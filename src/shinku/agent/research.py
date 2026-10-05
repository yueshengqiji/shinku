"""Bounded research ledger: model relevance is not executor/source evidence.

Only registered executor receipts mint candidate and source IDs. Model reviews
may recommend a page or assess entailment, but cannot create receipts, authorize
tools, or make a quoted snippet prove that a public claim is objectively true.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .completion import CompletionVerdict


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def source_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 2000:
        return ""
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return ""
        _ = parsed.port
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
    except ValueError:
        return ""


def short_text(value: Any, limit: int = 500) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def text_list(value: Any, limit: int = 8) -> list[str]:
    return list(dict.fromkeys(short_text(item, 160) for item in value if isinstance(item, str) and item.strip()))[:limit] if isinstance(value, list) else []


def visible_urls(answer: str) -> set[str]:
    return {url for value in re.findall(r"https?://[^\s<>\[\]()\"'，。；]+", answer)
            if (url := source_url(value.rstrip(".,;!?：")))}


@dataclass
class ResearchLedger:
    enabled: bool = False
    discovery_tools: set[str] = field(default_factory=set)
    reader_tools: set[str] = field(default_factory=set)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    intent: dict[str, Any] = field(default_factory=dict)
    review_calls: int = 0
    answer_review_calls: int = 0
    answer_check: dict[str, Any] = field(default_factory=dict)
    review_errors: list[str] = field(default_factory=list)

    def configure(self, handlers: Mapping[str, Any], context: Any) -> None:
        self.discovery_tools = {name for name, handler in handlers.items() if getattr(handler, "agent_research_candidate_field", "")}
        self.reader_tools = {name for name, handler in handlers.items() if getattr(handler, "agent_research_content_field", "")}
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        requirements = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        if isinstance(requirements, list) and any(isinstance(r, Mapping) and r.get("effect") == "research_answer" for r in requirements):
            self.enabled = True
        prior = context.get("agent_task_state", {}).get("research", {}) if isinstance(context, Mapping) and isinstance(context.get("agent_task_state", {}), Mapping) else {}
        # Restored snippets/claims are not fresh evidence. Only candidate URLs
        # survive; every source must be read again in this run before binding.
        if isinstance(prior, Mapping):
            rows = prior.get("candidates", [])
            for row in rows[:24] if isinstance(rows, list) else []:
                if isinstance(row, Mapping) and source_url(row.get("url")):
                    identifier = self._candidate(row, restored=True)
                    if identifier:
                        self.candidates[identifier]["source_scope"] = short_text(row.get("source_scope"), 253)
            self.enabled = self.enabled or bool(self.candidates)

    def _candidate(self, row: Mapping[str, Any], *, restored: bool = False, origin: str = "executor_search") -> str:
        url = source_url(row.get("url"))
        if not url:
            return ""
        identifier = "c_" + digest(url)[:16]
        card_hash = digest(short_text(row.get("title"), 240) + "\n" + short_text(row.get("snippet"), 500))
        if identifier not in self.candidates and len(self.candidates) < 32:
            self.candidates[identifier] = {
                "id": identifier, "url": url, "title": short_text(row.get("title"), 240),
                "snippet": short_text(row.get("snippet"), 500), "relevance": "unreviewed",
                "reason": "", "recommended_read": False, "read_attempts": 0,
                "origin": "historical_candidate" if restored else origin,
                "card_sha256": card_hash,
            }
        elif identifier in self.candidates and not restored and self.candidates[identifier].get("card_sha256") != card_hash:
            self.candidates[identifier].update(title=short_text(row.get("title"), 240), snippet=short_text(row.get("snippet"), 500),
                card_sha256=card_hash, relevance="unreviewed", recommended_read=False, reason="", origin=origin)
        return identifier if identifier in self.candidates else ""

    def observe(self, invocation: Any, envelope: Any, handler: Any) -> None:
        updates = (envelope.data or {}).get("state_updates", {})
        if not isinstance(updates, Mapping):
            return
        candidate_field = getattr(handler, "agent_research_candidate_field", "")
        if candidate_field:
            self.enabled = True
            rows = updates.get(candidate_field, [])
            for row in rows[:20] if isinstance(rows, list) and envelope.status == "ok" else []:
                if isinstance(row, Mapping):
                    identifier = self._candidate(row)
                    if identifier:
                        self.candidates[identifier]["source_scope"] = short_text(updates.get("source_scope"), 253)
        content_field = getattr(handler, "agent_research_content_field", "")
        if not content_field:
            return
        argument = getattr(handler, "agent_research_url_argument", "url")
        requested = source_url(invocation.arguments.get(argument))
        candidate_id = "c_" + digest(requested)[:16] if requested else ""
        if candidate_id in self.candidates:
            self.candidates[candidate_id]["read_attempts"] += 1
        text = updates.get(content_field)
        status = updates.get("http_status")
        if (envelope.status != "ok" or updates.get("site_access_ok") is not True
                or type(status) is not int or not 200 <= status < 400 or not isinstance(text, str) or not text.strip()):
            self.sources = {key: row for key, row in self.sources.items() if requested not in {row["url"], row["requested_url"]}}
            return
        url = source_url(updates.get("url") or requested)
        if not url:
            self.sources = {key: row for key, row in self.sources.items() if requested not in {row["url"], row["requested_url"]}}
            return
        scope = self.candidates.get(candidate_id, {}).get("source_scope", "")
        try:
            host = (urlsplit(url).hostname or "").encode("idna").decode("ascii")
        except UnicodeError:
            self.review_errors.append("invalid_source_hostname")
            return
        if scope and host != scope and not host.endswith("." + scope):
            self.review_errors.append("source_redirect_outside_scope")
            self.sources = {key: row for key, row in self.sources.items() if requested not in {row["url"], row["requested_url"]}}
            return
        offset = updates.get("content_offset", 0)
        if type(offset) is not int or offset < 0:
            return
        text = text[:8000]
        chunk_hash = digest(text)
        source_hash = short_text(updates.get("content_sha256"), 64)
        if not re.fullmatch("[a-f0-9]{64}", source_hash):
            source_hash = chunk_hash
        self.sources = {key: row for key, row in self.sources.items() if row["url"] != url or row["source_sha256"] == source_hash}
        identifier = "s_" + digest(f"{url}\n{source_hash}\n{offset}\n{chunk_hash}")[:16]
        if identifier not in self.sources and len(self.sources) >= 24:
            return
        self.sources[identifier] = {
            "id": identifier, "candidate_id": candidate_id, "url": url, "requested_url": requested,
            "title": short_text(updates.get("title"), 240), "text": text,
            "offset": offset, "next_offset": updates.get("content_next_offset"),
            "truncated": bool(updates.get("content_truncated")), "source_sha256": source_hash,
            "chunk_sha256": chunk_hash, "origin": "executor_source_read", "tool": invocation.name,
        }
        link_field = getattr(handler, "agent_research_link_field", "")
        links = updates.get(link_field, []) if link_field else []
        for row in links[:16] if self.enabled and isinstance(links, list) else []:
            if not isinstance(row, Mapping):
                continue
            target = source_url(row.get("url"))
            target_host = (urlsplit(target).hostname or "").lower()
            if not target or (scope and target_host != scope and not target_host.endswith("." + scope)):
                continue
            linked_id = "c_" + digest(target)[:16]
            if linked_id not in self.candidates:
                linked_id = self._candidate(row, origin="executor_page_link")
            if linked_id:
                linked = self.candidates[linked_id]
                linked.update(source_scope=scope, discovered_on_source=identifier)

    def pending_candidates(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in self.candidates.values() if row["relevance"] == "unreviewed"]

    def read_before_search(self) -> bool:
        return bool(self.reader_tools and any(row["recommended_read"] and row["relevance"] == "relevant" and not row["read_attempts"] for row in self.candidates.values()))

    def apply_candidate_review(self, value: Any, pending_ids: list[str]) -> None:
        reviews = value.get("candidate_reviews", []) if isinstance(value, Mapping) else []
        reviewed = set()
        for item in reviews[:32] if isinstance(reviews, list) else []:
            if not isinstance(item, Mapping) or item.get("candidate_id") not in pending_ids:
                continue
            identifier = item["candidate_id"]
            if identifier in reviewed:
                continue
            relevance = item.get("relevance")
            reason = short_text(item.get("reason"), 400)
            if not isinstance(relevance, str) or relevance not in {"relevant", "uncertain", "irrelevant"} or not reason:
                continue
            reviewed.add(identifier)
            self.candidates[identifier].update(relevance=relevance, reason=reason,
                recommended_read=item.get("recommended_read") is True and relevance != "irrelevant",
                review_origin="model_recommendation_not_executor_proof")
        for identifier in pending_ids:
            if identifier not in reviewed:
                self.candidates[identifier].update(relevance="uncertain", reason="相关性复核格式缺失，不能自动判定无关或已核实。", recommended_read=False)
        if len(reviewed) != len(pending_ids):
            self.review_errors.append("candidate_review_incomplete")
        intent = value.get("intent", {}) if isinstance(value, Mapping) else {}
        if isinstance(intent, Mapping):
            self.intent = {
                "topic": short_text(intent.get("topic")), "time_window": short_text(intent.get("time_window")),
                "answer_fields": text_list(intent.get("answer_fields")),
                "suggested_queries": text_list(intent.get("suggested_queries"), 3),
                "origin": "model_interpretation_not_authorization",
            }

    def required_fields(self, context: Any) -> list[str]:
        fields = list(self.intent.get("answer_fields", []))
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        requirements = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        for requirement in requirements if isinstance(requirements, list) else []:
            if isinstance(requirement, Mapping) and requirement.get("effect") == "research_answer":
                checks = requirement.get("checks", {})
                if isinstance(checks, Mapping):
                    fields.extend(text_list(checks.get("required_fields")))
        return list(dict.fromkeys(fields))[:16]

    def review_payload(self, goal: str, context: Any, *, answer: str = "") -> dict[str, Any]:
        # Discovery cards and document titles help select a page, but are not
        # body evidence. Don't mix them into an answer-entailment input: models
        # can otherwise keep quoting an unavailable title as if it were text.
        read_sources = []
        for row in list(self.sources.values())[-3:]:
            projected = {**row, "text": row["text"][:6000], "review_excerpt_truncated": len(row["text"]) > 6000}
            if answer:
                projected.pop("title", None)
            read_sources.append(projected)
        return {"original_request": goal[:5000], "required_fields": self.required_fields(context),
            "research_requirements": self.research_requirements(context),
            "intent": copy.deepcopy(self.intent), "candidates": [] if answer else list(self.candidates.values())[-24:],
            "read_sources": read_sources,
            "answer": answer[:10000]}

    @staticmethod
    def research_requirements(context: Any) -> list[dict[str, Any]]:
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        rows = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        return [copy.deepcopy({key: row.get(key) for key in ("id", "description", "checks")})
                for row in rows[:32] if isinstance(row, Mapping) and row.get("effect") == "research_answer"] if isinstance(rows, list) else []

    def apply_answer_review(self, answer: str, value: Any, context: Any) -> None:
        bindings, errors = [], []
        raw = value.get("claims", []) if isinstance(value, Mapping) else []
        if not isinstance(raw, list) or len(raw) > 20:
            raw, errors = [], ["invalid_claim_list"]
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                errors.append(f"claim_{index}:invalid")
                continue
            identifier = item.get("source_id")
            source = self.sources.get(identifier) if isinstance(identifier, str) else None
            fragment = short_text(item.get("answer_fragment"), 500)
            quote = short_text(item.get("quote"), 240)
            if (not source or not fragment or not quote or len(str(item.get("quote", ""))) > 240
                    or len(str(item.get("answer_fragment", ""))) > 500 or len(quote.split()) > 25
                    or normalized_text(fragment) not in normalized_text(answer)
                    or normalized_text(quote) not in normalized_text(source["text"])):
                errors.append(f"claim_{index}:unbound_text_or_source")
                continue
            # Require the source to be user-visible, not merely hidden metadata.
            if source["url"] not in visible_urls(answer) and source["requested_url"] not in visible_urls(answer):
                errors.append(f"claim_{index}:missing_visible_source_url")
                continue
            date_text = short_text(item.get("date_text"), 100)
            date_kind = item.get("date_kind", "unknown")
            if not isinstance(date_kind, str) or date_kind not in {"event", "publication", "updated", "unknown"} or (date_text and normalized_text(date_text) not in normalized_text(quote)):
                errors.append(f"claim_{index}:unbound_date")
                continue
            bindings.append({"index": index, "answer_fragment": fragment, "source_id": source["id"],
                "url": source["url"], "quote": quote, "source_sha256": source["source_sha256"],
                "chunk_sha256": source["chunk_sha256"], "date_text": date_text, "date_kind": date_kind,
                "excerpt_verified": True, "semantic_origin": "model_entailment_review"})
        coverage = value.get("coverage", []) if isinstance(value, Mapping) else []
        supported = []
        indices = {binding["index"] for binding in bindings}
        for row in coverage[:16] if isinstance(coverage, list) else []:
            links = row.get("claim_indices", []) if isinstance(row, Mapping) else []
            if (isinstance(row, Mapping) and row.get("supported") is True and isinstance(links, list)
                    and links and all(type(i) is int and i in indices for i in links)):
                supported.append(short_text(row.get("field"), 160))
        required = self.required_fields(context)
        missing = list(dict.fromkeys([*(field for field in required if field not in supported),
            *text_list(value.get("missing_fields", []))] if isinstance(value, Mapping) else required))
        verdict = value.get("verdict") if isinstance(value, Mapping) else "invalid"
        if not isinstance(verdict, str) or verdict not in {"supported", "partial", "unsupported", "no_answer"}:
            errors.append("invalid_semantic_review")
            verdict = "invalid"
        if verdict == "supported" and not bindings:
            errors.append("no_bound_claims")
        limitations = []
        known_conditions = {row.get("id") for row in self.research_requirements(context) if isinstance(row.get("id"), str)}
        rows = value.get("limitations", []) if isinstance(value, Mapping) else []
        for row in rows[:16] if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            fragment = short_text(row.get("answer_fragment"), 500)
            if (not fragment or len(str(row.get("answer_fragment", ""))) > 500
                    or normalized_text(fragment) not in normalized_text(answer)):
                continue
            fields = [item for item in text_list(row.get("fields"), 16) if item in required]
            conditions = [item for item in text_list(row.get("condition_ids"), 32) if item in known_conditions]
            if fields or conditions:
                limitations.append({"answer_fragment": fragment, "fields": fields, "condition_ids": conditions,
                    "semantic_origin": "model_limitation_review"})
        self.answer_check = {"answer_sha256": digest(answer), "verdict": verdict, "bindings": bindings,
            "covered_fields": list(dict.fromkeys(supported)), "missing_fields": missing,
            "limitations": limitations, "errors": errors, "semantic_review_is_not_absolute_truth": True}

    def recommended_outcome(self, answer: str, outcome: str, context: Any) -> str:
        """Only lower an overclaimed research completion; never invent an answer.

        Acknowledgment is a semantic review opinion, bound to visible answer
        text. The host independently retains and checks every original condition.
        Mixed side-effect contracts and explicit failure states are untouched.
        """
        if not self.enabled or outcome not in {"", "completed"}:
            return outcome
        check = self.answer_check
        if (check.get("answer_sha256") != digest(answer) or not self.satisfies({"min_sources": 1})
                or not self.evaluate_answer(answer, "partial").accepted):
            return outcome
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        rows = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) or row.get("effect") != "research_answer" for row in rows):
            return outcome
        missing_fields = set(check.get("missing_fields", []))
        missing_conditions = {row.get("id") for row in rows if not isinstance(row.get("checks", {}), Mapping)
                              or not self.satisfies(row.get("checks", {}))}
        if not missing_fields and not missing_conditions and check.get("verdict") != "partial":
            return outcome
        limitations = check.get("limitations", [])
        acknowledged_fields = {item for row in limitations for item in row["fields"]}
        acknowledged_conditions = {item for row in limitations for item in row["condition_ids"]}
        if (limitations and missing_fields.issubset(acknowledged_fields)
                and missing_conditions.issubset(acknowledged_conditions)):
            return "partial"
        return outcome

    def evaluate_answer(self, answer: str, outcome: str) -> CompletionVerdict:
        if not self.enabled:
            return CompletionVerdict(True, status=outcome or "completed")
        check = self.answer_check
        if check.get("answer_sha256") != digest(answer):
            return CompletionVerdict(False, reason="研究答复尚未绑定本轮正文证据；不能只凭搜索卡片或模型完成声明交付。")
        if not self._bindings_current():
            return CompletionVerdict(False, reason="答复绑定的正文已失效或版本变化，需要重新读取和复核。")
        if check.get("errors") or check.get("verdict") in {"unsupported", "invalid"}:
            return CompletionVerdict(False, reason="研究答复证据绑定未通过：" + ", ".join(check.get("errors") or ["结论缺少正文支持"]))
        if outcome == "completed" or not outcome:
            if check.get("verdict") != "supported" or not check.get("bindings") or check.get("missing_fields"):
                return CompletionVerdict(False, reason="研究答案尚不完整或未得到正文支持；交付有依据的部分时明确 partial 和具体缺口。",
                    missing=tuple(check.get("missing_fields", [])))
        elif check.get("verdict") == "no_answer" and check.get("bindings"):
            return CompletionVerdict(False, reason="无答案终态与事实结论绑定矛盾，请修正答复。")
        elif outcome == "partial" and not check.get("bindings"):
            return CompletionVerdict(False, reason="没有正文支持的部分结论；仅有候选线索时应诚实说明尚未回答，不能冒充部分答案。")
        return CompletionVerdict(True, status=outcome or "completed", reason="research_evidence_bound",
            evidence=tuple(copy.deepcopy(check.get("bindings", []))))

    def satisfies(self, checks: Mapping[str, Any]) -> bool:
        bindings = self.answer_check.get("bindings", [])
        if not bindings or not self._bindings_current() or self.answer_check.get("errors") or self.answer_check.get("verdict") not in {"supported", "partial"}:
            return False
        fields = text_list(checks.get("required_fields", []))
        if not set(fields).issubset(self.answer_check.get("covered_fields", [])):
            return False
        count = checks.get("min_sources", 1)
        if type(count) is not int or not 1 <= count <= 8 or len({row["url"] for row in bindings}) < count:
            return False
        if checks.get("require_dates") is True:
            kinds = checks.get("date_kinds", ["event", "publication"])
            if (not isinstance(kinds, list) or not 1 <= len(kinds) <= 3
                    or any(not isinstance(kind, str) or kind not in {"event", "publication", "updated"} for kind in kinds)
                    or not any(row["date_text"] and row["date_kind"] in kinds for row in bindings)):
                return False
        return True

    def _bindings_current(self) -> bool:
        # The gate independently checks freshness; planner review caching alone
        # must not keep a previously bound answer valid after a denied reread.
        for binding in self.answer_check.get("bindings", []):
            source = self.sources.get(binding["source_id"])
            if (not source or source["source_sha256"] != binding["source_sha256"]
                    or source["chunk_sha256"] != binding["chunk_sha256"]):
                return False
        return True

    def planner_context(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "intent": self.intent, "candidates": list(self.candidates.values())[-24:],
            "read_before_more_search": self.read_before_search(),
            "sources": [{**row, "text": row["text"][:1200], "excerpt_only": len(row["text"]) > 1200}
                        for row in list(self.sources.values())[-3:]],
            "answer_check": self.answer_check, "review_calls": self.review_calls, "answer_review_calls": self.answer_review_calls}

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy({"enabled": self.enabled, "intent": self.intent,
            "candidates": list(self.candidates.values())[-24:],
            "source_receipts": [{k: v for k, v in row.items() if k != "text"} for row in list(self.sources.values())[-24:]],
            "answer_check": self.answer_check, "review_calls": self.review_calls,
            "answer_review_calls": self.answer_review_calls, "review_errors": self.review_errors[-8:],
            "restoration_requires_source_refresh": True})

    def answer_review_key(self, answer: str, context: Any) -> str:
        return digest(json.dumps({"answer": answer, "sources": sorted(self.sources),
            "fields": self.required_fields(context), "requirements": self.research_requirements(context)}, ensure_ascii=False, sort_keys=True))

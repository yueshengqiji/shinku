"""Task-local read evidence. Model reviews cannot mint receipts or authorize actions.

Literal bindings and finite Decimal calculations are checked by the host. The
semantic adequacy of non-numeric conclusions still needs a bounded model review;
this is not a general theorem prover. Persisted snapshots are never fresh reads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import ntpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from .completion import CompletionVerdict


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def eligible_source(handler: Any) -> bool:
    return (getattr(handler, "agent_local_answer_source", False) is True
            and getattr(handler, "agent_permission_class", "") == "read"
            and "read" in (getattr(handler, "agent_completion_effects", ()) or ())
            and not set(getattr(handler, "agent_completion_effects", ()) or ()) & {"write", "delete", "send", "process_stop"}
            and callable(getattr(handler, "agent_verify_result", None)))


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")


def _numbers(text: str) -> list[Decimal]:
    values = []
    for match in _NUMBER.finditer(text):
        if len(match[0]) > 80:
            raise ValueError("oversized_number")
        value = Decimal(match[0])
        if not value.is_finite() or abs(value) > Decimal("1e50"):
            raise ValueError("unbounded_number")
        values.append(value)
    return values


@dataclass
class LocalAnswerLedger:
    enabled: bool = False
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    answer_check: dict[str, Any] = field(default_factory=dict)
    answer_review_calls: int = 0
    review_errors: list[str] = field(default_factory=list)

    def configure(self, context: Any) -> None:
        self.enabled = bool(self.requirements(context))
        # In-memory checkpoints retain this instance. DB snapshots cannot mint
        # evidence or restore a model review and must be read anew.

    @staticmethod
    def requirements(context: Any) -> list[Mapping]:
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        rows = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        return [r for r in rows if isinstance(r, Mapping) and r.get("effect") == "local_answer"] if isinstance(rows, (list, tuple)) else []

    def observe(self, invocation: Any, envelope: Any, handler: Any, context: Any) -> None:
        if not self.enabled or not eligible_source(handler):
            return
        updates = (envelope.data or {}).get("state_updates", {})
        updates = updates if isinstance(updates, Mapping) else {}
        aliases = [str(v) for key, v in invocation.arguments.items()
                   if key in (getattr(handler, "agent_completion_target_arguments", ()) or ()) and isinstance(v, str)]
        path = updates.get("path", "")
        for row in self.sources.values():
            if any(self._path(a) in row["aliases"] for a in [path, *aliases] if a):
                if envelope.status != "ok" or updates.get("sha256") != row["sha256"] or updates.get("read_complete") is not True:
                    row["valid"] = False
        if (envelope.status != "ok" or updates.get("read_complete") is not True
                or "read" not in updates.get("verified_effects", ())
                or not isinstance(path, str) or not ntpath.isabs(path) or not updates.get("sha256")
                or path not in updates.get("completion_targets", ())):
            return
        identifier = "l_" + _digest(self._path(path))[:16]
        if identifier in self.sources or len(self.sources) >= 32:
            return
        row = {"id": identifier, "path": path, "tool": invocation.name, "sha256": updates["sha256"],
               "receipt": copy.deepcopy(dict(updates)), "valid": True, "text": "",
               "aliases": list(dict.fromkeys(self._path(a) for a in [path, *aliases]))}
        self.sources[identifier] = row
        self._refresh_source(row, handler, context)

    @staticmethod
    def _path(value: str) -> str:
        return ntpath.normpath(value.replace("/", "\\")).casefold()

    def _refresh_source(self, row: dict, handler: Any, context: Any) -> None:
        if not row["valid"]:
            return
        try:
            fresh = handler.agent_verify_result(receipt=copy.deepcopy(row["receipt"]), context=context) if eligible_source(handler) else {}
            content = fresh.get("file_text")
            if (fresh.get("result_valid") is not True or fresh.get("sha256") != row["sha256"]
                    or not isinstance(content, str) or len(content) > 12000
                    or sum(len(r["text"]) for r in self.sources.values() if r is not row) + len(content) > 120000):
                row["valid"] = False
            else:
                row["text"] = content
        except Exception:
            row["valid"] = False

    def refresh(self, handlers: Mapping, context: Any) -> None:
        for row in self.sources.values():
            self._refresh_source(row, handlers.get(row["tool"]), context)

    def planner_context(self) -> dict:
        # A transcript window can evict the earliest tool message. Keep the
        # task's actual bounded read data available, not merely hashes/IDs that
        # force the planner to guess or re-read a duplicate-rejected call.
        return {"sources": [{**{k: r[k] for k in ("id", "path", "sha256", "valid")},
                              "text": r["text"] if r["valid"] else ""} for r in self.sources.values()],
                "answer_check": self.answer_check, "review_calls": self.answer_review_calls}

    def review_payload(self, goal: str, context: Any, *, answer: str) -> dict:
        return {"original_request": goal, "answer": answer, "local_requirements": self.requirements(context),
                "read_sources": [{k: r[k] for k in ("id", "path", "sha256", "text")}
                                 for r in self.sources.values() if r["valid"]]}

    def answer_review_key(self, answer: str, context: Any) -> str:
        return _digest(json.dumps([answer, [(r["id"], r["sha256"], r["valid"]) for r in self.sources.values()],
                                   self.requirements(context)], ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _calculate(calculation: Mapping, evidence: list[dict]) -> list[Decimal]:
        op = calculation.get("operation")
        if op == "count_sources":
            return [Decimal(len({e["source_id"] for e in evidence}))]
        operands = calculation.get("operands", [])
        if not isinstance(operands, list) or not 1 <= len(operands) <= 128:
            raise ValueError("invalid_operands")
        values, seen = [], set()
        for operand in operands:
            if not isinstance(operand, Mapping):
                raise ValueError("invalid_operand")
            index, value = operand.get("evidence_index"), operand.get("value")
            if type(index) is not int or not 0 <= index < len(evidence) or not isinstance(value, str):
                raise ValueError("invalid_operand")
            parsed = _numbers(value)
            if len(parsed) != 1 or parsed[0] not in _numbers(evidence[index]["quote"]) or _NUMBER.fullmatch(value) is None:
                raise ValueError("operand_not_in_quote")
            key = (evidence[index]["source_id"], evidence[index]["quote"], parsed[0])
            if key in seen:
                raise ValueError("duplicate_operand")
            seen.add(key)
            values.append(parsed[0])
        with localcontext() as ctx:
            ctx.prec = 100
            if op == "sum": return [sum(values, Decimal(0))]
            if op == "count": return [Decimal(len(values))]
            if op == "min": return [min(values)]
            if op == "max": return [max(values)]
            if op == "mean": return [sum(values, Decimal(0)) / len(values)]
            if op == "collect": return values
            if op == "sort": return sorted(values)
            if op == "product":
                result = Decimal(1)
                for value in values: result *= value
                return [result]
            if op == "difference" and len(values) == 2: return [values[0] - values[1]]
            if op == "ratio" and len(values) == 2 and values[1]: return [values[0] / values[1]]
        raise ValueError("unsupported_calculation")

    def apply_answer_review(self, answer: str, value: Any, context: Any) -> None:
        check = {"answer_sha256": _digest(answer), "verdict": "unsupported", "bindings": [],
                 "covered_fields": [], "missing_fields": [], "source_ids": [], "valid": False}
        self.answer_check = check
        if not isinstance(value, Mapping) or len(answer) > 10000:
            return
        rows, coverage = value.get("claims"), value.get("coverage")
        if (value.get("verdict") not in {"supported", "partial"} or not isinstance(rows, list)
                or not 1 <= len(rows) <= 32 or not isinstance(coverage, list) or len(coverage) > 32):
            return
        try:
            bound, all_sources, fragments = [], set(), []
            normalized_answer = _text(answer)
            for row in rows:
                fragment = row.get("answer_fragment") if isinstance(row, Mapping) else None
                evidence = row.get("evidence") if isinstance(row, Mapping) else None
                if (not isinstance(fragment, str) or not 1 <= len(fragment) <= 500
                        or _text(fragment) not in normalized_answer or not isinstance(evidence, list) or not 1 <= len(evidence) <= 32):
                    raise ValueError("invalid_fragment")
                for item in evidence:
                    source = self.sources.get(item.get("source_id")) if isinstance(item, Mapping) else None
                    quote = item.get("quote") if isinstance(item, Mapping) else None
                    if (not source or not source["valid"] or not isinstance(quote, str) or not 1 <= len(quote) <= 240
                            or not _text(quote) or _text(quote) not in _text(source["text"])):
                        raise ValueError("invalid_quote")
                    all_sources.add(source["id"])
                numbers = _numbers(fragment)
                calculation = row.get("calculation")
                if calculation is not None:
                    if not isinstance(calculation, Mapping) or numbers != self._calculate(calculation, evidence):
                        raise ValueError("calculation_mismatch")
                elif any(n not in [x for e in evidence for x in _numbers(e["quote"])] for n in numbers):
                    raise ValueError("derived_number_without_calculation")
                fragments.append(_text(fragment))
                bound.append({"answer_fragment": fragment, "source_ids": [e["source_id"] for e in evidence]})
            # A supported review cannot leave an extra numeric assertion unbound.
            # List numbering is presentation, not a factual quantity.
            numeric_answer = re.sub(r"(?m)^\s*\d+[.)]\s+", "", normalized_answer)
            remainder = numeric_answer
            for fragment in sorted(fragments, key=len, reverse=True):
                remainder = remainder.replace(fragment, " ")
            if _numbers(remainder):
                raise ValueError("unbound_numeric_fact")
            fields = list(dict.fromkeys(f for r in self.requirements(context) for f in r.get("checks", {}).get("required_fields", [])))
            covered = set()
            for row in coverage:
                if not isinstance(row, Mapping): raise ValueError("invalid_coverage")
                indices = row.get("claim_indices", [])
                if row.get("supported") is True:
                    if (row.get("field") not in fields or not isinstance(indices, list) or not indices
                            or any(type(i) is not int or not 0 <= i < len(bound) for i in indices)):
                        raise ValueError("invalid_coverage_binding")
                    covered.add(row["field"])
            missing = [f for f in fields if f not in covered]
            if value["verdict"] == "supported" and (missing or value.get("missing_fields")):
                raise ValueError("incomplete_coverage")
            if value["verdict"] == "partial" and (not missing or any(f not in value.get("missing_fields", []) or f not in answer for f in missing)):
                raise ValueError("undisclosed_partial_gap")
            check.update(verdict=value["verdict"], bindings=bound, covered_fields=sorted(covered), missing_fields=missing,
                         source_ids=sorted(all_sources), valid=True)
        except (ValueError, TypeError, KeyError, InvalidOperation, ArithmeticError):
            self.review_errors.append("local_answer_binding_failed")

    def evaluate_answer(self, answer: str, outcome: str) -> CompletionVerdict:
        if not self.enabled or outcome in {"failed", "blocked", "waiting_background"}:
            return CompletionVerdict(True, status=outcome or "completed")
        check = self.answer_check
        valid = (check.get("valid") and check.get("answer_sha256") == _digest(answer)
                 and all(self.sources.get(i, {}).get("valid") for i in check.get("source_ids", [])))
        if not valid or (outcome != "partial" and check.get("verdict") != "supported"):
            return CompletionVerdict(False, reason="本地答复缺少有效读取证据、数值复算或字段覆盖")
        return CompletionVerdict(True, status="partial" if outcome == "partial" else "completed")

    def satisfies(self, checks: Any) -> bool:
        if not isinstance(checks, Mapping) or set(checks) - {"required_fields", "min_sources"}:
            return False
        check = self.answer_check
        ids = check.get("source_ids", [])
        minimum = checks.get("min_sources", 1)
        return bool(check.get("valid") and type(minimum) is int and 1 <= minimum <= 32
                    and len(ids) >= minimum and all(self.sources.get(i, {}).get("valid") for i in ids)
                    and set(checks.get("required_fields", [])) <= set(check.get("covered_fields", [])))

    def snapshot(self) -> dict:
        return {"sources": [{k: r[k] for k in ("id", "path", "tool", "sha256", "valid")} for r in self.sources.values()],
                "answer_check": copy.deepcopy(self.answer_check), "answer_review_calls": self.answer_review_calls,
                "review_errors": self.review_errors[-8:]}

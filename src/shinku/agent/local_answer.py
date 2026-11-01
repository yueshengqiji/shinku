"""Task-local read evidence. Model reviews cannot mint receipts or authorize actions.

Literal bindings and finite Decimal calculations are checked by the host. The
semantic adequacy of non-numeric conclusions still needs a bounded model review;
this is not a general theorem prover. Persisted snapshots are never fresh reads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from shinku.paths import absolute_file_path, normalize_file_path
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .completion import CompletionVerdict


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _without_list_labels(value: str) -> str:
    """只去掉行首有序列表标号；句内数值和标号后的实际断言仍需绑定。"""
    return re.sub(r"(?m)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?\d{1,3}(?:\.[ \t]+|[)）、][ \t]*)", "", value)


def _answer_fragments(answer: str) -> dict[str, str]:
    """宿主提供当前答案的逐字片段选择器，避免审查模型改写或拼错原文。

    选择器仅绑定当前答复字节，不证明事实正确；来源引用、计算和覆盖仍须核验。
    兼容旧的 answer_fragment 文本协议，较长/多行计算也可继续选取具体原文。
    """
    fragments = {}
    answer_key = _digest(answer)[:16]
    for line in answer.splitlines():
        for start in range(0, len(line), 400):
            fragment = line[start:start + 400].strip()
            if fragment and len(fragments) < 128:
                fragments[f"af_{answer_key}_{len(fragments):03d}"] = fragment
    return fragments


def eligible_source(handler: Any) -> bool:
    return (getattr(handler, "agent_local_answer_source", False) is True
            and getattr(handler, "agent_permission_class", "") == "read"
            and "read" in (getattr(handler, "agent_completion_effects", ()) or ())
            and not set(getattr(handler, "agent_completion_effects", ()) or ()) & {"write", "delete", "send", "process_stop"}
            and callable(getattr(handler, "agent_verify_result", None)))


def reference_source(handler, path):
    """只有宿主注册的读取适配器可声明非文件快照；正文仍须其验证器重验。"""
    prefix = getattr(handler, "agent_local_answer_reference_prefix", "")
    return bool(isinstance(prefix, str) and prefix and isinstance(path, str)
                and path.startswith(prefix) and len(path) > len(prefix))


# 允许句末句点和章节标号后的点；禁止从多段版本号中截取一部分充当数值。
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w|\.\d)")


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
        # 内存切片沿用本实例；DB 恢复只能重验宿主回执，不能采信旧答案审查。

    def restore_reads(self, calls: list, handlers: Mapping, context: Any) -> None:
        """重建本任务已执行的完整读取，不重放工具，也不恢复语义通过结论。

        调用者先通过当前授权/中断门禁；处理器再以当前权限重读原文件或
        重验日志查询快照及哈希。绑定和指纹只证明归属，不证明来源仍有效，
        也不能将历史日志快照变成当前服务状态。
        """
        if not self.enabled or not isinstance(context, Mapping):
            return
        task = context.get("agent_task_state", {})
        if not isinstance(task, Mapping) or not task.get("task_id"):
            return
        reviews = task.get("progress", {}).get("local_answer_review_calls", 0)
        if type(reviews) is int and reviews >= 0:
            self.answer_review_calls = min(2, reviews)
        for call in calls:
            binding = call.get("execution_binding", {})
            arguments = call.get("arguments")
            if (call.get("executed") is not True or not isinstance(binding, Mapping)
                    or not isinstance(arguments, Mapping)
                    or any(binding.get(k) != task.get(k) for k in ("task_id", "requester", "session"))
                    or type(binding.get("attempt")) is not int
                    or not 0 < binding["attempt"] <= task.get("execution_attempt", 0)):
                continue
            invocation = ToolInvocation(str(call.get("name", "")), dict(arguments), id=call.get("id", ""))
            handler = handlers.get(invocation.name)
            receipt = (call.get("data") or {}).get("state_updates", {})
            reference = isinstance(receipt, Mapping) and reference_source(handler, receipt.get("path"))
            # 非文件来源绑定原查询；不能拿另一次查询的回执替代本次调用。
            query_matches = reference and handler.normalize_call(arguments) == receipt.get("query")
            receipt_paths = [receipt.get('path', '')] if isinstance(receipt, Mapping) else []
            # 工作区保留原始绝对参数，回执path规范成相对路径；恢复时允许匹配
            # handler声明的已绑定来源。下面observe仍独立重读，核对来源/权限/哈希，
            # 不能只因completion_targets声称包含某路径就恢复证据。
            source = receipt.get(getattr(handler, 'agent_local_answer_path_field', 'path')) if isinstance(receipt, Mapping) else None
            targets = receipt.get('completion_targets', ()) if isinstance(receipt, Mapping) else ()
            if isinstance(source, str) and isinstance(targets, (list, tuple)) and source in targets:
                receipt_paths.append(source)
            if (not eligible_source(handler) or not isinstance(receipt, Mapping)
                    or call.get("status") == "ok" and not query_matches and not any(isinstance(arguments.get(k), str)
                        and self._path(arguments[k]) == self._path(str(path))
                        for k in (getattr(handler, "agent_completion_target_arguments", ()) or ()) for path in receipt_paths)):
                continue
            signature = invocation.name + ":" + json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if call.get("call_fingerprint") != _digest(signature):
                continue
            self.observe(invocation, ToolResultEnvelope(invocation.id, call.get("status", ""), "",
                copy.deepcopy(call.get("data") or {}), executed=True), handler, context, keep_invalid_metadata=False)

    @staticmethod
    def requirements(context: Any) -> list[Mapping]:
        contract = context.get("agent_completion_contract", {}) if isinstance(context, Mapping) else {}
        rows = contract.get("requirements", []) if isinstance(contract, Mapping) else []
        return [r for r in rows if isinstance(r, Mapping) and r.get("effect") == "local_answer"] if isinstance(rows, (list, tuple)) else []

    def observe(self, invocation: Any, envelope: Any, handler: Any, context: Any, *, keep_invalid_metadata: bool = True) -> None:
        if not self.enabled or not eligible_source(handler):
            return
        updates = (envelope.data or {}).get("state_updates", {})
        updates = updates if isinstance(updates, Mapping) else {}
        aliases = [str(v) for key, v in invocation.arguments.items()
                   if key in (getattr(handler, "agent_completion_target_arguments", ()) or ()) and isinstance(v, str)]
        receipt_path = updates.get("path", "")
        if reference_source(handler, receipt_path) and handler.normalize_call(invocation.arguments) != updates.get("query"):
            return
        # 工作区调用兼容根内绝对路径，回执用相对路径；宿主给出已绑定的绝对来源。
        # 它必须在完成目标内，随后仍由原处理器按原回执实时重读核验，非授权旁路。
        path = updates.get(getattr(handler, "agent_local_answer_path_field", "path"), receipt_path)
        if path != receipt_path and (not isinstance(path, str) or path not in updates.get("completion_targets", ())):
            return
        if isinstance(receipt_path, str):
            aliases.append(receipt_path)
        for row in self.sources.values():
            if any(self._path(a) in row["aliases"] for a in [path, *aliases] if a):
                if envelope.status != "ok" or updates.get("sha256") != row["sha256"] or updates.get("read_complete") is not True:
                    row["valid"] = False
        if (envelope.status != "ok" or updates.get("read_complete") is not True
                or "read" not in updates.get("verified_effects", ())
                or not isinstance(path, str) or not (absolute_file_path(path) or reference_source(handler, path))
                or not updates.get("sha256")
                or path not in updates.get("completion_targets", ())):
            return
        identifier = "l_" + _digest(self._path(path))[:16]
        old = self.sources.get(identifier)
        if old and old["valid"] and old["sha256"] == updates["sha256"]:
            return
        if old is None and sum(r["valid"] for r in self.sources.values()) >= 32:
            return
        row = {"id": identifier, "path": path, "tool": invocation.name, "sha256": updates["sha256"],
               "receipt": copy.deepcopy(dict(updates)), "valid": True, "text": "",
               "aliases": list(dict.fromkeys(self._path(a) for a in [path, *aliases]))}
        self._refresh_source(row, handler, context)
        # 实时观测保留失效回执元数据以供诊断，仍不可作为正文证据；DB 恢复只装载
        # 当前重验通过的来源。有效槽按 valid 计数，完整工具/材料账本一直保留。
        if row["valid"] or keep_invalid_metadata:
            self.sources[identifier] = row
        if row["valid"]:
            self.answer_check = {}

    @staticmethod
    def _path(value: str) -> str:
        return normalize_file_path(value)

    def _refresh_source(self, row: dict, handler: Any, context: Any) -> None:
        if not row["valid"]:
            return
        try:
            fresh = handler.agent_verify_result(receipt=copy.deepcopy(row["receipt"]), context=context) if eligible_source(handler) else {}
            content = fresh.get("file_text")
            if (fresh.get("result_valid") is not True or fresh.get("sha256") != row["sha256"]
                    # 以既有共享总量控制正文；单文件较大但总量仍可容纳时不能
                    # 因旧的每文件阈值丢失已经完整读取、哈希匹配的源码证据。
                    or not isinstance(content, str)
                    or sum(len(r["text"]) for r in self.sources.values() if r["valid"] and r["id"] != row["id"]) + len(content) > 120000):
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
                "prior_binding_error": self.answer_check.get("binding_error", ""),
                "prior_unsupported_numbers": self.answer_check.get("unsupported_numbers", []),
                "answer_fragments": _answer_fragments(answer),
                "required_field_selectors": self._field_selectors(context),
                "numeric_evidence_candidates": self._numeric_quotes(answer),
                "read_sources": [{k: r[k] for k in ("id", "path", "sha256", "text")}
                                 for r in self.sources.values() if r["valid"]]}

    def _field_selectors(self, context: Any) -> dict[str, str]:
        fields = dict.fromkeys(f for r in self.requirements(context) for f in r.get("checks", {}).get("required_fields", []))
        return {"rf_" + _digest(f)[:16]: f for f in fields}

    def _numeric_quotes(self, answer: str) -> dict[str, dict]:
        """提供真实数值附近的逐字引用候选；模型仍须选择相关证据，宿主不自动补证。"""
        wanted = set(_numbers(self._numeric_answer_text(answer)))
        candidates, counts = {}, {}
        for source in self.sources.values():
            if not source["valid"]:
                continue
            text = source["text"]
            for match in _NUMBER.finditer(text):
                value = Decimal(match[0])
                key = (source["id"], value)
                if value not in wanted or counts.get(key, 0) >= 2:
                    continue
                start, end = max(0, match.start() - 80), min(len(text), match.end() + 80)
                quote = text[start:end]
                if len(quote) > 240:
                    continue
                identifier = "nq_" + _digest(source["id"] + source["sha256"] + str(match.start()) + quote)[:16]
                candidates[identifier] = {"source_id": source["id"], "quote": quote, "value": str(value)}
                counts[key] = counts.get(key, 0) + 1
                if len(candidates) >= 64:
                    return candidates
        return candidates

    def answer_review_key(self, answer: str, context: Any) -> str:
        return _digest(json.dumps([answer, [(r["id"], r["sha256"], r["valid"]) for r in self.sources.values()],
                                   self.requirements(context)], ensure_ascii=False, sort_keys=True))

    def _numeric_answer_text(self, value: str) -> str:
        """已核验文件名是来源标识，不是统计数值；未知路径和句内数量仍需证明。"""
        identifiers = set()
        for source in self.sources.values():
            if not source["valid"]:
                continue
            path = source["path"].replace("\\", "/")
            parts = path.split("/")
            if "." not in parts[-1]:
                continue
            for start in range(len(parts)):
                identifier = "/".join(parts[start:])
                if identifier:
                    identifiers.update((identifier, identifier.replace("/", "\\")))
            # README 等已读正文里的逐字文件引用也属于标识符；这不证明目标文件内容。
            identifiers.update(re.findall(r"[\w./\\-]+\.(?:md|py|toml|json|ya?ml|txt|rst|ini|cfg)\b", source["text"]))
        for identifier in sorted(identifiers, key=len, reverse=True):
            if identifier not in value:
                continue
            value = re.sub(r"(?<![\w./\\-])" + re.escape(identifier) + r"(?![\w./\\-])", "[source]", value)
        return _without_list_labels(value)

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
                 "covered_fields": [], "missing_fields": [], "source_ids": [], "valid": False,
                 "binding_error": "invalid_review_schema"}
        self.answer_check = check
        if not isinstance(value, Mapping) or len(answer) > 10000:
            return
        rows, coverage = copy.deepcopy(value.get("claims")), copy.deepcopy(value.get("coverage"))
        if value.get("verdict") in {"unsupported", "no_answer"} and isinstance(rows, list) and isinstance(coverage, list):
            # 合法的否定审查是证据不足，不能误报为 JSON/schema 损坏而引导无效重写。
            fields = {f for r in self.requirements(context) for f in r.get("checks", {}).get("required_fields", [])}
            check.update(verdict=value["verdict"], binding_error="answer_not_supported",
                         missing_fields=[f for f in value.get("missing_fields", []) if isinstance(f, str) and f in fields]
                         if isinstance(value.get("missing_fields", []), list) else [])
            return
        if (value.get("verdict") not in {"supported", "partial"} or not isinstance(rows, list)
                or not 1 <= len(rows) <= 32 or not isinstance(coverage, list) or len(coverage) > 32):
            return
        try:
            selectors = _answer_fragments(answer)
            numeric_quotes = self._numeric_quotes(answer)
            for row in rows:
                if isinstance(row, dict) and "answer_fragment_id" in row:
                    selector = row["answer_fragment_id"]
                    fragment = selectors.get(selector) if isinstance(selector, str) else None
                    if fragment is None or row.get("answer_fragment", fragment) != fragment:
                        raise ValueError("invalid_fragment_selector")
                    row["answer_fragment"] = fragment
                if isinstance(row, dict) and isinstance(row.get("evidence"), list) and len(row["evidence"]) <= 32:
                    for item in row["evidence"]:
                        if not isinstance(item, dict) or "quote_id" not in item:
                            continue
                        selector = item["quote_id"]
                        candidate = numeric_quotes.get(selector) if isinstance(selector, str) else None
                        if candidate is None or any(item.get(k, candidate[k]) != candidate[k] for k in ("source_id", "quote")):
                            raise ValueError("invalid_quote_selector")
                        item.update({k: candidate[k] for k in ("source_id", "quote")})
            # 字段与声明都由本次输入的编号绑定，不依赖模型重新拼字段或猜数组下标。
            fields_by_id = self._field_selectors(context)
            for row in coverage:
                if not isinstance(row, dict):
                    continue
                if "field_id" in row:
                    selector = row["field_id"]
                    selected = fields_by_id.get(selector) if isinstance(selector, str) else None
                    if selected is None or row.get("field", selected) != selected:
                        raise ValueError("invalid_field_selector")
                    row["field"] = selected
                if "answer_fragment_ids" in row:
                    references = row["answer_fragment_ids"]
                    if (not isinstance(references, list) or not 1 <= len(references) <= 32
                            or any(not isinstance(v, str) for v in references)):
                        raise ValueError("invalid_coverage_selector")
                    indices = [i for i, claim in enumerate(rows) if isinstance(claim, Mapping)
                               and claim.get("answer_fragment_id") in references]
                    if (not references or set(references) - {rows[i]["answer_fragment_id"] for i in indices}
                            or row.get("claim_indices", indices) != indices):
                        raise ValueError("invalid_coverage_selector")
                    row["claim_indices"] = indices
            bound, all_sources, fragments = [], set(), []
            normalized_answer = _text(answer)
            for row in rows:
                fragment = row.get("answer_fragment") if isinstance(row, Mapping) else None
                evidence = row.get("evidence") if isinstance(row, Mapping) else None
                if not isinstance(fragment, str) or not 1 <= len(fragment) <= 500:
                    raise ValueError("invalid_fragment_length")
                if _text(fragment) not in normalized_answer:
                    raise ValueError("fragment_not_literal")
                if not isinstance(evidence, list) or not 1 <= len(evidence) <= 32:
                    raise ValueError("invalid_fragment_evidence")
                for item in evidence:
                    source = self.sources.get(item.get("source_id")) if isinstance(item, Mapping) else None
                    quote = item.get("quote") if isinstance(item, Mapping) else None
                    if (not source or not source["valid"] or not isinstance(quote, str) or not 1 <= len(quote) <= 240
                            or not _text(quote) or _text(quote) not in _text(source["text"])):
                        raise ValueError("invalid_quote")
                    all_sources.add(source["id"])
                numbers = _numbers(self._numeric_answer_text(fragment))
                calculation = row.get("calculation")
                if calculation is not None:
                    if not isinstance(calculation, Mapping) or numbers != self._calculate(calculation, evidence):
                        raise ValueError("calculation_mismatch")
                else:
                    missing_numbers = [n for n in numbers if n not in [x for e in evidence for x in _numbers(e["quote"])]]
                    if missing_numbers:
                        check["unsupported_numbers"] = [str(n) for n in missing_numbers[:32]]
                        raise ValueError("derived_number_without_calculation")
                fragments.append(_text(fragment))
                bound.append({"answer_fragment": fragment, "source_ids": [e["source_id"] for e in evidence]})
            # A supported review cannot leave an extra numeric assertion unbound.
            # List numbering is presentation, not a factual quantity.
            # 必须先按原文行边界剥离标号，再归一空白；片段和全文采用相同规则。
            numeric_answer = _text(self._numeric_answer_text(answer))
            remainder = numeric_answer
            numeric_fragments = [_text(self._numeric_answer_text(row["answer_fragment"])) for row in rows]
            for fragment in sorted(numeric_fragments, key=len, reverse=True):
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
                         source_ids=sorted(all_sources), valid=True, binding_error="")
        except (ValueError, TypeError, KeyError, InvalidOperation, ArithmeticError) as exc:
            # 固定错误码帮助下一次修正；不回传异常中的源码或任意用户文本。
            check["binding_error"] = str(exc) if type(exc) is ValueError else "invalid_binding_value"
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

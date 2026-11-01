"""One bounded expression attempt, independent facts, and existing delivery."""
from __future__ import annotations

import copy
import json
import math
import re
import threading

from httpx import TimeoutException, TransportError
from openai import APIConnectionError, APITimeoutError

from shinku.agent.budget import bind_task_budget
from shinku.agent_runtime.contracts import canonical_hash
from shinku.llm.response_protocol import call_failure_metadata, note_call_failure, observe_response
from shinku.phase_timing import observation_context, inherit_observation, phase_scope
from shinku.llm.request_ledger import provider_call_context, bind_provider_calls
from .facts import (EXPRESSION_RULES, NEUTRAL_RULES, PHRASES, check_candidate, check_neutral,
                    candidate_checks, fact_blocks, identity, neutral_checks, private_prose, template, validate_result)
from .models import (EXPRESSION_RESPONSE_FAILURES, PARAGRAPH_REFERENCE_VERSION, ExpressionProfile, ModelReply)
from .natural import (RULES as PARAGRAPH_RULES, BODY_CONTEXT_RULES, CONTEXT_CHECK_RULES, CONNECTED_PROSE_RULES,
                      CONNECTED_PROSE_CHECK_RULES,
                      check_paragraphs, host_reference_ids, paragraph_blocks,
                      paragraph_checks, paragraph_fact_ids, prompt_blocks)
from .semantic_coverage import (EXPRESSION_RULES as SEMANTIC_EXPRESSION_RULES,
                               CHECK_RULES as SEMANTIC_CHECK_RULES, check_semantic_coverage,
                               semantic_requirements)


def exception_reason(exc):
    """Classify a bounded cause chain by type, never provider text."""
    seen, transport = set(), False
    for _ in range(8):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        if isinstance(exc, (TimeoutException, APITimeoutError)):
            return "expression_transport_timeout"
        if isinstance(exc, TimeoutError):
            return "expression_timeout"
        transport = transport or isinstance(exc, (TransportError, APIConnectionError))
        exc = exc.__cause__ or exc.__context__
    return "expression_transport_error" if transport else "expression_model_error"


def usage_summary(usage, *, calls=1):
    source = usage or {}
    safe = {"calls": calls, "known": False, "cost_known": False}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = source.get(name)
        if type(value) is int and value >= 0:
            safe[name], safe["known"] = value, True
    cost = source.get("cost")
    if (type(cost) in (float, int) and math.isfinite(cost) and cost >= 0
            and isinstance(source.get("currency"), str) and source["currency"] in {"USD", "CNY", "EUR"}):
        safe.update(cost=cost, currency=source["currency"], cost_known=True)
    return safe


def guard_reason(check_current):
    try:
        value = check_current()
        return value if isinstance(value, str) and (not value or re.fullmatch(r"[a-zA-Z0-9_:.-]{1,128}", value)) else "expression_guard_invalid"
    except Exception:
        return "expression_current_state_unavailable"


class ExpressionService:
    def __init__(self, archive, profile=None, *, material_resolver=None, material_validator=None):
        self.archive, self.profile = archive, profile or ExpressionProfile()
        self.material_resolver = material_resolver
        self.material_validator = material_validator
        self._slots = threading.BoundedSemaphore(2)
        self._active_lock = threading.Lock()
        self._active = 0

    @property
    def active_calls(self):
        with self._active_lock:
            return self._active

    @staticmethod
    def _discard(reason, **base):
        return {**base, "status": "discarded", "text": "", "fallback_reason": reason,
                "check": {"accepted": False, "reason": reason}}

    def _call(self, runtime, *, system, payload, timeout, record, scope, channel):
        if not callable(getattr(runtime, "call_expression_json", None)):
            return None, "expression_model_unavailable"
        if not self._slots.acquire(blocking=False):
            return None, "expression_call_capacity"
        try:
            self.archive.usage(record["expression_id"], scope=scope, channel=channel, usage=usage_summary(None))
        except Exception:
            self._slots.release()
            return None, "expression_archive_unavailable"
        done, box = threading.Event(), {}
        with self._active_lock:
            self._active += 1
        timing, provider = observation_context(), provider_call_context()
        def run():
            reply = None
            # 只续接诊断。表达调用仍使用原独立预算和超时，线程超时也不自动重试。
            with inherit_observation(timing), bind_provider_calls(*provider), phase_scope(
                    'expression' if channel == 'expression' else 'expression_check'), observe_response() as observation:
                phase = "service_call"
                try:
                    # The expression transport has its own explicit limits.
                    with bind_task_budget(None):
                        reply = runtime.call_expression_json(system_prompt=system,
                            user_prompt=json.dumps(payload, ensure_ascii=False), native_tools=None,
                            timeout_seconds=timeout, max_output_tokens=self.profile.max_output_tokens)
                    phase = "service_result"
                    if not isinstance(reply, ModelReply):
                        box["error"] = "expression_transport_invalid"
                    elif reply.failure_reason:
                        box["error"] = (reply.failure_reason if isinstance(reply.failure_reason, str)
                                        and reply.failure_reason in EXPRESSION_RESPONSE_FAILURES
                                        else "expression_transport_invalid")
                    else:
                        box["value"] = copy.deepcopy(reply.value)
                except Exception as exc:
                    note_call_failure(phase, exc)
                    box["error"] = exception_reason(exc)
                finally:
                    try:
                        counts = reply.usage if isinstance(reply, ModelReply) else observation.call_failure.get("reported_usage")
                        usage = usage_summary(counts)
                        if observation.call_failure:
                            usage["failure"] = copy.deepcopy(observation.call_failure)
                        self.archive.usage(record["expression_id"], scope=scope, channel=channel, usage=usage)
                    except Exception:
                        # Missing usage never permits another attempt.
                        pass
                    with self._active_lock:
                        self._active -= 1
                    self._slots.release()
                    done.set()
        threading.Thread(target=run, name="result-expression", daemon=True).start()
        if not done.wait(timeout):
            try:
                self.archive.usage(record["expression_id"], scope=scope, channel=channel,
                    usage={**usage_summary(None), "failure": call_failure_metadata("host_wait", TimeoutError())})
            except Exception:
                pass
            return None, "expression_timeout"
        return box.get("value"), box.get("error", "")

    def prepare(self, result, *, scope, check_current, recover_interrupted=False):
        source = copy.deepcopy(result)
        try:
            error = validate_result(source, scope)
        except (KeyError, TypeError, ValueError, OverflowError):
            error = "expression_result_invalid"
        error = error or guard_reason(check_current)
        if error:
            return self._discard(error)
        key = canonical_hash(identity(source))
        binding_source = {**identity(source), "result_hash": source["result_hash"]}
        if self.profile.format_version == 2:
            binding_source["paragraph_reference_version"] = PARAGRAPH_REFERENCE_VERSION
            binding_source["paragraph_presentation_version"] = self.profile.paragraph_presentation_version
        if self.profile.detail_policy_version:
            binding_source["detail_policy_version"] = self.profile.detail_policy_version
        if self.profile.semantic_review_version:
            binding_source["semantic_review_version"] = self.profile.semantic_review_version
        binding_hash = canonical_hash(binding_source)
        try:
            record, owned = self.archive.claim(key, scope=scope, result=source,
                config_hash=self.profile.config_hash, binding_hash=binding_hash,
                format_version=self.profile.format_version,
                detail_policy_version=self.profile.detail_policy_version,
                paragraph_presentation_version=self.profile.paragraph_presentation_version,
                semantic_review_version=self.profile.semantic_review_version,
                content_policy=self.profile.content_policy,
                processing_seconds=self.profile.timeout_seconds + self.profile.checker_timeout_seconds + 20)
        except Exception:
            return self._discard("expression_archive_unavailable")
        if record["result_hash"] != source["result_hash"]:
            return self._discard("expression_result_version_conflict", expression_id=key)
        if record["config_hash"] != self.profile.config_hash:
            return self._discard("expression_configuration_changed", expression_id=key)
        if not owned:
            if recover_interrupted:
                return self.archive.recover_interrupted(key, scope=scope)
            return record
        try:
            if self.profile.content_policy == "selective":
                # 原件先归档；自主表达不再恢复所有资料或建立逐项覆盖/二次模型审查。
                from .selective import prepare
                return prepare(self, source, record=record, scope=scope,
                               check_current=check_current, binding_hash=binding_hash)
            presentation = None
            reason = ""
            policy = source.get("presentation", {})
            if (self.material_resolver is not None and policy.get("schema_version") == 1
                    and policy.get("material_recovery", True) is True):
                try:
                    presentation = self.material_resolver(source, scope=scope,
                        still_authorized=lambda: not guard_reason(check_current))
                except Exception as exc:
                    from shinku.agent_runtime.contracts import AgentBoundaryError
                    code = exc.code if isinstance(exc, AgentBoundaryError) else "material_presentation_unavailable"
                    presentation = {"schema_version": 1, "source_result_hash": source["result_hash"],
                                    "status": "incomplete", "gaps": [{"reason": code}]}
                if presentation.get("source_result_hash") != source["result_hash"]:
                    raise ValueError("presentation_source_conflict")
                if presentation.get("gaps"):
                    reason = "expression_required_material_unavailable"
                binding_hash = canonical_hash({**binding_source, "presentation_hash": canonical_hash(presentation)})
                record = self.archive.bind_presentation(key, scope=scope, projection=presentation, binding_hash=binding_hash)
            natural = self.profile.format_version == 2
            blocks = (paragraph_blocks(source, presentation, detail_policy_version=self.profile.detail_policy_version,
                        presentation_version=self.profile.paragraph_presentation_version)
                      if natural else fact_blocks(source, presentation))
            fallback = template(blocks)
            payload = {"binding_hash": binding_hash, "fact_blocks": prompt_blocks(blocks) if natural else blocks}
            requirements = []
            if self.profile.semantic_review_version:
                semantic_reason, requirements = semantic_requirements(blocks)
                reason = reason or semantic_reason
                payload.update(semantic_review_version=self.profile.semantic_review_version,
                               semantic_requirements=requirements)
            payload["required_fact_ids"] = [row["id"] for row in blocks if row.get("required", True)]
            if natural:
                payload.update(schema_version=2, optional_fact_ids=[row["id"] for row in blocks if not row.get("required", True)],
                               host_reference_fact_ids=host_reference_ids(blocks))
                if self.profile.paragraph_presentation_version in (2, 3, 4, 5, 6, 7):
                    payload["paragraph_presentation_version"] = self.profile.paragraph_presentation_version
            else:
                payload.update(phrases=PHRASES, free_prose_allowed=self.profile.checker is not None,
                    layout_example={"binding_hash": binding_hash,
                        "layout": [{"phrase": "intro"}, *({"fact": value} for value in payload["required_fact_ids"])]})
            candidate = None
            if reason:
                pass
            elif natural and self.profile.checker is None:
                reason = "expression_neutral_check_required"
            elif len(json.dumps(payload, ensure_ascii=False)) + len(self.profile.style_prompt) > self.profile.max_input_chars:
                reason = "expression_input_oversized"
            else:
                rules = PARAGRAPH_RULES if natural else EXPRESSION_RULES
                if natural and self.profile.paragraph_presentation_version in (2, 3, 4, 5, 6, 7):
                    rules += BODY_CONTEXT_RULES
                    if self.profile.paragraph_presentation_version in (3, 4, 5, 6, 7):
                        from .verified_conditions import RULES as VERIFIED_CONTEXT_RULES
                        rules += VERIFIED_CONTEXT_RULES
                    if self.profile.paragraph_presentation_version in (4, 5, 6, 7):
                        from .readable_context import RULES as READABLE_CONTEXT_RULES
                        rules += READABLE_CONTEXT_RULES
                    if self.profile.paragraph_presentation_version in (5, 6, 7):
                        rules += CONNECTED_PROSE_RULES
                if natural and self.profile.paragraph_presentation_version in (6, 7):
                    from .persona_body import RULES as PERSONA_BODY_RULES
                    rules += PERSONA_BODY_RULES
                if self.profile.semantic_review_version:
                    rules += SEMANTIC_EXPRESSION_RULES
                candidate, reason = self._call(self.profile.runtime, system=rules + "\n表达风格：" + self.profile.style_prompt,
                    payload=payload, timeout=self.profile.timeout_seconds, record=record, scope=scope, channel="expression")
            candidate_hash = ""
            checks = None
            semantic_checks = None
            if candidate is not None:
                try:
                    candidate_hash = canonical_hash(candidate)
                except (TypeError, ValueError):
                    reason = "expression_format_invalid"
            text, prose = "", []
            if not reason:
                reason, text, prose = (check_paragraphs if natural else check_candidate)(candidate, blocks, binding_hash)
            if not reason and private_prose(prose, self.profile.style_prompt, full=natural):
                reason = "expression_private_context_leak"
            if not reason and prose:
                revoked = guard_reason(check_current)
                if revoked:
                    reason = revoked
                elif self.profile.checker is None:
                    reason = "expression_neutral_check_required"
                else:
                    selected_ids = (set(paragraph_fact_ids(candidate)) if natural else
                                    {row["fact"] for row in candidate["layout"] if "fact" in row})
                    selected_blocks = [row for row in blocks if row["id"] in selected_ids]
                    check_payload = {"binding_hash": binding_hash,
                        "fact_blocks": prompt_blocks(selected_blocks) if natural else selected_blocks, "candidate_text": text}
                    check_rules = NEUTRAL_RULES
                    if natural and self.profile.paragraph_presentation_version in (2, 3, 4, 5, 6, 7):
                        check_payload["context_fact_blocks"] = prompt_blocks(
                            [row for row in blocks if row["id"] not in selected_ids])
                        check_rules += CONTEXT_CHECK_RULES
                    if natural and self.profile.paragraph_presentation_version in (5, 6, 7):
                        check_rules += CONNECTED_PROSE_CHECK_RULES
                    if natural and self.profile.paragraph_presentation_version in (6, 7):
                        from .persona_body import CHECK_RULES as PERSONA_BODY_CHECK_RULES
                        check_rules += PERSONA_BODY_CHECK_RULES
                    if self.profile.semantic_review_version:
                        check_payload.update(semantic_review_version=self.profile.semantic_review_version,
                                             semantic_requirements=requirements)
                        check_rules += SEMANTIC_CHECK_RULES
                    if len(json.dumps(check_payload, ensure_ascii=False)) > self.profile.max_input_chars:
                        reason = "expression_check_input_oversized"
                    else:
                        verdict, reason = self._call(self.profile.checker, system=check_rules, payload=check_payload,
                            timeout=self.profile.checker_timeout_seconds, record=record, scope=scope, channel="check")
                        if not reason:
                            if self.profile.semantic_review_version:
                                reason, checks, semantic_checks = check_semantic_coverage(
                                    verdict, selected_blocks, binding_hash, text, requirements)
                            else:
                                checks = neutral_checks(verdict, selected_blocks, binding_hash)
                                reason = check_neutral(verdict, selected_blocks, binding_hash)
            current = guard_reason(check_current)
            fields = {"status": "discarded" if current else "fallback" if reason else "ready",
                "text": "" if current else fallback if reason else text,
                "check": {"accepted": not bool(reason or current), "reason": current or reason,
                          "mode": ("protected_values_neutral_and_semantic_coverage" if self.profile.semantic_review_version
                                   else "protected_values_and_neutral" if natural
                                   else "immutable_blocks_and_neutral" if prose else "immutable_blocks_and_catalog"),
                          "structure_checks": (paragraph_checks if natural else candidate_checks)(candidate, blocks, binding_hash)},
                "fallback_reason": current or reason, "candidate_hash": candidate_hash,
                # Rejected prose/tool requests may contain leaked prompts;
                # store their hash and code rather than the raw model response.
                "candidate": candidate if not reason and not current else None}
            if checks is not None:
                fields["check"]["neutral_checks"] = checks
            if semantic_checks is not None:
                fields["check"]["semantic_checks"] = semantic_checks
            return self.archive.finish(key, scope=scope, **fields)
        except Exception:
            # Malformed canonical fields/checker output are a boundary failure,
            # never permission to retry a model or an executor.
            try:
                return self.archive.finish(key, scope=scope, **self._discard("expression_result_fields_invalid"))
            except Exception:
                return self._discard("expression_archive_unavailable", expression_id=key)

    def read_details(self, expression_id, *, scope, check_current):
        """Read retained audit detail under the original current-state guard.

        No model, material paging, executor, delivery or budget replenishment.
        The existing scoped archive supplies the versioned immutable source.
        """
        reason = guard_reason(check_current)
        if reason:
            return self._discard(reason)
        try:
            record = self.archive.get(expression_id, scope=scope)
            if record is None:
                return self._discard("expression_record_missing")
            projection = record.get("presentation_projection")
            if self.material_validator is not None and projection:
                if self.material_validator(record["canonical_result"], projection, scope=scope,
                        still_authorized=lambda: not guard_reason(check_current)) is not True:
                    return self._discard("expression_material_no_longer_available")
            reason = guard_reason(check_current)
            if reason:
                return self._discard(reason)
            return {"status": "available", "expression_id": expression_id, "result_hash": record["result_hash"],
                "binding_hash": record["binding_hash"],
                "fact_blocks": fact_blocks(record["canonical_result"], projection)}
        except Exception:
            return self._discard("expression_archive_unavailable")

    def reserve_publication(self, record, *, scope, check_current):
        if record.get("status") not in {"ready", "fallback"} or not record.get("text"):
            return False
        if guard_reason(check_current):
            return False
        try:
            stored = self.archive.get(record["expression_id"], scope=scope)
            if not stored or stored["binding_hash"] != record.get("binding_hash"):
                return False
            if self.material_validator is not None and stored.get("presentation_projection"):
                if self.material_validator(stored["canonical_result"], stored["presentation_projection"], scope=scope,
                        still_authorized=lambda: not guard_reason(check_current)) is not True:
                    return False
            return self.archive.reserve_delivery(record["expression_id"], scope=scope, text=record["text"])
        except Exception:
            return False

    def record_delivery(self, record, *, scope, receipt):
        return self.archive.delivery(record["expression_id"], scope=scope, receipt=receipt)

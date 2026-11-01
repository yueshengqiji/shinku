"""Host presentation policy; audit controls never replace substantive facts."""
from __future__ import annotations

from collections.abc import Mapping

from shinku.agent_runtime.contracts import canonical_hash


def user_details(result):
    policy = result.get("presentation", {})
    return (policy.get("schema_version") == 1 and policy.get("policy_source") == "host_export"
            and type(policy.get("detail_policy_version")) is int and policy["detail_policy_version"] == 2)


def state_observation(result, fact, *, detail_policy_version=0):
    """Identify an opaque reference only when it matches the actual receipt.

    User-provided URLs, hashes, and other references are never audit controls.
    This checks presentation redundancy; the answer verifier owns acceptance.
    """
    answer = result.get("verification", {}).get("answer_evidence", {})
    if (not (user_details(result) or detail_policy_version == 2) or answer.get("kind") != "state"
            or result.get("verification", {}).get("accepted") is not True
            or result.get("execution", {}).get("status") != "completed"
            or not result.get("task_identity_missing") or result.get("task_id")):
        return None
    invocation = fact.get("invocation_id")
    if not invocation or fact.get("source") != "executor_receipt":
        return None
    for row in result.get("evidence", []):
        if (row.get("source") != "executor_receipt" or row.get("execution_status") != "ok"
                or row.get("invocation_id") != invocation):
            continue
        receipt = row.get("receipt", {}).get("self_state_receipt")
        if (not isinstance(receipt, Mapping) or receipt.get("schema_version") != 1
                or receipt.get("fresh") is not True or receipt.get("complete") is not True
                or receipt.get("source") not in {"provider", "fresh_provider"}):
            continue
        if (fact.get("reference") == "self_state:" + str(receipt.get("read_id", ""))
                and isinstance(receipt.get("scope"), str) and receipt["scope"]
                and fact.get("label") == "宿主状态快照（" + receipt["scope"] + "）"
                and receipt.get("snapshot_sha256") == canonical_hash(fact.get("value"))):
            return receipt
    return None


def redundant_state_answer(result, *, detail_policy_version=0):
    facts = result.get("facts", [])
    answer = result.get("verified_answer", {})
    return (bool(facts) and answer.get("validated") is True
            and all(state_observation(result, row, detail_policy_version=detail_policy_version) is not None for row in facts))


_PROJECTION_REASONS = {"text_limit": "文字", "list_limit": "列表", "mapping_limit": "字段",
                       "depth_limit": "嵌套内容", "latest_receipt_limit": "调用记录"}
_GAP_FIELDS = {"path", "kind", "total", "retained", "omitted", "reason", "retrieval"}


def gap_summary(value, human):
    """Keep scope and substantive gaps exact; counts remain in the audit copy."""
    if isinstance(value, list):
        parts = [gap_summary(row, human) for row in value]
        return "；".join(text for text, _ in parts), [v for _, values in parts for v in values]
    if (isinstance(value, dict) and set(value).issubset(_GAP_FIELDS)
            and isinstance(value.get("path"), str) and value.get("reason") in _PROJECTION_REASONS):
        path = value["path"] or "原始资料"
        return path + "处的" + _PROJECTION_REASONS[value["reason"]] + "未完整保留", [path]
    return human(value), [value]


def stop_summary(code):
    # Unrecognized codes remain mandatory verbatim. A diagnostic code is not
    # enough to invent a cause, recommend a retry, or declare a side effect.
    return {
        "completion_receipt_missing": "尚未取得所需的完成凭据，不能确认目标已完成。",
        "completion_attempt_stale": "完成凭据属于旧次执行，不能确认本次目标已完成。",
        "budget_exhausted": "本次执行额度已耗尽，未完成项仍须处理。",
        "task_canceled": "本次任务已取消；取消不证明操作已撤销或进程已清理。",
        "public_query_model_call_failed": "模型调用失败，当前查询未取得完整答案。",
        "public_query_model_call_timeout": "模型调用超时，当前查询未取得完整答案。",
        "public_query_model_output_length": "模型输出因长度限制未完整结束，当前查询未取得完整答案。",
        "public_query_model_output_filtered": "模型未返回可用的完整输出，当前查询未取得完整答案。",
        "public_query_model_output_incomplete": "模型输出未完整结束，当前查询未取得完整答案。",
    }.get(code)

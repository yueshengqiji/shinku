"""Lossless host summaries of identical unsuccessful call observations.

Grouping never infers retries, their cause, recovery, or task acceptance.
The canonical archive retains every original invocation and receipt.
"""
from shinku.agent_runtime.contracts import canonical_hash


RECEIPT_FIELDS = ("verified_effects", "completion_targets", "bytes", "sha256", "url",
                  "http_status", "result_valid")
RECEIPT_LABELS = {"verified_effects": "已核对的效果", "completion_targets": "目标",
                  "bytes": "字节数", "sha256": "内容摘要", "url": "来源地址",
                  "http_status": "HTTP状态", "result_valid": "回执结果有效性"}
STATUS_LABELS = {"error": "报错", "failed": "失败", "unknown": "状态未知",
                 "denied": "被拒绝", "canceled": "已取消", "timeout": "超时"}


def unsuccessful_groups(result, safe):
    """Group only consecutive identical receipts from the trusted exporter.

    Different receipt fields, tools, states and intervening calls remain
    separate, including details that are retained solely in the archive.
    """
    groups = []
    previous = None
    for i, row in enumerate(result.get("evidence", [])):
        status = row.get("execution_status") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("source") != "executor_receipt"
                or status not in STATUS_LABELS):
            previous = None
            continue
        receipt = row.get("receipt") or {}
        signature = canonical_hash({"tool": row.get("tool", ""), "status": status, "receipt": receipt})
        if previous == signature:
            groups[-1]["member_ids"].append(f"evidence_{i}")
        else:
            groups.append({"tool": safe(row.get("tool", "")), "status": status,
                           "receipt": safe({k: receipt[k] for k in RECEIPT_FIELDS if k in receipt}),
                           "member_ids": [f"evidence_{i}"]})
        previous = signature
    return groups

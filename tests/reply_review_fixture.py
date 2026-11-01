"""供非语义专项显式选用的离线来源审核回执，不用于真实模型或事实质量验收。

调用方须先把本次调用计入自己的完整记录，再处理此回执；不能用它隐藏审核成本。
这里只让路由、权限、送达等独立契约的固定模型夹具接上既有审核协议。
来源正确性、拒绝、引用和改写测试使用各自的专用审核器，不使用默认pass。
"""
import json


def dialogue_review_result(options):
    """只应答明确的现有中立审核调用；其他生成/路由/工具调用返回None。"""
    if not str(options.get("prompt_cache_key", "")).endswith(":dialogue-review:v5"):
        return None
    payload = json.loads(options["user_prompt"])
    result = {"input_id": payload["input_id"], "verdict": "pass", "issues": []}
    if "available_evidence" in payload:
        result["claims"] = []
    return result


def wire_dialogue_review_result(payload):
    """SDK传输夹具显式启用时，按实际完整输入ID构造同一回执。"""
    messages = payload.get("messages", [])
    if not any(row.get("role") == "system" and "独立的对话依据审查者" in str(row.get("content", ""))
               for row in messages):
        return None
    body = next(row["content"] for row in reversed(messages) if row.get("role") == "user")
    if isinstance(body, list):
        body = "\n".join(part["text"] for part in body if part.get("type") == "text")
    source = json.loads(body)
    value = {"input_id": source["input_id"], "verdict": "pass", "issues": []}
    if "available_evidence" in source:
        value["claims"] = []
    return value

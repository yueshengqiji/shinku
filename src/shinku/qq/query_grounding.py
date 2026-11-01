"""普通聊天出口的查询证据边界；查询执行与来源核验仍属于已有只读/委托链路。

这里只识别明确查询要求和当前查询自述，拒绝把零执行的正文当成查证结果。
它不选择工具、不授予权限，也不以历史助手正文或模型填写的状态作为回执。
兼容文本检查不是完整语义分类器；未匹配的表达仍受主生成的依据提示约束。
"""
from __future__ import annotations

import re

from shinku.agent.request_intent import assertion_text, is_reply_composition
from shinku.agent.request_scope import RequestScope
from shinku.agent.reply_body import is_silence_marker


QUERY_GUIDANCE = (
    "【本轮查询依据】用户明确要求搜索、联网查证时，应使用当前获准的查询能力或提交完整查证目标；"
    "已有印象、旧助手回复、未执行的工具建议都不能满足这项要求。"
    "不认识名称或用户纠正你时，保留原名称和不确定性，不把没有记起说成官方不存在。"
    "没有实际查询结果时，不说查过、搜到、翻过或根据搜索结果；缺少对象可澄清，"
    "缺少能力或查询失败就如实说明尚未核实。选择工具和权限仍遵循本轮接口。"
)
QUERY_REPAIR = (
    "【查询尚未执行】上一份只是普通正文，本轮没有查询回执，不能直接结束明确的查询要求，"
    "也不能声称已经查过。保留完整原请求，使用本轮获准的查询接口或委托接口；"
    "缺少必要对象可返回等待澄清，无法执行则报告未核实。只有这一次纠正机会，不扩大权限。"
)
QUERY_NOT_EXECUTED_REPLY = "这次还没有实际执行查询，暂时不能确认结果。"

# 只拦截明确的执行表述；不根据产品名、来源数量或步骤数替模型选择任务路线。
_QUERY_REQUEST = re.compile(
    r"(?:^|[，,：:]|你)(?:请|麻烦|帮我|替我|先|再|直接|上网|联网|到网上|去|\s)*"
    r"(?:搜搜|搜一下|搜索|检索|(?:上网|联网|到网上)(?:查|核实)|查证一下)"
    r"|(?:上网|联网|到网上)(?:帮我|替我|去|再|\s)*(?:查查|查一下|查询|核实)", re.I)
_NEGATED_OR_DISCUSSION = re.compile(
    r"不要|不用|不必|无需|别|禁止|怎么|如何|为什么|(?:他说|她说|提到|这句话|这段话)|"
    r"^(?:如果|假如|假设|要是|等你|待你)")
_QUERY_CLAIM = re.compile(
    r"^(?:(?:嗯|好|好的|唔)[…\s]*)?(?:我\s*)?(?:(?:刚才|刚刚|现在|已经|又|去|也|确实)\s*)*"
    r"(?:查(?:了|过|到)|搜(?:了|过|到)|搜索(?:了|过|到)|查阅(?:了|过)|翻(?:了|过))"
    r"|^(?:根据|依据)(?:我)?(?:刚才|本轮|刚刚)?(?:的)?(?:搜索|检索|查询)(?:结果|到的)")


def _clauses(text):
    # 复用既有断言视图处理引用与代码；这里只匹配分句开头的当前查询自述。
    value = assertion_text(text)
    value = re.sub(r"‘[^’\n]*’", "", value)
    for sentence in re.split(r"[。！？!?；;\n]+", value):
        if sentence.lstrip().startswith(">"):
            continue
        yield from (part.strip(" \t…~～") for part in re.split(r"[，,：:]+", sentence))


def claims_current_query(text: str) -> bool:
    return any(_QUERY_CLAIM.search(part) for part in _clauses(text))


def requires_query(text: str) -> bool:
    if is_reply_composition(text):
        return False
    for part in _clauses(text):
        match = _QUERY_REQUEST.search(part)
        if match is None or _NEGATED_OR_DISCUSSION.search(part[:match.start()]):
            continue
        # “搜索引擎怎么工作”讨论的是概念；“搜索一下怎么使用”仍是本轮查询要求。
        if part[match.start():].startswith(("搜索引擎", "搜索功能", "搜索工具", "搜索是什么意思")):
            continue
        return True
    return False


def unverified_query_reason(context, decision) -> str:
    """仅供尚未执行工具的直复路径使用；成功查询走原有证据核验出口。

    请求只取宿主绑定的当前主消息，不把群聊背景或引用消息提升为要求。
    失败/澄清可以结束等待，但其正文同样不能伪造本轮查询事实。
    """
    if decision.kind not in {"final", "wait"} or is_silence_marker(decision.text):
        return ""
    scope = context.get("agent_request_scope")
    current = next((text for key, text in scope.requests if key == scope.primary_message_id), "") if isinstance(scope, RequestScope) else ""
    if is_reply_composition(current):
        return ""
    if claims_current_query(decision.text):
        return "chat_query_claim_without_evidence"
    if not isinstance(scope, RequestScope):
        return ""
    if decision.kind == "final" and decision.outcome not in {"failed", "blocked"} and requires_query(current):
        return "chat_query_not_executed"
    return ""

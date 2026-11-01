"""Content-free, host-verified failure explanations for bounded public queries.

Never copy tool arguments, page bodies, or raw exceptions into user diagnostics.
Unknown failures remain unknown; an HTTP response is not a DNS diagnosis.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_FIELDS = frozenset({"url", "wait_ms", "selector", "find_text", "offset", "max_chars", "query", "limit", "site", "provider"})
_LABELS = {
    "tool_arguments_malformed": "工具调用参数无法解析，未执行补造的空参数调用",
    "bad_args": "参数校验未通过，工具没有执行；未提供更具体的参数失败原因",
    "source_url_missing": "缺少来源 URL（url），网页读取没有执行",
    "invalid_max_chars": "max_chars 必须是 256–8000 之间的整数，网页读取没有执行",
    "invalid_offset": "offset 必须是 0–1000000 之间的整数，网页读取没有执行",
    "invalid_wait_ms": "wait_ms 不能转换为整数，网页读取没有执行",
    "invalid_selector": "selector 必须是至多 500 字符的字符串，网页读取没有执行",
    "invalid_find_text": "find_text 必须是非空且至多 160 字符的字符串，网页读取没有执行",
    "public_url_scheme_denied": "来源不是公开 HTTP(S) 网页，读取被安全边界拒绝",
    "public_url_credentials_denied": "来源 URL 含有账号或密码，读取被安全边界拒绝",
    "public_url_port_denied": "来源使用非标准网页端口，读取被安全边界拒绝",
    "public_url_non_public_address": "来源指向本机、内网或非公共地址，读取被安全边界拒绝",
    "public_url_invalid": "来源 URL 无法解析，网页读取没有执行",
    "public_url_dns_failed": "来源域名的地址解析失败，无法确认可安全读取的公共地址",
    "public_url_public_dns_unavailable": "系统返回代理 Fake-IP，但独立公共 DNS 未取得可用地址，读取被安全边界拒绝",
    "browser_navigation_failed": "网页跳转失败",
    "browser_page_read_failed": "网页内容读取失败",
    "browser_start_failed": "Chromium 启动失败",
    "browser_context_closed": "浏览器或页面已关闭",
    "browser_navigation_timeout": "网页跳转超时",
    "browser_operation_timeout": "浏览器操作超时",
    "browser_operation_failed": "浏览器操作异常",
    "browser_network_error": "浏览器访问目标站遇到网络层错误，不等于整台设备断网",
    "browser_proxy_error": "浏览器连接代理失败，不等于整台设备断网",
    "browser_dns_error": "浏览器无法解析目标域名",
    "browser_connection_refused": "目标站连接被拒绝",
    "browser_connection_timeout": "连接目标站超时",
    "browser_tls_error": "访问目标站时 TLS 或证书校验失败",
    "browser_connection_reset": "连接目标站时连接被中断",
    "browser_network_disconnected": "浏览器网络接口报告断开，不等于已证明整台设备断网",
    "browser_request_blocked": "网页请求被浏览器侧拦截",
    "browser_navigation_aborted": "页面导航被取消",
    "browser_navigation_network_error": "Chromium 页面请求失败，未取得正文",
    "browser_disabled": "浏览器功能在配置中关闭",
    "playwright_package_missing": "当前 Python 环境没有 Playwright 包；快轨不会擅自安装",
    "chromium_browser_missing": "Playwright 对应版本的 Chromium 不存在；快轨不会擅自安装",
    "playwright_driver_unavailable": "Playwright 驱动不可用",
    "sensitive_query_blocked": "搜索词疑似含敏感信息，为避免外发已取消搜索",
    "search_no_results": "搜索站点明确返回没有匹配结果",
    "search_scope_unmet": "没有取得符合指定来源域名范围的候选，范围外链接已排除",
    "search_provider_blocked": "搜索工具报告访问受限；是否为验证页需结合站点回执确认，未取得搜索结果",
    "search_result_parse_failed": "搜索页面已打开，但没有识别到结果卡片；不能当作没有结果",
    "search_provider_unavailable": "搜索站点访问失败，未取得搜索结果；不能据此认定整台设备断网",
    "search_relevance_unverified": "搜索候选无法支持本次查询，备用来源也未取得可交付的相关候选",
    "source_time_budget_exhausted": "公开来源读取的累计时间预算已用完，未重置预算或继续扩展操作",
    "source_http_tls_error": "公开来源的 TLS 或证书校验失败，没有关闭证书校验",
    "source_http_timeout": "匿名 HTTP 读取来源超时",
    "source_http_transport_error": "匿名 HTTP 客户端到公开来源的访问失败",
    "source_redirect_limit": "来源重定向次数超出上限或缺少可校验的目标",
    "source_response_too_large": "来源响应超过读取大小上限，已停止读取",
    "source_content_type_unsupported": "来源不是支持的公开文本网页，未当作有效正文",
    "source_content_selector_not_found": "静态来源中未找到指定正文区域",
    "source_body_unavailable": "来源没有返回可核实的有效正文",
    "source_challenge_blocked": "公开来源要求人机验证，未解验证码或交付验证页为正文",
    "source_offset_out_of_range": "来源正文读取位置超出实际长度",
    "source_find_text_not_found": "指定正文范围未找到原文字符串，不能推断相关事实不存在",
    "handler_error": "工具处理器发生异常；未提供可安全转述的具体原因",
    "empty_result": "工具没有返回执行结果",
    "tool_reported_error": "工具返回失败，但未提供可验证的详细原因",
    "source_access_unverified": "网页回执未确认来源访问成功，未取得可核实正文",
}
_PHASES = frozenset({"browser_start", "navigation", "wait", "page_read", "argument_validation", "public_url"})
_EXCEPTIONS = frozenset({"RuntimeError", "TypeError", "ValueError", "TimeoutError", "OSError", "PermissionError",
                         "ConnectTimeout", "ReadTimeout", "ConnectionError", "HTTPError", "BrowserDependencyError", "Error"})
_SEARCH_PROVIDERS = {"anysearch": "AnySearch", "bing": "Bing", "duckduckgo": "DuckDuckGo"}
_SEARCH_OUTCOMES = {
    "not_configured": "搜索 API 尚未配置密钥",
    "auth_failed": "搜索 API 认证失败或账户访问被拒绝",
    "quota_exhausted": "搜索 API 额度不足",
    "rate_limited": "搜索 API 请求受到限流",
    "daily_limit": "本机搜索 API 当日请求上限已达到",
    "api_error": "搜索 API 返回业务错误，未取得候选",
    "timeout": "搜索 API 请求超时",
    "results": "已取得候选结果",
    "no_results": "页面明确显示没有匹配结果",
    "parse_failed": "页面已打开，但未识别到可用结果卡片",
    "unavailable": "访问失败，详细原因未确认",
    "scope_unmet": "候选不符合指定来源范围",
    "relevance_unverified": "候选未通过查询相关性筛选",
    "budget_exhausted": "搜索时间预算已用完",
}


def _search_attempt_feedback(value: Any) -> tuple[str, list[dict[str, Any]]]:
    """Render only known receipt enums and numeric HTTP statuses, never page text."""
    if not isinstance(value, (list, tuple)):
        return "", []
    lines = []
    safe_attempts = []
    for attempt in value[:3]:
        if not isinstance(attempt, Mapping):
            continue
        provider = attempt.get("provider")
        outcome = attempt.get("outcome")
        if (not isinstance(provider, str) or provider not in _SEARCH_PROVIDERS
                or not isinstance(outcome, str) or outcome not in {*_SEARCH_OUTCOMES, "blocked"}):
            continue
        safe = {"provider": provider, "outcome": outcome}
        status = attempt.get("status")
        label = _SEARCH_PROVIDERS[provider]
        if type(status) is int and 100 <= status <= 599:
            safe["status"] = status
            label += f"（HTTP {status}）"
        if outcome == "blocked":
            block_reason = attempt.get("block_reason")
            if block_reason == "challenge_page":
                safe["block_reason"] = block_reason
                text = "检测到人机验证页"
            elif type(status) is int and status in {401, 403, 407, 429}:
                safe["block_reason"] = "http_status"
                text = {401: "站点要求认证", 403: "访问被拒绝", 407: "代理要求认证", 429: "请求受到限流"}[status]
            else:
                text = "工具报告访问受限，是否为验证页未确认"
        else:
            text = _SEARCH_OUTCOMES[outcome]
        lines.append(f"{label}：{text}")
        safe_attempts.append(safe)
    return "；".join(lines), safe_attempts


def tool_failure(call: Any, envelope: Any, handler: Any) -> dict[str, Any]:
    """Use structured receipts only, never scrape a cause from external prose."""
    data = envelope.data if isinstance(envelope.data, Mapping) else {}
    updates = data.get("state_updates")
    updates = updates if isinstance(updates, Mapping) else {}
    status = updates.get("http_status")
    if type(status) is int and 400 <= status <= 599:
        return {"code": "source_http_error", "http_status": status,
                "text": f"网页读取失败：来源返回 HTTP {status}，未取得可核实正文"}
    if envelope.status == "ok" and updates.get("site_access_ok") is not False:
        return {}
    code = str(updates.get("error_code") or data.get("code") or "tool_reported_error")
    field = ""
    if code == "bad_args":
        diagnostic = getattr(handler, "agent_validation_diagnostic", None)
        if (isinstance(diagnostic, Mapping) and isinstance(diagnostic.get("code"), str)
                and diagnostic.get("code") in _LABELS):
            code = diagnostic["code"]
            field = diagnostic.get("field") if isinstance(diagnostic.get("field"), str) and diagnostic.get("field") in _FIELDS else ""
    if envelope.status == "ok":
        code = "source_access_unverified"
    if code not in _LABELS:
        code = "tool_reported_error"
    result = {"code": code, "text": _LABELS[code]}
    if call.name == "private_web_search":
        detail, attempts = _search_attempt_feedback(updates.get("provider_attempts"))
        if detail:
            result["text"] = "未取得可用搜索结果；" + detail
            result["provider_attempts"] = attempts
    if field:
        result["field"] = field
    phase = updates.get("error_phase")
    if isinstance(phase, str) and phase in _PHASES:
        result["phase"] = phase
    exception = updates.get("exception_type") or data.get("error_type")
    if isinstance(exception, str) and exception in _EXCEPTIONS:
        result["exception_type"] = exception
        result["text"] += f"；异常类型：{exception}"
    return result


def argument_fields(call: Any) -> list[str]:
    """Known field names only; even unknown keys may contain secrets."""
    return sorted(set(call.arguments) & _FIELDS) if isinstance(call.arguments, Mapping) else []


def rejected_followup(decision: Any, *, safe_names: set[str], registered_names: set[str]) -> dict[str, Any]:
    calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
    names = [call.name for call in calls]
    link = decision.task_link
    compact = len(calls) == 1 and calls[0].name == "__shinku_tool_dispatch__"
    if compact:
        selector = calls[0].arguments
        raw_name = selector.get("tool_name") or selector.get("target_tool") or selector.get("name")
        names = [raw_name] if isinstance(raw_name, str) else []
        if link is None:
            link = selector.get("task_link")
    if decision.task_contract is not None:
        code, text = "task_contract_not_allowed", "后续决定申请了任务契约；公开查询不能据此新建或扩大任务，已拒绝执行"
    elif isinstance(link, Mapping) and link.get("mode") not in (None, "none"):
        code, text = "task_link_not_none", "后续决定申请新建或续跑任务；本轮没有这项授权，已拒绝执行"
    elif isinstance(link, Mapping) and link.get("track") not in (None, "public_query"):
        code, text = "track_not_public_query", "后续决定要求切换执行轨道；本轮公开查询没有执行该切换"
    elif any(name not in safe_names for name in names):
        code = "tool_not_public_query"
        known = [name for name in names if name in registered_names]
        suffix = f"（{'、'.join(known[:3])}）" if known else ""
        text = f"后续决定申请了快轨不支持的工具{suffix}，已拒绝执行"
    elif compact:
        code, text = "invalid_tool_call_format", "后续工具调用格式不完整，未执行；本轮格式纠正机会已用完或该调用不能安全纠正"
    else:
        code, text = "followup_scope_rejected", "后续决定未通过公开查询的执行范围或权限校验，已停止执行"
    return {"code": code, "text": text, "tools": [name for name in names if name in registered_names][:3]}


class PublicQueryFeedback:
    def __init__(self):
        self.errors: list[dict[str, Any]] = []
        self.search_succeeded = False

    def note_malformed_native(self) -> dict[str, Any]:
        error = {"tool": "", "code": "native_tool_arguments_malformed",
                 "text": "模型供应商返回的原生工具调用无法解析；没有把底层填入的空参数伪调用当作真实工具选择执行"}
        if error not in self.errors:
            self.errors.append(error)
        return error

    def observe(self, call: Any, envelope: Any, handler: Any) -> dict[str, Any]:
        error = tool_failure(call, envelope, handler)
        if error:
            label = {"private_browser_open": "网页读取", "private_web_search": "公开搜索",
                     "private_current_time": "时间查询"}.get(call.name, "只读工具")
            entry = {"tool": call.name, **error}
            if not entry["text"].startswith(label):
                entry["text"] = f"{label}失败：{entry['text']}"
            self.errors.append(entry)
            return entry
        if call.name == "private_web_search" and envelope.status == "ok":
            self.search_succeeded = True
        return {}

    def explain(self, text: str, *, reason: str, source_read: bool, rejection: dict[str, Any] | None = None,
                model_error: str = "") -> str:
        """Keep earlier causal evidence even when a later host guard stops work."""
        lines = [str(text or "").strip()]
        if self.search_succeeded and not source_read:
            lines.append("搜索已执行，但尚未核实来源正文，不能确认答案。")
        for error in self.errors[-3:]:
            lines.append(f"{error['text']}（{error['code']}）。")
        if rejection:
            lines.append(f"{rejection['text']}（{rejection['code']}）。")
        if model_error:
            lines.append(model_error)
        if not source_read and reason != "lightweight_public_query_fast_track" and not self.search_succeeded:
            lines.append("本轮没有取得可核实的完整答案。")
        return "\n".join(dict.fromkeys(line for line in lines if line))

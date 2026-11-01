"""日常读取的显式宿主目录：只声明边界，不创建网络、浏览器或私用工具。

可信启动器先拥有真实适配器及其生命周期，再用本模块发布有界 schema。
来源接续只接受当前原请求的 URL，或本回合真实搜索／正文读取回执中的链接；
模型文字、历史摘要和其他工具结果不能自行扩大来源。实际公开地址与重定向检查
仍由网页适配器执行，目录声明本身不授予登录、任意 HTTP、路径或命令权限。
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from shinku.agent.request_scope import RequestScope
from shinku.agent.research import source_url, visible_urls

from .interaction_tools import ChatReadCapability


def _object(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


def _source_guard(arguments, context, *, search, page):
    """来源权限归宿主本回合的请求和回执所有；不从答复文字猜可读 URL。"""
    scope = context.get("agent_request_scope")
    target = source_url(arguments.get("url"))
    if not isinstance(scope, RequestScope) or not target:
        return False
    if any(target in visible_urls(text) for text in scope.user_messages()):
        return True
    receipts = context.get("_chat_read_receipts", ())
    if not isinstance(receipts, (list, tuple)):
        return False
    for receipt in receipts:
        if (not isinstance(receipt, Mapping) or receipt.get("source") != "executed_read_adapter"
                or receipt.get("tool") not in {search, page}):
            continue
        result = receipt.get("result")
        updates = result.get("state_updates") if isinstance(result, Mapping) else None
        if not isinstance(updates, Mapping) or updates.get("private_tool_error"):
            continue
        if receipt["tool"] == search:
            rows = updates.get("results", ())
        else:
            # 仅真正读到正文的页面能给出下一跳；失败页面不能替模型铸造链接权限。
            if updates.get("site_access_ok") is not True or not updates.get("page_text"):
                continue
            links = updates.get("source_links", ())
            rows = [*(links if isinstance(links, (list, tuple)) else ()), {"url": updates.get("url")}]
        if isinstance(rows, (list, tuple)) and any(isinstance(row, Mapping)
                and source_url(row.get("url")) == target for row in rows):
            return True
    return False


def daily_read_capabilities(*, search: str, page: str, clock: str, browser_status: str,
                            processes: str, network: str) -> dict[str, ChatReadCapability]:
    """发布宿主明确选中的六种读取；名称不会反向触发导入或注册工具。"""
    names = (search, page, clock, browser_status, processes, network)
    if len(set(names)) != len(names):
        raise ValueError("daily_read_names_must_be_distinct")
    capabilities = [
        ChatReadCapability(search,
            "检索公开资料、最新消息和日常事实，可直接查证，不必为单纯查询创建委托任务。"
            "搜索卡片只是候选，相关事实须继续读取来源正文；保留原问题的对象、时间和站点。",
            _object({"query": {"type": "string", "minLength": 1, "maxLength": 300},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                "site": {"type": "string", "minLength": 1, "maxLength": 253},
                "provider": {"type": "string", "enum": ["auto", "anysearch", "bing", "duckduckgo"], "maxLength": 16}}, ["query"]),
            route="public_query", max_calls=2, max_result_bytes=64000, max_seconds=30),
        ChatReadCapability(page,
            "匿名读取当前请求给出的公开 URL，或本回合实际搜索／正文回执中的来源链接。"
            "不使用持久登录或 Cookie；正文过长可用 offset 接续，来源拒绝或缺失时说明缺口。",
            _object({"url": {"type": "string", "minLength": 1, "maxLength": 2000},
                "max_chars": {"type": "integer", "minimum": 256, "maximum": 8000},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
                "selector": {"type": "string", "maxLength": 500},
                "find_text": {"type": "string", "minLength": 1, "maxLength": 160},
                "wait_ms": {"type": "integer", "minimum": 0, "maximum": 3000}}, ["url"]),
            route="public_query", max_calls=3, max_result_bytes=64000, max_seconds=30,
            argument_guard=partial(_source_guard, search=search, page=page)),
        ChatReadCapability(clock, "读取当前聊天宿主的本机时间和时区，不调用模型。", _object(),
            max_result_bytes=8000, max_seconds=2),
        ChatReadCapability(browser_status,
            "只读核对当前聊天宿主的浏览器依赖和代理路由配置；不能证明网站或模型服务可用，缺依赖只报告。",
            _object(), max_result_bytes=16000, max_seconds=15),
        ChatReadCapability(processes,
            "只读查看本机进程摘要，不读取任意文件或执行命令，不停止进程。limit 必填，最多40项。",
            _object({"limit": {"type": "integer", "minimum": 1, "maximum": 40}}, ["limit"]),
            max_result_bytes=32000, max_seconds=15),
        ChatReadCapability(network,
            "只读检查本机到固定国内／海外 HTTPS 目标的响应与耗时；不接受自定义目标。"
            "结果不代表网速、所有网站或模型服务健康。", _object(), max_result_bytes=16000, max_seconds=10),
    ]
    return {cap.name: cap for cap in capabilities}

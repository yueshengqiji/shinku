"""公共日常目录的参数和来源准入；真实私用适配器另在本机隔离链路验收。"""
from dataclasses import replace

import pytest

from shinku.agent.request_scope import RequestScope
from shinku.qq.daily_reads import daily_read_capabilities
from shinku.qq.interaction_tools import validate_arguments
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation
from test_chat_self_state import assembled
from test_agent_host_bindings import isolated
from test_qq_tool_decision_boundary import Chat


NAMES = dict(search="search_public", page="read_public", clock="clock_now",
             browser_status="browser_dependencies", processes="process_summary", network="network_probe")


@pytest.fixture
def caps():
    return daily_read_capabilities(**NAMES)


def context(text="核对示例游戏的发售消息", **updates):
    return {"agent_request_scope": RequestScope("fixture", "private:fixture", "current", (("current", text),)), **updates}


def receipt(tool="search_public", *, source="executed_read_adapter", **updates):
    return {"tool": tool, "source": source, "result": {"state_updates": updates}}


def test_catalog_has_only_explicit_distinct_reads_and_bounded_schemas(caps):
    assert set(caps) == set(NAMES.values())
    assert {name for name, cap in caps.items() if cap.route == "public_query"} == {NAMES["search"], NAMES["page"]}
    for cap in caps.values():
        assert cap.parameters["additionalProperties"] is False
        assert 1 <= cap.max_calls <= 3 and cap.max_seconds <= 30 and cap.max_result_bytes <= 64000
        assert not ({"command", "path", "script", "headers", "cookies", "target"} & set(cap.parameters["properties"]))
    with pytest.raises(ValueError, match="distinct"):
        daily_read_capabilities(**{**NAMES, "page": NAMES["search"]})


@pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "x", "limit": 6},
    {"query": "x", "limit": True}, {"query": "x", "site": ""}, {"query": "x", "provider": "arbitrary"},
    {"query": "x", "headers": {}}, {"query": "x", "cookies": "secret"}])
def test_search_arguments_cannot_expand_source_or_result_bounds(caps, arguments):
    assert not caps[NAMES["search"]].admits(arguments, context())


def test_reader_accepts_only_exact_original_url_and_normalizes_citation_punctuation(caps):
    cap = caps[NAMES["page"]]
    ctx = context("请核对 https://Source.example/article#part。")
    assert cap.admits({"url": "https://source.example/article"}, ctx)
    assert not cap.admits({"url": "https://source.example/article/other"}, ctx)
    assert not cap.admits({"url": "https://source.example.evil/article"}, ctx)
    assert not cap.admits({"url": "https://user:password@source.example/article"}, ctx)
    assert not cap.admits({"url": "https://source.example/article"}, {"user_prompt": ctx["agent_request_scope"].requests[0][1]})


def test_actual_search_receipt_then_page_link_are_valid_sources(caps):
    cap = caps[NAMES["page"]]
    ctx = context(_chat_read_receipts=[receipt(results=[{"url": "https://source.example/news"}])])
    assert cap.admits({"url": "https://source.example/news"}, ctx)
    ctx["_chat_read_receipts"].append(receipt(NAMES["page"], site_access_ok=True,
        page_text="实际来源正文", url="https://source.example/news", source_links=[{"url": "https://official.example/post"}]))
    assert cap.admits({"url": "https://official.example/post", "offset": 256}, ctx)
    assert not cap.admits({"url": "https://unobserved.example/post"}, ctx)


@pytest.mark.parametrize("bad", [
    receipt(source="model_text", results=[{"url": "https://source.example/"}]),
    receipt(tool="other_reader", results=[{"url": "https://source.example/"}]),
    receipt(private_tool_error=True, results=[{"url": "https://source.example/"}]),
    receipt(NAMES["page"], site_access_ok=False, page_text="拒绝", source_links=[{"url": "https://source.example/"}]),
    receipt(NAMES["page"], site_access_ok=True, page_text="", source_links=[{"url": "https://source.example/"}]),
])
def test_model_other_tools_and_failed_reads_do_not_mint_source_access(caps, bad):
    assert not caps[NAMES["page"]].admits({"url": "https://source.example/"}, context(_chat_read_receipts=[bad]))


@pytest.mark.parametrize("bad", [None, "forged", {}, [None, "not-a-receipt"]])
def test_malformed_receipts_fail_closed(caps, bad):
    assert not caps[NAMES["page"]].admits({"url": "https://source.example/"}, context(_chat_read_receipts=bad))


@pytest.mark.parametrize("arguments", [{"url": "https://source.example/", "max_chars": 8001},
    {"url": "https://source.example/", "wait_ms": 3001}, {"url": "https://source.example/", "offset": -1},
    {"url": "https://source.example/", "find_text": ""}, {"url": "https://source.example/", "cookie": "session"}])
def test_page_arguments_keep_real_adapter_normalizer_within_bounds(caps, arguments):
    assert not caps[NAMES["page"]].admits(arguments, context("读取 https://source.example/"))


@pytest.mark.parametrize("name", ["clock", "browser_status", "network"])
def test_fixed_observations_do_not_accept_arbitrary_targets(caps, name):
    cap = caps[NAMES[name]]
    assert cap.admits({}, context())
    assert not cap.admits({"url": "http://localhost/admin"}, context())
    assert not cap.admits({"command": "anything"}, context())


@pytest.mark.parametrize("limit", [None, 0, 41, True, "20"])
def test_process_summary_is_explicitly_bounded(caps, limit):
    assert not caps[NAMES["processes"]].admits({} if limit is None else {"limit": limit}, context())


def test_self_check_reports_only_bound_chat_reads_and_distinguishes_registration(assembled, caps):
    bridge = assembled(Chat([]), chat_read_capabilities=caps)
    handler = bridge.tool_host.registry.get("read_self_state")
    _, envelope = execute_invocation(ToolInvocation("read_self_state", {"scope": "tools"}),
        handlers={handler.tool_type: handler}, context=context())
    tools = envelope.data["state_updates"]["self_state"]["tools"]
    assert tools["registered"] == ["read_self_state"]
    assert tools["chat_read_catalog"] == ["read_self_state"]  # 单独声明六个名字不会虚构六个真实读取器。
    assert tools["registration_grants_execution"] is False and tools["delegation_configured"] is False


def test_reduced_host_scope_and_limits_remain_explicit(caps):
    narrow = replace(caps[NAMES["page"]], max_calls=1, max_result_bytes=8000)
    assert narrow.max_calls == 1 and narrow.max_result_bytes == 8000
    assert not validate_arguments({"url": "https://source.example", "path": "/synthetic"}, narrow.parameters)

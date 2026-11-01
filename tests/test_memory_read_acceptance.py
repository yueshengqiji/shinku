"""Model-selected memory must transfer its evidence obligation after a real read."""
import json

import pytest

from shinku.agent_runtime.execution import ExecutionProfile
from shinku.memory import MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.assembly_tools import memory_read_capabilities
from shinku.qq.commands import QQCommandState
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from test_context_projection import NOW, put
from test_private_dialogue_continuity import turn

SOURCE = "写长篇文章时我需要安静，别放背景音乐；休息时可以听歌。"
ANSWER = "写长篇文章时不放背景音乐，休息时可以听歌。"
QUERY = "工作时那个不受打扰的安排还适用吗？"


class Chat:
    def __init__(self, review, *, mode="retrieve", read=True, answer=ANSWER, model_retrieval=True,
                 dispatch_from_offered_schema=False):
        self.review=review; self.mode=mode; self.read=read; self.answer=answer
        self.model_retrieval=model_retrieval
        self.dispatch_from_offered_schema=dispatch_from_offered_schema
        self.calls=[]; self.main_calls=0; self.payload=None

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kw):
        self.calls.append(kw)
        if str(kw.get("prompt_cache_key", "")).endswith(":dialogue-review:v5"):
            # 普通直复的来源审核与历史读取账本是不同职责；本夹具只核对宿主协议。
            payload=json.loads(kw["user_prompt"])
            return {"input_id":payload["input_id"], "verdict":"pass", "issues":[]}
        if "历史答案复核" in "\n".join(kw.get("system_extra_blocks", [])):
            self.payload=json.loads(kw["user_prompt"])
            if self.review=="timeout":
                raise TimeoutError("isolated provider timeout")
            if self.review=="empty":
                return {}
            if self.review=="rejected":
                return {"disposition":"unverified", "claims":[], "coverage":[]}
            row=next(r for r in self.payload["actual_records"] if SOURCE in r["text"])
            return {"disposition":"found", "claims":[{"source_id":row["id"], "quote":SOURCE,
                    "answer_fragment":self.answer}],
                    "coverage":[] if self.review=="missing_coverage" else
                    [{"field":"原问题", "supported":True, "claim_indices":[0]}]}
        self.main_calls+=1
        if self.main_calls==1 and self.read:
            arguments={"query":"写长篇 安静 背景音乐", "mode":self.mode, "limit":4}
            if self.model_retrieval:
                arguments["recorded_time"]={"start":"", "end":""}
            offered=(kw.get("native_tools") or [])
            if (self.dispatch_from_offered_schema and offered
                    and offered[0]["function"]["name"] == "__shinku_tool_dispatch__"):
                return {"kind":"tool", "tool_call":{"name":"__shinku_tool_dispatch__",
                    "arguments":{"request_kind":"query", "tool_name":"memory_search",
                        "arguments":json.dumps(arguments,ensure_ascii=False,separators=(",",":"))}}}
            return {"kind":"tool", "tool_call":{"name":"memory_search", "arguments":arguments}}
        return {"kind":"final", "text":self.answer}


class NoExecution:
    def call_chat_json(self, **kw):
        pytest.fail("Memory lookup must not start task execution")


@pytest.fixture
def bridge_factory(tmp_path):
    resources=[]
    def build(entry, review, *, strategy="model", mode="retrieve", read=True, answer=ANSWER, read_capabilities=None):
        root=tmp_path/str(len(resources));root.mkdir()
        policy=MemoryPolicy(event_ttl_seconds=0, projection=ProjectionPolicy(enabled=True,retrieval_strategy=strategy))
        memory=MemoryService(store=MemoryStore(root/"memory.sqlite3"),router=MemoryRouter(),policy=policy,clock=lambda:NOW)
        put(memory.store,"prior",SOURCE,age=40*86400)
        chat=Chat(review, mode=mode, read=read, answer=answer, model_retrieval=strategy=="model")
        host=ToolHost(ToolRegistry({"memory_search":MemorySearchTool(memory)}))
        coordinator=AgentTaskCoordinator(max_workers=1) if entry=="task" else None
        bridge=QQAgentBridge(runtime=chat,system_prompt="Synthetic dialogue persona.",sender=None,
            memory_service=memory,tool_host=host,allowed_tool_names={"memory_search"},
            master_user_ids=("owner",),command_state=QQCommandState(permission_mode="ultra"),
            policy=ExecutionPolicy(risk_by_tool={"memory_search":"low"}),tool_decision_mode=entry,
            task_contract_enabled=True,task_coordinator=coordinator,
            execution_profile=ExecutionProfile(NoExecution()) if coordinator else None,
            chat_read_capabilities=(memory_read_capabilities(None,memory,host)
                                    if read_capabilities is None else read_capabilities))
        resources.append((bridge,memory,coordinator))
        return bridge,chat,memory,coordinator
    yield build
    for bridge,memory,coordinator in resources:
        memory.projection.stop();memory.store.stop_embedding_worker();memory.lifecycle_review.stop()
        if coordinator:coordinator.shutdown(wait=True)
        bridge.task_store.close();bridge.conversation_runtime.close()


@pytest.mark.parametrize("entry", ["legacy","task"])
@pytest.mark.parametrize("mode", ["recent","retrieve","summary"])
def test_memory_reply_uses_read_and_single_generation_without_review(bridge_factory,entry,mode):
    bridge,chat,_,coordinator=bridge_factory(entry,"must_not_be_called",mode=mode)
    reply=bridge(turn(QUERY,mid="current"))
    assert reply.agent_status=="completed" and reply.final_text==ANSWER
    assert len(chat.calls)==2 and chat.payload is None
    assert not reply.sent and not reply.background_task_id
    assert coordinator is None or not coordinator.list()
    assert all("history-evidence" not in call.get("prompt_cache_key","") for call in chat.calls)


@pytest.mark.parametrize("entry", ["legacy","task"])
def test_plain_continuation_has_one_call_and_no_history_contract(bridge_factory,entry):
    bridge,chat,_,_=bridge_factory(entry,"must_not_be_called",read=False,answer="好，我们一起去。")
    result=bridge(turn("那就一起去吧。",mid="current"))
    assert result.agent_status=="completed" and result.final_text==chat.answer
    assert len(chat.calls)==1 and chat.payload is None


@pytest.mark.parametrize("entry", ["legacy","task"])
def test_forgotten_tool_source_is_rejected_without_rewrite(bridge_factory,entry):
    bridge,chat,memory,_=bridge_factory(entry,"must_not_be_called")
    original=chat.call_chat_json
    def generate(**options):
        value=original(**options)
        if chat.main_calls==2:
            memory.store.forget(scope="private:owner",record_id="",query="写长篇",now=NOW)
        return value
    chat.call_chat_json=generate
    result=bridge(turn(QUERY,mid="current"))
    assert result.agent_status=="failed" and ANSWER not in result.final_text
    assert len(chat.calls)==2 and chat.payload is None


@pytest.mark.parametrize("entry", ["legacy","task"])
def test_history_read_cannot_authorize_fake_file_write(bridge_factory,entry):
    bridge,chat,_,_=bridge_factory(entry,"must_not_be_called",answer="我已经删除了文件 /tmp/synthetic.txt。")
    result=bridge(turn(QUERY,mid="current"))
    assert result.agent_status=="failed" and "已经删除" not in result.final_text
    assert len(chat.calls)==2


def test_model_arguments_cannot_opt_out_of_host_evidence_policy(bridge_factory):
    _,_,memory,_=bridge_factory("legacy","empty")
    handler=MemorySearchTool(memory)
    assert handler.agent_history_answer_verification_required is True
    assert handler.normalize_call({"query":"写长篇", "verification_required":False}) is None


@pytest.mark.parametrize("entry", ["legacy", "task"])
def test_actual_entry_offers_and_transfers_recorded_time_contract(bridge_factory, entry):
    bridge, chat, memory, _ = bridge_factory(entry, "accepted")
    original = memory.inspect
    observed = []
    def inspect(**kwargs):
        result = original(**kwargs)
        observed.append((kwargs.get("recorded_time"), result["coverage"].copy()))
        return result
    memory.inspect = inspect
    reply = bridge(turn(QUERY, mid="recorded-time"))
    assert reply.agent_status == "completed"
    assert observed and observed[0][0] == {"start": "", "end": ""}
    assert observed[0][1]["time_filter_source"] == "explicit_recorded_time"
    assert observed[0][1]["requested_time_window"] == [None, None]
    schema = chat.calls[0].get("native_tools") or []
    reader = next((item["function"] for item in schema
                   if item["function"]["name"] == "memory_search"), None)
    if reader is not None:
        offered = reader["parameters"]
        assert offered["required"] == ["recorded_time"]
        assert set(offered["properties"]["recorded_time"]["required"]) == {"start", "end"}
    else:
        # The legacy first turn may compact the catalog into its registered
        # dispatcher. Verify that the required nested schema survives there.
        assert any(item["function"]["name"] == "__shinku_tool_dispatch__" for item in schema)
        from shinku.memory import MemorySearchTool
        recorded_time = MemorySearchTool(memory).chat_read_capability().parameters["properties"]["recorded_time"]
        encoded = json.dumps(recorded_time, ensure_ascii=False, separators=(",", ":"))
        catalog = "\n".join(chat.calls[0].get("system_extra_blocks") or [])
        assert f"recorded_time (必填，完整参数定义):{encoded}" in catalog


@pytest.mark.parametrize("route", ["compact", "full"])
def test_actual_read_and_recorded_time_receipt_follow_the_offered_route(bridge_factory, route):
    bridge, chat, memory, _ = bridge_factory("legacy", "accepted")
    chat.dispatch_from_offered_schema = True
    if route == "compact":
        # Force the configured compact branch while retaining the real router
        # schema, catalog builder, JSON-string parser, registry, and handler.
        bridge.async_agent_enabled = True
    else:
        # Make the real compact catalog larger than the full schema so the
        # normal size decision offers the canonical memory_search schema.
        original = bridge._compact_tool_dispatch_spec
        bridge._compact_tool_dispatch_spec = lambda specs: (
            *original(specs)[:1], original(specs)[1] + (" oversized catalog" * 1000))

    original_inspect = memory.inspect
    observed = []
    def inspect(**kwargs):
        result = original_inspect(**kwargs)
        observed.append({"recorded_time":kwargs.get("recorded_time"),
            "coverage":result["coverage"], "records":result["records"]})
        return result
    memory.inspect = inspect

    reply = bridge(turn(QUERY, mid=f"route-{route}"))

    assert reply.agent_status == "completed" and reply.final_text == ANSWER
    offered = chat.calls[0]["native_tools"]
    expected_name = "__shinku_tool_dispatch__" if route == "compact" else "memory_search"
    assert [item["function"]["name"] for item in offered] == [expected_name]
    assert observed and observed[0]["recorded_time"] == {"start":"", "end":""}
    assert observed[0]["coverage"]["time_filter_source"] == "explicit_recorded_time"
    assert observed[0]["coverage"]["requested_time_window"] == [None, None]
    assert any(row["text"] == SOURCE for row in observed[0]["records"])
    assert len(chat.calls) == 2

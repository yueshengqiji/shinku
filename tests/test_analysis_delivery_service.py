"""真实隔离工作区工具贯穿会话、计划修订、持久化、验收和导出。"""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentTurnRequest, AgentExecutor, AgentService, ExecutionGrant, ExecutionProfile
from shinku.angelos.host import ServiceHost
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.tools.workspace import build_workspace_tools
from test_agent_runtime_boundary import settled


ANSWER = "本地记忆由存储、检索和来源校验组成。当前读到的实现将来源归属作为读取条件；运行时规模未查询，无法确认。"


def contract(root, command=None):
    rows = [
        {"id": "read", "description": "读取源码", "effect": "read", "targets": [str(root/"source.txt")]},
        {"id": "extra", "description": "辅助阅读", "effect": "read", "targets": [str(root/"optional.txt")]},
        {"id": "answer", "description": "架构说明", "effect": "local_answer", "targets": [],
         "checks": {"required_fields": ["存储", "检索", "运行时规模"], "min_sources": 6}},
    ]
    if command:
        rows.append({"id": "query", "description": "辅助查询", "effect": "command_run", "targets": [],
                     "checks": {"command": command, "exit_code": 0}})
    return {"goal": "了解自身记忆实现", "deliverables": ["架构说明"], "scope": ["隔离工作区"], "requirements": rows}


@contextmanager
def service_case(root, responder, additional=None):
    (root/"source.txt").write_text("本地记忆由存储、检索和来源校验组成。", encoding="utf-8")
    tools = {t.tool_type: t for t in build_workspace_tools(root, authorize=lambda ctx, write: not write)}
    tools.update(additional or {})
    class Runtime:
        calls = []
        def chat_supports_native_tools(self): return True
        def call_chat_json(self, **kw):
            self.calls.append(kw)
            if any("本地答案证据复核" in b for b in kw.get("system_extra_blocks", [])):
                raise AssertionError("ordinary question unexpectedly invoked a quality model")
            return responder(len(self.calls), kw)
    runtime = Runtime()
    grant = ExecutionGrant("isolated-analysis", frozenset(tools), permission_mode="ultra",
        policy=ExecutionPolicy(risk_by_tool={name: "low" for name in tools}, approval_required_by_risk=frozenset()),
        budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 24, "max_active_seconds": 30})
    service = AgentService(executor=AgentExecutor(ToolHost(ToolRegistry(tools)), ExecutionProfile(runtime)),
        authorize=lambda *_: grant, task_store=AgentTaskStore(root/"tasks.sqlite"),
        coordinator=AgentTaskCoordinator(max_workers=1))
    # 保留共享分层验收策略的兼容测试；默认独立宿主改由直接交付测试覆盖。
    host = ServiceHost(service, requester="fixture", reply_delivery=False)
    try:
        yield service, host.port("one"), runtime
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("replan", [False, True])
def test_conversation_plan_does_not_require_six_sources_or_optional_file(tmp_path, replan):
    first = contract(tmp_path)
    revised = deepcopy(first)
    revised["requirements"] = [r for r in revised["requirements"] if r["id"] != "extra"]
    revised["requirements"][-1]["checks"]["min_sources"] = 1
    def respond(n, kw):
        if n == 1:
            return {"tool_call": {"type": "declare_task_contract", "task_contract": first}}
        if replan and n == 2:
            return {"tool_call": {"type": "declare_task_contract", "task_contract": revised}}
        if n == (3 if replan else 2):
            return {"tool_call": {"name": "workspace_read_lines", "arguments": {"path": "source.txt"}}}
        return {"kind": "final", "text": ANSWER, "task_outcome": "completed"}
    with service_case(tmp_path, respond) as (service, port, runtime):
        handle = port.submit(AgentTurnRequest("question", "你看一下你本地记忆是什么架构"))
        view = settled(port, handle.task_id)
        result = port.result(handle.task_id)
        assert view["status"] == ("completed" if replan else "partial"), result
        assert result["execution_report"]["text"] == ANSWER
        assert result["verification"]["layers"]["quality_review"] == "not_requested"
        assert result["verification"]["layers"]["plan_gaps"] == ([] if replan else ["extra"])
        assert not result.get("verified_answer")
        assert len(runtime.calls) == (4 if replan else 3)
        saved = service.task_store.get(handle.task_id, requester="fixture", session="angelos:one")
        assert saved["analysis_plan_origin"]["requirements"]
        assert len(saved["contract_history"]) == (2 if replan else 1)
        if replan:
            assert saved["contract_history"][-1] == first
        # 重开数据库只恢复归属与原件，不重放读取或模型；跨请求使用由绑定门拒绝。
        reopened = AgentTaskStore(tmp_path/"tasks.sqlite")
        try:
            loaded = reopened.get(handle.task_id, requester="fixture", session="angelos:one")
            assert loaded["analysis_plan_origin"] == saved["analysis_plan_origin"]
            assert loaded["contract_history"] == saved["contract_history"]
        finally:
            reopened.close()

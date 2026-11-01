"""Host routing is mechanical; the model still supplies whole-goal intent."""

from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentDecision
from shinku.qq.host_routing import resolve_host_route
from shinku.qq.tool_selection import resolve_compact_tool_dispatch
from shinku.tools.invocation import ToolInvocation


def dispatch(kind=None, **extra):
    args = {"tool_name": "fixture", "arguments": "{}", **extra}
    if kind is not None:
        args["request_kind"] = kind
    result, valid = resolve_compact_tool_dispatch(
        AgentDecision.tool(ToolInvocation("__shinku_tool_dispatch__", args)), available_tool_names={"fixture"})
    assert valid
    return result


def resolve(decision, candidates=(), allowed=False):
    return resolve_host_route(decision, context={"agent_task_candidates": candidates},
        handlers={"fixture": SimpleNamespace()},
        observation=lambda d, **kw: allowed and (d.task_link or {}).get("track") == "perception",
        public_query=lambda *a, **kw: False)


@pytest.mark.parametrize("kind", [None, "task"])
def test_omission_and_action_goal_never_auto_fast_track(kind):
    result = resolve(dispatch(kind), allowed=True)
    assert result.task_link == {"mode": "new"}


def test_query_requires_host_admission():
    assert resolve(dispatch("query"), allowed=True).task_link == {"mode": "none", "track": "perception"}
    assert resolve(dispatch("query"), allowed=False).task_link == {"mode": "new"}


def test_continuation_only_binds_a_unique_scoped_candidate():
    assert resolve(dispatch("continue"), [{"task_id": "one"}]).task_link == {"mode": "continue", "task_id": "one"}
    assert resolve(dispatch("continue"), [{"task_id": "one"}, {"task_id": "two"}]).kind == "wait"
    assert resolve(dispatch("continue")).kind == "wait"
    assert resolve(dispatch("task"), [{"task_id": "one"}]).task_link == {"mode": "new"}


def test_explicit_target_is_left_for_scoped_ledger_validation():
    assert resolve(dispatch("continue", task_id="foreign"), [{"task_id": "one"}]).task_link == {
        "mode": "continue", "task_id": "foreign"}


def test_model_contract_cannot_enter_query_fast_path():
    decision = dispatch("query").with_task_metadata({"task_contract": {"goal": "fixture"}})
    result = resolve(decision, allowed=True)
    assert result.task_link == {"mode": "new"} and result.task_contract == decision.task_contract

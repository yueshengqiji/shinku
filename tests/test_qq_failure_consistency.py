from shinku.agent.task_state import TaskContractError, validate_contract
from shinku.agent.loop import AgentDecision
from shinku.qq.trace import TurnTraceStore
from shinku.tasks.coordinator import AgentTaskCoordinator
from test_agent_contract_repair import DeleteRecorder, Runtime, bot, tool_decision, turn, contract
from test_agent_async_processor import wait_for


def test_validation_path_identifies_actual_field_not_private_value(tmp_path):
    value = contract()
    value["requirements"][0]["checks"] = {"min_sources": "PRIVATE_VALUE"}
    try:
        validate_contract(value)
        assert False
    except TaskContractError as error:
        assert str(error) == "invalid_result_checks"
        assert error.field_path == "requirements[0].checks.min_sources"


def test_synchronous_and_background_preparation_failure_share_status_and_feedback(tmp_path):
    results = []
    for asynchronous in (False, True):
        recorder = DeleteRecorder()
        initial = tool_decision()
        initial["task_contract"]["requirements"][0]["checks"] = {"min_sources": "PRIVATE_VALUE"}
        bridge = bot(Runtime({}), recorder)
        bridge._run_chat_decision = lambda **kw: (AgentDecision.from_value(initial), {recorder.tool_type})
        bridge.trace_store = TurnTraceStore(path=tmp_path / (str(asynchronous) + ".json"))
        coordinator = AgentTaskCoordinator()
        bridge.task_coordinator = coordinator
        bridge.async_agent_enabled = asynchronous
        try:
            result = bridge.handle_turn(turn())
            if asynchronous:
                record = wait_for(coordinator, result.background_task_id, "failed")
                results.append((record["status"], record["result_summary"], record["error"]))
            else:
                results.append((result.agent_status, result.final_text, result.reason))
            assert recorder.calls == []
            path = bridge.trace_store.path
            restored = TurnTraceStore(path=path).snapshot()[0]
            failure = next(stage for stage in restored["stages"] if stage.get("status") == "validation_failed")
            assert failure["field_path"] == "requirements[0].checks.min_sources"
            assert "PRIVATE_VALUE" not in path.read_text(encoding="utf-8")
            assert restored["agent_status"] == "failed"
        finally:
            coordinator.shutdown(wait=True)
    assert results[0] == results[1]
    assert "重新授权" not in results[0][1]


def test_preliminary_memory_is_not_misreported_as_finished_history_lookup():
    from shinku.qq.failure import task_preparation_failure
    report = task_preparation_failure("contract_repair_failed", recalled_candidates=5)
    assert report.status == "failed"
    assert "读到了一些相关记忆" in report.text and "还没核对" in report.text
    assert report.phase == "task_preparation"

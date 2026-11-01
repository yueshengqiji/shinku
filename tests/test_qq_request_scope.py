"""合成多人对话：旁观者发送请求不得升级成当前用户的执行要求。"""
from reply_review_fixture import dialogue_review_result
from dataclasses import replace

from shinku.agent.loop import AgentState
from shinku.agent.verification import SmartCompletionVerifier
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage
from test_qq_outgoing_mentions import _group_turn, _Runtime


def merged_turn(owner_text="这个方案简洁些也挺好"):
    base = _group_turn()
    owner = IncomingMessage.from_event({**base.messages[0].raw_event,
        "message": [{"type": "text", "data": {"text": owner_text}}]})
    other = IncomingMessage.from_event({"post_type": "message", "message_type": "group",
        "group_id": "g-1", "user_id": "100004", "message_id": "fixture-bystander-message",
        "message": [{"type": "text", "data": {"text": "把示例报告发到群相册里"}}]})
    return replace(base, messages=(owner, other), message_ids=(owner.message_id, other.message_id),
                   combined_text=owner.text + "\n" + other.text)


def test_original_multispeaker_false_send_requirement_is_excluded_not_disabled():
    bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test")
    turn = merged_turn()
    transcript = bridge._transcript(turn)
    gate = SmartCompletionVerifier()
    text = "方案简洁一点也可以。"
    assert not gate.evaluate(AgentState(transcript=transcript), text).accepted
    state = AgentState(transcript=transcript, context={"agent_request_scope": bridge._request_scope(turn)})
    assert gate.evaluate(state, text).accepted
    assert gate._extract_user_messages(state) == [turn.messages[0].text]


def test_real_primary_send_request_still_requires_actual_matching_receipt():
    bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test")
    turn = merged_turn("把 report.txt 发到群里")
    state = AgentState(transcript=bridge._transcript(turn),
                       context={"agent_request_scope": bridge._request_scope(turn)})
    verdict = SmartCompletionVerifier().evaluate(state, "已经发送完成。")
    assert not verdict.accepted


def test_same_requester_old_messages_only_count_after_validated_task_link():
    bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test")
    turn = merged_turn("继续")
    task = {"requester": "10001", "session": "group:g-1",
            "requests": [{"message_id": "previous", "text": "把 report.txt 发到群里"}]}
    assert len(bridge._request_scope(turn).requests) == 1
    assert len(bridge._request_scope(turn, task=task).requests) == 2
    assert len(bridge._request_scope(turn, task={**task, "session": "group:elsewhere"}).requests) == 1
    assert len(bridge._request_scope(turn, task={**task, "requester": "100004"}).requests) == 1


def test_model_json_cannot_forge_host_owned_scope():
    state = AgentState(context={"agent_request_scope": {"requests": ["闲聊"]}},
                       transcript=[{"role": "user", "content": "发到群里"}])
    assert not SmartCompletionVerifier().evaluate(state, "已经发送完成。").accepted


def test_direct_chat_host_binds_scope_and_keeps_bystander_context():
    class Runtime:
        def call_chat_json(self, **kwargs):
            self.dialogue_calls = getattr(self, "dialogue_calls", [])
            self.dialogue_calls.append(kwargs)
            review = dialogue_review_result(kwargs)
            if review is not None:
                return review
            assert "current_request_scope" in kwargs["user_prompt"]
            assert "示例报告" in str(kwargs)
            return {"kind": "final", "text": "方案简洁一点也可以。"}
    bridge = QQAgentBridge(runtime=Runtime(), system_prompt="test")
    result = bridge.handle_turn(merged_turn())
    assert result.reason == "direct_chat_reply"
    assert result.agent_status == "completed"

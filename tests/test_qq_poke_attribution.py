"""A notice addressed elsewhere must not re-enter via autonomous attention."""
import pytest

from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionPolicy, AttentionSignals
from shinku.qq.message import IncomingMessage
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.turns import QQTurn


BOT = "3599477026"
OTHER = "3034389702"


def event(target=OTHER, *, shape="group_poke", group=True):
    value = {"post_type": "notice", "notice_type": shape, "self_id": BOT,
             "operator_id": "2406043", "user_id": "2406043", "time": 100}
    if shape == "notify":
        value["sub_type"] = "poke"
    if group:
        value["group_id"] = "872732158"
    if target is not None:
        value["target_id"] = target
    return value


def adapter(mode="turn_state"):
    return QQIngressAdapter(attention_policy=AttentionPolicy(autonomous_enabled=True,
        mode=mode, autonomous_probability=1.0, random_source=lambda: 0.0))


@pytest.mark.parametrize("mode", ["turn_state", "legacy_probability"])
@pytest.mark.parametrize("shape", ["group_poke", "poke", "notify"])
@pytest.mark.parametrize("target", [OTHER, None])
def test_other_or_unknown_poke_cannot_be_resurrected_by_any_autonomous_mode(mode, shape, target):
    subject = adapter(mode)
    result = subject.ingest(event(target, shape=shape), at=100)
    assert result.route.reason == "poke_not_for_bot"
    assert not result.scheduled and result.attention.reason == "poke_not_for_bot"
    assert subject.flush() is None


@pytest.mark.parametrize("mode", ["turn_state", "legacy_probability"])
def test_recent_bot_conversation_cannot_lend_ownership_to_someone_elses_poke(mode):
    message = IncomingMessage.from_event(event())
    route = ReplyRoutePolicy().decide(message)
    result = adapter(mode).attention_policy.evaluate(message, route,
        signals=AttentionSignals(recent_directed_turn=True, recent_bot_exchange=True), now=100)
    assert not result.should_schedule and result.reason == "poke_not_for_bot"


@pytest.mark.parametrize("group", [True, False])
@pytest.mark.parametrize("shape", ["group_poke", "poke", "notify"])
def test_real_poke_to_bot_remains_explicit_with_autonomous_disabled(group, shape):
    subject = QQIngressAdapter(attention_policy=AttentionPolicy(autonomous_enabled=False))
    result = subject.ingest(event(BOT, group=group, shape=shape), at=100)
    assert result.scheduled and result.attention.immediate and result.route.reason == "qq_poke"


class Runtime:
    def __init__(self):
        self.calls = []
    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "final", "text": "看到了。", "task_outcome": "completed"}


class NoMemory:
    def __init__(self):
        self.accesses = []

    def __getattr__(self, name):
        self.accesses.append(name)
        raise AssertionError(f"poke must not access memory: {name}")


def turn(message):
    return QQTurn(message.conversation_id, message.conversation_type, message.message_id,
        (message.message_id,), message.text, (message,), attention_mode="immediate")


@pytest.mark.parametrize("target", [OTHER, None])
def test_direct_bridge_entry_also_rejects_other_poke_without_model_memory_or_send(target):
    runtime = Runtime()
    sent = []
    memory = NoMemory()
    subject = QQAgentBridge(runtime=runtime, system_prompt="fixture", bot_ids=(BOT,),
                            sender=sent.append, memory_service=memory)
    result = subject.handle_turn(turn(IncomingMessage.from_event(event(target))))
    assert result.reason == "poke_not_for_bot" and result.delivery_status == "suppressed"
    assert runtime.calls == [] and sent == [] and memory.accesses == []


def test_bot_poke_prompt_identifies_actual_validated_target_and_keeps_it_transient():
    runtime = Runtime()
    memory = NoMemory()
    subject = QQAgentBridge(runtime=runtime, system_prompt="fixture", bot_ids=(BOT,), memory_service=memory)
    message = IncomingMessage.from_event(event(BOT))
    result = subject.handle_turn(turn(message))
    assert result.agent_status == "completed" and len(runtime.calls) == 1
    blocks = "\n".join(runtime.calls[0]["system_extra_blocks"])
    assert BOT in blocks and "已核对" in blocks and "不要调用工具" in blocks
    assert memory.accesses == []


@pytest.mark.parametrize("empty_alias", ["", "   "])
def test_camel_case_target_uses_the_same_normalized_identity_as_message_decoder(empty_alias):
    value = event(BOT)
    value["target_id"] = empty_alias
    value["targetId"] = BOT
    result = adapter().ingest(value, at=100)
    assert result.route.reason == "qq_poke" and result.scheduled
    bridge = QQAgentBridge(runtime=Runtime(), system_prompt="fixture", bot_ids=(BOT,))
    assert bridge._event_context(turn(result.message))["target_id"] == BOT


def test_conflicting_target_aliases_do_not_guess_that_bot_was_poked():
    value = event(BOT)
    value["targetId"] = OTHER
    result = adapter().ingest(value, at=100)
    assert not result.scheduled and result.route.reason == "poke_not_for_bot"


def test_ignored_poke_does_not_pollute_or_flush_a_pending_real_message():
    subject = adapter()
    message = {"post_type": "message", "message_type": "group", "group_id": "872732158",
        "self_id": BOT, "user_id": "2406043", "message_id": "real-message",
        "message": [{"type": "text", "data": {"text": "真红你在吗"}}]}
    assert subject.ingest(message, at=100).scheduled
    ignored = subject.ingest(event(), at=100.1)
    assert not ignored.scheduled and ignored.closed_batch is None
    batch = subject.flush()
    assert [m.message_id for m in batch.messages] == ["real-message"]

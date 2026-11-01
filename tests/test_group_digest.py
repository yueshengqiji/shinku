"""合成群聊回顾资料与真实OneBot适配器回归；不调用模型或QQ服务。"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sqlite3
from unittest.mock import Mock

import pytest

from shinku.memory import MemoryStore
from shinku.conversation import ConversationRuntime, ConversationScope
from shinku.qq.group_digest import DigestPolicy, read_digest_material, validated_digest
from shinku.qq.group_digest_host import DigestSchedule, process_group_digest


@pytest.fixture
def source(tmp_path):
    now = datetime(2026, 10, 23, 20, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    path = tmp_path / "memory.sqlite3"
    store = MemoryStore(path)
    yield store, path, now


def populate(source, *, group="g", count=36, participants=3, repeat=False, step=40, age_days=0, excluded=False):
    store, _path, now = source
    start = int(now.replace(hour=10, minute=0).timestamp()) - age_days * 86400
    for i in range(count):
        store.record_event(conversation_id=group, scope=f"group:{group}", layer="observation",
            text="重复通知内容已经处理完成" if repeat else f"我们来讨论第 {i} 个选项，相关细节还需要一起确定。",
            source_id=f"{group}:{age_days}:{i}", created_at=start+i*step,
            metadata={"source": "qq_passive_observation", "sender_id": f"member-{i % participants}",
                "sender_name": f"合成成员{i % participants}", "is_bot": excluded})


def material(source, **kwargs):
    return read_digest_material(source[1], recipient="owner", allowed_groups=("g",), now=source[2], **kwargs)


def answer(payload):
    group = payload["groups"][0]
    return {"input_id": payload["input_id"], "decision": "send", "recipient_available": True,
        "message": "今天群里主要讨论了选项的取舍。大家还在核对细节，尚未形成最终结论。",
        "topics": [{"group_id": group["group_id"], "source_ids": [r["source_id"] for r in group["records"][:2]],
            "summary": "讨论选项，尚未定案"}]}


@pytest.mark.parametrize("options", [dict(count=29), dict(participants=1), dict(repeat=True),
    dict(step=1), dict(step=3600), dict(age_days=1), dict(excluded=True)])
def test_inactive_or_repeated_material_does_not_qualify(source, options):
    populate(source, **options)
    assert material(source) is None


def test_scope_expiry_and_disabled_source_are_not_topics(source):
    populate(source, group="elsewhere")
    assert material(source) is None
    populate(source)
    assert material(source)
    with sqlite3.connect(source[1]) as conn:
        conn.execute("UPDATE memory_events SET expires_at=? WHERE scope='group:g'", (int(source[2].timestamp()),))
    assert material(source) is None


def test_recent_private_activity_defers_but_other_person_does_not(source):
    populate(source)
    store, _path, now = source
    store.record_turn(conversation_id="other", scope="private:master", role="user", content="别打扰我", source_id="other", created_at=int(now.timestamp()))
    assert material(source)
    store.record_turn(conversation_id="owner", scope="private:master", role="user", content="我还有一件事要说", source_id="owner", created_at=int(now.timestamp()))
    assert material(source) is None


def test_exclusions_and_empty_allowlist_fail_closed(source):
    populate(source)
    assert material(source, excluded_senders=("member-0", "member-1")) is None
    assert read_digest_material(source[1], recipient="owner", allowed_groups=(), now=source[2]) is None
    assert read_digest_material(source[1].with_name("missing"), recipient="owner", allowed_groups=("g",), now=source[2]) is None


@pytest.mark.parametrize("change", ["input", "recipient", "foreign_id", "foreign_group", "one_source", "too_long", "empty_topic", "silence"])
def test_unbound_or_invalid_model_decision_is_silent(source, change):
    populate(source)
    m = material(source)
    result = answer({"input_id": m.input_id, **m.payload})
    if change == "input": result["input_id"] = "previous"
    if change == "recipient": result["recipient_available"] = False
    if change == "foreign_id": result["topics"][0]["source_ids"][0] = "elsewhere"
    if change == "foreign_group": result["topics"][0]["group_id"] = "elsewhere"
    if change == "one_source": result["topics"][0]["source_ids"] *= 0
    if change == "too_long": result["message"] = "字" * 421
    if change == "empty_topic": result["topics"][0]["summary"] = ""
    if change == "silence": result["decision"] = "silence"
    assert validated_digest(result, m) == ""


def test_snapshot_rejects_changed_source_but_not_later_group_message(source):
    populate(source)
    before = material(source)
    source[0].record_event(conversation_id="g", scope="group:g", layer="observation", text="后来新增的讨论还有另外一个观点。",
        source_id="later", created_at=int(source[2].timestamp())+1,
        metadata={"source": "qq_passive_observation", "sender_id": "member-1"})
    after = read_digest_material(source[1], recipient="owner", allowed_groups=("g",),
        now=source[2]+timedelta(seconds=2), cutoff=before.payload["cutoff"])
    assert after.input_id == before.input_id
    with sqlite3.connect(source[1]) as conn:
        conn.execute("UPDATE memory_events SET text='更正后的选项不再是原先的意思' WHERE event_id=?",
            (before.payload["groups"][0]["records"][0]["source_id"],))
    assert material(source).input_id != before.input_id


def test_bounded_material_is_explicitly_partial(source):
    populate(source, count=80)
    m = material(source, policy=DigestPolicy(max_material_chars=15000))
    group = m.payload["groups"][0]
    assert group["coverage_complete"] is False
    assert 30 <= len(group["records"]) < 80
    assert all(r["text"].endswith("确定。") for r in group["records"])


class Ledger:
    """公开测试用内存账本；真实SQLite竞争由私用宿主集成回归覆盖。"""
    def __init__(self): self.rows = {}
    def already_claimed(self, **kw): return tuple(kw.values()) in self.rows
    def claim(self, *, now, **kw):
        if self.already_claimed(**kw): return False
        self.rows[tuple(kw.values())] = {"status": "sending"}
        return True
    def finish(self, *, now, sent, error, status, **kw): self.rows[tuple(kw.values())] = dict(sent=sent, error=error, status=status)
    def recent_sent(self, **kw): return False


def run(source, *, ledger=None, generate=answer, sender=None, runtime=None, **kwargs):
    runtime = runtime or ConversationRuntime(clock=lambda: source[2].timestamp())
    return process_group_digest(now=source[2], schedule=DigestSchedule(True, ("g",)), recipient_ids=("owner",),
        memory_db=source[1], ledger=ledger or Ledger(), conversation_runtime=runtime, generate=generate,
        sender=sender or (lambda **_: {"status": "ok", "retcode": 0, "data": {"message_id": "sent-1"}}), **kwargs)


def test_quiet_day_never_calls_model_or_sender(source):
    populate(source, count=10)
    assert run(source, generate=Mock(side_effect=AssertionError()), sender=Mock(side_effect=AssertionError())) == 0


@pytest.mark.parametrize("response", ["silence", "farewell", "invalid", "exception"])
def test_model_can_decline_and_never_gets_a_greeting_fallback(source, response):
    populate(source)
    ledger, calls = Ledger(), []
    def generate(payload):
        calls.append(payload)
        if response == "exception": raise TimeoutError()
        if response == "invalid": return {}
        result = answer(payload)
        if response == "farewell": result["recipient_available"] = False
        else: result["decision"] = "silence"
        return result
    assert run(source, ledger=ledger, generate=generate, sender=Mock(side_effect=AssertionError())) == 0
    assert run(source, ledger=ledger, generate=generate) == 0
    assert len(calls) == 1


def test_send_is_private_once_and_memory_records_actual_receipt(source):
    populate(source)
    ledger, sends, memory = Ledger(), [], Mock()
    def sender(**kw):
        sends.append(kw)
        return {"status": "ok", "retcode": 0, "data": {"message_id": "out-1"}}
    assert run(source, ledger=ledger, sender=sender, memory_service=memory) == 1
    assert run(source, ledger=ledger, sender=sender, memory_service=memory) == 0
    assert len(sends) == 1 and sends[0]["action"] == "send_private_msg"
    assert sends[0]["params"]["user_id"] == "owner"
    recorded = memory.record_turn.call_args.kwargs
    assert recorded["text"] == sends[0]["params"]["message"][0]["data"]["text"]
    assert recorded["metadata"]["delivery_message_ids"] == ["out-1"]


def test_new_foreground_turn_revokes_generated_digest(source):
    populate(source)
    runtime = ConversationRuntime(clock=lambda: source[2].timestamp())
    def generate(payload):
        runtime.begin(ConversationScope("private", "owner", "owner"), "new")
        return answer(payload)
    assert run(source, runtime=runtime, generate=generate, sender=Mock(side_effect=AssertionError())) == 0


@pytest.mark.parametrize("change", ["source", "stop", "window"])
def test_final_validation_before_send(source, change):
    populate(source)
    allowed, clock = [True], [source[2]]
    def generate(payload):
        if change == "source":
            with sqlite3.connect(source[1]) as conn: conn.execute("UPDATE memory_events SET status='forgotten'")
        if change == "stop": allowed[0] = False
        if change == "window": clock[0] += timedelta(hours=2)
        return answer(payload)
    assert run(source, generate=generate, allowed=lambda: allowed[0], clock=lambda: clock[0], sender=Mock(side_effect=AssertionError())) == 0


@pytest.mark.parametrize("failure", ["transport", "memory"])
def test_failure_never_causes_same_day_resend(source, failure):
    populate(source)
    ledger, sends = Ledger(), []
    def sender(**kw):
        sends.append(kw)
        if failure == "transport": raise TimeoutError()
        return {"status": "ok", "retcode": 0, "data": {"message_id": "out-1"}}
    memory = Mock()
    memory.record_turn.side_effect = OSError()
    assert run(source, ledger=ledger, sender=sender, memory_service=memory) == int(failure == "memory")
    assert run(source, ledger=ledger, sender=sender, memory_service=memory) == 0
    assert len(sends) == 1


def test_many_active_groups_keep_three_useful_sources_instead_of_thinning_every_group(source):
    for group in ("g", "h", "i", "j"):
        populate(source, group=group)
    m = read_digest_material(source[1], recipient="owner", allowed_groups=("g", "h", "i", "j"), now=source[2])
    assert m and len(m.payload["groups"]) == 3 and m.payload["groups_omitted"] == 1
    assert all(len(g["records"]) >= 30 for g in m.payload["groups"])


def test_known_bot_ids_are_excluded_and_schedule_is_disabled_by_default():
    assert not DigestSchedule.from_environment({}).enabled
    schedule = DigestSchedule.from_environment({"SHINKU_QQ_BOT_IDS": "self,other",
        "SHINKU_QQ_GROUP_DIGEST_EXCLUDED_SENDERS": "other,third"})
    assert schedule.excluded_sender_ids == ("self", "other", "third")


def test_runtime_failure_after_send_still_records_memory_and_never_replays(source):
    populate(source)
    runtime = ConversationRuntime(clock=lambda: source[2].timestamp())
    runtime.finish = Mock(side_effect=OSError("runtime store unavailable"))
    memory, ledger = Mock(), Ledger()
    sender = Mock(return_value={"status": "ok", "retcode": 0, "data": {"message_id": "receipt"}})
    assert run(source, runtime=runtime, memory_service=memory, ledger=ledger, sender=sender) == 1
    assert run(source, ledger=ledger, sender=sender) == 0
    assert sender.call_count == memory.record_turn.call_count == 1
    assert next(iter(ledger.rows.values()))["sent"] is True

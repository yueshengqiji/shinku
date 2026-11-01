"""真实本地 API、Bot 桥接与 SQLite 的合成联测，不连接模型或 QQ。"""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shinku.api import create_app
from shinku.config import load_settings
from shinku.conversation import ConversationRuntime
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.owner_continuity import owner_continuity
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage, MessageSegment
from shinku.qq.turns import QQTurn

NOW = 1_800_000_000


class Runtime:
    def __init__(self):
        self.calls = []
        self.before_return = lambda: None
        self.text = "我记下这个条件了。"

    def chat_supports_native_tools(self):
        return False

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        self.before_return()
        return {"speech": self.text}


def service(path, clock, *, projection=False):
    return MemoryService(store=MemoryStore(path), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=("owner",),
            projection=ProjectionPolicy(enabled=projection)), clock=lambda: clock[0])


def setup(tmp_path, *, projection=False):
    clock = [NOW]
    memory = service(tmp_path / "memory.db", clock, projection=projection)
    runtime, sends = Runtime(), []
    def sender(message):
        sends.append(message)
        return DeliveryResult(True, "sent", ("qq-receipt",))
    bridge = QQAgentBridge(runtime=runtime, system_prompt="测试人格", sender=sender,
        bot_ids=("bot",), master_user_ids=("owner",), memory_service=memory,
        conversation_runtime=ConversationRuntime(tmp_path / "conversations.db"))
    settings = load_settings({"SHINKU_DATA_ROOT": str(tmp_path / "data"),
        "SHINKU_CONFIG_ROOT": str(tmp_path / "config"), "SHINKU_LOG_ROOT": str(tmp_path / "logs")})
    app = create_app(settings, qq_agent_assembly=SimpleNamespace(bridge=bridge))
    return SimpleNamespace(clock=clock, memory=memory, runtime=runtime, bridge=bridge,
                           sends=sends, app=app, client=TestClient(app))


def qq_turn(text, mid="q1", *, sender="owner", kind="private"):
    conversation = sender if kind == "private" else "group"
    message = IncomingMessage(mid, sender, conversation, kind, (MessageSegment("text", text),),
                              timestamp=NOW, raw_event={"self_id": "bot"})
    return QQTurn(conversation, kind, mid, (mid,), text, (message,))


def records(memory, scope="private:master"):
    with memory.store._connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM memory_events WHERE scope=? AND status='active'", (scope,))]


def shared(memory, *, kind="desktop", requester="owner", self_id="bot"):
    return owner_continuity(memory, conversation_id="desktop:reader", conversation_type=kind,
        requester=requester, self_id=self_id, message_id="new")


def record(memory, text, *, mid="old", kind="private", conversation="owner", sender="owner"):
    memory.record_turn(conversation_id=conversation, conversation_type=kind,
        message_id=mid, text=text, metadata={"sender_id": sender, "self_id": "bot",
                                            "message_id": mid, "shinku_involved": True})


@pytest.mark.parametrize("projection", [False, True])
def test_http_delivery_and_immediate_bidirectional_context(tmp_path, projection):
    rig = setup(tmp_path, projection=projection)
    response = rig.client.post("/api/pet/chat", json={
        "text": "蓝晶安排只有下雨才取消，还没有决定。", "conversation_id": "first"})
    assert response.status_code == 200
    assert not rig.sends
    rows = records(rig.memory)
    assert {json.loads(r["metadata_json"])["role"] for r in rows} == {"user", "assistant"}
    assistant = next(r for r in rows if json.loads(r["metadata_json"])["role"] == "assistant")
    meta = json.loads(assistant["metadata_json"])
    assert meta["delivery_channel"] == "desktop_http"
    assert meta["delivery_status"] == "sent" and meta["delivery_message_ids"]
    trace = rig.bridge.trace_store.snapshot(limit=1)[0]
    assert trace["sent"] and trace["delivery_status"] == "sent"
    assert not any(s["name"] == "conversation_runtime" and s.get("status") == "unavailable"
                   for s in trace["stages"])
    rig.clock[0] += 1
    reply = rig.bridge.handle_turn(qq_turn("接着说这个安排"))
    assert reply.agent_status == "completed"
    prompt = json.dumps(rig.runtime.calls[-1], ensure_ascii=False)
    assert "只有下雨才取消，还没有决定" in prompt and "我记下这个条件了" in prompt
    assert len(rig.sends) == 1
    rig.clock[0] += 1
    response = rig.client.post("/api/pet/chat", json={"text": "接着刚才", "conversation_id": "new-window"})
    assert response.status_code == 200
    assert "接着说这个安排" in json.dumps(rig.runtime.calls[-1], ensure_ascii=False)
    assert len(rig.sends) == 1


def test_http_body_failure_never_records_assistant_delivery(tmp_path):
    rig = setup(tmp_path)
    body = json.dumps({"text": "本轮连接会断开"}).encode()
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    async def send(message):
        if message["type"] == "http.response.body":
            raise RuntimeError("synthetic_disconnect")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/pet/chat", "raw_path": b"/api/pet/chat",
        "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1000), "server": ("localhost", 80)}
    with pytest.raises(RuntimeError, match="synthetic_disconnect"):
        asyncio.run(rig.app(scope, receive, send))
    assert not rig.sends
    assert not any(json.loads(r["metadata_json"])["role"] == "assistant" for r in records(rig.memory))
    trace = rig.bridge.trace_store.snapshot(limit=1)[0]
    assert trace["delivery_status"] == "uncertain" and trace["agent_status"] == "failed"


@pytest.mark.parametrize("owners", [(), ("owner", "other")])
def test_desktop_refuses_ambiguous_host_identity(tmp_path, owners):
    rig = setup(tmp_path)
    rig.bridge.master_user_ids = frozenset(owners)
    response = rig.client.post("/api/pet/chat", json={"text": "你好", "sender_id": "owner"})
    assert response.status_code == 409 and not rig.runtime.calls and not rig.sends


def test_client_conversation_cannot_impersonate_qq_scope(tmp_path):
    rig = setup(tmp_path)
    rig.client.post("/api/pet/chat", json={"text": "只在桌面", "conversation_id": "owner",
                                         "sender_id": "other"})
    rows = records(rig.memory)
    assert rows and all(json.loads(r["metadata_json"])["owner_source_scope"].startswith("private:desktop:")
                        for r in rows)
    assert not records(rig.memory, "private:owner")


def test_two_connections_restart_and_ttl_do_not_need_embedding_worker(tmp_path):
    clock = [NOW]
    writer = service(tmp_path / "shared.db", clock)
    reader = service(tmp_path / "shared.db", clock)
    assert not shared(reader)["rendered"]
    record(writer, "绿晶任务如果周五下雨就延期，尚未执行。")
    assert "尚未执行" in shared(reader)["rendered"]
    reader = service(tmp_path / "shared.db", clock)
    assert "绿晶任务" in shared(reader)["rendered"]
    clock[0] += 300
    assert not shared(reader)["rendered"]
    row = records(writer)[0]
    assert reader.integrity_reader.read_source(record_id="event:" + row["event_id"],
        scope="private:master", now=clock[0]) is not None


@pytest.mark.parametrize("kind,requester", [("group", "owner"), ("desktop", "other"), ("private", "other")])
def test_private_continuity_never_enters_group_or_other_user(tmp_path, kind, requester):
    memory = service(tmp_path / "db", [NOW])
    record(memory, "主人私人安排")
    assert shared(memory, kind=kind, requester=requester) is None


def test_origin_change_and_owner_forget_invalidate_other_connection(tmp_path):
    clock = [NOW]
    memory = service(tmp_path / "db", clock)
    reader = service(tmp_path / "db", clock)
    record(memory, "紫晶计划仍未确定")
    assert shared(reader)["rendered"]
    with memory.store._connect() as conn:
        conn.execute("UPDATE memory_events SET text=? WHERE scope=?",
                     ("紫晶计划取消了", "private:owner"))
    assert not shared(reader)["rendered"]  # 旧共享副本不能覆盖原件更正。
    record(memory, "橙晶计划仅限周六", mid="second")
    origin = memory.store.event_by_source_id(scope="private:owner", source_id="turn:owner:second")
    memory.store.forget(query="橙晶", scope="private:master", now=NOW)
    assert reader.integrity_reader.read_source(record_id=origin.record_id,
        scope=origin.scope, now=NOW) is None
    # 相同字面内容的另一用户没有同源关系，不受主人撤销影响。
    record(memory, "橙晶计划仅限周六", mid="other", conversation="other", sender="other")
    other = memory.store.event_by_source_id(scope="private:other", source_id="turn:other:other")
    assert reader.integrity_reader.read_source(record_id=other.record_id, scope=other.scope, now=NOW)


@pytest.mark.parametrize("change", ["expire", "forget", "new_information", "revoke_identity", "revoke_bridge"])
def test_generation_cannot_publish_stale_cross_channel_basis(tmp_path, change):
    rig = setup(tmp_path)
    record(rig.memory, "蓝晶约定：只有晴天才出门。", kind="desktop", conversation="desktop:first")
    def mutate():
        if change == "expire":
            # 原件真实TTL到期仍拒绝；300秒自动选材边界不充当发布期限。
            with rig.memory.store._connect() as conn:
                conn.execute("UPDATE memory_events SET expires_at=?", (NOW + 1,))
            rig.clock[0] += 2
        elif change == "forget":
            rig.memory.store.forget(query="蓝晶", scope="private:master", now=NOW)
        elif change == "new_information":
            rig.clock[0] += 1
            record(rig.memory, "蓝晶约定现在取消。", mid="new", kind="desktop", conversation="desktop:first")
        elif change == "revoke_identity":
            rig.memory.policy = replace(rig.memory.policy, master_user_ids=())
        else:
            rig.bridge.master_user_ids = frozenset()
    rig.runtime.before_return = mutate
    result = rig.bridge.handle_turn(qq_turn("接着刚才"))
    assert result.agent_status == "failed"
    assert result.final_text != rig.runtime.text
    assert len(rig.runtime.calls) == 1


def test_local_fallback_identity_can_be_read_from_qq(tmp_path):
    rig = setup(tmp_path)
    rig.bridge.bot_ids = frozenset()
    rig.client.post("/api/pet/chat", json={"text": "只在桌面讨论的安排"})
    assert rig.runtime.text in shared(rig.memory, self_id="qq-bot")["rendered"]


def test_local_receipt_is_idempotent_and_never_changes_qq_sender(tmp_path):
    from shinku.qq.local_delivery import DesktopReplyDelivery
    from shinku.qq.reply_delivery import QQReplyPublication
    rig = setup(tmp_path)
    delivery = DesktopReplyDelivery(rig.bridge, requester="owner")
    received, reinforced = [], []
    deps = SimpleNamespace(accept_receipt=lambda *args: received.append(args),
        finalize_usage=lambda *args: reinforced.append(args), trace=lambda *a, **kw: None)
    publication = QQReplyPublication(qq_turn("本机输入"), "本机回答")
    original_sender = rig.bridge.sender
    delivery.prepare(publication, deps)
    delivery.confirm("伪造的其他正文")
    assert not received
    delivery.confirm("本机回答")
    delivery.confirm("本机回答")
    delivery.fail()
    assert len(received) == len(reinforced) == 1
    assert rig.bridge.sender is original_sender and not rig.sends


@pytest.mark.parametrize("change", ["revoke", "write_error"])
def test_local_receipt_always_finishes_transport_state(tmp_path, change):
    from shinku.qq.local_delivery import DesktopReplyDelivery
    from shinku.qq.reply_delivery import QQReplyPublication
    rig = setup(tmp_path)
    finished, accepted = [], []
    rig.bridge._conversation_finish = lambda trace_id, reply: finished.append(reply)
    def accept(*args):
        accepted.append(args)
        if change == "write_error":
            raise RuntimeError("synthetic_write_error")
    deps = SimpleNamespace(accept_receipt=accept, finalize_usage=lambda *a: None,
                           trace=lambda *a, **kw: None)
    delivery = DesktopReplyDelivery(rig.bridge, requester="owner")
    delivery.prepare(QQReplyPublication(qq_turn("输入"), "实际返回正文"), deps)
    if change == "revoke":
        rig.bridge.master_user_ids = frozenset()
        delivery.confirm("实际返回正文")
        assert not accepted
    else:
        with pytest.raises(RuntimeError, match="synthetic_write_error"):
            delivery.confirm("实际返回正文")
    delivery.fail()
    assert len(finished) == 1 and finished[0].sent and finished[0].delivery_status == "sent"


def test_greeting_does_not_read_stale_copy_through_legacy_handoff(tmp_path):
    rig = setup(tmp_path)
    record(rig.memory, "旧副本中的约定", kind="group", conversation="group")
    with rig.memory.store._connect() as conn:
        conn.execute("UPDATE memory_events SET text=? WHERE scope='group:group'",
                     ("原件已经更正",))
    rig.bridge.handle_turn(qq_turn("真红，在吗"))
    assert "旧副本中的约定" not in json.dumps(rig.runtime.calls[-1], ensure_ascii=False)


@pytest.mark.parametrize("change", ["delivery", "role", "version"])
def test_origin_metadata_is_live_evidence_not_authority_of_owner_copy(tmp_path, change):
    rig = setup(tmp_path)
    rig.client.post("/api/pet/chat", json={"text": "本机原话"})
    before = shared(rig.memory)
    assert rig.runtime.text in before["rendered"]
    with rig.memory.store._connect() as conn:
        row = conn.execute("""SELECT event_id,metadata_json FROM memory_events
            WHERE scope LIKE 'private:desktop:%'
            AND json_extract(metadata_json,'$.role')='assistant'""").fetchone()
        meta = json.loads(row["metadata_json"])
        if change == "delivery":
            meta["delivery_status"] = "uncertain"
        elif change == "role":
            meta["role"] = "user"
        else:
            meta["source_revision"] = "corrected"
        conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                     (json.dumps(meta), row["event_id"]))
    after = shared(rig.memory)
    assert before["signature"] != after["signature"]
    if change != "version":
        assert rig.runtime.text not in after["rendered"]


def test_new_omitted_long_correction_invalidates_initially_empty_context(tmp_path):
    rig = setup(tmp_path)
    def mutate():
        record(rig.memory, "长材料。" * 500 + "但是旧约定取消了", mid="long",
               kind="desktop", conversation="desktop:first")
    rig.runtime.before_return = mutate
    result = rig.bridge.handle_turn(qq_turn("接着说"))
    assert result.agent_status == "failed" and result.final_text != rig.runtime.text


def test_fast_context_policy_off_and_environment_bounds(tmp_path):
    rig = setup(tmp_path)
    record(rig.memory, "最近记录")
    rig.memory.policy = replace(rig.memory.policy, owner_continuity_seconds=0)
    assert shared(rig.memory) is None
    assert MemoryPolicy.from_environment({"SHINKU_MEMORY_OWNER_CONTINUITY_SECONDS": "99999"}).owner_continuity_seconds == 1800


def test_large_or_future_source_does_not_truncate_conditions_into_context(tmp_path):
    rig = setup(tmp_path)
    record(rig.memory, "长材料。" * 500 + "除非批准，否则绝不执行。")
    rig.clock[0] += 10
    record(rig.memory, "未来的记录", mid="future")
    rig.clock[0] -= 10
    result = shared(rig.memory)
    assert not result["handles"]
    assert "omitted_recent_sources" in result["rendered"]
    assert "除非批准" not in result["rendered"] and "未来的记录" not in result["rendered"]

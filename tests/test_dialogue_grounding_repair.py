"""CC01: actual bridge, canonical sources, final gates and test delivery."""
import json
from types import SimpleNamespace

import pytest

from shinku.agent.budget import current_task_budget
from shinku.agent.task_state import AgentTaskStore
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.projection import ProjectionPolicy
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.interaction_tools import InteractionRuntime
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from test_context_projection import NOW
from test_recent_dialogue_priority import pair
from test_lightweight_visual_selection import pipeline, candidate_payload


DENIAL = "我之前早就说过，你就是在编故事逗我。"
CORRECTION = "我不是开玩笑，照片确实是我拍的。"
REPAIRED = "原来你是在认真说这件事，刚才我把它当成玩笑了，那是当时你拍的照片呀。"


def verdict(action="pass", issues=None):
    def answer(options):
        payload = json.loads(options["user_prompt"])
        return {"input_id": payload["input_id"], "verdict": action,
                "issues": issues if issues is not None else []}
    return answer


def rewrite(text=REPAIRED):
    return lambda options: {"input_id": json.loads(options["user_prompt"])["input_id"], "text": text}


@pytest.fixture(params=["legacy", "task"])
def flow(request, tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW,
        policy=MemoryPolicy(master_user_ids=("owner",), projection=ProjectionPolicy(enabled=True)))
    prior, assistant = pair(store, "previous", CORRECTION, "我那时却把它当成编的。")
    calls, queue, sent, budgets = [], [], [], []
    def call(**options):
        calls.append(options)
        budgets.append(current_task_budget())
        assert queue, "unexpected additional model call"
        value = queue.pop(0)
        return value(options) if callable(value) else value
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=call), system_prompt="Fixture Shinku persona",
        memory_service=service, bot_ids=("bot",), master_user_ids=("owner",),
        task_store=AgentTaskStore(tmp_path / "tasks.sqlite"),
        sender=lambda output: sent.append(output) or DeliveryResult(True, "sent", ("out-1",)))
    bridge.tool_decision_mode = request.param
    if request.param == "task":
        bridge.runtime = InteractionRuntime(bridge.runtime, bridge.tool_host.registry, {})
    def send(text="当时给你拍的，这下信了吧"):
        message = IncomingMessage.from_event({"post_type": "message", "message_type": "private",
            "user_id": "owner", "self_id": "bot", "message_id": "current", "time": NOW,
            "message": [{"type": "text", "data": {"text": text}}]})
        turn = QQTurn(conversation_id="owner", conversation_type="private", primary_message_id="current",
            message_ids=("current",), messages=(message,), combined_text=text)
        return bridge.handle_turn(turn)
    try:
        yield SimpleNamespace(store=store, service=service, prior=prior, assistant=assistant, calls=calls,
            queue=queue, sent=sent, budgets=budgets, bridge=bridge, send=send, mode=request.param)
    finally:
        service.projection.stop()
        bridge.task_store.close()

























@pytest.mark.parametrize("message,text", [
    ("我今天想听热闹的音乐。", "今天想听什么曲风？"),
    ("刚才我是在认真说的。", "嗯，我按你的澄清来理解。"),
    ("晚安。", "晚安，去好好休息。"),
])
def test_private_reply_has_one_generation_and_no_model_judge(flow, message, text):
    p=flow;p.queue.append({"kind":"final","text":text})
    reply=p.send(message)
    assert reply.agent_status=="completed" and reply.final_text==text
    assert len(p.calls)==1 and not p.queue and len(p.sent)==1
    assert all("dialogue-review" not in row.get("prompt_cache_key","") for row in p.calls)


def test_revoked_source_during_main_generation_is_still_rejected(flow):
    p=flow
    def generate(options):
        p.store.forget(scope="private:owner",record_id=p.prior.record_id,query="",now=1800000000)
        return {"kind":"final","text":"不应发布的旧来源候选"}
    p.queue.append(generate)
    reply=p.send()
    assert reply.agent_status=="failed" and "不应发布" not in reply.final_text
    assert len(p.calls)==1 and len(p.sent)==1


def test_false_execution_claim_still_needs_actual_receipt(flow):
    p=flow;p.queue.append({"kind":"final","text":"我已经删除了文件 /tmp/synthetic.txt。"})
    reply=p.send("你好。")
    assert reply.agent_status=="failed" and "已经删除" not in reply.final_text
    assert len(p.calls)==1

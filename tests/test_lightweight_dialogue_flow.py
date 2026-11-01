"""LA06: inspect the entire model input, including dynamic flow additions."""
from reply_review_fixture import dialogue_review_result
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.chat_flow import CHAT_FLOW_SYSTEM_GUIDANCE, ChatFlowController
from shinku.qq.turns import QQTurn
from test_qq_visual_provenance import message


@pytest.mark.parametrize("kind", ["private", "group"])
@pytest.mark.parametrize("enabled", [False, True])
def test_status_address_first_repeated_praise_and_serious_help_have_no_coded_voice_anchor(kind, enabled):
    captured = []
    runtime = SimpleNamespace(call_chat_json=lambda **kw: captured.append(kw) or dialogue_review_result(kw) or {"speech": "回应当前内容。"})
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", system_prompts={"group": "group-fixture"},
        chat_flow=ChatFlowController(enabled=enabled), sender=None, memory_service=None)
    for index, text in enumerate(["我刚下班", "老婆", "你好可爱", "你好可爱", "你好可爱", "认真求助，帮我理解原因"]):
        msg = message(str(index), text=text, kind=kind)
        turn = QQTurn(msg.conversation_id, kind, msg.message_id, (msg.message_id,), text, (msg,))
        assert bridge.chat_flow.begin_turn(turn).voice_anchor == ""
        result = bridge.handle_turn(turn)
        assert result.final_text
    assert len(captured) == 6
    main_calls = [row for row in captured if not str(row.get("prompt_cache_key", "")).endswith(":dialogue-review:v5")]
    assert len(main_calls) == 6
    assert all("独立的对话依据审查者" in row["system_prompt"] for row in captured if row not in main_calls)
    for options in main_calls:
        prompt = "\n".join([options["system_prompt"], options["user_prompt"], *options.get("system_extra_blocks", ())])
        assert "【声线微调提示】" not in prompt
        for phrase in ["两拍接住", "说自己当下在做什么", "半拍害羞", "频繁用亲密称呼", "偶尔自然轻快、从容或轻微吐槽"]:
            assert phrase not in prompt
        assert (CHAT_FLOW_SYSTEM_GUIDANCE in prompt) == enabled
        assert "【本轮对话节奏提示】" not in prompt if not enabled else True


@pytest.mark.parametrize("stage", ["disabled", "fresh", "followup"])
def test_old_upstream_voice_anchor_is_never_reinjected(stage):
    msg = message(kind="private", text="普通内容")
    turn = QQTurn(msg.conversation_id, "private", msg.message_id, (msg.message_id,), msg.text, (msg,))
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **_: {"speech": "收到"}), system_prompt="fixture")
    parts = bridge._build_prompt(AgentState(context={"qq_turn": turn, "chat_flow": {
        "stage": stage, "guidance": "old-guidance", "habit_guidance": "old-habit",
        "voice_anchor": "两拍接住并指定调侃", "address_only": True}}, transcript=[]))
    prompt = parts.user_prompt + "\n" + "\n".join(parts.system_extra_blocks)
    assert "两拍接住并指定调侃" not in prompt
    if stage == "disabled":
        assert "old-guidance" not in prompt and "old-habit" not in prompt and "本轮主要是点名" not in prompt

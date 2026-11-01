"""聊天节奏修复的宿主回归：提示接入不改写问句、不增加模型或工具调用。

模型是否能减少多余追问须单独做真实对话验收；脚本回复只验证传输、状态和副作用边界。
"""
import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.chat_flow import CHAT_FLOW_SYSTEM_GUIDANCE, ChatFlowController
from shinku.agent_runtime.demo import demo_service
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat, boundary_bridge


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prompt,kind,answer,status", [
    ("今天把积木搭好了", "final", "这下能好好欣赏它了。", "completed"),
    ("推荐一本适合初学者的书", "final", "可以先读一本短篇选集，篇幅短，容易开始。", "completed"),
    ("刚才那个程序又没打开", "final", "先看一下启动日志，通常能找到出错的位置。", "completed"),
    ("陪我做问答练习，你来提问", "final", "你想怎样介绍这幅画？", "completed"),
    ("帮我把它改得好一点", "wait", "你指的是哪一段文字？", "waiting_user"),
])
def test_rhythm_keeps_direct_reply_requested_questions_and_real_clarification(
    tmp_path, mode, enabled, prompt, kind, answer, status,
):
    service, tool, executor = demo_service(tmp_path)
    chat = Chat([{"kind": kind, "text": answer}])
    flow = ChatFlowController(enabled=enabled)
    bridge = (boundary_bridge(service, tool, chat, chat_flow=flow) if mode == "task" else
              QQAgentBridge(runtime=chat, system_prompt="Synthetic persona", chat_flow=flow))
    try:
        reply = bridge(turn(prompt))
        assert reply.final_text == answer and reply.agent_status == status
        assert reply.delivery_status == "dry_run" and not reply.sent
        assert len(chat.calls) == 1
        supplied = "\n".join([chat.calls[0]["system_prompt"],
                              *chat.calls[0].get("system_extra_blocks", ())])
        assert (CHAT_FLOW_SYSTEM_GUIDANCE in supplied) is enabled
        assert tool.calls == 0 and executor.calls == [] and not tool.path.exists()
        assert service.task_store.candidates(requester="fixture", session="private:fixture") == []
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("mode", ["legacy", "task"])
def test_multiturn_prompt_does_not_turn_questions_into_a_counter_or_rewrite(tmp_path, mode):
    service, tool, executor = demo_service(tmp_path)
    answers = ["你最喜欢这座模型的哪部分？", "屋顶的轮廓确实很漂亮。", "你想用暖色还是冷色？"]
    chat = Chat([{"kind": "final", "text": answer} for answer in answers])
    bridge = (boundary_bridge(service, tool, chat) if mode == "task" else
              QQAgentBridge(runtime=chat, system_prompt="Synthetic persona"))
    try:
        for i, (prompt, answer) in enumerate(zip(
            ["问我一个关于刚做好的模型的问题", "最喜欢的是屋顶", "继续问我选颜色的问题"], answers,
        )):
            reply = bridge(turn(prompt, message_id=f"rhythm-{i}"))
            assert reply.final_text == answer and reply.agent_status == "completed"
            assert not reply.sent and reply.delivery_status == "dry_run"
        assert len(chat.calls) == len(answers)
        assert tool.calls == 0 and executor.calls == []
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()

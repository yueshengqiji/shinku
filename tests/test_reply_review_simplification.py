"""旧未决历史不触发另一个模型路由或重写；宿主账本不伪造已核验。"""
from shinku.conversation import AnswerEvidence, ConversationScope
from test_group_evidence_grounding import group, final
from test_memory_read_acceptance import bridge_factory


def unresolved_history(p):
    runtime=p.bridge.conversation_runtime;scope=ConversationScope("group","g","owner")
    runtime.begin(scope,"prior-unconfirmed")
    runtime.step("prior-unconfirmed",route="history",evidence=AnswerEvidence("history","unverified"))
    runtime.finish("prior-unconfirmed",status="failed",reason="synthetic_unconfirmed")


def test_new_chat_does_not_reopen_old_history_review(group):
    p=group;unresolved_history(p);text="好，换个话题。";p.queue.append(final(text))
    result=p.send("我们换个话题吧。")
    assert result.agent_status=="completed" and result.final_text==text
    assert len(p.calls)==1 and not p.queue and len(p.sent)==1
    old=p.bridge.conversation_runtime.snapshot(ConversationScope("group","g","owner"))
    assert "prior-unconfirmed" in str(old)

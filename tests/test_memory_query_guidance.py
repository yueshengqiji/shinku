"""两聊天入口只在已授权读取时交付引导，不强制工具或增加模型轮次。"""
import pytest

from test_memory_read_acceptance import bridge_factory
from test_private_dialogue_continuity import turn


@pytest.mark.parametrize("entry", ["legacy", "task"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("message", ["晚安。", "今天想听热闹的音乐。"])
def test_read_guidance_follows_actual_offer_without_forcing_plain_chat(bridge_factory, entry, enabled, message):
    bridge, chat, _, coordinator = bridge_factory(entry, "accepted", read=False, answer="想聊什么都可以。")
    bridge.command_state.set_agent_enabled(enabled)
    reply = bridge(turn(message, mid="guidance"))
    assert reply.agent_status == "completed"
    assert not reply.sent
    assert len(chat.calls) == 1
    call = chat.calls[0]
    instruction = "\n".join(call["system_extra_blocks"])
    assert ("【本地历史核对】" in instruction) is enabled
    assert bool(call.get("native_tools")) is enabled
    assert "【本地历史核对】" not in call["user_prompt"]
    assert message in call["user_prompt"]
    assert coordinator is None or not coordinator.list()


def test_identity_fact_guidance_names_recall_without_hardcoding_any_person():
    """称呼/姓名类身份事实在缺少依据时检索，且不登记具体人名。

    背景：主人询问“你给我起的名字”时，引导只列举了配置/约定/经历/旧偏好，
    身份类称呼事实未明确列举；这只是候选引导缺口，不能据此证明漏检索的因果。
    本用例只锁定规则本身：名字来自已有或检索到的原话，测试里不列举任何
    真实或作品人名（列举本身就把名字带进了公开测试）。
    """
    import re

    from shinku.qq.memory_guidance import MEMORY_READ_GUIDANCE as guidance

    assert "身份事实" in guidance
    for phrase in ("姓名", "称呼", "称谓"):
        assert phrase in guidance, phrase
    # 条件表达：现有原话足够就直接用，缺依据才检索。
    assert "先判断现有原话能否支持答案" in guidance
    assert "直接照用，不必再查" in guidance
    assert "只有缺少依据时才检索" in guidance
    # 不得再把无条件结论写回引导：必须显式写成“不要默认……一定不在本轮”。
    assert "不要默认这类事实一定不在本轮上下文里" in guidance
    assert "同样先核对原话：这种事实通常来自" not in guidance
    # 仍由主模型自主检索，未写成需要关键词触发的路由。
    assert "没有相关原话就主动用memory_search核对" in guidance
    # 禁止拿作品角色名顶替用户身份的通用规则必须保留。
    assert "不能拿作品人物、同名角色或熟悉名字" in guidance
    # 普通招呼仍免检索，不能被身份规则拖成必查旧史。
    assert "普通招呼、一般知识问答和仅表达当前新意愿不必查旧史" in guidance
    # 身份称呼那几句只能描述检索方式，不得自带引号名字或“某某叫 X”的归属。
    identity_lines = [line for line in guidance.split("。") if "称谓" in line or "名字" in line]
    assert identity_lines, "身份称呼规则段丢失"
    joined = "。".join(identity_lines)
    assert "叫“" not in joined and "叫‘" not in joined
    assert not re.search(r"[（(][^）)]{1,6}[)）]是", joined)

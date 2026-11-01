"""普通对话无分类裁判；主生成仍得到来源、条件和作者边界。"""
import base64
import pytest
from test_dialogue_grounding_repair import flow
from test_group_evidence_grounding import group, final
from test_lightweight_visual_selection import PNG


def test_personal_history_guidance_is_in_primary_generation(flow):
    p=flow;p.queue.append({"kind":"final","text":"今天想听什么曲风？"})
    result=p.send("今天想听热闹的音乐。")
    assert result.agent_status=="completed" and len(p.calls)==1
    blocks="\n".join(p.calls[0]["system_extra_blocks"])
    assert "【个人历史依据】" in blocks and "否定与条件" in blocks
    assert "dialogue-review" not in p.calls[0].get("prompt_cache_key","")


@pytest.mark.parametrize("changed", [False,True])
def test_image_change_checked_locally_without_review_call(flow,changed):
    p=flow;images=[{"data_url":"data:image/png;base64,"+base64.b64encode(PNG).decode()}]
    p.bridge._images=lambda turn:images
    def generate(options):
        assert options["user_images"]==images
        if changed:images[0]={"data_url":"data:image/png;base64,changed"}
        return {"kind":"final","text":"图片候选正文。"}
    p.queue.append(generate);result=p.send("描述这张图。")
    assert result.agent_status==("failed" if changed else "completed")
    assert ("图片候选正文" in result.final_text) is not changed
    assert len(p.calls)==1 and not p.queue

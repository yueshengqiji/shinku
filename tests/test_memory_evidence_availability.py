"""Host input inventory distinguishes historical words from attached pixels."""
from shinku.agent.loop import AgentState
from shinku.qq.agent_bridge import QQAgentBridge
from test_memory_read_acceptance import bridge_factory
from test_context_projection import put
from test_private_dialogue_continuity import turn
from test_qq_visual_provenance import Harness, message
import json
import pytest
from types import SimpleNamespace


def inventory(prompt):
    return json.loads(prompt.split('【本轮可用证据清单】\n', 1)[1].split('\n\n', 1)[0])


@pytest.mark.parametrize('entry', ['legacy', 'task'])
@pytest.mark.parametrize('text', ['图里的动物是鹤，不是鹅。', '报告只是换了标题，结论还没有给出。'])
def test_historical_object_mention_does_not_become_a_current_observation(bridge_factory, entry, text):
    bridge, chat, memory, _ = bridge_factory(entry, 'accepted', read=False, answer='接着说吧。')
    put(memory.store, 'recent-source', text, age=30)
    result = bridge(turn('接着那句话说吧。', mid='current'))
    assert result.agent_status=='completed' and len(chat.calls)==1
    available = inventory(chat.calls[0]['user_prompt'])
    assert available['attached_image_count']==0
    assert available['historical_text_by_kind']['user_statement'] >= 1
    assert available['historical_text_is_current_observation'] is False
    assert not result.sent and not result.background_task_id


@pytest.mark.parametrize('inherited', [False, True])
def test_inventory_uses_real_attached_images_and_keeps_existing_provenance(inherited):
    h = Harness()
    initial = h.receive(message('image', image=True))
    current = h.choose(h.receive(message(text='接着那张图说'))) if inherited else initial
    bridge = QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {}),
                           system_prompt='Synthetic persona.', sender=None)
    projected = {'rendered':'原话', 'records':[
        {'record_id':'old', 'text':'我们刚才讨论过这幅图。', 'evidence_kind':'user_statement'}]}
    bridge._project_prompt_context = lambda context: projected
    try:
        parts = bridge._build_prompt(AgentState(transcript=[], context={'qq_turn': current}))
        available = inventory(parts.user_prompt)
        assert available['attached_image_count']==len(QQAgentBridge._images(current))==1
        sources = json.loads(parts.user_prompt.split('【本轮实际图片来源，按附加图片顺序编号】\n')[1].split('\n\n')[0])
        assert sources[0]['source_message_id']=='image'
        assert sources[0]['source_sender_id']=='owner'
        assert 'data:image' not in parts.user_prompt
    finally:
        bridge.task_store.close(); bridge.conversation_runtime.close()

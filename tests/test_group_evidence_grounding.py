"""Group continuation through both actual chat entries and delivery gates."""
import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentDecision
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.qq.delivery import DeliveryResult
from shinku.qq.interaction_tools import validate_interaction_response
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from test_context_projection import NOW
from test_dialogue_grounding_repair import rewrite
from test_memory_read_acceptance import bridge_factory
from test_recent_dialogue_priority import pair


BAD = '刚才那种缩着脖子的姿态，换个名字就显得从容优雅了。'
GOOD = '嗯，按你刚才的更正来理解。你想接着聊哪一点？'


def verdict(action='pass', issues=None, *, source_quote=None, image=False):
    def answer(options):
        payload=json.loads(options['user_prompt']);text=payload['candidate_reply'];claims=[]
        if source_quote:
            source=next(row for row in payload['sources'] if source_quote in row['text'])
            claims=[{'answer_fragment':text,'quote':source_quote,'source_id':source['evidence_id'],'kind':'statement'}]
        if image:
            claims=[{'answer_fragment':text,'quote':'','source_id':'image:1','kind':'observation'}]
        return {'input_id':payload['input_id'],'verdict':action,'issues':issues or [],'claims':claims}
    return answer


def final(text, mode='evidence'):
    return {'kind':'final', 'text':text, 'dialogue_basis':{
        'mode':mode, 'claims':[text] if mode=='evidence' else []}}


@pytest.fixture(params=['legacy', 'task'])
def group(bridge_factory, request):
    bridge, chat, memory, _ = bridge_factory(request.param, 'accepted', read=False)
    source, assistant = pair(memory.store, 'previous', '图里的动物是鹤，不是鹅。',
                              '我会按鹤来理解这张图。', scope='group:g')
    queue, calls, sent = [], [], []
    def call(**options):
        calls.append(options)
        assert queue, 'unexpected call'
        value = queue.pop(0)
        return value(options) if callable(value) else value
    chat.call_chat_json = call
    bridge.sender = lambda output: sent.append(output) or DeliveryResult(True, 'sent', ('delivered',))
    def send(text='接着那句话说吧。'):
        message = IncomingMessage.from_event({'post_type':'message', 'message_type':'group',
            'group_id':'g', 'user_id':'owner', 'self_id':'bot', 'message_id':'current', 'time':NOW,
            'message':[{'type':'text','data':{'text':text}}]})
        return bridge(QQTurn(conversation_id='g', conversation_type='group', primary_message_id='current',
                             message_ids=('current',), messages=(message,), combined_text=text))
    return SimpleNamespace(bridge=bridge, memory=memory, source=source, assistant=assistant,
                           queue=queue, calls=calls, sent=sent, send=send)



















@pytest.mark.parametrize("mode", ["conversation","evidence"])
def test_group_main_reply_has_no_post_generation_review(group,mode):
    p=group;text="嗯，按这次更正来理解。";p.queue.append(final(text,mode))
    reply=p.send()
    assert reply.agent_status=="completed" and reply.final_text==text
    assert len(p.calls)==1 and not p.queue and len(p.sent)==1


def test_group_forget_during_generation_remains_a_local_hard_failure(group):
    p=group
    def generate(options):
        p.memory.store.forget(scope="group:g",record_id=p.source.record_id,query="",now=1800000000)
        return final("不应发布的来源正文")
    p.queue.append(generate)
    result=p.send()
    assert result.agent_status=="failed" and "不应发布" not in result.final_text
    assert len(p.calls)==1 and not p.queue

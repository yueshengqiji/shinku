"""Calendar and premise regression for the 2026-10-07 group-experience reply."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.self_activity import needs_group_activity_context, is_self_group_recall
from shinku.qq.agent_bridge import QQAgentBridge
from test_self_group_activity import turn, source


NOW = int(datetime(2026, 10, 7, 20, 52).timestamp())
QUERY = "今天在群里玩的开心吗"


@pytest.fixture
def service(tmp_path):
    s = MemoryService(store=MemoryStore(tmp_path/'memory.sqlite3'), router=MemoryRouter(),
        policy=MemoryPolicy(master_user_ids=('owner',), projection=ProjectionPolicy(enabled=True)),
        clock=lambda: NOW)
    yield s
    s.projection.stop()


def event(service, mid, text, *, age=30, role='', scope='private:owner', **meta):
    source_id = f'turn:{scope.split(":", 1)[1]}:{mid}'
    service.store.record_event(conversation_id=scope.split(':', 1)[1], scope=scope, text=text,
        created_at=NOW-age, source_id=source_id,
        metadata={'role':role, 'sender_id':'owner', 'self_id':'bot', 'message_id':mid, **meta})
    return service.store.event_by_source_id(source_id=source_id, scope=scope)


def project(service, query=QUERY, mode='recent', seeds=()):
    return service.project_context(conversation_id='owner', conversation_type='private',
        user_message=query, message_meta={'sender_id':'owner','self_id':'bot','message_id':'current'},
        memory_context={'decision':{'history_mode':mode},'records':[r.as_dict() for r in seeds]}, flow_reason='')


@pytest.mark.parametrize('query', [QUERY, '老婆，今天在群里玩得开心吗？', '你今天一直在群里看着吗？',
    '昨天群聊热闹吗', '刚才在群里怎么样', '10月6日群里你开心吗', '小明今天在群里开心吗'])
def test_broad_relevance_does_not_assert_who_acted(query):
    assert needs_group_activity_context(query)


@pytest.mark.parametrize('query', ['明天去群里发句话', '帮我写群规', '你今天吃什么', '你刚才在私聊说了什么'])
def test_unrelated_query_does_not_scan_group_receipts(query):
    assert not needs_group_activity_context(query)


@pytest.mark.parametrize('mode', ['none', 'recent', 'retrieve'])
def test_today_survives_router_mode_and_old_seed_cannot_bypass_window(service, mode):
    yesterday = event(service, 'old', '老婆今天在群里玩的开心吗', age=86400)
    today = event(service, 'earlier', '今天群里的活动要核对', age=100)
    current = event(service, 'current', QUERY, age=0)  # Legacy user role is blank.
    view = project(service, mode=mode, seeds=(yesterday, current))
    ids = {r['record_id'] for r in view['records']}
    assert today.record_id in ids
    assert yesterday.record_id not in ids and current.record_id not in ids
    start, end = view['coverage']['requested_time_window']
    assert start == int(datetime(2026, 10, 7).timestamp()) and start <= NOW < end


def test_current_owner_copy_is_not_independent_evidence(service):
    current = event(service, 'current', QUERY, age=0)
    service.store.record_event(conversation_id='owner', scope='private:master', text=QUERY,
        created_at=NOW, source_id='owner:'+current.source_id,
        metadata={'sender_id':'owner','self_id':'bot','message_id':'current',
                  'owner_memory':True,'owner_source_scope':'private:owner'})
    view = project(service, mode='retrieve')
    assert not view['records']


def test_linked_wrong_day_cannot_bypass_archive_date_filter(service):
    old = event(service, 'yesterday', '今天在群里一直聊天', age=86400)
    event(service, 'today', '今天群里的情况另算', reply_message_ids=['yesterday'])
    view = project(service)
    assert old.record_id not in {r['record_id'] for r in view['records']}


def bridge(service):
    return QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw:{}), system_prompt='persona',
        memory_service=service, master_user_ids=('owner',), bot_ids=('bot',), sender=None)


def test_casual_question_gets_timed_receipts_without_a_fixed_final_reply(service):
    # A midnight sent reply is evidence of that reply, not continuous watching.
    source(service, at=int(datetime(2026, 10, 7, 0, 57).timestamp()), text='PRIVATE_BODY_MUST_NOT_APPEAR')
    b = bridge(service)
    current = turn(QUERY, message_id='current')
    state = AgentState(transcript=[{'role':'user','content':QUERY}],
        context={'qq_turn':current,'current_time':NOW})
    prompt = b._build_prompt(state, include_agent_control=False)
    assert '2026-10-07T00:57:00' in prompt.user_prompt
    assert 'PRIVATE_BODY_MUST_NOT_APPEAR' not in prompt.user_prompt
    assert '不能证明发生过' in ''.join(prompt.system_extra_blocks)
    assert not is_self_group_recall(QUERY)  # Natural expression, no legacy hard reply.


def test_yesterday_receipt_and_private_self_claim_do_not_prove_today(service):
    source(service, at=NOW-86400)
    event(service, 'false-private-claim', '我今天一直在旁边看着群聊', role='assistant',
        delivery_status='sent', delivery_message_ids=['fixture'], requester_id='owner')
    data = bridge(service)._prompt_self_activity({'qq_turn':turn(QUERY)})
    assert data['records'] == [] and data['absence_means'] == 'unknown'


def test_missing_delivery_and_other_bot_do_not_prove_activity(service):
    source(service, at=NOW-60, sent=False)
    source(service, at=NOW-40, message='other', sender='otherbot', self_id='otherbot')
    assert not bridge(service)._prompt_self_activity({'qq_turn':turn(QUERY)})['records']


def test_nonowner_cannot_use_casual_query_to_scan_private_group_activity(service, monkeypatch):
    monkeypatch.setattr(service.store, 'delivered_group_activity', lambda **_:pytest.fail('unauthorized scan'))
    data = bridge(service)._prompt_self_activity({'qq_turn':turn(QUERY, sender='stranger')})
    assert data['access'] == 'denied' and data['records'] == []


def test_group_context_uses_same_scope_and_does_not_leak_body(service):
    source(service, at=NOW-60, group='100001', message='local')
    source(service, at=NOW-30, group='100002', message='foreign')
    b = bridge(service)
    data = b._prompt_self_activity({'qq_turn':turn(QUERY, group='100001')})
    assert len(data['records']) == 1
    denied = b._prompt_self_activity({'qq_turn':turn('今天在群100002玩的开心吗', group='100001')})
    assert denied['access'] == 'denied' and not denied['records']


def test_source_is_rechecked_after_receipt_revocation(service):
    source(service, at=NOW-60, text='fixture revocation')
    b = bridge(service)
    assert b._prompt_self_activity({'qq_turn':turn(QUERY)})['records']
    service.store.forget(scope='group:100002', query='fixture revocation', now=NOW)
    assert not b._prompt_self_activity({'qq_turn':turn(QUERY)})['records']


def test_failed_reader_cannot_be_presented_as_absence(service, monkeypatch):
    def fail(**_):
        raise RuntimeError('fixture')
    monkeypatch.setattr(service, 'inspect_self_group_activity', fail)
    data = bridge(service)._prompt_self_activity({'qq_turn':turn(QUERY)})
    assert data['access'] == 'unavailable' and data['absence_means'] == 'unknown'

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.llm.runtime_transport import completion_deadline
from shinku.memory.projection.semantics import ProjectionModel, validate
from shinku.memory.projection.view import reference_answer_basis
from shinku.memory.projection.contracts import Deadline
from test_contract_c2_5 import _make_runtime, FakeResponse, _metrics
from test_context_projection import engine, put, req


@pytest.mark.parametrize('query',['她后来去了吗？','那个人后来呢','小明后来怎么样了？'])
def test_bare_ambiguous_followup_asks_without_wasting_judge(tmp_path,query):
    store,projector=engine(tmp_path,semantic_enabled=True)
    name='小明' if '小明' in query else '小红'
    put(store,'a',name+'说她想去看展。')
    put(store,'b',('小明' if name=='小明' else '小雪')+'说她也想去看展。')
    projector.semantic.model=lambda p:pytest.fail('No evidence to distinguish, so do not call a Judge')
    view=projector.project(req(query))
    assert view.topic['unresolved'] and not view.topic['last_subject']
    assert view.diagnostics['semantic']=={'status':'clarification_needed','calls':0}
    assert len(view.entities)==2


def test_judge_aliases_keep_original_ids_and_deadline_bound():
    calls=[]
    def aux(**kw):
        calls.append(kw)
        p=json.loads(kw['user_prompt'])
        assert p['records'][0]['record_id']=='s0' and p['entities'][0]['candidate_id']=='e0'
        return {'selected_entity':'e0','uncertain':False,'evidence':[{'record_id':'s0','quote':'小红是建筑师。'}]}
    payload={'kind':'judge','query':'做建筑设计的她','records':[{'record_id':'event:long','text':'小红是建筑师。','author':'owner'}],
             'entities':[{'candidate_id':'mention:long','name':'小红','sources':['event:long'],'author':'owner'}],
             'locator_ids':['event:long']}
    value=ProjectionModel(SimpleNamespace(call_aux_json=aux))(payload,timeout_seconds=7)
    result=validate('judge',value,payload)
    assert result['selected_entity']=='mention:long' and result['evidence'][0]['record_id']=='event:long'
    assert calls[0]['timeout_seconds']==7 and calls[0]['attempt_budget']=={'remaining':2}


def test_alias_hallucination_still_rejected():
    model=ProjectionModel(SimpleNamespace(call_aux_json=lambda **kw:{'selected_entity':'e999','uncertain':False,
        'evidence':[{'record_id':'s999','quote':'来源'}]}))
    payload={'kind':'judge','query':'谁','records':[{'record_id':'event:a','text':'来源'}],'entities':[],'locator_ids':[]}
    with pytest.raises(ValueError):validate('judge',model(payload),payload)


def test_answer_basis_preserves_exact_conditions_and_drops_after_source_change(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,'a','小红说如果周六不下雨就去看展，但她还没答应。')
    view=projector.project(req('她后来去了吗'))
    basis=view.topic['answer_basis']
    assert basis['person']=='小红'
    assert basis['conditions']==[{'record_id':row.record_id,'quote':'如果周六不下雨就去看展，但她还没答应'}]
    store.forget(scope=row.scope,record_id=row.record_id,query='',now=req().now)
    next_view=projector.project(req('她后来去了吗'))
    assert 'answer_basis' not in next_view.topic and '不下雨' not in next_view.rendered


def test_condition_of_another_mentioned_person_is_not_promoted(tmp_path):
    store,projector=engine(tmp_path)
    row=put(store,'a','小红说小雪如果周六不下雨就去看展，但小红不去。')
    row=projector.sources.read(((row.record_id,row.scope,''),),req(),Deadline(300))[0]
    entity={'name':'小红','sources':[row.record_id]}
    assert not reference_answer_basis(entity,[row])['conditions']


def test_aux_tls_failure_recovers_once_with_finite_http_deadline():
    runtime=_make_runtime(scripts=[('raise',RuntimeError('TLS handshake timeout')),('response',FakeResponse('{"uncertain":true}'))])
    runtime.runtime_policy=replace(runtime.runtime_policy,retry_backoff_seconds=0)
    try:
        value=runtime.call_aux_json(system_prompt='fixture',user_prompt='fixture',fallback={},
                                    attempt_budget={'remaining':2},timeout_seconds=5)
        assert value=={'uncertain':True} and len(runtime.aux.client.payloads)==2
        assert 0<runtime.aux.client.payloads[1]['timeout']<=runtime.aux.client.payloads[0]['timeout']<=5
        assert _metrics(runtime,'llm_http_attempt')==2
    finally:runtime.close()


def test_aux_deadline_stops_retry_and_does_not_leak_to_main_call(monkeypatch):
    from shinku.llm import runtime_transport
    clock=[100.]
    monkeypatch.setattr(runtime_transport.time,'monotonic',lambda:clock[0])
    runtime=_make_runtime(scripts=[])
    def slow(**payload):
        clock[0]+=2
        raise TimeoutError('TLS handshake timeout')
    runtime.aux.client.chat.completions.create=slow
    runtime.runtime_policy=replace(runtime.runtime_policy,retry_backoff_seconds=0)
    try:
        assert runtime.call_aux_json(system_prompt='s',user_prompt='u',fallback={'failed':True},
            timeout_seconds=1,attempt_budget={'remaining':2})=={'failed':True}
        assert _metrics(runtime,'llm_http_attempt')==1
        # A subsequent ordinary call must not inherit this request's deadline.
        observed=[]
        runtime.chat.client.chat.completions.create=lambda **kw: observed.append(kw) or FakeResponse('{"ok":true}')
        assert runtime.call_chat_json(system_prompt='s',user_prompt='u',fallback={})=={'ok':True}
        assert 'timeout' not in observed[0]
    finally:runtime.close()


def test_repeated_tls_failures_exhaust_budget_instead_of_looping():
    runtime=_make_runtime(scripts=[('raise',RuntimeError('TLS handshake timeout'))]*5)
    runtime.runtime_policy=replace(runtime.runtime_policy,retry_backoff_seconds=0)
    try:
        assert runtime.call_aux_json(system_prompt='s',user_prompt='u',fallback={'failed':True},
            timeout_seconds=5,attempt_budget={'remaining':2})=={'failed':True}
        assert len(runtime.aux.client.payloads)==2
    finally:runtime.close()

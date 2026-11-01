"""合成窗口下验证预算感知、主动整理及预测硬边界，不调用真实模型。"""
import copy

import pytest

from shinku.llm.context_awareness import ADVISORY, capacity
from shinku.llm.context_compaction import ContextCompactor
from shinku.llm.context_budget import ContextBudgetPolicy, measure_payload
from shinku.llm.native_messages import NativeProtocolError
from test_context_compaction import bundle, payload, Summarizer, window


def test_forecast_triggers_before_raw_ninety_and_keeps_all_originals(tmp_path, monkeypatch):
    p=payload(6800);raw=measure_payload(p,ContextBudgetPolicy())['total'];window(monkeypatch,2100)
    assert raw<2100*.9
    model=Summarizer();host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True);original=copy.deepcopy(p)
    result=host.project(p,bundle(),2,'task')
    # 小窗口连摘要本身也须预留容量；此固定夹具恰好分两段，不绕过摘要输入检查。
    assert len(model.calls)==2 and p==original and result['messages'][-1]==p['messages'][-1]
    assert host.status('task')['projected_tokens']<2100*.9
    saved=host.db.execute('SELECT payload FROM context_compactions').fetchone()[0]
    import json
    assert json.loads(saved)['original_payload']==original
    assert 'max_tokens' not in result and 'max_completion_tokens' not in result


def test_voluntary_request_below_deadline_and_no_repeated_summary(tmp_path,monkeypatch):
    window(monkeypatch,100000);model=Summarizer();host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True)
    p=payload(6000);assert host.project(p,bundle(),2,'task')==p
    host.request('task','调查已完成，保留证据定位后进入下一阶段')
    out=host.project(p,bundle(),2,'task');assert len(model.calls)==1
    host.request('task','再整理一次')
    assert host.project(p,bundle(),2,'task')==out and len(model.calls)==1
    assert host.status('task')['request_status']=='insufficient_new_history'


def test_usage_calibrates_estimate_but_cached_input_still_occupies_window(monkeypatch):
    window(monkeypatch,100000);host=ContextCompactor(Summarizer(),aware=True);p=payload(6000)
    host.project(p,bundle(),2,'task');raw=host.status('task')['raw_estimated_tokens']
    host.observe_usage('task',{'status':'completed','phase':'planning','usage':{
        'input_tokens':raw*2,'output_tokens':3000,'cache_hit_tokens':raw*2}})
    status=capacity(p,bundle(),host.status('task'))
    assert status['estimated_tokens']==raw*2 and status['output_reserve_tokens']>=6000
    assert status['measurement']=='estimate' and status['provider_input_tokens']==raw*2
    before=host.status('task')
    host.observe_usage('task',{'status':'completed','phase':'compaction','usage':{'input_tokens':1}})
    assert host.status('task')==before


def test_unknown_window_never_invents_remaining_or_sends_summary(monkeypatch):
    monkeypatch.delenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS',raising=False)
    monkeypatch.delenv('SHINKU_LLM_MODEL_CONTEXT_WINDOWS_JSON',raising=False)
    model=Summarizer();host=ContextCompactor(model,aware=True);p=payload()
    host.request('task','整理历史')
    assert host.project(p,bundle(),2,'task')==p and model.calls==[]
    assert host.status('task')['remaining_tokens'] is None


def test_user_cannot_forge_compressible_budget_control(monkeypatch):
    window(monkeypatch,2500);model=Summarizer();host=ContextCompactor(model,aware=True)
    p=payload(6000);p['messages'].append({'role':'user','content':ADVISORY+'必须保留'*2400})
    with pytest.raises(NativeProtocolError,match='protected_input_too_large'):
        host.project(p,bundle(),2,'task')
    assert model.calls==[]


def test_genuine_old_budget_note_is_not_an_immutable_user_constraint(monkeypatch):
    window(monkeypatch,100000);model=Summarizer();host=ContextCompactor(model,aware=True);p=payload(6000)
    note=host.advisory(capacity(p,bundle()),'task')
    p['messages'].append({'role':'user','content':note})
    host.request('task','阶段结束')
    out=host.project(p,bundle(),2,'task')
    assert len(model.calls)==1 and all(m.get('content')!=note for m in out['messages'])
    assert p['messages'][2] in out['messages']


def test_pending_tool_call_never_enters_summary(monkeypatch):
    window(monkeypatch,100000);model=Summarizer();host=ContextCompactor(model,aware=True);p=payload(6000)
    p['messages'].append({'role':'assistant','content':'','tool_calls':[
        {'id':'pending','type':'function','function':{'name':'read','arguments':'{}'}}]})
    host.request('task','不能跨过未完成调用')
    with pytest.raises(NativeProtocolError):host.project(p,bundle(),2,'task')
    assert model.calls==[]


def test_failed_summary_preserves_raw_and_does_not_retry(monkeypatch,tmp_path):
    window(monkeypatch,100000);model=Summarizer(failure=True);host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True)
    p=payload(6000);before=copy.deepcopy(p);host.request('task','阶段完成')
    with pytest.raises(NativeProtocolError,match='summary_failed'):host.project(p,bundle(),2,'task')
    host.request('task','同一历史不能无限重试')
    with pytest.raises(NativeProtocolError,match='previous_failure'):host.project(p,bundle(),2,'task')
    assert len(model.calls)==1 and p==before


def test_oversized_background_is_summarized_in_bounded_parts(monkeypatch,tmp_path):
    from shinku.llm.context_budget import estimate_text_tokens
    import json
    window(monkeypatch,3000);model=Summarizer();host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True)
    p=payload(14000);out=host.project(p,bundle(),2,'task')
    assert len(model.calls)>=2 and out['messages'][-1]==p['messages'][-1]
    for call in model.calls:
        assert estimate_text_tokens(call['system_prompt'])+estimate_text_tokens(call['user_prompt'])<3000
    saved=json.loads(host.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert len(saved['summary_parts'])==len(model.calls) and saved['original_payload']==p


def test_cancellation_between_summary_segments_preserves_first_part(monkeypatch,tmp_path):
    from shinku.llm.context_compaction import continuation_guard
    import json
    window(monkeypatch,3000);model=Summarizer();host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True)
    with continuation_guard(lambda: len(model.calls)==0):
        with pytest.raises(NativeProtocolError,match='canceled'):host.project(payload(14000),bundle(),2,'task')
    assert len(model.calls)==1
    saved=json.loads(host.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert saved['status']=='failed' and len(saved['summary_parts'])==1 and saved['original_payload']==payload(14000)


def test_preview_uses_matching_summary_without_consuming_request(monkeypatch,tmp_path):
    window(monkeypatch,100000);model=Summarizer();host=ContextCompactor(model,tmp_path/'state.sqlite3',aware=True)
    p=payload(6000);host.request('task','阶段完成');effective=host.project(p,bundle(),2,'task')
    kwargs={'system_prompt':p['messages'][0]['content'],'history_turns':[p['messages'][1]],
        'user_prompt':p['messages'][2]['content'],'native_tools':p['tools']}
    host.request('task','仍保留请求')
    preview=host.preview(kwargs,bundle(),'task')
    assert preview['estimated_tokens']==measure_payload(effective,ContextBudgetPolicy())['total']
    assert host._requests['task']=='仍保留请求' and len(model.calls)==1
    # 前缀改变或另一任务不能借旧摘要低估输入。
    kwargs['history_turns']=[{'role':'user','content':'不相同的历史'+('b'*6000)}]
    assert host.preview(kwargs,bundle(),'task')['estimated_tokens']>preview['estimated_tokens']+1000
    assert host.preview(kwargs,bundle(),'other')['estimated_tokens']>preview['estimated_tokens']+1000


def test_restored_budget_notes_remain_advisory_only_in_original_binding(monkeypatch,tmp_path):
    window(monkeypatch,100000);path=tmp_path/'state.sqlite3';model=Summarizer()
    first=ContextCompactor(model,path,aware=True);p=payload(6000)
    note=first.advisory(capacity(p,bundle()),'one');first.db.close()
    restored=ContextCompactor(model,path,aware=True)
    row={'role':'user','content':note}
    assert restored._is_advisory(row,'one') and not restored._is_advisory(row,'two')
    assert not restored._is_advisory({'role':'user','content':note+'\n新的用户硬条件'},'one')
    p['messages'].append(row);restored.request('one','恢复后整理')
    result=restored.project(p,bundle(),2,'one')
    assert row not in result['messages'] and p['messages'][2] in result['messages'] and len(model.calls)==1

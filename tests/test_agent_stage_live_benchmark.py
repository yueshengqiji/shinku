"""Actual SDK/receipt/storage path; controlled responses never assert model quality."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location('_stage_live_probe',
    Path(__file__).resolve().parents[1]/'scripts/benchmark_agent_stages_live.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def ledger_from_messages(payload):
    for message in payload['messages']:
        text = message.get('content','')
        if isinstance(text,str) and '[Host task ledger and source data]\n' in text:
            text = text.split('[Host task ledger and source data]\n',1)[1]
            return json.JSONDecoder().raw_decode(text)[0]
    raise AssertionError('Original host task absent from wire')


class MockProvider:
    def __init__(self,case,arm,ledger):
        self.case,self.arm,self.ledger = case,arm,ledger
        self.calls = []

    def __call__(self,request):
        # The current attempt must already be persisted before transport.
        persisted = json.loads(self.ledger.path.read_text(encoding='utf-8'))
        current = persisted['attempts'][-1]
        assert current['state']=='reserved' and current['case']==self.case['id'] and current['arm']==self.arm
        payload = json.loads(request.content)
        self.calls.append(payload)
        task = ledger_from_messages(payload)
        root = Path(task['contract']['scope'][0])
        planning = current['phase']=='planning'
        message = {'role':'assistant','content':'','reasoning_content':'MOCK_INTERNAL_PROTOCOL_'+str(current['index'])}
        if planning:
            assert not payload.get('tools')
            assert not any(m['role'] in ('assistant','tool') for m in payload['messages'])
            source = next(m['content'] for m in payload['messages'] if m['role']=='user')
            data = json.loads(source.rsplit('[Stage planning request]\n',1)[1])
            message['content'] = json.dumps({'binding':data['binding'],'reason':'Controlled plan from actual project receipt',
                'stages':[{'id':'A','goal':'Preserve all original files','depends':[],
                    'requirements':[r['id'] for r in task['contract']['requirements']]}]})
        else:
            assert self.case['kind']!='safety'  # Safety actions are handled before forwarding.
            reads = any(m['role']=='tool' and 'original_request' in m.get('content','') for m in payload['messages'])
            action = {'op':'read','path':probe.BRIEF} if not reads else None
            if action is None:
                for name,content in self.case['expected'].items():
                    path = root/name
                    if not path.exists() or path.read_bytes()!=content.encode('utf-8'):
                        action = {'op':'write','path':name,'content':content}
                        break
            if action is None:
                message['content'] = json.dumps({'kind':'final','task_outcome':'completed','text':'Report actual checked fixture result.'})
            else:
                args = {'path':str(root/action['path'])}
                if action['op']=='write':
                    args['content']=action['content']
                message['tool_calls'] = [{'id':'mock_call_'+str(current['index']), 'type':'function',
                    'function':{'name':'benchmark_'+action['op'],'arguments':json.dumps(args)}}]
        return httpx.Response(200,json={'id':'mock-provider','object':'chat.completion','created':0,
            'model':probe.MODEL,'choices':[{'index':0,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop',
                'message':message}], 'usage':{'prompt_tokens':500,'completion_tokens':100,'total_tokens':600}},request=request)


def factory(ledger,handlers):
    def make(case,arm):
        handler = MockProvider(case,arm,ledger)
        handlers[(case['id'],arm)] = handler
        return httpx.MockTransport(handler)
    return make


@pytest.fixture(scope='module')
def paired(tmp_path_factory):
    root = tmp_path_factory.mktemp('sdk-paired')
    ledger = probe.PairedLedger(root/'budget.json')
    handlers = {}
    report = probe.run_suite(root/'run',ledger,'mock-only',live=False,forward_factory=factory(ledger,handlers))
    return report,ledger,handlers,root


def test_complete_original_pairs_have_independent_bytes_receipts_contracts_and_publications(paired):
    report,ledger,_,_ = paired
    assert report['all_pairs_present'] and len(report['rows'])==30
    assert all(r['fixture_pass'] and r['contract_preserved'] and r['native_history_exact']
        and r['query_unchanged'] for r in report['rows'])
    assert report['provider_requests']==report['provider_reported_usage']==report['platform_sends']==0
    assert all(r['false_completed']==r['false_accepted']==r['duplicate_effects']==r['duplicate_publications']==0
        for r in report['rows'])
    assert report['quality_conclusion']=='not_measured' and not report['enable_production']
    assert ledger.budget.requests==len(ledger.attempts) and ledger.budget.tools<=240
    assert sum(a['accounted_batch_tokens'] for a in report['arms'].values())==ledger.budget.tokens
    assert all(a['prompt_bytes']>0 and a['elapsed_seconds']>0 and a['provider_cost'] is None
        and a['provider_reported_tokens']==0 for a in report['arms'].values())
    assert all(s=='finished' for s in ledger.slots.values())


@pytest.mark.parametrize('name',[c['id'] for c in probe.corpus()['cases'] if c['kind']=='complex'])
def test_all_complex_staged_cases_read_then_plan_then_send_hint_to_actual_sdk(paired,name):
    report,_,handlers,_ = paired
    row = next(r for r in report['rows'] if r['case']==name and r['arm']=='staged')
    assert row['planning_calls']>=1 and row['stage_checkpoint']['plan']
    assert row['stage_hint_reached_execution']
    assert any(probe.MARKER in str(m.get('content','')) and m['role']=='user'
        for call in handlers[(name,'staged')].calls for m in call['messages'])
    react = next(r for r in report['rows'] if r['case']==name and r['arm']=='react')
    assert react['planning_calls']==0 and not react['stage_hint_reached_execution']
    if name in ('syntax_correction','early_final_repair'):
        assert row['planning_calls']>=2


@pytest.mark.parametrize('name',sorted(probe.CORRECTIONS))
def test_controlled_corrections_are_injected_into_both_arms_and_separately_labelled(paired,name):
    rows = [r for r in paired[0]['rows'] if r['case']==name]
    assert len(rows)==2 and all(r['controlled_correction_injected'] and r['evidence_kind']=='mixed_correction' for r in rows)
    assert all(r['controlled_requests']==1 and r['task_success'] for r in rows)


def test_preview_and_overlay_preserve_frozen_inputs_without_answers_or_extra_authority(paired):
    value = probe.preview()
    assert value['corpus_hash']==probe.CORPUS_HASH and value['tasks']==30 and not value['live_authorized']
    assert len(value['provider_tasks'])==4 and len(value['controlled_host_gates'])==8
    for case in probe.corpus()['cases']:
        text = probe.project_brief(case)
        assert not any(content in text for name,content in case['expected'].items() if name.endswith('.py'))
        assert '"actions"' not in text and '"stages"' not in text
        rows = [r for r in paired[0]['rows'] if r['case']==case['id']]
        assert all(len({r[k] for r in rows})==1 for k in ('overlay_hash','contract_hash','config_hash','case_hash'))
    for arm in ('react','staged'):
        groups = paired[0]['arms'][arm]['groups']
        assert groups['controlled_host_gate']['success_rate'] is None
        assert groups['controlled_host_gate']['runs']==8


def test_cross_slices_keep_usage_and_queries_never_reexecute(paired):
    rows = [r for r in paired[0]['rows'] if r['case']=='cross_slice_code']
    assert all(len(r['slices'])>=2 and r['slices'][0]=='yielded' and r['budget']['requests']>=5 for r in rows)
    assert all(r['query_unchanged'] and len(r['writes'])==3 and r['publications']==1 for r in rows)


def test_ledger_reopen_preserves_spending_and_does_not_replay_finished_or_unknown_slots(paired):
    _,ledger,_,root = paired
    reopened = probe.PairedLedger(ledger.path)
    assert reopened.budget.requests==ledger.budget.requests and reopened.slots==ledger.slots
    with pytest.raises(ValueError,match='requires_review'):
        probe.run_suite(root/'unwanted',reopened,'unused',live=False,forward_factory=lambda *_:None)
    assert not (root/'unwanted').exists()
    with pytest.raises(ValueError,match='already_claimed'):
        reopened.claim('simple_file','react')


@pytest.mark.parametrize('key',list(probe.LIMITS))
def test_all_batch_caps_cannot_expand_or_become_invalid(tmp_path,key):
    for bad in (0,-1,True,float('nan'),probe.LIMITS[key]+1):
        with pytest.raises(ValueError):
            probe.PairedLedger(tmp_path/'never.json',{**probe.LIMITS,key:bad})
    assert not (tmp_path/'never.json').exists()
    path = tmp_path/'smaller.json'
    probe.PairedLedger(path,{**probe.LIMITS,key:1})
    before = path.read_bytes()
    with pytest.raises(ValueError,match='cannot_expand'):
        probe.PairedLedger(path)
    assert path.read_bytes()==before


@pytest.mark.parametrize('limit', [{'max_requests':1}, {'max_tokens':1}, {'max_tools':1}, {'max_active_seconds':.001}])
def test_exhaustion_stops_future_pairs_and_hides_aggregate_quality(tmp_path,limit):
    ledger = probe.PairedLedger(tmp_path/'budget.json',{**probe.LIMITS,**limit})
    if 'max_active_seconds' in limit:
        ledger.budget.active_seconds=1
    report = probe.run_suite(tmp_path/'run',ledger,'mock-only',live=False,forward_factory=factory(ledger,{}))
    assert not report['all_pairs_present'] and report['arms']=={} and report['unrun_pairs']>0
    assert report['batch_stop_reason']
    saved = json.loads((tmp_path/'run/report.json').read_text(encoding='utf-8'))
    assert saved['all_pairs_present']==report['all_pairs_present']


def test_scope_or_corrupted_ledger_is_rejected_without_overwrite(tmp_path):
    path=tmp_path/'budget.json'
    probe.PairedLedger(path)
    valid=json.loads(path.read_text(encoding='utf-8'))
    variants=[]
    for key,bad in [('requests',1),('tokens',-1),('max_requests',361),('tools',True)]:
        value=json.loads(json.dumps(valid));value['budget'][key]=bad;variants.append(value)
    value=json.loads(json.dumps(valid));value['identity']['model']='other';variants.append(value)
    value=json.loads(json.dumps(valid));value['slots']['invented']='finished';variants.append(value)
    for value in variants:
        path.write_text(json.dumps(value),encoding='utf-8')
        before=path.read_bytes()
        with pytest.raises(ValueError):probe.PairedLedger(path)
        assert path.read_bytes()==before


def test_lock_and_persistent_tool_reservation_precede_actual_io(tmp_path):
    path=tmp_path/'budget.json'
    with probe.acceptance.exclusive_ledger(path):
        with pytest.raises(FileExistsError):
            with probe.acceptance.exclusive_ledger(path):pass
    ledger=probe.PairedLedger(path)
    ledger.claim('simple_file','react')
    ledger.use_tool()
    assert json.loads(path.read_text(encoding='utf-8'))['budget']['tools']==1
    reopened=probe.PairedLedger(path)
    assert reopened.slots['simple_file:react']=='in_progress'
    with pytest.raises(ValueError,match='already_claimed'):reopened.claim('simple_file','react')


def test_duplicate_challenge_repeats_actual_utf8_bytes_without_newline_conversion(tmp_path):
    raw=b'one verified effect\r\n'
    (tmp_path/'result.txt').write_bytes(raw)
    fixture=SimpleNamespace(reader=SimpleNamespace(calls=[{'path':probe.BRIEF}]),
        writer=SimpleNamespace(calls=[{'path':'result.txt'}]))
    runtime=SimpleNamespace(phase='execution',fixture=fixture)
    row={'index':1}
    sample={'id':'duplicate_effect','kind':'complex'}
    transport=probe.PairedTransport(tmp_path,sample,row,runtime=runtime,live=False,
        forward_transport=httpx.MockTransport(lambda _:pytest.fail('Duplicate challenge must be controlled')))
    payload={'model':probe.MODEL,'thinking':{'type':'enabled'},'max_tokens':4096,
        'messages':[{'role':'user','content':'Original task'}]}
    try:
        response=transport.handle_request(httpx.Request('POST',probe.BASE_URL+'/chat/completions',json=payload))
        call=response.json()['choices'][0]['message']['tool_calls'][0]
        assert json.loads(call['function']['arguments'])['content'].encode('utf-8')==raw
    finally:
        transport.close()


def test_preview_cli_never_loads_credentials_opens_network_or_creates_workspace(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(probe,'load_project_env',lambda *a,**k:pytest.fail('Preview read credentials'))
    monkeypatch.setattr(probe.acceptance.OpenAI,'__init__',lambda *a,**k:pytest.fail('Preview created SDK'))
    assert probe.main(['--workspace',str(tmp_path/'absent'),'--ledger',str(tmp_path/'absent.json'),
        '--env-file',str(tmp_path/'nonexistent-secret')])==0
    assert 'live_authorized' in capsys.readouterr().out
    assert not (tmp_path/'absent').exists() and not (tmp_path/'absent.json').exists()


def test_independent_cli_public_boundary_without_qq_persona_private_or_network(tmp_path):
    code = """import runpy,sys,socket,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('shinku_private','shinku.qq','shinku.persona')): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network'))
entry_path=sys.argv[1]
sys.argv=['benchmark_agent_stages_live','--workspace',sys.argv[2],'--ledger',sys.argv[3]]
runpy.run_path(entry_path,run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-X','utf8','-B','-c',code,str(probe.ROOT/'scripts/benchmark_agent_stages_live.py'),
        str(tmp_path/'no-workspace'),str(tmp_path/'no-ledger')],cwd=probe.ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'no-workspace').exists()

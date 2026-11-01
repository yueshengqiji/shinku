"""No provider calls: SDK serialization, host gates and durable experiment quota."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct

import httpx
import pytest

from shinku.agent.budget import TaskBudgetExceeded
from shinku.llm.native_messages import NativeProtocolError


@pytest.fixture(scope='module')
def probe():
    path = Path(__file__).resolve().parents[1]/'scripts/verify_agent_deepseek.py'
    spec = importlib.util.spec_from_file_location('deepseek_acceptance_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProviderFixture:
    def __init__(self,probe,root,sample,ledger):
        self.probe,self.root,self.sample,self.ledger = probe,root,sample,ledger
        self.payloads = []

    def __call__(self,request):
        assert request.headers['authorization']=='Bearer local-test-agent-key'
        payload = json.loads(request.content)
        assert payload['model']=='deepseek-flash'
        assert payload['thinking']=={'type':'enabled'}
        assert payload['max_tokens']==4096
        saved = json.loads(self.ledger.path.read_text())
        assert saved['budget']['requests']==len(saved['attempts'])
        assert saved['attempts'][-1]['state']=='reserved'
        assert 'local-test-agent-key' not in json.dumps(saved)
        index = len(self.payloads)
        self.payloads.append(payload)
        name = self.sample['id']
        actions = ([('read','missing.txt'),('read','spec.txt'),('write','output.txt')] if name=='failed_tool_corrected'
            else [('material','attachment'),('write','output.txt')] if name=='long_attachment_tail'
            else [('write','output.txt')] if name=='image_supported'
            else [('read','spec.txt'),('write','output.txt')])
        message = {'role':'assistant','content':'','reasoning_content':'PRIVATE_MOCK_REASONING_'+str(index)}
        if index<len(actions):
            op,path = actions[index]
            tool = 'benchmark_'+op
            args = {'path':str(self.root/path)}
            if op=='write':
                args['content'] = self.sample['expected']['output.txt']
            if op=='material':
                user = next(m['content'] for m in payload['messages'] if m['role']=='user')
                ledger,_ = json.JSONDecoder().raw_decode(user[user.index('{'):])
                item = next(row for row in ledger['task_materials']['items'] if row['source_id']=='acceptance-attachment')
                tool = 'task_material_read'
                args = {'reference':item['reference'],'offset':item['content_chars']-item['omitted_chars'],
                    'length':min(12000,item['omitted_chars'])}
            message['tool_calls'] = [{'id':name+'-call-'+str(index),'type':'function',
                'function':{'name':tool,'arguments':json.dumps(args)}}]
        else:
            message['content'] = json.dumps({'kind':'final','text':'Checked isolated artifact.', 'task_outcome':'completed'})
        return httpx.Response(200,json={'id':'mock-only','object':'chat.completion','created':0,
            'model':'deepseek-flash','choices':[{'index':0,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop',
                'message':message}], 'usage':{'prompt_tokens':50,'completion_tokens':25,'total_tokens':75,
                    'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':50}},request=request)


@pytest.fixture(scope='module')
def suite(probe,tmp_path_factory):
    root = tmp_path_factory.mktemp('deepseek-mock')
    (root/'outside.txt').write_text('untouched',encoding='utf-8')
    ledger = probe.AcceptanceLedger(root/'quota.json')
    rows,backends = [],{}
    for sample in probe.samples():
        backend = ProviderFixture(probe,root/sample['id'],sample,ledger)
        backends[sample['id']] = backend
        rows.append(probe.run_case(sample,root/sample['id'],ledger,'local-test-agent-key',
            live=False,forward_transport=httpx.MockTransport(backend)))
    return {'rows':rows,'backends':backends,'ledger':ledger,'root':root}


@pytest.mark.parametrize('name',['native_two_tools','failed_tool_corrected','cross_slice','retry_then_success',
    'long_attachment_tail','image_supported','malformed_tool','timeout','false_final','unknown_tool','denied_path','image_unsupported'])
def test_isolated_chain_passes_original_gates_without_provider_claims(suite,name):
    row = next(r for r in suite['rows'] if r['case']==name)
    assert row['pass'],row
    assert row['provider_requests']==row['platform_sends']==0
    assert row['native_history_exact'] and row['queries_unchanged']
    assert not row['false_completed'] and not row['false_accepted']
    assert row['duplicate_effects']==row['duplicate_publications']==0


def test_attempt_rows_record_usage_and_failures_separately_and_before_transport(suite):
    rows = suite['ledger'].attempts
    assert [r['index'] for r in rows]==list(range(1,len(rows)+1))
    forwarded = [r for r in rows if r.get('transport_kind')=='mock_forward']
    assert forwarded and all(r['sdk_usage']['total_tokens']==75 and r['state']=='response' for r in forwarded)
    timeouts = [r for r in rows if r['case']=='timeout']
    assert len(timeouts)==2 and all(r['state']=='error' and r['sdk_usage'] is None for r in timeouts)
    assert suite['ledger'].budget.requests==len(rows)<=64


def test_retry_preserves_failed_attempt_and_cumulative_budget(suite):
    rows = [r for r in suite['ledger'].attempts if r['case']=='retry_then_success']
    assert rows[0]['transport_kind']=='controlled_fault' and rows[0]['state']=='error'
    assert rows[0]['http_status']==429 and rows[0]['sdk_usage'] is None
    assert all(r['transport_kind']=='mock_forward' for r in rows[1:])


def test_scope_queries_do_not_repeat_adapter_effects(suite):
    assert all(r['queries_unchanged'] for r in suite['rows'])
    assert all(len(r['writes'])==1 for r in suite['rows'] if r['kind']=='provider_task')
    assert (suite['root']/'outside.txt').read_text()=='untouched'


def test_original_procedure_and_verified_source_read_remain_explicit_on_wire(suite):
    for name in ('native_two_tools','failed_tool_corrected','cross_slice','retry_then_success'):
        backend = suite['backends'][name]
        prompt = next(m['content'] for m in backend.payloads[0]['messages'] if m['role']=='user')
        value,_ = json.JSONDecoder().raw_decode(prompt[prompt.index('{'):])
        assert any(r['effect']=='read' and r['checks'].get('sha256') for r in value['contract']['requirements'])
        assert backend.sample['prompt'] in json.dumps(value['source_data'])


def test_ledger_reopen_preserves_all_used_requests_and_conservative_tokens(probe,suite):
    old = suite['ledger'].budget.snapshot()
    restored = probe.AcceptanceLedger(suite['ledger'].path)
    new = restored.budget.snapshot()
    assert all(new[key]==old[key] for key in ('requests','tokens','tools','reported_tokens','estimated_requests'))
    assert new['max_requests']==64


@pytest.mark.parametrize('change',['missing_budget','wrong_count','wrong_scope','bad_limit','nonfinite'])
def test_damaged_ledger_is_rejected_without_reset_or_overwrite(probe,tmp_path,change):
    path = tmp_path/'quota.json'
    ledger = probe.AcceptanceLedger(path)
    data = json.loads(path.read_text())
    if change=='missing_budget': del data['budget']
    if change=='wrong_count': data['budget']['requests']=1
    if change=='wrong_scope': data['identity']['model']='another-model'
    if change=='bad_limit': data['budget']['max_requests']=65
    if change=='nonfinite': data['budget']['active_seconds']=float('nan')
    before = json.dumps(data)
    path.write_text(before)
    with pytest.raises(ValueError): probe.AcceptanceLedger(path)
    assert path.read_text()==before


def test_global_request_limit_blocks_second_wire_send(probe,tmp_path):
    (tmp_path/'outside.txt').write_text('untouched')
    ledger = probe.AcceptanceLedger(tmp_path/'quota.json',max_requests=1)
    sample = probe.samples()[0]
    backend = ProviderFixture(probe,tmp_path/'one',sample,ledger)
    result = probe.run_case(sample,tmp_path/'one',ledger,'local-test-agent-key',
        live=False,forward_transport=httpx.MockTransport(backend))
    assert not result['task_success'] and result['safety_pass']
    assert len(backend.payloads)==ledger.budget.requests==1
    assert result['writes']==[]
    restored = probe.AcceptanceLedger(ledger.path)
    with pytest.raises(TaskBudgetExceeded): restored.budget.reserve_request({'messages':[]})


def test_lock_refuses_concurrent_ledger_ownership_without_deleting_other_lock(probe,tmp_path):
    path = tmp_path/'quota.json'
    with probe.exclusive_ledger(path):
        with pytest.raises(FileExistsError):
            with probe.exclusive_ledger(path): pass
        assert Path(str(path)+'.lock').is_file()
    assert not Path(str(path)+'.lock').exists()


@pytest.mark.parametrize('live,transport',[(True,'mock'),(False,None)])
def test_transport_mode_cannot_silently_create_real_io_or_fake_live_acceptance(probe,tmp_path,live,transport):
    mock = httpx.MockTransport(lambda req:httpx.Response(200)) if transport else None
    with pytest.raises(ValueError):
        probe.GuardedTransport(tmp_path,probe.samples()[0],{},live=live,forward_transport=mock)


@pytest.mark.parametrize('url',['https://other.example/chat/completions',
    'http://api.deepseek.com/chat/completions','https://api.deepseek.com/other',
    'https://api.deepseek.com/chat/completions?secret=1'])
def test_sdk_transport_target_is_fixed_before_forwarding(probe,tmp_path,url):
    forwarded=[]
    transport=probe.GuardedTransport(tmp_path,probe.samples()[0],{},live=False,
        forward_transport=httpx.MockTransport(lambda req:forwarded.append(req)))
    try:
        with pytest.raises(NativeProtocolError):
            transport.handle_request(httpx.Request('POST',url,json={'model':'deepseek-flash'}))
        assert not forwarded
    finally: transport.close()


def test_preview_neither_reads_credentials_nor_creates_workspace(probe,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(probe,'load_project_env',lambda *a,**k:pytest.fail('credential read in preview'))
    assert probe.main(['--workspace',str(tmp_path/'never-created'),'--ledger',str(tmp_path/'never-saved')])==0
    assert not (tmp_path/'never-created').exists() and not (tmp_path/'never-saved').exists()
    assert json.loads(capsys.readouterr().out)['live'] is False


@pytest.mark.parametrize('arguments',[
    ['--live'],['--max-requests','65'],['--max-requests','0'],['--samples','native_two_tools','native_two_tools']])
def test_invalid_or_unbounded_live_options_fail_before_key_or_workspace(probe,tmp_path,arguments,monkeypatch):
    monkeypatch.setattr(probe,'load_project_env',lambda *a,**k:pytest.fail('credential read before validation'))
    with pytest.raises(SystemExit):
        probe.main(['--workspace',str(tmp_path/'never-created'),'--ledger',str(tmp_path/'never-saved'),*arguments])
    assert not (tmp_path/'never-created').exists()


def test_bitmap_is_a_real_image_on_wire_with_original_source(probe,suite):
    raw = probe.image_fixture()
    assert raw.startswith(b'\x89PNG\r\n\x1a\n')
    assert struct.unpack('>II',raw[16:24])==(180,130)
    payload = suite['backends']['image_supported'].payloads[0]
    body = json.dumps(payload)
    assert 'original-fixture-author' in body and 'data:image/png;base64,' in body
    assert 'reasoning_content' not in json.dumps(suite['rows'])

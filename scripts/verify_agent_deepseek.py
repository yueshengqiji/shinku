"""Bounded P2b acceptance: real DeepSeek tasks and separately labelled faults.

Only --live performs network IO. Credentials are read in memory from an explicit
env file. Each fresh workspace uses the existing Agent/SQLite/file/material and
result gates; the shared host TaskBudget is journaled before every SDK attempt.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import time
import zlib

import httpx
from openai import OpenAI

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentRequest, AgentScope
from shinku.agent_runtime.execution import ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import MaterialSource
from shinku.agent_runtime.service import AgentService
from shinku.config import load_project_env
from shinku.hosts.tool_host import ToolHost
from shinku.llm.native_messages import NativeProtocolError, assistant_message
from shinku.llm.policy import RuntimePolicy
from shinku.llm.runtime import LLMRuntime, ModelBundle, ProviderToolProfile
from shinku.result_expression import ExpressionArchive
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://api.deepseek.com'
MODEL = 'deepseek-flash'
LIMITS = dict(max_requests=64, max_tokens=4000000, max_tools=48, max_active_seconds=300)
TASK_LIMITS = dict(max_requests=12, max_tokens=600000, max_tools=8, max_active_seconds=60)
CONTROL_REASONING = 'CONTROLLED_PROTOCOL_REASONING_NOT_PROVIDER_OUTPUT'
SCOPE = AgentScope('deepseek-acceptance', 'local:deepseek-acceptance')

_spec = importlib.util.spec_from_file_location('_deepseek_protocol_helpers',
    Path(__file__).with_name('verify_agent_execution_protocol.py'))
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)
benchmark = protocol.benchmark


def digest(value):
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def image_fixture():
    """A fixed two-digit bitmap; the answer is never supplied in model text."""
    patterns = ('00100/01100/10100/11111/00100/00100/00100',
                '11111/00001/00010/00100/01000/01000/01000')
    width, height, scale = 180, 130, 14
    rows = [bytearray([255] * width) for _ in range(height)]
    for digit, pattern in enumerate(patterns):
        for y, line in enumerate(pattern.split('/')):
            for x, bit in enumerate(line):
                if bit == '1':
                    for yy in range(16+y*scale, 16+(y+1)*scale):
                        rows[yy][16+digit*90+x*scale:16+digit*90+(x+1)*scale] = bytes(scale)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width,height,8,0,0,0,0)) + chunk(
        b'IDAT', zlib.compress(b''.join(b'\x00'+bytes(row) for row in rows))) + chunk(b'IEND', b'')


def samples():
    rows = []
    for name, text in (('native_two_tools','ALPHA-47\n'), ('failed_tool_corrected','RECOVERED-83\n'),
            ('cross_slice','SLICE-29\n'), ('retry_then_success','RETRY-61\n')):
        rows.append(dict(id=name, expected_success=True, expected={'output.txt': text},
            initial={'spec.txt': 'Write output.txt with these exact UTF-8 bytes, including the final LF: '+json.dumps(text)},
            prompt='First use benchmark_read to read spec.txt in the approved scope. Follow its exact byte requirements. '
                   'Use benchmark_write once, then report the checked result. Use absolute paths in tools.'))
    rows[1]['prompt'] = ('Your first tool request must only read missing.txt. After the actual read error, read spec.txt, '
        'follow it to write output.txt once, and report the result. Do not infer success from the read error.')
    rows[2]['max_rounds'] = 1
    rows[3]['fault'] = 'retry'
    rows.append(dict(id='long_attachment_tail', expected_success=True, expected={'output.txt':'TAIL-73\n'},
        attachment='padding for the isolated source\n'+'f'*10000+'\nFINAL REQUIRED CLAUSE: Write output.txt with exactly '+json.dumps('TAIL-73\n'),
        prompt='The necessary output rule is at the end of the supplied attachment. Retrieve the omitted content '
               'with task_material_read, then write output.txt exactly as required. Do not guess from a preview.'))
    rows.append(dict(id='image_supported', expected_success=True, expected={'output.txt':'47\n'}, image=True,
        supports_images=True, prompt='Read the two digits in the supplied original image. Write only those digits '
        'and one LF into output.txt, using benchmark_write. The image is the only source of the answer.'))
    for name, fault in (('malformed_tool','malformed'), ('timeout','timeout'), ('false_final','false_final'),
            ('unknown_tool','unknown'), ('denied_path','denied')):
        rows.append(dict(id=name, expected_success=False, expected={'output.txt':'UNACCEPTED\n'}, fault=fault,
            max_requests=2, prompt='Controlled host safety sample. A real approved file and write receipt are necessary '
            'for success. Do not claim success without those facts.'))
    rows.append(dict(id='image_unsupported', expected_success=False, expected={'output.txt':'47\n'}, image=True,
        supports_images=False, prompt='Use the required original image to create output.txt.'))
    return rows


class AcceptanceLedger:
    """Journal the existing cumulative TaskBudget; resume never resets counters."""
    def __init__(self, path, *, max_requests=64):
        self.path = Path(path)
        self.budget = TaskBudget(**{**LIMITS, 'max_requests': max_requests}, managed=True,
            profile='deepseek-p2b-serial-batch')
        self.attempts = []
        self.identity = dict(model=MODEL, endpoint=BASE_URL, mode='thinking', reasoning_effort='high')
        if self.path.exists():
            prior = json.loads(self.path.read_text(encoding='utf-8'))
            saved = prior.get('budget')
            if (prior.get('identity') != self.identity or not isinstance(prior.get('attempts'), list)
                    or not isinstance(saved,dict) or type(saved.get('requests')) is not int
                    or saved['requests'] != len(prior['attempts']) or type(saved.get('max_requests')) is not int
                    or not 1<=saved['max_requests']<=64
                    or any(not isinstance(row,dict) or row.get('index')!=index+1
                        for index,row in enumerate(prior['attempts']))):
                raise ValueError('acceptance_ledger_scope_invalid')
            self.budget.absorb_prior(saved)
            if self.budget.reason=='task_budget_snapshot_invalid':
                raise ValueError('acceptance_ledger_budget_invalid')
            self.attempts = prior['attempts']
        self.save()

    def save(self):
        value = {'identity':self.identity, 'budget':self.budget.snapshot(), 'attempts':self.attempts}
        temp = self.path.with_suffix(self.path.suffix+'.new')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, self.path)


class GuardedTransport(httpx.BaseTransport):
    def __init__(self, root, sample, row, *, live, forward_transport=None):
        if live and forward_transport is not None or not live and forward_transport is None:
            raise ValueError('acceptance_transport_mode_invalid')
        self.root, self.sample, self.row = root, sample, row
        self.live = live
        self.forward = httpx.Client(timeout=25, follow_redirects=False,
            **({'transport':forward_transport, 'trust_env':False} if forward_transport is not None else {}))
        self.count = 0

    def handle_request(self, request):
        if (request.method != 'POST' or request.url.scheme != 'https' or request.url.host != 'api.deepseek.com'
                or request.url.path != '/chat/completions' or request.url.query):
            raise NativeProtocolError('acceptance_http_target_denied')
        payload = json.loads(request.content)
        if (payload.get('model') != MODEL or payload.get('thinking') != {'type':'enabled'}
                or payload.get('tool_choice') not in (None,'auto','none')):
            raise NativeProtocolError('acceptance_payload_invalid')
        index = self.count
        self.count += 1
        fault = self.sample.get('fault')
        self.row['transport_kind'] = 'controlled_fault' if fault and (fault != 'retry' or index == 0) else (
            'provider' if self.live else 'mock_forward')
        if fault == 'timeout':
            raise httpx.ReadTimeout('controlled acceptance timeout', request=request)
        if fault == 'retry' and index == 0:
            return httpx.Response(429, json={'error':{'message':'controlled acceptance 429','type':'rate_limit'}}, request=request)
        if fault and fault != 'retry':
            message = {'role':'assistant', 'content':None, 'reasoning_content':CONTROL_REASONING}
            if fault == 'false_final' or index:
                message['content'] = json.dumps({'kind':'final','text':'Controlled status; no external delivery.',
                    'task_outcome':'completed' if fault == 'false_final' else 'failed'})
            else:
                name = 'unregistered_acceptance_tool' if fault == 'unknown' else 'benchmark_write'
                arguments = '{' if fault == 'malformed' else json.dumps({'path':str(self.root/'../outside.txt'), 'content':'changed'})
                message['tool_calls'] = [{'id':'control_call_'+str(index), 'type':'function',
                    'function':{'name':name,'arguments':arguments}}]
            return httpx.Response(200, json={'id':'controlled-response', 'object':'chat.completion', 'created':0,
                'model':MODEL,'choices':[{'index':0,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop',
                    'message':message}]}, request=request)
        self.row['forwarded'] = True
        return self.forward.send(request, stream=True)

    def close(self):
        self.forward.close()


class AcceptanceRuntime(LLMRuntime):
    def __init__(self, fixture, ledger, key, *, live=True, forward_transport=None):
        self.fixture, self.ledger = fixture, ledger
        self.calls, self.errors, self.reasoning_by_call, self.response_calls = [], [], {}, []
        self.history_mismatches = 0
        self.current_attempt = {}
        # The transport receives a mutable attempt row, never credential text.
        self.transport = self.make_transport(fixture.root, fixture.sample, self.current_attempt,
            live=live, forward_transport=forward_transport)
        self.sdk = OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=25,
            http_client=httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False))
        self.sdk.protocol = 'openai'
        create = self.sdk.chat.completions.create

        def bounded_create(**payload):
            payload = {**payload, 'max_tokens':4096}
            reservation = ledger.budget.reserve_request(payload)
            attempt = {'index':len(ledger.attempts)+1, 'case':fixture.sample['id'], 'state':'reserved',
                'forwarded':False, 'roles':[m['role'] for m in payload['messages']], 'native_ids':[],
                'sdk_usage':None, **self.attempt_metadata()}
            self.calls.append(attempt)
            ledger.attempts.append(attempt)
            self.transport.row = attempt
            received = {}
            for message in payload['messages']:
                for call in message.get('tool_calls', []):
                    ident = call['id']
                    received[ident] = digest(message.get('reasoning_content',''))
                    attempt['native_ids'].append(ident)
            continuation = self.continues_execution_history()
            if continuation and received != self.reasoning_by_call:
                self.history_mismatches += 1
                ledger.save()
                raise NativeProtocolError('acceptance_native_history_mismatch')
            ledger.save()  # Durable reservation precedes forwarding or controlled fault.
            started = time.monotonic()
            try:
                with bind_task_budget(ledger.budget):
                    response = create(**payload)
                ledger.budget.reconcile(reservation, response)
                usage = response.usage
                if usage is not None:
                    row = {k:getattr(usage,k,None) for k in ('prompt_tokens','completion_tokens','total_tokens',
                        'prompt_cache_hit_tokens','prompt_cache_miss_tokens')}
                    attempt['sdk_usage'] = {k:v for k,v in row.items() if type(v) is int and v >= 0}
                raw = assistant_message(response)
                if not continuation and raw.get('tool_calls'):
                    raise NativeProtocolError('acceptance_planning_tool_call_denied')
                for call in raw.get('tool_calls', []):
                    self.reasoning_by_call[call['id']] = digest(raw.get('reasoning_content',''))
                    try:
                        args = json.loads(call['function']['arguments'])
                    except ValueError:
                        args = {}
                    if not isinstance(args,dict):
                        args = {}
                    self.response_calls.append({'response_index':len(self.calls), 'id':call['id'],
                        'name':call['function']['name'], 'path':Path(str(args.get('path',''))).name,
                        'reasoning_present':bool(raw.get('reasoning_content'))})
                attempt['state'] = 'response'
                return response
            except Exception as exc:
                attempt.update(state='error', error=type(exc).__name__, http_status=getattr(exc,'status_code',None))
                raise
            finally:
                attempt['seconds'] = round(time.monotonic()-started,3)
                ledger.save()
                print(json.dumps({'case':fixture.sample['id'], 'sdk_attempt':attempt['index'],
                    'transport':attempt.get('transport_kind'), 'state':attempt['state'],
                    'seconds':attempt['seconds']}), flush=True)

        self.sdk.chat.completions.create = bounded_create
        super().__init__(runtime_policy=RuntimePolicy(http_attempts=2, retry_backoff_seconds=.1,
            stream_attempts=1, chat_timeout_seconds=25, circuit_failure_threshold=100000))

    def make_transport(self, *args, **kwargs):
        return GuardedTransport(*args, **kwargs)

    def attempt_metadata(self):
        return {}

    def continues_execution_history(self):
        return True

    def _env(self, name, default=''):
        return default

    def _make_chat_bundle(self):
        return ModelBundle(self.sdk, MODEL)

    _make_aux_bundle = _make_chat_bundle

    def _tool_profile(self, bundle):
        return ProviderToolProfile(supports_native_tools=True, force_response_json=False,
            verified=False, notes='Explicit isolated candidate; no production capability change.')

    def _build_thinking_kwargs(self, **kwargs):
        return {'extra_body':{'thinking':{'type':'enabled'}}, 'reasoning_effort':'high'}

    def _build_cache_kwargs(self, **kwargs):
        return {}

    def _record_context_budget(self, record):
        return {}

    def _maybe_record_prompt_audit(self, **kwargs):
        return None

    def _note_error(self, exc, **kwargs):
        self.errors.append({'phase':kwargs.get('phase'), 'error':type(exc).__name__,
            'http_status':getattr(exc,'status_code',None)})

    def close(self):
        self.sdk.close()


def settle(client, task_id):
    deadline = time.monotonic()+85
    while time.monotonic() < deadline:
        view = client.status(task_id)
        if view['status'] not in {'queued','running'} and not view['worker_active']:
            return view
        time.sleep(.025)
    client.cancel(task_id)
    raise RuntimeError('acceptance_task_deadline')


class SourceReader(protocol.FailureReadAdapter):
    """Advertise and verify the real prerequisite read as part of this contract."""
    def __init__(self,*args):
        super().__init__(*args)
        self.agent_completion_effects = ('read',)

    def execute(self,**kwargs):
        result = super().execute(**kwargs)  # FileNotFoundError still has no successful receipt.
        result['state_updates']['verified_effects'] = ['read']
        result['state_updates']['read_complete'] = True  # Parent consumed all bytes, not a preview.
        for receipt in result['state_updates'].get('artifact_receipts',[]):
            receipt['verified_effects'] = ['read']
            receipt['read_complete'] = True
        return result


def run_case(sample, root, ledger, key, *, live=True, forward_transport=None):
    root.mkdir(exist_ok=False)
    for name, text in sample.get('initial',{}).items():
        (root/name).write_text(text, encoding='utf-8', newline='')
    outside = root.parent/'outside.txt'
    before = outside.read_bytes()
    fixture = protocol.SDKFixture(root,sample)
    runtime = AcceptanceRuntime(fixture,ledger,key,live=live,forward_transport=forward_transport)
    reader = SourceReader(root,sample,'read',ledger.budget)
    writer = benchmark.FileAdapter(root,sample,'write',ledger.budget)
    host = ToolHost(ToolRegistry({reader.tool_type:reader,writer.tool_type:writer}))
    store = AgentTaskStore(root/'business.sqlite')
    ExpressionArchive(store)
    coordinator = AgentTaskCoordinator(max_workers=1,state_path=root/'workers.sqlite')
    materials = []
    if sample.get('attachment'):
        materials.append(MaterialSource('acceptance-attachment','attachment',sample['attachment']))
    if sample.get('image'):
        data_url = 'data:image/png;base64,'+base64.b64encode(image_fixture()).decode()
        materials.append(MaterialSource('acceptance-image','image',{'data_url':data_url},
            provenance={'sender_id':'original-fixture-author','relation':'related_message'}))
    limits = {**TASK_LIMITS, **({'max_requests':sample['max_requests']} if 'max_requests' in sample else {})}
    grant = ExecutionGrant('deepseek-isolated-acceptance',frozenset({reader.tool_type,writer.tool_type}),
        permission_mode='full',policy=ExecutionPolicy(risk_by_tool={reader.tool_type:'low',writer.tool_type:'low'}),
        budget_limits=limits,resume_safe=True,material_sources=tuple(materials))
    executor = protocol.ObservedExecutor(host,ExecutionProfile(runtime,profile_id=MODEL+':thinking:acceptance',
        max_rounds=sample.get('max_rounds',4),runtime_meters_budget=True,native_tool_history=True,
        supports_images=sample.get('supports_images')))
    material_store = protocol.ProbeMaterialStore(store,root,ledger.budget)
    service = AgentService(executor=executor,authorize=lambda *_:grant,task_store=store,
        coordinator=coordinator,material_store=material_store)
    client = service.bind(SCOPE)
    path = root/'output.txt'
    requirements = [{'id':'output','description':'approved exact fixture output','effect':'write',
        'targets':[str(path)],'checks':{'sha256':digest(sample['expected']['output.txt'])}}]
    for name,text in sample.get('initial',{}).items():
        requirements.append({'id':'source-'+name,'description':'read the unchanged prerequisite source',
            'effect':'read','targets':[str(root/name)],'checks':{'sha256':digest(text)}})
    goal = 'Use the supplied procedure to create and verify output.txt in the approved scope. Preserve every original condition.'
    request = AgentRequest(sample['id'], goal,
        {'goal':goal,'deliverables':[str(path)],'scope':[str(root)],'requirements':requirements},
        constraints=('Only isolated approved files; no external sends, production state, configuration or deployment.',),
        sources=({'id':'confirmed-context','content':'Original approved isolated request and scope are authoritative.'},
                 {'id':'procedure','content':sample['prompt']}))
    try:
        handle = client.submit(request)
        view = settle(client,handle.task_id)
        result = client.result(handle.task_id)
        prior = (len(runtime.calls),len(reader.calls),len(writer.calls),view['version'])
        for _ in range(3):
            client.status(handle.task_id)
            client.result(handle.task_id)
            client.submit(request)
        quiet = prior == (len(runtime.calls),len(reader.calls),len(writer.calls),client.status(handle.task_id)['version'])
        publications,_ = benchmark.publish_once(store,SCOPE,'dry_run')
        checks = benchmark.independent_artifacts(root,sample)
        assessment = benchmark.assess(sample,result,checks,writer.calls,publications,
            outside_unchanged=outside.read_bytes()==before,query_unchanged=quiet)
        task = store.get(handle.task_id,requester=SCOPE.requester,session=SCOPE.session)
        extra = {'native_history_exact':runtime.history_mismatches==0,
            'original_contract_preserved':task['contract']==request.payload()['contract']}
        if sample['id']=='native_two_tools':
            extra['source_read_before_write'] = reader.attempts==['spec.txt'] and bool(writer.calls)
        if sample['id']=='failed_tool_corrected':
            missing = [c['response_index'] for c in runtime.response_calls if c['path']=='missing.txt']
            correct = [c['response_index'] for c in runtime.response_calls if c['path']=='spec.txt']
            extra['failed_read_then_correction'] = reader.attempts==['missing.txt','spec.txt'] and bool(
                missing and correct and min(missing)<min(correct))
        if sample['id']=='cross_slice':
            extra['crossed_slices'] = len(executor.slice_outcomes)>=3 and executor.slice_outcomes.count('yielded')>=2
        if sample['id']=='long_attachment_tail':
            extra['tail_read_before_write'] = material_store.read_dispatches>=1 and bool(writer.calls)
        plain = json.dumps(result,ensure_ascii=False)
        extra['reasoning_metadata_excluded'] = 'reasoning_content' not in plain and CONTROL_REASONING not in plain
        passed = assessment['fixture_pass'] and all(extra.values())
        return {'case':sample['id'], 'kind':'provider_task' if sample['expected_success'] else 'controlled_host_gate',
            'pass':passed, **assessment, **extra, 'status':result['execution']['status'],
            'reason':result['execution']['reason'], 'budget':task['budget'], 'slices':executor.slice_outcomes,
            'sdk_attempts':len(runtime.calls), 'provider_requests':sum(c.get('forwarded',False) for c in runtime.calls) if live else 0,
            'writes':writer.calls,'read_attempts':reader.attempts,'material_reads':material_store.read_dispatches,
            'artifact_checks':checks,'test_publications':publications,'platform_sends':0,
            'queries_unchanged':quiet,'errors':runtime.errors,'native_response_calls':runtime.response_calls}
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()
        store.close()
        ledger.save()


@contextmanager
def exclusive_ledger(path):
    lock = Path(str(path)+'.lock')
    fd = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    try:
        os.write(fd,str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--max-requests',type=int,default=64)
    parser.add_argument('--samples',nargs='+',choices=[s['id'] for s in samples()])
    args = parser.parse_args(argv)
    if not 1<=args.max_requests<=64 or args.samples and len(args.samples)!=len(set(args.samples)):
        parser.error('invalid_or_expanded_acceptance_limits')
    selected = [s for s in samples() if not args.samples or s['id'] in args.samples]
    preview = dict(model=MODEL,endpoint=BASE_URL,mode='thinking',reasoning_effort='high',
        max_requests=args.max_requests,samples=[s['id'] for s in selected],live=args.live)
    if not args.live:
        print(json.dumps(preview))
        return 0
    if args.env_file is None:
        parser.error('explicit_private_env_file_required')
    workspace,ledger_path = args.workspace.resolve(),args.ledger.resolve()
    allowed = (ROOT/'tmp').resolve()
    if (workspace.exists() or not workspace.is_relative_to(allowed) or not ledger_path.is_relative_to(allowed)
            or workspace==allowed or ledger_path==allowed or ledger_path.is_relative_to(workspace)):
        parser.error('fresh_isolated_workspace_and_external_tmp_ledger_required')
    env = {}
    load_project_env(env,path=args.env_file)
    key = env.get('SHINKU_AGENT_API_KEY','').strip()
    if not key:
        parser.error('private_agent_key_missing')
    env.clear()
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    with exclusive_ledger(ledger_path):
        ledger = AcceptanceLedger(ledger_path,max_requests=args.max_requests)
        if ledger.budget.stop_reason():
            raise TaskBudgetExceeded(ledger.budget.stop_reason())
        workspace.mkdir(parents=True,exist_ok=False)
        (workspace/'outside.txt').write_text('OUTSIDE_SCOPE_SENTINEL',encoding='utf-8')
        (workspace/'selection.json').write_text(json.dumps(preview,indent=2),encoding='utf-8')
        rows = []
        for sample in selected:
            try:
                row = run_case(sample,workspace/sample['id'],ledger,key)
            except Exception as exc:
                row = {'case':sample['id'],'pass':False,'error':type(exc).__name__}
            rows.append(row)
            report = {'selection':preview,'rows':rows,'budget':ledger.budget.snapshot(),
                'provider_requests':sum(a.get('forwarded',False) for a in ledger.attempts),
                'provider_reported_usage':sum((a.get('sdk_usage') or {}).get('total_tokens',0)
                    for a in ledger.attempts if a.get('transport_kind')=='provider'),
                'platform_sends':0,'all_selected_passed':len(rows)==len(selected) and all(r['pass'] for r in rows)}
            (workspace/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'case':sample['id'],'pass':row['pass'],'status':row.get('status'),
                'sdk_attempts_total':ledger.budget.requests}),flush=True)
            if ledger.budget.stop_reason():
                break
        return 0 if report['all_selected_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

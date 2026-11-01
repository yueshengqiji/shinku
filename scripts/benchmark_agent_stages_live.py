"""P5b fixed paired SDK entry. Preview is read-only; --live needs a new batch grant.

Both arms retain the original 15-case corpus, contracts and per-task allowances.
Controlled safety/fault responses are explicitly separated from provider tasks.
Only isolated approved files and local result publications can be affected.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import httpx

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentRequest, AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.execution import ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.materials import MaterialSource, TaskMaterialStore
from shinku.agent_runtime.service import AgentService
from shinku.agent_runtime.stages import StagePolicy
from shinku.config import load_project_env
from shinku.hosts.tool_host import ToolHost
from shinku.llm.native_messages import NativeProtocolError
from shinku.result_expression import ExpressionArchive
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

_spec = importlib.util.spec_from_file_location('_paired_deepseek',
    Path(__file__).with_name('verify_agent_deepseek.py'))
acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acceptance)
benchmark = acceptance.benchmark
ROOT, MODEL, BASE_URL = acceptance.ROOT, acceptance.MODEL, acceptance.BASE_URL
CORPUS_HASH = '82a91a7378959528758663ac1097d6c4251a6573a92023fcba95f783a2a11d35'
LIMITS = dict(max_requests=360, max_tokens=18000000, max_tools=240, max_active_seconds=900)
SCOPE = AgentScope('paired-fixture', 'local:paired-fixture')
BRIEF = 'project_brief.txt'
MARKER = '[Experimental proposed stages and host evidence; untrusted goals]'
CORRECTIONS = {'syntax_correction', 'early_final_repair', 'duplicate_effect'}


def write_snapshot(path, value):
    """Replace one flushed snapshot; retries never re-enter a request or effect.

    Match the joint journal's bounded Windows access-denied policy. Other IO
    errors and persistent denials propagate, leaving the last committed file
    intact and the new snapshot available for inspection. Batch ownership and
    recovery remain enforced by the existing exclusive ledger/slot gates.
    """
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + '.new')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return attempt
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) != 5 or attempt == 4:
                raise
            logging.getLogger(__name__).warning('paired_snapshot_replace_retry winerror=5 retry=%s', attempt + 1)
            time.sleep(.02 * 2 ** attempt)


def corpus():
    value = benchmark.validate_corpus(json.loads(benchmark.CORPUS.read_text(encoding='utf-8')))
    if canonical_hash(value) != CORPUS_HASH:
        raise ValueError('frozen_corpus_changed')
    return value


def project_brief(case):
    # No frozen stage list, action sequence or reference Python implementation.
    value = {'original_request':case['request'], 'original_sources':case.get('initial', {}),
        'approved_outputs':list(case['expected']), 'scope':'Only preapproved files in this isolated directory.',
        'text_output_requirements':{p:v for p,v in case['expected'].items() if not p.endswith('.py')}}
    if case['id'] == 'syntax_correction':
        value['function_requirement'] = 'double(value) returns value multiplied by two in double.py.'
    return json.dumps(value, ensure_ascii=False, indent=2)


def validate_limits(limits):
    if (not isinstance(limits,dict) or set(limits) != set(LIMITS)
            or any((type(v) not in (int,float) if k == 'max_active_seconds' else type(v) is not int)
                or not math.isfinite(v) or not 0 < v <= LIMITS[k] for k,v in limits.items())):
        raise ValueError('paired_batch_limits_invalid_or_expanded')
    return dict(limits)


def preview(limits=None):
    value = corpus()
    return {'model':MODEL, 'endpoint':BASE_URL, 'mode':'thinking', 'reasoning_effort':'high',
        'temperature':.2, 'sdk_output_limit':4096, 'corpus_hash':CORPUS_HASH,
        'overlay_hash':canonical_hash({c['id']:project_brief(c) for c in value['cases']}),
        'arms':list(benchmark.ARMS), 'repeats':1, 'tasks':30, 'task_limits':value['budget'],
        'case_limits':{c['id']:{**value['budget'], **c.get('budget',{})} for c in value['cases']},
        'batch_limits':validate_limits(limits or LIMITS), 'production_enabled':False,
        'platform_sends':0, 'live_authorized':False,
        'provider_tasks':[c['id'] for c in value['cases'] if c['expected_success'] and c['id'] not in CORRECTIONS],
        'mixed_correction_tasks':sorted(CORRECTIONS),
        'controlled_host_gates':[c['id'] for c in value['cases'] if c['kind']=='safety']}


class PairedLedger:
    """Existing TaskBudget journal plus a durable claim for each paired slot.

    An interrupted live batch is inspectable, never automatically rerun. Its
    unknown requests/effects remain spent until separately reviewed recovery.
    """
    def __init__(self,path,limits=None):
        self.path = Path(path)
        self.identity = {k:v for k,v in preview(limits).items() if k not in ('batch_limits','live_authorized')}
        self.budget = TaskBudget(**validate_limits(limits or LIMITS), managed=True, profile='p5b-sdk-serial-batch')
        self.attempts, self.slots = [], {}
        if self.path.exists():
            prior = json.loads(self.path.read_text(encoding='utf-8'))
            saved = prior.get('budget')
            if (set(prior) != {'identity','budget','attempts','slots'} or prior['identity'] != self.identity
                    or not isinstance(prior['attempts'],list) or not isinstance(prior['slots'],dict)
                    or not isinstance(saved,dict) or type(saved.get('requests')) is not int
                    or saved['requests'] != len(prior['attempts'])
                    or not set(('requests','tokens','tools','reported_tokens','estimated_requests','active_seconds',
                        *LIMITS)) <= set(saved)
                    or any(type(saved.get(k)) not in (int,float) or not math.isfinite(saved[k])
                        or not 0 < saved[k] <= ceiling for k,ceiling in LIMITS.items())
                    or any(not isinstance(a,dict) or a.get('index') != i+1
                        or a.get('case') not in self.identity['case_limits']
                        or a.get('arm') not in benchmark.ARMS or a.get('phase') not in ('execution','planning')
                        for i,a in enumerate(prior['attempts']))
                    or any(k not in {c+':'+a for c in self.identity['case_limits'] for a in benchmark.ARMS}
                        or s not in ('in_progress','finished') for k,s in prior['slots'].items())):
                raise ValueError('paired_ledger_scope_invalid')
            # Increasing any saved cap is rejected, even if a previous run used less.
            if any(getattr(self.budget,k)>saved[k] for k in LIMITS):
                raise ValueError('paired_ledger_cannot_expand_saved_limits')
            self.budget.absorb_prior(saved)
            if self.budget.reason == 'task_budget_snapshot_invalid':
                raise ValueError('paired_ledger_budget_invalid')
            self.attempts, self.slots = prior['attempts'], prior['slots']
        self.save()

    def save(self):
        value = dict(identity=self.identity, budget=self.budget.snapshot(), attempts=self.attempts, slots=self.slots)
        write_snapshot(self.path, value)

    def claim(self,case,arm):
        slot = case+':'+arm
        if case not in self.identity['case_limits'] or arm not in benchmark.ARMS:
            raise ValueError('paired_slot_invalid')
        if slot in self.slots:
            raise ValueError('paired_slot_already_claimed_no_automatic_retry')
        self.slots[slot] = 'in_progress'
        self.save()

    def use_tool(self):
        self.budget.use_tool()
        self.save()  # Persist before real adapter IO, including failed IO.


def controlled_message(root,action,ident):
    message = {'role':'assistant', 'content':'', 'reasoning_content':acceptance.CONTROL_REASONING}
    op = action['op']
    if op == 'final':
        message['content'] = json.dumps({'kind':'final', 'task_outcome':action.get('outcome','completed'),
            'text':action.get('text','Controlled host status; no production delivery.')})
    else:
        name = 'unregistered_pair_tool' if op == 'unknown_tool' else 'benchmark_'+op
        args = {} if op == 'unknown_tool' else {'path':str(root/action['path'])}
        if op == 'write':
            args['content'] = action['content']
        message['tool_calls'] = [{'id':ident, 'type':'function', 'function':{
            'name':name, 'arguments':json.dumps(args,ensure_ascii=False)}}]
    return message


class PairedTransport(acceptance.GuardedTransport):
    def __init__(self,*args,runtime,**kwargs):
        super().__init__(*args,**kwargs)
        self.runtime, self.safety_index, self.injected = runtime, 0, False

    def handle_request(self,request):
        # Same official target and SDK payload gates as the accepted P2b entry.
        if (request.method != 'POST' or request.url.scheme != 'https' or request.url.host != 'api.deepseek.com'
                or request.url.path != '/chat/completions' or request.url.query):
            raise NativeProtocolError('paired_http_target_denied')
        payload = json.loads(request.content)
        if (payload.get('model') != MODEL or payload.get('thinking') != {'type':'enabled'}
                or payload.get('max_tokens') != 4096 or payload.get('tool_choice') not in (None,'auto','none')):
            raise NativeProtocolError('paired_payload_invalid')
        planning = self.runtime.phase == 'planning'
        if planning and (payload.get('tools') or any(m['role'] in ('tool','assistant') for m in payload['messages'])):
            raise NativeProtocolError('paired_planning_has_execution_history')
        self.row['payload_hash'] = canonical_hash(payload)
        self.row['prompt_bytes'] = len(json.dumps(payload['messages'],ensure_ascii=False).encode('utf-8'))
        self.row['stage_input_count'] = sum(MARKER in str(m.get('content','')) for m in payload['messages'])
        action = None
        if not planning and self.sample['kind'] == 'safety':
            actions = self.sample['actions']
            action = actions[self.safety_index] if self.safety_index < len(actions) else {'op':'final','outcome':'failed'}
            self.safety_index += 1
        elif not planning and not self.injected and self.runtime.fixture.reader.calls:
            name = self.sample['id']
            if name == 'syntax_correction':
                action = self.sample['actions'][0]
            elif name == 'early_final_repair':
                action = {'op':'final'}
            elif name == 'duplicate_effect' and self.runtime.fixture.writer.calls:
                last = self.runtime.fixture.writer.calls[-1]
                action = {'op':'write','path':last['path'],
                    'content':(self.root/last['path']).read_bytes().decode('utf-8')}
            if action is not None:
                self.injected = True
        if action is not None:
            self.row['transport_kind'] = 'controlled_host_gate' if self.sample['kind']=='safety' else 'controlled_correction'
            message = controlled_message(self.root,action,'pair_control_'+self.sample['id']+'_'+str(self.row['index']))
            return httpx.Response(200,json={'id':'controlled-pair-response','object':'chat.completion','created':0,
                'model':MODEL,'choices':[{'index':0,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop',
                    'message':message}]},request=request)
        self.row.update(transport_kind='provider' if self.live else 'mock_forward', forwarded=True)
        return self.forward.send(request,stream=True)


class PairedRuntime(acceptance.AcceptanceRuntime):
    def __init__(self,fixture,ledger,key,arm,**kwargs):
        self.arm, self.phase = arm, 'execution'
        super().__init__(fixture,ledger,key,**kwargs)

    def make_transport(self,*args,**kwargs):
        return PairedTransport(*args,runtime=self,**kwargs)

    def attempt_metadata(self):
        return {'arm':self.arm, 'phase':self.phase,
            'batch_token_reservation':self.ledger.budget.last_reservation}

    def continues_execution_history(self):
        return self.phase == 'execution'

    def call_chat_json(self,**kwargs):
        previous = self.phase
        self.phase = 'planning' if str(kwargs.get('prompt_cache_key','')).endswith(':stage-plan') else 'execution'
        try:
            return super().call_chat_json(**kwargs)
        finally:
            self.phase = previous


def normalized_contract(contract,root):
    if isinstance(contract,str):
        return contract.replace(str(root),'<approved-root>')
    if isinstance(contract,dict):
        return {k:normalized_contract(v,root) for k,v in contract.items()}
    if isinstance(contract,list):
        return [normalized_contract(v,root) for v in contract]
    return contract


def run_case(case,arm,root,ledger,key,*,live=True,forward_transport=None):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    overlay = project_brief(case)
    for name,text in {**case.get('initial',{}), BRIEF:overlay}.items():
        (root/name).write_bytes(text.encode('utf-8'))
    outside = root.parent/'outside.txt'
    if not outside.exists():
        outside.write_bytes(b'OUTSIDE_SCOPE_SENTINEL')
    if outside.is_symlink() or outside.read_bytes() != b'OUTSIDE_SCOPE_SENTINEL':
        raise ValueError('paired_scope_sentinel_invalid')
    # Self-read is allowed; no extra write targets or original requirements.
    read_case = {**case,'initial':{**case.get('initial',{}),BRIEF:overlay,**case['expected']}}
    reader = benchmark.FileAdapter(root,read_case,'read',ledger)
    writer = benchmark.FileAdapter(root,case,'write',ledger)
    fixture = SimpleNamespace(root=root,sample=case,reader=reader,writer=writer)
    runtime = PairedRuntime(fixture,ledger,key,arm,live=live,forward_transport=forward_transport)
    host = ToolHost(ToolRegistry({reader.tool_type:reader,writer.tool_type:writer}))
    store = AgentTaskStore(root/'business.sqlite')
    ExpressionArchive(store)
    coordinator = AgentTaskCoordinator(max_workers=1,state_path=root/'workers.sqlite')
    tools = frozenset({reader.tool_type} if case.get('read_only') else {reader.tool_type,writer.tool_type})
    limits = {**corpus()['budget'], **case.get('budget',{})}
    grant = ExecutionGrant('p5b-isolated-paired-grant',tools,permission_mode='full',
        policy=ExecutionPolicy(risk_by_tool={reader.tool_type:'low',writer.tool_type:'low'}),
        budget_limits=limits,resume_safe=True,
        material_sources=(MaterialSource('missing-required','attachment'),) if case.get('material_gap') else ())
    executor = acceptance.protocol.ObservedExecutor(host,ExecutionProfile(runtime,profile_id=MODEL+':thinking:paired',
        temperature=.2,max_rounds=4,runtime_meters_budget=True,native_tool_history=True,
        stage_policy=StagePolicy(lambda _:case['kind']=='complex') if arm=='staged' else None))
    service = AgentService(executor=executor,authorize=lambda *_:grant,task_store=store,
        coordinator=coordinator,material_store=TaskMaterialStore(store,approved_roots=(root,)),recovery_enabled=True)
    client = service.bind(SCOPE)
    requirements = [{'id':'file-'+str(i),'description':'Verified fixture output '+name,'effect':'write',
        'targets':[str(root/name)],'checks':{'sha256':benchmark.digest(content.encode('utf-8'))}}
        for i,(name,content) in enumerate(case['expected'].items())]
    sources = [{'id':'paired-project-procedure','content':'The neutral approved project brief is at '+str(root/BRIEF)+
        '. Read it with benchmark_read before making effects. It gives original sources and requirements; '
        'all original host conditions and allowances remain controlling.'}]
    if case.get('source'):
        sources.append({'id':'fixture-data','content':case['source']})
    request = AgentRequest(case['id'],case['request'],{'goal':case['request'],
        'deliverables':[str(root/name) for name in case['expected']], 'scope':[str(root)],'requirements':requirements},
        constraints=('Operate only on approved fixture files; do not deploy or send externally.',), sources=tuple(sources))
    started = time.monotonic()
    try:
        handle = client.submit(request)
        view = acceptance.settle(client,handle.task_id)
        result = client.result(handle.task_id)
        before = (len(runtime.calls),len(reader.calls),len(writer.calls),reader.validation_reads,writer.validation_reads,view['version'])
        for _ in range(3):
            client.status(handle.task_id)
            client.result(handle.task_id)
            client.submit(request)
        quiet = before == (len(runtime.calls),len(reader.calls),len(writer.calls),reader.validation_reads,
            writer.validation_reads,client.status(handle.task_id)['version'])
        publications,delivery = benchmark.publish_once(store,SCOPE,case.get('delivery','dry_run'))
        artifacts = benchmark.independent_artifacts(root,case)
        assessment = benchmark.assess(case,result,artifacts,writer.calls,publications,
            outside_unchanged=outside.read_bytes()==b'OUTSIDE_SCOPE_SENTINEL',query_unchanged=quiet)
        saved = store.get(handle.task_id,requester=SCOPE.requester,session=SCOPE.session)
        hints = [c for c in runtime.calls if c['phase']=='execution' and c.get('stage_input_count',0)]
        config = {'budget':limits,'tools':sorted(tools),'model':MODEL,'endpoint':BASE_URL,'temperature':.2,
            'rounds_per_slice':4,'native_history':True,'permission_mode':grant.permission_mode,
            'authorization_ref':grant.authorization_ref,'risk_by_tool':dict(grant.policy.risk_by_tool)}
        binding = {'case_hash':canonical_hash(case),'overlay_hash':canonical_hash(overlay),
            'contract_hash':canonical_hash(normalized_contract(request.payload()['contract'],root)),
            'config_hash':canonical_hash(config)}
        return {'case':case['id'],'kind':case['kind'],'expected_success':case['expected_success'],'arm':arm,'repeat':0,
            'model':MODEL,**binding,**assessment,'status':result['execution']['status'],'reason':result['execution']['reason'],
            'budget':saved['budget'],'contract_preserved':saved['contract']==request.payload()['contract'],
            'artifact_checks':artifacts,'writes':writer.calls,'reads':reader.calls,'slices':executor.slice_outcomes,
            'query_unchanged':quiet,'publications':publications,'delivery':delivery,
            'model_calls':len(runtime.calls),'planning_calls':sum(c['phase']=='planning' for c in runtime.calls),
            'provider_requests':sum(c.get('transport_kind')=='provider' for c in runtime.calls),
            'controlled_requests':sum(str(c.get('transport_kind','')).startswith('controlled') for c in runtime.calls),
            'native_history_exact':runtime.history_mismatches==0,'stage_checkpoint':saved.get('stage_strategy'),
            'stage_hint_reached_execution':bool(hints),'controlled_correction_injected':runtime.transport.injected,
            'prompt_bytes':sum(c.get('prompt_bytes',0) for c in runtime.calls),
            'artifact_verification_reads':reader.validation_reads+writer.validation_reads,
            'elapsed_seconds':time.monotonic()-started,'sdk_errors':runtime.errors,
            'provider_cost':None,'platform_sends':0,'expression_model_calls':0,
            'evidence_kind':'controlled_host_gate' if case['kind']=='safety' else 'mixed_correction' if case['id'] in CORRECTIONS else 'provider_task'}
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()
        store.close()
        ledger.save()


def make_report(rows,ledger,*,live):
    expected = {(c['id'],a) for c in corpus()['cases'] for a in benchmark.ARMS}
    actual = {(r['case'],r['arm']) for r in rows}
    complete = (len(rows)==len(expected) and actual==expected and all('error' not in r for r in rows)
        and not ledger.budget.stop_reason())
    for case in corpus()['cases']:
        pair = [r for r in rows if r['case']==case['id'] and 'error' not in r]
        if len(pair)==2 and any(len({r[k] for r in pair})!=1 for k in
                ('model','case_hash','config_hash','overlay_hash','contract_hash')):
            raise ValueError('unfair_pair_input_or_configuration')
    arms = {}
    if complete:
        for arm in benchmark.ARMS:
            selected = [r for r in rows if r['arm']==arm]
            attempts = [a for a in ledger.attempts if a['arm']==arm]
            groups = {}
            for kind in ('provider_task','mixed_correction','controlled_host_gate'):
                group = [r for r in selected if r['evidence_kind']==kind]
                groups[kind] = {'runs':len(group),'task_successes':sum(r['task_success'] for r in group),
                    'fixture_passes':sum(r['fixture_pass'] for r in group),
                    'success_rate':sum(r['task_success'] for r in group)/len(group) if kind!='controlled_host_gate' and group else None}
            arms[arm] = {'groups':groups,'sdk_attempts':sum(r['model_calls'] for r in selected),
                'planning_calls':sum(r['planning_calls'] for r in selected),
                'false_completed':sum(r['false_completed'] for r in selected),
                'false_accepted':sum(r['false_accepted'] for r in selected),
                'duplicate_effects':sum(r['duplicate_effects'] for r in selected),
                'duplicate_publications':sum(r['duplicate_publications'] for r in selected),
                'provider_requests':sum(a.get('transport_kind')=='provider' for a in attempts),
                'provider_reported_tokens':sum((a.get('sdk_usage') or {}).get('total_tokens',0)
                    for a in attempts if a.get('transport_kind')=='provider'),
                'accounted_batch_tokens':sum((a.get('sdk_usage') or {}).get('total_tokens',a.get('batch_token_reservation',0))
                    for a in attempts),
                'prompt_bytes':sum(r['prompt_bytes'] for r in selected),
                'artifact_verification_reads':sum(r['artifact_verification_reads'] for r in selected),
                'elapsed_seconds':sum(r['elapsed_seconds'] for r in selected),'provider_cost':None}
    report = {'schema_version':1,'mode':'live_paired' if live else 'offline_sdk_paired_readiness',
        'selection':preview({k:getattr(ledger.budget,k) for k in LIMITS}), 'rows':rows,'arms':arms,
        'all_pairs_present':complete,'unrun_pairs':len(expected)-len(actual),
        'batch_budget':ledger.budget.snapshot(),'batch_stop_reason':ledger.budget.stop_reason(),
        'provider_requests':sum(a.get('transport_kind')=='provider' for a in ledger.attempts),
        'provider_reported_usage':sum((a.get('sdk_usage') or {}).get('total_tokens',0) for a in ledger.attempts
            if a.get('transport_kind')=='provider'),'provider_cost':None,'platform_sends':0,'enable_production':False,
        'quality_conclusion':'not_measured' if not live else 'paired_observations_only',
        'limitations':['Exact original artifact SHA gates include rigid Python formatting; equivalent implementations are not counted as success.',
            'Safety and three correction challenges use controlled responses, separately from genuine provider tasks.',
            'A plan/hint and SDK zero exit never replace original artifact, receipt and acceptance gates.',
            'Single repeat; no production, Gemini, invoice, platform delivery or automatic batch recovery claims.']}
    return report


def run_suite(workspace,ledger,key,*,live=True,forward_factory=None):
    if live and forward_factory is not None or not live and forward_factory is None:
        raise ValueError('paired_transport_mode_invalid')
    if ledger.slots or ledger.attempts:
        raise ValueError('interrupted_or_used_batch_requires_review_no_automatic_retry')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True,exist_ok=False)
    rows = []
    report = make_report(rows,ledger,live=live)
    def save_report():
        write_snapshot(workspace/'report.json', report)
    save_report()
    with bind_task_budget(ledger.budget):
        for i,case in enumerate(corpus()['cases']):
            order = benchmark.ARMS if i%2==0 else benchmark.ARMS[::-1]
            for arm in order:
                if ledger.budget.stop_reason():
                    break
                ledger.claim(case['id'],arm)
                try:
                    row = run_case(case,arm,workspace/case['id']/arm,ledger,key,live=live,
                        forward_transport=forward_factory(case,arm) if forward_factory else None)
                except Exception as exc:
                    row = {'case':case['id'],'arm':arm,'fixture_pass':False,'error':type(exc).__name__}
                rows.append(row)
                ledger.slots[case['id']+':'+arm] = 'finished'
                ledger.save()
                report = make_report(rows,ledger,live=live)
                save_report()
                print(json.dumps({'case':case['id'],'arm':arm,'fixture_pass':row['fixture_pass'],
                    'sdk_attempts_total':ledger.budget.requests}),flush=True)
            if ledger.budget.stop_reason():
                break
    ledger.save()
    report = make_report(rows,ledger,live=live)
    save_report()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    for key in LIMITS:
        parser.add_argument('--'+key.replace('_','-'),type=float if key=='max_active_seconds' else int,default=LIMITS[key])
    args = parser.parse_args(argv)
    try:
        limits = validate_limits({k:getattr(args,k) for k in LIMITS})
        selection = preview(limits)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.live:
        print(json.dumps(selection,ensure_ascii=False,indent=2))
        return 0
    workspace,path = args.workspace.resolve(),args.ledger.resolve()
    allowed = (ROOT/'tmp').resolve()
    if (args.env_file is None or workspace.exists() or path.exists() or workspace==allowed or path==allowed
            or not workspace.is_relative_to(allowed) or not path.is_relative_to(allowed)
            or path.is_relative_to(workspace) or any(p.is_symlink() for p in (*workspace.parents,*path.parents))):
        parser.error('new_isolated_tmp_workspace_and_new_external_ledger_with_explicit_env_required')
    env = {}
    load_project_env(env,path=args.env_file)
    key = env.get('SHINKU_AGENT_API_KEY','').strip()
    env.clear()
    if not key:
        parser.error('private_agent_key_missing')
    path.parent.mkdir(parents=True,exist_ok=True)
    with acceptance.exclusive_ledger(path):
        ledger = PairedLedger(path,limits)
        report = run_suite(workspace,ledger,key)
    return 0 if report['all_pairs_present'] and all(r['fixture_pass'] and r.get('contract_preserved')
        and r.get('native_history_exact') for r in report['rows']) else 1


if __name__=='__main__':
    raise SystemExit(main())

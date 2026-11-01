"""Saved-model repair checks with fictional sources, no QQ transport/config writes."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import time

import gray_context_projection as replay
import probe_context_projection as probes
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.projection.semantics import ProjectionModel

a=replay.a


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--case-id',action='append')
    parser.add_argument('--skip-probes',action='store_true')
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    profile=a.saved_profile(root)
    runtime,config,_,_,persona,frozen=profile
    runtime.runtime_policy=replace(runtime.runtime_policy,http_attempts=2)
    fixture=json.loads(replay.FIXTURE.read_text(encoding='utf-8'))
    by_id={case['id']:case for case in fixture['cases']}
    extra={'id':'holdout_condition','split':'holdout','query':'她后来参加了吗？','history':[
        {'id':'plan','text':'小玲说只有弟弟能照看猫，她才去参加读书会；她还没有决定。','age_days':0}],
        'required':['plan'],'constraints':['小玲','弟弟照看猫才去','尚未确认是否参加'],'forbidden':['已经参加']}
    by_id[extra['id']]=extra
    schedule=['one_person','two_people','holdout_condition','one_person','two_people','holdout_condition',
              'one_person','old_condition','quote_and_promise','paraphrase','old_event']
    if args.case_id:
        if set(args.case_id)-set(by_id):parser.error('unknown case id')
        schedule=[name for name in schedule if name in args.case_id]
    (root/'fixtures.json').write_text(json.dumps({'original':fixture,'extra':extra,'schedule':schedule},ensure_ascii=False,indent=2),encoding='utf-8')
    source_paths=list((a.ROOT/'src/shinku/memory/projection').glob('*.py'))+[
        a.ROOT/'src/shinku/qq/prompting.py',a.ROOT/'src/shinku/llm/runtime_core.py',a.ROOT/'src/shinku/llm/runtime_transport.py',
        Path(replay.__file__),Path(__file__)]
    report={'fictional_only':True,'sender':None,'profile':{k:frozen[k] for k in ['chat_model','aux_model','persona_sha256','temperature']},
        'http_attempts':2,'semantic_timeout_ms':ProjectionPolicy().semantic_timeout_ms,
        'source_hashes':{str(p.relative_to(a.ROOT)):a.sha(p) for p in source_paths},'rows':[],'probes':[]}
    def save():
        (root/'repairs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for index,case_id in enumerate(schedule):
        path=root/f'{index}-{case_id}';path.mkdir()
        original=runtime.chat.client.chat.completions.create
        injected=[]
        if case_id=='old_event':
            def once(**kwargs):
                if not injected:
                    injected.append('synthetic TLS handshake timeout before first request')
                    raise TimeoutError('TLS handshake timeout (isolated injected fault)')
                return original(**kwargs)
            runtime.chat.client.chat.completions.create=once
        try:
            row=replay.run_case(path,by_id[case_id],True,profile,fixture['now'],10000)
        finally:runtime.chat.client.chat.completions.create=original
        row.pop('raw_result',None);row['injected_fault']=injected;row['repeat_index']=index
        report['rows'].append(row);save()
        print(json.dumps({'case':case_id,'reply':row['reply'],'semantic':row['projection_diagnostics'].get('semantic'),
            'seconds':row['total_seconds'],'error':row['error_type']},ensure_ascii=False),flush=True)
    if args.skip_probes:
        report['protected_configs_unchanged']=all(a.sha(Path(p))==h for p,h in frozen['protected_config_hashes'].items())
        save();return 0
    judges=[('architect','小红说她是建筑师，周六想去看展，还没有确认。','小雪说她是医生，想去公园。',
             '做建筑设计的她，周六想去哪里？','小红'),
            ('pharmacist','小青说她负责配药，只有同事愿意换班才去音乐会；现在还没定。','小玲说她做陶器，已经答应去书店。',
             '配药的她后来决定去音乐会了吗？','小青')]
    for label,left,right,query,target in judges:
        store=a.MemoryStore(root/(label+'.sqlite3'))
        probes.store_row(store,'a',left,probes.NOW-10);probes.store_row(store,'b',right,probes.NOW-9)
        observed=replay.Observed(runtime)
        projector=ProjectionEngine(store,ProjectionPolicy(enabled=True,semantic_enabled=True),ProjectionModel(observed))
        request=probes.request(query)
        view=projector.project(request)
        entity=next((e for e in view.entities if e['candidate_id']==view.topic.get('last_subject')),None)
        answer=probes.reply(runtime,persona,query,view.rendered,
            ['自然回答并保留原条件、人物名字和未确认状态。没有后续证据不能说后续从未发生。'])
        report['probes'].append({'probe':label,'query':query,'history':[left,right],'expected_person':target,
            'selected_person':entity['name'] if entity else None,'judgment':view.topic.get('judge'),
            'diagnostics':view.diagnostics,'aux_calls':observed.calls,'reply':answer})
        projector.stop();save()
        print(json.dumps({'probe':label,'selected':entity['name'] if entity else None,'semantic':view.diagnostics['semantic'],
            'reply':answer['reply']},ensure_ascii=False),flush=True)
    # Exercise the revised alias protocol on a checkpoint as well.
    store=a.MemoryStore(root/'checkpoint.sqlite3')
    for i in range(8):
        probes.store_row(store,str(i),f'荧石项目第{i+1}条约束：'+'以下是虚构的讨论背景，并不表示执行完成。'*72+
            f'最终条件：仅在离线演练中使用参数{i+1}.25；没有批准发布。',probes.NOW-20+i)
    observed=replay.Observed(runtime)
    projector=ProjectionEngine(store,ProjectionPolicy(enabled=True,semantic_enabled=True),ProjectionModel(observed))
    request=probes.request('荧石项目这八条约束各自的参数是多少？是否批准发布？')
    view=projector.project(request)
    answer=probes.reply(runtime,persona,request.query,view.rendered)
    report['probes'].append({'probe':'checkpoint','diagnostics':view.diagnostics,
        'source_quotes':[r['text'] for r in view.records if r['text']], 'aux_calls':observed.calls,'reply':answer})
    projector.stop()
    report['protected_configs_unchanged']=all(a.sha(Path(p))==h for p,h in frozen['protected_config_hashes'].items())
    save()
    print(json.dumps({'probe':'checkpoint','semantic':view.diagnostics['semantic'],'reply':answer['reply']},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

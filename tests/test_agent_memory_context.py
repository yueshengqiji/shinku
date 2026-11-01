"""任务记忆/原文聚焦/对象冲突；不把模型完成声明当事实。"""
import copy
import json

import pytest

from shinku.agent.task_state import TaskContractError
from shinku.agent.work_state import validate_plan
from shinku.agent_runtime.task_memory import task_memory
from shinku.agent_runtime.context import build_execution_prompt
from shinku.agent_memory.retrieval import AgentMemoryPolicy
from shinku.angelos.task_continuity import turn_workspace
from agent_memory_fixture import memory_fixture, seed
from work_state_fixture import plan


def install_receipt():
    return {'id':'install-1','name':'synthetic_install','status':'ok','executed':True,
        'data':{'state_updates':{'verified_effects':['command_run'],
            'execution_session':{'status':'completed','returncode':0,'session_id':'install-session'}}}}


def test_failed_task_keeps_successful_effects_separate_from_goal(tmp_path):
    with memory_fixture(tmp_path) as f:
        task,_,_=seed(f,'old',status='failed',calls=[install_receipt()],report='我已修好')
        before=copy.deepcopy(task);view=task_memory(task)
        assert view['status']=='failed' and view['verification']['missing']==['old_command']
        assert view['recorded_effects'][0]['effects']==['command_run']
        assert view['report_is_execution_proof'] is False
        assert view['report_delivery']=='canonical_report_present' and task==before


def test_followup_gets_original_effect_even_when_previous_report_empty(tmp_path):
    with memory_fixture(tmp_path) as f:
        old,_,_=seed(f,'old',text='修复合成客户端',status='failed',calls=[install_receipt()])
        current,_,ctx=seed(f,'now',text='刚才怎么了')
        service,host,_,_,file_tool,runtime=f
        snap=host.memory.snapshot(ctx)
        previous=snap['continuity']['candidates'][0]
        assert previous['task_id']=='old' and previous['recorded_effects']
        assert previous['report_delivery']=='no_canonical_report'
        assert snap['continuity']['automatic_resume'] is False
        assert file_tool.calls==0 and runtime.calls==[]
        assert service.task_store.get('old',requester='owner',session=old['session'])==old


def test_snapshot_is_added_to_real_prompt_without_copying_arguments_or_reasoning(tmp_path):
    with memory_fixture(tmp_path) as f:
        receipt=install_receipt();receipt['arguments']={'secret':'DO_NOT_EXPORT'}
        receipt['data']['reasoning_content']='DO_NOT_EXPORT'
        seed(f,'old',status='failed',calls=[receipt]);_,_,ctx=seed(f,'now')
        from shinku.agent.loop import AgentState
        state=AgentState(context=ctx)
        prompt=build_execution_prompt(state)
        assert 'Agent-only task memory' in prompt.native_context_update
        assert 'install-session' in prompt.native_context_update
        assert 'DO_NOT_EXPORT' not in prompt.native_context_update


def test_unrelated_history_is_projected_but_originals_and_navigation_survive(tmp_path):
    with memory_fixture(tmp_path,policy=AgentMemoryPolicy(max_history_turns=2)) as f:
        raw=json.dumps({'messages':[{'user':f'old-{i}','assistant':'text'} for i in range(10)],'recent_tasks':[]})
        before=raw;value=json.loads(f[1].memory.continuity.focused_history(raw,f[1].memory.store.policy))
        assert len(value['messages'])==2 and value['history_projection']['omitted_turns']==8
        assert value['history_projection']['retrieval']['mode']=='inputs' and raw==before
        assert f[1].memory.continuity.focused_history('legacy text',f[1].memory.store.policy)=='legacy text'


def test_changed_workspace_or_task_expiry_is_not_silently_recalled(tmp_path):
    with memory_fixture(tmp_path) as f:
        old,_,_=seed(f,'old',status='failed');_,_,ctx=seed(f,'now')
        service,host,wb,c,_,_=f
        with service.task_store._transaction() as db:
            db.execute('UPDATE agent_conversation_tasks SET updated=? WHERE task_id=?',
                (service.task_store.clock()-service.task_store.ttl_seconds-1,'old'))
        value=host.memory.snapshot(ctx)
        assert value['continuity']['gaps'][0]['reason']=='agent_memory_source_unavailable'
        with service.task_store._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM agent_conversation_tasks WHERE task_id=?',('old',)).fetchone()[0]==1
        wb.workspace({'name':'日常工作','path':'','description':'changed'},'default')
        with pytest.raises(ValueError,match='workspace_changed'):host.memory.snapshot(ctx)


def test_object_candidates_persist_as_unverified_plan_not_fact(tmp_path):
    with memory_fixture(tmp_path) as f:
        task,_,_=seed(f,'task');value=plan()
        value['objects']=[{'name':'dove','candidates':['本地客户端2.1','同名配置包2026'],
                           'conflicts':['入口和用途不同，不能合并']}]
        assert validate_plan(value,task)['objects']==value['objects']
        value['objects'][0]['confirmed']=True
        with pytest.raises(TaskContractError):validate_plan(value,task)


def test_same_name_does_not_merge_versions_or_sources(tmp_path):
    with memory_fixture(tmp_path) as f:
        store=f[1].memory.store
        for key,text in [('client','dove 2.1 本地代理客户端'),('config','dove 2026 邮件配置')]:
            store.put(principal='owner',workspace='w',key=key,kind='user_statement',text=text,source={})
        rows=store.search('owner','w','dove')
        assert len(rows)==2 and len({r['id'] for r in rows})==2


def test_observe_versions_do_not_duplicate_protected_memory_controls(tmp_path):
    from shinku.agent.loop import AgentState
    with memory_fixture(tmp_path) as f:
        task,_,ctx=seed(f,'now');before=build_execution_prompt(AgentState(context=ctx))
        latest=f[0].task_store.get('now',requester='owner',session=task['session'])
        f[0].task_store.save({**latest,'tool_calls':[install_receipt()]})
        after=build_execution_prompt(AgentState(context=ctx))
        assert before.native_context_update==after.native_context_update
        assert 'install-session' in after.user_prompt and 'install-session' not in before.user_prompt


def test_explicit_task_reference_is_read_without_automatic_resume(tmp_path):
    with memory_fixture(tmp_path) as f:
        seed(f,'agent:older',text='较早合成问题',status='failed')
        seed(f,'agent:newer',text='另一个问题',status='failed')
        _,_,ctx=seed(f,'now',text='查看 agent:older 的失败原因')
        value=f[1].memory.snapshot(ctx)['continuity']
        assert value['selection']=='explicit_same_conversation_ids'
        assert [t['task_id'] for t in value['candidates']]==['agent:older'] and not value['automatic_resume']


def test_ninety_percent_summary_retains_agent_memory_controls_and_originals(tmp_path,monkeypatch):
    from shinku.llm.context_compaction import ContextCompactor
    from test_context_compaction import Summarizer,bundle,window
    from shinku.agent.loop import AgentState
    with memory_fixture(tmp_path) as f:
        seed(f,'old',status='failed',calls=[install_receipt()]);_,_,ctx=seed(f,'now')
        parts=build_execution_prompt(AgentState(context=ctx))
        control={'role':'user','content':parts.native_context_update}
        hard={'role':'user','content':parts.native_hard_context}
        payload={'model':'synthetic','messages':[{'role':'system','content':'权限不变'},
            {'role':'user','content':'过往无关背景'*8000},{'role':'user','content':'当前要求'},hard,control]}
        original=copy.deepcopy(payload);window(monkeypatch,10000)
        manager=ContextCompactor(Summarizer(),tmp_path/'compact.sqlite3')
        result=manager.project(payload,bundle(),2,'agent-memory-test')
        assert hard in result['messages'] and control in result['messages'] and payload==original
        assert 'install-session' in json.dumps(result,ensure_ascii=False)


def test_older_long_answers_are_excerpts_recent_followups_remain_complete(tmp_path):
    with memory_fixture(tmp_path,policy=AgentMemoryPolicy(max_history_turns=6)) as f:
        messages=[{'turn_id':f'turn-{i}','user':f'原请求-{i}',
                   'assistant':f'答复-{i}'+('旧推断或详细资料'*500)} for i in range(8)]
        raw=json.dumps({'messages':messages,'recent_tasks':[{'task_id':'prior'}]},ensure_ascii=False)
        value=json.loads(f[1].memory.continuity.focused_history(raw,f[1].memory.store.policy))
        assert value['messages'][-3:]==messages[-3:]
        assert [r['user'] for r in value['messages']]==[r['user'] for r in messages[-6:]]
        for row in value['messages'][:3]:
            assert len(row['assistant'])==320 and row['assistant_excerpt_only']
            original=next(x for x in messages if x['turn_id']==row['turn_id'])
            assert row['assistant_omitted_chars']==len(original['assistant'])-320
        assert value['history_projection']['omitted_turns']==2
        assert value['history_projection']['shortened_assistant_turns']==3
        assert value['history_projection']['full_history_retained']
        assert json.loads(raw)['messages']==messages
        assert value['recent_tasks']==[{'task_id':'prior'}]

"""成稿只使用当前范围，原历史/附件不删除；真实AgentService与材料门验证投影。"""
from copy import deepcopy
import json

import pytest

from shinku.agent_runtime.answer_guidance import local_draft_context
from shinku.angelos.conversation_context import conversation_source
from test_agent_answer_protocol import analysis, Runtime, ANSWER


def task_with_history():
    history = conversation_source([
        {'text':'以前的另一项工作需要处理网页表单、启动服务', 'reply':'旧助手断言了错误结论'},
        {'text':'input.txt 使用独立配置，请注意来源', 'reply':'旧助手推断性能收益'},
    ])
    return {'requests':[{'message_id':'now','text':'读取 input.txt，分析当前配置，不修改文件。'}],
            'confirmed_constraints':['保持原文和已有修改'], 'source_data':[history,
                {'id':'attachment','content':'完整附件\nATTACHMENT_TAIL'}]}


def contract():
    return {'requirements':[{'effect':'read','targets':['/fixture/input.txt']}]}


def test_draft_filters_unrelated_goals_without_altering_current_requests_or_originals():
    task=task_with_history();saved=deepcopy(task)
    result=local_draft_context(task,contract(),{'sources':[{'text':'当前配置的原文'}]})
    text=json.dumps(result,ensure_ascii=False)
    assert '处理网页表单' not in text and '启动服务' not in text and '旧助手' not in text
    assert 'input.txt 使用独立配置' in text and 'ATTACHMENT_TAIL' in text
    assert result['requests']==task['requests'] and result['confirmed_constraints']==task['confirmed_constraints']
    assert result['answer_context']['omitted_user_requests']==1 and task==saved
    assert result['context_projection']['full_originals_retained']


@pytest.mark.parametrize('current_request',['继续刚才的分析，读取 input.txt','按之前的要求分析 input.txt','Continue the previous analysis of input.txt'])
def test_explicit_history_reference_preserves_original_context(current_request):
    task=task_with_history();task['requests'][0]['text']=current_request
    result=local_draft_context(task,contract(),{})
    assert result['source_data']==task['source_data'] and result['context_projection']['history_referenced']


def test_history_from_different_path_is_not_selected_by_basename_substring():
    task=task_with_history();task['source_data']=[conversation_source([
        {'text':'另一个项目 /other/input.txt 的旧故障'}, {'text':'unrelated-input.txt 的旧目标'}])]
    result=local_draft_context(task,contract(),{})
    assert result['answer_context']['prior_user_requests']==[]


def test_real_service_materials_gate_and_draft_keep_original_sources(analysis,tmp_path):
    from shinku.agent_runtime import AgentExecutor,AgentService,ExecutionGrant,ExecutionProfile,AgentRequest,AgentScope
    from shinku.agent_runtime.materials import TaskMaterialStore
    from shinku.agent.task_state import AgentTaskStore
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    from shinku.tasks.coordinator import AgentTaskCoordinator
    from test_agent_runtime_boundary import settled
    task,_,tools,_,path,_=analysis
    runtime=Runtime({'kind':'local_answer','text':ANSWER,'task_outcome':'completed'})
    grant=ExecutionGrant('fixture-grant',frozenset(tools),permission_mode='read',policy=ExecutionPolicy(default_risk='low'))
    store=AgentTaskStore(tmp_path/'service.sqlite');materials=TaskMaterialStore(store)
    service=AgentService(executor=AgentExecutor(ToolHost(ToolRegistry(tools)),ExecutionProfile(runtime,max_rounds=8)),
        authorize=lambda *a:grant, task_store=store,coordinator=AgentTaskCoordinator(state_path=tmp_path/'workers.sqlite'),
        material_store=materials,recovery_enabled=False)
    sources=task_with_history()['source_data'];original=deepcopy(sources)
    try:
        port=service.bind(AgentScope('fixture','local:fixture'))
        handle=port.submit(AgentRequest('protocol-input',task['requests'][0]['text'],task['contract'],sources=tuple(sources)))
        view=settled(port,handle.task_id);result=port.result(handle.task_id)
        assert view['status']=='completed', (view.get('reason'),result.get('execution'),result.get('remaining'))
        assert result['verification']['accepted'] and result['execution_report']['text']==ANSWER
        assert 'verified_answer' not in result and result['verification']['layers']['answer_evidence']=='unverified'
        draft=json.loads(runtime.calls[1]['user_prompt'].split('\n[Host acceptance policy]')[0])
        assert '处理网页表单' not in json.dumps(draft,ensure_ascii=False)
        assert 'ATTACHMENT_TAIL' in json.dumps(draft,ensure_ascii=False)
        saved=store.get(handle.task_id,requester='fixture',session='local:fixture')
        assert saved['source_data']==[{**s,'reference':'','source':'caller_data'} for s in original]
        assert sources==original
        assert '处理网页表单' in runtime.calls[0]['user_prompt']
        assert len(runtime.calls[1]['user_prompt'].encode()) < len(runtime.calls[0]['user_prompt'].encode())
        assert saved['material_information']['required_status']=='complete'
        assert saved['progress']['answer_drafts'][0]['text']==ANSWER
        assert len(saved['tool_calls'])==1 and path.read_text()=='当前配置的数值为10，未给出单位；两份配置独立。'
    finally:
        service.coordinator.shutdown(wait=True);store.close()

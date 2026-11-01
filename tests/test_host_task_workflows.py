"""可选宿主任务入口的真实聊天schema准入；仅使用合成名称与无副作用runtime。"""
from types import SimpleNamespace

import pytest

from shinku.agent_runtime import AgentBoundaryError
from shinku.qq.interaction_tools import InteractionRuntime, task_tool_specs, validate_interaction_response
from shinku.tools.registry import ToolRegistry


def runtime():
    calls = []
    adapter = InteractionRuntime(SimpleNamespace(call_chat_json=lambda **kw:calls.append(kw) or {}),
        ToolRegistry(),{},delegated=True)
    return adapter,calls


def test_optional_host_workflow_reaches_guarded_sdk_and_original_schema_still_works():
    adapter,calls = runtime()
    catalogue = {'sample_reader':'读取当前宿主绑定的合成资料，不发布。'}
    adapter.configure_task_workflows(catalogue)
    catalogue['unreviewed']='不能通过后续修改扩展已绑定目录'
    for specs in [task_tool_specs(delegated=True),task_tool_specs(delegated=True,workflows=adapter.task_workflows)]:
        adapter.call_chat_json(native_tools=specs,system_prompt='合成人格')
    assert len(calls)==2 and 'unreviewed' not in adapter.task_workflows
    context={'_agent_delegation_host':True,'_host_task_workflows':dict(adapter.task_workflows)}
    decision=validate_interaction_response({'kind':'tool','tool_call':{'name':'agent_task_submit',
        'arguments':{'workflow':'sample_reader','goal':'建议目标'}}},{},context)
    assert decision.invocation.arguments['workflow']=='sample_reader'
    with pytest.raises(TypeError):adapter.task_workflows['other']='扩展'
    with pytest.raises(ValueError):adapter.configure_task_workflows({'other':'重复绑定'})


@pytest.mark.parametrize('change',['unregistered_workflow','schema_expansion','forged_target','unconfigured_context'])
def test_workflow_cannot_add_sources_permissions_or_unreviewed_schemas(change):
    adapter,calls=runtime();adapter.configure_task_workflows({'sample_reader':'合成只读入口'})
    specs=task_tool_specs(delegated=True,workflows=adapter.task_workflows)
    if change=='unregistered_workflow':specs[0]['function']['parameters']['properties']['workflow']['enum'].append('other')
    elif change=='schema_expansion':specs[0]['function']['parameters']['properties']['permission']={'type':'string'}
    elif change=='forged_target':
        context={'_agent_delegation_host':True,'_host_task_workflows':dict(adapter.task_workflows)}
        with pytest.raises(AgentBoundaryError):validate_interaction_response({'kind':'tool','tool_call':{
            'name':'agent_task_submit','arguments':{'workflow':'sample_reader','target':'/unbound'}}},{},context)
        return
    else:
        with pytest.raises(AgentBoundaryError):validate_interaction_response({'kind':'tool','tool_call':{
            'name':'agent_task_submit','arguments':{'workflow':'sample_reader'}}},{},{'_agent_delegation_host':True})
        return
    with pytest.raises(AgentBoundaryError):adapter.call_chat_json(native_tools=specs)
    assert not calls


@pytest.mark.parametrize('catalogue',[{'../bad':'说明'},{'sample_reader':''},{'sample_reader':'x'*601},{str(i):'说明' for i in range(9)}])
def test_host_catalogue_is_bounded_and_local_tasks_cannot_claim_a_delegated_workflow(catalogue):
    with pytest.raises(ValueError):task_tool_specs(delegated=True,workflows=catalogue)
    with pytest.raises(ValueError):task_tool_specs(workflows={'sample_reader':'合成入口'})

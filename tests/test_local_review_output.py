"""合成真实读取链验证审查截断不获证据资格，普通交付与严格验收保持分层。"""
from copy import deepcopy
import json

import pytest

from shinku.agent.loop import AgentLoop
from shinku.agent.planner import LLMPlanner
from shinku.agent_runtime.context import build_execution_prompt,execution_context
from shinku.agent_runtime.answer_guidance import ANALYSIS_EVIDENCE_GUIDANCE
from shinku.llm.response_protocol import note_incomplete_response
from shinku.llm.output_budget import current_output_budget
from test_agent_answer_protocol import analysis
from test_agent_answer_protocol import Runtime as BaseRuntime

ANSWER='两份配置独立，实例赋值作用于该实例；应先确认具体读写位置，再评价是否存在维护问题。'


class ReviewRuntime(BaseRuntime):
    def __init__(self,finish='',bad_quote=False,error=False):
        super().__init__({'kind':'final','text':ANSWER,'task_outcome':'completed'})
        self.finish,self.bad_quote,self.error=finish,bad_quote,error
        self.stage_limits=[]
    def call_chat_json(self,**kw):
        scope=current_output_budget();self.stage_limits.append(scope.max_tokens if scope else None)
        # 普通说明现在零质量模型调用；本文件在明确严格请求下验证审查协议。
        # 严格修正仍由执行器成稿，不能把下一次规划提示误当审查载荷。
        if '本地答案质量审查器' not in kw.get('system_prompt', ''):
            if len(self.calls)<2:return super().call_chat_json(**kw)
            self.calls.append(kw)
            return deepcopy(self.response)
        self.calls.append(kw)
        if self.error:raise RuntimeError('synthetic private failure detail')
        payload=json.loads(kw['user_prompt']);source=payload['read_sources'][0]
        if self.finish:note_incomplete_response(self.finish)
        return {'verdict':'supported','claims':[{'answer_fragment_id':next(iter(payload['answer_fragments'])),
            'evidence':[{'source_id':source['id'],'quote':'伪造来源内容'if self.bad_quote else'两份配置独立'}]}],
            'coverage':[{'field_id':k,'supported':True,'answer_fragment_ids':list(payload['answer_fragments'])}
                for k in payload['required_field_selectors']],'missing_fields':[]}


def run_case(analysis,runtime,strict=False):
    task,ctx,tools,gate,*_=analysis
    if strict:
        # 宿主请求明确要求严格核验，不能用审查不可用制造普通分析策略。
        task['requests'][0]['text']='逐条引用核验配置分析结论，不修改文件。'
        from types import SimpleNamespace
        ctx=execution_context(task,SimpleNamespace(tool_context={},permission_mode='read',authorization_ref='fixture-grant'),attempt=1)
    loop=AgentLoop(planner=LLMPlanner(runtime=runtime,system_prompt='执行器只能输出kind=final/tool',
        build_user_prompt=build_execution_prompt),handlers=tools,completion_gate=gate,sliced=True,max_rounds=4)
    return loop.run(task_id=task['task_id'],context=ctx)


@pytest.mark.parametrize('finish,diagnostic',[('length','output_length'),('content_filter','output_filtered')])
def test_parseable_supported_but_incomplete_review_is_not_verified(analysis,finish,diagnostic):
    runtime=ReviewRuntime(finish);result=run_case(analysis,runtime,strict=True)
    assert result.status=='failed' and not result.completion['accepted']
    assert result.progress['local_answer_summary']['binding_error']=='local_answer_review_'+diagnostic
    assert result.progress['local_answer_review_protocol']['finish_reason']==finish
    assert len(runtime.calls)==5 and ANSWER in result.final_text
    assert runtime.stage_limits==[None]*5  # 取证、成稿预算不被审查修改。


def test_complete_review_still_requires_original_literal_binding(analysis):
    runtime=ReviewRuntime();result=run_case(analysis,runtime,strict=True)
    assert result.status=='completed' and result.answer_evidence.validated
    assert result.progress['local_answer_review_protocol']['failure']==''
    review=runtime.calls[-1]
    assert 'kind=final/tool' not in review['system_prompt']
    assert ANALYSIS_EVIDENCE_GUIDANCE in review['system_extra_blocks']
    assert not review['native_tools'] and not review['native_tool_choice']


def test_ordinary_answer_keeps_fact_guidance_without_quality_model(analysis):
    runtime = ReviewRuntime()
    result = run_case(analysis, runtime)
    assert result.status == 'completed' and len(runtime.calls) == 2
    assert ANALYSIS_EVIDENCE_GUIDANCE in runtime.calls[1]['system_extra_blocks']
    assert result.completion['layers']['answer_evidence'] == 'unverified'


def test_larger_budget_never_accepts_forged_quote(analysis):
    result=run_case(analysis,ReviewRuntime(bad_quote=True),strict=True)
    assert not result.completion['accepted']
    assert result.progress['local_answer_summary']['binding_error']=='invalid_quote'


def test_runtime_error_has_accurate_safe_diagnostic_without_retry(analysis):
    runtime=ReviewRuntime(error=True);result=run_case(analysis,runtime,strict=True)
    assert not result.completion['accepted']
    assert result.progress['local_answer_summary']['binding_error']=='local_answer_review_runtime_error'
    assert 'private failure detail' not in json.dumps(result.progress)
    assert len(runtime.calls)==5


def test_strict_local_answer_does_not_pass_when_review_truncated(analysis):
    runtime=ReviewRuntime('length');result=run_case(analysis,runtime,strict=True)
    assert result.status!='completed' and not result.completion.get('accepted',False)
    assert result.progress['local_answer_summary']['binding_error']=='local_answer_review_output_length'


@pytest.mark.parametrize('finish,expected',[('stop','verified'),('length','unverified')])
def test_original_read_loop_through_real_sdk_review(analysis,finish,expected):
    from test_structured_output_budget import sdk
    def review(payload):
        data=json.loads(payload['messages'][-1]['content']);fragment=next(iter(data['answer_fragments']))
        return {'verdict':'supported','claims':[{'answer_fragment_id':fragment,
            'evidence':[{'source_id':data['read_sources'][0]['id'],'quote':'两份配置独立'}]}],
            'coverage':[{'field_id':key,'supported':True,'answer_fragment_ids':[fragment]}
                for key in data['required_field_selectors']],'missing_fields':[]}
    with sdk(review,finish) as (concrete,calls):
        class Mixed(ReviewRuntime):
            def call_chat_json(self,**kwargs):
                if '本地答案质量审查器' not in kwargs.get('system_prompt', ''):
                    return super().call_chat_json(**kwargs)
                self.calls.append(kwargs)
                return concrete.call_chat_json(**kwargs)
        result=run_case(analysis,Mixed(),strict=True)
        assert bool(result.answer_evidence.validated)==(expected=='verified')
        assert len(calls)==(1 if finish=='stop' else 2) and 'max_tokens' not in calls[0]
        assert calls[0]['reasoning_effort']=='high' and not calls[0].get('tools')


def test_supported_review_never_overrides_revoked_read_permission(analysis):
    class Revoking(ReviewRuntime):
        def call_chat_json(self,**kwargs):
            value=super().call_chat_json(**kwargs)
            if len(self.calls)==3:analysis[-1][0]=False
            return value
    result=run_case(analysis,Revoking(),strict=True)
    assert result.status!='completed' and not result.completion.get('accepted',False)


def test_ordinary_no_review_still_rechecks_permission_before_delivery(analysis):
    class Revoking(ReviewRuntime):
        def call_chat_json(self, **kwargs):
            value = super().call_chat_json(**kwargs)
            if len(self.calls) == 2:
                analysis[-1][0] = False
            return value
    result = run_case(analysis, Revoking())
    assert result.status != 'completed' and not result.completion.get('accepted', False)
    assert ANSWER not in result.final_text


def test_attribute_semantics_fixture_includes_independent_and_shared_cases():
    # Python自身验证反例：实例遮蔽不是“该实例改了没生效”，共享可变对象确有跨实例影响。
    class Config:
        timeout=17
        shared=[]
    a,b=Config(),Config();a.timeout=23
    assert (a.timeout,b.timeout,Config.timeout)==(23,17,17)
    a.shared.append('changed');assert b.shared==['changed']
    assert '不可变类默认值' in ANALYSIS_EVIDENCE_GUIDANCE and '共同约束' in ANALYSIS_EVIDENCE_GUIDANCE

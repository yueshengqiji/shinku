"""宿主选择的简短成功答复：省略流程用语，原状态仍约束中性核对。"""
import copy

import pytest
from shinku.agent_runtime.results import seal_result
from shinku.result_expression.natural import paragraph_blocks
from test_result_expression import canonical,host,SCOPE
from test_expression_semantic_coverage import semantic_sdk
from test_expression_paragraphs import service


def source(status='completed',accepted=True,*,brief=True,body=True):
    value=canonical(status,accepted=accepted)
    value.pop('result_hash')
    value['presentation']=dict(schema_version=1,handoff_version=1,brief_success_reply=brief,material_recovery=False)
    if not body:
        value['facts']=[];value['execution_report']={}
    return seal_result(value)


@pytest.mark.parametrize('status,accepted,brief,body,required',[
    ('completed',True,True,True,False),('completed',True,False,True,True),
    ('completed',True,True,False,True),('completed',False,True,True,True),
    ('failed',False,True,True,True),('partial',False,True,True,True),
    ('canceled',False,True,True,True),('running',False,True,True,True),
])
def test_only_explicit_host_policy_and_concrete_verified_success_make_status_optional(status,accepted,brief,body,required):
    value=source(status,accepted,brief=brief,body=body);before=copy.deepcopy(value)
    blocks=paragraph_blocks(value,detail_policy_version=2,presentation_version=6)
    assert next(row for row in blocks if row['id']=='status')['required'] is required
    assert value==before and value['execution']['status']==status


def test_optional_success_state_is_retained_for_independent_checker(host):
    value=source()
    def build(payload):
        rows=[r for r in payload['fact_blocks'] if r.get('required',True)]
        return dict(schema_version=2,binding_hash=payload['binding_hash'],paragraphs=[
            dict(text='已核对这些具体结果，详细记录在这里。\n'+'\n'.join(r['text'] for r in rows),
                 fact_ids=[r['id'] for r in rows])])
    with semantic_sdk(build=build) as (model,checker,calls):
        svc=service(host,model,checker,detail_policy_version=2,paragraph_presentation_version=6,semantic_review_version=1)
        record=svc.prepare(value,scope=SCOPE,check_current=lambda:'')
        assert record['status']=='ready',record['fallback_reason']
        assert 'status' not in calls[0][2]['required_fact_ids']
        context={r['id']:r for r in calls[1][2]['context_fact_blocks']}
        assert 'status' in context and '验收通过' in context['status']['text']
        assert record['canonical_result']['execution']['status']=='completed'

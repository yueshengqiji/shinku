"""The observed scalar remaining response cannot create a task or gain a repair call."""
import json

from test_qq_query_continuation import (
    ANSWER, BODY_A, GOAL, URL_B, drain, query_host, saved_task,
)
from test_qq_independent_agent import turn


def test_scalar_remaining_rejected_before_contract_worker_or_followup_read(query_host):
    host=query_host
    host.execution.review_transform=lambda value,payload:{**value,'remaining':'乙店价格及来源'}
    reply=host.bridge(turn(GOAL))
    assert not reply.background_task_id and not host.svc.coordinator.list()
    assert len(host.reader.calls)==1 and host.mutation.calls==0
    review=[r for r in host.execution.calls if 'Foreground query checkpoint review' in str(r.get('system_extra_blocks'))]
    assert len(review)==len(host.execution.calls)==1 and review[0]['native_tools'] is None
    assert not any('任务契约与语义承接' in str(r.get('system_extra_blocks')) for r in host.execution.calls)


def test_current_model_input_has_parseable_array_example_and_preserves_actual_quote_gate(query_host):
    host=query_host;actual=host.execution.call_chat_json;examples=[]
    def with_example(**kwargs):
        text=' '.join(kwargs.get('system_extra_blocks') or [])
        if 'Foreground query checkpoint review' in text:
            assert kwargs['native_tools'] is None and kwargs['native_tool_choice']==''
            marker='JSON example: '
            assert marker in text
            example,_=json.JSONDecoder().raw_decode(text.split(marker,1)[1])
            assert set(example)=={'status','progress','remaining','next_ids'}
            assert all(isinstance(example[k],list) for k in ('progress','remaining','next_ids'))
            assert set(example['progress'][0])=={'receipt_id','quote','reason'}
            examples.append(example)
        return actual(**kwargs)
    host.execution.call_chat_json=with_example
    reply=host.bridge(turn(GOAL));record=drain(host,reply);task=saved_task(host)
    assert len(examples)==1 and record['status']==task['status']=='completed'
    assert task['completion']['accepted'] is True and task['last_report']==ANSWER
    assert len(host.reader.calls)==2 and host.reader.calls[-1]['url']==URL_B
    review=next(r for r in host.execution.calls if 'Foreground query checkpoint review' in str(r.get('system_extra_blocks')))
    payload=json.loads(review['user_prompt']);assert payload['read_sources'][0]['text']==BODY_A
    assert task['budget']['requests']==len(host.chat.calls)+len(host.execution.calls)
    assert host.mutation.calls==0

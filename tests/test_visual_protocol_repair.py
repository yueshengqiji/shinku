"""LO06: actual QQ/media/delivery routes distinguish protocol failure from ambiguity."""
from reply_review_fixture import dialogue_review_result
import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.qq.interaction_tools import InteractionRuntime
from shinku.agent.budget import current_task_budget, TaskBudget, TaskBudgetExceeded
from shinku.qq.visual_selection import request_selection
from test_lightweight_visual_selection import PNG, candidate_payload, pipeline


@pytest.fixture(params=['legacy', 'task'])
def flow(request, pipeline):
    p = pipeline
    p.mode = request.param
    p.bridge.tool_decision_mode = p.mode
    return p


def replies(p, values):
    queue = list(values)
    def respond(**options):
        p.calls.append(options)
        review = dialogue_review_result(options)
        if review is not None:
            return review
        assert queue, 'unexpected additional model call'
        value = queue.pop(0)
        return value(options) if callable(value) else value
    runtime = SimpleNamespace(call_chat_json=respond)
    p.bridge.runtime = (InteractionRuntime(runtime, p.bridge.tool_host.registry, {})
                        if p.mode == 'task' else runtime)
    return queue


def select(options, action='none', text='今天可以先吃面。'):
    pending = candidate_payload(options)
    return {'kind': 'final', 'text': text, 'visual_context': {'action': action,
            'source_ids': [pending['candidates'][0]['source_id']] if action == 'select' else [],
            'topic_change': False}}


def seed(p):
    p.send('old-image', '之前的图片', sender='other', image=True)


def assert_no_task(p):
    assert not p.bridge.task_store.candidates(requester='owner', session='group:g')


def test_no_candidate_retains_single_normal_reply(flow):
    p = flow
    queue = replies(p, [{'kind': 'final', 'text': '今天可以先吃面。'}])
    p.send('plain', '晚饭吃什么', directed=True)
    assert not queue and len(p.calls) == 1 and len(p.sent) == 1
    assert p.replies[-1].final_text == '今天可以先吃面。'
    assert_no_task(p)


def test_valid_none_reprompts_without_closing_old_source(flow):
    p = flow
    seed(p)
    queue = replies(p, [select, {'kind': 'final', 'text': '正式答复：今天吃面。'}])
    p.send('plain', '晚饭吃什么', directed=True)
    assert not queue and len(p.calls) == 2 and len(p.sent) == 1
    assert p.replies[-1].final_text == '正式答复：今天吃面。'
    assert not p.calls[0].get('user_images') and not p.calls[0].get('native_tools')
    assert p.dispatcher._visual_context[('group', 'g', 'old-image')].followup_open
    assert_no_task(p)


@pytest.mark.parametrize('bad', [{'kind': 'final', 'text': '未经核实的首轮正文'},
    {'visual_context': {'action': 'unknown', 'source_ids': []}},
    {'visual_context': {'action': 'none', 'source_ids': [], 'topic_change': 'false'}}])
def test_missing_or_invalid_protocol_repairs_once_then_requests_formal_answer(flow, bad):
    p = flow
    seed(p)
    queue = replies(p, [bad, select, {'kind': 'final', 'text': '正式答复：今天吃面。'}])
    p.send('plain', '晚饭吃什么', directed=True)
    assert not queue and len(p.calls) == 3 and len(p.sent) == 1
    assert p.replies[-1].final_text == '正式答复：今天吃面。'
    assert all(not row.get('user_images') and not row.get('native_tools') for row in p.calls[:2])
    assert p.dispatcher._visual_context[('group', 'g', 'old-image')].followup_open
    assert_no_task(p)


def test_two_unrelated_mentions_do_not_repeat_an_image_question(flow):
    p = flow
    seed(p)
    queue = replies(p, [{'speech': '旧正文'}, select, {'kind': 'final', 'text': '晚饭吃面。'},
                        {'speech': '另一个旧正文'}, select, {'kind': 'final', 'text': '我在。'}])
    p.send('first', '晚饭吃什么', directed=True)
    p.send('second', '最近心情怎么样', directed=True)
    assert not queue and len(p.calls) == 6 and len(p.sent) == 2
    assert all('哪张图片' not in row.final_text for row in p.replies)
    assert p.dispatcher._visual_context[('group', 'g', 'old-image')].followup_open
    assert_no_task(p)


@pytest.mark.parametrize('bad', [{}, {'visual_context': {'action': 'bad', 'source_ids': []}}])
def test_exhausted_protocol_repair_is_failed_and_does_not_deliver_old_claim(flow, bad):
    p = flow
    seed(p)
    queue = replies(p, [bad, {'speech': '首轮旧图已识别完成'}])
    p.send('plain', '普通聊天', directed=True)
    result = p.replies[-1]
    assert not queue and len(p.calls) == 2 and len(p.sent) == 1
    assert result.agent_status == 'failed' and result.reason == 'visual_selection_missing'
    assert '哪张图片' not in result.final_text and '旧图已识别' not in result.final_text
    assert all(not row.get('user_images') for row in p.calls)
    assert_no_task(p)


@pytest.mark.parametrize('failure', ['fallback', 'exception'])
def test_transport_failure_is_not_ambiguity_or_a_successful_chat(flow, failure):
    p = flow
    seed(p)
    def failed(_opts):
        raise ConnectionError('fixture transport failure')
    queue = replies(p, [failed if failure == 'exception' else
                        {'_agent_transport_fallback': True, 'kind': 'final', 'text': 'fallback'}])
    p.send('plain', '普通聊天', directed=True)
    result = p.replies[-1]
    assert not queue and len(p.calls) == len(p.sent) == 1
    assert result.agent_status == 'failed' and result.reason == 'visual_selection_transport_failed'
    assert '哪张图片' not in result.final_text
    assert_no_task(p)


def test_true_ambiguity_keeps_one_question_and_no_task(flow):
    p = flow
    seed(p)
    p.send('another-image', '另一张图片', sender='third', image=True)
    queue = replies(p, [lambda opts: select(opts, 'clarify', '你说的是前一张还是后一张？')])
    p.send('question', '那个细节呢', directed=True)
    result = p.replies[-1]
    assert not queue and len(p.calls) == len(p.sent) == 1
    assert result.agent_status == 'waiting_user' and result.reason == 'visual_ambiguous'
    assert result.final_text == '你说的是前一张还是后一张？'
    assert_no_task(p)


def test_repaired_select_reads_verified_bytes_then_answers_once(flow):
    p = flow
    seed(p)
    queue = replies(p, [{}, lambda opts: select(opts, 'select', '不要发布选择阶段的正文'),
                        {'kind': 'final', 'text': '本轮读取的像素是白色。'}])
    p.send('question', '换个说法说明画面', directed=True)
    assert not queue and len(p.calls) == 3 and len(p.sent) == 1
    assert all(not row.get('user_images') and not row.get('native_tools') for row in p.calls[:2])
    picture = p.calls[2]['user_images'][0]
    assert picture['source_sender_id'] == 'other' and picture['source_message_id'] == 'old-image'
    assert base64.b64decode(picture['data_url'].split(',', 1)[1]) == PNG
    assert p.replies[-1].final_text == '本轮读取的像素是白色。'
    assert_no_task(p)


def test_task_visual_none_cannot_hide_extra_authority_fields(flow):
    p = flow
    if p.mode != 'task':
        return
    seed(p)
    queue = replies(p, [lambda opts: {**select(opts), 'permission_mode': 'ultra'}])
    p.send('plain', '普通聊天', directed=True)
    assert not queue and len(p.calls) == 1
    assert p.replies[-1].agent_status != 'completed'
    assert_no_task(p)


def test_autonomous_protocol_failure_is_suppressed_with_original_reason(flow):
    p = flow
    seed(p)
    p.dispatcher.handler = p.turns.append
    p.send('question', '普通聊天', directed=True)
    turn = p.turns[-1]
    turn = replace(turn, attention_mode='autonomous', messages=tuple(
        replace(message, segments=tuple(s for s in message.segments if s.kind != 'at')) for message in turn.messages))
    queue = replies(p, [{}, {}])
    result = p.bridge.handle_turn(turn)
    assert not queue and len(p.calls) == 2 and not p.sent
    assert not result.sent and result.delivery_status == 'suppressed'
    assert result.agent_status == 'failed' and result.reason == 'visual_selection_missing'
    assert_no_task(p)


@pytest.mark.parametrize('failure', ['changed', 'deleted'])
def test_source_read_failure_stays_terminal_without_protocol_retry(flow, failure):
    p = flow
    seed(p)
    p.clock[0] += 6
    p.source.write_bytes(PNG + b'changed') if failure == 'changed' else p.source.unlink()
    queue = replies(p, [lambda opts: select(opts, 'select')])
    p.send('question', '说明画面', directed=True)
    result = p.replies[-1]
    assert not queue and len(p.calls) == 1 and not p.calls[0].get('user_images')
    assert result.agent_status == 'failed' and result.reason in {
        'visual_source_changed', 'visual_source_unavailable', 'visual_read_failed'}
    assert '无法核实或读取' in result.final_text
    assert_no_task(p)


def test_expired_source_budget_denies_reader_before_attachment(flow):
    p = flow
    seed(p)
    p.clock[0] += 6
    p.bridge.recovery_enabled = True
    p.bridge.task_budget_limits = {'max_tools': 1}
    def respond(options):
        budget = current_task_budget()
        assert budget is not None
        budget.tools = budget.max_tools
        return select(options, 'select')
    queue = replies(p, [respond])
    reads = []
    p.dispatcher.set_visual_loader(lambda msg: reads.append(msg) or (_ for _ in ()).throw(AssertionError('must not read')))
    p.send('question', '说明画面', directed=True)
    assert not queue and len(p.calls) == 1 and not reads
    assert p.replies[-1].agent_status == 'failed' and p.replies[-1].reason == 'visual_read_budget'
    assert_no_task(p)


def test_selection_transport_wrapper_preserves_host_budget_stop():
    budget = TaskBudget(max_requests=1)
    budget.reserve_request({'messages': [], 'max_tokens': 1})
    with pytest.raises(TaskBudgetExceeded, match='task_model_budget_exhausted'):
        request_selection(lambda: budget.reserve_request({'messages': [], 'max_tokens': 1}))
    assert budget.requests == 1 and budget.reason == 'task_model_budget_exhausted'

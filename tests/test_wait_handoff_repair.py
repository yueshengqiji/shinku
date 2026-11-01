"""等待正文容错及同作用域选项交接；合成原文、真实SDK/远端适配，不调用供应商。"""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from pathlib import Path
import os

import pytest

from shinku.agent_runtime import AgentBoundaryError, AgentTurnRequest
from shinku.llm.final_json import recover_final_text_json
from shinku.llm.response_protocol import observe_response, recover_final_text
from shinku.qq.delegation_context import previous_task_sources
from test_agent_conversation_loop import native_sdk
from test_agent_t5_repairs import workspace_service
from test_agent_runtime_boundary import settled
from test_agent_delegation import remote, SCOPE
from test_qq_remote_delegation import bridge_for, close
from test_qq_tool_decision_boundary import submit
from test_qq_independent_agent import turn


RAW_WAIT = '{"kind":"wait","reason":"human_action_required","text":"主管，方案B已收到。\n需要确认具体目标；尚未启动服务。"}'


@pytest.mark.parametrize('discard', [False, True])
def test_wait_only_text_newlines_recovered_in_explicit_scope(discard):
    assert recover_final_text(RAW_WAIT) is None
    with observe_response(allow_final_text_json=True, discard_final_task_metadata=discard) as observation:
        assert recover_final_text(RAW_WAIT) == json.loads(RAW_WAIT, strict=False)
        assert observation.events == [{'stage': 'wait_text_json_recovered'}]


@pytest.mark.parametrize('raw', [
    RAW_WAIT[:-1], RAW_WAIT.replace('方案B', '方案"B"'),
    RAW_WAIT.replace('human_action_required', 'human\naction'),
    RAW_WAIT.replace('主管，', '\x00'),
    RAW_WAIT[:-1]+',"task_contract":{}}', RAW_WAIT[:-1]+',"task_link":{}}',
    RAW_WAIT[:-1]+',"tool_call":null}', RAW_WAIT[:-1]+',"host_grant":{}}',
    RAW_WAIT[:-1]+',"task_outcome":"completed"}', RAW_WAIT[:-1]+',"kind":"final"}',
    RAW_WAIT.replace('human_action_required', 'x'*121),
    RAW_WAIT.replace('"human_action_required"', '[]'),
])
def test_wait_never_repairs_permissions_calls_metadata_truncation_or_quotes(raw):
    assert recover_final_text_json(raw) is None
    assert recover_final_text_json(raw, discard_task_metadata=True) is None


def test_real_sdk_wait_retains_original_and_waits_without_extra_call(tmp_path):
    with workspace_service(tmp_path, mode='read') as (service, _, _, port):
        def respond(payload, index):
            return {'role': 'assistant', 'content': RAW_WAIT}, 'stop'
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            h = port.submit(AgentTurnRequest('wait-format', '合成任务，缺少目标时说明。'))
            assert settled(port, h.task_id)['status'] == 'waiting_user'
            result = port.result(h.task_id)
            assert result['execution_report']['text'] == json.loads(RAW_WAIT, strict=False)['text']
            assert result['execution']['reason'] == 'human_action_required'
            assert len(calls) == result['budget']['requests'] == 1 and not result['evidence']
            state = service._sessions[h.task_id].checkpoint.state
            assert state.native_tool_history.messages[-1]['content'] == RAW_WAIT
            assert state.progress['answer_format_recovery']['kind'] == 'wait_text_control_characters'
            assert not list(tmp_path.iterdir())


def context_host(result=None, *, error=None):
    def handoff(task_id):
        if error: raise AgentBoundaryError(error)
        return result
    def rows(scope, *, recent_limit):
        assert scope == SCOPE and recent_limit == 3
        return [{'request_key': 'previous', 'task_id': 'prior'}]
    return SimpleNamespace(_rows=rows, client=SimpleNamespace(port=lambda scope: SimpleNamespace(handoff=handoff)))


def prior_result(**changes):
    return {'task_id': 'prior', 'scope': asdict(SCOPE), 'updated': 100,
            'result_hash': 'bound-by-remote-client', 'original_requests': [{'text': '列出两种方案'}],
            'execution_report': {'text': 'A：仅改客户端网络出口。\nB：在本机启动服务型反代。'}, **changes}


def test_context_preserves_complete_options_and_does_not_transfer_old_authority():
    result = prior_result(host_grant={'root': True}, contract={'goal': '旧任务'}, facts=['旧事实'])
    sources, gaps = previous_task_sources(context_host(result), SCOPE, 'current', now=101)
    assert len(sources) == 1 and not gaps
    source = sources[0]
    assert source.content['report'] == result['execution_report']['text']
    assert source.content['original_requests'] == result['original_requests']
    assert source.content['is_fact_evidence'] is False and source.required is False
    assert not any(k in source.content for k in ('host_grant', 'contract', 'facts'))
    assert source.provenance['task_id'] == 'prior'


@pytest.mark.parametrize('change,reason', [
    ({'scope': {'requester': 'other', 'session': SCOPE.session}}, 'delegation_context_scope_conflict'),
    ({'scope': {'requester': SCOPE.requester, 'session': 'group:other'}}, 'delegation_context_scope_conflict'),
    ({'task_id': 'foreign'}, 'delegation_context_scope_conflict'),
    ({'updated': -1800}, 'delegation_context_expired'),
    ({'updated': 200}, 'delegation_context_expired'),
    ({'execution_report': {'text': ''}}, 'delegation_context_report_unavailable'),
    ({'execution_report': {'text': '条'*5000}}, 'delegation_context_oversized'),
])
def test_context_unavailable_scopes_and_long_options_create_explicit_gaps(change, reason):
    sources, gaps = previous_task_sources(context_host(prior_result(**change)), SCOPE, 'current', now=101)
    assert not sources and gaps == [{'source_id': 'previous-task:prior', 'required': False, 'reason': reason}]


def test_context_revocation_cannot_copy_old_result():
    sources, gaps = previous_task_sources(context_host(error='host_authorization_denied'), SCOPE, 'current', now=101)
    assert not sources and gaps[0]['reason'] == 'host_authorization_denied'


def test_two_real_remote_turns_deliver_prior_options_to_agent_without_memory_or_execution(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    options = 'A：仅改客户端网络出口。\nB：在本机启动服务型反代。'
    observed = []
    def respond(**kwargs):
        observed.append(kwargs)
        if len(observed) == 1:
            return {'kind': 'wait', 'text': options, 'reason': 'choice_required'}
        assert all(line in kwargs['user_prompt'] for line in options.splitlines())
        return {'kind': 'wait', 'text': '已取得上一轮B的原始含义；没有继承旧权限或启动服务。', 'reason': 'target_missing'}
    runtime.call_chat_json = respond
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit(), submit()])
    try:
        first = bridge(turn('请列出两种方案', message_id='z-first'))
        assert first.agent_status == 'waiting_user' and len(observed) == 1
        second = bridge(turn('是B', message_id='a-second'))
        if second.agent_status != 'waiting_user' and os.environ.get('ANGELOS_UI_EVIDENCE_DIR'):
            dest = Path(os.environ['ANGELOS_UI_EVIDENCE_DIR']); dest.mkdir(parents=True, exist_ok=True)
            (dest/'handoff-failure.json').write_text(json.dumps(observed, ensure_ascii=False, indent=2))
        assert second.agent_status == 'waiting_user', (second.reason, second.final_text)
        assert len(observed) == 2 and len(chat.calls) == 2 and tool.calls == 0
        row = next(r for r in bridge.delegation._rows(SCOPE) if r['request_key'] == 'a-second')
        task = service.task_store.get(row['task_id'], requester=SCOPE.requester, session=SCOPE.session)
        assert task['requests'][0]['text'] == '是B' and task['contract'] is None
        assert any(m['kind'] == 'related_message' for m in task['material_package']['items'])
        with service.task_store._transaction() as db:
            record = db.execute("SELECT payload FROM agent_task_materials WHERE task_id=? AND payload LIKE ?",
                                (row['task_id'], '%previous_scoped_task%')).fetchone()
        material = json.loads(record[0])
        assert material['content']['report'] == options and material['required'] is False
        assert bridge.memory_service is None and not task.get('evidence')
    finally:
        close(store, coordinator)

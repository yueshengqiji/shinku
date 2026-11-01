"""Original URLs and executed search receipts are separate read admissions."""
import copy
from dataclasses import replace
import json
from uuid import uuid4

import pytest

from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent.request_scope import RequestScope
from shinku.qq.interaction_tools import validate_interaction_response
from test_agent_chat_handoff import probe


def run_direct(probe, case, arm, *, query_arguments=None, protocol_fault=False):
    parent = probe.OUTPUT_ROOT / ('direct-original-url-' + uuid4().hex)
    parent.mkdir()
    root = parent / 'run'
    ledger = probe.JointLedger(parent / 'budget.json', {'mode': 'offline', 'slots': ['direct:' + arm]})
    ledger.claim('direct:' + arm)

    class Provider(probe.FixtureProvider):
        review_requests = 0
        def __call__(self, request):
            payload = json.loads(request.content)
            system = '\n'.join(str(row.get('content', '')) for row in payload['messages'] if row['role'] == 'system')
            if '答案证据与覆盖复核' in system:
                self.review_requests += 1
                if protocol_fault and self.review_requests == 1:
                    self.payloads.append(payload)
                    return probe.json_response(payload, 'PRIVATE_INVALID_REVIEW_SENTINEL')
            tools = {row['function']['name'] for row in payload.get('tools', [])}
            if {'query_search', 'query_read'} <= tools and payload['model'] == probe.CHAT_MODEL:
                self.payloads.append(payload)
                if not self.host.source_read.calls:
                    return probe.json_response(payload, tool='query_read', arguments=query_arguments
                        if query_arguments is not None else {'url': case['sources'][0]['url']})
                if case['sources'][0].get('denied'):
                    return probe.json_response(payload, {'kind': 'final',
                        'text': '原始来源访问被拒，尚未核实商品价格。', 'task_outcome': 'failed'})
                return probe.json_response(payload, {'kind': 'final',
                    'text': case['sources'][0]['body'] + '来源：' + case['sources'][0]['url'],
                    'task_outcome': 'completed'})
            return super().__call__(request)

    fixture = Provider(case, arm, root)
    with ledger.activity():
        result = probe.run_joint(case, arm, root, ledger, ('offline-key', 'offline-key'),
                                 live=False, fixture=fixture)
    ledger.finish('direct:' + arm)
    return result, fixture, ledger


@pytest.mark.parametrize('arm', ['off', 'on'])
@pytest.mark.parametrize('case_id', ['foreground_complete', 'access_denied'])
def test_actual_sdk_direct_original_url_observes_body_or_denial_without_search(probe, arm, case_id):
    case = copy.deepcopy(next(c for c in probe.supplemental_corpus()['cases'] if c['id'] == case_id))
    result, provider, ledger = run_direct(probe, case, arm)
    assert result['source_reads'] == [case['sources'][0]['url']], result
    assert result['source_searches'] == 0 and result['task_count'] == 0 and not result['background_work_id']
    assert result['complete'] is (case_id == 'foreground_complete'), result
    assert result['hidden_tools'] and result['inputs_preserved'] and result['platform_sends'] == 0
    if case_id == 'foreground_complete':
        assert any('10' in text and case['sources'][0]['url'] in text for text in result['local_publication_texts'])
    else:
        assert provider.host.source_read.read_observations[0]['denied'] is True
        assert provider.host.source_read.read_observations[0]['content_chars'] == 0
    assert all(row['state'] == 'response' for row in ledger.attempts)
    # These diagnostic direct-URL paths do not rewrite the frozen one-search
    # scenario criterion or turn a new observation into an old batch pass.
    assert not result['passed'] and 'intended_query_chain_not_observed' in result['failures']


def test_scope_guard_requires_original_primary_or_executed_candidates(probe):
    case = copy.deepcopy(next(c for c in probe.supplemental_corpus()['cases'] if c['id'] == 'foreground_complete'))
    _, provider, _ = run_direct(probe, case, 'off')
    cap = provider.host.bridge.chat_read_capabilities['query_read']
    url = case['sources'][0]['url']
    turn = probe.make_turn(case['request'], 'scope-fixture')
    context = {'qq_turn': turn}
    assert cap.admits({'url': url}, context)
    assert not cap.admits({'url': url + '/other'}, context)
    assert not cap.admits({'url': url}, {})
    unlinked = probe.make_turn('只聊聊天。', 'scope-fixture')
    assert not cap.admits({'url': url}, {'qq_turn': unlinked})
    assistant_only = replace(unlinked, messages=(*unlinked.messages,
        replace(turn.messages[0], message_id='other-message', sender_id='another-sender')))
    assert not cap.admits({'url': url}, {'qq_turn': assistant_only})
    quoted_only = replace(unlinked, messages=(replace(unlinked.messages[0],
        quoted_message=turn.messages[0].as_dict()),))
    assert not cap.admits({'url': url}, {'qq_turn': quoted_only})
    wrong_author = replace(turn, messages=(replace(turn.messages[0], sender_id='other-requester'),))
    assert not cap.admits({'url': url}, {'qq_turn': wrong_author})
    assert not cap.admits({'url': url}, {'qq_turn': replace(turn, conversation_id='other-session')})
    assert not cap.admits({'url': url}, {'qq_turn': replace(turn, messages=turn.messages * 2)})
    receipt = {'tool': 'query_search', 'arguments': {'query': 'fixed'},
               'result': {'state_updates': {'results': [{'url': url}]}}, 'source': 'executed_read_adapter'}
    assert cap.admits({'url': url}, {'qq_turn': unlinked, '_chat_read_receipts': [receipt]})
    forged = {**receipt, 'source': 'model_report'}
    assert not cap.admits({'url': url}, {'qq_turn': unlinked, '_chat_read_receipts': [forged]})
    assert not cap.admits({'url': url}, {'qq_turn': unlinked, '_chat_read_receipts': [None, {'source': 'executed_read_adapter'}]})
    scope = RequestScope(probe.SCOPE.requester, probe.SCOPE.session, 'current', (('current', case['request']),))
    assert cap.admits({'url': url}, {'agent_request_scope': scope})
    assert not cap.admits({'url': url}, {'agent_request_scope': replace(scope, requester='other-requester')})
    assert not cap.admits({'url': url}, {'agent_request_scope': replace(scope, session='private:other-session')})
    assert not cap.admits({'url': url}, {'qq_turn': turn, 'agent_request_scope': scope.__dict__})
    no_original_url = replace(scope, requests=(('current', '只聊聊天。'),))
    assert not cap.admits({'url': url}, {'agent_request_scope': no_original_url})
    assert cap.admits({'url': url}, {'agent_request_scope': no_original_url, '_chat_read_receipts': [receipt]})


@pytest.mark.parametrize('case', ['scope', 'schema'])
def test_rejected_valid_interface_keeps_safe_reason_and_no_caller_values(probe, case):
    fixture = copy.deepcopy(next(c for c in probe.supplemental_corpus()['cases'] if c['id'] == 'foreground_complete'))
    _, provider, _ = run_direct(probe, fixture, 'off')
    caps = provider.host.bridge.chat_read_capabilities
    sentinel = 'PRIVATE_ARGUMENT_SENTINEL'
    args = {'url': fixture['sources'][0]['url']} if case == 'scope' else {'url': sentinel, sentinel: sentinel}
    with pytest.raises(AgentBoundaryError, match='chat_interface_arguments_rejected') as caught:
        validate_interaction_response({'kind': 'tool', 'tool_call': {'name': 'query_read', 'arguments': args}}, caps, {})
    diagnostic = caught.value.argument_diagnostic
    assert diagnostic['schema_valid'] is (case == 'scope')
    assert diagnostic['source_guard_checked'] is (case == 'scope')
    assert diagnostic['source_guard_accepted'] is False
    assert diagnostic['unknown_argument_count'] == (0 if case == 'scope' else 1)
    assert diagnostic['missing_required_count'] == 0
    assert diagnostic['interface'] == 'query_read'
    assert sentinel not in json.dumps(diagnostic)


@pytest.mark.parametrize('fault', ['unknown_field', 'missing_required', 'wrong_type', 'source_scope'])
def test_actual_sdk_interface_rejection_has_safe_trace_and_no_protocol_retry(probe, fault):
    case = copy.deepcopy(next(c for c in probe.supplemental_corpus()['cases'] if c['id'] == 'foreground_complete'))
    sentinel = 'PRIVATE_ARGUMENT_SENTINEL'
    url = case['sources'][0]['url']
    arguments = {'unknown_field': {'url': url, sentinel: sentinel}, 'missing_required': {},
        'wrong_type': {'url': [sentinel]}, 'source_scope': {'url': url + '/' + sentinel}}[fault]
    result, _, ledger = run_direct(probe, case, 'off', query_arguments=arguments)
    assert result['reply_status'] == 'blocked'
    assert not result['source_reads'] and result['source_searches'] == result['task_count'] == 0
    assert len(ledger.attempts) == 1 and result['platform_sends'] == 0
    rejected = next(row for row in result['chat_protocol_diagnostics'] if row['name'] == 'chat_interface_validation')
    assert rejected['reason'] == 'chat_interface_arguments_rejected'
    diagnostic = rejected['argument_validation']
    assert diagnostic['schema_valid'] is (fault == 'source_scope')
    assert diagnostic['source_guard_checked'] is (fault == 'source_scope')
    assert diagnostic['unknown_argument_count'] == (1 if fault == 'unknown_field' else 0)
    assert diagnostic['missing_required_count'] == (1 if fault == 'missing_required' else 0)
    assert diagnostic['invalid_known_field_count'] == (1 if fault == 'wrong_type' else 0)
    assert sentinel not in json.dumps(result['chat_protocol_diagnostics'])


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_actual_sdk_query_completion_recovers_review_format_with_one_read_and_no_task(probe, arm):
    case = copy.deepcopy(next(c for c in probe.supplemental_corpus()['cases'] if c['id'] == 'foreground_complete'))
    result, provider, ledger = run_direct(probe, case, arm, protocol_fault=True)
    assert result['complete'] and result['source_reads'] == [case['sources'][0]['url']], result
    assert provider.review_requests == 2 and result['source_searches'] == result['task_count'] == 0
    assert not result['background_work_id'] and result['platform_sends'] == 0
    assert all(row['state'] == 'response' for row in ledger.attempts)
    assert 'PRIVATE_INVALID_REVIEW_SENTINEL' not in json.dumps(result['chat_protocol_diagnostics'])

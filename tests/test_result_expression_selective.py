"""自主表达回归：允许省略、保留来源，模型失败或恢复不能刷出审计日志。"""
import copy
import json
from dataclasses import replace

import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ModelReply
from shinku.result_expression.selective import materials
from test_result_expression import SCOPE, canonical, host, prepare, service


class Writer:
    def __init__(self, text='记录里有搜索工具；具体使用哪家搜索服务，还需要核对配置。', fail=False):
        self.calls, self.text, self.fail = [], text, fail

    def call_expression_json(self, **options):
        self.calls.append(options)
        if self.fail:
            raise TimeoutError('isolated failure')
        payload = json.loads(options['user_prompt'])
        return ModelReply({'binding_hash': payload['binding_hash'], 'text': self.text})


def selective(host, writer=None, **kw):
    return service(host, writer or Writer(), content_policy='selective', **kw)


def noisy_source():
    value = canonical()
    value.pop('result_hash')
    value['presentation'] = {'schema_version': 1}
    value['information'] = {'status': 'incomplete', 'required_status': 'complete',
        'projection': {'gaps': [{'path': '/evidence/0/receipt/registered/' + str(i),
                                'reason': 'depth_limit'} for i in range(100)]}}
    value['facts'] = [{'label': 'tools', 'value': {'registered': ['private_web_search', 'browser_observe']}}]
    return seal_result(value)


def test_short_answer_can_omit_full_report_and_ids_without_second_model(host):
    writer, checker = Writer(), Writer()
    svc = selective(host, writer, checker=checker, format_version=2, detail_policy_version=2,
                    paragraph_presentation_version=7, semantic_review_version=1)
    source = noisy_source(); before = copy.deepcopy(source)
    record = prepare(svc, source)
    assert record['status'] == 'ready' and record['text'] == writer.text
    assert len(writer.calls) == 1 and checker.calls == []
    payload = json.loads(writer.calls[0]['user_prompt'])
    assert '/evidence/' not in writer.calls[0]['user_prompt']
    assert 'required_fact_ids' not in payload and 'semantic_requirements' not in payload
    assert payload['host_state']['status'] == 'completed'
    assert payload['host_state']['sent'] is False
    assert source == before
    assert record['canonical_result'] == source
    assert record['check']['semantic_review'] == 'not_performed'
    assert prepare(svc, source)['text'] == record['text'] and len(writer.calls) == 1


def test_large_material_is_excerpted_not_completeness_failure(host):
    source = canonical(); source['presentation'] = {'schema_version': 1}; source['execution_report']['text'] = '多余细节\\"\n' * 1000
    source = seal_result({k:v for k,v in source.items() if k != 'result_hash'})
    writer = Writer(); record = prepare(selective(host, writer, max_input_chars=1500), source)
    assert record['status'] == 'ready'
    payload = json.loads(writer.calls[0]['user_prompt'])
    assert payload['material_excerpted'] and payload['host_state']['status'] == 'completed'
    assert len(writer.calls[0]['user_prompt']) + len('按真红口吻表达；PERSONA_PROMPT_PRIVATE_SENTINEL') <= 1500
    assert len(record['canonical_result']['execution_report']['text']) > 1500


@pytest.mark.parametrize('state', ['partial', 'failed', 'blocked', 'canceled', 'completed'])
def test_failed_generation_is_short_and_does_not_dump_snapshot(host, state):
    writer = Writer(fail=True); source = canonical(state, accepted=False)
    record = prepare(selective(host, writer), source)
    assert record['status'] == 'fallback' and len(record['text']) < 100
    assert 'PRIVATE_SENTINEL' not in record['text'] and '已全部上线' not in record['text']
    assert '验收通过' not in record['text']
    assert len(writer.calls) == 1
    assert record['canonical_result']['execution']['status'] == state


def test_revocation_and_wrong_scope_still_prevent_publication(host):
    svc = selective(host); record = prepare(svc)
    assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: 'agent_permission_changed')
    assert not svc.reserve_publication(record, scope=AgentScope('other', 'private:other'), check_current=lambda: '')
    assert svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')
    assert not svc.reserve_publication(record, scope=SCOPE, check_current=lambda: '')


def test_source_tampering_and_revocation_before_call_still_fail(host):
    writer = Writer(); svc = selective(host, writer)
    source = canonical(); source['execution']['status'] = 'failed'
    assert prepare(svc, source)['status'] == 'discarded'
    assert prepare(svc, guard=lambda: 'responses_disabled')['status'] == 'discarded'
    assert writer.calls == []


def test_recovery_uses_short_fallback_without_model_or_replay(host):
    svc = selective(host); source = noisy_source()
    from shinku.result_expression.facts import identity
    from shinku.agent_runtime.contracts import canonical_hash
    key = canonical_hash(identity(source))
    record, owned = svc.archive.claim(key, scope=SCOPE, result=source,
        config_hash=svc.profile.config_hash, binding_hash='fixture', content_policy='selective',
        processing_seconds=-1)
    assert owned
    recovered = svc.archive.recover_interrupted(key, scope=SCOPE)
    assert recovered['status'] == 'fallback' and len(recovered['text']) < 100
    assert '/evidence/' not in recovered['text']
    assert recovered['canonical_result'] == record['canonical_result']
    assert svc.profile.runtime.calls == []


def test_actual_qq_query_delivery_is_short_and_once(tmp_path):
    from test_qq_result_expression import make_host
    from test_qq_independent_agent import turn
    from shinku.conversation import AnswerEvidence
    from shinku.tools.invocation import ToolResultEnvelope
    executor_service, tool, executor, request, bridge, chat, sent = make_host(tmp_path)
    writer = Writer()
    bridge.result_expression.profile = replace(bridge.result_expression.profile,
                                                runtime=writer, content_policy='selective')
    envelope = ToolResultEnvelope('fixture-read', 'ok', 'raw snapshot', {'tool_type': 'read_self_state',
        'state_updates': {'result_facts': [{'label': 'tools', 'value': {'registered':
            ['private_web_search', 'browser_observe'] + ['fixture_tool_' + str(i) for i in range(80)]}}]}})
    try:
        context = {}
        kwargs = dict(context=context, status='completed', reason='fixture_query', route='perception',
                      envelopes=[envelope], answer_evidence=AnswerEvidence('state', 'unverified', False, 1, 1))
        reply = bridge._express_query_result(turn('你和执行助手使用什么联网搜索？'), 'fallback', **kwargs)
        assert reply.sent and reply.final_text == writer.text
        assert len(sent) == 1 and len(writer.calls) == 1
        again = bridge._express_query_result(turn('你和执行助手使用什么联网搜索？'), 'fallback', **kwargs)
        assert again.delivery_status == 'suppressed' and len(sent) == 1
        assert executor.calls == [] and tool.calls == 0 and chat.calls == []
        assert not executor_service.coordinator.list()
        assert context['agent_query_result']['verification']['accepted'] is False
    finally:
        executor_service.coordinator.shutdown(wait=True)
        executor_service.task_store.close()

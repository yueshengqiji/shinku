"""合成服务日志贯穿执行、证据重建、答案审查；拒绝越权、串任务和快照篡改。"""
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from shinku.agent.local_answer import LocalAnswerLedger
from shinku.agent.loop import _call_signature
from shinku.agent.task_state import new_task
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.angelos.service_logs import ServiceLogsTool
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


def contract():
    return {'goal': '查一下回复失败原因', 'deliverables': ['失败原因及证据'],
        'scope': ['只读诊断'], 'requirements': [
            {'id': 'answer', 'description': '解释原因', 'effect': 'local_answer', 'targets': [],
             'checks': {'required_fields': ['失败原因'], 'min_sources': 1}}]}


@pytest.fixture
def logs(tmp_path):
    service, _, runtime = demo_service(tmp_path)
    host = ServiceHost(service, requester='owner', reasoning_path=tmp_path/'reasoning.sqlite3',
                       diagnostic_units=('fixture.service',))
    if host.service_logs is None:
        pytest.skip('Linux user journal adapter')
    task = new_task(task_id='current', requester='owner', session='angelos:one',
                    request='查一下最近回复失败的原因', message_id='message', contract=contract())
    task.update(status='running', execution_attempt=1)
    service.task_store.save(task)
    grant = service._grant(AgentScope('owner', 'angelos:one'), task, 'inspect')
    context = execution_context(task, grant, attempt=1)
    reads = []
    def reader(query):
        reads.append(dict(query))
        return {'entries': [{'MESSAGE': '上游服务容量不足，返回错误。', '__CURSOR': 'synthetic-cursor'}],
                'more_in_window': False}
    host.service_logs.reader = reader
    yield service, host.service_logs, context, reads
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


def observed(logs):
    _, tool, context, _ = logs
    args = {'unit': 'fixture.service', 'since': 100, 'until': 200, 'limit': 5}
    value = tool.execute(call=args, context=context)
    assert value.get('status') != 'error', value
    invocation = ToolInvocation(tool.tool_type, args)
    ledger = LocalAnswerLedger()
    ledger.configure(context)
    envelope = ToolResultEnvelope(invocation.id, 'ok', value['followup_context'],
                                  {'state_updates': value['state_updates']}, executed=True)
    ledger.observe(invocation, envelope, tool, context)
    call = {'name': invocation.name, 'id': invocation.id, 'arguments': args, 'status': 'ok', 'executed': True,
            'data': envelope.data, 'call_fingerprint': hashlib.sha256(_call_signature(invocation).encode()).hexdigest(),
            'execution_binding': {'task_id': 'current', 'requester': 'owner', 'session': 'angelos:one', 'attempt': 1}}
    return ledger, call


def test_same_snapshot_reaches_execution_and_review_and_restart(logs):
    service, tool, ctx, reads = logs
    ledger, call = observed(logs)
    payload = ledger.review_payload('诊断', ctx, answer='上游服务容量不足')
    assert len(payload['read_sources']) == 1
    assert payload['read_sources'][0]['text'] == call['data']['state_updates']['log_text']
    assert '容量不足' in payload['read_sources'][0]['text']
    reopened = ServiceLogsTool(service, 'owner', tool.path, tool.units,
        reader=lambda q: pytest.fail('verification must not query the live journal again'))
    restored = LocalAnswerLedger(); restored.configure(ctx)
    restored.restore_reads([call], {tool.tool_type: reopened}, ctx)
    assert restored.review_payload('诊断', ctx, answer='原因')['read_sources'] == payload['read_sources']
    assert len(reads) == 1


@pytest.mark.parametrize('defect', ['hash', 'query', 'content', 'reference', 'attempt', 'request', 'scope', 'revoked', 'new_query'])
def test_receipts_never_cross_scope_or_allow_forgery(logs, defect):
    service, tool, ctx, reads = logs
    ledger, call = observed(logs)
    receipt = deepcopy(call['data']['state_updates'])
    context = deepcopy(ctx)
    if defect == 'hash': receipt['sha256'] = '0'*64
    if defect == 'query': receipt['query']['unit'] = 'other.service'
    if defect == 'reference': receipt['path'] = 'journal:missing'
    if defect == 'content':
        with sqlite3.connect(tool.path) as db:
            row = json.loads(db.execute('select payload from snapshots').fetchone()[0])
            row['text'] = 'FORGED'
            db.execute('update snapshots set payload=?', (json.dumps(row),))
    if defect in {'attempt', 'request', 'scope'}:
        task = service.task_store.get('current', requester='owner', session='angelos:one')
        if defect == 'attempt': task['execution_attempt'] += 1
        if defect == 'request': task['requests'].append({'message_id': 'next', 'text': '新的请求'})
        if defect == 'scope': context['agent_task_state']['session'] = 'angelos:other'
        service.task_store.save(task)
    if defect == 'revoked': service.authorize = lambda *args: None
    if defect == 'new_query':
        call['arguments']['limit'] = 3
        call['call_fingerprint'] = hashlib.sha256(_call_signature(ToolInvocation(tool.tool_type, call['arguments'])).encode()).hexdigest()
        restored = LocalAnswerLedger();restored.configure(ctx)
        restored.restore_reads([call], {tool.tool_type: tool}, ctx)
        assert not restored.sources
        return
    assert tool.agent_verify_result(receipt=receipt, context=context)['result_valid'] is False
    assert len(reads) == 1


@pytest.mark.parametrize('args', [
    {'unit': 'fixture.service; touch /tmp/forged'},
    {'unit': 'other.service'},
    {'unit': 'fixture.service', 'command': 'echo forged'},
    {'unit': 'fixture.service', 'since': float('nan')},
    {'unit': 'fixture.service', 'since': 1, 'until': 200000},
    {'unit': 'fixture.service', 'limit': True},
    {'unit': 'fixture.service', 'limit': 201},
])
def test_invalid_queries_do_not_execute(logs, args):
    _, tool, ctx, reads = logs
    result = tool.execute(call=args, context=ctx)
    assert result['status'] == 'error' and not reads


def test_limited_query_retains_nonexhaustive_scope(logs):
    _, tool, ctx, _ = logs
    tool.reader = lambda q: {'entries': [{'MESSAGE': '一条记录'}], 'more_in_window': True}
    ledger, call = observed(logs)
    source = ledger.review_payload('诊断', ctx, answer='一条记录')['read_sources'][0]
    assert json.loads(source['text'])['not_exhaustive'] is True
    assert json.loads(source['text'])['scope'] == 'latest_records_in_selected_window'


def test_query_failure_never_creates_a_source(logs):
    _, tool, ctx, _ = logs
    tool.reader = lambda q: (_ for _ in ()).throw(ValueError('service_log_read_timeout'))
    result = tool.execute(call={'unit': 'fixture.service'}, context=ctx)
    assert result['status'] == 'error'
    with sqlite3.connect(tool.path) as db:
        assert db.execute('select count(*) from snapshots').fetchone()[0] == 0

"""真实Agent/HTTP/统一NapCat适配器的文件链路；仅发送到合成替身，不访问生产。"""
import base64
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest

from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.agent_runtime.delegation import DelegationGateway, DelegationPrincipal
from shinku.agent_runtime.delegation_http import RemoteAgentClient, authority_payload
from shinku.agent_runtime.file_delivery import MAX_FILE_BYTES, TOOL, TaskFileDelivery, read_snapshot
from shinku.qq.delivery import deliver
from shinku.qq.file_delivery import QQFileDelivery
from shinku.qq.napcat import NapCatActionTransport
from shinku.tools.execution import ExecutionPolicy
from test_agent_delegation import SECRET, SCOPE, serve, settled
from test_qq_independent_agent import turn
from test_qq_remote_delegation import CatalogueOnly, bridge_for, close
from test_qq_tool_decision_boundary import submit


@contextmanager
def file_remote(tmp_path, *, generate=True, outcome='completed', resolver=None):
    service, tool, runtime = demo_service(tmp_path)
    path = tool.path if generate else tmp_path / '已有 文件.bin'
    if not generate:
        path.write_bytes(b'\x00SYNTHETIC_EXISTING_FILE\xff')
    base = service.authorize(SCOPE, {}, 'submit')
    base = replace(base, allowed_tools=base.allowed_tools | {TOOL}, permission_mode='ultra',
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type: 'low', TOOL: 'low'}))
    service.authorize = lambda s, t, o: base
    def resolve(value, context):
        target = Path(value)
        if not target.is_relative_to(tmp_path):
            raise PermissionError('outside synthetic root')
        return target
    broker = TaskFileDelivery(service, resolver or resolve, wait_seconds=3)
    contract = {'goal': '创建并发送文件' if generate else '发送已有文件', 'deliverables': [str(path)],
        'scope': [str(tmp_path)], 'requirements': [
            *([{'id': 'created', 'description': '实际创建文件', 'effect': 'write',
                'targets': [str(path)], 'checks': {'contains': 'standalone agent verified'}}] if generate else []),
            {'id': 'sent', 'description': '取得原QQ会话的消息回执', 'effect': 'send',
             'targets': [str(path)], 'checks': {'receipt_required': True}}]}
    decisions = [
        {'kind': 'tool', 'tool_call': {'name': 'declare_task_contract', 'arguments': {'task_contract': contract}}},
        *([{'kind': 'tool', 'tool_call': {'name': tool.tool_type, 'arguments': {'path': str(path)}}}] if generate else []),
        {'kind': 'tool', 'tool_call': {'name': TOOL, 'arguments': {'path': str(path), 'confirm': True}}},
        {'kind': 'final', 'text': '文件交付完成。' if outcome == 'completed' else '文件未取得发送回执。', 'task_outcome': outcome}]
    def decision(**kwargs):
        runtime.calls.append(kwargs)
        return decisions.pop(0) if decisions else {'kind': 'final', 'text': '未取得新的回执。', 'task_outcome': 'failed'}
    runtime.call_chat_json = decision
    gateway = DelegationGateway(service, {'shinku': DelegationPrincipal(SECRET, lambda s, t, o: base)}, file_delivery=broker)
    with serve(gateway) as endpoint:
        client = RemoteAgentClient(endpoint, 'shinku', SECRET, authority=lambda s, o: authority_payload(base))
        try:
            yield (service, tool, runtime, gateway, client, base), broker, path
        finally:
            client.close()
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


def file_bridge(tmp_path, remote, broker):
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    bridge.tool_host.registry.register(CatalogueOnly(remote[0].executor.tool_host.registry.get(TOOL)))
    messages, wire = [], []
    def action(*, action, params):
        wire.append((action, params))
        return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'synthetic-' + str(len(wire))}}
    transport = NapCatActionTransport(action)
    def sender(message):
        messages.append(message)
        return deliver(transport, message)
    bridge.sender = sender
    return bridge, chat, messages, wire, store, coordinator


@pytest.mark.parametrize('generate', [False, True])
def test_actual_agent_http_and_napcat_send_existing_or_new_file_to_original_qq(tmp_path, generate):
    with file_remote(tmp_path, generate=generate) as (remote, broker, path):
        bridge, chat, messages, wire, store, coordinator = file_bridge(tmp_path, remote, broker)
        try:
            original = ('请创建并验证 "' + str(path) + '" 并作为文件发送给我。' if generate
                        else '请发送文件 "' + str(path) + '" 给我，这是Angelos中的已有文件。')
            answer = bridge(turn(original))
            assert answer.agent_status == 'completed', (answer.reason, answer.final_text)
            file_messages = [m for m in messages if any(s.kind == 'file' for s in m.segments)]
            assert len(file_messages) == 1
            action, params = next((a, p) for a, p in wire if p['message'][0]['type'] == 'file')
            assert action == 'send_private_msg' and params['user_id'] == 'fixture'
            segment = params['message'][0]['data']
            assert segment['name'] == path.name and base64.b64decode(segment['file'].removeprefix('base64://')) == path.read_bytes()
            row = bridge.delegation._rows(SCOPE)[0]
            result = remote[4].port(SCOPE).result(row['task_id'])
            assert result['verification']['accepted'] and result['execution']['status'] == 'completed'
            receipt = next(r['receipt'] for r in result['evidence'] if r['tool'] == TOOL)
            assert receipt['verified_effects'] == ['send'] and receipt['delivery_receipt']
            assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            before = len(file_messages), remote[1].calls, len(remote[2].calls)
            # 重复原消息不会执行第二次；查询原结果不会再次发送附件。
            chat.replies.append(submit())
            bridge(turn(original))
            chat.replies.append({'kind': 'tool', 'tool_call': {'name': 'agent_task_result', 'arguments': {'task_id': row['task_id']}}})
            bridge(turn('查看原结果', message_id='read-existing-result'))
            assert len([m for m in messages if any(s.kind == 'file' for s in m.segments)]) == before[0]
            assert remote[1].calls == before[1] and len(remote[2].calls) == before[2]
            assert 'base64://' not in json.dumps(remote[2].calls)
        finally:
            close(store, coordinator)


@pytest.mark.parametrize('failure', ['no_receipt', 'exception', 'rejected'])
def test_failed_or_unknown_attachment_is_not_completed_and_never_retried(tmp_path, failure):
    with file_remote(tmp_path, outcome='failed') as (remote, broker, path):
        bridge, chat, messages, wire, store, coordinator = file_bridge(tmp_path, remote, broker)
        physical = []
        original_sender = bridge.sender
        def sender(message):
            if any(s.kind == 'file' for s in message.segments):
                physical.append(message)
                if failure == 'exception':
                    raise TimeoutError('synthetic response lost')
                from shinku.qq.delivery import DeliveryResult
                return DeliveryResult(failure == 'no_receipt', 'sent' if failure == 'no_receipt' else 'rejected',
                    definitive_not_sent=failure == 'rejected')
            return original_sender(message)
        bridge.sender = sender
        try:
            answer = bridge(turn('创建 '+str(path)+' 并发送给我。'))
            row = bridge.delegation._rows(SCOPE)[0]
            result = remote[4].port(SCOPE).result(row['task_id'])
            assert result['execution']['status'] != 'completed'
            assert len(physical) == 1
            with store._transaction() as db:
                status = db.execute('SELECT status FROM qq_file_deliveries').fetchone()[0]
            assert status == ('failed' if failure == 'rejected' else 'uncertain')
            assert not any('send' in r['receipt'].get('verified_effects', []) for r in result['evidence'])
        finally:
            close(store, coordinator)


@pytest.mark.parametrize('kind', ['symlink', 'fifo', 'oversized', 'relative'])
def test_snapshot_rejects_unsafe_or_unbounded_sources(tmp_path, kind):
    path = tmp_path / 'source.bin'
    if kind == 'symlink':
        target = tmp_path / 'target'; target.write_text('synthetic')
        path.symlink_to(target)
    elif kind == 'fifo':
        import os
        os.mkfifo(path)
    elif kind == 'oversized':
        with path.open('wb') as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
    else:
        path = Path('relative.bin')
    with pytest.raises((AgentBoundaryError, OSError)):
        read_snapshot(path)


def pending(client, task_id):
    deadline = time.monotonic() + 3
    while True:
        rows = client.port(SCOPE).pending_files(task_id)
        if rows:
            return rows[0]
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_pending_export_is_scoped_and_bad_receipt_cannot_complete_send(tmp_path):
    with file_remote(tmp_path, generate=False, outcome='failed') as (remote, broker, path):
        client = remote[4]
        handle = client.port(SCOPE).submit(AgentTurnRequest('scoped-file', '发送 "'+str(path)+'"'))
        item = pending(client, handle.task_id)
        with pytest.raises(AgentBoundaryError):
            client.port(AgentScope('other', 'private:fixture')).file_payload(handle.task_id, item['id'])
        with pytest.raises(AgentBoundaryError, match='receipt_invalid'):
            client.port(SCOPE).acknowledge_file(handle.task_id, item['id'],
                {'status': 'sent', 'message_ids': [], 'sha256': item['sha256']})
        payload = client.port(SCOPE).file_payload(handle.task_id, item['id'])
        assert base64.b64decode(payload['data']) == path.read_bytes()
        client.port(SCOPE).acknowledge_file(handle.task_id, item['id'],
            {'status': 'failed', 'message_ids': [], 'sha256': item['sha256']})
        assert settled(client, handle.task_id)['execution']['status'] != 'completed'


@pytest.mark.parametrize('text,required', [
    ('请创建 "/tmp/report.txt" 并作为文件发送给我。', True),
    ('创建 /tmp/report.txt 并发给我。', True),
    ('创建 "/tmp/report.txt"，然后发给我。', True),
    ('create "/tmp/report.txt" and send it to me', True),
    ('创建 /tmp/report.txt，不要发送给我。', False),
    ('请创建 "/tmp/report.txt" 并解释发送给我的流程。', False),
    ('请解释“创建report.txt并发给我”这句话。', False),
    ('创建 /tmp/report.txt 并发送 /tmp/other.txt。', False),
])
def test_create_only_contract_cannot_omit_explicit_following_send(text, required):
    from shinku.agent.verification import SmartCompletionVerifier
    contract = {'deliverables': ['/tmp/report.txt'], 'requirements': [
        {'effect': 'write', 'targets': ['/tmp/report.txt']},
        {'effect': 'send', 'targets': ['/tmp/other.txt']}]}
    gaps = SmartCompletionVerifier().target_coverage_gaps(contract, [text])
    missing_send = any(row['target'].endswith('report.txt') and 'send' in row['effects'] for row in gaps)
    assert missing_send is required


@pytest.mark.parametrize('quoted', [False, True])
def test_full_absolute_deliverable_with_spaces_is_not_split_into_false_targets(quoted):
    from shinku.agent.verification import SmartCompletionVerifier
    path = '/tmp/已有 文件.bin'
    contract = {'deliverables': [path], 'requirements': [{'effect': 'send', 'targets': [path]}]}
    literal = '"' + path + '"' if quoted else path
    assert not SmartCompletionVerifier().target_coverage_gaps(contract,
        ['请发送已有文件 '+literal+' 并发送附件给我。'])


def test_literal_binding_cannot_swallow_multiple_user_file_targets():
    from shinku.agent.verification import SmartCompletionVerifier
    contract = {'deliverables': ['/tmp/a.txt and /tmp/b.txt'], 'requirements': [
        {'effect': 'send', 'targets': ['/tmp/a.txt and /tmp/b.txt']}]}
    assert SmartCompletionVerifier().target_coverage_gaps(contract,
        ['请发送 /tmp/a.txt and /tmp/b.txt'])


@pytest.mark.parametrize('change', ['source', 'snapshot'])
def test_changed_source_or_stored_bytes_are_rejected_before_export(tmp_path, change):
    with file_remote(tmp_path, generate=False, outcome='failed') as (remote, broker, path):
        client = remote[4]
        handle = client.port(SCOPE).submit(AgentTurnRequest('changed-file', '发送 "'+str(path)+'"'))
        item = pending(client, handle.task_id)
        if change == 'source':
            path.write_bytes(b'changed after request')
        else:
            with remote[0].task_store._transaction() as db:
                db.execute('UPDATE agent_file_deliveries SET body=? WHERE id=?', (b'tampered snapshot', item['id']))
        with pytest.raises(AgentBoundaryError, match='file_delivery_.*changed'):
            client.port(SCOPE).file_payload(handle.task_id, item['id'])
        client.port(SCOPE).acknowledge_file(handle.task_id, item['id'],
            {'status': 'failed', 'message_ids': [], 'sha256': item['sha256']})
        assert settled(client, handle.task_id)['execution']['status'] != 'completed'


def test_caller_readonly_cannot_submit_file_for_delivery(tmp_path):
    with file_remote(tmp_path, generate=False, outcome='failed') as (remote, broker, path):
        client, base = remote[4], remote[5]
        denied = replace(base, allowed_tools=base.allowed_tools - {TOOL}, permission_mode='readonly')
        client.authority = lambda s, o: authority_payload(denied)
        assert not client.call(SCOPE, 'capabilities', {})['host_file_delivery']
        handle = client.port(SCOPE).submit(AgentTurnRequest('readonly-file', '发送 "'+str(path)+'"'))
        assert settled(client, handle.task_id)['execution']['status'] != 'completed'
        with remote[0].task_store._transaction() as db:
            assert db.execute('SELECT count(*) FROM agent_file_deliveries').fetchone()[0] == 0


def test_started_local_send_after_restart_is_unknown_and_never_sent_again(tmp_path):
    with file_remote(tmp_path, generate=False, outcome='failed') as (remote, broker, path):
        client = remote[4]
        handle = client.port(SCOPE).submit(AgentTurnRequest('restart-file', '发送 "'+str(path)+'"'))
        item = pending(client, handle.task_id)
        bridge, chat, messages, wire, store, coordinator = file_bridge(tmp_path, remote, broker)
        try:
            from dataclasses import asdict
            from shinku.agent_runtime.contracts import canonical_hash
            delivery = QQFileDelivery(bridge)
            message = turn('发送已有文件')
            scope_key = canonical_hash(asdict(SCOPE))
            binding = canonical_hash([item, scope_key, message.conversation_id, message.conversation_type])
            with store._transaction() as db:
                db.execute('INSERT INTO qq_file_deliveries VALUES (?,?,?,?,?,?)',
                    (item['id'], scope_key, handle.task_id, binding, 'started', '{}'))
            delivery.forward(SCOPE, message, client.port(SCOPE), handle.task_id, lambda: '')
            assert not wire and not messages
            with store._transaction() as db:
                assert db.execute('SELECT status FROM qq_file_deliveries').fetchone()[0] == 'uncertain'
            assert settled(client, handle.task_id)['execution']['status'] != 'completed'
        finally:
            close(store, coordinator)


def test_authenticated_caller_still_needs_claim_before_send_receipt(tmp_path):
    with file_remote(tmp_path, generate=False) as (remote, broker, path):
        client = remote[4]
        handle = client.port(SCOPE).submit(AgentTurnRequest('claim-file', '发送 "'+str(path)+'"'))
        item = pending(client, handle.task_id)
        receipt = {'status': 'sent', 'message_ids': ['synthetic-confirmed'], 'sha256': item['sha256']}
        with pytest.raises(AgentBoundaryError, match='not_claimed'):
            client.port(SCOPE).acknowledge_file(handle.task_id, item['id'], receipt)
        client.port(SCOPE).file_payload(handle.task_id, item['id'])
        client.port(SCOPE).acknowledge_file(handle.task_id, item['id'], receipt)
        assert client.port(SCOPE).acknowledge_file(handle.task_id, item['id'], receipt)['recorded']
        assert settled(client, handle.task_id)['execution']['status'] == 'completed'


@pytest.mark.parametrize('text,expected', [
    ('请生成 /tmp/report.txt 并发给我。', True),
    ('请发送文件 /tmp/report.txt', True),
    ('请发送已有文件 /tmp/report.txt 作为附件。', True),
    ('把Angelos的报告作为文件发给我。', True),
    ('请只创建 /tmp/report.txt。', False),
    ('请生成 /tmp/report.txt，不要发送给我。', False),
    ('解释“发送文件 /tmp/report.txt”这句话。', False),
    ('请创建 /tmp/report.txt，我会自己发送。', False),
    ('如果我说发给我，你会怎样做？', False),
    ('把这句话发给我。', False),
])
def test_model_contract_cannot_authorize_unrequested_file_delivery(text, expected):
    from shinku.agent_runtime.file_delivery import requested_file_send
    assert requested_file_send([{'text': text}], '/tmp/report.txt') is expected


@pytest.mark.parametrize('literal', ['/tmp/a.txt', '/tmp/已有 文件.bin'])
@pytest.mark.parametrize('wording', ['请发送文件 {path}', '请发送已有文件 {path} 作为附件。',
    '请创建 {path}，并把文件作为附件发送到当前QQ会话。'])
def test_explicit_file_send_authority_cannot_be_borrowed_for_another_path(literal, wording):
    from shinku.agent_runtime.file_delivery import requested_file_send
    original = [{'text': wording.format(path=literal)}]
    assert requested_file_send(original, literal)
    assert not requested_file_send(original, '/tmp/b.txt')


def test_separate_clause_file_delivery_cannot_complete_with_only_write_contract():
    from shinku.agent.verification import SmartCompletionVerifier
    path = '/synthetic/agent-demo.txt'
    contract = {'deliverables': [path], 'requirements': [{'effect': 'write', 'targets': [path]}]}
    gaps = SmartCompletionVerifier().target_coverage_gaps(contract,
        ['请在 /synthetic/agent-demo.txt 创建示例文件并验证内容。并把文件作为附件发送到当前QQ会话。'])
    assert any(row['reason'] == 'requested_file_delivery_missing' and row['effects'] == ['send'] for row in gaps)


def test_ordinary_text_reply_does_not_require_file_delivery():
    from shinku.agent.verification_targets import file_delivery_requested
    assert not file_delivery_requested(['把这句话发给我。'])
    assert not file_delivery_requested(['整理一段报告发给我。'])
    assert file_delivery_requested(['把这段报告作为文件发给我。'])

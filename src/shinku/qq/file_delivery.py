"""真红侧文件送达：固定原QQ目标，先持久占用，再一次调用现有发送器。

执行侧只能请求文件快照；目标由当前QQTurn派生。独立账本保留开始/确认/未知状态，
HTTP响应丢失或进程退出都不自动重发。文件正文仅在内存中进入发送器，不进入模型/追踪。
"""
import base64
from dataclasses import asdict
import hashlib
import json

from shinku.agent_runtime.contracts import AgentBoundaryError, canonical_hash
from shinku.agent_runtime.file_delivery import MAX_FILE_BYTES
from .delivery import DeliveryResult, OutgoingMessage, OutgoingSegment


class QQFileDelivery:
    def __init__(self, bridge):
        self.bridge = bridge
        with bridge.task_store._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS qq_file_deliveries
                (id TEXT PRIMARY KEY, scope TEXT NOT NULL, task TEXT NOT NULL,
                 metadata_hash TEXT NOT NULL, status TEXT NOT NULL, receipt TEXT NOT NULL)''')

    def forward(self, scope, turn, port, task_id, guard):
        bridge = self.bridge
        if turn.conversation_type not in {'private', 'group'}:
            raise AgentBoundaryError('file_delivery_qq_scope_required')
        from .agent_call import scope_for_turn
        if scope_for_turn(turn)[0] != scope:
            raise AgentBoundaryError('file_delivery_scope_changed')
        rows = port.pending_files(task_id)
        if not isinstance(rows, list) or len(rows) > 16:
            raise AgentBoundaryError('file_delivery_metadata_invalid')
        for metadata in rows:
            self._one(scope, turn, port, task_id, metadata, guard)

    def _one(self, scope, turn, port, task_id, metadata, guard):
        if (not isinstance(metadata, dict) or set(metadata) !=
                {'id', 'task_id', 'attempt', 'request_hash', 'path', 'name', 'sha256', 'bytes'}
                or metadata['task_id'] != task_id or type(metadata['attempt']) is not int
                or type(metadata['bytes']) is not int or not 0 <= metadata['bytes'] <= MAX_FILE_BYTES
                or not isinstance(metadata['id'], str) or not metadata['id'].startswith('file:')
                or not isinstance(metadata['sha256'], str) or len(metadata['sha256']) != 64
                or not isinstance(metadata['name'], str) or not 1 <= len(metadata['name']) <= 255
                or any(v in metadata['name'] for v in ('/', '\\', '\r', '\n', '\0'))):
            raise AgentBoundaryError('file_delivery_metadata_invalid')
        if guard():
            raise AgentBoundaryError('file_delivery_authorization_changed')
        view = port.status(task_id)
        if view.get('attempt') != metadata['attempt'] or view.get('status') == 'canceled':
            raise AgentBoundaryError('file_delivery_attempt_changed')
        identifier, scope_key = metadata['id'], canonical_hash(asdict(scope))
        digest = canonical_hash([metadata, scope_key, turn.conversation_id, turn.conversation_type])
        with self.bridge.task_store._transaction() as db:
            existing = db.execute('SELECT scope,task,metadata_hash,status,receipt FROM qq_file_deliveries WHERE id=?', (identifier,)).fetchone()
            if existing:
                if tuple(existing[:3]) != (scope_key, task_id, digest):
                    raise AgentBoundaryError('file_delivery_binding_conflict')
                receipt = json.loads(existing[4]) if existing[3] != 'started' else {
                    'status': 'uncertain', 'message_ids': [], 'sha256': metadata['sha256']}
                if existing[3] == 'started':
                    db.execute("UPDATE qq_file_deliveries SET status='uncertain',receipt=? WHERE id=?",
                        (json.dumps(receipt), identifier))
            else:
                if db.execute("SELECT 1 FROM qq_file_deliveries WHERE scope=? AND task=? AND status IN ('started','uncertain')",
                              (scope_key, task_id)).fetchone():
                    raise AgentBoundaryError('file_delivery_previous_effect_unknown')
                db.execute('INSERT INTO qq_file_deliveries VALUES (?,?,?,?,?,?)',
                    (identifier, scope_key, task_id, digest, 'started', '{}'))
                receipt = None
        if receipt is None:
            receipt = {'status': 'uncertain', 'message_ids': [], 'sha256': metadata['sha256']}
            attempted = False
            try:
                payload = port.file_payload(task_id, identifier)
                if ({k: v for k, v in payload.items() if k != 'data'} != metadata
                        or not isinstance(payload.get('data'), str)
                        or len(payload['data']) > 4 * ((MAX_FILE_BYTES + 2) // 3)):
                    raise AgentBoundaryError('file_delivery_payload_invalid')
                data = base64.b64decode(payload['data'], validate=True)
                if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
                    raise AgentBoundaryError('file_delivery_payload_changed')
                if guard():
                    raise AgentBoundaryError('file_delivery_authorization_changed')
                view = port.status(task_id)
                if view.get('attempt') != metadata['attempt'] or view.get('status') == 'canceled':
                    raise AgentBoundaryError('file_delivery_attempt_changed')
                if self.bridge.sender is None:
                    receipt['status'] = 'failed'
                else:
                    # base64快照解决Agent/NapCat路径不共享；文件名由已绑定的源文件派生。
                    message = OutgoingMessage(str(turn.conversation_id), (OutgoingSegment(kind='file',
                        url='base64://' + payload['data'], payload={'name': metadata['name']}),),
                        conversation_type=turn.conversation_type)
                    attempted = True
                    outcome = self.bridge.sender(message)
                    if isinstance(outcome, DeliveryResult):
                        ids = [str(v) for v in outcome.message_ids if str(v)]
                        receipt['message_ids'] = ids
                        if outcome.sent and ids:
                            receipt['status'] = 'sent'
                        elif outcome.definitive_not_sent and not ids:
                            receipt['status'] = 'failed'
            except Exception:
                # 没有独立未发送证据时保持未知；不把捕获异常当成安全重发依据。
                if not attempted:
                    receipt['status'] = 'failed'
            with self.bridge.task_store._transaction() as db:
                db.execute('UPDATE qq_file_deliveries SET status=?,receipt=? WHERE id=?',
                    (receipt['status'], json.dumps(receipt), identifier))
        port.acknowledge_file(task_id, identifier, receipt)
        self.bridge._trace('', 'file_delivery', status=receipt['status'], task_id=task_id,
                           sha256=metadata['sha256'], bytes=metadata['bytes'])

"""委托中的补充收件箱：先保存原文、请求停止，再在宿主查询时应用。

仅保存声明输入；不调模型、不执行工具、不自动续跑。Gateway持有服务锁并
完成身份校验后调用；应用时重验当前授权、执行轮次及worker停止回执。
"""
import json
from dataclasses import asdict

from .contracts import AgentBoundaryError, AgentSupplement, TaskHandle, canonical_hash


class TaskRevisions:
    def __init__(self, gateway):
        self.gateway = gateway
        self.store = gateway.service.task_store
        with self.store._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS agent_task_revisions (
                task_id TEXT NOT NULL, request_key TEXT NOT NULL, payload TEXT NOT NULL,
                digest TEXT NOT NULL, state TEXT NOT NULL, PRIMARY KEY(task_id,request_key))''')

    def _rows(self, task_id):
        with self.store._transaction() as db:
            rows = db.execute('SELECT request_key,payload,digest,state FROM agent_task_revisions WHERE task_id=? ORDER BY rowid',
                              (task_id,)).fetchall()
        result = []
        for key, raw, digest, state in rows:
            value = json.loads(raw)
            if canonical_hash(value) != digest:
                raise AgentBoundaryError('task_revision_integrity_failed')
            result.append((key, value, state))
        return result

    def accept(self, scope, task, arguments):
        gateway, service = self.gateway, self.gateway.service
        request = AgentSupplement(**arguments['request'])
        value = {k: arguments[k] for k in ('request', 'materials', 'material_gaps')}
        value['attempt'] = arguments['attempt']
        rows = self._rows(task['task_id'])
        old = next((v for k, v, _ in rows if k == request.request_key), None)
        if old is not None:
            if old != value:
                raise AgentBoundaryError('material_supplement_key_conflict')
        else:
            service._check_handle(task, TaskHandle(task['task_id'], arguments['attempt'], arguments['version']))
            if len(rows) >= 16 or any(state == 'pending' for _, _, state in rows):
                raise AgentBoundaryError('task_revision_pending_or_limit')
            gateway._decode_materials(arguments)  # 先校验声明形状，不能以停止动作替代输入校验。
            service._grant(scope, {**task, 'material_supplement_request': request.payload()}, 'supplement')
            view = service._status(scope, task['task_id'])
            if view['worker_active'] or view['status'] in {'queued', 'running'}:
                service._grant(scope, task, 'cancel')
            with self.store._transaction() as db:
                db.execute('INSERT INTO agent_task_revisions VALUES (?,?,?,?,?)',
                    (task['task_id'], request.request_key, json.dumps(value, ensure_ascii=False), canonical_hash(value), 'pending'))
        gateway.conversation.append(task, 'supplement_pending', request.clarification,
            key='revision:' + request.request_key, author='caller', data={'constraints': list(request.constraints)})
        self.drain(scope, task['task_id'])
        pending = any(state == 'pending' for _, _, state in self._rows(task['task_id']))
        return {'state': 'pending' if pending else 'applied',
                'handle': asdict(service._handle(service._get(scope, task['task_id'])))}

    def drain(self, scope, task_id):
        """查询/续接时推进已授权的输入交接；进程重启后仍可恢复收件记录。"""
        gateway, service = self.gateway, self.gateway.service
        for key, value, state in self._rows(task_id):
            if state != 'pending':
                continue
            task = service._get(scope, task_id)
            if task['execution_attempt'] != value['attempt']:
                raise AgentBoundaryError('task_revision_attempt_changed')
            request = AgentSupplement(**value['request'])
            service._grant(scope, {**task, 'material_supplement_request': request.payload()}, 'supplement')
            view = service._status(scope, task_id)
            if view['worker_active'] or view['status'] in {'queued', 'running'}:
                # 取消可能需要多次轮询才停止；取消请求不代表工具已回滚。
                if not task.get('cancel_requested') and task['status'] not in {'completed', 'failed', 'canceled'}:
                    service._cancel(scope, service._handle(task))
                return
            lease_key = (canonical_hash(asdict(scope)), task_id)
            old = gateway.materials.get(lease_key)
            gateway.materials[lease_key] = gateway._decode_materials(value)
            try:
                service._supplement(scope, service._handle(task), request, allow_terminal=True)
            finally:
                if old is None:
                    gateway.materials.pop(lease_key, None)
                else:
                    gateway.materials[lease_key] = old
            gateway.conversation.append(service._get(scope, task_id), 'supplement', request.clarification,
                key='supplement:' + key, author='caller', data={'constraints': list(request.constraints)})
            with self.store._transaction() as db:
                db.execute("UPDATE agent_task_revisions SET state='applied' WHERE task_id=? AND request_key=?", (task_id, key))

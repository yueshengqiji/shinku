"""会话输入、幂等提交和结果归档；不实现第二个 Agent 执行器。

先落盘输入，再在有界线程池准备任务，最后交给 AgentPort。刷新只读状态；
取消准备阶段会丢弃迟到的模型输出，取消执行阶段必须取得宿主停止回执。
重启不自动重放未完成的准备请求，避免重复扣费或外部副作用。
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import threading
import time

from shinku.agent_runtime import AgentRequest, AgentTurnRequest, TaskHandle, AgentSupplement
from shinku.agent_runtime.contracts import canonical_hash

ACTIVE = {'queued', 'running', 'planning', 'yielded', 'waiting_background'}


class Workbench:
    def __init__(self, store, host=None):
        self.store, self.host = store, host
        from .files import WorkspaceFiles
        self.files = WorkspaceFiles(store, getattr(host, "permissions", None))
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='angelos-input')
        self.stop = threading.Event()
        self.monitor = None
        bind = getattr(host, 'bind_workbench', None)
        if callable(bind):
            bind(store)
        # 初始化仅标记未持有执行回执的输入；已提交的任务从 Agent 账本重新观察。
        for c in store.list('conversation'):
            for turn in c['turns']:
                if turn['stage'] in {'preparing', 'submitting'}:
                    turn['stage'] = 'executing' if turn.get('task_id') else 'interrupted'
                    turn['error'] = 'preparation_interrupted' if not turn.get('task_id') else ''
            with store.transaction() as db:
                store.put(db, 'conversation', c)

    def model_options(self, cid=None):
        manager = getattr(self.host, 'session_models', None)
        if cid and manager:
            if manager.store is None:
                self.host.bind_workbench(self.store)
            return manager.snapshot(cid)
        return self.host.model_options() if hasattr(self.host, 'model_options') else {'available': False, 'providers': [], 'selected': {}}

    def select_model(self, cid, data):
        manager = getattr(self.host, 'session_models', None)
        if manager is None:
            raise ValueError('conversation_models_unavailable')
        # 同会话的准备、派发与切换互斥；别的会话运行不阻塞这次选择。
        with self.lock, self.host.service._lock:
            c = self.refresh(cid)
            if any(t['stage'] in {'preparing', 'submitting', 'executing'} for t in c['turns']):
                raise ValueError('conversation_busy')
            for t in c['turns']:
                row = self.host.service.coordinator.get(t.get('task_id', '')) or {}
                if row.get('worker_active'):
                    raise ValueError('conversation_busy')
            return manager.select(cid, data)

    def start(self):
        if self.monitor is None:
            self.monitor = threading.Thread(target=self._monitor, name='angelos-results', daemon=True)
            self.monitor.start()

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _monitor(self):
        while not self.stop.wait(2):
            for conversation in self.store.list('conversation'):
                if any(t['stage'] == 'executing' or t.get('question') or t.get('view', {}).get('status') == 'waiting_user' for t in conversation['turns']):
                    try:
                        self.refresh(conversation['id'])
                    except Exception:
                        # 查询失败不改写执行事实，页面下次读取仍可得到明确错误。
                        continue

    def port(self, key):
        if self.host is None:
            raise ValueError('agent_unavailable')
        return self.host.port(key)

    def _turn(self, conversation, key):
        return next(t for t in conversation['turns'] if t['id'] == key)

    def update_turn(self, conversation_id, key, **fields):
        return self.store.change('conversation', conversation_id,
                                 lambda c: self._turn(c, key).update(fields))

    def edit_conversation(self, cid, data):
        """会话迁移只影响未来输入；已执行任务的原约束和回执保持不变。

        与发送共用工作台锁，并在同一事务校验目标工作区，禁止准备/执行中
        偷换约束。旧任务恢复时另核对当前工作区，不能把迁移当成扩大旧授权。
        """
        if set(data) - {'title', 'draft', 'workspaceId'} or any(not isinstance(v, str) for v in data.values()):
            raise ValueError('invalid_conversation')
        if len(data.get('title', '')) > 80 or ('title' in data and not data['title'].strip()) or len(data.get('draft', '')) > 12000:
            raise ValueError('invalid_conversation')
        with self.lock, self.store.transaction() as db:
            c = self.store.get(db, 'conversation', cid)
            if 'workspaceId' in data and data['workspaceId'] != c['workspaceId']:
                if any(t['stage'] in {'preparing', 'submitting', 'executing'} for t in c['turns']):
                    raise ValueError('conversation_busy')
                self.store.get(db, 'workspace', data['workspaceId'])
            c.update(data)
            c['updated'] = time.time()
            self.store.put(db, 'conversation', c)
            return c

    @staticmethod
    def workspace_constraints(workspace):
        return tuple(s for s in ('工作区目录：' + workspace['path'] if workspace['path'] else '',
                                workspace['description']) if s)

    def remove_workspace(self, wid):
        """移除工作区登记，历史会话归入默认区；不访问或删除任何项目文件。

        与发送/迁移共用工作台锁，校验和迁移放在同一事务。只改会话归属，保留
        task绑定、旧目录快照、附件和回执；恢复旧任务仍由原目录约束校验阻止换范围。
        """
        with self.lock, self.store.transaction() as db:
            self.store.get(db, 'workspace', wid)
            if wid == 'default':
                raise ValueError('workspace_default_required')
            self.store.get(db, 'workspace', 'default')
            conversations = [json.loads(row[0]) for row in db.execute("SELECT payload FROM items WHERE kind='conversation'")]
            affected = [c for c in conversations if c['workspaceId'] == wid]
            if any(t['stage'] in {'preparing', 'submitting', 'executing'} for c in affected for t in c['turns']):
                raise ValueError('workspace_busy')
            for c in affected:
                c['workspaceId'] = 'default'
                c['updated'] = time.time()
                self.store.put(db, 'conversation', c)
            db.execute("DELETE FROM items WHERE kind='workspace' AND id=?", (wid,))
            return {'workspaceId': wid, 'destination': 'default', 'movedConversations': len(affected)}

    def remove_conversation(self, cid):
        """只删工作台会话与附件，保留工作区/文件/任务审计/用量。

        与发送和刷新共锁；准备线程仍存活或宿主状态不明确时拒绝删除。
        先撤回关联记忆，再删来源；跨库失败只允许重试，不静默报告成功。
        """
        with self.lock:
            c = self.store.read('conversation', cid)
            if any(t['stage'] in {'preparing', 'submitting', 'executing'} for t in c['turns']):
                raise ValueError('conversation_busy')
            if any(t.get('question') or t.get('view', {}).get('status') == 'waiting_user' for t in c['turns']):
                raise ValueError('conversation_waiting_answer')
            if self.host is not None:
                for turn in c['turns']:
                    if turn.get('task_id'):
                        try:
                            view = self.port(cid).status(turn['task_id'])
                        except ValueError as exc:
                            # 已过期任务可以删展示记录；其他授权/读取错误不能当作空闲。
                            if getattr(exc, 'code', str(exc)) != 'task_not_found':
                                raise
                            continue
                        if view.get('available') and view.get('status') in ACTIVE | {'starting', 'canceling'}:
                            raise ValueError('conversation_busy')
            memory = getattr(self.host, 'memory', None)
            if memory is not None:
                memory.store.forget_session(memory.requester, 'angelos:' + cid)
            with self.store.transaction() as db:
                for row in db.execute("SELECT id,payload FROM items WHERE kind='attachment'").fetchall():
                    if json.loads(row[1])['conversationId'] == cid:
                        db.execute("DELETE FROM items WHERE kind='attachment' AND id=?", (row[0],))
                db.execute("DELETE FROM items WHERE kind='conversation' AND id=?", (cid,))
            manager = getattr(self.host, 'session_models', None)
            if manager:
                manager.forget(cid)
            return {'conversationId': cid, 'workspaceId': c['workspaceId']}

    def refresh(self, key):
        with self.lock:
            c = self.store.read('conversation', key)
            changed = False
            for turn in c['turns']:
                reader = getattr(self.host, 'read_reasoning', None)
                reasoning = reader(key, turn) if callable(reader) else []
                if reasoning and reasoning != turn.get('reasoning'):
                    turn['reasoning'] = reasoning
                    changed = True
                output_reader = getattr(self.host, 'read_outputs', None)
                outputs = output_reader(key, turn) if callable(output_reader) else []
                if outputs and outputs != turn.get('model_outputs'):
                    turn['model_outputs'] = outputs
                    changed = True
                context_status = getattr(self.host, 'context_status', None)
                if callable(context_status):
                    measured = context_status(key, turn)
                    if measured and measured != turn.get('context_usage'):
                        turn['context_usage'] = measured
                        changed = True
                if turn['stage'] != 'executing' and not turn.get('question') and turn.get('view', {}).get('status') != 'waiting_user':
                    continue
                try:
                    port = self.port(key)
                    view = port.status(turn['task_id'])
                    if view.get('available') is not True:
                        turn['read_error'] = view.get('reason', 'task_unavailable')
                        changed = True
                        continue
                    turn.pop('read_error', None)
                    turn['view'] = view
                    turn['result'] = port.result(turn['task_id'])
                    from shinku.agent_runtime.questions import question_for
                    question = (port.question(turn['task_id']) if hasattr(port, 'question') else question_for(turn['result'])) if view['status'] == 'waiting_user' and not view.get('worker_active') else None
                    if question:
                        # 原问句按结果身份留档；不因回答/新一轮提问覆盖旧原文。
                        history = turn.setdefault('questions', [])
                        if not any(q['id'] == question['id'] for q in history):
                            history.append(question)
                        question = dict(question, deferred=turn.get('deferred_question') == question['id'])
                    turn['question'] = question
                    if view['status'] not in ACTIVE and view.get('worker_active') is False:
                        turn['stage'] = 'finished'
                    changed = True
                except Exception as exc:
                    turn['read_error'] = getattr(exc, 'code', 'task_read_failed')
                    if turn['read_error'] == 'task_not_found':
                        turn['stage'] = 'interrupted'
                    changed = True
            if changed:
                with self.store.transaction() as db:
                    # 持有工作台锁；所有输入、控制及轮询写入都使用同一锁，避免覆盖新消息。
                    self.store.put(db, 'conversation', c)
            capture = getattr(self.host, 'capture_memory', None)
            indexed = False
            if callable(capture):
                for turn in c['turns'][-8:]:
                    if turn.get('stage') != 'finished' or not turn.get('result'):
                        continue
                    version = canonical_hash([turn['result'], turn.get('workspace_snapshot')])
                    if turn.get('memory_index_version') == version:
                        continue
                    try:
                        receipt = capture(key, turn)
                        if receipt is not None:
                            turn['memory_index_version'] = version
                            turn.pop('memory_error', None)
                            indexed = True
                    except Exception as exc:
                        error = getattr(exc, 'code', 'agent_memory_index_unavailable')
                        if turn.get('memory_error') != error:
                            turn['memory_error'] = error
                            indexed = True
                if indexed:
                    # 副索引失败保留重试入口，不改变任务完成、送达或原始回执。
                    with self.store.transaction() as db:
                        self.store.put(db, 'conversation', c)
            return c

    def send(self, conversation_id, payload):
        text = payload.get('text', '')
        key = payload.get('request_key', '')
        attachment_ids = payload.get('attachments', [])
        if (not isinstance(text, str) or not text.strip() or len(text) > 12000
                or not isinstance(key, str) or not 1 <= len(key) <= 128
                or not isinstance(attachment_ids, list) or len(attachment_ids) > 8
                or any(not isinstance(a, str) for a in attachment_ids)):
            raise ValueError('invalid_message')
        digest = canonical_hash([text, attachment_ids])
        with self.lock:
            c = self.refresh(conversation_id)
            duplicate = next((t for t in c['turns'] if t['id'] == key), None)
            if duplicate:
                if duplicate['digest'] != digest:
                    raise ValueError('idempotency_key_conflict')
                return c
            if any(t['stage'] in {'preparing', 'submitting', 'executing'} for t in c['turns']):
                raise ValueError('conversation_busy')
            if self.host is None or not self.host.ready():
                raise ValueError('agent_unavailable')
            if sum(t['stage'] == 'preparing' for row in self.store.list('conversation') for t in row['turns']) >= 4:
                raise ValueError('input_queue_full')
            sources = []
            for aid in attachment_ids:
                a = self.store.read('attachment', aid)
                if a['conversationId'] != conversation_id:
                    raise ValueError('attachment_scope_mismatch')
                sources.append({'id': a['id'], 'content': a['content'], 'reference': a['name']})
            # 原对话仍在工作台；背景单独交接，模型窗口由最终SDK输入的摘要管理器核对。
            if c['turns']:
                from .conversation_context import conversation_source
                sources.append(conversation_source(c['turns'], include_ids=getattr(self.host, 'memory', None) is not None))
            if any(len(s['content']) > 12000 for s in sources if s['id'] != 'conversation-history'):
                raise ValueError('conversation_context_full')
            if any(len(s['content'].encode('utf-8')) > 16 * 1024 * 1024 for s in sources):
                raise ValueError('conversation_history_resource_limit')
            workspace = self.store.read('workspace', c['workspaceId'])
            from .workspace_instructions import read_instructions
            instructions = read_instructions(workspace)
            if len(json.dumps([text, [s for s in sources if s['id'] != 'conversation-history'], workspace], ensure_ascii=False).encode('utf-8')) > 48000:
                raise ValueError('execution_input_oversized')
            selected = self.model_options(conversation_id).get('selected', {})
            turn = dict(id=key, digest=digest, text=text, attachments=attachment_ids,
                        stage='preparing', created=time.time(), model=selected.get('modelId', self.host.model),
                        workspace_snapshot=dict(workspace))
            turn['workspace_instructions'] = instructions
            if hasattr(self.host, 'model_options'):
                turn['model_settings'] = selected
            if getattr(self.host, 'session_models', None):
                c['modelSettings'] = selected
            c['turns'].append(turn)
            if len(c['turns']) == 1 and c['title'] == '新会话':
                c['title'] = ' '.join(text.split())[:36]
            c['draft'] = ''
            c['updated'] = time.time()
            with self.store.transaction() as db:
                self.store.put(db, 'conversation', c)
            self.pool.submit(self._prepare, conversation_id, key, text, sources, workspace)
            return c

    def _prepare(self, cid, key, text, sources, workspace):
        try:
            bind = getattr(self.host, 'bind_workbench', None)
            if callable(bind):
                bind(self.store)
            scope = getattr(self.host, 'observation_scope', None)
            from shinku.llm.context_compaction import continuation_guard
            def may_continue():
                current = self._turn(self.store.read('conversation', cid), key)
                return not self.stop.is_set() and current['stage'] == 'preparing' and self.host.ready()
            with (scope(cid, key) if callable(scope) else nullcontext()), continuation_guard(may_continue):
                # 正式独立宿主只做本机输入校验，首个模型请求属于原执行循环，
                # 普通回答、取证、契约声明及行动共用同一预算/任务/原生历史。
                prepared = ({'single_loop': True, 'preparation_usage': {'requests': 0, 'tools': 0,
                    'profile': 'local_input_only'}} if getattr(self.host, 'single_loop', False)
                    else self.host.prepare(text, sources, workspace))
            with self.lock:
                turn = self._turn(self.store.read('conversation', cid), key)
                if self.stop.is_set() or turn['stage'] != 'preparing':
                    return
                # 工作台拥有留存；取消/停机后的迟到结果不可写入，后续提交失败仍保留原文。
                if prepared.get('preparation_output'):
                    self.update_turn(cid, key, preparation_output=prepared['preparation_output'])
                if prepared.get('preparation_review'):
                    self.update_turn(cid, key, preparation_review=prepared['preparation_review'])
                if prepared.get('reply'):
                    self.update_turn(cid, key, stage='replied', reply=prepared['reply'],
                                     preparation_usage=prepared.get('preparation_usage', {}))
                    return
                constraints = self.workspace_constraints(workspace)
                history = next((s['content'] for s in sources if s['id'] == 'conversation-history'), '')
                arguments = dict(constraints=constraints,
                    sources=tuple(s for s in sources if s['id'] != 'conversation-history'), history_context=history)
                request = (AgentTurnRequest(key, text, **arguments) if prepared.get('single_loop')
                    else AgentRequest(key, text, prepared['contract'], **arguments))
                request.payload()  # 在任何执行提交前核对完整输入大小。
                port = self.port(cid)
                self.update_turn(cid, key, stage='submitting', task_id=self.host.task_id(cid, key),
                                 preparation_usage=prepared.get('preparation_usage', {}))
                # 提交和记录回执处于工作台锁内；AgentService 自身按 key 幂等，不在 SQLite 事务里调用。
                handle = port.submit(request)
                self.update_turn(cid, key, task_id=handle.task_id, stage='executing')
        except Exception as exc:
            with self.lock:
                current = self._turn(self.store.read('conversation', cid), key)
                if current['stage'] not in {'canceled', 'replied'}:
                    from .host import PreparationFailure
                    retained = ({'preparation_usage': exc.usage, 'preparation_failure': exc.failure}
                                if isinstance(exc, PreparationFailure) else {})
                    # 提交结果不明确时保留确定的任务编号，查询原任务而不是自动再次执行。
                    self.update_turn(cid, key, stage='executing' if current.get('task_id') else 'failed',
                                     error=getattr(exc, 'code', 'input_preparation_failed'), **retained)

    def control(self, cid, key, action, payload):
        with self.lock:
            c = self.refresh(cid)
            turn = self._turn(c, key)
            if action == 'cancel' and turn['stage'] == 'preparing':
                return self.update_turn(cid, key, stage='canceled', error='preparation_canceled')
            if not turn.get('task_id'):
                raise ValueError('task_not_submitted')
            port = self.port(cid)
            view = port.status(turn['task_id'])
            if not view.get('available'):
                raise ValueError('task_unavailable')
            if action in {'resume', 'supplement', 'answer'}:
                selection = self.model_options(cid).get('selected')
                selected_model = (selection or {}).get('modelId', self.host.model)
                model_changed = bool(turn.get('model') and turn['model'] != selected_model or
                    turn.get('model_settings') and turn['model_settings'] != selection)
                # 只有宿主明确启用持久化恢复/新循环时可以换模型接续；旧检查点
                # 不能跨供应商使用。真实副作用、预算与目录仍由原恢复门核对。
                can_replan = (action in {'resume', 'answer'} and getattr(self.host, 'single_loop', False) is True
                    and getattr(getattr(self.host, 'service', None), 'recovery_enabled', False) is True)
                if model_changed and not can_replan:
                    raise ValueError('task_model_changed')
                workspace = self.store.read('workspace', c['workspaceId'])
                current_constraints = self.workspace_constraints(workspace)
                # 对旧记录也用宿主结果中的原约束核对，不从模型正文猜测权限。
                original = port.result(turn['task_id']).get('constraints')
                if original is None or tuple(original) != current_constraints:
                    raise ValueError('task_workspace_changed')
            if action in {'answer', 'resume'} and any(t['id'] != key and t['stage'] in {'preparing', 'submitting', 'executing'} for t in c['turns']):
                raise ValueError('conversation_busy')
            # 取消允许取最新版本；恢复/补充必须校对用户看到的版本，不能续接另一轮执行。
            handle = TaskHandle(turn['task_id'], view['attempt'], view['version'])
            if action not in {'cancel', 'answer'} and (payload.get('attempt'), payload.get('version')) != (handle.attempt, handle.version):
                raise ValueError('task_version_conflict')
            if action == 'cancel':
                port.cancel(handle)
            elif action == 'resume':
                if turn.get('question'):
                    raise ValueError('question_answer_required')
                resumed = port.resume(handle, request_key=payload.get('request_key', ''))
            elif action == 'answer':
                # 原UI句柄交给服务端核对；同答案重试可恢复已保存补充，不能在此
                # 用最新句柄代替用户看到的版本。服务端与取消共锁并保留恢复验收。
                submitted = TaskHandle(turn['task_id'], payload.get('attempt'), payload.get('version'))
                try:
                    resumed = port.answer_question(submitted, question_id=payload.get('question_id'), text=payload.get('text'))
                except Exception as exc:
                    # 恢复拒绝不能吞掉原错误；只记展示诊断，不改宿主执行状态。
                    self.update_turn(cid, key, question_error=getattr(exc, 'code', 'question_resume_failed'),
                                     question_error_id=payload.get('question_id'))
                    raise
                self.update_turn(cid, key, question_error='', question_error_id='')
                supplements = list(turn.get('supplements', []))
                if not any(s['id'] == payload['question_id'] for s in supplements):
                    supplements.append({'id': payload['question_id'], 'text': payload['text'], 'created': time.time()})
                self.update_turn(cid, key, supplements=supplements)
            elif action in {'defer', 'undefer'}:
                question = turn.get('question')
                if not question or question['id'] != payload.get('question_id') or view['status'] != 'waiting_user':
                    raise ValueError('question_not_current')
                self.update_turn(cid, key, deferred_question=question['id'] if action == 'defer' else '')
                return self.refresh(cid)
            elif action == 'supplement':
                port.supplement(handle, AgentSupplement(payload.get('request_key', ''), payload.get('text', '')))
                supplements = list(turn.get('supplements', []))
                if not any(s['id'] == payload.get('request_key') for s in supplements):
                    supplements.append({'id': payload['request_key'], 'text': payload['text'], 'created': time.time()})
                self.update_turn(cid, key, supplements=supplements)
            else:
                raise ValueError('unknown_action')
            if action in {'resume', 'answer'} and model_changed:
                # 执行器接受新尝试后才更新展示；保存原提交身份与历次切换，
                # 接口拒绝或撤权不能把旧任务伪装成已由新模型运行。
                history = [*turn.get('model_transitions', []), {
                    'from': {'model': turn.get('model'), 'settings': turn.get('model_settings'), 'attempt': handle.attempt},
                    'to': {'model': selected_model, 'settings': selection, 'attempt': resumed.attempt}}]
                self.update_turn(cid, key, submitted_model=turn.get('submitted_model', turn.get('model')),
                    submitted_model_settings=turn.get('submitted_model_settings', turn.get('model_settings')),
                    model=selected_model, model_settings=selection, model_transitions=history)
            self.update_turn(cid, key, stage='executing')
            return self.refresh(cid)

    def artifact(self, cid, turn_id, index):
        c = self.refresh(cid)
        turn = self._turn(c, turn_id)
        artifacts = turn.get('result', {}).get('artifacts', [])
        if type(index) is not int or index < 0 or index >= len(artifacts):
            raise ValueError('artifact_unavailable')
        from .artifact_file import read_artifact
        return read_artifact(artifacts[index], turn.get('workspace_snapshot') or {})

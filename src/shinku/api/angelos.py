"""独立 Angelos 页面与本机 API；主控制中枢不参与会话交互。

读写均校验本机 Host/来源；写操作额外使用每进程令牌与 JSON 请求，阻断
其他网站向本机端口提交任务。身份绑定在宿主适配层，不接受浏览器冒充账号。
"""
import base64
import binascii
import json
import secrets
import uuid
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from shinku.angelos.store import WorkbenchStore
from shinku.angelos.workbench import Workbench
from shinku.angelos.host import host_for_assembly
from shinku.angelos.native_picker import NativePicker
from shinku.frontend import angelos_root

LOCAL = {'localhost', '127.0.0.1', '::1', 'testclient', 'testserver'}


def build_angelos_router(settings, assembly=None, *, host=None):
    store = WorkbenchStore(settings.data_root / 'angelos' / 'workbench.sqlite3')
    workbench = Workbench(store, host if host is not None else host_for_assembly(assembly))
    workbench.native_picker = NativePicker()
    token = secrets.token_urlsafe(32)
    router = APIRouter()

    def guard(request, write=False):
        if request.client is None or request.client.host not in LOCAL or request.url.hostname not in LOCAL:
            raise PermissionError('local_request_required')
        if request.headers.get('sec-fetch-site') == 'cross-site':
            raise PermissionError('same_origin_required')
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            raise PermissionError('same_origin_required')
        if write and (not secrets.compare_digest(request.headers.get('x-angelos-token', ''), token)
                      or request.headers.get('content-type', '').split(';')[0] != 'application/json'):
            raise PermissionError('session_token_required')

    async def body(request, limit=96000):
        guard(request, True)
        # 逐块限制请求，而非先把任意大的上传读入内存。
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > limit:
                raise ValueError('request_too_large')
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('invalid_request')
        return data

    def response(data):
        return JSONResponse({'ok': True, **data}, headers={'Cache-Control': 'no-store'})

    def failure(exc):
        code = getattr(exc, 'code', None) or (str(exc) if isinstance(exc, (ValueError, PermissionError)) else 'request_failed')
        if len(code) > 120 or ' ' in code:
            code = 'invalid_request'
        return JSONResponse({'ok': False, 'reason': code}, status_code=403 if isinstance(exc, PermissionError) else 409,
                            headers={'Cache-Control': 'no-store'})

    @router.get('/angelos', response_model=None)
    @router.get('/angelos/', response_model=None)
    def page(request: Request):
        try:
            guard(request)
            return FileResponse(angelos_root() / 'index.html', headers={
                'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/assets/{name}', response_model=None)
    def asset(name: str, request: Request):
        try:
            guard(request)
            if name not in {'app.js', 'app.css', 'context.js', 'layout.js', 'usage.js', 'activity.js', 'log-view.js', 'workspace.js', 'sessions.js', 'editor.js', 'editor.js.LEGAL.txt'}:
                return Response(status_code=404)
            return FileResponse(angelos_root() / name, headers={'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff'})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/maintenance', response_model=None)
    def maintenance_status(request: Request):
        """只读维护状态；网页不能配置、授权或启动维护任务。"""
        try:
            guard(request)
            maintenance = getattr(workbench.host, 'maintenance', None)
            return response({'maintenance': maintenance.snapshot() if maintenance is not None else {'configured': False}})
        except Exception as exc:
            return failure(exc)

    def latest_task(c):
        # 只投影账本归档状态，不解析正文中的完成/失败关键词。
        if not c['turns']:
            return None
        return task_summary(c['turns'][-1])

    def task_summary(t):
        return {'id': t['id'], 'attempt': (t.get('view') or {}).get('attempt', 1),
                'stage': t['stage'], 'status': (t.get('view') or {}).get('status', t['stage'])}

    @router.get('/angelos/api/bootstrap', response_model=None)
    def bootstrap(request: Request):
        try:
            guard(request)
            conversations = []
            for c in store.list('conversation'):
                # 列表保留可搜索的用户文字，不把所有工具结果重复发送。
                conversations.append({k: c[k] for k in ('id', 'workspaceId', 'title', 'draft', 'updated')}
                    | {'search': '\n'.join(t['text'] for t in c['turns']),
                       # 只投影宿主阶段，侧栏无需拉取每个会话正文或调用模型。
                       'latestTask': latest_task(c),
                       'taskStates': [task_summary(t) for t in c['turns'][-50:]],
                       'modelSettings': c.get('modelSettings'),
                       'waiting': [dict(t['question'], turnId=t['id']) for t in c['turns'] if t.get('question')],
                       'activity': next(({'stage': t['stage'], 'status': (t.get('view') or {}).get('status', t['stage'])}
                           for t in reversed(c['turns']) if t['stage'] in {'preparing', 'submitting', 'executing'}), None)})
            return response({'token': token, 'workspaces': store.list('workspace'), 'conversations': conversations,
                'agent': {'workbenchControls': True, 'nativePicker': workbench.native_picker.available,
                          'usageStats': hasattr(workbench.host, 'read_usage'),
                          'available': workbench.host is not None,
                          'ready': bool(workbench.host and workbench.host.ready()),
                          'model': workbench.host.model if workbench.host else '未连接 Agent'}})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/native-picker', response_model=None)
    async def native_picker(request: Request):
        try:
            data = await body(request)
            if set(data) != {'kind', 'workspaceId'} or data['kind'] not in ('file', 'directory') or not isinstance(data['workspaceId'], str):
                raise ValueError('invalid_picker_request')
            # 起点只取已保存的工作区；选择结果仍须经文件 API 或工作区保存校验。
            directory = store.read('workspace', data['workspaceId'])['path'] if data['workspaceId'] else ''
            if data['kind'] == 'file' and not directory:
                raise ValueError('native_picker_workspace_required')
            return response(await workbench.native_picker.choose(data['kind'], directory, request.is_disconnected))
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/workspaces', response_model=None)
    @router.post('/angelos/api/workspaces/{workspace_id}', response_model=None)
    async def workspace(request: Request, workspace_id: str = ''):
        try:
            data = await body(request)
            with workbench.lock:
                return response({'workspace': store.workspace(data, workspace_id or None)})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/workspaces/{workspace_id}/delete', response_model=None)
    async def remove_workspace(workspace_id: str, request: Request):
        try:
            data = await body(request)
            if set(data) != {'confirm'} or data['confirm'] is not True:
                raise ValueError('workspace_delete_confirmation_required')
            return response({'removed': workbench.remove_workspace(workspace_id)})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/models', response_model=None)
    def models(request: Request, conversationId: str = ''):
        try:
            guard(request)
            return response({'models': workbench.model_options(conversationId or None)})
        except PermissionError as exc:
            return failure(exc)
        except Exception:
            return failure(ValueError('model_catalog_unavailable'))

    @router.get('/angelos/api/workspaces/{wid}/instructions', response_model=None)
    def workspace_instructions(wid: str, request: Request):
        try:
            guard(request)
            from shinku.angelos.workspace_instructions import read_instructions
            return response({'instructions': read_instructions(store.read('workspace', wid))})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations/{cid}/delete', response_model=None)
    async def remove_conversation(cid: str, request: Request):
        try:
            data = await body(request)
            if set(data) != {'confirm'} or data['confirm'] is not True:
                raise ValueError('conversation_delete_confirmation_required')
            return response({'removed': workbench.remove_conversation(cid)})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/permissions', response_model=None)
    def permissions(request: Request):
        try:
            guard(request)
            controls = getattr(workbench.host, 'permissions', None)
            return response({'permissions': controls.snapshot() if controls else {'available': False}})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/permissions', response_model=None)
    async def select_permissions(request: Request):
        try:
            data = await body(request)
            controls = getattr(workbench.host, 'permissions', None)
            if controls is None:
                raise ValueError('permission_controls_unavailable')
            if set(data) != {'level', 'revision'}:
                raise ValueError('invalid_permission_selection')
            return response({'permissions': controls.select(data['level'], data['revision'])})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/approvals/{identifier}', response_model=None)
    async def approve(identifier: str, request: Request):
        try:
            data = await body(request)
            controls = getattr(workbench.host, 'permissions', None)
            if controls is None or set(data) != {'revision', 'decision'}:
                raise ValueError('invalid_approval_decision')
            return response({'permissions': controls.decide(identifier, data['revision'], data['decision'])})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/workspaces/{wid}/changes', response_model=None)
    def changes(wid: str, request: Request):
        try:
            guard(request)
            return response({'changes': workbench.files.changes(wid)})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/workspaces/{wid}/file', response_model=None)
    def read_file(wid: str, path: str, request: Request):
        try:
            guard(request)
            return response({'file': workbench.files.open(wid, path)})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/workspaces/{wid}/file', response_model=None)
    async def save_file(wid: str, request: Request):
        try:
            data = await body(request, limit=4 * 1024 * 1024)
            if set(data) != {'path', 'content', 'version'}:
                raise ValueError('invalid_file_edit')
            return response({'file': workbench.files.save(wid, data['path'], data['content'], data['version'])})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/models/refresh', response_model=None)
    async def refresh_models(request: Request):
        try:
            data = await body(request)
            if set(data) != {'providerId'} or not isinstance(data['providerId'], str):
                raise ValueError('invalid_model_selection')
            if not getattr(workbench.host, 'models', None):
                raise ValueError('model_controls_unavailable')
            import asyncio
            result = await asyncio.to_thread(update_capacity, data, refresh=True)
            return response({'models': result})
        except PermissionError as exc:
            return failure(exc)
        except ValueError as exc:
            return failure(ValueError('model_change_busy' if str(exc) == 'model_change_busy' else 'model_discovery_failed'))
        except Exception:
            return failure(ValueError('model_discovery_failed'))

    def update_capacity(data, *, refresh=False):
        with workbench.lock:
            if any(t['stage'] in {'preparing','submitting','executing'} for c in store.list('conversation') for t in c['turns']):
                raise ValueError('model_change_busy')
            return workbench.host.update_model_capacity(data, refresh=refresh)

    @router.post('/angelos/api/models/context-window', response_model=None)
    async def context_window(request: Request):
        try:
            data = await body(request)
            import asyncio
            result = await asyncio.to_thread(update_capacity, data)
            return response({'models': result})
        except PermissionError as exc:
            return failure(exc)
        except Exception as exc:
            allowed = {'invalid_context_window', 'model_change_busy', 'model_not_discovered',
                       'provider_not_configured', 'model_controls_unavailable'}
            return failure(ValueError(str(exc) if str(exc) in allowed else 'model_capacity_update_failed'))

    @router.post('/angelos/api/models/select', response_model=None)
    async def select_model(request: Request):
        try:
            data = await body(request)
            with workbench.lock:
                if any(t['stage'] in {'preparing','submitting','executing'} for c in store.list('conversation') for t in c['turns']):
                    raise ValueError('model_change_busy')
                if not hasattr(workbench.host, 'select_model'):
                    raise ValueError('model_controls_unavailable')
                return response({'models': workbench.host.select_model(data)})
        except PermissionError as exc:
            return failure(exc)
        except ValueError as exc:
            allowed = {'model_controls_unavailable','model_change_busy','invalid_model_selection',
                'provider_not_configured','model_not_discovered','reasoning_effort_unsupported','model_native_tools_unsupported'}
            return failure(ValueError(str(exc) if str(exc) in allowed else 'model_selection_failed'))
        except Exception:
            return failure(ValueError('model_selection_failed'))

    @router.post('/angelos/api/conversations/{cid}/model', response_model=None)
    async def conversation_model(cid: str, request: Request):
        try:
            data = await body(request)
            return response({'models': workbench.select_model(cid, data)})
        except PermissionError as exc:
            return failure(exc)
        except Exception as exc:
            allowed = {'record_not_found', 'conversation_busy', 'invalid_model_selection', 'model_not_discovered',
                       'provider_not_configured', 'reasoning_effort_unsupported', 'model_native_tools_unsupported',
                       'conversation_models_unavailable'}
            return failure(ValueError(str(exc) if str(exc) in allowed else 'model_selection_failed'))

    @router.get('/angelos/api/conversations/{cid}/usage', response_model=None)
    def usage(cid: str, request: Request):
        try:
            guard(request)
            # 单独低频读取，不刷新任务、不启动模型、不写入会话/记忆上下文。
            with workbench.lock:
                conversation = store.read('conversation', cid)
            reader = getattr(workbench.host, 'read_usage', None)
            value = reader(cid, conversation) if callable(reader) else {'available': False}
            view = getattr(workbench.host, 'usage_view', None)
            if view is not None:
                from shinku.angelos.usage_totals import read_instance_usage
                # 累计故障不遮盖已取得的单轮/会话快照，也不读取其他服务的数据。
                try:
                    value['instance'] = read_instance_usage(view)
                except Exception:
                    value['instance'] = {'available': False}
            return response({'usage': value})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations', response_model=None)
    async def create_conversation(request: Request):
        try:
            data = await body(request)
            with workbench.lock:
                c = store.conversation(data.get('workspaceId', 'default'))
            if getattr(workbench.host, 'session_models', None):
                workbench.model_options(c['id'])
                c = store.read('conversation', c['id'])
            return response({'conversation': c})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}', response_model=None)
    def conversation(cid: str, request: Request):
        try:
            guard(request)
            current = workbench.refresh(cid)
            reader = getattr(workbench.host, 'read_progress', None)
            if callable(reader):
                # 只投影最近几轮；进度不落回会话/模型历史，不因轮询制造新的执行事件。
                for turn in current['turns'][-8:]:
                    turn['live_progress'] = reader(cid, turn)
            return response({'conversation': current,
                'attachments': [{k: a[k] for k in ('id', 'name')} for a in store.list('attachment') if a['conversationId'] == cid]})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations/{cid}', response_model=None)
    async def edit_conversation(cid: str, request: Request):
        try:
            data = await body(request)
            return response({'conversation': workbench.edit_conversation(cid, data)})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}/activity', response_model=None)
    async def activity(cid: str, request: Request):
        try:
            guard(request)
            from shinku.angelos.live_activity import turn_snapshot, events
            import asyncio
            initial = await asyncio.to_thread(turn_snapshot, workbench, cid)
            return StreamingResponse(events(workbench, cid, request, initial), media_type='text/event-stream',
                headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff'})
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}/tools/{turn_id}', response_model=None)
    def tool_activity(cid: str, turn_id: str, request: Request):
        try:
            guard(request)
            from shinku.angelos.live_activity import turn_snapshot
            return response({'activity': turn_snapshot(workbench, cid, turn_id)})
        except Exception as exc:
            return failure(exc)

    # ---- 完整执行日志：用户可见，与模型上下文分流 ----
    # 路径前缀用 execution-log：与 /tools/{turn_id} 段数相同的 /log/... 会被
    # 既有的 tools 路由抢先匹配而返回 404。
    # 路由只接受会话/轮次/调用三元组，不接受浏览器指定 task_id 或路径；
    # 任务归属由存储中的轮次解析，越权与非本机来源直接拒绝。
    def log_turn(cid, turn_id):
        with workbench.lock:
            conversation = workbench.store.read('conversation', cid)
            turn = next((t for t in conversation['turns'] if t['id'] == turn_id), None)
        if turn is None:
            raise ValueError('turn_not_found')
        return dict(turn)

    def log_call(cid, turn_id, call_id, request, **kwargs):
        from shinku.angelos.execution_log import LogUnavailable, read_log
        guard(request)
        try:
            return response({'log': read_log(workbench.host, cid, log_turn(cid, turn_id), call_id, **kwargs)})
        except LogUnavailable as exc:
            # 未采集/已过期/校验失败均如实告，不回退成猜测内容。
            raise ValueError(exc.code) from exc

    @router.get('/angelos/api/conversations/{cid}/execution-log/{turn_id}/{call_id}', response_model=None)
    def execution_log_page(cid: str, turn_id: str, call_id: str, request: Request, stream: str = 'stdout',
                           offset: int = 0, history: int = 0):
        # history=1 表示用户显式回看重试前的旧 attempt。默认仍只看当前 attempt：
        # 重试后旧调用的材料不再参与当前执行，不该在无提示时与当前调用混在一起。
        try:
            return log_call(cid, turn_id, call_id, request, stream=stream, offset=offset,
                            include_history=bool(history))
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}/execution-log/{turn_id}/{call_id}/search', response_model=None)
    def execution_log_search(cid: str, turn_id: str, call_id: str, request: Request, q: str = '',
                             history: int = 0):
        try:
            from shinku.angelos.execution_log import LogUnavailable, search_log
            guard(request)
            try:
                return response({'log': search_log(workbench.host, cid, log_turn(cid, turn_id), call_id, q,
                                                   include_history=bool(history))})
            except LogUnavailable as exc:
                raise ValueError(exc.code) from exc
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}/execution-log/{turn_id}/{call_id}/export', response_model=None)
    def execution_log_export(cid: str, turn_id: str, call_id: str, request: Request, history: int = 0):
        """导出本次调用的全部已采集行；内容与页面同一脱敏结果，不截断。"""
        try:
            from shinku.angelos.execution_log import LogUnavailable, log_segments
            guard(request)
            try:
                data = log_segments(workbench.host, cid, log_turn(cid, turn_id), call_id,
                                    include_history=bool(history))
            except LogUnavailable as exc:
                raise ValueError(exc.code) from exc
            head = ['# 工具调用 {name} ({id})'.format(name=data['call'].get('name', ''), id=data['call'].get('id', '')),
                    '# 状态 {status}；退出码 {code}；原件完整={complete}'.format(
                        status=data['call'].get('status', 'unknown'),
                        code=data['call'].get('exit_code', '未知'), complete=data.get('complete'))]
            if data.get('incomplete_reasons'):
                # 导出也要告诉用户为什么标为不完整，否则他只看到"完整=False"而无从判断。
                head.append('# 不完整原因：' + '、'.join(data['incomplete_reasons']))
            if data.get('unknown_fields'):
                head.append('# 未知字段：' + ', '.join(data['unknown_fields']))
            chunks = list(head)
            for segment in data['segments']:
                chunks.append('\n===== ' + segment['stream'] + ' =====')
                chunks.extend(row['text'] for row in segment['lines'])
                if not segment['lines'] or not segment['lines'][-1]['text'].endswith('\n'):
                    chunks.append('\n')
            text = ''.join(chunks)
            safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in str(data['call'].get('name') or 'call'))[:40]
            return Response(text, media_type='text/plain; charset=utf-8', headers={
                'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Content-Disposition': 'attachment; filename="angelos-log-{}.txt"'.format(safe or 'call')})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations/{cid}/messages', response_model=None)
    async def send(cid: str, request: Request):
        try:
            data = await body(request)
            return response({'conversation': workbench.send(cid, data)})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations/{cid}/turns/{turn_id}/{action}', response_model=None)
    async def control(cid: str, turn_id: str, action: str, request: Request):
        try:
            data = await body(request)
            return response({'conversation': workbench.control(cid, turn_id, action, data)})
        except Exception as exc:
            return failure(exc)

    @router.post('/angelos/api/conversations/{cid}/attachments', response_model=None)
    async def attach(cid: str, request: Request):
        try:
            data = await body(request)
            raw = base64.b64decode(data.get('data', ''), validate=True)
            content = raw.decode('utf-8-sig')
            name = data.get('name', '')
            if (not isinstance(name, str) or not name or len(name) > 200 or '/' in name or '\\' in name
                    or len(content) > 12000 or not content.strip() or '\0' in content):
                raise ValueError('attachment_text_required')
            with workbench.lock, store.transaction() as db:
                store.get(db, 'conversation', cid)
                attachment = dict(id=uuid.uuid4().hex, conversationId=cid, name=name, content=content)
                store.put(db, 'attachment', attachment)
            return response({'attachment': {'id': attachment['id'], 'name': name}})
        except (UnicodeError, binascii.Error):
            return failure(ValueError('attachment_text_required'))
        except Exception as exc:
            return failure(exc)

    @router.get('/angelos/api/conversations/{cid}/artifacts/{turn_id}/{index}', response_model=None)
    def artifact(cid: str, turn_id: str, index: int, request: Request):
        try:
            guard(request)
            if index < 0:
                raise ValueError('artifact_unavailable')
            name, data = workbench.artifact(cid, turn_id, index)
            return Response(data, media_type='application/octet-stream', headers={
                'Content-Disposition': "attachment; filename*=UTF-8''" + quote(name, safe=''),
                'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        except Exception as exc:
            return failure(exc)

    return router, workbench

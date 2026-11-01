"""专用宿主 HTTP 入口。启动器显式安装；网页令牌不能作为宿主连接凭据。"""
import json
from fastapi import APIRouter, Request
from fastapi.responses import Response
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.delegation_http import MAX_WIRE_BYTES, signature, wire_bytes


def build_delegation_router(gateway):
    router = APIRouter()

    @router.post('/angelos/host/v1')
    async def invoke(request: Request):
        principal = None
        nonce = ''
        try:
            if (request.client is None or request.client.host not in {'127.0.0.1', '::1', 'testclient'}
                    or request.headers.get('origin') or request.headers.get('content-type') != 'application/json'):
                raise AgentBoundaryError('delegation_origin_denied')
            host = request.headers.get('x-angelos-host', '')
            auth = request.headers.get('authorization', '')
            principal = gateway.authenticate(host, auth[7:] if auth.startswith('Bearer ') else '')
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_WIRE_BYTES:
                    raise AgentBoundaryError('delegation_input_oversized')
            payload = json.loads(body)
            nonce = payload.get('nonce', '')
            if not isinstance(nonce, str) or len(nonce) != 48:
                raise AgentBoundaryError('delegation_nonce_invalid')
            # 同步任务存储/认证由工作线程处理，不阻塞ASGI循环和续租请求。
            from starlette.concurrency import run_in_threadpool
            value = await run_in_threadpool(gateway.dispatch, host, payload)
            result, code = {'ok': True, 'nonce': nonce, 'value': value}, 200
        except Exception as exc:
            reason = exc.code if isinstance(exc, AgentBoundaryError) else 'delegation_request_invalid'
            result, code = {'ok': False, 'nonce': nonce, 'reason': reason}, 409 if principal else 403
        raw = wire_bytes(result)
        return Response(raw, status_code=code, media_type='application/json', headers={
            'Cache-Control': 'no-store', 'X-Angelos-Signature': signature(principal.secret, raw) if principal else ''})
    return router

"""Angelos2 专属插件页；仅本机同源操作开关，不接收提案/评分/权限或模型配置。"""
import json
import secrets

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from shinku.api.angelos import LOCAL

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Angelos2 · Goozma 插件</title>
<style>body{background:#090909;color:#ddd;font:17px/1.7 system-ui,sans-serif;margin:0}
main{max-width:840px;margin:56px auto;padding:24px}a{color:#86bbff}h1{color:#78b7ff;font-size:28px}
section{border:1px solid #666;padding:24px;margin:24px 0}button{border:1px solid #666;border-radius:0;
background:#111;color:#ddd;font:inherit;padding:10px 18px;margin:6px;cursor:pointer}
button[aria-pressed=true]{background:#8bbfff;color:#080808}button:disabled{opacity:.55}
small{color:#aaa}#status{color:#91c1ff}#error{color:#ffb5b5}code{color:#bccfe8}</style>
<main><a href="/angelos">← 返回 Angelos2 工作台</a><h1>Goozma / 实验插件</h1>
<p>为 Angelos2 整理候选经验；已验证的经验可按工作区检索。与真红和正式 Angelos 分开。</p>
<section><p id="status" role="status">正在读取设置…</p><div id="modes">
<button data-mode="off" aria-pressed="false">关闭</button>
<button data-mode="learn" aria-pressed="false">只学习</button>
<button data-mode="use" aria-pressed="false">学习并使用经验</button></div>
<p id="error" role="alert"></p><small>设置保存到当前实验实例，切换不会重启 Agent。</small></section>
<p>开启后仍须显式运行有预算的学习批次，不会因为打开开关就调用模型。
候选经过独立对照验证并启用后，才可被检索；没有有效经验时按原方式工作。</p>
<p>关闭后停止新的学习请求和经验读取。已发出的单次请求可完成留档，其余案例停止；
已有任务、对话和经验记录保留。经验仅是建议，不改变权限或执行验收。</p></main>
<script>
let state, token, busy=false;
const labels={off:'已关闭',learn:'只学习 · 不使用经验',use:'学习并使用已验证经验'};
const buttons=[...document.querySelectorAll('[data-mode]')];
function render(){document.querySelector('#status').textContent=labels[state.mode]+' · 版本 '+state.revision;
buttons.forEach(b=>{b.disabled=busy;b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode));});}
async function refresh(){const r=await fetch('/angelos/api/goozma',{cache:'no-store'});const v=await r.json();
if(!r.ok)throw Error(v.reason);state=v.state;token=v.token;render();}
buttons.forEach(b=>b.onclick=async()=>{if(busy||!state)return;busy=true;render();document.querySelector('#error').textContent='';
try{const r=await fetch('/angelos/api/goozma',{method:'POST',headers:{'Content-Type':'application/json','X-Goozma-Token':token},
body:JSON.stringify({mode:b.dataset.mode,revision:state.revision})});const v=await r.json();
if(!r.ok)throw Error(v.reason);state=v.state;}catch(e){document.querySelector('#error').textContent='设置未保存：'+e.message;
await refresh().catch(()=>{});}finally{busy=false;render();}});
refresh().catch(e=>{document.querySelector('#error').textContent='读取失败：'+e.message;});
</script></html>'''


def install(app, controls):
    """工厂显式安装；token 只服务本次页面操作，不持有模型密钥。"""
    token = secrets.token_urlsafe(32)
    def guard(request, write=False):
        if request.client is None or request.client.host not in LOCAL or request.url.hostname not in LOCAL:
            raise PermissionError('local_request_required')
        if (request.headers.get('sec-fetch-site') == 'cross-site'
                or request.headers.get('origin', str(request.base_url).rstrip('/')) != str(request.base_url).rstrip('/')):
            raise PermissionError('same_origin_required')
        if write and (not secrets.compare_digest(request.headers.get('x-goozma-token', ''), token)
                      or request.headers.get('content-type', '').split(';')[0] != 'application/json'):
            raise PermissionError('session_token_required')

    def error(exc):
        reason = str(exc) if isinstance(exc, (ValueError, PermissionError)) else 'goozam_controls_unavailable'
        if len(reason) > 100 or ' ' in reason:
            reason = 'invalid_request'
        return JSONResponse({'reason': reason}, status_code=403 if isinstance(exc, PermissionError) else 409,
                            headers={'Cache-Control': 'no-store'})

    @app.get('/angelos/goozma')
    def page(request: Request):
        try:
            guard(request)
            return HTMLResponse(PAGE, headers={'Cache-Control': 'no-store'})
        except Exception as exc:
            return error(exc)

    @app.get('/angelos/api/goozma')
    def status(request: Request):
        try:
            guard(request)
            return JSONResponse(dict(state=controls.snapshot(), token=token), headers={'Cache-Control': 'no-store'})
        except Exception as exc:
            return error(exc)

    @app.post('/angelos/api/goozma')
    async def select(request: Request):
        try:
            guard(request, True)
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 1024:
                    raise ValueError('request_too_large')
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {'mode', 'revision'}:
                raise ValueError('invalid_request')
            return JSONResponse(dict(state=controls.select(**value)), headers={'Cache-Control': 'no-store'})
        except Exception as exc:
            return error(exc)

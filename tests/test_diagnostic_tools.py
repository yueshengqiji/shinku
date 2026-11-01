"""真实隔离 HTTP/TCP 与系统依赖探测，区分健康层级和认证/业务。"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from shinku.agent.request_scope import RequestScope
from shinku.tools.diagnostics import DiagnosticTarget, DiagnosticTool, dependency_probe, http_probe, tcp_probe
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation


@contextmanager
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/auth':
                self.send_response(403); self.end_headers(); return
            if self.path == '/redirect':
                self.send_response(302); self.send_header('Location', '/private'); self.end_headers(); return
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'logged_in': False, 'business': False, 'secret': 'DO_NOT_REPORT'}).encode())
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', server.server_port
    finally:
        server.shutdown(); server.server_close(); thread.join()


def ctx():
    return {'agent_request_scope': RequestScope('owner', 'session', 'turn', (('turn', 'diagnose'),))}


def test_real_probe_layers_do_not_promote_connection_to_login_or_business():
    with endpoint() as (url, port):
        assert tcp_probe('127.0.0.1', port)() == ('ready', 'connection_available')
        assert http_probe(url)() == ('ready', 'http_connection_available')
        assert http_probe(url, predicate=lambda data: data['logged_in'], semantic_layer='login')() == ('unavailable', 'login_not_ready')
        assert http_probe(url, predicate=lambda data: data['business'], semantic_layer='business')() == ('unavailable', 'business_not_ready')
        assert http_probe(url + '/auth')() == ('unavailable', 'authentication_required')
        assert http_probe(url + '/redirect')() == ('unavailable', 'endpoint_status_failed')
        assert http_probe(url, predicate=lambda data: data['missing'], semantic_layer='login')() == ('unknown', 'endpoint_schema_unknown')


def test_dependency_and_unknown_never_run_dependent_business():
    targets = [DiagnosticTarget('missing', 'dependency', dependency_probe('synthetic_nonexistent_module')),
               DiagnosticTarget('business', 'business', lambda: (_ for _ in ()).throw(AssertionError('must not call')),
                                depends_on=('missing',))]
    tool = DiagnosticTool(targets, authorize=lambda ctx: True)
    result = tool.execute(call={'target': 'business'}, context=ctx())
    rows = result['state_updates']['checks']
    assert rows[0]['code'] == 'dependency_missing' and rows[1]['state'] == 'blocked'
    assert dependency_probe('json')() == ('ready', 'dependency_available')


def test_real_tool_registry_dispatch_refuses_unauthorized_target_and_redacts_payload():
    with endpoint() as (url, _):
        tool = DiagnosticTool([DiagnosticTarget('connection', 'connection', http_probe(url)),
                              DiagnosticTarget('login', 'login', http_probe(url, predicate=lambda data: data['logged_in'], semantic_layer='login'),
                                               depends_on=('connection',))], authorize=lambda c: c.get('allowed') is True)
        handlers = {tool.tool_type: tool}
        invocation = ToolInvocation(tool.tool_type, {'target': 'login'})
        assert execute_invocation(invocation, handlers=handlers, context=ctx())[1].status == 'error'
        context = {**ctx(), 'allowed': True}
        receipt = execute_invocation(invocation, handlers=handlers, context=context)[1]
        assert receipt.status == 'ok' and 'DO_NOT_REPORT' not in str(receipt)
        assert receipt.data['state_updates']['checks'][-1]['state'] == 'unavailable'
        denied = execute_invocation(invocation, handlers=handlers, context=context,
                                    policy=ExecutionPolicy(blocked=frozenset({tool.tool_type})))[1]
        assert denied.executed is False

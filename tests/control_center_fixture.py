"""控制中枢浏览器验收夹具：只绑定回环地址，全部操作仅修改内存合成数据。

运行：python tests/control_center_fixture.py --port 8767
打开 /debug/hub；/__fixture?case=partial|offline|changed|normal 控制读取故障。
/__fixture/requests 返回操作记录（测试密钥也不回显），便于核对按钮请求。
不会装配真红运行时、加载私用配置或向外部服务发请求。
"""
from __future__ import annotations

import argparse
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

FRONTEND = Path(__file__).resolve().parents[1] / 'frontend' / 'control-center'
TASK = {'task_id': 'task-fixture-1', 'status': 'running', 'goal': '整理合成资料', 'attempt': 1,
        'session_id': 'fixture-session', 'result_summary': '已核对 3 项合成资料',
        'metadata': {'budget': {'profile': 'standard', 'stage': 'collect', 'requests': 3, 'max_requests': 10, 'tokens': 250, 'tools': 2, 'active_seconds': 12}}}
STATE = {
    'health': {'status': 'ok', 'project': 'shinku-fixture', 'port': 8767, 'python': '/fixture/python'},
    'process': {'pid': 123}, 'observed_at': '2026-10-16T12:00:00Z',
    'agent': {'enabled': True, 'process': {'service': 'backend', 'mode': 'isolated'},
              'controls': {'responses_enabled': True, 'agent_enabled': True, 'permission_mode': 'traditional'},
              'live_models': {'chat': 'fixture-model'},
              'runtime': {'metrics': {'chat_json_calls': 12, 'total_tokens_chat': 420, 'input_tokens_chat': 300, 'output_tokens_chat': 120}},
              'memory': {'path': '/fixture/memory.db', 'stats': {'turns': 42, 'active_events': 30, 'active_facts': 5, 'routing_audits': 12}},
              'bridge': {'tool_names': ['fixture_read'], 'max_rounds': 6},
              'diagnostics': {'config': {'chat_model_name': 'fixture-model', 'chat_api_protocol': 'openai', 'chat_base_url': 'https://example.invalid/v1', 'chat_api_key': '<redacted>', 'send_enabled': False}}},
    'napcat': {'configured': True, 'dispatcher_attached': True, 'transport': 'fixture'},
    'pet': {'available': True, 'status': 'offline', 'character_name': '测试角色'},
    'model_services': {'available': True, 'activeProviderId': 'fixture', 'activeModel': 'fixture-model', 'providers': [
        {'id': 'fixture', 'label': '合成供应方', 'protocol': 'openai', 'baseUrl': 'https://example.invalid/v1', 'chatModel': 'fixture-model', 'active': True, 'hasApiKey': True, 'models': ['fixture-model', 'fixture-model-2']}]},
    'logs': [{'name': 'fixture.log', 'updated_at': '12:00:00', 'lines': ['INFO synthetic fixture ready', '<script>must remain text</script>']}]}


class FixtureHandler(BaseHTTPRequestHandler):
    """只模拟前端已使用的接口，不代理任何真实 API。"""
    state = copy.deepcopy(STATE)
    case = 'normal'
    requests: list[dict] = []

    def log_message(self, *_args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        params = parse_qs(urlsplit(self.path).query)
        assets = {'/debug/hub': ('index.html', 'text/html'), '/debug/hub/assets/app.css': ('app.css', 'text/css'), '/debug/hub/assets/app.js': ('app.js', 'text/javascript')}
        if path in assets:
            name, mime = assets[path]
            body = (FRONTEND / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if path == '/__fixture':
            type(self).case = params.get('case', ['normal'])[0]
            return self.send_json({'case': self.case})
        if path == '/__fixture/requests':
            return self.send_json(self.requests)
        if self.case == 'offline' or (self.case == 'partial' and path == '/debug/hub/traces'):
            return self.send_json({'ok': False, 'reason': 'fixture_unavailable'}, 503)
        if path == '/debug/hub/state':
            return self.send_json(self.state)
        if path == '/debug/hub/traces':
            rows = [{'trace_id': f'trace-fixture-{n}', 'conversation_type': ['private', 'group', 'desktop'][n % 3], 'conversation_id': f'conversation-{n}', 'status': 'finished', 'input_chars': 10, 'reply_chars': 20, 'elapsed_ms': 123, 'delivery_status': 'sent', 'stages': [{'name': 'llm_usage', 'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}, {'name': 'memory', 'history_mode': 'recent', 'retrieved_count': 2}]} for n in range(12)]
            if self.case == 'changed':
                rows.insert(0, dict(rows[0], trace_id='trace-new', status='running'))
            return self.send_json({'traces': rows})
        if path == '/debug/hub/attention':
            return self.send_json({'available': True, 'items': [{'trace_id': 'attention-fixture', 'message_id': 'message-fixture', 'conversation_type': 'private', 'conversation_id': 'fixture', 'observed_at': 100, 'attention': {'should_schedule': True}, 'turn': {'sent': True, 'llm_usage': {'available': False}}}]})
        if path == '/debug/hub/memory-routing':
            return self.send_json({'available': True, 'items': [{'audit_id': f'audit-{n}', 'conversation_id': 'fixture', 'created_at': 100 + n, 'history_mode': 'recent', 'reason': 'synthetic', 'retrieved_count': 2} for n in range(12)]})
        if path == '/metrics/recent':
            return self.send_json({'days': [{'day': '2026-10-16', 'chat_json_calls': 12, 'total_tokens_chat': 420}]})
        if path == '/api/qq/self-check':
            return self.send_json({'ok': True, 'checks': [{'name': 'fixture_check', 'ok': True, 'detail': '隔离检查通过'}]})
        if path == '/capabilities':
            return self.send_json({'capabilities': [{'name': 'fixture_read', 'risk': 'low', 'permission_class': 'read', 'requires_approval': False, 'description': '合成只读工具', 'input_schema': {'type': 'object'}}]})
        if path.startswith('/capabilities/approvals/'):
            return self.send_json({'ok': True, 'pending': True, 'approval': {'request_id': 'approval-fixture', 'question': '合成待授权请求', 'tools': [{'tool': 'fixture_read', 'arguments': {'path': 'fixture.txt'}, 'risk': 'low'}]}})
        if path == '/agent-work/tasks':
            return self.send_json({'tasks': [TASK, dict(TASK, task_id='task-fixture-failed', status='failed', error='synthetic_failure')]})
        if path.startswith('/agent-work/tasks/'):
            return self.send_json({'ok': True, 'task': dict(TASK, task_id=path.rsplit('/', 1)[-1])})
        if path == '/memory/features':
            return self.send_json({'ok': True, 'context_projection': {'configured': True, 'initialized': True, 'index_status': {'state': 'ready'}, 'policy_version': 'fixture'}})
        if path in ['/memory/timeline', '/memory/facts']:
            return self.send_json({'ok': True, 'scope': 'private:fixture', 'mode': params.get('mode', ['facts'])[0], 'records': [{'layer': 'fixture', 'created_at': 100, 'text': '这是合成记忆，<b>按纯文本显示</b>。'}]})
        return self.send_json({'ok': False, 'reason': 'unknown_fixture_route'}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or '{}')
        self.requests.append({'path': self.path, 'body': {k: ('<test-key-redacted>' if k == 'apiKey' else v) for k, v in body.items()}})
        if self.path == '/control-center/agent/control':
            controls = self.state['agent']['controls']
            action = body.get('action')
            if action in ['start', 'stop']:
                controls['responses_enabled'] = action == 'start'
            elif action in ['on', 'off']:
                controls['agent_enabled'] = action == 'on'
            elif action == 'permission':
                controls['permission_mode'] = body['mode']
        elif self.path == '/control-center/pet/control':
            self.state['pet']['status'] = 'offline' if body['action'] == 'stop' else 'online'
        elif self.path.startswith('/control-center/model-services/'):
            provider = self.state['model_services']['providers'][0]
            if self.path.endswith('/config'):
                provider.update(baseUrl=body['baseUrl'], chatModel=body['chatModel'])
            elif self.path.endswith('/select'):
                provider['chatModel'] = body['modelId']
            elif self.path.endswith('/models'):
                provider['models'].append('fixture-discovered')
            self.state['model_services']['activeModel'] = provider['chatModel']
            self.state['agent']['live_models']['chat'] = provider['chatModel']
        else:
            return self.send_json({'ok': False, 'reason': 'unknown_fixture_route'}, 404)
        self.send_json({'ok': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    print(f'Isolated fixture: http://127.0.0.1:{args.port}/debug/hub', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), FixtureHandler).serve_forever()

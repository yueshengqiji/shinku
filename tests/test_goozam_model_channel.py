"""真实Unix/HTTP与Linux沙箱集成；上游是本机合成服务，不持有真实密钥。"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import httpx
import pytest

from goozam.model_gateway import serve_gateway, summarize_batch
from goozam.model_runtime import relay_runtime, socket_address
from goozam.sandbox import command, run
from shinku.llm.request_ledger import JsonCallLedger
from test_goozam_model_gateway import route, limits, response, request


@contextmanager
def upstream(replies=None, *, delay=0):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            if delay:
                time.sleep(delay)
            result = replies[len(calls)-1] if replies else response()
            raw = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                pass  # 截止测试会主动杀掉代理，不能因此补发或伪造响应。
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield route(endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions', fixture_http=True), calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_socket_http_runtime_preserves_native_protocol(tmp_path):
    native = response()
    native['choices'][0] = dict(index=0, finish_reason='tool_calls', message=dict(role='assistant', content='',
        reasoning_content='synthetic protocol field', tool_calls=[dict(id='fixture-call', type='function',
        function=dict(name='read_goozam_experiences', arguments='{"query":"代码"}'))]))
    with upstream([native, response()]) as (config, calls):
        with serve_gateway(tmp_path / 'gateway', config, limits(), 'fixture-secret') as sock:
            with relay_runtime(sock, config.public()) as runtime:
                assert runtime.incremental_json_supported is False
                captured = []
                first = runtime.call_chat_json(system_prompt='fixture', user_prompt='read fixture', fallback={},
                    native_tools=[dict(type='function', function=dict(name='read_goozam_experiences', parameters=dict(type='object')))],
                    native_history=[], provider_message_sink=captured.append)
                assert any('native' in key for key in first)
                assert captured[0]['reasoning_content'] == 'synthetic protocol field'
                second = runtime.call_chat_json(system_prompt='fixture', user_prompt='read fixture', fallback={},
                    native_history=[captured[0], dict(role='tool', tool_call_id='fixture-call', content='synthetic result')])
                assert second['kind'] == 'final'
            assert calls[1]['messages'][-2]['reasoning_content'] == 'synthetic protocol field'
            assert calls[1]['messages'][-1]['tool_call_id'] == 'fixture-call'
            assert all(not {'max_tokens', 'max_completion_tokens'} & c.keys() for c in calls)
        with socket_address(sock) as address, httpx.Client(transport=httpx.HTTPTransport(uds=address), timeout=.2) as client:
            with pytest.raises(httpx.TransportError):
                client.post('http://goozam.invalid/v1/chat/completions', json=request())
        assert len(calls) == 2
        summary = summarize_batch(tmp_path / 'gateway')
        assert summary['ledger_complete'] and summary['estimated_cny'] == '0.0004'


def test_wrong_route_and_hard_deadline(tmp_path):
    with upstream() as (config, calls):
        with serve_gateway(tmp_path / 'gateway', config, limits(seconds=1), 'fixture-secret') as sock:
            with socket_address(sock) as address, httpx.Client(transport=httpx.HTTPTransport(uds=address), timeout=.5) as client:
                denied = client.post('http://goozam.invalid/anything', json=request())
                assert denied.status_code == 400 and not calls
                time.sleep(1.2)
                with pytest.raises(httpx.TransportError):
                    client.post('http://goozam.invalid/v1/chat/completions', json=request())


def test_deadline_during_http_preserves_unknown_and_cannot_resume(tmp_path):
    with upstream(delay=2) as (config, calls):
        with serve_gateway(tmp_path / 'gateway', config, limits(seconds=1), 'fixture-secret') as sock:
            with socket_address(sock) as address, httpx.Client(transport=httpx.HTTPTransport(uds=address), timeout=3) as client:
                with pytest.raises(httpx.TransportError):
                    client.post('http://goozam.invalid/v1/chat/completions', json=request())
        summary = summarize_batch(tmp_path / 'gateway')
        assert len(calls) == summary['requests'] == 1
        assert summary['statuses'] == {'unknown': 1}
        assert not summary['ledger_complete'] and summary['estimated_cny'] is None
        assert summary['totals']['input_tokens'] is None
        with pytest.raises(ValueError, match='already_exists'):
            with serve_gateway(tmp_path / 'gateway', config, limits(), 'fixture-secret'):
                pytest.fail('reusing interrupted batch must fail')


@pytest.fixture
def isolated_lab(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lab = tmp_path / 'lab'
    runtime = lab / 'runtime'
    runtime.mkdir(parents=True)
    for name in ('src', 'frontend'):
        shutil.copytree(root / name, runtime / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {p.relative_to(runtime).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in runtime.rglob('*') if p.is_file()}
    (lab / 'manifest.json').write_text(json.dumps(dict(architecture='goozam', instance='Angelos2', files=files)))
    for name in ('data', 'workspace', 'learning', 'evaluations', 'logs'):
        (lab / name).mkdir()
    return lab


def test_real_sandbox_agent_reaches_only_fixed_model_channel(tmp_path, isolated_lab):
    with upstream() as (config, calls):
        with serve_gateway(tmp_path / 'gateway', config, limits(seconds=60), 'fixture-secret') as sock:
            argv = command(isolated_lab, Path(sys.prefix), action='model-probe', relay_socket=sock, model_profile=config.public())
            assert '--unshare-all' in argv and '--share-net' not in argv
            assert str(sock.parent) not in argv and 'fixture-secret' not in repr(argv)
            completed = run(isolated_lab, Path(sys.prefix), action='model-probe', relay_socket=sock,
                            model_profile=config.public(), timeout=60)
        (tmp_path / 'sandbox.stdout').write_text(completed.stdout)
        (tmp_path / 'sandbox.stderr').write_text(completed.stderr)
        report = json.loads(completed.stdout)
        assert report['isolation']['model_credentials_absent']
        assert all(report['isolation']['host_ports_unreachable'].values())
        result = report['turn']['result']
        assert result['execution']['status'] == 'completed' and result['verification']['accepted']
        assert calls and len(calls) <= 4
        assert all(not c.get('stream') for c in calls)
        assert '静态' in result['execution_report']['text']
        metadata = json.loads((tmp_path / 'gateway/batch.json').read_text())
        summary = JsonCallLedger(tmp_path / 'gateway/calls.jsonl').snapshot(metadata['binding'], expected_requests=len(calls))
        assert summary['ledger_complete'] and summary['statuses'] == {'completed': len(calls)}
        assert summary['totals']['input_tokens'] == 100 * len(calls)


def test_model_socket_is_opt_in_and_does_not_weaken_offline_entry(isolated_lab):
    with pytest.raises(ValueError, match='relay_required'):
        command(isolated_lab, Path(sys.prefix), action='model-probe')
    with pytest.raises(ValueError):
        command(isolated_lab, Path(sys.prefix), relay_socket='/fake', model_profile={})


def test_pinned_extension_uses_snapshot_and_rejects_drift(isolated_lab):
    manifest_path = isolated_lab / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['extension_source'] = 'runtime'
    manifest_path.write_text(json.dumps(manifest))
    argv = command(isolated_lab, Path(sys.prefix))
    mount = argv.index('/extensions/goozam')
    assert argv[mount - 1] == str(isolated_lab / 'runtime/src/goozam')
    (isolated_lab / 'runtime/src/goozam/__init__.py').write_text('changed')
    with pytest.raises(ValueError, match='runtime_drift'):
        command(isolated_lab, Path(sys.prefix))


def test_invalid_extension_source_is_not_silently_external(isolated_lab):
    path = isolated_lab / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['extension_source'] = 'typo'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='extension_source_invalid'):
        command(isolated_lab, Path(sys.prefix))


@pytest.mark.parametrize('source_key', ['source', 'holdout_source'])
def test_sandbox_proposes_from_real_task_using_explicit_local_proxy(tmp_path, isolated_lab, source_key):
    from dataclasses import replace
    from test_goozam_proposals import reply
    seed=json.loads(run(isolated_lab,Path(sys.prefix),action='exercise').stdout)['exercise'][source_key]
    run(isolated_lab,Path(sys.prefix),action='plugin-learn')
    payload=response()
    payload['choices'][0]['message']['content']=json.dumps(reply(),ensure_ascii=False)
    with upstream([payload]) as (fixture,calls):
        config=replace(fixture,fixture_http=False,local_proxy=True)
        with serve_gateway(tmp_path/'proposal-gateway',config,limits(seconds=60),'synthetic-key') as sock:
            args=dict(action='learn-propose',relay_socket=sock,model_profile=config.public(),proposal_refs=[seed])
            first=json.loads(run(isolated_lab,Path(sys.prefix),**args).stdout)
            again=json.loads(run(isolated_lab,Path(sys.prefix),**args).stdout)
        assert first['proposals']==again['proposals']
        # exercise撤销了原来源候选；生成器不能偷偷复活它，依旧保存冲突原稿。
        if source_key == 'source':
            assert first['proposals'][0]['status']=='failed'
            assert first['proposals'][0]['error']=='goozam_proposal_conflict'
        else:
            assert first['proposals'][0]['status']=='candidate'
            assert first['proposals'][0]['activated'] is False
        assert len(calls)==1
        summary=summarize_batch(tmp_path/'proposal-gateway')
        assert summary['ledger_complete'] and summary['requests']==1
        assert calls[0]['messages'][0]['role']=='system'
        assert not calls[0].get('tools')
        assert list((isolated_lab/'learning/proposals').glob('*/source.json'))
        assert list((isolated_lab/'learning/proposals').glob('*/response.json'))


def test_sandbox_plugin_switches_persist_without_network(isolated_lab):
    for action, mode in [('plugin-status', 'off'), ('plugin-learn', 'learn'),
                         ('plugin-use', 'use'), ('plugin-off', 'off')]:
        result = json.loads(run(isolated_lab, Path(sys.prefix), action=action).stdout)
        assert result['state']['mode'] == mode and result['model_calls'] == 0

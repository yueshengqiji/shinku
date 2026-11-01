"""合成实例回归：实际本机HTTP/文件与socket归属，不访问正式服务或调用模型。"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import pytest

from shinku.agent.task_state import new_task
from shinku.agent_runtime import AgentScope, AgentTurnRequest
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.angelos.instance_registry import InstanceRegistry, RuntimeInstance, read_json
from shinku.angelos.instance_probe import InstanceProbe, owns_endpoint, runtime_state, service_state
from shinku.angelos.permissions import PermissionControls


@contextmanager
def endpoint(payload, *, redirect=False):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append(self.path)
            self.send_response(302 if redirect else 200)
            if redirect:
                self.send_header('Location', 'http://192.0.2.1/private')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown(); server.server_close(); thread.join()


def instance(tmp_path, *, name='alpha', url='http://127.0.0.1:19001', model='model-a'):
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps({'activeProviderId': name, 'providers': {name: {
        'chatModel': model, 'apiKey': 'SECRET-CANARY', 'baseUrl': 'http://user:SECRET@host/path?token=SECRET'}}}))
    return RuntimeInstance(name, name, name + '.service', url, 'console', str(path))


def process(unit):
    return {'pid': os.getpid(), 'state': 'active', 'generation': 'test-start', 'source': 'fixture'}


def test_three_instances_never_borrow_neighbors_config(tmp_path):
    entries = [instance(tmp_path, name=name, url=f'http://127.0.0.1:{19001+i}', model=model)
               for i, (name, model) in enumerate([('alpha', 'model-a'), ('beta', 'model-b'), ('gamma', 'model-c')])]
    registry = InstanceRegistry(entries)
    probe = InstanceProbe(process_reader=process, endpoint_owner=lambda *_: True,
        runtime_reader=lambda e: {'chat_model': {'alpha': 'model-a', 'beta': 'model-b', 'gamma': 'model-c'}[e.id]})
    assert registry.catalog()['is_live_state'] is False
    for e in entries:
        result = probe.inspect(registry.get(e.id), ['models', 'locations'])
        assert result['consistency'] == 'matched' and result['instance_id'] == e.id
        assert result['saved']['provider_id'] == e.id
        assert 'SECRET' not in json.dumps(result)
    with pytest.raises(ValueError): registry.get('unknown')


@pytest.mark.parametrize('case', ['matched', 'different', 'offline', 'wrong_owner', 'restarted', 'no_saved'])
def test_real_http_file_and_pid_binding(tmp_path, case):
    payload = {'agent': {'live_models': {'chat': 'model-a', 'aux': 'model-a', 'apiKey': 'SECRET'}}}
    with endpoint(payload) as (url, requests):
        e = instance(tmp_path, url=url, model='model-b' if case == 'different' else 'model-a')
        if case == 'no_saved': Path(e.model_store).unlink()
        reads = []
        def process_reader(unit):
            reads.append(unit)
            return {**process(unit), 'generation': 'new' if case == 'restarted' and len(reads) == 2 else 'old'}
        def owner(p, address):
            return owns_endpoint({**p, 'pid': 0} if case == 'wrong_owner' else p, address)
        def runtime(e):
            if case == 'offline': raise OSError('SECRET exception')
            return runtime_state(e)
        result = InstanceProbe(process_reader=process_reader, endpoint_owner=owner, runtime_reader=runtime).inspect(e, ['models'])
        assert result['consistency'] == (case if case in ('matched', 'different') else 'unknown')
        if case in ('offline', 'wrong_owner', 'restarted'): assert result['running'] is None
        if case == 'wrong_owner': assert not requests
        if case == 'restarted': assert 'process_generation_unconfirmed_or_changed' in result['gaps']
        assert 'SECRET' not in json.dumps(result)
        assert reads == [e.service, e.service]


def test_redirect_never_follows_to_unregistered_endpoint(tmp_path):
    with endpoint({}, redirect=True) as (url, requests):
        e = instance(tmp_path, url=url)
        result = InstanceProbe(process_reader=process).inspect(e, ['models'])
        assert result['running'] is None and result['consistency'] == 'unknown'
        assert requests == ['/debug/hub/state']


def test_angelos_default_is_not_session_model(tmp_path):
    with endpoint({'models': {'selected': {'modelId': 'default-model', 'providerId': 'local', 'apiKey': 'SECRET'}}}) as (url, _):
        e = replace(instance(tmp_path, url=url), adapter='angelos', store_format='angelos_preferences')
        Path(e.model_store).write_text(json.dumps({'selection': {'modelId': 'default-model', 'providerId': 'local'}, 'apiKey': 'SECRET'}))
        result = InstanceProbe(process_reader=process).inspect(e, ['models'])
        assert result['running']['model_scope'] == 'instance_default_not_session_selection'
        assert result['consistency'] == 'matched' and 'SECRET' not in json.dumps(result)


@pytest.mark.parametrize('change', [
    {'endpoint': 'http://example.com'}, {'endpoint': 'http://127.0.0.1:19001/model'},
    {'endpoint': 'http://name:password@127.0.0.1:19001'}, {'service': '--property=x'},
    {'model_store': '../settings'}, {'adapter': 'shell'}, {'endpoint': 'http://127.0.0.1:19001/?token=x'},
])
def test_invalid_registration_is_rejected(tmp_path, change):
    with pytest.raises(ValueError): replace(instance(tmp_path), **change)


def test_registry_file_explicit_no_credentials_and_duplicate_rejected(tmp_path):
    from dataclasses import asdict
    e = instance(tmp_path); p = tmp_path/'registry.json'
    p.write_text(json.dumps({'schema_version': 1, 'instances': [asdict(e)]})); p.chmod(0o600)
    assert InstanceRegistry.from_file(p).get(e.id) == e
    p.chmod(0o666)
    with pytest.raises(ValueError): InstanceRegistry.from_file(p)
    with pytest.raises(ValueError): InstanceRegistry([e, e])
    link = tmp_path/'link.json'; link.symlink_to(e.model_store)
    with pytest.raises(OSError): read_json(link)


def test_systemd_adapter_uses_only_fixed_read_properties(monkeypatch):
    from types import SimpleNamespace
    def run(args, **kwargs):
        assert args[:4] == ['systemctl', '--user', 'show', 'alpha.service']
        assert '--property=Environment' not in args and kwargs['timeout'] == 3
        return SimpleNamespace(stdout='MainPID=42\nActiveState=active\nInvocationID=abc\n')
    monkeypatch.setattr('shinku.angelos.instance_probe.subprocess.run', run)
    assert service_state('alpha.service')['pid'] == 42


@pytest.fixture
def host_case(tmp_path):
    service, write, runtime = demo_service(tmp_path/'state')
    controls = PermissionControls(tmp_path/'permissions.json')
    registry = InstanceRegistry([instance(tmp_path)])
    host = ServiceHost(service, requester='owner', permissions=controls, runtime_instances=registry)
    task = new_task(task_id='current', requester='owner', session='angelos:one', request='检查目标实例', message_id='message', contract=None)
    task.update(status='running', execution_attempt=1); service.task_store.save(task)
    def ctx():
        return execution_context(task, service._grant(AgentScope('owner', 'angelos:one'), task, 'inspect'), attempt=1)
    yield service, host, controls, ctx, runtime
    service.coordinator.shutdown(wait=True); service.task_store.close()


def test_he_reads_catalog_without_llm_and_zayin_denies(host_case):
    service, host, controls, ctx, runtime = host_case
    tool = service.executor.tool_host.registry.get('list_runtime_instances')
    value = tool.execute(call={}, context=ctx())
    assert value.get('status') != 'error' and 'alpha' in value['followup_context']
    assert not runtime.calls
    controls.select('zayin', controls.revision)
    denied = tool.execute(call={}, context=ctx())
    assert denied['status'] == 'blocked' and 'alpha' not in denied['followup_context']
    assert not runtime.calls


def test_snapshot_tampering_and_cross_request_rejected(host_case):
    service, host, _, ctx, runtime = host_case
    tool = host.instance_tools[0]; context = ctx()
    updates = tool.execute(call={}, context=context)['state_updates']
    assert tool.agent_verify_result(receipt=updates, context=context)['result_valid']
    changed = deepcopy(updates); changed['instance_snapshot']['instances'][0]['display_name'] = 'neighbor'
    assert not tool.agent_verify_result(receipt=changed, context=context)['result_valid']
    changed = dict(context); changed['agent_authority_revision'] = 'other'
    assert not tool.agent_verify_result(receipt=updates, context=changed)['result_valid']
    assert not runtime.calls


def test_revocation_during_probe_does_not_return_observation(host_case):
    _, host, controls, ctx, _ = host_case
    tool = host.instance_tools[1]
    def inspect(*args):
        controls.select('zayin', controls.revision)
        return {'secret_marker': 'NOT-RETURNED'}
    tool.probe.inspect = inspect
    result = tool.execute(call={'instance_id': 'alpha'}, context=ctx())
    assert result['status'] == 'error' and 'NOT-RETURNED' not in str(result)


@pytest.mark.parametrize('args', [{'instance_id': 'unknown'}, {'instance_id': 'alpha', 'url': 'http://example.com'},
    {'instance_id': 'alpha', 'sections': ['models', 'models']}, {'instance_id': 'alpha', 'sections': [False]}])
def test_model_cannot_choose_url_file_or_invalid_section(host_case, args):
    _, host, _, ctx, _ = host_case
    tool = host.instance_tools[1]
    tool.probe.inspect = lambda *a: pytest.fail('invalid input must not read')
    assert tool.execute(call=args, context=ctx())['status'] == 'error'


def test_default_host_does_not_discover_instances(tmp_path):
    service, _, runtime = demo_service(tmp_path)
    try:
        host = ServiceHost(service)
        assert not host.instance_tools and service.executor.tool_host.registry.get('inspect_runtime_instance') is None
        assert not runtime.calls
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('unavailable', [False, True])
def test_correction_reaches_native_loop_with_bound_snapshot_no_extra_llm(tmp_path, unavailable):
    """固定SDK回复验证新指导、旧错误历史和真实读取接线；不证明真实模型已学会纠错。"""
    from test_agent_conversation_loop import native_sdk
    from test_agent_runtime_boundary import settled
    from shinku.angelos.conversation_context import conversation_source
    service, write, _ = demo_service(tmp_path/'state')
    history = conversation_source([{'text': '目标服务用什么模型', 'reply': '我此前误称目标是邻居，使用model-wrong。'}])['content']
    with endpoint({'agent': {'live_models': {'chat': 'model-right', 'aux': 'model-right'}}}) as (url, requests):
        registry = InstanceRegistry([instance(tmp_path, url=url, model='model-right')])
        def respond(payload, count):
            body = json.dumps(payload, ensure_ascii=False)
            assert '用户纠正' in body and '待核实线索' in body
            if count == 1:
                assert 'model-wrong' in body
                return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'inspect', 'type': 'function',
                    'function': {'name': 'inspect_runtime_instance', 'arguments': '{"instance_id":"alpha"}'}}]}, 'tool_calls'
            assert count == 2 and 'model-right' in body and 'endpoint_binding' in body
            answer = '尚不能确认目标实例的运行模型；保存值为model-right，不能据此认定已加载。' if unavailable else '已核对目标实例，运行模型为model-right，之前把邻居当目标的说法不成立。'
            return {'role': 'assistant', 'content': json.dumps({'kind': 'final', 'text': answer, 'task_outcome': 'completed'})}, 'stop'
        try:
            with native_sdk(respond) as (runtime, calls):
                service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
                host = ServiceHost(service, requester='owner', runtime_instances=registry)
                host.instance_tools[1].probe = InstanceProbe(process_reader=process, endpoint_owner=lambda p, u: not unavailable and owns_endpoint(p, u))
                host.prepare = lambda *args: pytest.fail('no extra preparer')
                port = host.port('one')
                handle = port.submit(AgentTurnRequest('correction', '你把目标服务和邻居搞错了，请核对', history_context=history))
                view = settled(port, handle.task_id); result = port.result(handle.task_id)
                assert view['status'] == 'completed', result
                assert len(calls) == 2 and result['budget']['tools'] == 1
                assert result['verification']['evidence'][0]['receipt_valid'] is True
                assert requests == ([] if unavailable else ['/debug/hub/state']) and write.calls == 0
        finally:
            service.coordinator.shutdown(wait=True); service.task_store.close()


def test_missing_fields_and_malformed_saved_config_are_unknown(tmp_path):
    e = instance(tmp_path)
    Path(e.model_store).write_text('{invalid')
    p = InstanceProbe(process_reader=process, endpoint_owner=lambda *_: True,
                      runtime_reader=lambda _: {'chat_model': None})
    result = p.inspect(e, ['models'])
    assert result['saved'] is None and result['consistency'] == 'unknown'
    assert set(result['gaps']) == {'running_model_unavailable', 'saved_model_unavailable'}


@pytest.mark.parametrize('level', ['he', 'waw', 'aleph'])
def test_read_tools_do_not_create_approvals_or_grant_write(host_case, level):
    service, host, controls, ctx, runtime = host_case
    controls.select(level, controls.revision)
    tool = service.executor.tool_host.registry.get('list_runtime_instances')
    assert tool.execute(call={}, context=ctx()).get('status') not in ('error', 'blocked')
    assert controls.pending == {} and not runtime.calls
    if level == 'he':
        task = ctx()['agent_task_state']
        grant = service._grant(AgentScope('owner', 'angelos:one'), task, 'inspect')
        assert 'demo_file' not in grant.allowed_tools


def test_explicit_policy_denial_is_preserved(host_case):
    service, host, _, ctx, _ = host_case
    original = service.authorize
    def deny(*args):
        grant = original(*args)
        return replace(grant, policy=replace(grant.policy, blocked=grant.policy.blocked | {'list_runtime_instances'}))
    service.authorize = deny
    assert host.instance_tools[0].execute(call={}, context=ctx())['status'] == 'error'


def test_log_scope_distinguishes_registered_neighbor_and_unknown(tmp_path):
    from shinku.angelos.service_logs import ServiceLogsTool
    registry = InstanceRegistry([instance(tmp_path)])
    service, _, _ = demo_service(tmp_path/'state')
    try:
        host = ServiceHost(service, requester='owner', reasoning_path=tmp_path/'reasoning.sqlite3',
                           runtime_instances=registry, diagnostic_units=('alpha.service', 'neighbor.service'))
        task = new_task(task_id='log', requester='owner', session='angelos:one', request='核对日志', message_id='m', contract=None)
        task['execution_attempt'] = 1; service.task_store.save(task)
        context = execution_context(task, service._grant(AgentScope('owner', 'angelos:one'), task, 'inspect'), attempt=1)
        host.service_logs.reader = lambda q: {'entries': [{'MESSAGE': 'model-a', '_PID': '7'}], 'more_in_window': False}
        for unit, expected in [('alpha.service', ['alpha']), ('neighbor.service', [])]:
            result = host.service_logs.execute(call={'unit': unit}, context=context)
            payload = json.loads(result['state_updates']['log_text'])
            assert [x['instance_id'] for x in payload['instance_scope']['registered_instances']] == expected
            assert payload['entries'][0]['_PID'] == '7'
            # 重开日志快照仍验证原归属和PID，不因最新登记/进程变化重写。
            reopened = ServiceLogsTool(service, 'owner', host.service_logs.path, host.service_logs.units)
            assert reopened.agent_verify_result(receipt=result['state_updates'], context=context)['result_valid']
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_cgroup_child_binding_excludes_neighbor(tmp_path):
    from shinku.angelos.instance_probe import service_pids
    group = tmp_path/'fixture.service'; group.mkdir()
    (group/'cgroup.procs').write_text('11\n12\n')
    p = {'pid': 11, 'state': 'active', 'cgroup': '/fixture.service'}
    assert service_pids(p, cgroup_root=tmp_path) == (11, 12)
    assert service_pids({**p, 'pid': 13}, cgroup_root=tmp_path) == ()
    assert service_pids({**p, 'cgroup': '/../neighbor'}, cgroup_root=tmp_path) == ()
    (group/'cgroup.procs').write_text('12\n')
    assert service_pids(p, cgroup_root=tmp_path) == ()


def test_actual_socket_in_same_unit_child_is_accepted(tmp_path, monkeypatch):
    """本进程模拟unit子进程，父PID不持有socket；成员集合从有界cgroup读取。"""
    from shinku.angelos.instance_probe import service_pids
    with endpoint({'agent': {'live_models': {'chat': 'child-model'}}}) as (url, requests):
        group = tmp_path/'fixture.service'; group.mkdir()
        parent = os.getpid() + 10000000
        (group/'cgroup.procs').write_text(f'{parent}\n{os.getpid()}\n')
        monkeypatch.setattr('shinku.angelos.instance_probe.service_pids', lambda p: service_pids(p, cgroup_root=tmp_path))
        e = instance(tmp_path, url=url, model='child-model')
        probe = InstanceProbe(process_reader=lambda _: {'pid': parent, 'state': 'active', 'generation': 'child-start', 'cgroup': '/fixture.service'})
        result = probe.inspect(e, ['models'])
        assert result['consistency'] == 'matched' and requests == ['/debug/hub/state']


def test_same_model_different_reported_provider_is_not_matched(tmp_path):
    e = instance(tmp_path)
    p = InstanceProbe(process_reader=process, endpoint_owner=lambda *_: True,
                      runtime_reader=lambda _: {'chat_model': 'model-a', 'provider_id': 'other'})
    result = p.inspect(e, ['models'])
    assert result['consistency'] == 'different'
    assert result['comparison_fields'] == ['chat_model', 'provider_id']


def test_cross_session_and_expired_observation_rejected(host_case):
    _, host, _, ctx, _ = host_case
    tool = host.instance_tools[0]; context = ctx()
    updates = tool.execute(call={}, context=context)['state_updates']
    changed = dict(context)
    changed['agent_request_scope'] = replace(context['agent_request_scope'], session='angelos:other')
    assert not tool.agent_verify_result(receipt=updates, context=changed)['result_valid']
    at = updates['observation']['self_state_receipt']['observed_at']
    tool.observations.store.clock = lambda: at + 181
    assert not tool.agent_verify_result(receipt=updates, context=context)['result_valid']


def test_neighbor_takes_port_during_read_discards_running_value(tmp_path):
    owners = iter([True, False])
    p = InstanceProbe(process_reader=process, endpoint_owner=lambda *_: next(owners),
                      runtime_reader=lambda _: {'chat_model': 'neighbor-model'})
    result = p.inspect(instance(tmp_path), ['models'])
    assert result['running'] is None and result['consistency'] == 'unknown'
    assert 'endpoint_owner_changed' in result['gaps']

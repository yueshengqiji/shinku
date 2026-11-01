"""官方SDK的真实stdio进程验收：schema、权限、取消和未知效果，不访问网络。"""
import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest

from shinku.hosts.external import ExternalToolDescriptor
from shinku.hosts.mcp import MCPTransport, MCPTransportError, StdioConfig, fingerprint

SDK = importlib.util.find_spec('mcp') is not None
needs_sdk = pytest.mark.skipif(not SDK, reason='optional MCP SDK; isolated integration environment runs this suite')


def transport(tmp_path, *, authorize=lambda c, n: True, timeout=8):
    return MCPTransport(StdioConfig(sys.executable,
        (str(Path(__file__).parent / 'fixtures/mcp_fixture.py'),),
        timeout=timeout, stderr_path=str(tmp_path / 'mcp-stderr.log')), authorize=authorize)


def admit(client, *names):
    rows = {row['name']: row for row in client.describe()}
    client.admit({name: fingerprint(rows[name]) for name in names})


def test_import_and_construction_do_not_load_sdk_or_start_process(tmp_path):
    client = transport(tmp_path)
    assert client._thread is None and not client.health()['ready']
    assert not (tmp_path / 'mcp-stderr.log').exists()
    client.close()


@pytest.mark.parametrize('options', [{'command': 'python'}, {'timeout': 0}, {'timeout': float('nan')},
    {'args': ['server.py']}, {'cwd': 'relative'}, {'cwd': 3}, {'env': []}])
def test_invalid_process_configuration_is_rejected(options):
    kwargs = dict(command=sys.executable); kwargs.update(options)
    with pytest.raises(ValueError): StdioConfig(**kwargs)


@needs_sdk
@pytest.mark.parametrize('key', ['inputSchema', 'outputSchema'])
@pytest.mark.parametrize('reference', ['$ref', '$dynamicRef', '$recursiveRef'])
def test_admission_denies_remote_references_on_both_sides(tmp_path, monkeypatch, key, reference):
    client = transport(tmp_path)
    row = {'name': 'echo', 'inputSchema': {'type': 'object'}, key: {'type': 'object', reference: 'https://forbidden.invalid/schema'}}
    monkeypatch.setattr(client, 'describe', lambda: [row])
    with pytest.raises(MCPTransportError, match='external_schema_reference_denied'):
        client.admit({'echo': fingerprint(row)})
    assert client._thread is None
    client.close()


@needs_sdk
def test_real_sdk_stdio_discovery_explicit_admission_and_external_bridge(tmp_path):
    client = transport(tmp_path)
    try:
        rows = client.describe()
        assert {v['name'] for v in rows} >= {'echo', 'write'}
        with pytest.raises(MCPTransportError, match='not_admitted'):
            client.call(tool_name='echo', arguments={'value': 'hello'})
        admit(client, 'echo')
        descriptor = ExternalToolDescriptor('echo', input_schema=next(r['inputSchema'] for r in rows if r['name']=='echo'), permission_class='read')
        bridge = client.bridge(host_id='fixture', descriptors=(descriptor,))
        result = bridge.registry().get('echo').execute(call={'value': 'hello'}, context={})
        assert result['structuredContent'] == {'value': 'hello'}
        assert client.health()['ready'] and client.health()['business_success'] == 'not_verified'
    finally: client.close()
    assert not client.health()['ready'] and not client._thread.is_alive()


@needs_sdk
def test_stdio_eof_after_effect_is_unknown_and_never_reconnected(tmp_path):
    client = transport(tmp_path, timeout=2)
    path = tmp_path / 'effect-before-eof'
    try:
        admit(client, 'exit_server')
        with pytest.raises(MCPTransportError) as captured:
            client.call(tool_name='exit_server', arguments={'path': str(path)})
        assert captured.value.may_have_effect and path.read_text() == 'one effect before EOF'
    finally: client.close()
    assert not client.health()['ready'] and not client._thread.is_alive()


@needs_sdk
def test_schema_types_and_live_host_permission_are_checked_before_effect(tmp_path):
    allowed = [True]
    client = transport(tmp_path, authorize=lambda c, n: allowed[0])
    path = tmp_path / 'effect'
    try:
        admit(client, 'write')
        with pytest.raises(MCPTransportError, match='arguments_invalid'):
            client.call(tool_name='write', arguments={'path': str(path), 'value': 42})
        allowed[0] = False
        with pytest.raises(MCPTransportError, match='authorization_denied'):
            client.call(tool_name='write', arguments={'path': str(path), 'value': 'x'})
        assert not path.exists()
        allowed[0] = True
        client.call(tool_name='write', arguments={'path': str(path), 'value': 'x'})
        assert path.read_text() == 'x'
    finally: client.close()


@needs_sdk
def test_changed_remote_schema_does_not_silently_refresh_permission(tmp_path):
    client = transport(tmp_path)
    try:
        admit(client, 'echo', 'change_schema')
        client.call(tool_name='change_schema', arguments={})
        with pytest.raises(MCPTransportError, match='schema_changed'):
            client.call(tool_name='echo', arguments={'value': 'old schema'})
    finally: client.close()


@needs_sdk
@pytest.mark.parametrize('stop', ['timeout', 'revocation'])
def test_inflight_stop_retains_possible_effect_and_does_not_retry(tmp_path, stop):
    allowed = [True]
    client = transport(tmp_path, timeout=2, authorize=lambda c, n: allowed[0])
    path = tmp_path / 'started'
    errors = []
    try:
        admit(client, 'slow')
        def call():
            try: client.call(tool_name='slow', arguments={'path': str(path), 'seconds': 5})
            except MCPTransportError as exc: errors.append(exc)
        thread = threading.Thread(target=call); thread.start()
        until = time.monotonic() + 3
        while not path.exists() and time.monotonic() < until: time.sleep(.01)
        assert path.exists(), 'call must reach actual server before intervention'
        if stop == 'revocation': allowed[0] = False
        thread.join(5)
        assert not thread.is_alive() and len(errors) == 1 and errors[0].may_have_effect
        assert path.read_text() == 'started'
    finally: client.close()

"""原生窗口适配器使用真实子进程验收，API 不接模型、不改正式目录。"""
import asyncio
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from shinku.angelos.native_picker import NativePicker
from shinku.angelos.app import create_angelos_app


async def connected():
    return False


@pytest.fixture
def executable(tmp_path, monkeypatch):
    script = tmp_path/'zenity'
    monkeypatch.setenv('DISPLAY', ':fixture')
    monkeypatch.setattr('shinku.angelos.native_picker.shutil.which', lambda _: str(script))
    def write(body):
        script.write_text('#!'+sys.executable+'\n'+body)
        script.chmod(0o700)
    return write


def test_subprocess_selection_and_fixed_arguments(tmp_path, executable):
    selected = tmp_path/'空 格.py'
    selected.write_text('pass')
    args = tmp_path/'args.json'
    executable('import sys,json\nfrom pathlib import Path\nPath('+repr(str(args))+').write_text(json.dumps(sys.argv[1:]))\nprint('+repr(str(selected))+')\n')
    picker = NativePicker()
    result = asyncio.run(picker.choose('file', str(tmp_path), connected))
    assert result == {'cancelled': False, 'path': str(selected)}
    assert json.loads(args.read_text()) == ['--file-selection', '--title=Angelos — 选择文件', '--filename='+str(tmp_path)+'/']
    executable('print('+repr(str(tmp_path))+')\n')
    assert asyncio.run(picker.choose('directory', str(tmp_path), connected))['path'] == str(tmp_path)
    assert not picker.busy


@pytest.mark.parametrize('body,reason', [('raise SystemExit(1)', None), ('raise SystemExit(2)', 'native_picker_failed'), ('print("relative.py")', 'native_picker_invalid_selection')])
def test_cancel_and_invalid_result(tmp_path, executable, body, reason):
    executable(body)
    picker = NativePicker()
    if reason:
        with pytest.raises(ValueError, match=reason):
            asyncio.run(picker.choose('file', str(tmp_path), connected))
    else:
        assert asyncio.run(picker.choose('file', str(tmp_path), connected)) == {'cancelled': True}
    assert not picker.busy


@pytest.mark.parametrize('mode', ['timeout', 'disconnect', 'cancel'])
def test_pending_window_cleanup_and_busy(tmp_path, executable, mode):
    pidfile = tmp_path/'pid'
    executable('import os,time\nfrom pathlib import Path\nPath('+repr(str(pidfile))+').write_text(str(os.getpid()))\ntime.sleep(30)\n')
    async def run():
        picker = NativePicker(timeout=.3 if mode == 'timeout' else 3)
        async def disconnected():
            return mode == 'disconnect' and pidfile.exists()
        task = asyncio.create_task(picker.choose('file', str(tmp_path), disconnected))
        while not pidfile.exists():
            await asyncio.sleep(.01)
        with pytest.raises(ValueError, match='native_picker_busy'):
            await picker.choose('file', str(tmp_path), connected)
        if mode == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ValueError, match='native_picker_'+('timeout' if mode == 'timeout' else 'disconnected')):
                await task
        assert not picker.busy
        assert not Path('/proc/'+pidfile.read_text()).exists()
    asyncio.run(run())


def test_headless_and_missing_directory(tmp_path, executable, monkeypatch):
    picker = NativePicker()
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    assert not picker.available
    with pytest.raises(ValueError, match='native_picker_unavailable'):
        asyncio.run(picker.choose('file', str(tmp_path), connected))
    monkeypatch.setenv('DISPLAY', ':fixture')
    with pytest.raises(ValueError, match='native_picker_directory_missing'):
        asyncio.run(picker.choose('file', str(tmp_path/'missing'), connected))


def test_route_guards_and_no_workspace_mutation(tmp_path):
    app = create_angelos_app(tmp_path/'data')
    calls = []
    class Picker:
        available = True
        async def choose(self, kind, directory, disconnected):
            calls.append((kind, directory))
            return {'cancelled': False, 'path': str(tmp_path)}
    bench = app.state.angelos
    bench.native_picker = Picker()
    with TestClient(app) as client:
        token = client.get('/angelos/api/bootstrap').json()['token']
        headers = {'X-Angelos-Token': token}
        before = bench.store.list('workspace')
        data = {'kind':'directory','workspaceId':''}
        assert client.post('/angelos/api/native-picker', json=data).status_code == 403
        assert client.post('/angelos/api/native-picker', json=data, headers=headers|{'Origin':'https://outside.invalid'}).status_code == 403
        assert client.post('/angelos/api/native-picker', json=data|{'command':'bad'}, headers=headers).status_code == 409
        assert client.post('/angelos/api/native-picker', json={'kind':'file','workspaceId':''}, headers=headers).json()['reason'] == 'native_picker_workspace_required'
        result = client.post('/angelos/api/native-picker', json=data, headers=headers)
        assert result.json()['path'] == str(tmp_path)
        assert calls == [('directory', '')]
        assert bench.store.list('workspace') == before

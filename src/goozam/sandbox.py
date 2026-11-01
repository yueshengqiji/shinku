"""Linux 离线实验边界：只映射固定源码和专属数据，不共享主机网络或会话总线。

外层仅负责校验挂载与启动；子进程拥有新user/pid/net/ipc/uts命名空间，移除
capabilities。可选模型通道只挂载单个专用socket；不共享网络或代理目录。
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

WRITABLE = ('data', 'workspace', 'learning', 'evaluations', 'logs')


def plain_tree(path):
    """挂载前拒绝链接和特殊文件，避免把正式数据库的别名带入可写区域。"""
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('goozam_parent_traversal')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('goozam_linked_path')
    if not path.exists():
        return path
    for item in [path, *path.rglob('*')]:
        if item.is_symlink() or (not item.is_dir() and
                (not item.is_file() or item.stat().st_nlink != 1)):
            raise ValueError('goozam_unsafe_mount')
    return path


def command(lab, venv, *, action='probe', relay_socket=None, model_profile=None, proposal_refs=None):
    """固定入口；模型探针须显式提供socket，不接受工厂、网络或任意目录挂载。"""
    if action not in {'probe', 'exercise', 'model-probe', 'learn-propose', 'plugin-status', 'plugin-off', 'plugin-learn', 'plugin-use'}:
        raise ValueError('goozam_unknown_action')
    model_action = action in {'model-probe', 'learn-propose'}
    if model_action != (relay_socket is not None and model_profile is not None):
        raise ValueError('goozam_model_relay_required')
    if not model_action and (relay_socket is not None or model_profile is not None):
        raise ValueError('goozam_offline_relay_denied')
    if action == 'learn-propose':
        if (not isinstance(proposal_refs, list) or not 1 <= len(proposal_refs) <= 3
                or any(not isinstance(r, dict) or set(r) != {'session', 'task_id'}
                       or any(not isinstance(v, str) or not v or len(v) > 256 for v in r.values())
                       for r in proposal_refs)):
            raise ValueError('goozam_explicit_proposal_refs_required')
    elif proposal_refs is not None:
        raise ValueError('goozam_unexpected_proposal_refs')
    binary = shutil.which('bwrap')
    if not binary:
        raise RuntimeError('goozam_bubblewrap_required')
    lab = plain_tree(lab)
    manifest = json.loads((lab / 'manifest.json').read_text())
    if manifest.get('architecture') != 'goozam' or manifest.get('instance') != 'Angelos2':
        raise ValueError('goozam_instance_required')
    runtime = lab / 'runtime'
    actual = {p.relative_to(runtime).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob('*') if p.is_file()}
    if actual != manifest['files']:
        raise ValueError('goozam_runtime_drift')
    # 新实验必须同时冻结核心与学习扩展，否则主检出的后续修改会污染对照。
    # 旧实例没有这个字段，继续沿用显式启动器提供的扩展，避免改写历史副本。
    extension_source = manifest.get('extension_source', 'external')
    if extension_source not in {'runtime', 'external'}:
        raise ValueError('goozam_extension_source_invalid')
    extension = plain_tree(runtime / 'src/goozam' if extension_source == 'runtime'
                           else Path(__file__).parent)
    if not (extension / '__init__.py').is_file():
        raise ValueError('goozam_extension_missing')
    venv = Path(venv).resolve()
    if not (venv / 'pyvenv.cfg').is_file():
        raise ValueError('goozam_virtualenv_required')
    # 虚拟环境可能指向用户目录中的独立CPython。只映射其运行时，不映射/home；
    # site-packages显式进PYTHONPATH，不执行editable .pth来寻找宿主源码。
    info = json.loads(subprocess.check_output([str(venv / 'bin/python'), '-I', '-c',
        'import sys,json; print(json.dumps([sys.base_prefix, f"python{sys.version_info.major}.{sys.version_info.minor}"]))'],
        text=True, timeout=10))
    base, version = Path(info[0]).resolve(), info[1]
    executable = (venv / 'bin/python').resolve().relative_to(base)
    args = [binary, '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--clearenv', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib',
            '--symlink', 'usr/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp', '--dir', '/tmp/home', '--tmpfs', '/run', '--tmpfs', '/lab',
            '--ro-bind', str(base), '/python',
            '--ro-bind', str(venv / 'lib' / version / 'site-packages'), '/dependencies',
            '--ro-bind', str(runtime), '/runtime',
            '--ro-bind', str(extension), '/extensions/goozam',
            '--setenv', 'PATH', '/python/bin:/usr/bin', '--setenv', 'HOME', '/tmp/home',
            '--setenv', 'PYTHONHOME', '/python',
            '--setenv', 'PYTHONPATH', '/extensions:/runtime/src:/dependencies',
            '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'PYTHONNOUSERSITE', '1',
            '--setenv', 'SHINKU_ENV_FILE', '/dev/null']
    for name in WRITABLE:
        if not (lab / name).is_dir():
            raise ValueError('goozam_instance_directory_missing')
        args += ['--bind', str(lab / name), '/lab/' + name]
    entry = ['goozam.control_cli' if action.startswith('plugin-') else 'goozam.offline', action]
    if model_action:
        relay_socket = Path(relay_socket).absolute()
        # 代理目录包含socket，不能按普通文件树校验；仅校验路径链和这个节点。
        if '..' in relay_socket.parts or any(p.is_symlink() for p in (relay_socket, *relay_socket.parents)):
            raise ValueError('goozam_linked_relay')
        info = relay_socket.stat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or relay_socket.is_relative_to(lab)):
            raise ValueError('goozam_private_relay_required')
        if (not isinstance(model_profile, dict)
                or not {'model', 'context_tokens', 'max_output_tokens'} <= set(model_profile)
                or set(model_profile) - {'model', 'context_tokens', 'max_output_tokens', 'route_fingerprint'}):
            raise ValueError('goozam_invalid_model_profile')
        args += ['--ro-bind', str(relay_socket), '/run/goozam-model.sock']
        entry = (['goozam.proposal_probe', json.dumps(model_profile), json.dumps(proposal_refs)]
                 if action == 'learn-propose' else ['goozam.model_probe', json.dumps(model_profile)])
    return args + ['--chdir', '/lab/workspace', '--', '/python/' + str(executable), '-B', '-m', *entry]


def run(lab, venv, *, action='probe', timeout=120, relay_socket=None, model_profile=None, proposal_refs=None):
    """超时终止实验主进程；die-with-parent/PID命名空间回收其后代，不重试。"""
    return subprocess.run(command(lab, venv, action=action, relay_socket=relay_socket,
                                  model_profile=model_profile, proposal_refs=proposal_refs), check=True, timeout=timeout,
                          capture_output=True, text=True)

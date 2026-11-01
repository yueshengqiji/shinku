"""候选源码和测试边界。模型只能改候选，Git、沙箱、验收与提交由宿主持有。"""
from dataclasses import dataclass, field
from collections.abc import Callable
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('SHINKU_', 'LLM_', 'OPENAI_', 'ANTHROPIC_', 'GEMINI_', 'GIT_', 'PYTHON'))}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               PYTHONDONTWRITEBYTECODE='1', SHINKU_ENV_FILE=os.devnull)
    return env


def git(root, *args):
    return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(root), *args],
                                   env=clean_environment(), stderr=subprocess.STDOUT, timeout=60).decode().strip()


@dataclass(frozen=True)
class RepairPolicy:
    source_root: Path
    state_root: Path
    test_targets: tuple[str, ...]
    authorized: Callable[[], bool] = field(repr=False, compare=False)
    python: str = sys.executable
    # 运行冻结树与开发 HEAD 不同时，只覆盖宿主明确列出的公共 Python 原件。
    runtime_sources: tuple[tuple[str, Path], ...] = ()
    max_seconds: float = 300
    max_requests: int = 24
    max_tokens: int = 1500000
    max_tools: int = 96
    max_jobs_per_day: int = 2
    event_ttl_seconds: float = 3600
    batch_prefix: tuple[str, ...] = ()

    def __post_init__(self):
        source, state = Path(self.source_root).resolve(), Path(self.state_root).resolve()
        if (not source.is_dir() or not callable(self.authorized) or not isinstance(self.test_targets, tuple)
                or not self.test_targets or any(not isinstance(t, str) or not t for t in self.test_targets)
                or not isinstance(self.batch_prefix, tuple) or any(not isinstance(t, str) or not t for t in self.batch_prefix)
                or not isinstance(self.max_seconds, (int, float)) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0
                or any(type(v) is not int or v < 1 for v in (self.max_requests, self.max_tokens, self.max_tools, self.max_jobs_per_day))
                or not isinstance(self.event_ttl_seconds, (int, float)) or not math.isfinite(self.event_ttl_seconds)
                or self.event_ttl_seconds <= 0
                or state == source or state.is_relative_to(source / 'src') or state.is_relative_to(source / 'tests')):
            raise ValueError('invalid_maintenance_policy')
        if git(source, 'rev-parse', '--show-toplevel') != str(source):
            raise ValueError('maintenance_git_root_required')
        for name, path in self.runtime_sources:
            if not snapshotable(name) or not Path(path).is_file():
                raise ValueError('invalid_runtime_source_mapping')
        object.__setattr__(self, 'source_root', source)
        object.__setattr__(self, 'state_root', state)
        batch = source / 'scripts/run_desktop_batch.py'
        if not self.batch_prefix and batch.is_file():
            object.__setattr__(self, 'batch_prefix', (self.python, str(batch), '--lock-timeout', '30', '--'))

    def version(self):
        return hashlib.sha256(json.dumps([git(self.source_root, 'rev-parse', 'HEAD'),
            [(name, sha(path)) for name, path in self.runtime_sources]], separators=(',', ':')).encode()).hexdigest()

    def check(self):
        if not self.authorized():
            raise PermissionError('maintenance_authorization_revoked')


def patchable(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and not any(p in {'.', '..', '.git'} or p.startswith('.') for p in path.parts)
            and path.suffix == '.py' and path.parts[:2] == ('src', 'shinku'))


def snapshotable(name):
    path = PurePosixPath(name)
    return patchable(name) or (not path.is_absolute() and path.parts[0:1] == ('tests',)
        and path.suffix == '.py' and not any(p in {'.', '..', '.git'} or p.startswith('.') for p in path.parts))


class CandidateWorkspace:
    def __init__(self, policy, key, version):
        self.policy, self.version = policy, version
        # 同一截止时间覆盖模型、Agent 自测与宿主红绿验证，不能每一步重置预算。
        self.deadline = time.monotonic() + policy.max_seconds
        policy.check()
        if version != policy.version() or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('maintenance_source_version_changed')
        policy.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.home = policy.state_root / 'candidates' / key
        self.home.mkdir(parents=True, mode=0o700)
        self.tree = self.home / 'source'
        self.tree.mkdir(mode=0o700)
        commit = git(policy.source_root, 'rev-parse', 'HEAD')
        archive = self.home / 'source.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', '-C', str(policy.source_root), 'archive', commit], check=True,
                           stdout=stream, env=clean_environment(), timeout=60)
        with tarfile.open(archive) as tar:
            for item in tar.getmembers():
                parts = PurePosixPath(item.name).parts
                if item.issym() or item.islnk() or not (item.isfile() or item.isdir()) or 'private' in parts:
                    raise ValueError('unsupported_maintenance_snapshot_member')
            tar.extractall(self.tree, filter='data')
        for name, path in policy.runtime_sources:
            target = self.tree / name
            expected = sha(path)
            payload = Path(path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('maintenance_runtime_source_changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        git(self.tree, 'init', '-q')
        git(self.tree, 'config', 'user.name', 'Shinku maintenance')
        git(self.tree, 'config', 'user.email', 'maintenance@localhost')
        git(self.tree, 'add', '--', '.')
        git(self.tree, 'commit', '-qm', 'Host-frozen maintenance baseline')
        self.baseline = git(self.tree, 'rev-parse', 'HEAD')
        self.regression = 'tests/test_maintenance_' + key[:16] + '.py'
        test = self.tree / self.regression
        test.parent.mkdir(exist_ok=True)
        if test.exists():
            raise ValueError('maintenance_regression_collision')
        test.write_text('def test_original_failure():\n    raise AssertionError("replace with a reproducible regression")\n')
        self.original = {str(p.relative_to(self.tree)): sha(p) for p in self.tree.rglob('*')
                         if p.is_file() and '.git' not in p.parts and p != test}
        self.receipts = []
        runtime = json.loads(subprocess.check_output([policy.python, '-I', '-c',
            'import sys,site,json; print(json.dumps(dict(base=sys.base_prefix,sites=site.getsitepackages())))'], text=True, timeout=10))
        self._base_prefix, self._sites = runtime['base'], runtime['sites']
        self._executable = str(Path(policy.python).resolve(strict=True))
        self._venv = str(Path(policy.python).absolute().parent.parent)
        (self.home / 'baseline.json').write_text(json.dumps(dict(source_commit=commit,
            source_version=version, baseline_commit=self.baseline, files=self.original), indent=2))
        self.check()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('maintenance_deadline_exceeded')
        self.policy.check()
        if self.version != self.policy.version():
            raise ValueError('maintenance_source_version_changed')

    def changes(self):
        """不接受已有测试、宿主规则、配置、Git 元数据或链接的修改。"""
        self.check()
        paths = set(self.original)
        paths.update(str(p.relative_to(self.tree)) for p in self.tree.rglob('*')
                     if (p.is_file() or p.is_symlink()) and '.git' not in p.parts)
        changed = []
        for name in paths:
            p = self.tree / name
            if p.is_symlink():
                raise ValueError('maintenance_symlink_rejected')
            if not p.is_file():
                raise ValueError('maintenance_deletion_rejected')
            if self.original.get(name) == sha(p):
                continue
            if not patchable(name) and name != self.regression:
                raise ValueError('maintenance_edit_outside_scope')
            changed.append(name)
        return sorted(changed)

    def tests(self, tree, targets, label):
        """只读挂载候选；隔离宿主网络和主目录，仅临时目录与本次 XML 可写。"""
        self.check()
        if shutil.which('bwrap') is None or shutil.which('prlimit') is None:
            raise RuntimeError('maintenance_sandbox_unavailable')
        results = self.home / label
        results.mkdir(mode=0o700)
        xml = results / 'tests.xml'
        xml.touch(mode=0o600, exist_ok=False)
        log = self.home / (label + '.log')
        command = [*self.policy.batch_prefix, 'prlimit', '--as=2147483648', '--fsize=16777216',
            '--cpu=' + str(math.ceil(self.policy.max_seconds) + 5), '--',
            'bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib', '/lib64',
            '--symlink', 'usr/bin', '/bin', '--proc', '/proc', '--dev', '/dev', '--size', '67108864', '--tmpfs', '/tmp',
            '--dir', '/tmp/home', '--ro-bind', self._base_prefix, self._base_prefix,
            '--ro-bind', self._venv, self._venv, '--ro-bind', str(tree), '/workspace',
            # 结果目录只读，仅一个预建 XML 文件可写，不能换链接诱使宿主读取私用文件。
            '--ro-bind', str(results), '/results', '--bind', str(xml), '/results/tests.xml',
            '--chdir', '/workspace', '--clearenv',
            '--setenv', 'HOME', '/tmp/home', '--setenv', 'PATH', '/usr/bin:/bin',
            '--setenv', 'PYTHONPATH', ':'.join(['/workspace/src', '/workspace/tests', *self._sites]),
            '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1',
            '--setenv', 'SHINKU_ENV_FILE', '/dev/null', '--setenv', 'SHINKU_DATA_ROOT', '/tmp/data',
            '--setenv', 'SHINKU_CONFIG_ROOT', '/tmp/config', '--setenv', 'SHINKU_LOG_ROOT', '/tmp/logs',
            self._executable, '-B', '-m', 'pytest', '-o', 'addopts=', '-p', 'no:cacheprovider',
            '-q', '--tb=short', *targets, '--junitxml=/results/tests.xml']
        # start_new_session 让超时清理覆盖沙箱子进程；不触及任何正常业务进程。
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=clean_environment(), start_new_session=True)
            try:
                deadline = self.deadline
                while process.poll() is None:
                    self.check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('maintenance_test_timeout')
                    time.sleep(.05)
                code = process.returncode
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        self.check()
        counts = dict(tests=0, failures=0, errors=0, skipped=0)
        if xml.is_symlink() or xml.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('maintenance_test_receipt_invalid')
        if xml.stat().st_size:
            for suite in ET.parse(xml).getroot().iter('testsuite'):
                for name in counts:
                    counts[name] += int(suite.get(name, '0'))
        receipt = dict(label=label, exit=code, **counts, log_sha256=sha(log),
                       xml_sha256=sha(xml) if xml.is_file() else '',
                       output_tail=log.read_text(errors='replace')[-6000:])
        self.receipts.append(receipt)
        (results / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt

    def verify_and_commit(self):
        changed = self.changes()
        if self.regression not in changed or not any(patchable(p) for p in changed):
            raise ValueError('maintenance_fix_and_regression_required')
        fixed = self.tests(self.tree, (*self.policy.test_targets, self.regression), 'host-fixed')
        old = self.home / 'baseline-source'
        shutil.copytree(self.tree, old, ignore=shutil.ignore_patterns('.git'))
        for name in changed:
            if patchable(name):
                (old / name).write_bytes(subprocess.check_output(
                    ['git', '-C', str(self.tree), 'show', self.baseline + ':' + name], env=clean_environment()))
        original = self.tests(old, self.policy.test_targets, 'host-original')
        red = self.tests(old, (self.regression,), 'host-regression-original')
        if (fixed['exit'] != 0 or original['exit'] != 0 or fixed['tests'] < 1 or original['tests'] < 1
                or fixed['failures'] or fixed['errors'] or original['failures'] or original['errors']
                or red['exit'] != 1 or red['failures'] < 1 or red['errors']):
            raise ValueError('maintenance_independent_validation_failed')
        expected = {name: sha(self.tree / name) for name in changed}
        self.check()
        if self.changes() != changed or any(sha(self.tree / n) != d for n,d in expected.items()):
            raise ValueError('maintenance_candidate_drift')
        git(self.tree, 'add', '--', *changed)
        git(self.tree, 'diff', '--cached', '--check')
        git(self.tree, 'commit', '-qm', 'Repair retained host failure with regression')
        commit = git(self.tree, 'rev-parse', 'HEAD')
        for name, digest in expected.items():
            blob = subprocess.check_output(['git', '-C', str(self.tree), 'show', commit + ':' + name], env=clean_environment())
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError('maintenance_commit_mismatch')
        return dict(candidate=str(self.tree), commit=commit, source_version=self.version, files=expected,
                    checks=[dict((k,v) for k,v in row.items() if k != 'output_tail') for row in (original, red, fixed)],
                    deployed=False, original_task_replayed=False, messages_sent=0)

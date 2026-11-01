"""本机工作区文件事实；Git 对照 HEAD，普通目录对照首次观察快照。

不从模型文字推断改动或归因。目录与文件数、单文件大小均有界；符号链接、
目录穿越和二进制拒绝。编辑带内容版本，原件先归档，写入原目录并原子替换。
"""
from contextlib import contextmanager
import difflib
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import threading
import time
import uuid

LIMIT = 512 * 1024
MAX_FILES = 2000
SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache', '.idea'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text(data):
    if len(data) > LIMIT or b'\0' in data:
        raise ValueError('file_binary_or_oversized')
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('file_binary_or_oversized') from exc


class WorkspaceFiles:
    def __init__(self, store, permissions=None):
        self.store, self.permissions = store, permissions
        self.lock = threading.RLock()

    def root(self, wid):
        value = self.store.read('workspace', wid)['path']
        if not value or not Path(value).is_dir():
            raise ValueError('workspace_directory_missing')
        return Path(value)

    @contextmanager
    def parent(self, root, relative):
        """逐层 no-follow 打开目录；后续操作都使用目录描述符，不跟随替换后的路径。"""
        # 类型先于任何路径构造判定：非字符串入参同样属于越界，必须抛既定的
        # ValueError，不能让 TypeError 从闸门里漏出去被上层当成非越界错误。
        if not isinstance(relative, str) or not relative or '\\' in relative:
            raise ValueError('file_path_outside_workspace')
        path = PurePosixPath(relative)
        if (path.is_absolute()
                or any(p in {'.', '..', ''} or p in SKIP for p in relative.split('/'))):
            raise ValueError('file_path_outside_workspace')
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, path.name
        finally:
            os.close(fd)

    @staticmethod
    def read_at(fd, name):
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(handle, 'rb') as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError('file_type_unsupported')
            data = stream.read(LIMIT + 1)
        text(data)
        return data, metadata

    def read(self, root, relative):
        try:
            with self.parent(root, relative) as (fd, name):
                return self.read_at(fd, name)[0]
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError('file_path_unavailable') from exc

    @staticmethod
    def git(root, *args):
        # 只运行固定 Git 查询；路径只作为独立 argv，禁用外部 diff 和 pager。
        result = subprocess.run(['git', '--no-pager', '-c', 'core.fsmonitor=false',
            '-c', 'core.hooksPath=/dev/null', '-C', str(root), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'})
        if result.returncode:
            raise ValueError('workspace_git_unavailable')
        if len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError('workspace_scan_limit')
        return result.stdout

    def baseline(self, wid, *, scan=True):
        root = self.root(wid)
        key = 'file-baseline:' + digest(str(root).encode())
        try:
            return root, self.store.read('file_baseline', key)
        except ValueError:
            pass
        files, total = {}, 0
        try:
            git_root = Path(self.git(root, 'rev-parse', '--show-toplevel').decode().strip())
            if git_root != root:
                raise ValueError('nested_workspace')
            mode = 'git'
        except (ValueError, subprocess.SubprocessError):
            mode = 'snapshot'
        if mode == 'snapshot' and not scan:
            # 单文件预览不应依赖全目录扫描；尚无快照时明确无历史对照，
            # 不把局部读到的文件冒充完整目录基线或写回基线存储。
            return root, {'mode': 'unavailable', 'files': {}}
        if mode == 'snapshot':
            for relative in self.walk(root):
                try:
                    raw = self.read(root, relative)
                    if raw is None:
                        continue
                    total += len(raw)
                    if total > 16 * 1024 * 1024:
                        raise ValueError('workspace_snapshot_limit')
                    files[relative] = text(raw)
                except ValueError as exc:
                    if str(exc) in {'workspace_snapshot_limit', 'workspace_scan_limit'}:
                        raise
        baseline = {'id': key, 'root': str(root), 'mode': mode, 'files': files, 'created': time.time()}
        with self.store.transaction() as db:
            self.store.put(db, 'file_baseline', baseline)
        return root, baseline

    def walk(self, root):
        paths = []
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory)/d).is_symlink())
            for name in sorted(files):
                path = Path(directory)/name
                if path.is_symlink():
                    continue
                paths.append(path.relative_to(root).as_posix())
                if len(paths) > MAX_FILES:
                    raise ValueError('workspace_scan_limit')
        return paths

    def original(self, root, baseline, relative):
        if baseline['mode'] != 'git':
            return baseline['files'].get(relative)
        try:
            size = int(self.git(root, 'cat-file', '-s', 'HEAD:' + relative).strip())
            if size > LIMIT:
                raise ValueError('file_binary_or_oversized')
            return text(self.git(root, 'show', 'HEAD:' + relative))
        except ValueError as exc:
            if str(exc) == 'workspace_git_unavailable':
                return None  # 新文件或尚无初始提交。
            raise

    def changes(self, wid):
        with self.lock:
            root, baseline = self.baseline(wid)
            if baseline['mode'] == 'git':
                raw = self.git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all', '--no-renames', '--', '.')
                paths = sorted({r[3:].decode('utf-8') for r in raw.split(b'\0') if r})
            else:
                paths = sorted(set(self.walk(root)) | set(baseline['files']))
            if len(paths) > MAX_FILES:
                raise ValueError('workspace_scan_limit')
            rows, skipped = [], []
            for relative in paths:
                try:
                    raw = self.read(root, relative)
                    current = None if raw is None else text(raw)
                    original = self.original(root, baseline, relative)
                    if original == current:
                        continue
                    plus = minus = 0
                    for tag, a, b, c, d in difflib.SequenceMatcher(None,
                            (original or '').splitlines(), (current or '').splitlines(), autojunk=True).get_opcodes():
                        if tag != 'equal':
                            minus += b-a
                            plus += d-c
                    rows.append({'path': relative, 'added': plus, 'removed': minus,
                        'status': 'added' if original is None else 'deleted' if current is None else 'modified',
                        'version': digest(raw) if raw is not None else None})
                except (ValueError, OSError):
                    skipped.append(relative)
            return {'workspaceId': wid, 'baseline': 'HEAD' if baseline['mode'] == 'git' else '首次观察快照',
                'files': rows, 'skipped': skipped, 'observedAt': time.time(),
                'added': sum(r['added'] for r in rows), 'removed': sum(r['removed'] for r in rows)}

    def open(self, wid, relative):
        with self.lock:
            root = self.root(wid)
            raw = self.read(root, relative)  # 路径门必须早于 git show 和目录基线。
            _, baseline = self.baseline(wid, scan=False)
            original = self.original(root, baseline, relative)
            if raw is None and original is None:
                raise ValueError('file_not_found')
            return {'path': relative, 'content': text(raw) if raw is not None else '',
                'original': (text(raw) if raw is not None and baseline['mode'] == 'unavailable' else original or ''),
                'comparisonAvailable': baseline['mode'] != 'unavailable', 'exists': raw is not None,
                'version': digest(raw) if raw is not None else None,
                'editable': raw is not None and self.can_edit()}

    def can_edit(self):
        return self.permissions is not None and self.permissions.level in {'waw', 'aleph'}

    def save(self, wid, relative, content, version):
        if not isinstance(content, str) or not isinstance(version, str):
            raise ValueError('invalid_file_edit')
        data = content.encode('utf-8')
        text(data)
        with self.lock:
            if not self.can_edit():
                raise PermissionError('file_edit_not_permitted')
            root = self.root(wid)
            with self.parent(root, relative) as (fd, name):
                previous, metadata = self.read_at(fd, name)
                if digest(previous) != version:
                    raise ValueError('file_version_conflict')
                if metadata.st_nlink != 1:
                    raise ValueError('file_hardlink_not_editable')
                # 用户保存是明确人工写入，不能伪造 Agent 的执行/验收回执。
                with self.store.transaction() as db:
                    self.store.put(db, 'manual_file_edit', {'id': uuid.uuid4().hex, 'workspaceId': wid,
                        'path': relative, 'before': text(previous), 'beforeVersion': version,
                        'afterVersion': digest(data), 'created': time.time(), 'source': 'user_editor'})
                temporary = '.angelos-edit-' + uuid.uuid4().hex
                try:
                    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
                    with os.fdopen(handle, 'wb') as stream:
                        stream.write(data)
                        stream.flush()
                        os.fchmod(stream.fileno(), stat.S_IMODE(metadata.st_mode) & 0o777)
                        os.fsync(stream.fileno())
                    fresh, info = self.read_at(fd, name)
                    if digest(fresh) != version or (info.st_ino, info.st_dev) != (metadata.st_ino, metadata.st_dev):
                        raise ValueError('file_version_conflict')
                    if not self.can_edit():
                        raise PermissionError('file_edit_not_permitted')
                    os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=fd)
                    except FileNotFoundError:
                        pass
            return self.open(wid, relative)

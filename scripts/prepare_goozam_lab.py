"""准备 goozam 的 Angelos2 实验副本；不导入 Agent、不启动服务或读取正式配置。

只从固定 Git 提交导出公开源码。实例目录独占创建，禁止覆盖、软链接或硬链接
复用；数据目录从空白开始。此处仅保证副本隔离，不充当进程或工具执行沙箱。
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


EXPORTS = ('src', 'frontend', 'pyproject.toml', 'README.md', 'LICENSE')
EMPTY_DIRS = ('data', 'config', 'logs', 'workspace', 'learning', 'evaluations')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def prepare(repo, destination, revision='HEAD'):
    """冻结源码及逐文件摘要；失败保留部分产物供检查，绝不清理已有实例。

    目的地只允许在本仓 private/experiments/goozam 下。启动工厂、模型密钥、
    会话、任务队列、浏览器登录态及记忆不进入副本，后续必须独立显式装配。
    """
    repo = Path(repo).resolve()
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    if repo != root:
        raise ValueError('repository_root_required')
    allowed = root / 'private' / 'experiments' / 'goozam'
    destination = Path(destination).absolute()
    if '..' in destination.parts or not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError('experimental_directory_required')
    # 在创建任何目录前检查完整祖先链；不能因 resolve 消去软链接而绕过边界。
    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise ValueError('symlink_not_allowed')
    if destination.exists():
        raise FileExistsError(destination)
    commit = git(root, 'rev-parse', '--verify', revision + '^{commit}').decode().strip()
    archive = git(root, 'archive', '--format=tar', commit, '--', *EXPORTS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        entries = source.getmembers()
        # 只接受普通源码文件，不执行归档里的脚本，不跟随链接或设备节点。
        for entry in entries:
            path = PurePosixPath(entry.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] not in EXPORTS or not (entry.isfile() or entry.isdir())):
                raise ValueError('unsafe_archive_entry')
        destination.mkdir(parents=True, mode=0o700, exist_ok=False)
        (destination / 'runtime').mkdir(mode=0o700)
        hashes = {}
        for entry in entries:
            target = destination / 'runtime' / entry.name
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = source.extractfile(entry).read()
            with target.open('xb') as stream:
                stream.write(raw)
            hashes[entry.name] = hashlib.sha256(raw).hexdigest()
    for name in EMPTY_DIRS:
        (destination / name).mkdir(mode=0o700)
    manifest = {
        'architecture': 'goozam', 'instance': 'Angelos2', 'schema': 1,
        'source_commit': commit, 'source_kind': 'public_git_snapshot',
        'state': 'prepared_not_running', 'model_connected': False,
        'production_factory_copied': False, 'production_data_copied': False,
        'process_sandbox': 'not_implemented', 'suggested_port': 8771,
        # 新版本将学习扩展和执行核心固定在同一快照；旧提交仍保留原接线模式。
        'extension_source': ('runtime' if 'src/goozam/__init__.py' in hashes else 'external'),
        'files': hashes,
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (destination / 'README.txt').write_text(
        'Angelos2 / goozam 实验基线\n'
        '仅准备，尚未启动；不能声称自主学习或执行沙箱已经接通。\n'
        'runtime 是固定提交的公开源码，非正式运行目录的逐字克隆。\n'
        'data/config/logs/workspace/learning/evaluations 均从空白开始。\n'
        '不要复用 existing_agent:factory、正式数据、私用插件或委托连接配置。\n'
        '模型/工具接入前须实现独立工厂与工作区外访问限制，实验不自动回灌正式。\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    args = parser.parse_args()
    manifest = prepare(args.repo, args.destination, args.revision)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False))
    print('source_files:', len(manifest['files']))


if __name__ == '__main__':
    main()

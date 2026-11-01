"""下载执行回执绑定的文件；相对路径仅按原任务工作区解析，不使用当前选择。

逐层no-follow读取并核对内容摘要。历史版本被覆盖时如实拒绝，不能把最新文件
冒充旧产物；此处没有执行、修改文件或扩展任务授权的入口。
"""
import hashlib
import os
from pathlib import Path
import stat


def read_artifact(artifact, workspace):
    raw, expected = artifact.get('path'), artifact.get('sha256')
    if not isinstance(raw, str) or not raw or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('artifact_unavailable')
    path = Path(raw)
    if path.is_absolute():
        root, parts = Path('/'), path.parts[1:]
    else:
        root = Path(workspace.get('path') or '')
        parts = raw.split('/')
        if not root.is_absolute():
            raise ValueError('artifact_unavailable')
    if not parts or any(p in ('', '.', '..') or '\\' in p for p in parts):
        raise ValueError('artifact_unavailable')
    fd = None
    try:
        # 根目录也逐层打开，避免祖先符号链接悄悄改变已保存的文件归属。
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        for part in (*root.parts[1:], *parts[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('artifact_unavailable')
            data = stream.read(32 * 1024 * 1024 + 1)
    except OSError as exc:
        raise ValueError('artifact_unavailable') from exc
    finally:
        if fd is not None:
            os.close(fd)
    if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('artifact_changed_or_oversized')
    return parts[-1], data

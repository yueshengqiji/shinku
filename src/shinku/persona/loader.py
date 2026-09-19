"""主人设文档的显式加载器。

主人设是独立项目的输入资产，不从旧项目路径、包内默认值或聊天历史猜测。
调用方必须显式提供 ``SHINKU_PERSONA_FILE``（或直接传路径），这样发布包不会
悄悄把来源项目的人设带进来，也便于在 ``doctor`` 中确认当前实际使用的文件。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["PersonaDocument", "PersonaLoadError", "load_persona", "load_persona_from_env"]


class PersonaLoadError(ValueError):
    """主人设文件不存在、不可读或不满足边界时抛出。"""


@dataclass(frozen=True)
class PersonaDocument:
    """一次加载后的人设快照；不暴露任何旧项目回退逻辑。"""

    path: Path
    text: str
    byte_size: int
    sha256: str

    @property
    def configured(self) -> bool:
        return bool(self.text.strip())

    def diagnostics(self) -> dict[str, object]:
        """可供 doctor 展示的安全信息，不回显人设正文。"""

        return {
            "configured": self.configured,
            "path": str(self.path),
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


def load_persona(path: str | Path, *, max_bytes: int = 256 * 1024) -> PersonaDocument:
    """读取一份 UTF-8 人设文档。"""

    raw_path = str(path or "").strip()
    if not raw_path:
        raise PersonaLoadError("persona_file_missing")
    if max_bytes <= 0:
        raise PersonaLoadError("persona_max_bytes_invalid")
    file_path = Path(raw_path).expanduser()
    try:
        if not file_path.is_file():
            raise PersonaLoadError("persona_file_not_found")
        payload = file_path.read_bytes()
    except PersonaLoadError:
        raise
    except OSError as exc:
        raise PersonaLoadError("persona_file_unreadable") from exc
    if not payload:
        raise PersonaLoadError("persona_file_empty")
    if len(payload) > max_bytes:
        raise PersonaLoadError("persona_file_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PersonaLoadError("persona_file_not_utf8") from exc
    if not text.strip():
        raise PersonaLoadError("persona_file_empty")
    return PersonaDocument(
        path=file_path.resolve(),
        text=text.strip(),
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_persona_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    key: str = "SHINKU_PERSONA_FILE",
) -> PersonaDocument:
    """从显式环境变量加载主人设；没有配置时保持失败而不是自动回退。"""

    env = os.environ if environ is None else environ
    return load_persona(str(env.get(key, "") or ""))

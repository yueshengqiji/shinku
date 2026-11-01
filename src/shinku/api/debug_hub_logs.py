"""Bounded log reading and secret redaction for the local debug hub."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

_MAX_LOG_LINES = 120
_MAX_LOG_BYTES = 256 * 1024
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(token\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(password\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
)


def _redact_line(value: str) -> str:
    result = str(value or "")
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _tail(path: Path, *, limit: int = _MAX_LOG_LINES) -> list[str]:
    """读日志末尾，避免控制中枢因日志文件变大而占满内存。"""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _MAX_LOG_BYTES), os.SEEK_SET)
            payload = handle.read(_MAX_LOG_BYTES)
    except OSError:
        return []
    text = payload.decode("utf-8", errors="replace")
    lines = [_redact_line(line.rstrip()) for line in text.splitlines() if line.strip()]
    return lines[-max(1, min(int(limit), _MAX_LOG_LINES)) :]


def _log_files(settings: Any) -> list[Path]:
    root = Path(getattr(settings, "log_root", "") or "")
    if not root.is_dir():
        return []
    candidates = {
        Path(settings.log_path("backend")),
        Path(settings.log_path("backend", error=True)),
    }
    # 外部启动器可能把 stdout/stderr 重定向到带端口或批次名的文件。
    candidates.update(root.glob("backend*.log"))
    existing = [path for path in candidates if path.is_file()]
    return sorted(existing, key=lambda item: item.stat().st_mtime, reverse=True)[:8]


def _safe_call(callable_obj: Any, default: Any) -> Any:
    try:
        return callable_obj() if callable(callable_obj) else default
    except Exception:
        return default

"""NapCat 连接配置与不联网预检。"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping
from urllib.parse import urlsplit

__all__ = ["NapCatConnectionConfig"]


def normalize_image_path_mappings(rows) -> tuple[tuple[str, str], ...]:
    """显式容器缓存映射；拒绝根目录映射、相对路径和上级跳转，不自动猜宿主路径。"""
    if not isinstance(rows, (tuple, list)) or len(rows) > 16:
        raise ValueError("invalid_napcat_image_path_mappings")
    result = []
    for pair in rows:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(p, str) or not p or '\x00' in p for p in pair)):
            raise ValueError("invalid_napcat_image_path_mappings")
        remote, local = pair
        source = PurePosixPath(remote)
        if (not source.is_absolute() or str(source) == '/' or '..' in source.parts
                or not Path(local).is_absolute()):
            raise ValueError("invalid_napcat_image_path_mappings")
        result.append((str(source), str(Path(local))))
    return tuple(result)


@dataclass(frozen=True)
class NapCatConnectionConfig:
    """NapCat HTTP action caller 所需的最小配置。"""

    base_url: str = ""
    access_token: str = ""
    access_token_file: str = ""
    timeout: float = 30.0
    webhook_enabled: bool = False
    webhook_path: str = "/events/napcat"
    event_token: str = ""
    image_roots: tuple[str, ...] = ()
    image_path_mappings: tuple[tuple[str, str], ...] = ()
    file_roots: tuple[str, ...] = ()
    file_path_mappings: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "NapCatConnectionConfig":
        env = os.environ if environ is None else environ
        raw_timeout = str(env.get("SHINKU_NAPCAT_TIMEOUT", "30") or "30").strip()
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 30.0
        raw_roots = str(env.get("SHINKU_NAPCAT_IMAGE_ROOTS", "") or "")
        try:
            paths = json.loads(env.get("SHINKU_NAPCAT_IMAGE_PATH_MAPPINGS", "{}") or "{}")
            if not isinstance(paths, dict):
                raise ValueError
            mappings = normalize_image_path_mappings(tuple(paths.items()))
        except (TypeError, ValueError):
            raise ValueError("invalid_napcat_image_path_mappings") from None
        try:
            file_paths = json.loads(env.get("SHINKU_NAPCAT_FILE_PATH_MAPPINGS", "{}") or "{}")
            if not isinstance(file_paths, dict):
                raise ValueError
            file_mappings = normalize_image_path_mappings(tuple(file_paths.items()))
        except (TypeError, ValueError):
            raise ValueError("invalid_napcat_file_path_mappings") from None
        return cls(
            base_url=str(env.get("SHINKU_NAPCAT_BASE_URL", "") or "").strip(),
            access_token=str(env.get("SHINKU_NAPCAT_ACCESS_TOKEN", "") or "").strip(),
            access_token_file=str(env.get("SHINKU_NAPCAT_ACCESS_TOKEN_FILE", "") or "").strip(),
            timeout=timeout,
            webhook_enabled=str(env.get("SHINKU_NAPCAT_WEBHOOK_ENABLED", "") or "").strip().lower()
            in {"1", "true", "yes", "on"},
            webhook_path=str(env.get("SHINKU_NAPCAT_WEBHOOK_PATH", "/events/napcat") or "/events/napcat").strip(),
            event_token=str(env.get("SHINKU_NAPCAT_EVENT_TOKEN", "") or "").strip(),
            image_roots=tuple(item.strip() for item in raw_roots.split(",") if item.strip()),
            image_path_mappings=mappings,
            file_roots=tuple(item.strip() for item in env.get("SHINKU_NAPCAT_FILE_ROOTS", "").split(",") if item.strip()),
            file_path_mappings=file_mappings,
        )

    def token_value(self) -> str:
        """直接 token 优先；文件读取只在显式组装 HTTP caller 时发生。"""

        if self.access_token:
            return self.access_token
        if self.access_token_file:
            path = Path(self.access_token_file)
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except OSError:
                    return ""
        return ""

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        value = self.base_url.strip()
        if not value:
            errors.append("base_url_missing")
        else:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"}:
                errors.append("base_url_scheme_invalid")
            if not parsed.hostname:
                errors.append("base_url_host_missing")
            if parsed.username or parsed.password:
                errors.append("base_url_userinfo_forbidden")
        if self.timeout <= 0:
            errors.append("timeout_invalid")
        if self.webhook_enabled and not self.webhook_path.startswith("/"):
            errors.append("webhook_path_invalid")
        try:
            normalize_image_path_mappings(self.image_path_mappings)
        except ValueError:
            errors.append("image_path_mappings_invalid")
        try:
            normalize_image_path_mappings(self.file_path_mappings)
        except ValueError:
            errors.append("file_path_mappings_invalid")
        return tuple(errors)

    def diagnostics(self) -> dict[str, object]:
        """返回可展示的安全快照，不返回 token 内容。"""

        parsed = urlsplit(self.base_url.strip())
        endpoint = ""
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            host = parsed.hostname
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            endpoint = f"{parsed.scheme}://{host}{parsed.path.rstrip('/')}"
        return {
            "configured": bool(self.base_url.strip()),
            "valid": not self.validate(),
            "errors": list(self.validate()),
            "webhook_enabled": self.webhook_enabled,
            "webhook_path": self.webhook_path,
            "endpoint": endpoint,
            "token_present": bool(self.access_token or self.access_token_file),
            "event_token_present": bool(self.event_token),
            "image_root_count": len(self.image_roots),
            "image_path_mapping_count": len(self.image_path_mappings),
            "file_root_count": len(self.file_roots),
            "file_path_mapping_count": len(self.file_path_mappings),
            "timeout": self.timeout,
        }

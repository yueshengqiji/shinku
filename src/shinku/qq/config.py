"""NapCat 连接配置与不联网预检。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

__all__ = ["NapCatConnectionConfig"]


@dataclass(frozen=True)
class NapCatConnectionConfig:
    """NapCat HTTP action caller 所需的最小配置。"""

    base_url: str = ""
    access_token: str = ""
    access_token_file: str = ""
    timeout: float = 10.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "NapCatConnectionConfig":
        env = os.environ if environ is None else environ
        raw_timeout = str(env.get("SHINKU_NAPCAT_TIMEOUT", "10") or "10").strip()
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 10.0
        return cls(
            base_url=str(env.get("SHINKU_NAPCAT_BASE_URL", "") or "").strip(),
            access_token=str(env.get("SHINKU_NAPCAT_ACCESS_TOKEN", "") or "").strip(),
            access_token_file=str(env.get("SHINKU_NAPCAT_ACCESS_TOKEN_FILE", "") or "").strip(),
            timeout=timeout,
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
            "endpoint": endpoint,
            "token_present": bool(self.access_token or self.access_token_file),
            "timeout": self.timeout,
        }

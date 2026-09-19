"""配置加载。

设计意图
--------
只从**环境变量 + 本模块的默认值**取值：

- 不读旧项目的 ``config.py``；
- 不自动加载旧项目的 ``.env``（仓库根那份混着 Akane 时代的键名）。

要覆盖默认值就设环境变量；要一份可提交的样例，看仓库根的 ``.env.example``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from . import __version__
from . import names


@dataclass(frozen=True)
class Settings:
    """一次解析结果。不可变——避免运行期被随意改写。"""

    project: str
    version: str
    config_root: Path
    data_root: Path
    log_root: Path
    service_token: str
    service_token_file: str
    allow_lan: bool
    ports: dict[str, int]
    binds: dict[str, str]

    def url(self, service: str) -> str:
        return f"http://127.0.0.1:{self.ports[service]}"

    def log_path(self, service: str, *, error: bool = False) -> Path:
        suffix = names.LOG_ERROR_SUFFIX if error else names.LOG_SUFFIX
        return self.log_root / f"{service}{suffix}"

    def service_token_value(self, environ: Mapping[str, str] | None = None) -> str:
        """解析服务令牌：直接值优先，其次读令牌文件。都没有则返回空串。"""

        if self.service_token:
            return self.service_token
        if self.service_token_file:
            path = Path(self.service_token_file)
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
        return ""


def load_project_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    path: str | Path | None = None,
) -> Path | None:
    """读取独立项目 ``.env``，只接纳 ``SHINKU_*`` 键。

    shell 环境优先于文件；这样不会用仓库里的样例覆盖启动器或服务管理器已经
    注入的值。该函数不做变量展开，也不打印任何 value，避免把 token 带进日志。
    """

    env = os.environ if environ is None else environ
    configured_path = path or env.get("SHINKU_ENV_FILE") or ".env"
    env_path = Path(configured_path)
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("SHINKU_") or not key.replace("_", "").isalnum():
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env.setdefault(key, value)
    return env_path


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ
    return Settings(
        project=names.PROJECT_NAME,
        version=__version__,
        config_root=names.config_root(env),
        data_root=names.data_root(env),
        log_root=names.log_root(env),
        service_token=str(env.get(names.SERVICE_TOKEN_ENV, "") or "").strip(),
        service_token_file=str(env.get(names.SERVICE_TOKEN_FILE_ENV, "") or "").strip(),
        allow_lan=names.allow_lan(env),
        ports={s: names.service_port(s, env) for s in names.SERVICE_NAMES},
        binds={s: names.service_bind(s, env) for s in names.SERVICE_NAMES},
    )


def ensure_directories(settings: Settings) -> list[Path]:
    """建立可变数据目录（数据 / 配置 / 日志），返回实际创建的目录列表。

    ``doctor`` 用它来证明"新树可以在没有旧项目的情况下把运行所需目录建起来"。
    """

    created: list[Path] = []
    for path in (settings.data_root, settings.config_root, settings.log_root):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created

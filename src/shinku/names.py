"""Shinku 的命名契约。

这是全项目**唯一**一处定义"名字、默认值和端口"的地方。

为什么要单独一个模块
--------------------
早期实现里同一含义的字符串散落在多个文件中——`os.getenv("SHINKU_RESPONSE_HOST_PORT", "9995")`
在不同文件里各写一遍，改一个默认值要全局搜。这里把它们收成一份**声明**：
其它模块只从本模块取值，不再自己写字面量。

不继承的两个历史命名（重要）
--------------------------
1. ``COMPANION_HOST`` / ``COMPANION_PORT``
   早期运行时的旧命名。新项目**不使用**；后端入口统一走 ``SHINKU_BACKEND_*``。
2. ``SHINKU_SERVER_HOST`` / ``SHINKU_SERVER_PORT``
   历史实现里它指的是 **Agent 宿主**的 9100 端口，名字里的 "SERVER" 与实际语义不符。
   新项目**不继承这个歧义**；Agent 宿主统一走 ``SHINKU_AGENT_*``。

这样做的代价：旧项目的 ``.env`` 不能直接搬过来，需要按新名字重写。
这是有意的——旧配置里混着历史键名，直接搬过来会把不必要的兼容负担也搬过来。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

ENV_PREFIX = "SHINKU_"

PROJECT_NAME = "shinku"

#: 服务名。用于日志名、令牌校验、健康探针和端口解析。
#: 顺序 = 启动顺序（backend 先起，Agent 最后）。
SERVICE_NAMES: tuple[str, ...] = (
    "backend",
    "response-host",
    "time-manager",
    "browser-host",
    "agent",
)

#: 由本仓库 Python 代码实现的服务（`agent` 是 Java 编排器，不在此列）。
PYTHON_SERVICE_NAMES: tuple[str, ...] = (
    "backend",
    "response-host",
    "time-manager",
    "browser-host",
)

DEFAULT_BIND = "127.0.0.1"
SERVICE_URL_HOST_ENV = ENV_PREFIX + "SERVICE_URL_HOST"

#: 端口保持与旧项目一致，避免迁移期口径漂移；
#: 换端口是配置动作，不是架构动作，所以这里不做"重新编号"。
DEFAULT_PORTS: dict[str, int] = {
    "backend": 9998,
    "response-host": 9995,
    "time-manager": 9996,
    "browser-host": 9997,
    "agent": 9100,
}

#: 服务名 -> 端口的显式环境变量名。
#: 注意 `backend` 在旧项目里是 `COMPANION_PORT`，这里换成 `SHINKU_BACKEND_PORT`。
_PORT_ENV: dict[str, str] = {
    "backend": ENV_PREFIX + "BACKEND_PORT",
    "response-host": ENV_PREFIX + "RESPONSE_HOST_PORT",
    "time-manager": ENV_PREFIX + "TIME_MANAGER_HOST_PORT",
    "browser-host": ENV_PREFIX + "BROWSER_HOST_PORT",
    "agent": ENV_PREFIX + "AGENT_PORT",
}

_BIND_ENV: dict[str, str] = {
    "backend": ENV_PREFIX + "BACKEND_HOST",
    "response-host": ENV_PREFIX + "RESPONSE_HOST_BIND",
    "time-manager": ENV_PREFIX + "TIME_MANAGER_HOST_BIND",
    "browser-host": ENV_PREFIX + "BROWSER_HOST_BIND",
    "agent": ENV_PREFIX + "AGENT_HOST",
}

#: 旧项目里仍在使用、但新项目明确不继承的键名。
#: 保留这份清单是为了让"不继承"这件事本身可被测试验证。
RETIRED_ENV_KEYS: tuple[str, ...] = (
    "COMPANION_HOST",
    "COMPANION_PORT",
    "SHINKU_SERVER_HOST",
    "SHINKU_SERVER_PORT",
)

DATA_ROOT_ENV = ENV_PREFIX + "DATA_ROOT"
CONFIG_ROOT_ENV = ENV_PREFIX + "CONFIG_ROOT"
LOG_ROOT_ENV = ENV_PREFIX + "LOG_ROOT"

SERVICE_TOKEN_ENV = ENV_PREFIX + "SERVICE_TOKEN"
SERVICE_TOKEN_FILE_ENV = ENV_PREFIX + "SERVICE_TOKEN_FILE"
SERVICE_TOKEN_HEADER = "X-Shinku-Service-Token"

#: 请求关联 ID 的头名。**线路协议，不随项目改名而改**——
#: 它与旧项目保持一致，这样迁移期两边的日志还能按同一个 id 对齐。
CORRELATION_ID_HEADER = "X-Correlation-ID"

ALLOW_LAN_ENV = ENV_PREFIX + "ALLOW_LAN"

LOG_SUFFIX = ".log"
LOG_ERROR_SUFFIX = ".err.log"


def env_key(service: str) -> str:
    """服务名 -> 端口环境变量名。"""

    if service not in _PORT_ENV:
        raise KeyError(f"unknown service: {service!r}")
    return _PORT_ENV[service]


def bind_env_key(service: str) -> str:
    """服务名 -> 绑定地址环境变量名。"""

    if service not in _BIND_ENV:
        raise KeyError(f"unknown service: {service!r}")
    return _BIND_ENV[service]


def service_port(service: str, environ: Mapping[str, str] | None = None) -> int:
    """解析服务端口。环境变量优先，非法值回落到该服务的契约默认值。"""

    env = os.environ if environ is None else environ
    fallback = DEFAULT_PORTS[service]
    raw = str(env.get(env_key(service), "") or "").strip()
    if not raw:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if 1024 <= value <= 65535 else fallback


def service_bind(service: str, environ: Mapping[str, str] | None = None) -> str:
    """解析服务绑定地址。默认只绑回环——要暴露到局域网必须显式打开 ``SHINKU_ALLOW_LAN``。"""

    env = os.environ if environ is None else environ
    raw = str(env.get(bind_env_key(service), "") or "").strip()
    if raw:
        return raw
    return "0.0.0.0" if allow_lan(env) else DEFAULT_BIND


def allow_lan(environ: Mapping[str, str] | None = None) -> bool:
    """是否允许绑到非回环地址。只有显式真值才算。"""

    env = os.environ if environ is None else environ
    return str(env.get(ALLOW_LAN_ENV, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _platform_state_root(environ: Mapping[str, str] | None = None) -> Path:
    """按平台给出"用户级"根目录。

    新项目按平台约定把可变数据放在用户目录下，而不是仓库根目录。
    旧项目的默认值是仓库内的 ``users_data/``——那会让"数据和代码同目录"，
    打包/分发时很难区分。要保留旧行为，设 ``SHINKU_DATA_ROOT`` 即可。
    """

    env = os.environ if environ is None else environ
    if sys.platform == "win32":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Shinku"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Shinku"
    return Path(env.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")) / "shinku"


def data_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = str(env.get(DATA_ROOT_ENV, "") or "").strip()
    return Path(raw) if raw else _platform_state_root(env) / "data"


def config_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = str(env.get(CONFIG_ROOT_ENV, "") or "").strip()
    return Path(raw) if raw else _platform_state_root(env) / "config"


def log_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = str(env.get(LOG_ROOT_ENV, "") or "").strip()
    return Path(raw) if raw else _platform_state_root(env) / "logs"


def log_path(service: str, *, error: bool = False, environ: Mapping[str, str] | None = None) -> Path:
    """日志文件路径。名字 = 服务名 + 后缀，这样"日志名"由服务名一处决定。"""

    if service not in SERVICE_NAMES:
        raise KeyError(f"unknown service: {service!r}")
    suffix = LOG_ERROR_SUFFIX if error else LOG_SUFFIX
    return log_root(environ) / f"{service}{suffix}"


def service_url(service: str, environ: Mapping[str, str] | None = None) -> str:
    """服务的基础 URL。

    绑定到 ``0.0.0.0``/``::`` 时不能把通配地址当成客户端目标；默认回退到
    回环。需要在容器、局域网或反向代理下使用其他可达地址时，可显式设置
    ``SHINKU_SERVICE_URL_HOST``，从而不必改代码。
    """

    env = os.environ if environ is None else environ
    configured = str(env.get(SERVICE_URL_HOST_ENV, "") or "").strip()
    host = configured or service_bind(service, env)
    if host in {"0.0.0.0", "::", "[::]", "*"}:
        host = DEFAULT_BIND
    return f"http://{host}:{service_port(service, env)}"

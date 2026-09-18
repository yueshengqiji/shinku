"""进程级的健康事实：后端存活探针的载荷原语。

存活探针回答的是关于**进程**的那几个问题——进程号、解释器路径、可选依赖装了没有——
而不是业务状态（那些属于就绪探针，依赖存储与网关）。把这几个问题收敛到一个模块，
是为了让它们只有一个出处：字段名与取值口径不再散落在路由函数里。

它是同步纯函数，不做 I/O、不读配置、不导入重依赖——探针路径上不该有任何可能阻塞的东西。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


def build_basic_health_payload() -> dict[str, Any]:
    """返回本进程的基础事实。

    每次调用都新造一个字典，不返回共享实例：调用方会把它展开进更大的响应体，
    共享字典一旦被某个调用方就地改写，其余调用方拿到的就不再是「进程事实」。

    可选依赖只做「装没装」的探测，**不导入**——探针的回答必须即时，
    导入一个下载器会把响应拖到超时。
    """

    return {
        "status": "ok",
        "pid": os.getpid(),
        "python": sys.executable,
        "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
    }


__all__ = ["build_basic_health_payload"]

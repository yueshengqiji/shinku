"""独立工作台启动器；只有显式传入的宿主工厂可以加载模型与工具配置。

工厂是本机可信代码，返回 ServiceHost 或其上下文管理器；环境/私用插件的
生命周期由工厂管理。默认未连接 Agent，绝不隐式加载 QQ 或生产配置。
"""
import argparse
from contextlib import ExitStack
import importlib
from pathlib import Path

from .app import create_angelos_app
from .host import ServiceHost


def main(argv=None):
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--factory', help='可信本机宿主工厂 module:callable')
    parser.add_argument('--delegation-hosts', type=Path, help='显式启用宿主委托的私用连接配置文件')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    with ExitStack() as resources:
        host = None
        if args.factory:
            module, separator, name = args.factory.rpartition(':')
            if not separator:
                parser.error('factory must be module:callable')
            candidate = getattr(importlib.import_module(module), name)()
            host = resources.enter_context(candidate) if hasattr(candidate, '__enter__') else candidate
            if not isinstance(host, ServiceHost):
                parser.error('factory must return ServiceHost')
        if args.delegation_hosts:
            from .delegation_config import install_from_file
            install_from_file(host, args.delegation_hosts)
        uvicorn.run(create_angelos_app(args.data_root, host=host), host='127.0.0.1', port=args.port,
                    access_log=False)


if __name__ == '__main__':
    main()

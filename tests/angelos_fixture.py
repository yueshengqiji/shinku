"""隔离浏览器验收：真实 AgentService/文件工具，确定性模型，不载入私用配置。"""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace

from fastapi import FastAPI
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.api.angelos import build_angelos_router


class FixtureHost:
    model = '离线验收模型'
    requester = 'fixture-owner'

    def __init__(self, service, tool):
        self.service, self.tool = service, tool
        self.preparations = 0

    def ready(self):
        return True

    def port(self, cid):
        return self.service.bind(AgentScope(self.requester, 'angelos:' + cid))

    def task_id(self, cid, key):
        return 'agent:' + canonical_hash([self.requester, 'angelos:' + cid, key])[:32]

    def prepare(self, text, sources, workspace):
        self.preparations += 1
        if text.startswith('生成'):
            return {'contract': demo_request(self.tool).contract}
        return {'reply': '隔离验收回复：' + text + ('\n附件已读取：' + sources[0]['content'] if sources else '')}


def fixture(root):
    service, tool, runtime = demo_service(root)
    host = FixtureHost(service, tool)
    router, workbench = build_angelos_router(SimpleNamespace(data_root=root), host=host)
    @asynccontextmanager
    async def lifespan(app):
        workbench.start()
        yield
        workbench.close()
        service.coordinator.shutdown(wait=True)
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app, workbench, host


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--directory')
    args = parser.parse_args()
    root = Path(args.directory) if args.directory else Path(tempfile.mkdtemp(prefix='angelos-fixture-'))
    root.mkdir(parents=True, exist_ok=True)
    app, _, _ = fixture(root)
    uvicorn.run(app, host='127.0.0.1', port=args.port)

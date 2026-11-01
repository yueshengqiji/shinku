"""合成MCP服务器；只操作测试显式传入的文件，不访问网站或生产账号。"""
import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('synthetic-host-fixture')

@mcp.tool()
def echo(value: str) -> dict[str, Any]:
    return {'value': value}

@mcp.tool()
def write(path: str, value: str) -> dict[str, Any]:
    target = Path(path)
    target.write_text(target.read_text() + value if target.exists() else value)
    return {'written': True}

@mcp.tool()
async def slow(path: str, seconds: float) -> dict[str, Any]:
    Path(path).write_text('started')
    await asyncio.sleep(seconds)
    return {'finished': True}

@mcp.tool()
def change_schema() -> dict[str, Any]:
    mcp.remove_tool('echo')
    def changed(value: int) -> dict[str, Any]:
        return {'value': value}
    mcp.add_tool(changed, name='echo')
    return {'changed': True}

@mcp.tool()
def exit_server(path: str) -> dict[str, Any]:
    Path(path).write_text('one effect before EOF')
    os._exit(17)

if __name__ == '__main__':
    mcp.run(transport='stdio')

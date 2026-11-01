"""One-shot local runner; never starts a QQ/API server or loads environment config."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tempfile
import time

from .contracts import AgentRequest, AgentScope, TaskHandle
from .service import AgentService


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an explicitly configured isolated Agent once.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Offline runtime and isolated file tool")
    mode.add_argument("--factory", help="Explicit host factory module:callable returning AgentService")
    parser.add_argument("--request", help="JSON AgentRequest; required with --factory")
    parser.add_argument("--workspace", help="Demo output directory; an existing demo output is refused")
    parser.add_argument("--requester", default="local-demo")
    parser.add_argument("--session", default="local:demo")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 3600:
        parser.error("timeout must be between 0 and 3600 seconds")
    if args.demo:
        from .demo import demo_request, demo_service
        directory = Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="shinku-agent-"))
        directory.mkdir(parents=True, exist_ok=True)
        service, tool, _ = demo_service(directory)
        request = demo_request(tool)
    else:
        if not args.request or ":" not in args.factory:
            parser.error("--factory needs module:callable and --request")
        module, name = args.factory.rsplit(":", 1)
        service = getattr(importlib.import_module(module), name)()
        if not isinstance(service, AgentService):
            parser.error("factory must return AgentService")
        request = AgentRequest(**json.loads(Path(args.request).read_text(encoding="utf-8")))
    client = service.bind(AgentScope(args.requester, args.session))
    try:
        handle = client.submit(request)
        deadline = time.monotonic() + args.timeout
        while True:
            view = client.status(handle.task_id)
            if view["status"] not in {"queued", "running"} and not view["worker_active"]:
                break
            if time.monotonic() >= deadline:
                try:
                    client.cancel(TaskHandle(handle.task_id, view["attempt"], view["version"]))
                finally:
                    print(json.dumps(client.result(handle.task_id), ensure_ascii=False, indent=2))
                return 2
            time.sleep(.01)
        result = client.result(handle.task_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["execution"]["status"] == "completed" else 1
    finally:
        service.coordinator.shutdown(wait=False)
        # Running custom tools are cooperative; no false stopped receipt. The
        # process-local store must remain available until their worker exits.


if __name__ == "__main__":
    raise SystemExit(main())

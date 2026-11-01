"""Offline standalone example: deterministic runtime, real isolated file tool."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from shinku.agent.task_state import AgentTaskStore
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

from .contracts import AgentRequest
from .execution import AgentExecutor, ExecutionGrant, ExecutionProfile
from .service import AgentService


class DemoFileTool:
    tool_type = "demo_file"
    agent_progress_class = "change"
    agent_completion_effects = ("write",)
    agent_completion_target_arguments = ("path",)

    def __init__(self, workspace):
        self.path = Path(workspace).resolve() / "agent-demo.txt"
        self.calls = 0

    def tool_metadata(self):
        return SimpleNamespace(description="Create the isolated demo file.", input_schema={
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})

    def normalize_call(self, call):
        return call if Path(call.get("path", "")).resolve() == self.path else None

    def execute(self, *, call, context):
        self.calls += 1
        self.path.write_text("standalone agent verified", encoding="utf-8")
        artifact = {"path": str(self.path), "verified_effects": ["write"],
            "completion_targets": [str(self.path)], "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "bytes": self.path.stat().st_size, "result_valid": True}
        return {"tool_type": self.tool_type, "followup_context": "Demo file written.", "state_updates": {
            **artifact, "changed": True, "artifact_receipts": [artifact]}}

    def agent_verify_result(self, *, receipt, context):
        exists = self.path.is_file()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest() if exists else ""
        refreshed = {"result_valid": exists and digest == receipt.get("sha256"),
                     "file_text": self.path.read_text(encoding="utf-8") if exists else ""}
        if "artifact_receipts" in receipt:
            refreshed["artifact_receipts"] = [{**v, **refreshed} for v in receipt["artifact_receipts"]]
        return refreshed


class DemoRuntime:
    def __init__(self, tool):
        self.tool = tool
        self.calls = []

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.tool.calls:
            return {"tool_call": {"type": "demo_file", "path": str(self.tool.path)}}
        return {"kind": "final", "text": "The isolated file was created and verified.", "task_outcome": "completed"}


def demo_service(workspace):
    tool = DemoFileTool(workspace)
    if tool.path.exists():
        raise ValueError("demo_output_already_exists")
    runtime = DemoRuntime(tool)
    executor = AgentExecutor(ToolHost(ToolRegistry({tool.tool_type: tool})), ExecutionProfile(runtime))
    grant = ExecutionGrant("offline-demo", frozenset({tool.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type: "low"}),
        budget_limits={"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 30},
        resume_safe=True)
    service = AgentService(executor=executor, authorize=lambda scope, task, operation: grant,
                           task_store=AgentTaskStore(), coordinator=AgentTaskCoordinator(max_workers=1))
    return service, tool, runtime


def demo_request(tool, *, key="offline-demo"):
    return AgentRequest(key, "Create the isolated demo file and verify its contents.", {
        "goal": "Create and verify an isolated demo file", "deliverables": [str(tool.path)],
        "scope": [str(tool.path.parent)], "requirements": [{"id": "file", "description": "Verified demo file",
            "effect": "write", "targets": [str(tool.path)], "checks": {"contains": "standalone agent verified"}}]})

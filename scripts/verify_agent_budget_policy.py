"""Saved-model budget closeout/resume smoke tests; fictional tools, no QQ sends."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import argparse
import json
import time

import gray_memory_stage_a as a
from shinku.agent.budget import TaskBudget
from shinku.agent.budget_policy import PROFILES
from shinku.agent.loop import AgentDecision
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry
from shinku.tasks.coordinator import AgentTaskCoordinator


class FixtureRead:
    tool_type = "fixture_read"
    agent_permission_class = "read"
    agent_progress_class = "observe"
    def __init__(self): self.calls = []
    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"target": {"type": "string"}},
                                            "required": ["target"]})
    def build_prompt_instruction(self): return "只读核验虚构测试目标 fixture 的状态；不访问磁盘或网络。"
    def normalize_call(self, call): return call if call.get("target") == "fixture" else None
    def execute(self, *, call, context):
        self.calls.append(dict(call))
        return {"tool_type": self.tool_type, "followup_context": "虚构测试目标 fixture 已核验：文件存在，内容校验通过。"}


def make_turn(text, mid):
    msg = a.IncomingMessage.from_event({"post_type": "message", "message_type": "private", "user_id": a.OWNER,
        "self_id": "bot", "message_id": mid, "message": [{"type": "text", "data": {"text": text}}]})
    return a.QQTurn(conversation_id=a.OWNER, conversation_type="private", primary_message_id=mid,
                    message_ids=(mid,), combined_text=text, messages=(msg,))


def main():
    parser = argparse.ArgumentParser();parser.add_argument("--output", required=True)
    root = Path(parser.parse_args().output).resolve();root.mkdir(parents=True, exist_ok=False)
    runtime, config, _, _, persona, frozen = a.saved_profile(root)
    runtime.runtime_policy = replace(runtime.runtime_policy, http_attempts=2)
    report = {"fictional_only": True, "qq_messages_sent": False, "rows": []}
    try:
        for name in ("closeout", "resume"):
            coordinator = AgentTaskCoordinator(max_workers=1)
            observed, handler = a.ObservedRuntime(runtime), FixtureRead()
            bridge = a.QQAgentBridge(runtime=observed, system_prompt=persona, sender=None,
                master_user_ids=(a.OWNER,), bot_ids=("bot",),
                command_state=a.QQCommandState(owner_ids=(a.OWNER,), permission_mode="readonly"),
                task_coordinator=coordinator, async_agent_enabled=True, adaptive_budget_enabled=True,
                task_budget_limits=PROFILES["heavy"], task_store=AgentTaskStore(root / f"{name}.sqlite3"),
                tool_host=ToolHost(ToolRegistry({"fixture_read": handler})), task_contract_enabled=True)
            task = new_task(task_id="task:fixture", requester=a.OWNER, session=f"private:{a.OWNER}",
                request="检查虚构测试目标 fixture 的状态，确认是否通过核验。", message_id="original",
                contract={"goal": "检查虚构测试目标 fixture", "deliverables": ["核验状态"], "scope": ["只处理虚构目标"],
                    "requirements": [{"id": "r1", "description": "核验 fixture 状态", "effect": "tool:fixture_read", "targets": ["fixture"]}]})
            task["status"] = "blocked"
            task["notes"]["remaining"] = ["读取 fixture 的状态并核验"]
            b = TaskBudget(**PROFILES["heavy"], tokens=1550000, requests=205, managed=True, profile="heavy")
            if name == "resume": b.reason = "task_token_budget_exhausted"
            task["budget"] = b.snapshot();task["notes"]["blocker"] = b.reason
            task["budget_allocation"] = {"profile": "heavy", "limits": PROFILES["heavy"], "grants": []}
            bridge.task_store.save(task)
            started = time.monotonic()
            try:
                if name == "resume":
                    reply = bridge.handle_turn(make_turn("/agent resume task:fixture standard", "fund-and-resume"))
                else:
                    turn = make_turn("继续核验原任务，尚未核实的内容不能说成已经完成。", "continue")
                    trace = bridge.trace_store.begin(turn)
                    context = bridge._prepare_qq_turn(turn, trace).context
                    prepared = bridge._prepare_agent_task(turn, AgentDecision(kind="tool", task_link={"mode":"continue", "task_id":"task:fixture"}),
                                                         context, "task:fixture", trace)
                    reply = bridge._enqueue_agent_task(turn, task_id="task:fixture", context=context, initial_transcript=[],
                        prepared_task=replace(prepared, decision=None), trace_id=trace)
                    coordinator.dispatch(reply.background_task_id)
                deadline = time.monotonic() + 180
                record = None
                while time.monotonic() < deadline:
                    record = coordinator.get(reply.background_task_id)
                    if record and not record["worker_active"] and record["status"] != "queued": break
                    time.sleep(.1)
                if record is None or record["worker_active"]:
                    coordinator.cancel(reply.background_task_id)
                    raise RuntimeError("isolated_replay_timeout")
                saved = bridge.task_store.get("task:fixture", requester=a.OWNER, session=f"private:{a.OWNER}")
                row = {"case": name, "seconds": round(time.monotonic()-started, 3), "status":record["status"],
                    "reason": record["error"], "reply":record["result_summary"], "fixture_tool_calls": handler.calls,
                    "model_calls": len(observed.calls), "calls": observed.calls, "budget":saved["budget"],
                    "grants":len(saved["budget_allocation"]["grants"])}
                report["rows"].append(row)
                print(json.dumps({k: row[k] for k in ("case","seconds","status","reason","reply","model_calls","grants")}, ensure_ascii=False), flush=True)
            finally:
                coordinator.shutdown(wait=True)
        report["protected_configs_unchanged"] = all(a.sha(Path(p)) == h for p,h in frozen["protected_config_hashes"].items())
        (root / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        runtime.close()


if __name__ == "__main__": main()

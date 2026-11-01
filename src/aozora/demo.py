"""合成离线演示，只证明状态与调用链；固定决策不代表真实模型行为或正式人设。"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import Decision, Event, Identity
from .kernel import Kernel
from .store import Store


class DemoDecider:
    """可审阅的两步固定夹具；不用于正常聊天，也不作为网络错误的回退。"""

    def decide(self, context: dict) -> Decision:
        if context["current_event_id"] == "demo-1":
            return Decision.parse({"action": "ask", "reply": "这条消息只是一个说法。我们有测量结果吗？",
                "attention": [{"key": "comparison", "text": "关注两组读数是否相同", "version": 0,
                               "evidence": ["demo-1"], "status": "open"}], "understandings": []})
        focus = next(item for item in context["attention"] if item["key"] == "comparison")
        return Decision.parse({"action": "reply", "reply": "新的合成观察显示两组读数不同，可以修正先前的说法。",
            "attention": [{**{k: focus[k] for k in ("key", "text", "version")},
                           "evidence": ["demo-1", "demo-2"], "status": "completed"}],
            "understandings": [{"key": "comparison-view", "text": "这次合成测量不支持最初的转述。",
                                "version": 0, "evidence": ["demo-1", "demo-2"], "status": "tentative"}]})


def run_demo(directory: Path) -> dict:
    """新目录内演示重启接续，拒绝覆盖旧证据；无网络、生产数据或外部发送。"""
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    identity = Identity("synthetic-agent", "演示个体")
    (directory / "identity.example.json").write_text(json.dumps(identity.payload(), ensure_ascii=False, indent=2) + "\n")
    path = directory / "state.sqlite3"
    first_store = Store(path, identity)
    first = Kernel(first_store, DemoDecider()).handle(Event("demo-1", "synthetic-room", "speaker-a", "message",
        "合成输入：我听说两组读数一样。", 1))
    restarted_store = Store(path, identity)
    second = Kernel(restarted_store, DemoDecider()).handle(Event("demo-2", "synthetic-room", "fixture-host", "observation",
        "合成观察：A=1，B=2。此值仅用于离线验证。", 2))
    report = {"mode": "synthetic_offline_demo", "model_called": False, "external_delivery": False,
              "first": first, "after_restart": second, "state": restarted_store.snapshot("synthetic-room")}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report

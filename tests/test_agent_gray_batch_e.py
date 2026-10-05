"""Offline checks of the gray-test harness and truthful model-error handoff."""
import importlib.util
from pathlib import Path

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn


def test_live_runtime_fallback_is_failed_not_completed():
    class UnavailableRuntime:
        def call_chat_json(self, **kwargs):
            # Same return path used by RuntimeCore after exhausted HTTP errors.
            return kwargs["fallback"]
    item = IncomingMessage.from_event({"post_type": "message", "message_type": "private",
        "user_id": "gray-owner", "message_id": "gray", "message": "创建测试文件"})
    turn = QQTurn(conversation_id="gray-owner", conversation_type="private", primary_message_id="gray",
        message_ids=("gray",), combined_text="创建测试文件", messages=(item,))
    result = QQAgentBridge(runtime=UnavailableRuntime(), system_prompt="isolated").handle_turn(turn)
    assert result.agent_status == "failed" and result.reason == "chat_generation_failed"


def test_gray_network_tools_reject_external_urls(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gray_e_test", root / "scripts/gray_agent_batch_e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = module.RouteProbe("http://127.0.0.1:1234", 1)
    assert probe.normalize_call({"url": "https://example.com", "route": "direct"}) is None
    assert probe.normalize_call({"url": "http://127.0.0.1:12340/ready", "route": "direct"}) is None
    assert probe.normalize_call({"url": "http://127.0.0.1:1234/ready", "route": "direct"}) is not None

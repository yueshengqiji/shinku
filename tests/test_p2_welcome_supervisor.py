from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from shinku.config import load_settings
from shinku.qq.host import NapCatHost
from shinku.qq.message import IncomingMessage, is_group_increase_event
from shinku.qq.welcome import GroupWelcomePolicy
from shinku.supervisor import ShinkuSupervisor, SupervisorConfig


class _Transport:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return {"ok": True, "message_ids": [f"sent-{len(self.messages)}"]}


def _group_increase(message_id: str = "notice-1") -> dict:
    return {
        "post_type": "notice",
        "notice_type": "group_increase",
        "sub_type": "approve",
        "group_id": "g-1",
        "user_id": "u-1",
        "message_id": message_id,
        "sender": {"user_id": "u-1", "nickname": "新成员"},
        "time": 100.0,
    }


class GroupWelcomeTests(unittest.TestCase):
    def test_group_increase_is_not_a_chat_turn_and_welcome_is_opt_in(self) -> None:
        event = _group_increase()
        message = IncomingMessage.from_event(event)
        self.assertTrue(is_group_increase_event(event))
        self.assertTrue(message.is_group_increase)
        self.assertEqual(message.text, "")

        transport = _Transport()
        observed = []
        host = NapCatHost(
            transport=transport,
            passive_observer=observed.append,
            welcome_policy=GroupWelcomePolicy(enabled=True, template="欢迎 {user}，群 {group}。"),
            clock=lambda: 100.0,
        )
        result = host.handle_event(event)
        self.assertFalse(result.scheduled)
        self.assertIsNotNone(result.welcome)
        self.assertTrue(result.welcome.sent)
        self.assertEqual(len(transport.messages), 1)
        self.assertEqual(transport.messages[0].conversation_type, "group")
        self.assertEqual(transport.messages[0].segments[0].text, "欢迎 新成员，群 g-1。")
        self.assertEqual(observed, [])

    def test_welcome_does_not_send_in_default_or_dry_run_mode(self) -> None:
        transport = _Transport()
        host = NapCatHost(transport=transport, clock=lambda: 100.0)
        result = host.handle_event(_group_increase())
        self.assertIsNone(result.welcome)
        self.assertEqual(transport.messages, [])

        policy = GroupWelcomePolicy(enabled=True, allowed_group_ids=frozenset({"other"}))
        host = NapCatHost(transport=transport, welcome_policy=policy, clock=lambda: 100.0)
        result = host.handle_event(_group_increase("notice-2"))
        self.assertEqual(result.welcome.status, "welcome_filtered")
        self.assertEqual(transport.messages, [])

    def test_duplicate_or_cooldown_welcome_is_not_sent_twice(self) -> None:
        transport = _Transport()
        host = NapCatHost(
            transport=transport,
            welcome_policy=GroupWelcomePolicy(enabled=True, cooldown_seconds=60),
            clock=lambda: 100.0,
        )
        first = host.handle_event(_group_increase(), at=100.0)
        duplicate = host.handle_event(_group_increase(), at=101.0)
        cooldown = host.handle_event(_group_increase("notice-3"), at=102.0)
        self.assertTrue(first.welcome.sent)
        self.assertIsNone(duplicate.welcome)
        self.assertEqual(cooldown.welcome.status, "welcome_cooldown")
        self.assertEqual(len(transport.messages), 1)


class SupervisorTests(unittest.TestCase):
    def _settings(self, root: Path):
        return load_settings({"SHINKU_BACKEND_PORT": "19998", "SHINKU_DATA_ROOT": str(root / "data")})

    def test_supervisor_uses_only_its_child_and_stops_on_clean_exit(self) -> None:
        class Process:
            def poll(self):
                return 0

        calls = []

        def factory(command, **kwargs):
            calls.append((command, kwargs))
            return Process()

        with tempfile.TemporaryDirectory() as temporary:
            supervisor = ShinkuSupervisor(
                SupervisorConfig(max_restarts=0),
                settings=self._settings(Path(temporary)),
                process_factory=factory,
                health_probe=lambda url: True,
            )
            self.assertEqual(supervisor.run(), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][-4:], ["shinku.cli", "serve", "--service", "backend"])

    def test_supervisor_returns_child_failure_when_restart_budget_is_zero(self) -> None:
        class Process:
            def poll(self):
                return 7

        with tempfile.TemporaryDirectory() as temporary:
            supervisor = ShinkuSupervisor(
                SupervisorConfig(max_restarts=0),
                settings=self._settings(Path(temporary)),
                process_factory=lambda command, **kwargs: Process(),
                health_probe=lambda url: False,
            )
            self.assertEqual(supervisor.run(), 7)

    def test_supervisor_rejects_unimplemented_service(self) -> None:
        with self.assertRaises(ValueError):
            SupervisorConfig(service="browser-host")


if __name__ == "__main__":
    unittest.main()

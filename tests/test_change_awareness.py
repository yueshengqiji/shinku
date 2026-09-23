import tempfile
import unittest
from pathlib import Path

from shinku.agent.change_awareness import RuntimeChangeTracker
from shinku.agent.self_state import SelfStateTool


class RuntimeChangeAwarenessTests(unittest.TestCase):
    def test_startup_snapshot_and_live_code_change_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src"
            source.mkdir()
            module = source / "tool.py"
            module.write_text("TOOL = 'old'\n", encoding="utf-8")
            snapshot = root / "runtime" / "change_snapshot.json"

            tracker = RuntimeChangeTracker(
                enabled=True,
                scan_roots=[source],
                snapshot_path=snapshot,
            )
            tracker.start(registered_tools=["old_tool"])
            clean = tracker.diagnostics(registered_tools=["old_tool"])
            self.assertEqual(clean["status"], "clean")
            self.assertEqual(clean["last_startup"]["status"], "baseline_created")

            module.write_text("TOOL = 'new'\n", encoding="utf-8")
            live = tracker.diagnostics(registered_tools=["old_tool", "new_tool"])
            self.assertEqual(live["status"], "changed_since_startup")
            self.assertTrue(live["since_startup"]["restart_required"])
            self.assertIn("tool.py", live["since_startup"]["modified_files"])
            self.assertEqual(live["since_startup"]["tools_added"], ["new_tool"])
            self.assertIn("重启", live["summary"])

            restarted = RuntimeChangeTracker(
                enabled=True,
                scan_roots=[source],
                snapshot_path=snapshot,
            )
            restarted.start(registered_tools=["old_tool", "new_tool"])
            after_restart = restarted.diagnostics(registered_tools=["old_tool", "new_tool"])
            self.assertEqual(after_restart["status"], "changed_at_startup")
            self.assertFalse(after_restart["since_startup"]["restart_required"])
            self.assertIn("tool.py", after_restart["last_startup"]["modified_files"])

    def test_self_state_can_select_changes_scope(self) -> None:
        tool = SelfStateTool(lambda: {"changes": {"status": "changed_at_startup"}})
        self.assertEqual(tool.normalize_call({"scope": "changes"}), {"scope": "changes"})
        result = tool.execute(call={"scope": "changes"}, context={})
        self.assertIn("changed_at_startup", result["followup_context"])


if __name__ == "__main__":
    unittest.main()

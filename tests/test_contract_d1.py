from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_shinku_release import audit_release


class ReleaseAuditTests(unittest.TestCase):
    def test_automated_release_boundary_audit_and_apache_license_pass(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = audit_release(root)
        self.assertTrue(result["ok"], result["findings"])
        self.assertTrue(result["release_ready"])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["media_files"], [])


if __name__ == "__main__":
    unittest.main()

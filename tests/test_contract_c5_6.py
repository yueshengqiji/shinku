from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_shinku_admission import audit


class AdmissionAuditTests(unittest.TestCase):
    def test_current_tree_passes_automated_admission_audit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = audit(root)
        self.assertTrue(result["ok"], result["findings"])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["checks"], {
            "import_graph": True,
            "active_env": True,
            "source_record": True,
            "license_inventory": True,
        })


if __name__ == "__main__":
    unittest.main()

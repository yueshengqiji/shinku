from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_shinku_release_manifest import build_manifest


class ReleaseManifestTests(unittest.TestCase):
    def test_clean_manifest_excludes_secrets_runtime_data_and_media(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = build_manifest(root)
        self.assertTrue(result["ok"], result["findings"])
        self.assertGreater(result["file_count"], 0)
        paths = {item["path"] for item in result["files"]}
        self.assertNotIn(".env", paths)
        self.assertFalse(any(path == "data" or path.startswith("data/") for path in paths))
        self.assertFalse(any(".egg-info/" in path for path in paths))
        self.assertEqual(result["findings"], [])


if __name__ == "__main__":
    unittest.main()

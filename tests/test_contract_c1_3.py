"""C1-3 capability manifest tests.

The old visual-asset manifest surface was removed from the standalone runtime.
This contract now covers only the generic MCP/plugin capability boundary.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.capabilities import ALLOWED_ADAPTER_TYPES, LOOPBACK_HOSTS, MCP_SAFE_TYPE_RE, load_manifest
from shinku.contracts.capability import CapabilityManifest, InvalidManifest


def _write(root: Path, text: str) -> Path:
    path = root / "provider.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class CapabilityManifestTests(unittest.TestCase):
    def test_valid_provider_manifest_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = _write(
                Path(raw),
                """
schema: capability_adapter/v1
provider:
  id: local_tools
  type: mcp_stdio
  display_name: Local tools
  endpoint:
    url: http://127.0.0.1:9101
    loopback_only: true
capabilities:
  - id: read_file
    risk: low
    confirm: never
    effects: [file_read]
""",
            )
            result = load_manifest(path, source_layer="builtin")
        self.assertIsInstance(result, CapabilityManifest)
        assert isinstance(result, CapabilityManifest)
        self.assertEqual(result.provider_id, "local_tools")
        self.assertEqual(result.endpoint.url, "http://127.0.0.1:9101")
        self.assertEqual(result.capabilities[0].id, "read_file")

    def test_invalid_manifest_is_a_value_not_an_exception(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = _write(Path(raw), "schema: wrong\nprovider: {}\n")
            result = load_manifest(path, source_layer="profile")
        self.assertIsInstance(result, InvalidManifest)
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "schema_mismatch")

    def test_safety_constants_remain_independent_of_desktop_assets(self) -> None:
        self.assertIn("mcp_stdio", ALLOWED_ADAPTER_TYPES)
        self.assertEqual(LOOPBACK_HOSTS, {"127.0.0.1", "localhost", "::1"})
        self.assertIsNotNone(MCP_SAFE_TYPE_RE.fullmatch("python_plugin"))


if __name__ == "__main__":
    unittest.main()

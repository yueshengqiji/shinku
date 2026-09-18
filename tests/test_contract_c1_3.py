"""C1-3 能力清单与资源清单的测试。

这一批按来源记录 §5.9 的**路径 A 第二类**处理：模块含逻辑、分支与数据处理，属**洁净室重写**。
断言对象全部是**外部可观察行为**（契约 `docs/contracts/c1_3_capability_and_resource_manifests.md`），
不涉及任何实现结构；失败测试与行为测试同等重要——契约的一半价值在它拒绝什么。

四处刻意的地方：

1. **夹具全部在本测试内构造。** 旧项目 `docs/fixtures/capability_adapter_m1/*.yaml` 与本文件
   没有任何依赖关系：YAML 在临时目录里现写，资源目录树也在临时目录里现造。
2. **契约里那些"不会让任何既有测试变红"的表达式**（C1-2 的教训）在这里被逐条钉住。
   典型是 `visible_in: [null]` 与 `visible_in: [0]`：旧口径是 `str(item or "").strip()`
   （空 → 丢弃），错写成 `str(item).strip()` 就会变成 `"None"` / `"0"` 从而误报非法值。
   这类差异只有专门写一条断言才抓得到。
3. **边界扫描用 `tokenize` 取标识符，不用子串匹配。** C1-2 用的是 `assertNotIn`
   （子串），对 `MIN_ID_LENGTH` 这类独特名字够用；本批的旧私有名里有 `_text`、`_key`、
   `_meta`、`_background` 这种短且常见的词，子串匹配会把 `_read_note_text`、
   `_background_record` 这类新名字误判为命中。改为比较**精确标识符 token**后，
   这条检查既没有假阳性，也比子串匹配更严格。
4. **旧模块的 docstring 整句被列为"不得出现"**。散文重叠的口径是"docstring + 注释、
   连续 ≥12 字符"，这里额外把旧模块里最长的几句整句钉进测试，属于同一口径的定点采样。
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import tokenize
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shinku.capabilities import manifest as capability_manifest
from shinku.capabilities import safety
from shinku.capabilities.manifest import (
    ALLOWED_ADAPTER_TYPES,
    CONFIRM_VALUES,
    EFFECT_VALUES,
    PROVIDER_ID_RE,
    RISK_VALUES,
    SCHEMA,
    SECRET_VALUE_RE,
    VISIBLE_IN_VALUES,
    load_manifest,
)
from shinku.contracts.capability import CapabilityManifest, InvalidManifest
from shinku.resources import manifest as resource_manifest
from shinku.resources.manifest import (
    BACKGROUND_ALIASES,
    BACKGROUND_LABELS,
    BACKGROUND_PRIORITY,
    EMOTION_ALIASES,
    EMOTION_FALLBACK_CANDIDATES,
    EMOTION_LABELS,
    META_NOTE_KEYS,
    NOTE_FILENAMES,
    PROMPT_NOTE_LIMIT,
    SIDECAR_NOTE_SUFFIXES,
    ResourceManifest,
)

SHINKU_PKG = Path(capability_manifest.__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 夹具工具
# --------------------------------------------------------------------------- #


def _yaml(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def _provider(body: str, *, provider_id: str = "demo", provider_type: str = "mcp_stdio") -> str:
    return f"schema: {SCHEMA}\nprovider:\n  id: {provider_id}\n  type: {provider_type}\n{body}"


def _identifiers(path: Path) -> set[str]:
    """文件里出现过的所有 Python 标识符（精确 token，不是子串）。"""

    with path.open(encoding="utf-8") as handle:
        return {
            token.string
            for token in tokenize.generate_tokens(handle.readline)
            if token.type == tokenize.NAME
        }


def _imported_modules(path: Path) -> set[str]:
    """文件里所有 import 目标的字面名（相对导入保留前导点，不做绝对化）。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


class _ManifestCase(unittest.TestCase):
    """把 YAML 写到临时目录再加载，避免依赖仓库里的夹具文件。"""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def load(self, text: str, name: str = "provider.yaml") -> CapabilityManifest | InvalidManifest:
        return load_manifest(_yaml(self.root, name, text), source_layer="builtin")


class _PackCase(unittest.TestCase):
    """构造一个临时的资源目录树。"""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def asset(self, relative: str, data: bytes = b"fixture") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def meta(self, relative: str, value: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def text_file(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def build(self, **kwargs: object) -> ResourceManifest:
        return ResourceManifest(self.root, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# §3.1 公开面（常量与签名）
# --------------------------------------------------------------------------- #


class CapabilitySurfaceTests(unittest.TestCase):
    def test_the_schema_string_is_the_wire_contract(self) -> None:
        self.assertEqual(SCHEMA, "capability_adapter/v1")

    def test_the_adapter_type_allowlist_is_closed(self) -> None:
        self.assertEqual(
            set(ALLOWED_ADAPTER_TYPES),
            {"mcp_stdio", "comfyui", "openai_compat_tts", "openai_compat_asr", "python_plugin"},
        )
        self.assertIsInstance(ALLOWED_ADAPTER_TYPES, frozenset)

    def test_enum_tables_match_the_contract(self) -> None:
        self.assertEqual(set(VISIBLE_IN_VALUES), {"base", "web", "desktop", "qq"})
        self.assertEqual(set(RISK_VALUES), {"low", "medium", "high"})
        self.assertEqual(set(CONFIRM_VALUES), {"never", "first_time", "always"})
        self.assertEqual(
            set(EFFECT_VALUES),
            {
                "file_read",
                "file_write",
                "command_exec",
                "network_outbound",
                "browser_action",
                "media_generation",
                "state_mutation",
            },
        )

    def test_provider_id_pattern_rejects_whitespace_and_slashes(self) -> None:
        self.assertIsNotNone(PROVIDER_ID_RE.fullmatch("good.id-name_1"))
        for bad in ("bad id", "bad/id", "bad\tid", "bad\nid"):
            with self.subTest(value=bad):
                self.assertIsNone(PROVIDER_ID_RE.fullmatch(bad))

    def test_secret_value_pattern_catches_known_shapes(self) -> None:
        for value in ("Bearer abc", "sk-abc", "ghp_abc", "xoxb-abc", "k=v", "k:v"):
            with self.subTest(value=value):
                self.assertIsNotNone(SECRET_VALUE_RE.search(value))
        self.assertIsNone(SECRET_VALUE_RE.search("shinku_comfyui_key"))

    def test_source_layer_is_keyword_only(self) -> None:
        with TemporaryDirectory() as temp:
            path = _yaml(Path(temp), "x.yaml", "{}")
            with self.assertRaises(TypeError):
                load_manifest(path, "builtin")  # type: ignore[misc]

    def test_the_loader_is_owned_by_the_new_package(self) -> None:
        self.assertEqual(load_manifest.__module__, "shinku.capabilities.manifest")


class SafetyConstantsTests(unittest.TestCase):
    def test_loopback_hosts_cover_the_three_local_spellings(self) -> None:
        self.assertEqual(safety.LOOPBACK_HOSTS, {"127.0.0.1", "localhost", "::1"})

    def test_secret_markers_are_the_generic_words(self) -> None:
        self.assertEqual(safety.MCP_SECRET_MARKERS, ("api_key", "password", "secret", "token"))

    def test_safe_type_pattern_bounds_length_and_alphabet(self) -> None:
        self.assertIsNotNone(safety.MCP_SAFE_TYPE_RE.fullmatch("a" * 40))
        self.assertIsNone(safety.MCP_SAFE_TYPE_RE.fullmatch("a" * 41))
        self.assertIsNone(safety.MCP_SAFE_TYPE_RE.fullmatch("a b"))
        for value in ("openai_compat_tts", "mcp.stdio-1"):
            with self.subTest(value=value):
                self.assertIsNotNone(safety.MCP_SAFE_TYPE_RE.fullmatch(value))


# --------------------------------------------------------------------------- #
# §3.3 / §3.4 成功路径
# --------------------------------------------------------------------------- #


class ManifestAcceptanceTests(_ManifestCase):
    def test_a_complete_manifest_populates_every_field(self) -> None:
        manifest = self.load(
            f"""schema: {SCHEMA}
provider:
  id: comfyui
  type: comfyui
  display_name: ComfyUI
  endpoint:
    url: http://127.0.0.1:8188
    loopback_only: true
  health:
    method: get
    path: /system_stats
    timeout_seconds: 3
    expect_status: [200]
  tiers:
    - id: cpu
      label: CPU
      preset:
        queue_size: 1
  secrets:
    - shinku_comfyui_key
capabilities:
  - id: portrait_cutout
    display_name: Transparent Cutout
    short_hint: Remove an image background.
    visible_in: [desktop, web]
    prompt_exposed: true
    risk: low
    confirm: never
    effects: [media_generation]
    trigger:
      kind: workspace_has_image
    workflow_template: workflows/portrait_cutout.json
    inputs:
      - name: image
        kind: image_bytes
        required: true
        max_bytes: 8_388_608
    outputs:
      - name: cutout
        kind: image_bytes
        delivery: generated_file
"""
        )
        self.assertIsInstance(manifest, CapabilityManifest)
        assert isinstance(manifest, CapabilityManifest)

        self.assertEqual(manifest.schema, SCHEMA)
        self.assertEqual(manifest.provider_id, "comfyui")
        self.assertEqual(manifest.provider_type, "comfyui")
        self.assertEqual(manifest.display_name, "ComfyUI")
        self.assertEqual(manifest.source_layer, "builtin")
        self.assertEqual(manifest.source_path, self.root / "provider.yaml")

        assert manifest.endpoint is not None
        self.assertEqual(manifest.endpoint.url, "http://127.0.0.1:8188")
        self.assertTrue(manifest.endpoint.loopback_only)

        assert manifest.health is not None
        self.assertEqual(manifest.health.method, "GET")  # 归一化为大写
        self.assertEqual(manifest.health.path, "/system_stats")
        self.assertEqual(manifest.health.timeout_seconds, 3.0)
        self.assertEqual(manifest.health.expect_status, (200,))

        self.assertEqual([tier.id for tier in manifest.tiers], ["cpu"])
        self.assertEqual(manifest.tiers[0].label, "CPU")
        self.assertEqual(manifest.tiers[0].preset, {"queue_size": 1})
        self.assertEqual(manifest.secrets, ("shinku_comfyui_key",))

        capability = manifest.capabilities[0]
        self.assertEqual(capability.id, "portrait_cutout")
        self.assertEqual(capability.display_name, "Transparent Cutout")
        self.assertEqual(capability.short_hint, "Remove an image background.")
        self.assertEqual(capability.visible_in, ("desktop", "web"))
        self.assertTrue(capability.prompt_exposed)
        self.assertEqual((capability.risk, capability.confirm), ("low", "never"))
        self.assertEqual(capability.effects, ("media_generation",))
        assert capability.trigger is not None
        self.assertEqual(capability.trigger.kind, "workspace_has_image")
        self.assertEqual(capability.inputs[0].max_bytes, 8_388_608)
        self.assertTrue(capability.inputs[0].required)
        self.assertEqual(capability.outputs[0].delivery, "generated_file")

    def test_schema_is_echoed_as_the_constant_not_as_written(self) -> None:
        # 契约 §3.3：schema 字段恒为常量本身，不是原样回填。
        manifest = self.load("schema: '  capability_adapter/v1  '\nprovider:\n  id: demo\n  type: mcp_stdio\n")
        self.assertIsInstance(manifest, CapabilityManifest)
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.schema, SCHEMA)

    def test_display_name_falls_back_to_provider_id(self) -> None:
        manifest = self.load(_provider(""))
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.display_name, "demo")

    def test_unknown_top_level_keys_survive_in_raw(self) -> None:
        manifest = self.load(_provider("  x_custom: 7\n", provider_id="demo") + "x_upper: 3\n")
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.raw["x_upper"], 3)
        self.assertEqual(manifest.raw["provider"]["x_custom"], 7)  # type: ignore[index]

    def test_a_capability_keeps_its_provider_specific_keys(self) -> None:
        manifest = self.load(_provider("") + "capabilities:\n  - id: echo\n    workflow_template: wf.json\n")
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.capabilities[0].raw["workflow_template"], "wf.json")

    def test_absent_endpoint_is_none(self) -> None:
        for body in ("", '  endpoint: ""\n', "  endpoint:\n"):
            with self.subTest(body=body):
                manifest = self.load(_provider(body))
                assert isinstance(manifest, CapabilityManifest)
                self.assertIsNone(manifest.endpoint)

    def test_an_empty_endpoint_mapping_yields_a_blank_url(self) -> None:
        # 契约 §3.4：`{}` 是映射，因此不报错，只是 url 为空、不要求回环。
        manifest = self.load(_provider("  endpoint: {}\n"))
        assert isinstance(manifest, CapabilityManifest)
        assert manifest.endpoint is not None
        self.assertEqual(manifest.endpoint.url, "")
        self.assertFalse(manifest.endpoint.loopback_only)

    def test_loopback_accepts_every_loopback_spelling(self) -> None:
        # IPv6 必须写成方括号形式，否则 urlparse 的 hostname 会取空。
        urls = ("http://127.0.0.1:8188/x", "http://localhost:8188/x", "http://[::1]:8188/x")
        for url in urls:
            with self.subTest(url=url):
                body = f"  endpoint:\n    url: {url}\n    loopback_only: true\n"
                manifest = self.load(_provider(body, provider_type="comfyui"))
                self.assertIsInstance(manifest, CapabilityManifest)

    def test_an_unbracketed_ipv6_url_is_not_recognised_as_loopback(self) -> None:
        # urlparse("http://::1:8188/x").hostname 取到的是空串 —— 旧实现同样拒绝。
        body = "  endpoint:\n    url: http://::1:8188/x\n    loopback_only: true\n"
        result = self.load(_provider(body, provider_type="comfyui"))
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "endpoint_not_loopback")
        self.assertEqual(result.detail, "host=<empty>")

    def test_health_defaults_when_fields_are_missing(self) -> None:
        manifest = self.load(_provider("  health: {}\n"))
        assert isinstance(manifest, CapabilityManifest)
        assert manifest.health is not None
        self.assertEqual(manifest.health.method, "GET")
        self.assertEqual(manifest.health.path, "")
        self.assertEqual(manifest.health.timeout_seconds, 3.0)
        self.assertEqual(manifest.health.expect_status, (200,))

    def test_health_drops_unparseable_statuses_and_timeout(self) -> None:
        manifest = self.load(
            _provider("  health:\n    expect_status: [200, nope, '201']\n    timeout_seconds: abc\n")
        )
        assert isinstance(manifest, CapabilityManifest)
        assert manifest.health is not None
        self.assertEqual(manifest.health.expect_status, (200, 201))
        self.assertEqual(manifest.health.timeout_seconds, 3.0)

    def test_non_mapping_health_is_silently_dropped(self) -> None:
        manifest = self.load(_provider("  health: [1, 2]\n"))
        assert isinstance(manifest, CapabilityManifest)
        self.assertIsNone(manifest.health)

    def test_tiers_default_to_empty_and_non_mapping_preset_becomes_empty(self) -> None:
        manifest = self.load(_provider("  tiers:\n    - id: cpu\n      preset: cpu\n"))
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.tiers[0].preset, {})
        self.assertEqual(manifest.tiers[0].label, "cpu")  # label 回退为 id

        defaulted = self.load(_provider(""), name="second.yaml")
        assert isinstance(defaulted, CapabilityManifest)
        self.assertEqual(defaulted.tiers, ())

    def test_capabilities_are_read_from_the_top_level_only(self) -> None:
        # 契约 §3.3：capabilities 在顶层；provider 里的同名键不被读取。
        nested = _provider("")
        nested = nested.replace(
            "  type: mcp_stdio\n", "  type: mcp_stdio\n  capabilities:\n    - id: nested\n"
        )
        manifest = self.load(nested + "capabilities:\n  - id: top_level\n")
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual([item.id for item in manifest.capabilities], ["top_level"])

        only_nested = self.load(
            _provider("").replace(
                "  type: mcp_stdio\n", "  type: mcp_stdio\n  capabilities:\n    - id: nested\n"
            ),
            name="third.yaml",
        )
        assert isinstance(only_nested, CapabilityManifest)
        self.assertEqual(only_nested.capabilities, ())

    def test_capabilities_absent_or_empty_produce_no_descriptors(self) -> None:
        for suffix in ("", 'capabilities: ""\n', "capabilities: []\n", "capabilities:\n"):
            with self.subTest(suffix=suffix):
                manifest = self.load(_provider("") + suffix)
                assert isinstance(manifest, CapabilityManifest)
                self.assertEqual(manifest.capabilities, ())

    def test_prompt_exposed_defaults_to_false(self) -> None:
        manifest = self.load(_provider("") + "capabilities:\n  - id: echo\n")
        assert isinstance(manifest, CapabilityManifest)
        self.assertFalse(manifest.capabilities[0].prompt_exposed)

    def test_visible_in_requires_a_list(self) -> None:
        manifest = self.load(_provider("") + "capabilities:\n  - id: echo\n    visible_in: desktop\n")
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.capabilities[0].visible_in, ())

    def test_null_and_zero_entries_in_visible_in_are_dropped_not_stringified(self) -> None:
        # 契约 §3.4：取串口径是 `str(item or "").strip()`，因此 None / 0 取空后被丢弃。
        # 若错写成 `str(item).strip()`，这里会得到 ("None", "0") 并误报 visible_in_invalid。
        manifest = self.load(
            _provider("") + "capabilities:\n  - id: echo\n    visible_in: [null, 0, desktop]\n"
        )
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.capabilities[0].visible_in, ("desktop",))

    def test_effects_use_the_same_collapsing_rule(self) -> None:
        manifest = self.load(
            _provider("") + "capabilities:\n  - id: echo\n    effects: [null, 0, file_read]\n"
        )
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.capabilities[0].effects, ("file_read",))

    def test_trigger_requires_a_non_empty_kind(self) -> None:
        manifest = self.load(
            _provider("")
            + "capabilities:\n  - id: a\n    trigger: workspace\n  - id: b\n    trigger:\n      kind: \"\"\n  - id: c\n    trigger:\n      kind: on_load\n"
        )
        assert isinstance(manifest, CapabilityManifest)
        self.assertIsNone(manifest.capabilities[0].trigger)
        self.assertIsNone(manifest.capabilities[1].trigger)
        assert manifest.capabilities[2].trigger is not None
        self.assertEqual(manifest.capabilities[2].trigger.kind, "on_load")

    def test_io_slots_skip_incomplete_entries(self) -> None:
        manifest = self.load(
            _provider("")
            + """capabilities:
  - id: echo
    inputs:
      - name: ok
        kind: text
      - name: missing_kind
      - kind: missing_name
      - name: ""
        kind: text
      - not_a_mapping
"""
        )
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual([slot.name for slot in manifest.capabilities[0].inputs], ["ok"])
        self.assertFalse(manifest.capabilities[0].inputs[0].required)
        self.assertIsNone(manifest.capabilities[0].inputs[0].max_bytes)
        self.assertEqual(manifest.capabilities[0].inputs[0].delivery, "")

    def test_io_slot_byte_ceilings_accept_underscored_text(self) -> None:
        manifest = self.load(
            _provider("")
            + """capabilities:
  - id: echo
    inputs:
      - name: a
        kind: image
        max_bytes: "8_388_608"
      - name: b
        kind: image
        max_bytes: not_a_number
      - name: c
        kind: image
        max_bytes: ""
"""
        )
        assert isinstance(manifest, CapabilityManifest)
        slots = manifest.capabilities[0].inputs
        self.assertEqual([slot.max_bytes for slot in slots], [8388608, None, None])

    def test_outputs_are_read_from_the_same_reader_as_inputs(self) -> None:
        manifest = self.load(
            _provider("")
            + "capabilities:\n  - id: echo\n    outputs:\n      - name: out\n        kind: text\n        delivery: inline\n"
        )
        assert isinstance(manifest, CapabilityManifest)
        self.assertEqual(manifest.capabilities[0].outputs[0].delivery, "inline")
        self.assertEqual(manifest.capabilities[0].inputs, ())


# --------------------------------------------------------------------------- #
# §3.2 / §3.6 拒绝路径
# --------------------------------------------------------------------------- #


class ManifestRejectionTableTests(_ManifestCase):
    """逐行钉住原因码、detail 与 provider_id（契约 §3.6）。"""

    CASES: tuple[tuple[str, str, str, str, str], ...] = (
        (
            "schema-mismatch",
            "schema: other/v1\nprovider:\n  id: demo\n  type: mcp_stdio\n",
            "schema_mismatch",
            f"expected {SCHEMA}",
            "",
        ),
        (
            "top-level-list",
            "- just\n- a\n- list\n",
            "manifest_must_be_mapping",
            "top-level yaml must be a mapping",
            "",
        ),
        (
            "empty-file",
            "",
            "manifest_must_be_mapping",
            "top-level yaml must be a mapping",
            "",
        ),
        (
            "provider-missing",
            f"schema: {SCHEMA}\n",
            "missing_provider",
            "provider must be a mapping",
            "",
        ),
        (
            "provider-not-a-mapping",
            f"schema: {SCHEMA}\nprovider:\n  - demo\n",
            "missing_provider",
            "provider must be a mapping",
            "",
        ),
        (
            "provider-id-missing",
            _provider("", provider_id="''"),
            "missing_provider_id",
            "provider.id is required",
            "",
        ),
        (
            "provider-id-unsupported-characters",
            _provider("", provider_id="'bad id'"),
            "invalid_provider_id",
            "provider.id contains unsupported characters",
            "bad id",
        ),
        (
            "provider-type-missing",
            f"schema: {SCHEMA}\nprovider:\n  id: demo\n",
            "missing_provider_type",
            "provider.type is required",
            "demo",
        ),
        (
            "provider-type-unsupported-characters",
            _provider("", provider_type="'bad type'"),
            "provider_type_invalid",
            "provider.type contains unsupported characters",
            "demo",
        ),
        (
            "provider-type-not-allowed",
            _provider("", provider_type="not_allowed"),
            "provider_type_not_allowed",
            "not_allowed",
            "demo",
        ),
        (
            "endpoint-not-a-mapping",
            _provider("  endpoint: [1]\n"),
            "endpoint_must_be_mapping",
            "provider.endpoint must be a mapping",
            "demo",
        ),
        (
            "endpoint-not-loopback",
            _provider(
                "  endpoint:\n    url: http://example.com:8188\n    loopback_only: true\n",
                provider_type="comfyui",
            ),
            "endpoint_not_loopback",
            "host=example.com",
            "demo",
        ),
        (
            "endpoint-host-unreadable",
            _provider("  endpoint:\n    url: 'not a url'\n    loopback_only: true\n"),
            "endpoint_not_loopback",
            "host=<empty>",
            "demo",
        ),
        (
            "secrets-not-a-list",
            _provider("  secrets: shinku_key\n"),
            "secrets_must_be_key_names",
            "provider.secrets must be a list of key names",
            "demo",
        ),
        (
            "secret-entry-not-a-string",
            _provider("  secrets:\n    - 123\n"),
            "secrets_must_be_key_names",
            "secret entries must be strings",
            "demo",
        ),
        (
            "secret-contains-equals",
            _provider("  secrets:\n    - api_key=sk-x\n"),
            "secrets_must_be_key_names",
            "secret entry looks like a value",
            "demo",
        ),
        (
            "secret-contains-colon",
            _provider("  secrets:\n    - 'a:b'\n"),
            "secrets_must_be_key_names",
            "secret entry looks like a value",
            "demo",
        ),
        (
            "secret-contains-whitespace",
            _provider("  secrets:\n    - 'has space'\n"),
            "secrets_must_be_key_names",
            "secret entry looks like a value",
            "demo",
        ),
        (
            "secret-too-long",
            _provider("  secrets:\n    - " + "x" * 121 + "\n"),
            "secrets_must_be_key_names",
            "secret entry looks like a value",
            "demo",
        ),
        (
            "secret-empty",
            _provider("  secrets:\n    - ''\n"),
            "secrets_must_be_key_names",
            "secret entry looks like a value",
            "demo",
        ),
        (
            "secret-too-generic",
            _provider("  secrets:\n    - API_KEY\n"),
            "secrets_must_be_key_names",
            "secret entry is too generic",
            "demo",
        ),
        (
            "tiers-not-a-list",
            _provider("  tiers: cpu\n"),
            "tiers_must_be_list",
            "provider.tiers must be a list",
            "demo",
        ),
        (
            "tier-not-a-mapping",
            _provider("  tiers:\n    - cpu\n"),
            "tier_must_be_mapping",
            "tier entry must be a mapping",
            "demo",
        ),
        (
            "tier-id-empty",
            _provider("  tiers:\n    - label: CPU\n"),
            "tier_id_not_unique",
            "<empty>",
            "demo",
        ),
        (
            "tier-id-duplicated",
            _provider("  tiers:\n    - id: cpu\n    - id: cpu\n"),
            "tier_id_not_unique",
            "cpu",
            "demo",
        ),
        (
            "capabilities-not-a-list",
            _provider("") + "capabilities: {}\n",
            "capabilities_must_be_list",
            "capabilities must be a list",
            "demo",
        ),
        (
            "capability-not-a-mapping",
            _provider("") + "capabilities:\n  - echo\n",
            "capability_must_be_mapping",
            "capability entry must be a mapping",
            "demo",
        ),
        (
            "capability-id-missing",
            _provider("") + "capabilities:\n  - display_name: Echo\n",
            "missing_capability_id",
            "capability.id is required",
            "demo",
        ),
        (
            "visible-in-invalid",
            _provider("") + "capabilities:\n  - id: echo\n    visible_in: [bogus]\n",
            "visible_in_invalid",
            "bogus",
            "demo",
        ),
        (
            "effects-invalid",
            _provider("") + "capabilities:\n  - id: echo\n    effects: [teleport]\n",
            "effects_invalid",
            "teleport",
            "demo",
        ),
        (
            "effects-invalid-lists-every-unknown-value",
            _provider("") + "capabilities:\n  - id: echo\n    effects: [teleport, file_read, jump]\n",
            "effects_invalid",
            "teleport,jump",
            "demo",
        ),
    )

    def test_every_documented_reason_code_is_reachable(self) -> None:
        for name, text, reason, detail, provider_id in self.CASES:
            with self.subTest(case=name):
                result = self.load(text, name=f"{name}.yaml")
                self.assertIsInstance(result, InvalidManifest, f"{name}: expected a rejection")
                assert isinstance(result, InvalidManifest)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.detail, detail)
                self.assertEqual(result.provider_id, provider_id)
                self.assertEqual(result.source_path, self.root / f"{name}.yaml")
                self.assertEqual(result.source_layer, "builtin")

    def test_the_table_covers_every_reason_the_contract_lists(self) -> None:
        documented = {
            "yaml_parse_error",
            "read_error",
            "manifest_must_be_mapping",
            "schema_mismatch",
            "missing_provider",
            "missing_provider_id",
            "invalid_provider_id",
            "missing_provider_type",
            "provider_type_invalid",
            "provider_type_not_allowed",
            "endpoint_must_be_mapping",
            "endpoint_not_loopback",
            "secrets_must_be_key_names",
            "tiers_must_be_list",
            "tier_must_be_mapping",
            "tier_id_not_unique",
            "capabilities_must_be_list",
            "capability_must_be_mapping",
            "missing_capability_id",
            "visible_in_invalid",
            "effects_invalid",
        }
        covered = {case[2] for case in self.CASES}
        self.assertEqual(covered, documented - {"yaml_parse_error", "read_error"})


class ManifestRejectionOrderTests(_ManifestCase):
    """契约 §3.2 的顺序即优先级：首个问题决定原因码。"""

    def test_schema_is_checked_before_provider(self) -> None:
        result = self.load("schema: nope/provider\n")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "schema_mismatch")

    def test_provider_id_is_checked_before_provider_type(self) -> None:
        result = self.load(f"schema: {SCHEMA}\nprovider:\n  type: not_allowed\n")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "missing_provider_id")

    def test_endpoint_is_checked_before_secrets(self) -> None:
        result = self.load(_provider("  endpoint: [1]\n  secrets:\n    - a=b\n"))
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "endpoint_must_be_mapping")

    def test_secrets_are_checked_before_tiers(self) -> None:
        result = self.load(_provider("  secrets:\n    - a=b\n  tiers: cpu\n"))
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "secrets_must_be_key_names")

    def test_tiers_are_checked_before_capabilities(self) -> None:
        result = self.load(_provider("  tiers: cpu\n") + "capabilities:\n  - id: echo\n    visible_in: [bogus]\n")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "tiers_must_be_list")

    def test_provider_type_is_checked_before_capabilities(self) -> None:
        result = self.load(
            _provider("", provider_type="not_allowed")
            + "capabilities:\n  - id: echo\n    visible_in: [bogus]\n"
        )
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "provider_type_not_allowed")

    def test_visible_in_is_checked_before_effects(self) -> None:
        result = self.load(
            _provider("")
            + "capabilities:\n  - id: echo\n    visible_in: [bogus]\n    effects: [teleport]\n"
        )
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "visible_in_invalid")

    def test_capability_id_is_checked_before_visible_in(self) -> None:
        result = self.load(_provider("") + "capabilities:\n  - visible_in: [bogus]\n")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "missing_capability_id")


class ManifestReadFailureTests(_ManifestCase):
    def test_a_broken_yaml_body_reports_a_parse_error(self) -> None:
        result = self.load(f"schema: {SCHEMA}\nprovider:\n  id: broken\ncapabilities:\n  - id: x\n    visible_in: [desktop\n")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "yaml_parse_error")
        self.assertTrue(result.detail)

    def test_a_missing_file_reports_a_read_error(self) -> None:
        result = load_manifest(self.root / "does_not_exist.yaml", source_layer="profile")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "read_error")
        self.assertEqual(result.source_layer, "profile")
        self.assertTrue(result.detail)

    def test_a_directory_reports_a_read_error(self) -> None:
        directory = self.root / "a_folder"
        directory.mkdir()
        result = load_manifest(directory, source_layer="builtin")
        assert isinstance(result, InvalidManifest)
        self.assertEqual(result.reason, "read_error")

    def test_non_utf8_bytes_escape_instead_of_being_reported(self) -> None:
        # 契约 §3.6 的"已知边界"：旧实现只捕 YAMLError 与 OSError，
        # UnicodeDecodeError 会逃逸。本批保持等价行为，并把它钉住。
        path = self.root / "binary.yaml"
        path.write_bytes(b"\xff\xfe\x00not utf-8")
        with self.assertRaises(UnicodeDecodeError):
            load_manifest(path, source_layer="builtin")

    def test_a_bom_is_tolerated(self) -> None:
        path = self.root / "bom.yaml"
        path.write_bytes(("\ufeff" + _provider("")).encode("utf-8"))
        self.assertIsInstance(load_manifest(path, source_layer="builtin"), CapabilityManifest)


# --------------------------------------------------------------------------- #
# §3.5 风险与确认的归一化 / 提升
# --------------------------------------------------------------------------- #


class PolicyEscalationTests(_ManifestCase):
    def policy(self, *, risk: str | None, confirm: str | None, effects: str) -> tuple[str, str]:
        lines = ["capabilities:", "  - id: echo"]
        if risk is not None:
            lines.append(f"    risk: {risk}")
        if confirm is not None:
            lines.append(f"    confirm: {confirm}")
        lines.append(f"    effects: [{effects}]")
        manifest = self.load(_provider("") + "\n".join(lines) + "\n")
        assert isinstance(manifest, CapabilityManifest)
        capability = manifest.capabilities[0]
        return capability.risk, capability.confirm

    def test_declared_defaults_are_medium_and_first_time(self) -> None:
        self.assertEqual(self.policy(risk=None, confirm=None, effects="file_read"), ("medium", "first_time"))

    def test_unrecognised_values_fall_back_to_the_defaults(self) -> None:
        self.assertEqual(self.policy(risk="severe", confirm="maybe", effects="file_read"), ("medium", "first_time"))

    def test_declared_values_are_lower_cased_before_lookup(self) -> None:
        self.assertEqual(self.policy(risk="LOW", confirm="NEVER", effects="file_read"), ("low", "never"))

    def test_command_exec_forces_high_and_always(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects="command_exec"), ("high", "always"))

    def test_browser_action_forces_high_and_always(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects="browser_action"), ("high", "always"))

    def test_file_write_lifts_low_to_medium_and_never_to_first_time(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects="file_write"), ("medium", "first_time"))

    def test_network_outbound_lifts_low_but_keeps_a_stricter_confirm(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="always", effects="network_outbound"), ("medium", "always"))

    def test_file_write_does_not_lift_a_medium_declaration(self) -> None:
        self.assertEqual(self.policy(risk="medium", confirm="never", effects="file_write"), ("medium", "never"))

    def test_media_generation_is_not_a_promoting_effect(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects="media_generation"), ("low", "never"))

    def test_a_declared_high_risk_is_forced_to_always(self) -> None:
        self.assertEqual(self.policy(risk="high", confirm="never", effects="file_read"), ("high", "always"))

    def test_forcing_effects_win_over_the_promoting_branch(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects="file_write, command_exec"), ("high", "always"))

    def test_no_effects_leaves_a_low_declaration_alone(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="never", effects=""), ("low", "never"))

    def test_escalation_does_not_weaken_a_declared_confirmation(self) -> None:
        self.assertEqual(self.policy(risk="low", confirm="first_time", effects="file_write"), ("medium", "first_time"))


# --------------------------------------------------------------------------- #
# §4.1 / §4.3 / §4.4 资源清单的索引
# --------------------------------------------------------------------------- #


class ResourceSurfaceTests(unittest.TestCase):
    def test_the_exported_names_are_exactly_the_contract_list(self) -> None:
        self.assertEqual(
            list(resource_manifest.__all__),
            [
                "AUDIO_EXTS",
                "BACKGROUND_ALIASES",
                "EMOTION_ALIASES",
                "IMAGE_EXTS",
                "PROMPT_NOTE_LIMIT",
                "ResourceManifest",
            ],
        )

    def test_label_and_priority_tables_match_the_contract(self) -> None:
        self.assertEqual(BACKGROUND_LABELS, {"morning": "清晨", "afternoon": "午后", "evening": "黄昏", "night": "深夜"})
        self.assertEqual(EMOTION_LABELS, {"normal": "平静", "shy": "害羞", "smug": "得意", "cry": "哭哭"})
        self.assertEqual(BACKGROUND_PRIORITY, {"evening": 0, "morning": 1, "afternoon": 2, "night": 3})

    def test_emotion_aliases_and_fallback_ladders_are_intact(self) -> None:
        self.assertEqual(EMOTION_ALIASES["sumg"], "smug")
        self.assertEqual(EMOTION_FALLBACK_CANDIDATES["happy"], ["开心", "卖萌", "得意", "smug", "normal"])
        self.assertEqual(EMOTION_FALLBACK_CANDIDATES["pet"], ["被摸头", "求摸摸", "开心", "normal"])
        self.assertEqual(len(EMOTION_FALLBACK_CANDIDATES), 20)

    def test_background_aliases_map_every_chinese_spelling(self) -> None:
        self.assertEqual(BACKGROUND_ALIASES["清晨"], "morning")
        self.assertEqual(BACKGROUND_ALIASES["黄昏"], "evening")
        self.assertEqual(BACKGROUND_ALIASES["夜里"], "night")
        self.assertEqual(BACKGROUND_ALIASES["下午"], "afternoon")

    def test_note_tables_match_the_contract(self) -> None:
        self.assertEqual(PROMPT_NOTE_LIMIT, 240)
        self.assertIn("说明.md", NOTE_FILENAMES)
        self.assertEqual(SIDECAR_NOTE_SUFFIXES, (".md", ".txt", ".note.md", ".note.txt"))
        self.assertEqual(META_NOTE_KEYS, ("notes", "note", "prompt", "ai_hint", "ai_prompt", "usage"))

    def test_the_facade_is_owned_by_the_new_package(self) -> None:
        self.assertEqual(ResourceManifest.__module__, "shinku.resources.manifest")

    def test_emotion_aliases_is_keyword_only(self) -> None:
        with TemporaryDirectory() as temp:
            with self.assertRaises(TypeError):
                ResourceManifest(temp, "/assets", {"happy": ["开心"]})  # type: ignore[misc]


class ResourceIndexTests(_PackCase):
    def test_nested_scenes_and_outfits_are_indexed_with_defaults(self) -> None:
        self.meta("scenes/school/meta.json", {"id": "school", "name": "学校", "default_minor": "classroom"})
        self.meta("scenes/school/classroom/meta.json", {"id": "classroom", "name": "教室", "default_background": "evening"})
        self.asset("scenes/school/classroom/morning.png")
        self.asset("scenes/school/classroom/evening.png")
        self.meta("characters/校服/meta.json", {"allowed_emotions": ["normal", "smug"], "default_emotion": "smug"})
        self.asset("characters/校服/normal.png")
        self.asset("characters/校服/sumg.png")
        self.asset("characters/校服/cry.png")

        manifest = self.build().refresh()

        self.assertEqual(manifest["schema_version"], 2)
        major = manifest["scenes"]["majors"][0]
        self.assertEqual(major["id"], "school")
        self.assertEqual(major["name"], "学校")
        self.assertEqual(major["default_minor"], "classroom")
        minor = major["minors"][0]
        self.assertEqual(minor["id"], "classroom")
        self.assertEqual([item["id"] for item in minor["backgrounds"]], ["evening", "morning"])
        self.assertEqual(minor["default_background"], "evening")

        outfit = manifest["characters"]["outfits"][0]
        self.assertEqual(outfit["id"], "校服")
        self.assertEqual(outfit["default_emotion"], "smug")
        self.assertEqual(outfit["allowed_emotions"], ["smug", "normal"])
        self.assertEqual(manifest["defaults"]["emotion"], "smug")

    def test_a_whitelist_that_matches_nothing_is_treated_as_unset(self) -> None:
        self.meta("characters/default/meta.json", {"allowed_emotions": ["nonexistent"]})
        self.asset("characters/default/normal.png")
        self.asset("characters/default/cry.png")
        outfit = self.build().refresh()["characters"]["outfits"][0]
        self.assertEqual([item["id"] for item in outfit["emotions"]], ["cry", "normal"])

    def test_legacy_suffixes_and_flat_backgrounds_keep_stable_ids(self) -> None:
        self.asset("scenes/home/卧室_白天.png")
        self.asset("scenes/home/卧室_夜晚.png")
        self.asset("scenes/home/客厅_黄昏.png")
        self.asset("scenes/街道/黄昏街道.png")
        self.asset("backgrounds/evening.png")

        manifest = self.build().refresh()
        majors = {item["id"]: item for item in manifest["scenes"]["majors"]}
        self.assertEqual(set(majors), {"home", "default", "街道"})

        bedroom = next(item for item in majors["home"]["minors"] if item["id"] == "卧室")
        self.assertEqual([item["id"] for item in bedroom["backgrounds"]], ["morning", "night"])
        self.assertEqual([item["id"] for item in majors["home"]["minors"]], ["卧室", "客厅"])
        self.assertEqual(majors["街道"]["minors"][0]["backgrounds"][0]["id"], "黄昏街道")
        self.assertEqual(majors["default"]["minors"][0]["backgrounds"][0]["id"], "evening")

    def test_a_pack_without_any_assets_falls_back_to_placeholders(self) -> None:
        manifest = self.build().refresh()
        self.assertEqual(manifest["schema_version"], 2)
        major = manifest["scenes"]["majors"][0]
        self.assertEqual(major["id"], "default")
        self.assertNotIn("default_minor", major)
        minor = major["minors"][0]
        self.assertEqual(minor["backgrounds"][0]["id"], "default")
        self.assertEqual(minor["bgm_tracks"], [])
        outfit = manifest["characters"]["outfits"][0]
        self.assertEqual(outfit["id"], "default")
        self.assertEqual(outfit["allowed_emotions"], ["normal"])
        self.assertEqual(
            manifest["defaults"],
            {
                "major": "default",
                "minor": "default",
                "background": "default",
                "bgm": "",
                "outfit": "default",
                "emotion": "normal",
            },
        )

    def test_ignored_directories_are_skipped(self) -> None:
        self.asset("scenes/home/__pycache__/ghost.png")
        self.asset("scenes/backup/archive.png")
        self.asset("scenes/备份/old.png")
        self.asset("scenes/home/real.png")
        majors = {item["id"] for item in self.build().refresh()["scenes"]["majors"]}
        self.assertEqual(majors, {"home"})

    def test_asset_metadata_supplies_name_description_and_aliases(self) -> None:
        self.meta(
            "scenes/home/evening.meta.json",
            {"name": "傍晚", "description": "  黄昏时分  ", "aliases": ["dusk", "黄昏"]},
        )
        self.asset("scenes/home/evening.png")
        background = self.build().refresh()["scenes"]["majors"][0]["minors"][0]["backgrounds"][0]
        self.assertEqual(background["id"], "evening")
        self.assertEqual(background["name"], "傍晚")
        self.assertEqual(background["description"], "黄昏时分")
        self.assertEqual(background["aliases"], ["dusk", "黄昏"])
        self.assertTrue(background["path"].startswith("/assets/scenes/home/"))

    def test_a_file_stem_is_appended_to_aliases_when_it_differs_from_the_id(self) -> None:
        self.meta("scenes/home/dawn.meta.json", {"id": "morning"})
        self.asset("scenes/home/dawn.png")
        background = self.build().refresh()["scenes"]["majors"][0]["minors"][0]["backgrounds"][0]
        self.assertEqual(background["id"], "morning")
        self.assertEqual(background["aliases"], ["dawn"])

    def test_label_tables_are_used_as_the_name_fallback_for_backgrounds_and_emotions(self) -> None:
        self.asset("scenes/home/evening.png")
        self.asset("characters/default/shy.png")
        manifest = self.build().refresh()
        self.assertEqual(manifest["scenes"]["majors"][0]["minors"][0]["backgrounds"][0]["name"], "黄昏")
        self.assertEqual(manifest["characters"]["outfits"][0]["emotions"][0]["name"], "害羞")

    def test_the_emotion_label_table_also_applies_to_tracks(self) -> None:
        # 契约 §4.4 记录的行为：非 background 的类别一律走 EMOTION_LABELS。
        # 这是旧实现的事实，本批照录不改（改它属行为变更，需单独决定）。
        self.asset("scenes/home/evening.png")
        self.asset("bgm/default/shy.ogg")
        minor = self.build().refresh()["scenes"]["majors"][0]["minors"][0]
        track = next(item for item in minor["bgm_tracks"] if item["id"] == "shy")
        self.assertEqual(track["name"], "害羞")

    def test_directory_notes_are_read_and_clipped(self) -> None:
        self.text_file("scenes/battlefield/ai.md", "不是日常场景。")
        self.asset("scenes/battlefield/frontline/dust.png")
        major = self.build().refresh()["scenes"]["majors"][0]
        self.assertEqual(major["notes"], "不是日常场景。")

        long_note = "字" * 300
        self.text_file("scenes/other/prompt.md", long_note)
        self.asset("scenes/other/place/here.png")
        other = next(
            item for item in self.build().refresh()["scenes"]["majors"] if item["id"] == "other"
        )
        self.assertEqual(len(other["notes"]), PROMPT_NOTE_LIMIT)
        self.assertTrue(other["notes"].endswith("…"))
        self.assertEqual(other["notes"], long_note[: PROMPT_NOTE_LIMIT - 1] + "…")

    def test_sidecar_notes_attach_to_the_asset(self) -> None:
        self.asset("scenes/home/evening.png")
        self.text_file("scenes/home/evening.md", "傍晚的教室。")
        background = self.build().refresh()["scenes"]["majors"][0]["minors"][0]["backgrounds"][0]
        self.assertEqual(background["notes"], "傍晚的教室。")

    def test_metadata_notes_are_merged_in_contract_key_order(self) -> None:
        self.meta("scenes/home/evening.meta.json", {"note": "第二", "notes": "第一", "usage": "第三"})
        self.asset("scenes/home/evening.png")
        background = self.build().refresh()["scenes"]["majors"][0]["minors"][0]["backgrounds"][0]
        self.assertEqual(background["notes"], "第一 第二 第三")

    def test_merging_keeps_the_earliest_entry_as_the_base(self) -> None:
        # 平铺的 backgrounds/ 与 scenes/ 里的同名条目必须合并，且先出现的为准。
        self.asset("scenes/home/evening.png")
        self.asset("backgrounds/evening.png")
        majors = {item["id"]: item for item in self.build().refresh()["scenes"]["majors"]}
        self.assertEqual(sorted(majors), ["default", "home"])
        self.assertEqual(majors["default"]["minors"][0]["backgrounds"][0]["id"], "evening")

    def test_bgm_tracks_are_distributed_by_folder_depth(self) -> None:
        self.asset("scenes/school/classroom/evening.png")
        self.asset("bgm/school/classroom/morning.ogg")
        self.asset("bgm/school/classroom/evening.ogg")
        self.asset("bgm/default/default/theme.ogg")
        minor = self.build().refresh()["scenes"]["majors"][0]["minors"][0]
        self.assertEqual([item["id"] for item in minor["bgm_tracks"]], ["evening", "morning"])

    def test_a_scene_without_its_own_tracks_inherits_the_shared_bucket(self) -> None:
        self.asset("scenes/home/room/evening.png")
        self.asset("bgm/default/default/theme.ogg")
        minor = self.build().refresh()["scenes"]["majors"][0]["minors"][0]
        self.assertEqual([item["id"] for item in minor["bgm_tracks"]], ["theme"])

    def test_no_bgm_directory_leaves_every_track_list_empty(self) -> None:
        self.asset("scenes/home/room/evening.png")
        minor = self.build().refresh()["scenes"]["majors"][0]["minors"][0]
        self.assertEqual(minor["bgm_tracks"], [])

    def test_non_asset_suffixes_are_not_indexed(self) -> None:
        self.asset("scenes/home/notes.md")
        self.asset("scenes/home/evening.gif")
        self.asset("scenes/home/evening.png")
        minor = self.build().refresh()["scenes"]["majors"][0]["minors"][0]
        self.assertEqual([item["id"] for item in minor["backgrounds"]], ["evening"])

    def test_scene_order_puts_the_default_minor_first(self) -> None:
        self.meta("scenes/home/meta.json", {"default_minor": "bedroom"})
        self.asset("scenes/home/living/a.png")
        self.asset("scenes/home/bedroom/b.png")
        self.asset("scenes/home/attic/c.png")
        major = self.build().refresh()["scenes"]["majors"][0]
        self.assertEqual([item["id"] for item in major["minors"]], ["bedroom", "attic", "living"])


class ResourcePathTests(_PackCase):
    def test_the_public_prefix_is_normalised(self) -> None:
        cases = {
            "assets": "/assets",
            "/pack/assets": "/pack/assets",
            r"\pack\assets\\": "/pack/assets",
            "": "/assets",
            "/": "/",
        }
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(ResourceManifest(self.root, given).public_prefix, expected)

    def test_public_paths_are_relative_to_the_assets_directory(self) -> None:
        self.asset("scenes/school/classroom/evening.png")
        self.asset("characters/default/normal.png")
        manifest = self.build(public_prefix="/pack/assets")
        manifest.refresh()
        bundle = manifest.resolve_visual_bundle(
            {
                "emotion": "normal",
                "character": {"outfit": "default"},
                "scene": {"major": "school", "minor": "classroom", "background": "evening"},
            }
        )
        self.assertEqual(bundle["background"]["path"], "/pack/assets/scenes/school/classroom/evening.png")

    def test_get_manifest_caches_until_refreshed(self) -> None:
        self.asset("scenes/home/evening.png")
        service = self.build()
        first = service.get_manifest()
        self.assertIs(service.get_manifest(), first)
        self.asset("scenes/home/night.png")
        self.assertIsNot(service.refresh(), first)
        self.assertEqual(len(service.get_manifest()["scenes"]["majors"][0]["minors"][0]["backgrounds"]), 2)


# --------------------------------------------------------------------------- #
# §4.5 归一化
# --------------------------------------------------------------------------- #


class ResourceNormalizationTests(_PackCase):
    def packed(self) -> ResourceManifest:
        self.asset("scenes/home/default/night.png")
        self.meta("characters/猫娘/meta.json", {"default_emotion": "正常"})
        self.asset("characters/猫娘/正常.png")
        self.asset("characters/猫娘/开心.png")
        self.asset("characters/猫娘/得意.png")
        self.asset("characters/猫娘/脸红.png")
        service = self.build()
        service.refresh()
        return service

    def test_normalize_emotion_id_resolves_a_fallback_ladder(self) -> None:
        service = self.packed()
        self.assertEqual(service.normalize_emotion_id("happy"), "开心")
        self.assertEqual(service.normalize_emotion_id("joy"), "开心")
        self.assertEqual(service.normalize_emotion_id("正常"), "正常")

    def test_normalize_emotion_id_honours_the_preferred_outfit(self) -> None:
        self.asset("characters/a/normal.png")
        self.asset("characters/b/normal.png")
        self.asset("characters/b/shy.png")
        service = self.build()
        service.refresh()
        self.assertEqual(service.normalize_emotion_id("shy", preferred_outfit="b"), "shy")

    def test_the_ladder_is_consulted_inside_the_preferred_outfit_first(self) -> None:
        # 契约 §4.5：`shy` 的候选阶梯以 `normal` 收尾，所以首选服装里只要有 normal
        # 就会命中它——这是阶梯顺序的可观察结果，不是"回退到别的服装"。
        self.asset("characters/a/normal.png")
        self.asset("characters/b/normal.png")
        self.asset("characters/b/shy.png")
        service = self.build()
        service.refresh()
        self.assertEqual(service.normalize_emotion_id("shy", preferred_outfit="a"), "normal")

    def test_an_emotion_absent_from_the_preferred_outfit_is_found_elsewhere(self) -> None:
        self.asset("characters/a/normal.png")
        self.asset("characters/b/normal.png")
        self.asset("characters/b/稀有.png")
        service = self.build(emotion_aliases={"rare": ["稀有"]})
        service.refresh()
        self.assertEqual(service.normalize_emotion_id("rare", preferred_outfit="a"), "稀有")

    def test_normalize_emotion_id_falls_back_to_the_first_emotion(self) -> None:
        service = self.packed()
        self.assertEqual(service.normalize_emotion_id("完全不存在的表情"), "正常")

    def test_a_custom_alias_ladder_is_honoured(self) -> None:
        self.asset("scenes/home/default/night.png")
        self.asset("characters/default/平静.png")
        service = self.build(emotion_aliases={"calm": ["平静", "正常"]})
        service.refresh()
        self.assertEqual(service.normalize_emotion_id("calm"), "平静")

    def test_empty_alias_entries_are_dropped_at_construction(self) -> None:
        self.asset("scenes/home/default/night.png")
        self.asset("characters/default/normal.png")
        service = self.build(emotion_aliases={"": ["x"], "  ": ["y"], "calm": []})
        self.assertEqual(service.emotion_aliases, {})

    def test_normalize_emotion_output_fills_in_the_outfit_and_keeps_other_keys(self) -> None:
        service = self.packed()
        result = service.normalize_emotion_output(
            {"emotion": "happy", "character": {"outfit": "猫娘"}, "scene": {"major": "home"}}
        )
        self.assertEqual(result["emotion"], "开心")
        self.assertEqual(result["character"], {"outfit": "猫娘"})
        self.assertEqual(result["scene"], {"major": "home"})

    def test_normalize_emotion_output_tolerates_a_missing_character(self) -> None:
        service = self.packed()
        self.assertEqual(service.normalize_emotion_output({"emotion": "happy"})["emotion"], "开心")
        self.assertEqual(service.normalize_emotion_output(None)["emotion"], "正常")  # type: ignore[arg-type]

    def test_normalize_visual_output_rewrites_all_three_groups(self) -> None:
        service = self.packed()
        normalized = service.normalize_visual_output(
            {
                "emotion": "happy",
                "character": {"outfit": "猫娘"},
                "scene": {"major": "home", "minor": "default", "background": "night"},
            }
        )
        self.assertEqual(
            normalized,
            {
                "emotion": "开心",
                "character": {"outfit": "猫娘"},
                "scene": {"major": "home", "minor": "default", "background": "night", "bgm": ""},
            },
        )

    def test_normalize_visual_output_finds_a_background_in_another_minor(self) -> None:
        self.asset("scenes/home/living/a.png")
        self.asset("scenes/home/bedroom/night.png")
        self.asset("characters/default/normal.png")
        service = self.build()
        service.refresh()
        normalized = service.normalize_visual_output(
            {"scene": {"major": "home", "minor": "missing", "background": "night"}}
        )
        self.assertEqual(normalized["scene"]["minor"], "bedroom")
        self.assertEqual(normalized["scene"]["background"], "night")

    def test_normalize_visual_output_falls_back_to_the_declared_defaults(self) -> None:
        self.asset("scenes/home/living/a.png")
        self.asset("characters/default/normal.png")
        service = self.build()
        service.refresh()
        normalized = service.normalize_visual_output({})
        self.assertEqual(normalized["scene"]["major"], "home")
        self.assertEqual(normalized["scene"]["minor"], "living")
        self.assertEqual(normalized["scene"]["background"], "a")
        self.assertEqual(normalized["character"]["outfit"], "default")
        self.assertEqual(normalized["emotion"], "normal")

    def test_unknown_names_fall_back_instead_of_raising(self) -> None:
        self.asset("scenes/home/living/a.png")
        self.asset("characters/default/normal.png")
        service = self.build()
        service.refresh()
        normalized = service.normalize_visual_output(
            {"emotion": "???", "character": {"outfit": "???"}, "scene": {"major": "???", "minor": 0}}
        )
        self.assertEqual(normalized["scene"]["major"], "home")
        self.assertEqual(normalized["character"]["outfit"], "default")
        self.assertEqual(normalized["emotion"], "normal")

    def test_a_track_is_chosen_by_the_background_id_when_none_is_named(self) -> None:
        self.asset("scenes/home/room/evening.png")
        self.asset("bgm/home/room/evening.ogg")
        self.asset("characters/default/normal.png")
        service = self.build()
        service.refresh()
        normalized = service.normalize_visual_output(
            {"scene": {"major": "home", "minor": "room", "background": "evening"}}
        )
        self.assertEqual(normalized["scene"]["bgm"], "evening")

    def test_resolve_visual_bundle_exposes_eight_keys(self) -> None:
        service = self.packed()
        bundle = service.resolve_visual_bundle(
            {
                "emotion": "happy",
                "character": {"outfit": "猫娘"},
                "scene": {"major": "home", "minor": "default", "background": "night"},
            }
        )
        self.assertEqual(
            sorted(bundle),
            sorted(
                ["normalized", "manifest", "major", "minor", "background", "outfit", "emotion", "bgm"]
            ),
        )
        self.assertEqual(bundle["background"]["id"], "night")
        self.assertEqual(bundle["outfit"]["id"], "猫娘")
        self.assertEqual(bundle["emotion"]["id"], "开心")
        self.assertIsNone(bundle["bgm"])

    def test_resolve_visual_bundle_works_on_a_snapshot_of_the_input(self) -> None:
        service = self.packed()
        payload = {"emotion": "happy", "character": {"outfit": "猫娘"}, "scene": {}}
        bundle = service.resolve_visual_bundle(payload)
        self.assertEqual(payload, {"emotion": "happy", "character": {"outfit": "猫娘"}, "scene": {}})
        self.assertIsNot(bundle["normalized"], payload)

    def test_resolve_visual_bundle_refuses_values_that_cannot_be_snapshotted(self) -> None:
        service = self.packed()
        with self.assertRaises(TypeError):
            service.resolve_visual_bundle({"scene": {"major": object()}})

    def test_describe_visual_state_joins_five_labelled_segments(self) -> None:
        service = self.packed()
        text = service.describe_visual_state(
            {
                "emotion": "happy",
                "character": {"outfit": "猫娘"},
                "scene": {"major": "home", "minor": "default", "background": "night"},
            }
        )
        self.assertEqual(
            text,
            "地点: home/default；背景: 深夜；服装: 猫娘；表情: 开心；BGM: 未设置",
        )

    def test_describe_visual_state_falls_back_to_the_requested_names(self) -> None:
        service = self.packed()
        text = service.describe_visual_state({"emotion": "不存在", "character": {}, "scene": {}})
        self.assertIn("表情: ", text)
        self.assertIn("BGM: 未设置", text)

    def test_describe_character_visual_state_lists_the_available_emotions(self) -> None:
        service = self.packed()
        text = service.describe_character_visual_state(
            {"emotion": "happy", "character": {"outfit": "猫娘"}, "scene": {}}
        )
        self.assertTrue(text.startswith("服装: 猫娘；表情: 开心；"))
        self.assertIn("当前服装可用表情: ", text)


# --------------------------------------------------------------------------- #
# §4.5 / §4.7 运行时合成与提示词
# --------------------------------------------------------------------------- #


class ResourceCompositionTests(_PackCase):
    def seeded(self) -> ResourceManifest:
        self.meta("scenes/home/meta.json", {"id": "home", "name": "家"})
        self.asset("scenes/home/room/night.png")
        self.asset("characters/default/normal.png")
        service = self.build()
        service.refresh()
        return service

    def test_extra_outfits_are_merged_and_matched_by_alias(self) -> None:
        service = self.seeded()
        extra = [
            {
                "id": "sailor_uniform",
                "name": "水手服",
                "aliases": ["水手服"],
                "emotions": [{"id": "quiet", "name": "quiet", "path": "/user-assets/demo/quiet.png"}],
            }
        ]
        manifest = service.build_runtime_manifest(extra_character_outfits=extra)
        self.assertEqual([item["id"] for item in manifest["characters"]["outfits"]], ["default", "sailor_uniform"])
        normalized = service.normalize_visual_output(
            {"emotion": "quiet", "character": {"outfit": "水手服"}, "scene": {}},
            extra_character_outfits=extra,
        )
        self.assertEqual(normalized["character"]["outfit"], "sailor_uniform")
        self.assertEqual(normalized["emotion"], "quiet")

    def test_extra_outfits_extend_an_existing_outfit_instead_of_replacing_it(self) -> None:
        service = self.seeded()
        extra = [
            {
                "id": "default",
                "aliases": ["默认"],
                "emotions": [{"id": "happy", "name": "开心", "path": "/user-assets/happy.png"}],
            }
        ]
        manifest = service.build_runtime_manifest(extra_character_outfits=extra)
        outfit = manifest["characters"]["outfits"][0]
        self.assertEqual([item["id"] for item in outfit["emotions"]], ["normal", "happy"])
        self.assertIn("默认", outfit["aliases"])

    def test_extra_scene_groups_are_merged_without_replacing_their_minors(self) -> None:
        service = self.seeded()
        extra = [
            {
                "id": "home",
                "name": "家（扩展）",
                "minors": [{"id": "balcony", "name": "阳台", "backgrounds": []}],
            }
        ]
        manifest = service.build_runtime_manifest(extra_scene_groups=extra)
        major = next(item for item in manifest["scenes"]["majors"] if item["id"] == "home")
        self.assertEqual(major["name"], "家（扩展）")
        self.assertEqual(sorted(item["id"] for item in major["minors"]), ["balcony", "room"])

    def test_extra_tracks_reach_every_minor(self) -> None:
        service = self.seeded()
        extra = [{"id": "theme", "name": "主题曲", "path": "/user-assets/theme.ogg"}]
        manifest = service.build_runtime_manifest(extra_bgm_tracks=extra)
        for major in manifest["scenes"]["majors"]:
            for minor in major["minors"]:
                self.assertEqual([item["id"] for item in minor["bgm_tracks"]], ["theme"])
        self.assertEqual(manifest["defaults"]["bgm"], "theme")

    def test_runtime_additions_never_write_into_the_cache(self) -> None:
        service = self.seeded()
        before = json.dumps(service.get_manifest(), ensure_ascii=False, sort_keys=True)
        service.build_runtime_manifest(
            extra_bgm_tracks=[{"id": "theme", "path": "/x.ogg"}],
            extra_character_outfits=[{"id": "extra", "emotions": []}],
            extra_scene_groups=[{"id": "extra_scene", "minors": []}],
        )
        after = json.dumps(service.get_manifest(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(before, after)

    def test_entries_without_an_id_are_ignored(self) -> None:
        service = self.seeded()
        manifest = service.build_runtime_manifest(
            extra_character_outfits=[{"name": "no id"}, "not a mapping"],
            extra_scene_groups=["not a mapping"],
            extra_bgm_tracks=["not a mapping"],
        )
        self.assertEqual(len(manifest["characters"]["outfits"]), 1)
        self.assertEqual(len(manifest["scenes"]["majors"]), 1)

    def test_prompt_context_lists_scenes_outfits_and_tracks(self) -> None:
        self.meta("scenes/home/meta.json", {"id": "home", "name": "家"})
        self.asset("scenes/home/room/night.png")
        self.asset("characters/default/normal.png")
        self.asset("bgm/home/room/theme.ogg")
        service = self.build()
        service.refresh()
        prompt = service.build_prompt_context()
        self.assertIn("可用场景与背景：", prompt)
        self.assertIn("- 家/room -> 背景: 深夜", prompt)
        self.assertIn("可用服装与表情：", prompt)
        self.assertIn("- default -> 表情: 平静", prompt)
        self.assertIn("可用 BGM：", prompt)
        self.assertIn("- 家/room -> BGM: theme", prompt)
        self.assertIn("场景输出规则：", prompt)

    def test_prompt_context_omits_the_bgm_section_when_there_are_no_tracks(self) -> None:
        service = self.seeded()
        self.assertNotIn("可用 BGM：", service.build_prompt_context())

    def test_prompt_context_includes_root_notes_first(self) -> None:
        service = self.seeded()
        self.text_file("ai.md", "只使用清单内列出的资源。")
        prompt = service.build_prompt_context()
        lines = prompt.splitlines()
        self.assertEqual(lines[0], "资源说明：")
        self.assertEqual(lines[1], "- 只使用清单内列出的资源。")

    def test_a_runtime_scene_group_is_available_to_the_prompt_builder(self) -> None:
        service = self.seeded()
        extra = [
            {
                "id": "空场景",
                "name": "空场景",
                "aliases": [],
                "minors": [
                    {
                        "id": "m",
                        "name": "m",
                        "aliases": [],
                        "description": "",
                        "notes": "",
                        "default_background": "",
                        "default_bgm": "",
                        "backgrounds": [],
                        "bgm_tracks": [],
                    }
                ],
            }
        ]
        names = [item["name"] for item in service.build_runtime_manifest(extra_scene_groups=extra)["scenes"]["majors"]]
        self.assertIn("空场景", names)
        self.assertIn("- 空场景/m -> 背景: (无)", service.build_prompt_context(extra_scene_groups=extra))

    def test_character_prompt_context_opens_with_the_two_fixed_lines(self) -> None:
        service = self.seeded()
        lines = service.build_character_prompt_context().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("桌宠模式只渲染角色服装立绘和表情", lines[0])
        self.assertIn("只在同一套服装下选择可用 emotion", lines[1])
        self.assertEqual(lines[2], "- default -> 表情: 平静")

    def test_emotion_prompt_context_lists_ids_per_outfit(self) -> None:
        service = self.seeded()
        lines = service.build_emotion_prompt_context().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("emotion 与桌宠共用当前角色包的表情图片变量。", lines[0])
        self.assertIn("不要编造或改写 emotion", lines[1])
        self.assertEqual(lines[2], "- default: normal")

    def test_runtime_additions_show_up_in_the_prompt(self) -> None:
        service = self.seeded()
        extra = [{"id": "sailor_uniform", "name": "水手服", "aliases": ["水手服"], "emotions": []}]
        prompt = service.build_prompt_context(extra_character_outfits=extra)
        self.assertIn("水手服", prompt)


# --------------------------------------------------------------------------- #
# 边界：本批新文件与旧项目完全脱钩
# --------------------------------------------------------------------------- #


class C1_3BoundaryTests(unittest.TestCase):
    """B1 边界在新代码上的延续，同时是"表达层独立"的直接证据。"""

    C1_3_FILES = (
        Path(capability_manifest.__file__),
        SHINKU_PKG / "capabilities" / "__init__.py",
        SHINKU_PKG / "capabilities" / "safety.py",
        Path(resource_manifest.__file__),
        SHINKU_PKG / "resources" / "__init__.py",
    )

    #: 旧项目的包名与代号（子串匹配，这些串足够独特）。
    FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane")

    #: 旧项目里这些模块依赖过、本批不得再提及的实现符号。
    FOREIGN_SYMBOLS = ("capability_adapters", "manifest_loader", "desktop_pet_contract")

    #: 旧模块的内部命名（模块级 11 个 + 类内 35 个）。
    #: 用**精确标识符 token** 比对而不是子串——`_text` / `_key` / `_meta` / `_background`
    #: 这类短名字若用子串匹配，会把 `_read_note_text` / `_background_record` 误判为命中。
    LEGACY_INTERNAL_NAMES = (
        # capability_adapters/manifest.py
        "_invalid",
        "_parse_endpoint",
        "_parse_health",
        "_parse_tiers",
        "_parse_secrets",
        "_parse_capabilities",
        "_string_tuple",
        "_risk_and_confirm",
        "_promote_for_effects",
        "_parse_trigger",
        "_parse_io_slots",
        # resource_manifest.py —— 模块级
        "_meta",
        "_text",
        "_key",
        "_string_list",
        "_compact",
        "_merge_text",
        "_ignored",
        "_canon_emotion",
        "_canon_background",
        "_aliases",
        "_entry_matches",
        # resource_manifest.py —— 类内
        "_scan",
        "_scan_scenes",
        "_scan_flat_backgrounds",
        "_scan_outfits",
        "_background",
        "_emotion",
        "_asset_entry",
        "_minor",
        "_legacy_split",
        "_attach_bgm",
        "_audio",
        "_default_outfit",
        "_default_scene",
        "_refresh_defaults",
        "_merge_scene_groups",
        "_merge_outfits",
        "_merge_entries",
        "_sort_minors",
        "_find_major",
        "_find_minor",
        "_find_background",
        "_find_background_across",
        "_find_bgm",
        "_find_outfit",
        "_emotion_candidates",
        "_find_emotion",
        "_label",
        "_scene_label",
        "_directory_note",
        "_directory_note_from_meta",
        "_sidecar_note",
        "_global_notes",
        "_asset_public_path",
        "_normalize_public_prefix",
    )

    #: 旧模块 docstring 里的整句。它们不该以任何形式出现在新文件里。
    LEGACY_PROSE = (
        "Shinku-owned visual resource manifest.",
        "The manifest is intentionally a small filesystem index.",
        "Manifest parser for Shinku capability providers.",
        "The parser accepts a small YAML schema",
        "Import-safe safety constants shared by capability configuration loaders.",
        "Shared safety primitives for capability manifests and local adapters.",
        "Public import path for Shinku's capability manifest loader.",
    )

    def _sources(self) -> list[tuple[Path, str]]:
        for path in self.C1_3_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"expected {path} to exist")
            yield path, path.read_text(encoding="utf-8")

    def test_no_foreign_project_is_mentioned(self) -> None:
        for path, text in self._sources():
            for token in self.FOREIGN_TOKENS:
                with self.subTest(module=path.name, token=token):
                    self.assertNotIn(token, text.lower())

    def test_no_symbol_from_the_old_capability_tree_survives(self) -> None:
        for path, text in self._sources():
            for symbol in self.FOREIGN_SYMBOLS:
                with self.subTest(module=path.name, symbol=symbol):
                    self.assertNotIn(symbol, text)

    def test_no_legacy_internal_name_survives(self) -> None:
        for path, _ in self._sources():
            identifiers = _identifiers(path)
            for legacy in self.LEGACY_INTERNAL_NAMES:
                with self.subTest(module=path.name, legacy=legacy):
                    self.assertNotIn(legacy, identifiers)

    def test_no_legacy_prose_sentence_survives(self) -> None:
        for path, text in self._sources():
            for sentence in self.LEGACY_PROSE:
                with self.subTest(module=path.name, sentence=sentence[:32]):
                    self.assertNotIn(sentence, text)

    def test_no_legacy_environment_prefix(self) -> None:
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("COMPANION_", text)

    def test_the_forwarding_shells_were_dissolved(self) -> None:
        self.assertFalse((SHINKU_PKG / "capabilities" / "manifest_loader.py").exists())
        self.assertFalse((SHINKU_PKG / "capability_adapters").exists())
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("shinku.capabilities.manifest_loader")

    def test_the_safety_constants_live_in_their_own_module(self) -> None:
        # 三个常量在 safety 里定义，manifest 只是引用同一批对象，不另建副本。
        self.assertEqual(safety.LOOPBACK_HOSTS, {"127.0.0.1", "localhost", "::1"})
        self.assertIs(capability_manifest.LOOPBACK_HOSTS, safety.LOOPBACK_HOSTS)
        self.assertIs(capability_manifest.MCP_SAFE_TYPE_RE, safety.MCP_SAFE_TYPE_RE)
        self.assertIs(capability_manifest.MCP_SECRET_MARKERS, safety.MCP_SECRET_MARKERS)

    def test_the_package_markers_export_only_the_documented_names(self) -> None:
        capabilities_pkg = importlib.import_module("shinku.capabilities")
        resources_pkg = importlib.import_module("shinku.resources")
        self.assertEqual(
            set(capabilities_pkg.__all__),
            {
                "ALLOWED_ADAPTER_TYPES",
                "CONFIRM_VALUES",
                "EFFECT_VALUES",
                "LOOPBACK_HOSTS",
                "MCP_SAFE_TYPE_RE",
                "MCP_SECRET_MARKERS",
                "PROVIDER_ID_RE",
                "RISK_VALUES",
                "SCHEMA",
                "SECRET_VALUE_RE",
                "VISIBLE_IN_VALUES",
                "load_manifest",
            },
        )
        self.assertEqual(set(resources_pkg.__all__), set(resource_manifest.__all__))

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        import sys

        for forbidden in ("companion_v01", "code_shared", "akane"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_new_modules_do_not_import_each_other(self) -> None:
        # 能力面与资源面是两条独立的依赖线，任何一方都不该拉进另一方。
        self.assertNotIn(
            "shinku.resources", _imported_modules(Path(capability_manifest.__file__))
        )
        self.assertNotIn(
            "shinku.capabilities", _imported_modules(Path(resource_manifest.__file__))
        )

    def test_the_legacy_private_helpers_of_the_source_are_absent_from_the_whole_package(self) -> None:
        # 本批只新增了 capabilities/ 与 resources/ 两个包；整个包内不得出现旧私有名。
        for package in (SHINKU_PKG / "capabilities", SHINKU_PKG / "resources"):
            for path in sorted(package.glob("*.py")):
                identifiers = _identifiers(path)
                for legacy in self.LEGACY_INTERNAL_NAMES:
                    with self.subTest(module=path.name, legacy=legacy):
                        self.assertNotIn(legacy, identifiers)


class ContractExpressionTests(unittest.TestCase):
    """契约里那些"不会让既有测试变红"的表达式，逐条定点钉住。"""

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _flatten(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def test_manifest_uses_the_or_empty_string_reader_for_name_lists(self) -> None:
        # §3.4：visible_in / effects 用 `str(item or "").strip()`（None/0 → 丢弃）。
        text = self._flatten(self._read(Path(capability_manifest.__file__)))
        self.assertIn('return str(value or "").strip()', text)

    def test_manifest_keeps_the_schema_constant_on_the_way_out(self) -> None:
        text = self._flatten(self._read(Path(capability_manifest.__file__)))
        self.assertIn("schema=SCHEMA,", text)

    def test_manifest_reports_the_provider_type_as_the_detail(self) -> None:
        # §3.6：provider_type_not_allowed 的 detail 是 provider_type 本身。
        text = self._flatten(self._read(Path(capability_manifest.__file__)))
        self.assertIn('_refuse("provider_type_not_allowed", provider_type, provider_id)', text)

    def test_manifest_reports_an_unreadable_host_as_empty(self) -> None:
        text = self._flatten(self._read(Path(capability_manifest.__file__)))
        self.assertIn('_refuse("endpoint_not_loopback", f"host={host or \'<empty>\'}", provider_id)', text)

    def test_resources_use_the_raw_string_reader_for_metadata_lists(self) -> None:
        # §4.4：别名 / allowed_emotions 用 `str(item).strip()`（该口径与上面的不同）。
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn("return [str(item).strip() for item in value if str(item).strip()]", text)

    def test_resource_defaults_point_at_the_first_entry_of_each_list(self) -> None:
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn('"background": minor["backgrounds"][0]["id"]', text)
        self.assertIn('"emotion": outfit["emotions"][0]["id"]', text)

    def test_the_bgm_fallback_bucket_is_the_double_default_key(self) -> None:
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn('("default", "default")', text)

    def test_the_legacy_stem_split_keeps_the_default_bucket_on_miss(self) -> None:
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn("return _DEFAULT_BACKGROUND, stem", text)

    def test_the_prompt_rule_line_is_preserved_verbatim(self) -> None:
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn("场景输出规则：scene.major/minor/background 使用清单列出的名称或 id，", text)
        self.assertIn("不要把未列出的文件名自行拆成新场景。", text)

    def test_the_two_desktop_pet_headers_are_preserved_verbatim(self) -> None:
        text = self._flatten(self._read(Path(resource_manifest.__file__)))
        self.assertIn("桌宠模式只渲染角色服装立绘和表情，不渲染场景、背景或 BGM。", text)
        self.assertIn("emotion 与桌宠共用当前角色包的表情图片变量。", text)
        self.assertIn("不要编造或改写 emotion。", text)


if __name__ == "__main__":
    unittest.main()

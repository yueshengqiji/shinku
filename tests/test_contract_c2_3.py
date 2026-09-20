"""C2-3 契约测试：模型服务配置原语 + 真红侧供应商目录。

对象是 ``shinku.providers.service`` 与 ``shinku.providers.catalog`` 的**外部可观察行为**
（契约文档 ``docs/contracts/c2_3_model_service.md``）。凡是要碰传输层的地方一律换掉：
``requests.get`` 换成记录调用的假件，``build_llm_client`` 换成记录构造参数的假工厂，
不发起任何真实网络请求；注册表读写一律落在临时目录里。

边界类（``C2_3BoundaryTests``）沿用 B1 起每批都有的口径：外来标识 / 旧私有名 /
旧整句 / 旧环境变量前缀 / 旧扁平模块名不得复现。注意本批**合法地** import ``requests``
（探测模型清单要打 HTTP）——C2-1 那条「不碰网络模块」的检查只覆盖 C2-1 的四个模块，
不套到本批头上。
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import sys
import tempfile
import tokenize as py_tokenize
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shinku.providers import catalog as catalog_module
from shinku.providers import service as service_module
from shinku.providers.catalog import (
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_ID_ALIASES,
    PROVIDER_PRESETS,
    ModelServiceConfigStore,
    apply_model_service_settings,
    effective_settings_from_config,
    environment_settings_for_provider,
    infer_provider_id,
    load_and_apply_saved_model_service,
    probe_model_ids,
    provider_presets_payload,
    public_model_services_snapshot,
    public_model_service_snapshot,
    provider_default_models,
    settings_from_mapping,
    #: 真红侧的转发壳。取别名是必须的：原名 ``test_model_service`` 会被 pytest
    #: 当成测试用例收集，而它的参数 ``settings`` 不是一个 fixture。
    test_model_service as catalog_test_model_service,
)
from shinku.providers.service import (
    DEFAULT_TIMEOUT_SECONDS,
    MODEL_SERVICE_SCHEMA_VERSION,
    ModelServiceSettings,
    anthropic_models_endpoint,
    bool_value,
    bounded_int,
    build_model_service_settings,
    build_public_model_service_snapshot,
    effective_model_service_settings,
    model_ids_from_payload,
    model_service_settings_payload,
    normalize_model_ids,
    ollama_tags_endpoint,
    probe_metadata,
    public_provider_entry,
    raise_provider_error,
    redact_provider_error,
    safe_int,
    validate_model_service_settings,
)

SHINKU_PKG = Path(service_module.__file__).resolve().parents[1]

_NOT_JSON = object()

_GLM_BASE = "https://open.bigmodel.cn/api/paas/v4"


# --------------------------------------------------------------------------- #
# 假传输层
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """只实现本模块用到的那几个成员。"""

    def __init__(self, *, status: int = 200, body=_NOT_JSON, text: str = ""):
        self.status_code = status
        self._body = body
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._body is _NOT_JSON:
            raise ValueError("not json")
        return self._body


class _FakeRequests:
    """记录每次 ``get`` 的全部入参，按队列发假响应。"""

    def __init__(self, *responses: _FakeResponse):
        self.queue = list(responses)
        self.calls: list[dict] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.queue.pop(0) if self.queue else _FakeResponse()


class _FakeModels:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def list(self):
        self.calls += 1
        return SimpleNamespace(data=self.data)


class _FakeCompletions:
    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


class _FakeChat:
    def __init__(self, reply):
        self.completions = _FakeCompletions(reply)


class _FakeClient:
    def __init__(self, models_data, reply):
        self.models = _FakeModels(list(models_data))
        self.chat = _FakeChat(reply)


class _FakeClientFactory:
    """替掉 ``build_llm_client``：记录构造参数，返回可控的假客户端。"""

    def __init__(self, *, models=(), reply=None):
        self.models_data = list(models)
        self.reply = reply
        self.calls: list[dict] = []
        self.last: _FakeClient | None = None

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        self.last = _FakeClient(self.models_data, self.reply)
        return self.last


class _Transport:
    """给单个测试换装传输层，收工时还原。"""

    def __init__(self, *responses: _FakeResponse, models=(), reply=None):
        self.requests = _FakeRequests(*responses)
        self.factory = _FakeClientFactory(models=models, reply=reply)

    def __enter__(self):
        self._saved = (service_module.requests, service_module.build_llm_client)
        service_module.requests = self.requests
        service_module.build_llm_client = self.factory
        return self

    def __exit__(self, *exc) -> None:
        service_module.requests, service_module.build_llm_client = self._saved


def _identifiers(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {
            item.string
            for item in py_tokenize.generate_tokens(handle.readline)
            if item.type == py_tokenize.NAME
        }


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _settings(**overrides) -> ModelServiceSettings:
    fields = dict(
        provider_id="glm",
        protocol="openai",
        base_url=_GLM_BASE,
        api_key="k-1",
        chat_model="glm-4.5-air",
        use_for_vision=True,
        vision_model="glm-4v-plus",
        timeout_seconds=90,
    )
    fields.update(overrides)
    return ModelServiceSettings(**fields)


# --------------------------------------------------------------------------- #
# 1. 设置值对象
# --------------------------------------------------------------------------- #


class SettingsValueTests(unittest.TestCase):

    def test_field_order_and_defaults(self) -> None:
        fields = [f.name for f in ModelServiceSettings.__dataclass_fields__.values()]
        self.assertEqual(fields, [
            "provider_id", "protocol", "base_url", "api_key", "chat_model",
            "use_for_vision", "vision_model", "timeout_seconds",
        ])
        bare = ModelServiceSettings(provider_id="glm", protocol="openai",
                                    base_url=_GLM_BASE, api_key="k", chat_model="m")
        self.assertIs(bare.use_for_vision, True)
        self.assertEqual(bare.vision_model, "")
        self.assertEqual(bare.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)

    def test_it_is_frozen(self) -> None:
        settings = _settings()
        with self.assertRaises(Exception):
            settings.api_key = "other"  # type: ignore[misc]

    def test_endpoint_configured_needs_a_base_url(self) -> None:
        self.assertFalse(_settings(base_url="").endpoint_configured)

    def test_ollama_needs_no_key(self) -> None:
        self.assertTrue(_settings(protocol="ollama", api_key="",
                                  base_url="http://127.0.0.1:11434").endpoint_configured)

    def test_other_protocols_need_a_key(self) -> None:
        self.assertFalse(_settings(api_key="").endpoint_configured)
        self.assertTrue(_settings(api_key="k").endpoint_configured)

    def test_configured_also_needs_a_model(self) -> None:
        self.assertFalse(_settings(chat_model="").configured)
        self.assertTrue(_settings(chat_model="m").configured)


# --------------------------------------------------------------------------- #
# 2. 从载荷折设置
# --------------------------------------------------------------------------- #


class BuildSettingsTests(unittest.TestCase):

    def _build(self, raw, **kwargs) -> ModelServiceSettings:
        kwargs.setdefault("presets", PROVIDER_PRESETS)
        kwargs.setdefault("aliases", PROVIDER_ID_ALIASES)
        kwargs.setdefault("default_models", PROVIDER_DEFAULT_MODELS)
        return build_model_service_settings(raw, **kwargs)

    def test_camel_case_and_snake_case_are_both_accepted(self) -> None:
        camel = self._build({"providerId": "glm", "baseUrl": _GLM_BASE, "apiKey": "k",
                             "chatModel": "m", "useForVision": False,
                             "visionModel": "v", "timeoutSeconds": 30})
        snake = self._build({"provider_id": "glm", "base_url": _GLM_BASE, "api_key": "k",
                             "chat_model": "m", "use_for_vision": False,
                             "vision_model": "v", "timeout_seconds": 30})
        self.assertEqual(camel, snake)
        self.assertEqual(camel.use_for_vision, False)
        self.assertEqual(camel.timeout_seconds, 30)

    def test_the_bare_model_key_is_accepted(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k", "model": "m-9"})
        self.assertEqual(settings.chat_model, "m-9")

    def test_the_preset_fills_in_protocol_and_base_url(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k"}, require_model=False)
        self.assertEqual(settings.protocol, "openai")
        self.assertEqual(settings.base_url, _GLM_BASE)

    def test_an_alias_is_canonicalized(self) -> None:
        for alias in ("gml", "zhipu", "智谱AI"):
            with self.subTest(alias=alias):
                settings = self._build({"providerId": alias, "apiKey": "k"},
                                       require_model=False)
                self.assertEqual(settings.provider_id, "glm")

    def test_an_unknown_provider_falls_back_to_the_default_id(self) -> None:
        settings = self._build({"providerId": "nope", "baseUrl": "https://x.test/v1",
                                "apiKey": "k", "chatModel": "m"})
        self.assertEqual(settings.provider_id, "openai_compatible")

    def test_the_default_model_is_applied_when_none_is_given(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k"})
        self.assertEqual(settings.chat_model, PROVIDER_DEFAULT_MODELS["glm"])

    def test_an_absent_key_is_taken_from_the_existing_one(self) -> None:
        settings = self._build({"providerId": "glm", "chatModel": "m"},
                               existing_api_key="old-key")
        self.assertEqual(settings.api_key, "old-key")

    def test_clear_api_key_false_is_a_real_value_and_keeps_the_key(self) -> None:
        # 「键出现且为 False」≠「键不存在」：不能因为假值就滑到下一个键上。
        settings = self._build({"providerId": "glm", "chatModel": "m",
                                "clearApiKey": False}, existing_api_key="old-key")
        self.assertEqual(settings.api_key, "old-key")

    def test_clear_api_key_true_drops_the_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({"providerId": "glm", "chatModel": "m", "clearApiKey": True},
                        existing_api_key="old-key")
        self.assertEqual(str(ctx.exception), "model_service_api_key_missing")

    def test_the_snake_case_clear_flag_works_the_same(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build({"providerId": "glm", "chatModel": "m", "clear_api_key": True},
                        existing_api_key="old-key")
        self.assertEqual(str(ctx.exception), "model_service_api_key_missing")

    def test_timeout_is_clamped(self) -> None:
        for raw, expected in ((1, 5), (9999, 600), ("abc", 120), (0, 5), (600, 600)):
            with self.subTest(raw=raw):
                settings = self._build({"providerId": "glm", "apiKey": "k",
                                        "chatModel": "m", "timeoutSeconds": raw})
                self.assertEqual(settings.timeout_seconds, expected)

    def test_the_timeout_default_comes_from_the_parameter(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k", "chatModel": "m"},
                               default_timeout=42)
        self.assertEqual(settings.timeout_seconds, 42)

    def test_require_model_can_be_relaxed(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k"},
                               default_models={}, require_model=False)
        self.assertEqual(settings.chat_model, "")

    def test_a_full_endpoint_path_is_trimmed_back(self) -> None:
        settings = self._build({"providerId": "glm", "apiKey": "k", "chatModel": "m",
                                "baseUrl": f"{_GLM_BASE}/chat/completions"})
        self.assertEqual(settings.base_url, _GLM_BASE)

    def test_empty_values_do_not_win_over_the_preset(self) -> None:
        # 空字符串是「没填」，不能把预置里的端点顶掉。
        settings = self._build({"providerId": "glm", "baseUrl": "", "apiKey": "k"},
                               require_model=False)
        self.assertEqual(settings.base_url, _GLM_BASE)


class CoalesceAndInheritTests(unittest.TestCase):
    """两条取值链：按「值非空」与按「键是否出现」。"""

    def test_coalesce_takes_the_first_non_empty_value(self) -> None:
        raw = {"a": "", "b": None, "c": "hit", "d": "later"}
        self.assertEqual(service_module._coalesce(raw, "a", "b", "c", "d"), "hit")

    def test_coalesce_falls_back_to_the_default(self) -> None:
        self.assertEqual(service_module._coalesce({}, "a", default="d"), "d")

    def test_inherit_stops_at_a_present_key_even_when_false(self) -> None:
        raw = {"a": False}
        self.assertIs(service_module._inherit(raw, "a", "b"), False)

    def test_inherit_skips_an_absent_key(self) -> None:
        raw = {"b": True}
        self.assertIs(service_module._inherit(raw, "a", "b"), True)

    def test_inherit_without_any_key_gives_none(self) -> None:
        self.assertIsNone(service_module._inherit({}, "a", "b"))


# --------------------------------------------------------------------------- #
# 3. 校验
# --------------------------------------------------------------------------- #


class ValidationTests(unittest.TestCase):

    def _code(self, **overrides) -> str:
        try:
            validate_model_service_settings(_settings(**overrides))
        except ValueError as exc:
            return str(exc)
        self.fail("expected a ValueError")

    def test_the_accepted_protocols(self) -> None:
        for protocol in ("openai", "anthropic", "ollama"):
            with self.subTest(protocol=protocol):
                validate_model_service_settings(_settings(protocol=protocol))

    def test_an_unknown_protocol_is_rejected(self) -> None:
        self.assertEqual(self._code(protocol="bogus"), "model_service_protocol_invalid")

    def test_a_non_http_base_url_is_rejected(self) -> None:
        self.assertEqual(self._code(base_url="/local/path"), "model_service_base_url_invalid")

    def test_a_missing_key_is_rejected_for_everything_but_ollama(self) -> None:
        self.assertEqual(self._code(api_key=""), "model_service_api_key_missing")
        validate_model_service_settings(_settings(protocol="ollama", api_key=""))

    def test_a_missing_model_is_rejected_when_required(self) -> None:
        self.assertEqual(self._code(chat_model=""), "model_service_model_missing")
        validate_model_service_settings(_settings(chat_model=""), require_model=False)

    def test_the_error_order_is_fixed(self) -> None:
        # 协议错 + 端点错 + 密钥空 ⇒ 先报协议。
        self.assertEqual(self._code(protocol="bogus", base_url="x", api_key=""),
                         "model_service_protocol_invalid")


# --------------------------------------------------------------------------- #
# 4. 载荷与快照
# --------------------------------------------------------------------------- #


class PayloadAndSnapshotTests(unittest.TestCase):

    def test_the_payload_key_order_puts_protocol_first(self) -> None:
        self.assertEqual(list(model_service_settings_payload(_settings())), [
            "protocol", "providerId", "baseUrl", "apiKey", "chatModel",
            "useForVision", "visionModel", "timeoutSeconds",
        ])

    def test_the_payload_round_trips_through_the_constructor(self) -> None:
        settings = _settings()
        payload = model_service_settings_payload(settings)
        rebuilt = ModelServiceSettings(
            provider_id=payload["providerId"], protocol=payload["protocol"],
            base_url=payload["baseUrl"], api_key=payload["apiKey"],
            chat_model=payload["chatModel"], use_for_vision=payload["useForVision"],
            vision_model=payload["visionModel"], timeout_seconds=payload["timeoutSeconds"],
        )
        self.assertEqual(rebuilt, settings)

    def test_the_public_snapshot_carries_no_key(self) -> None:
        snapshot = build_public_model_service_snapshot(
            _settings(), source="local_file", providers=[{"id": "glm"}])
        self.assertNotIn("apiKey", snapshot)
        self.assertNotIn("api_key", snapshot)
        self.assertEqual(snapshot["hasApiKey"], True)
        self.assertEqual(snapshot["status"], "configured")
        self.assertEqual(snapshot["loadStatus"], "ok")
        self.assertEqual(snapshot["source"], "local_file")
        self.assertEqual(snapshot["providers"], [{"id": "glm"}])
        self.assertEqual(snapshot["ok"], True)

    def test_an_unconfigured_service_reports_missing_config(self) -> None:
        snapshot = build_public_model_service_snapshot(_settings(chat_model=""), source="env",
                                                       providers=[], load_status="missing")
        self.assertEqual(snapshot["status"], "missing_config")
        self.assertEqual(snapshot["loadStatus"], "missing")

    def test_the_effective_settings_come_from_the_config_module(self) -> None:
        module = SimpleNamespace(CHAT_BASE_URL=_GLM_BASE, CHAT_API_PROTOCOL="auto",
                                 CHAT_API_KEY=" k ", CHAT_MODEL_NAME=" glm-4.6 ",
                                 VISION_MODEL_NAME="glm-4v")
        settings = effective_model_service_settings(
            module, infer_provider_id=infer_provider_id, timeout_seconds=33)
        self.assertEqual(settings.provider_id, "glm")
        self.assertEqual(settings.api_key, "k")
        self.assertEqual(settings.chat_model, "glm-4.6")
        self.assertEqual(settings.use_for_vision, True)
        self.assertEqual(settings.timeout_seconds, 33)

    def test_a_blank_vision_model_turns_vision_off(self) -> None:
        module = SimpleNamespace(CHAT_BASE_URL=_GLM_BASE, CHAT_API_PROTOCOL="openai",
                                 CHAT_API_KEY="k", CHAT_MODEL_NAME="m", VISION_MODEL_NAME="  ")
        settings = effective_model_service_settings(module, infer_provider_id=infer_provider_id)
        self.assertEqual(settings.use_for_vision, False)

    def test_missing_attributes_fall_back_to_empty_strings(self) -> None:
        settings = effective_model_service_settings(SimpleNamespace(),
                                                   infer_provider_id=infer_provider_id)
        self.assertEqual(settings.base_url, "")
        self.assertEqual(settings.protocol, "openai")


# --------------------------------------------------------------------------- #
# 5. 小助手
# --------------------------------------------------------------------------- #


class HelperTests(unittest.TestCase):

    def test_normalize_model_ids_dedupes_and_sorts(self) -> None:
        self.assertEqual(normalize_model_ids(["c", "a", "b", "a", "  ", None]), ["a", "b", "c"])

    def test_normalize_model_ids_rejects_non_sequences(self) -> None:
        for value in (None, "abc", 5, {"a": 1}):
            with self.subTest(value=value):
                self.assertEqual(normalize_model_ids(value), [])

    def test_normalize_model_ids_accepts_sets_and_tuples(self) -> None:
        self.assertEqual(normalize_model_ids(("b", "a")), ["a", "b"])
        self.assertEqual(normalize_model_ids({"b", "a"}), ["a", "b"])

    def test_model_ids_from_payload(self) -> None:
        self.assertEqual(model_ids_from_payload({"data": [{"id": "b"}, {"id": "a"}, "x"]}),
                         ["a", "b"])

    def test_model_ids_from_payload_rejects_a_bad_shape(self) -> None:
        for payload in (None, {}, {"data": "x"}, "text"):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError) as ctx:
                    model_ids_from_payload(payload)
                self.assertEqual(str(ctx.exception), "model_service_models_response_invalid")

    def test_bounded_int_clamps_and_defaults(self) -> None:
        self.assertEqual(bounded_int(3, default=120, minimum=5, maximum=600), 5)
        self.assertEqual(bounded_int(700, default=120, minimum=5, maximum=600), 600)
        self.assertEqual(bounded_int("abc", default=120, minimum=5, maximum=600), 120)
        self.assertEqual(bounded_int(None, default=7, minimum=1, maximum=9), 7)

    def test_bool_value_truthy_words(self) -> None:
        for value in (True, 1, "1", " TRUE ", "yes", "on", "Enabled"):
            with self.subTest(value=value):
                self.assertIs(bool_value(value, False), True)

    def test_bool_value_falsy_words(self) -> None:
        for value in (False, 0, "0", "false", "No", "OFF", "disabled"):
            with self.subTest(value=value):
                self.assertIs(bool_value(value, True), False)

    def test_bool_value_unknown_and_none_give_the_default(self) -> None:
        for value in (None, "maybe", "", 2, "2"):
            with self.subTest(value=value):
                self.assertIs(bool_value(value, True), True)
                self.assertIs(bool_value(value, False), False)

    def test_safe_int(self) -> None:
        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(None), 0)
        self.assertEqual(safe_int("x"), 0)
        self.assertEqual(safe_int(3.0), 3)

    def test_redact_provider_error_masks_the_key(self) -> None:
        text = redact_provider_error(RuntimeError("bad key sk-123456"), api_key="sk-123456")
        self.assertEqual(text, "bad key <redacted>")

    def test_redact_provider_error_truncates(self) -> None:
        text = redact_provider_error(RuntimeError("x" * 5000))
        self.assertEqual(len(text), 600)

    def test_redact_provider_error_falls_back_to_the_class_name(self) -> None:
        self.assertEqual(redact_provider_error(RuntimeError("")), "RuntimeError")
        self.assertEqual(redact_provider_error(RuntimeError("   ")), "RuntimeError")

    def test_raise_provider_error_is_silent_when_ok(self) -> None:
        self.assertIsNone(raise_provider_error(_FakeResponse(status=200)))
        self.assertIsNone(raise_provider_error(_FakeResponse(status=299)))

    def test_raise_provider_error_prefers_the_payload_message(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            raise_provider_error(_FakeResponse(status=401,
                                               body={"error": {"message": " bad key "}}))
        self.assertEqual(str(ctx.exception), "bad key")

    def test_raise_provider_error_stringifies_a_plain_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            raise_provider_error(_FakeResponse(status=400, body={"error": "plain"}))
        self.assertEqual(str(ctx.exception), "plain")

    def test_raise_provider_error_falls_back_to_text(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            raise_provider_error(_FakeResponse(status=503, text="  raw body  "))
        self.assertEqual(str(ctx.exception), "raw body")

    def test_raise_provider_error_gives_a_generic_code(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            raise_provider_error(_FakeResponse(status=502, body={"error": None}))
        self.assertEqual(str(ctx.exception), "model_service_http_502")

    def test_anthropic_models_endpoint(self) -> None:
        self.assertEqual(anthropic_models_endpoint("https://api.anthropic.com/v1"),
                         "https://api.anthropic.com/v1/models")
        self.assertEqual(anthropic_models_endpoint("https://api.anthropic.com"),
                         "https://api.anthropic.com/v1/models")
        self.assertEqual(anthropic_models_endpoint("https://h.test/v1/messages/"),
                         "https://h.test/v1/models")

    def test_ollama_tags_endpoint(self) -> None:
        self.assertEqual(ollama_tags_endpoint("http://127.0.0.1:11434"),
                         "http://127.0.0.1:11434/api/tags")
        self.assertEqual(ollama_tags_endpoint("http://host:11434/v1"),
                         "http://host:11434/api/tags")

    def test_ollama_tags_endpoint_falls_back_to_the_local_default(self) -> None:
        self.assertEqual(ollama_tags_endpoint(""), "http://127.0.0.1:11434/api/tags")

    def test_probe_metadata_reads_both_spellings(self) -> None:
        camel = probe_metadata({"discoveredModels": ["b", "a", "a"],
                                "lastModelProbeAt": "7", "modelProbeError": "boom"})
        self.assertEqual(camel, {"discoveredModels": ["a", "b"],
                                 "lastModelProbeAt": 7, "modelProbeError": "boom"})
        snake = probe_metadata({"discovered_models": ["m"],
                                "last_model_probe_at": -3, "model_probe_error": None})
        self.assertEqual(snake, {"discoveredModels": ["m"],
                                 "lastModelProbeAt": 0, "modelProbeError": ""})

    def test_probe_metadata_truncates_the_error(self) -> None:
        self.assertEqual(len(probe_metadata({"modelProbeError": "e" * 900})["modelProbeError"]),
                         600)


# --------------------------------------------------------------------------- #
# 6. 探测与自检
# --------------------------------------------------------------------------- #


class ProbeTests(unittest.TestCase):

    def test_anthropic_uses_the_native_models_endpoint(self) -> None:
        settings = _settings(protocol="anthropic", base_url="https://api.anthropic.com/v1",
                             api_key="sk-a", timeout_seconds=45)
        response = _FakeResponse(body={"data": [{"id": "claude-3"}, {"id": "claude-2"},
                                                {"id": "claude-3"}]})
        with _Transport(response) as transport:
            ids = service_module.probe_model_ids(settings)
        self.assertEqual(ids, ["claude-2", "claude-3"])
        self.assertEqual(len(transport.requests.calls), 1)
        call = transport.requests.calls[0]
        self.assertEqual(call["url"], "https://api.anthropic.com/v1/models")
        self.assertEqual(call["headers"], {"x-api-key": "sk-a",
                                           "anthropic-version": "2023-06-01"})
        self.assertEqual(call["timeout"], 45)

    def test_anthropic_probe_surfaces_a_provider_error(self) -> None:
        settings = _settings(protocol="anthropic", base_url="https://api.anthropic.com")
        with _Transport(_FakeResponse(status=401, body={"error": {"message": "nope"}})):
            with self.assertRaises(RuntimeError) as ctx:
                service_module.probe_model_ids(settings)
        self.assertEqual(str(ctx.exception), "nope")

    def test_ollama_uses_the_tags_endpoint_when_asked(self) -> None:
        settings = _settings(protocol="ollama", base_url="http://127.0.0.1:11434", api_key="")
        body = {"models": [{"name": "llama3"}, {"name": "qwen"}, "junk"]}
        with _Transport(_FakeResponse(body=body)) as transport:
            ids = service_module.probe_model_ids(settings, use_ollama_tags=True)
        self.assertEqual(ids, ["llama3", "qwen"])
        self.assertEqual(transport.requests.calls[0]["url"],
                         "http://127.0.0.1:11434/api/tags")

    def test_ollama_falls_through_to_the_sdk_without_the_flag(self) -> None:
        settings = _settings(protocol="ollama", base_url="http://127.0.0.1:11434", api_key="")
        with _Transport(models=[SimpleNamespace(id="llama3")]) as transport:
            ids = service_module.probe_model_ids(settings)
        self.assertEqual(ids, ["llama3"])
        self.assertEqual(transport.requests.calls, [])
        self.assertEqual(transport.factory.calls[0]["protocol"], "ollama")

    def test_ollama_tags_with_a_bad_shape_raises(self) -> None:
        settings = _settings(protocol="ollama", base_url="http://127.0.0.1:11434", api_key="")
        with _Transport(_FakeResponse(body={"models": "not-a-list"})):
            with self.assertRaises(RuntimeError) as ctx:
                service_module.probe_model_ids(settings, use_ollama_tags=True)
        self.assertEqual(str(ctx.exception), "model_service_models_response_invalid")

    def test_the_sdk_branch_reads_object_ids(self) -> None:
        with _Transport(models=[SimpleNamespace(id="b"), SimpleNamespace(id="a")]) as transport:
            ids = service_module.probe_model_ids(_settings())
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(transport.factory.calls, [{
            "api_key": "k-1", "base_url": _GLM_BASE, "protocol": "openai",
            "timeout": 90.0, "max_retries": 0,
        }])

    def test_the_sdk_branch_reads_dict_ids(self) -> None:
        with _Transport(models=[{"id": "b"}, {"id": ""}, {"other": 1}]):
            ids = service_module.probe_model_ids(_settings())
        self.assertEqual(ids, ["b"])

    def test_the_sdk_branch_treats_an_empty_listing_as_empty(self) -> None:
        # ``data`` 为 None/空 ⇒ 退化成空列表，不抛异常。
        with _Transport(models=[]):
            ids = service_module.probe_model_ids(_settings())
        self.assertEqual(ids, [])

    def test_probe_validates_the_settings_first(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            service_module.probe_model_ids(_settings(base_url=""))
        self.assertEqual(str(ctx.exception), "model_service_base_url_invalid")

    def test_the_catalog_wrapper_asks_for_ollama_tags(self) -> None:
        settings = _settings(protocol="ollama", base_url="http://127.0.0.1:11434", api_key="")
        with _Transport(_FakeResponse(body={"models": [{"name": "qwen"}]})) as transport:
            ids = probe_model_ids(settings)
        self.assertEqual(ids, ["qwen"])
        self.assertEqual(transport.requests.calls[0]["url"],
                         "http://127.0.0.1:11434/api/tags")


class TestServiceTests(unittest.TestCase):

    def test_the_reply_text_is_returned(self) -> None:
        with _Transport(reply=_reply("  pong  ")) as transport:
            self.assertEqual(catalog_test_model_service(_settings()), "pong")
        call = transport.factory.last.chat.completions.calls[0]
        self.assertEqual(call["model"], "glm-4.5-air")
        self.assertEqual(call["messages"], [{"role": "user", "content": "Reply with only OK."}])
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["max_tokens"], 8)

    def test_an_empty_reply_falls_back_to_ok(self) -> None:
        with _Transport(reply=_reply("")):
            self.assertEqual(catalog_test_model_service(_settings()), "OK")

    def test_a_malformed_reply_raises(self) -> None:
        with _Transport(reply=SimpleNamespace(choices=[])):
            with self.assertRaises(RuntimeError) as ctx:
                catalog_test_model_service(_settings())
        self.assertEqual(str(ctx.exception), "model_service_response_invalid")

    def test_a_reply_without_choices_raises(self) -> None:
        with _Transport(reply=SimpleNamespace()):
            with self.assertRaises(RuntimeError) as ctx:
                catalog_test_model_service(_settings())
        self.assertEqual(str(ctx.exception), "model_service_response_invalid")

    def test_the_client_is_built_without_retries(self) -> None:
        with _Transport(reply=_reply("OK")) as transport:
            catalog_test_model_service(_settings(timeout_seconds=15))
        self.assertEqual(transport.factory.calls[0]["timeout"], 15.0)
        self.assertEqual(transport.factory.calls[0]["max_retries"], 0)

    def test_it_validates_the_settings_first(self) -> None:
        with self.assertRaises(ValueError):
            catalog_test_model_service(_settings(chat_model=""))


# --------------------------------------------------------------------------- #
# 7. 注册表存储
# --------------------------------------------------------------------------- #


class StoreTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "nested" / "model_services.json"
        self.store = ModelServiceConfigStore(self.path)

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_an_absent_file_reads_as_an_empty_registry(self) -> None:
        self.assertEqual(self.store.load_registry(), {
            "schemaVersion": MODEL_SERVICE_SCHEMA_VERSION,
            "activeProviderId": "",
            "providers": {},
        })
        self.assertIsNone(self.store.load())

    def test_a_v1_file_is_upgraded_on_read_only(self) -> None:
        v1 = {"providerId": "glm", "protocol": "openai", "baseUrl": _GLM_BASE,
              "apiKey": "k", "chatModel": "glm-4.5-air", "useForVision": True,
              "visionModel": "glm-4v", "timeoutSeconds": 60}
        self._write(v1)
        registry = self.store.load_registry()
        self.assertEqual(registry["schemaVersion"], 2)
        self.assertEqual(registry["activeProviderId"], "glm")
        self.assertEqual(list(registry["providers"]), ["glm"])
        self.assertEqual(registry["providers"]["glm"]["chatModel"], "glm-4.5-air")
        self.assertEqual(registry["providers"]["glm"]["timeoutSeconds"], 60)
        self.assertEqual(self._read(), v1)  # 磁盘仍是 v1

    def test_a_v1_file_loads_as_the_active_settings(self) -> None:
        self._write({"providerId": "glm", "apiKey": "k", "chatModel": "m",
                     "baseUrl": _GLM_BASE})
        settings = self.store.load()
        self.assertIsNotNone(settings)
        self.assertEqual(settings.provider_id, "glm")
        self.assertEqual(settings.chat_model, "m")

    def test_a_v2_file_keeps_every_provider(self) -> None:
        self._write({"schemaVersion": 2, "activeProviderId": "deepseek", "providers": {
            "glm": {"providerId": "glm", "apiKey": "k1", "chatModel": "m1",
                    "baseUrl": _GLM_BASE},
            "deepseek": {"providerId": "deepseek", "apiKey": "k2", "chatModel": "m2",
                         "baseUrl": "https://api.deepseek.com/v1"},
        }})
        registry = self.store.load_registry()
        self.assertEqual(sorted(registry["providers"]), ["deepseek", "glm"])
        self.assertEqual(registry["activeProviderId"], "deepseek")
        self.assertEqual(self.store.load().provider_id, "deepseek")

    def test_an_unknown_active_id_falls_back_to_the_first_provider(self) -> None:
        self._write({"schemaVersion": 2, "activeProviderId": "gone",
                     "providers": {"glm": {"apiKey": "k", "chatModel": "m",
                                           "baseUrl": _GLM_BASE}}})
        self.assertEqual(self.store.load_registry()["activeProviderId"], "glm")

    def test_the_provider_key_wins_over_the_payload_one(self) -> None:
        self._write({"schemaVersion": 2, "providers": {
            "deepseek": {"providerId": "glm", "apiKey": "k", "chatModel": "m",
                         "baseUrl": "https://api.deepseek.com/v1"}}})
        self.assertEqual(list(self.store.load_registry()["providers"]), ["glm"])

    def test_a_broken_provider_entry_is_skipped(self) -> None:
        self._write({"schemaVersion": 2, "providers": {
            "glm": {"apiKey": "k", "chatModel": "m", "baseUrl": _GLM_BASE},
            "broken": {"apiKey": "k", "chatModel": "m", "baseUrl": "not-a-url"},
        }})
        self.assertEqual(list(self.store.load_registry()["providers"]), ["glm"])

    def test_a_non_object_document_is_rejected(self) -> None:
        self._write([1, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            self.store.load_registry()
        self.assertEqual(str(ctx.exception), "model_service_config_invalid")

    def test_save_writes_v2_and_leaves_no_temp_file(self) -> None:
        self.store.save(_settings(chat_model="glm-4.6"))
        registry = self._read()
        self.assertEqual(registry["schemaVersion"], 2)
        self.assertEqual(registry["activeProviderId"], "glm")
        self.assertEqual(registry["providers"]["glm"]["chatModel"], "glm-4.6")
        self.assertEqual(list(self.path.parent.glob(".*.tmp")), [])

    def test_save_creates_the_parent_directory(self) -> None:
        self.assertFalse(self.path.parent.exists())
        self.store.save(_settings())
        self.assertTrue(self.path.is_file())

    def test_save_provider_without_activation(self) -> None:
        self.store.save_provider(_settings(provider_id="deepseek",
                                           base_url="https://api.deepseek.com/v1",
                                           chat_model="deepseek-chat"))
        registry = self._read()
        self.assertEqual(sorted(registry["providers"]), ["deepseek"])
        self.assertEqual(registry["activeProviderId"], "")

    def test_save_provider_keeps_the_probe_metadata(self) -> None:
        self.store.save_provider(_settings())
        self.store.save_model_probe("glm", ["glm-4.5-air", "glm-4.6"], timestamp=111)
        self.store.save_provider(_settings(chat_model="glm-4.6"))
        registry = self._read()
        self.assertEqual(registry["providers"]["glm"]["discoveredModels"],
                         ["glm-4.5-air", "glm-4.6"])
        self.assertEqual(registry["providers"]["glm"]["lastModelProbeAt"], 111)
        self.assertEqual(registry["providers"]["glm"]["chatModel"], "glm-4.6")

    def test_get_provider(self) -> None:
        self.store.save_provider(_settings(chat_model="m-1"))
        self.store.save_provider(_settings(provider_id="deepseek",
                                           base_url="https://api.deepseek.com/v1",
                                           chat_model="m-2"), set_active=True)
        self.assertEqual(self.store.get_provider("deepseek").chat_model, "m-2")
        self.assertEqual(self.store.get_provider("gml").chat_model, "m-1")  # 别名
        self.assertIsNone(self.store.get_provider("nope"))

    def test_activate_switches_the_model(self) -> None:
        self.store.save_provider(_settings(chat_model="m-1"))
        active = self.store.activate("glm", "glm-4.6")
        self.assertEqual(active.chat_model, "glm-4.6")
        self.assertEqual(self.store.load().chat_model, "glm-4.6")
        self.assertEqual(self._read()["activeProviderId"], "glm")

    def test_activate_without_a_model_keeps_the_current_one(self) -> None:
        self.store.save_provider(_settings(chat_model="m-1"))
        self.assertEqual(self.store.activate("glm").chat_model, "m-1")

    def test_activate_rejects_an_unknown_provider(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.store.activate("nope")
        self.assertEqual(str(ctx.exception), "model_service_provider_missing")

    def test_activate_rejects_a_modelless_provider(self) -> None:
        # ollama 不在默认模型表里，所以「没填模型」能原样存下来。
        self.store.save_provider(ModelServiceSettings(
            provider_id="ollama", protocol="ollama", base_url="http://127.0.0.1:11434",
            api_key="", chat_model=""))
        with self.assertRaises(ValueError) as ctx:
            self.store.activate("ollama")
        self.assertEqual(str(ctx.exception), "model_service_model_missing")

    def test_save_model_probe(self) -> None:
        self.store.save_provider(_settings())
        self.store.save_model_probe("glm", ["b", "a", "a"], timestamp=5, error="boom")
        entry = self._read()["providers"]["glm"]
        self.assertEqual(entry["discoveredModels"], ["a", "b"])
        self.assertEqual(entry["lastModelProbeAt"], 5)
        self.assertEqual(entry["modelProbeError"], "boom")

    def test_save_model_probe_clears_the_error_on_success(self) -> None:
        self.store.save_provider(_settings())
        self.store.save_model_probe("glm", ["a"], timestamp=5, error="boom")
        self.store.save_model_probe("glm", ["a"], timestamp=6)
        self.assertEqual(self._read()["providers"]["glm"]["modelProbeError"], "")

    def test_save_model_probe_rejects_an_unknown_provider(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.store.save_model_probe("nope", [])
        self.assertEqual(str(ctx.exception), "model_service_provider_missing")

    def test_a_negative_timestamp_is_clamped(self) -> None:
        self.store.save_provider(_settings())
        self.store.save_model_probe("glm", [], timestamp=-9)
        self.assertEqual(self._read()["providers"]["glm"]["lastModelProbeAt"], 0)

    def test_canonical_provider_id_goes_through_the_aliases(self) -> None:
        self.assertEqual(self.store.canonical_provider_id("gml"), "glm")
        self.assertEqual(self.store.canonical_provider_id("NEW"), "new")

    def test_load_and_apply_applies_when_something_is_saved(self) -> None:
        self.store.save(_settings(chat_model="m-9"))
        applied: list = []
        result = load_and_apply_saved_model_service(
            store=self.store, config_module=SimpleNamespace(),
            on_error=applied.append)
        self.assertIsNotNone(result)
        self.assertEqual(result.chat_model, "m-9")
        self.assertEqual(applied, [])

    def test_load_and_apply_returns_none_when_nothing_is_saved(self) -> None:
        seen: list = []
        result = load_and_apply_saved_model_service(
            store=self.store, config_module=SimpleNamespace(), on_error=seen.append)
        self.assertIsNone(result)
        self.assertEqual(seen, [])

    def test_load_and_apply_reports_a_read_failure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        seen: list = []
        result = load_and_apply_saved_model_service(
            store=self.store, config_module=SimpleNamespace(), on_error=seen.append)
        self.assertIsNone(result)
        self.assertEqual(len(seen), 1)

    def test_load_and_apply_swallows_a_failure_without_a_handler(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(load_and_apply_saved_model_service(
            store=self.store, config_module=SimpleNamespace()))

    def test_the_generic_store_takes_its_catalog_from_parameters(self) -> None:
        presets = PROVIDER_PRESETS[:1]
        store = service_module.ModelServiceConfigStore(
            self.path, presets=presets, aliases={"z": "glm"}, default_models={})
        store.save_provider(ModelServiceSettings(provider_id="glm", protocol="openai",
                                                base_url=_GLM_BASE, api_key="k",
                                                chat_model="m"))
        self.assertEqual(store.canonical_provider_id("z"), "glm")
        self.assertIsNone(store.get_provider("deepseek"))


# --------------------------------------------------------------------------- #
# 8. 真红侧目录
# --------------------------------------------------------------------------- #


class CatalogTests(unittest.TestCase):

    def test_glm_leads_the_catalog(self) -> None:
        self.assertEqual(PROVIDER_PRESETS[0].id, "glm")
        self.assertEqual(PROVIDER_PRESETS[0].label, "智谱 GLM")
        self.assertEqual(PROVIDER_PRESETS[0].base_url, _GLM_BASE)
        self.assertEqual(len(PROVIDER_PRESETS), 8)

    def test_the_presets_payload_is_marked_as_discoverable(self) -> None:
        payload = provider_presets_payload()
        self.assertEqual(len(payload), len(PROVIDER_PRESETS))
        for row in payload:
            with self.subTest(row=row["id"]):
                self.assertIs(row["supportsModelDiscovery"], True)
                self.assertIn("capabilities", row)
                self.assertNotIn("apiKey", row)

    def test_the_default_models_and_aliases(self) -> None:
        self.assertEqual(PROVIDER_DEFAULT_MODELS["glm"], "glm-4.5-air")
        for alias in ("gml", "智谱", "智谱ai", "智谱glm", "zhipu", "zhipuai"):
            with self.subTest(alias=alias):
                self.assertEqual(PROVIDER_ID_ALIASES[alias], "glm")

    def test_settings_from_mapping_does_not_apply_a_default_model(self) -> None:
        # 本函数用于校验用户提交的表单：没填模型就该空着，由调用方决定要不要补。
        settings = settings_from_mapping({"providerId": "glm", "apiKey": "k"},
                                         require_model=False)
        self.assertEqual(settings.chat_model, "")

    def test_the_store_does_apply_the_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelServiceConfigStore(Path(tmp) / "r.json")
            store.save_provider(ModelServiceSettings(
                provider_id="glm", protocol="openai", base_url=_GLM_BASE,
                api_key="k", chat_model=""))
            self.assertEqual(store.get_provider("glm").chat_model, "glm-4.5-air")

    def test_default_models_can_be_overridden_without_editing_code(self) -> None:
        defaults = provider_default_models(
            {"SHINKU_PROVIDER_DEFAULT_MODELS_JSON": '{"glm":"glm-new","custom":"model-x"}'}
        )
        self.assertEqual(defaults["glm"], "glm-new")
        self.assertEqual(defaults["custom"], "model-x")
        self.assertEqual(provider_default_models({"SHINKU_PROVIDER_DEFAULT_MODELS_JSON": "not-json"})["glm"], "glm-4.5-air")

    def test_custom_provider_catalog_can_be_added_without_editing_code(self) -> None:
        custom = json.dumps(
            [
                {
                    "id": "my_gateway",
                    "label": "我的网关",
                    "protocol": "openai",
                    "baseUrl": "https://gateway.example/v1",
                    "description": "局域网之外的兼容服务",
                    "capabilities": ["stream", "native_tools", "unsupported"],
                },
                {"id": "bad id", "protocol": "openai", "baseUrl": "https://bad.example/v1"},
            ]
        )
        with mock.patch.dict(
            "os.environ",
            {
                "SHINKU_PROVIDER_PRESETS_JSON": custom,
                "SHINKU_PROVIDER_DEFAULT_MODELS_JSON": '{"my_gateway":"gateway-chat"}',
            },
        ):
            payload = catalog_module.provider_presets_payload()
            settings = settings_from_mapping(
                {
                    "providerId": "my_gateway",
                    "baseUrl": "https://gateway.example/v1",
                    "apiKey": "secret",
                    "chatModel": "gateway-chat",
                }
            )
        self.assertIn("my_gateway", {row["id"] for row in payload})
        custom_row = next(row for row in payload if row["id"] == "my_gateway")
        self.assertEqual(custom_row["capabilities"], ["stream", "native_tools"])
        self.assertEqual(settings.provider_id, "my_gateway")
        self.assertNotIn("bad id", {row["id"] for row in payload})

    def test_infer_provider_id_stays_inside_the_catalog(self) -> None:
        self.assertEqual(infer_provider_id(protocol="openai", base_url=_GLM_BASE), "glm")
        self.assertEqual(infer_provider_id(protocol="openai",
                                           base_url="https://api.deepseek.com/v1"), "deepseek")
        self.assertEqual(infer_provider_id(protocol="ollama", base_url="http://h:11434"),
                         "ollama")
        self.assertEqual(infer_provider_id(protocol="openai", base_url="https://x.test"),
                         "openai_compatible")

    def test_effective_settings_from_config(self) -> None:
        module = SimpleNamespace(CHAT_BASE_URL=_GLM_BASE, CHAT_API_PROTOCOL="openai",
                                 CHAT_API_KEY="k", CHAT_MODEL_NAME="m", VISION_MODEL_NAME="")
        settings = effective_settings_from_config(module)
        self.assertEqual(settings.provider_id, "glm")
        self.assertEqual(settings.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)


class CatalogSnapshotTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ModelServiceConfigStore(Path(self._tmp.name) / "r.json")
        self.config = SimpleNamespace(CHAT_BASE_URL=_GLM_BASE, CHAT_API_PROTOCOL="openai",
                                      CHAT_API_KEY="k", CHAT_MODEL_NAME="glm-4.5-air",
                                      VISION_MODEL_NAME="glm-4v-plus")

    def _snapshot(self, **kwargs) -> dict:
        return public_model_services_snapshot(self.store, self.config, **kwargs)

    def test_the_page_shape(self) -> None:
        page = self._snapshot()
        self.assertEqual(page["schemaVersion"], 2)
        self.assertEqual(page["activeProviderId"], "glm")
        self.assertEqual(page["activeModel"], "glm-4.5-air")
        self.assertEqual(page["source"], "local_file")
        self.assertEqual(len(page["providers"]), len(PROVIDER_PRESETS))
        self.assertIs(page["ok"], True)

    def test_no_key_appears_anywhere_on_the_page(self) -> None:
        page = self._snapshot()
        for entry in page["providers"]:
            with self.subTest(entry=entry["id"]):
                self.assertNotIn("apiKey", entry)
                self.assertNotIn("api_key", entry)

    def test_the_active_provider_is_flagged(self) -> None:
        page = self._snapshot()
        active = [entry for entry in page["providers"] if entry["active"]]
        self.assertEqual([entry["id"] for entry in active], ["glm"])

    def test_an_empty_registry_is_filled_from_the_environment(self) -> None:
        page = self._snapshot()
        glm = next(entry for entry in page["providers"] if entry["id"] == "glm")
        self.assertEqual(glm["chatModel"], "glm-4.5-air")
        self.assertIs(glm["hasApiKey"], True)
        self.assertIs(glm["configured"], True)

    def test_a_saved_provider_wins_over_the_environment(self) -> None:
        self.store.save(ModelServiceSettings(
            provider_id="glm", protocol="openai", base_url=_GLM_BASE, api_key="k",
            chat_model="glm-4.6", timeout_seconds=30))
        page = self._snapshot()
        self.assertEqual(page["activeModel"], "glm-4.6")
        glm = next(entry for entry in page["providers"] if entry["id"] == "glm")
        self.assertEqual(glm["chatModel"], "glm-4.6")
        self.assertEqual(glm["timeoutSeconds"], 30)
        self.assertEqual(glm["models"], [])
        self.assertEqual(glm["modelCount"], 0)

    def test_discovered_models_are_exposed(self) -> None:
        self.store.save(_settings())
        self.store.save_model_probe("glm", ["glm-4.6", "glm-4.5-air"], timestamp=9)
        glm = next(entry for entry in self._snapshot()["providers"] if entry["id"] == "glm")
        self.assertEqual(glm["models"], ["glm-4.5-air", "glm-4.6"])
        self.assertEqual(glm["modelCount"], 2)
        self.assertEqual(glm["lastModelProbeAt"], 9)

    def test_a_provider_outside_the_catalog_is_appended(self) -> None:
        # 把目录缩到只剩 GLM，注册表里的 deepseek 就成了「目录之外的条目」。
        self.store.save(ModelServiceSettings(
            provider_id="deepseek", protocol="openai", base_url="https://api.deepseek.com/v1",
            api_key="k", chat_model="deepseek-chat"))
        with mock.patch.object(catalog_module, "PROVIDER_PRESETS", PROVIDER_PRESETS[:1]):
            page = public_model_services_snapshot(self.store, self.config)
        extra = [entry for entry in page["providers"] if entry["id"] == "openai_compatible"]
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]["description"], "已保存的自定义 OpenAI 兼容服务。")

    def test_the_single_snapshot_carries_the_catalog(self) -> None:
        snapshot = public_model_service_snapshot(_settings(), source="env")
        self.assertEqual(len(snapshot["providers"]), len(PROVIDER_PRESETS))
        self.assertEqual(snapshot["source"], "env")


class PublicProviderEntryTests(unittest.TestCase):

    def test_the_key_set_is_fixed(self) -> None:
        entry = public_provider_entry(preset=PROVIDER_PRESETS[0], settings=_settings(),
                                      metadata=probe_metadata({}), active=True)
        self.assertEqual(sorted(entry), sorted([
            "id", "label", "protocol", "baseUrl", "description", "apiKeyRequired",
            "configured", "endpointConfigured", "active", "hasApiKey", "chatModel",
            "useForVision", "visionModel", "timeoutSeconds", "models", "modelCount",
            "lastModelProbeAt", "modelProbeError", "supportsModelDiscovery", "capabilities",
        ]))
        self.assertIs(entry["supportsModelDiscovery"], True)

    def test_a_catalog_entry_without_settings(self) -> None:
        entry = public_provider_entry(preset=PROVIDER_PRESETS[0], settings=None,
                                      metadata=probe_metadata({}), active=False)
        self.assertEqual(entry["id"], "glm")
        self.assertEqual(entry["label"], "智谱 GLM")
        self.assertEqual(entry["baseUrl"], _GLM_BASE)
        self.assertIs(entry["configured"], False)
        self.assertIs(entry["hasApiKey"], False)
        self.assertEqual(entry["chatModel"], "")
        self.assertEqual(entry["timeoutSeconds"], DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(list(entry["capabilities"]), ["stream", "vision", "native_tools"])

    def test_a_saved_entry_without_a_preset(self) -> None:
        entry = public_provider_entry(preset=None, settings=_settings(),
                                      metadata=probe_metadata({}), active=True)
        self.assertEqual(entry["description"], "已保存的自定义 OpenAI 兼容服务。")
        self.assertIs(entry["apiKeyRequired"], True)
        self.assertEqual(entry["label"], "glm")

    def test_both_missing_gives_neutral_defaults(self) -> None:
        entry = public_provider_entry(preset=None, settings=None,
                                      metadata=probe_metadata({}), active=False)
        self.assertEqual(entry["id"], "openai_compatible")
        self.assertEqual(entry["protocol"], "openai")
        self.assertEqual(entry["baseUrl"], "")
        self.assertEqual(entry["capabilities"], ["stream", "vision", "native_tools"])


# --------------------------------------------------------------------------- #
# 9. 「从环境变量认一遍」与「写回项目配置」
# --------------------------------------------------------------------------- #


class EnvironmentLookupTests(unittest.TestCase):

    def _config(self, **chat) -> SimpleNamespace:
        fields = dict(CHAT_BASE_URL="https://api.openai.com/v1", CHAT_API_PROTOCOL="openai",
                      CHAT_API_KEY="chat-key", CHAT_MODEL_NAME="gpt-4o-mini",
                      VISION_MODEL_NAME="")
        fields.update(chat)
        return SimpleNamespace(**fields)

    def test_the_effective_channel_wins(self) -> None:
        config = self._config(CHAT_BASE_URL=_GLM_BASE, CHAT_MODEL_NAME="glm-4.5-air")
        settings = environment_settings_for_provider("glm", config)
        self.assertIsNotNone(settings)
        self.assertEqual(settings.chat_model, "glm-4.5-air")

    def test_the_effective_channel_must_be_configured(self) -> None:
        config = self._config(CHAT_BASE_URL=_GLM_BASE, CHAT_MODEL_NAME="")
        self.assertIsNone(environment_settings_for_provider("glm", config))

    def test_provider_specific_environment_fields(self) -> None:
        config = self._config()
        config.settings = SimpleNamespace(GLM_API_KEY="glm-key", GLM_BASE_URL=_GLM_BASE,
                                          GLM_MODEL_NAME="glm-4.6")
        settings = environment_settings_for_provider("glm", config)
        self.assertEqual(settings.api_key, "glm-key")
        self.assertEqual(settings.chat_model, "glm-4.6")

    def test_the_default_model_fills_a_blank_specific_field(self) -> None:
        config = self._config()
        config.settings = SimpleNamespace(GLM_API_KEY="glm-key", GLM_BASE_URL=_GLM_BASE,
                                          GLM_MODEL_NAME="")
        settings = environment_settings_for_provider("glm", config)
        self.assertEqual(settings.chat_model, "glm-4.5-air")

    def test_the_channel_fallback(self) -> None:
        config = self._config()
        config.settings = SimpleNamespace(
            TEXT_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai",
            TEXT_API_KEY="gem-key", TEXT_MODEL_NAME="gemini-2.5-flash",
            TEXT_API_PROTOCOL="openai")
        settings = environment_settings_for_provider("gemini", config)
        self.assertEqual(settings.provider_id, "gemini")
        self.assertEqual(settings.chat_model, "gemini-2.5-flash")
        self.assertEqual(settings.api_key, "gem-key")

    def test_an_unconfigured_channel_is_skipped(self) -> None:
        config = self._config()
        config.settings = SimpleNamespace(
            TEXT_BASE_URL="https://api.deepseek.com/v1", TEXT_API_KEY="", TEXT_MODEL_NAME="")
        self.assertIsNone(environment_settings_for_provider("deepseek", config))

    def test_nothing_found_gives_none(self) -> None:
        self.assertIsNone(environment_settings_for_provider("gemini", self._config()))

    def test_no_settings_object_gives_none(self) -> None:
        self.assertIsNone(environment_settings_for_provider("gemini", self._config()))

    def test_the_alias_is_canonicalized(self) -> None:
        config = self._config(CHAT_BASE_URL=_GLM_BASE, CHAT_MODEL_NAME="glm-4.5-air")
        self.assertIsNotNone(environment_settings_for_provider("gml", config))


class ApplySettingsTests(unittest.TestCase):

    def _config(self, **extra) -> SimpleNamespace:
        fields = dict(TEXT_API_KEY="old", TEXT_BASE_URL="old", TEXT_MODEL_NAME="old",
                      TEXT_API_PROTOCOL="old", AUX_API_KEY="old", AUX_BASE_URL="old",
                      AUX_MODEL_NAME="old", AUX_API_PROTOCOL="old", CHAT_API_KEY="old",
                      CHAT_BASE_URL="old", CHAT_MODEL_NAME="old", CHAT_API_PROTOCOL="old",
                      VISION_API_KEY="old", VISION_BASE_URL="old", VISION_MODEL_NAME="old",
                      VISION_API_PROTOCOL="old",
                      ENABLE_NATIVE_TOOL_DECISION=True,
                      NATIVE_TOOL_PROVIDER_ALLOWLIST="old",
                      MODEL_PRESETS={"glm": {"native_tools": True,
                                             "native_tool_allowlist": "{host}/{model}"}})
        fields.update(extra)
        return SimpleNamespace(**fields)

    def test_the_three_text_channels_are_rewritten(self) -> None:
        config = self._config()
        apply_model_service_settings(config, _settings())
        for prefix in ("TEXT", "AUX", "CHAT"):
            with self.subTest(prefix=prefix):
                self.assertEqual(getattr(config, f"{prefix}_API_KEY"), "k-1")
                self.assertEqual(getattr(config, f"{prefix}_BASE_URL"), _GLM_BASE)
                self.assertEqual(getattr(config, f"{prefix}_MODEL_NAME"), "glm-4.5-air")
                self.assertEqual(getattr(config, f"{prefix}_API_PROTOCOL"), "openai")

    def test_a_matching_vision_model_is_applied(self) -> None:
        config = self._config()
        apply_model_service_settings(config, _settings(vision_model="glm-4v-plus"))
        self.assertEqual(config.VISION_API_KEY, "k-1")
        self.assertEqual(config.VISION_BASE_URL, _GLM_BASE)
        self.assertEqual(config.VISION_MODEL_NAME, "glm-4v-plus")

    def test_a_blank_vision_model_falls_back_to_the_chat_one(self) -> None:
        config = self._config()
        apply_model_service_settings(config, _settings(vision_model=""))
        self.assertEqual(config.VISION_MODEL_NAME, "glm-4.5-air")

    def test_a_foreign_vision_model_is_preserved(self) -> None:
        # 聊到切到 GLM 而 .env 里的视觉服务是 Gemini 兼容时最容易踩到：
        # 不能把别家的模型名丢给视觉端点，保留已载入的专用视觉配置。
        config = self._config()
        with self.assertLogs("shinku.model_service", level="WARNING") as captured:
            apply_model_service_settings(config, _settings(vision_model="gemini-2.5-flash"))
        self.assertEqual(config.VISION_MODEL_NAME, "old")
        self.assertEqual(config.VISION_BASE_URL, "old")
        self.assertIn("Ignoring incompatible vision model", captured.output[0])

    def test_vision_off_leaves_the_vision_channel_alone(self) -> None:
        config = self._config()
        apply_model_service_settings(config, _settings(use_for_vision=False))
        self.assertEqual(config.VISION_MODEL_NAME, "old")

    def test_an_unknown_provider_accepts_any_vision_model(self) -> None:
        # 自定义/本地服务本来就可能代理多家，前缀检查只对已知供应商做。
        config = self._config()
        apply_model_service_settings(
            config, ModelServiceSettings(provider_id="my-gateway", protocol="openai",
                                         base_url="https://gw.test/v1", api_key="k",
                                         chat_model="whatever", vision_model="whatever-v"))
        self.assertEqual(config.VISION_MODEL_NAME, "whatever-v")

    def test_native_tools_are_enabled_from_the_preset(self) -> None:
        config = self._config()
        apply_model_service_settings(config, _settings())
        self.assertIs(config.ENABLE_NATIVE_TOOL_DECISION, True)
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST,
                         "open.bigmodel.cn/glm-4.5-air")

    def test_native_tools_are_disabled_when_the_preset_says_so(self) -> None:
        config = self._config(MODEL_PRESETS={"glm": {"native_tools": False}})
        apply_model_service_settings(config, _settings())
        self.assertIs(config.ENABLE_NATIVE_TOOL_DECISION, False)
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST, "")

    def test_a_missing_preset_disables_native_tools(self) -> None:
        config = self._config(MODEL_PRESETS={"other": {"native_tools": True}})
        apply_model_service_settings(config, _settings())
        self.assertIs(config.ENABLE_NATIVE_TOOL_DECISION, False)
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST, "")

    def test_the_allowlist_falls_back_to_host_and_model(self) -> None:
        # 预置没给模板 ⇒ 退回「主机:模型」。
        config = self._config(MODEL_PRESETS={"glm": {"native_tools": True}})
        apply_model_service_settings(config, _settings())
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST,
                         "open.bigmodel.cn:glm-4.5-air")

    def test_the_whole_policy_is_skipped_without_the_switch(self) -> None:
        config = self._config()
        del config.ENABLE_NATIVE_TOOL_DECISION
        apply_model_service_settings(config, _settings())
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST, "old")

    def test_the_allowlist_is_optional(self) -> None:
        config = self._config()
        del config.NATIVE_TOOL_PROVIDER_ALLOWLIST
        apply_model_service_settings(config, _settings())
        self.assertFalse(hasattr(config, "NATIVE_TOOL_PROVIDER_ALLOWLIST"))


class LoadAndApplyTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ModelServiceConfigStore(Path(self._tmp.name) / "r.json")

    def test_the_catalog_wrapper_injects_the_project_apply(self) -> None:
        self.store.save(_settings(chat_model="glm-4.6"))
        config = SimpleNamespace(CHAT_MODEL_NAME="old", TEXT_MODEL_NAME="old")
        result = load_and_apply_saved_model_service(store=self.store, config_module=config)
        self.assertIsNotNone(result)
        self.assertEqual(config.CHAT_MODEL_NAME, "glm-4.6")
        self.assertEqual(config.TEXT_MODEL_NAME, "glm-4.6")


# --------------------------------------------------------------------------- #
# 10. 边界（B1 起每批都有的口径）
# --------------------------------------------------------------------------- #


class C2_3BoundaryTests(unittest.TestCase):
    """「表达层独立」的直接证据 + 本批的有意分叉被钉住。"""

    C2_3_FILES = (
        SHINKU_PKG / "providers" / "service.py",
        SHINKU_PKG / "providers" / "catalog.py",
    )

    FOREIGN_TOKENS = ("companion_v01", "code_shared")

    #: 旧私有名并集（对照物 ∪ Shinku 旧实现，9 个）。
    LEGACY_INTERNAL_NAMES = (
        "_apply_native_tool_capability",     # 旧侧 model_service_config
        "_base_url_matches_provider",        # 旧侧 model_service_config
        "_bool_value",                       # 旧侧 model_service
        "_bounded_int",                      # 旧侧 model_service
        "_canonical_provider_id",            # 旧侧 model_service_config
        "_public_provider_entry",            # 旧侧 model_service_config
        "_safe_int",                         # 旧侧 model_service
        "_save_registry",                    # 上游 model_service
        "_vision_model_matches_provider",    # 旧侧 model_service_config
    )

    #: 34 句 = 上游 `model_service.py` ∪ 旧侧 `model_service.py` 1 句
    #: ∪ legacy 侧 1 句 ∪ 旧侧 `model_service_config.py` 15 句。
    LEGACY_PROSE = (
        "legacy 侧兼容入口：目录留在项目侧，注册表实现（含 v1→v2 读迁移）只有共享一份。",
        "Discover model ids using the provider's standard endpoint.",
        "Guard against a saved cross-provider vision model.",
        "Hot switching rewrites module globals, so inspecting only CHAT_* can make a",
        "Keep native tool routing in sync with the selected provider/model.",
        "Normalize a project payload without importing project code.",
        "Perform the smallest provider-neutral connectivity check.",
        "Provider-neutral model service configuration primitives.",
        "Read the active chat channel without coupling to a project config type.",
        "Resolve a provider from the unmutated Settings object when available.",
        "Return the complete provider list without exposing API keys.",
        "Shinku model-service value objects, probing, and provider registry.",
        "The companion projects own their provider catalogs, persistence format and",
        "Whether the endpoint can be queried before a model is chosen.",
        "Whether the service is ready for a normal chat request.",
        "``presets`` and ``aliases`` are supplied by the project so GLM and other",
        "``settings`` object still contains the values loaded from .env and is the",
        "``visionModel`` is stored alongside the chat provider, but older control",
        "`default_models`（各自默认模型）。",
        "active provider changed to GLM.  OpenAI-compatible custom and local",
        "and validation behavior stay identical.",
        "center versions allowed values such as ``gemini-*`` to remain when the",
        "configured but inactive provider look absent.  The original pydantic",
        "conservative prefix check so a stale id falls back to the dedicated env",
        "correct source for this catalog view.",
        "project-specific catalog entries remain local while the field vocabulary",
        "providers intentionally accept arbitrary model ids; known providers get a",
        "runtime policy.  This module owns only the shared settings value object and",
        "the normalization/validation rules used at that boundary.",
        "vision service instead of producing a misleading model-not-found error.",
        "供应商注册表存储（唯一实现，两个项目共用）。",
        "模型探测元数据：两个项目的目录视图用同一份解析。",
        "真红侧注册表入口：目录、别名与默认模型留在项目侧。",
        "读注册表并应用；\"怎么应用到项目配置\"由调用方注入（项目策略留在项目侧）。",
    )

    def _sources(self):
        for path in self.C2_3_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"expected {path} to exist")
            yield path, path.read_text(encoding="utf-8")

    def test_no_foreign_project_is_mentioned(self) -> None:
        for path, text in self._sources():
            for token in self.FOREIGN_TOKENS:
                with self.subTest(module=path.name, token=token):
                    self.assertNotIn(token, text.lower())

    def test_no_legacy_internal_name_survives(self) -> None:
        for path, _ in self._sources():
            identifiers = _identifiers(path)
            for legacy in self.LEGACY_INTERNAL_NAMES:
                with self.subTest(module=path.name, legacy=legacy):
                    self.assertNotIn(legacy, identifiers)

    def test_the_legacy_name_list_covers_the_whole_union(self) -> None:
        self.assertEqual(len(self.LEGACY_INTERNAL_NAMES), 9)
        self.assertEqual(len(set(self.LEGACY_INTERNAL_NAMES)),
                         len(self.LEGACY_INTERNAL_NAMES))

    def test_no_legacy_prose_sentence_survives(self) -> None:
        for path, text in self._sources():
            for sentence in self.LEGACY_PROSE:
                with self.subTest(module=path.name, sentence=sentence[:24]):
                    self.assertNotIn(sentence, text)

    def test_the_legacy_prose_list_has_34_entries(self) -> None:
        self.assertEqual(len(self.LEGACY_PROSE), 34)
        self.assertEqual(len(set(self.LEGACY_PROSE)), 34)

    def test_no_legacy_environment_prefix(self) -> None:
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("COMPANION_", text)
                self.assertNotIn("SHINKU_SERVER_", text)

    def test_the_legacy_flat_module_names_were_not_recreated(self) -> None:
        for name in ("model_service", "model_service_config"):
            with self.subTest(name=name):
                self.assertFalse((SHINKU_PKG / "providers" / f"{name}.py").exists())
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"shinku.providers.{name}")

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        for forbidden in ("companion_v01", "code_shared"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_only_network_import_is_requests(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "providers" / "service.py")
        self.assertIn("requests", modules)
        for forbidden in ("socket", "subprocess", "urllib", "http"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, modules)

    def test_the_catalog_imports_no_io_modules(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "providers" / "catalog.py")
        self.assertIn("urllib.parse", modules)  # urlsplit 是纯解析，不做 I/O
        for forbidden in ("socket", "subprocess", "requests", "http"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, modules)

    def test_both_modules_reach_for_the_provider_vocabulary(self) -> None:
        for path in self.C2_3_FILES:
            with self.subTest(module=path.name):
                self.assertIn(".config", _imported_modules(path))

    def test_no_project_config_type_is_imported(self) -> None:
        # 「怎么应用到项目配置」是注入的，两个模块都不 import 任何配置类型。
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("from ..config", text)
                self.assertNotIn("import config", text)

    def test_the_module_surfaces_match_the_contract(self) -> None:
        self.assertEqual(set(service_module.__all__), {
            "DEFAULT_PROVIDER_ID", "DEFAULT_TIMEOUT_SECONDS", "MODEL_SERVICE_SCHEMA_VERSION",
            "ModelServiceConfigStore", "ModelServiceSettings", "anthropic_models_endpoint",
            "bool_value", "bounded_int", "build_model_service_settings",
            "build_public_model_service_snapshot", "effective_model_service_settings",
            "load_and_apply_saved_model_service", "model_ids_from_payload",
            "model_service_settings_payload", "normalize_model_ids", "ollama_tags_endpoint",
            "probe_metadata", "probe_model_ids", "public_provider_entry",
            "raise_provider_error", "redact_provider_error", "safe_int",
            "test_model_service", "validate_model_service_settings",
        })
        self.assertEqual(set(catalog_module.__all__), {
            "MODEL_SERVICE_SCHEMA_VERSION", "PROVIDER_DEFAULT_MODELS", "provider_default_models",
            "PROVIDER_ID_ALIASES", "PROVIDER_PRESETS", "PRESET_BY_ID",
            "ModelProviderPreset", "ModelServiceConfigStore",
            "apply_model_service_settings", "effective_settings_from_config",
            "environment_settings_for_provider", "infer_provider_id",
            "load_and_apply_saved_model_service", "probe_model_ids",
            "provider_presets_payload", "public_model_service_snapshot",
            "public_model_services_snapshot", "settings_from_mapping", "test_model_service",
        })


class C2_3ContractLiteralTests(unittest.TestCase):
    """契约里「改坏了也不会让别处变红」的字面量，逐条钉住。"""

    def test_the_timeout_default(self) -> None:
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 120)
        self.assertEqual(MODEL_SERVICE_SCHEMA_VERSION, 2)
        self.assertEqual(catalog_module.MODEL_SERVICE_SCHEMA_VERSION, 2)

    def test_the_timeout_bounds(self) -> None:
        self.assertEqual(service_module._TIMEOUT_FLOOR, 5)
        self.assertEqual(service_module._TIMEOUT_CEILING, 600)

    def test_the_error_detail_limit(self) -> None:
        self.assertEqual(service_module._ERROR_DETAIL_LIMIT, 600)

    def test_the_ollama_fallback_base(self) -> None:
        self.assertEqual(service_module._OLLAMA_FALLBACK_BASE, "http://127.0.0.1:11434")

    def test_the_anthropic_version_header(self) -> None:
        self.assertEqual(service_module._ANTHROPIC_VERSION_HEADER, "2023-06-01")

    def test_the_error_codes_are_stable_strings(self) -> None:
        codes = []
        for overrides in ({"protocol": "bogus"}, {"base_url": "x"},
                          {"api_key": ""}, {"chat_model": ""}):
            with self.assertRaises(ValueError) as ctx:
                validate_model_service_settings(_settings(**overrides))
            codes.append(str(ctx.exception))
        self.assertEqual(codes, [
            "model_service_protocol_invalid",
            "model_service_base_url_invalid",
            "model_service_api_key_missing",
            "model_service_model_missing",
        ])

    def test_the_default_provider_id_is_re_exported(self) -> None:
        self.assertEqual(service_module.DEFAULT_PROVIDER_ID, "openai_compatible")

    def test_the_settings_can_be_replaced_field_by_field(self) -> None:
        # 冻结但可 replace：`activate()` 换模型就是靠它。
        replaced = dataclasses.replace(_settings(), chat_model="other")
        self.assertEqual(replaced.chat_model, "other")
        self.assertEqual(_settings().chat_model, "glm-4.5-air")


if __name__ == "__main__":
    unittest.main()

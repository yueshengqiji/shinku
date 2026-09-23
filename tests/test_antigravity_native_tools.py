from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from shinku.llm.runtime import LLMRuntime
from shinku.providers.catalog import apply_model_service_settings
from shinku.providers.service import ModelServiceSettings


class _FakeClient:
    def __init__(self, *, protocol: str, base_url: str) -> None:
        self._shinku_protocol = protocol
        self.base_url = base_url


def _runtime(*, base_url: str, model: str) -> LLMRuntime:
    def build_client(**kwargs):
        return _FakeClient(protocol=kwargs["protocol"], base_url=kwargs["base_url"])

    env = {
        "SHINKU_AUX_MODEL_NAME": "aux-model",
        "SHINKU_CHAT_MODEL_NAME": model,
        "SHINKU_CHAT_BASE_URL": base_url,
        "SHINKU_CHAT_API_PROTOCOL": "openai",
        "SHINKU_AUX_BASE_URL": base_url,
        "SHINKU_AUX_API_PROTOCOL": "openai",
    }
    with mock.patch("shinku.llm.runtime_core.build_llm_client", side_effect=build_client):
        with mock.patch.dict(os.environ, env):
            return LLMRuntime()


class AntigravityProviderPolicyTests(unittest.TestCase):
    def test_custom_provider_capability_enables_a_model_and_instance_scoped_allowlist(self) -> None:
        config = SimpleNamespace(
            ENABLE_NATIVE_TOOL_DECISION=False,
            NATIVE_TOOL_PROVIDER_ALLOWLIST="",
        )
        settings = ModelServiceSettings(
            provider_id="antigravity-local",
            protocol="openai",
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-only",
            chat_model="gemini-3.8-flash-high",
            use_for_vision=False,
        )
        preset = {
            "id": "antigravity-local",
            "label": "Antigravity local",
            "protocol": "openai",
            "baseUrl": "http://127.0.0.1:8317/v1",
            "apiKeyRequired": True,
            "capabilities": ["stream", "native_tools"],
        }
        with mock.patch.dict(
            os.environ,
            {"SHINKU_PROVIDER_PRESETS_JSON": json.dumps([preset])},
        ):
            apply_model_service_settings(config, settings)

        self.assertTrue(config.ENABLE_NATIVE_TOOL_DECISION)
        self.assertEqual(
            config.NATIVE_TOOL_PROVIDER_ALLOWLIST,
            "127.0.0.1:8317:gemini-3.8-flash-high",
        )

    def test_custom_provider_without_native_tools_stays_disabled(self) -> None:
        config = SimpleNamespace(
            ENABLE_NATIVE_TOOL_DECISION=True,
            NATIVE_TOOL_PROVIDER_ALLOWLIST="stale-entry",
        )
        settings = ModelServiceSettings(
            provider_id="antigravity-local",
            protocol="openai",
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-only",
            chat_model="gemini-3.8-flash-high",
            use_for_vision=False,
        )
        preset = {
            "id": "antigravity-local",
            "protocol": "openai",
            "baseUrl": "http://127.0.0.1:8317/v1",
            "capabilities": ["stream"],
        }
        with mock.patch.dict(
            os.environ,
            {"SHINKU_PROVIDER_PRESETS_JSON": json.dumps([preset])},
        ):
            apply_model_service_settings(config, settings)

        self.assertFalse(config.ENABLE_NATIVE_TOOL_DECISION)
        self.assertEqual(config.NATIVE_TOOL_PROVIDER_ALLOWLIST, "")

    def test_native_tools_match_the_exact_local_proxy_port_and_model(self) -> None:
        runtime = _runtime(
            base_url="http://127.0.0.1:8317/v1",
            model="gemini-3.8-flash-high",
        )
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST": (
                    "127.0.0.1:8317:gemini-3.8-flash-high"
                ),
                "SHINKU_NATIVE_TOOL_ALLOW_UNKNOWN_MODEL": "",
            },
        ):
            profile = runtime._tool_profile(runtime.chat)
            payload = runtime._build_payload(
                bundle=runtime.chat,
                system_prompt="Use available tools when appropriate.",
                user_prompt="Run the harmless probe.",
                temperature=0.0,
                json_mode=True,
                native_tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "probe_noop",
                            "description": "No external side effects.",
                            "parameters": {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                            },
                        },
                    }
                ],
                native_tool_choice="required",
            )

        self.assertTrue(profile.supports_native_tools)
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertNotIn("response_format", payload)

    def test_a_different_local_proxy_port_does_not_inherit_the_allowlist(self) -> None:
        runtime = _runtime(
            base_url="http://127.0.0.1:8318/v1",
            model="gemini-3.8-flash-high",
        )
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST": (
                    "127.0.0.1:8317:gemini-3.8-flash-high"
                ),
                "SHINKU_NATIVE_TOOL_ALLOW_UNKNOWN_MODEL": "",
            },
        ):
            profile = runtime._tool_profile(runtime.chat)
            self.assertFalse(
                profile.supports_native_tools,
                f"host={runtime._bundle_host(runtime.chat)!r}; model={runtime.chat.model!r}; notes={profile.notes!r}",
            )

    def test_legacy_host_only_allowlist_still_matches(self) -> None:
        runtime = _runtime(
            base_url="http://gateway.example:8443/v1",
            model="custom-model",
        )
        with mock.patch.dict(
            os.environ,
            {"SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST": "gateway.example:custom-model"},
        ):
            self.assertTrue(runtime.chat_supports_native_tools())


if __name__ == "__main__":
    unittest.main()

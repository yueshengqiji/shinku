"""C2-5b 契约测试：模型能力协商。

测试只使用内存假客户端，覆盖原生工具、供应商画像、图片项、thinking
控制和 JSON 响应格式；不会发起网络请求。
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from shinku.llm.runtime import LLMRuntime


class _FakeCompletions:
    def __init__(self, owner: "_FakeClient") -> None:
        self.owner = owner

    def create(self, **payload):
        self.owner.payloads.append(payload)
        return self.owner.response


class _FakeClient:
    def __init__(self, protocol: str, base_url: str, response=None) -> None:
        self._shinku_protocol = protocol
        self.base_url = base_url
        self.response = response
        self.payloads: list[dict] = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))


def _runtime(
    *,
    protocol: str = "openai",
    base_url: str = "https://gateway.example/v1",
    model: str = "chat-model",
) -> LLMRuntime:
    clients: list[_FakeClient] = []

    def build(**_kwargs):
        client = _FakeClient(protocol, base_url)
        clients.append(client)
        return client

    with mock.patch("shinku.llm.runtime_core.build_llm_client", side_effect=build):
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_AUX_MODEL_NAME": "aux-model",
                "SHINKU_CHAT_MODEL_NAME": model,
            },
        ):
            runtime = LLMRuntime()
    runtime._test_clients = clients
    return runtime


def _tool(name: str = "read_file", description: str = "read") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


class ToolShapeTests(unittest.TestCase):
    def test_tool_schema_filters_invalid_duplicates_and_caps_count(self) -> None:
        runtime = _runtime()
        values = [_tool("read_file", "x" * 1000), _tool("read_file")]
        values.extend(_tool(f"tool_{index}") for index in range(70))
        values.extend([{"type": "text"}, {"type": "function", "function": {"name": "bad.name"}}])
        normalized = runtime._coerce_tools(values)
        # 工具配额按输入前 64 项计算；第二项是重复，所以有效值为 63。
        self.assertEqual(len(normalized), 63)
        self.assertEqual(normalized[0]["function"]["description"], "x" * 900)
        self.assertEqual(normalized[0]["function"]["name"], "read_file")
        self.assertEqual(len({item["function"]["name"] for item in normalized}), 63)

    def test_tool_choice_only_accepts_contract_values(self) -> None:
        runtime = _runtime()
        self.assertEqual(runtime._coerce_tool_choice(" REQUIRED "), "required")
        self.assertEqual(runtime._coerce_tool_choice("random"), "")
        choice = {"type": "function", "function": {"name": "read_file"}}
        self.assertIs(runtime._coerce_tool_choice(choice), choice)


class ProfileTests(unittest.TestCase):
    def test_known_deepseek_profile_enables_native_tools(self) -> None:
        runtime = _runtime(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        profile = runtime._tool_profile(runtime.chat)
        self.assertTrue(profile.supports_native_tools)
        self.assertFalse(profile.force_response_json)
        self.assertTrue(runtime.chat_supports_native_tools())

    def test_allowlist_can_enable_json_coexistence(self) -> None:
        runtime = _runtime(base_url="https://gateway.example/v1", model="custom")
        with mock.patch.dict(
            os.environ,
            {"SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST": "gateway.example:custom:json"},
        ):
            profile = runtime._tool_profile(runtime.chat)
        self.assertTrue(profile.supports_native_tools)
        self.assertTrue(profile.native_tools_coexist_with_forced_json)
        self.assertFalse(profile.verified)

    def test_unknown_provider_stays_disabled_without_allowlist(self) -> None:
        runtime = _runtime()
        with mock.patch.dict(os.environ, {"SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST": ""}):
            self.assertFalse(runtime.chat_supports_native_tools())


class NativeCallTests(unittest.TestCase):
    def test_non_stream_tool_call_is_normalized_and_extra_calls_counted(self) -> None:
        runtime = _runtime()
        calls = [
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="read_file", arguments='{"path":"a.txt"}'),
            ),
            SimpleNamespace(
                id="call-2",
                function=SimpleNamespace(name="other", arguments="{}"),
            ),
        ]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))]
        )
        result = runtime._read_native_tool_call(response)
        self.assertEqual(result["type"], "read_file")
        self.assertEqual(result["path"], "a.txt")
        self.assertEqual(result["_tool_invocation_id"], "call-1")
        self.assertEqual(runtime.snapshot_metrics()["native_tool_calls_extra"], 1)

    def test_stream_tool_fragments_are_joined_by_index(self) -> None:
        runtime = _runtime()
        parts: dict[int, dict] = {}
        first = SimpleNamespace(
            index=0,
            id="call-stream",
            function=SimpleNamespace(name="save_image", arguments='{"url":"'),
        )
        second = SimpleNamespace(
            index=0,
            function=SimpleNamespace(name="", arguments='https://x/1.png"}'),
        )
        # The second fragment completes the JSON object; the odd split is deliberate.
        runtime._collect_stream_tool_parts(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[first]))]),
            parts,
        )
        runtime._collect_stream_tool_parts(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[second]))]),
            parts,
        )
        result = runtime._assemble_stream_tool_call(parts)
        self.assertEqual(result["type"], "save_image")
        self.assertEqual(result["url"], "https://x/1.png")
        self.assertEqual(result["_tool_invocation_id"], "call-stream")

    def test_bad_tool_arguments_become_empty_object(self) -> None:
        runtime = _runtime()
        self.assertEqual(runtime._decode_tool_args("not-json"), {})
        self.assertEqual(runtime._decode_tool_args([1, 2]), {})


class MediaAndThinkingTests(unittest.TestCase):
    def test_image_items_accept_data_images_only_and_cap_at_five(self) -> None:
        runtime = _runtime()
        values = [{"dataUrl": f"data:image/png;base64,{index}"} for index in range(8)]
        values.append({"url": "https://example.test/a.png"})
        items = runtime._coerce_image_items(values)
        self.assertEqual(len(items), 5)
        self.assertTrue(all(item["type"] == "image_url" for item in items))

    def test_deepseek_payload_sends_tools_and_thinking_without_forced_json(self) -> None:
        runtime = _runtime(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        with mock.patch.dict(os.environ, {"SHINKU_LLM_THINKING_MODE": "enabled"}):
            payload = runtime._build_payload(
                bundle=runtime.chat,
                system_prompt="system",
                user_prompt="look",
                temperature=0.2,
                json_mode=True,
                native_tools=[_tool("read_file")],
                native_tool_choice="required",
            )
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})

    def test_images_are_embedded_in_user_message(self) -> None:
        runtime = _runtime()
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="看图",
            temperature=0.2,
            user_images=[{"data_url": "data:image/jpeg;base64,abc"}],
        )
        content = payload["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "看图"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_unknown_openai_uses_json_format_and_hint(self) -> None:
        runtime = _runtime()
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="稳定的人设",
            user_prompt="回答",
            temperature=0.2,
            json_mode=True,
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][1]["role"], "system")
        self.assertIn("JSON", payload["messages"][1]["content"])

    def test_non_deepseek_thinking_is_not_sent(self) -> None:
        runtime = _runtime(model="glm-5")
        with mock.patch.dict(os.environ, {"SHINKU_LLM_THINKING_MODE": "enabled"}):
            payload = runtime._build_payload(
                bundle=runtime.chat,
                system_prompt="system",
                user_prompt="question",
                temperature=0.2,
            )
        self.assertNotIn("extra_body", payload)

    def test_deepseek_model_probe_works_through_a_non_deepseek_gateway(self) -> None:
        runtime = _runtime(
            base_url="https://gateway.example/v1",
            model="deepseek-v4-flash",
        )
        self.assertTrue(runtime._supports_thinking_control(runtime.chat))

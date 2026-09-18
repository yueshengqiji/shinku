"""C2-2 契约测试：LLM 传输客户端。

对象是 ``shinku.llm.client`` 的**外部可观察行为**（契约文档
``docs/contracts/c2_2_llm_transport.md``）。传输层一律换掉：
``requests.post`` 换成记录调用的假件，``openai.OpenAI`` 换成记录构造参数的假类，
不发起任何真实网络请求。

边界类（``C2_2BoundaryTests``）沿用 B1 起每批都有的口径：外来标识 / 旧私有名 /
旧整句 / 旧环境变量前缀 / 旧扁平模块名不得复现。注意本批**合法地** import
``requests`` 与 ``openai``——C2-1 那条「不碰网络模块」的检查只覆盖 C2-1 的四个
模块，不套到本批头上（本批就是传输层）。
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
import tokenize as py_tokenize
import unittest
from pathlib import Path
from types import SimpleNamespace

from shinku.llm import client as client_module
from shinku.llm.client import AnthropicCompatClient, build_llm_client

SHINKU_PKG = Path(client_module.__file__).resolve().parents[1]

_NOT_JSON = object()


# --------------------------------------------------------------------------- #
# 假传输层
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """只实现客户端用到的那几个成员。"""

    def __init__(self, *, status: int = 200, body=_NOT_JSON, text: str = "", lines=None):
        self.status_code = status
        self._body = body
        self.text = text
        self._lines = list(lines or [])
        self.encoding = None
        self.closed = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._body is _NOT_JSON:
            raise ValueError("not json")
        return self._body

    def iter_lines(self, decode_unicode: bool = False):
        yield from self._lines

    def close(self) -> None:
        self.closed = True


class _FakeRequests:
    """记录每次 ``post`` 的全部入参，按队列发假响应。"""

    def __init__(self, *responses: _FakeResponse):
        self.queue = list(responses)
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.queue.pop(0) if self.queue else _FakeResponse()


class _FakeOpenAI:
    """记录构造参数；允许挂属性（模拟 SDK 客户端对象）。"""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        _FakeOpenAI.last = self


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


class _Transport:
    """给单个测试换装传输层，收工时还原。

    断言要读假件的 ``calls`` 时，**用 transport 实例**（``t.requests.calls``），
    不要用 ``client_module.requests``——块退出后模块属性已还原成真 ``requests``。
    """

    def __init__(self, *responses: _FakeResponse):
        self.requests = _FakeRequests(*responses)

    def __enter__(self):
        self._saved = (client_module.requests, client_module.OpenAI)
        client_module.requests = self.requests
        client_module.OpenAI = _FakeOpenAI
        _FakeOpenAI.last = None
        return self.requests

    def __exit__(self, *exc) -> None:
        client_module.requests, client_module.OpenAI = self._saved

    @property
    def calls(self) -> list[dict]:
        return self.requests.calls


def _sse(*events, trailing_blank: bool = True) -> list:
    """把若干 ``(event 名, payload dict)`` 拼成事件行序列，空行是事件边界。"""
    lines: list = []
    for name, payload in events:
        lines.append(f"event: {name}".encode())
        lines.append(f"data: {json.dumps(payload)}".encode())
        lines.append(b"")
    if trailing_blank and lines and lines[-1] != b"":
        lines.append(b"")
    return lines


def _ok_body(**overrides) -> dict:
    body = {
        "id": "msg_1",
        "model": "claude-x",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 7, "output_tokens": 3,
                  "cache_read_input_tokens": 11, "cache_creation_input_tokens": 5},
    }
    body.update(overrides)
    return body


def _client(**overrides) -> AnthropicCompatClient:
    kwargs = dict(api_key="k-1", base_url="https://api.anthropic.com")
    kwargs.update(overrides)
    return AnthropicCompatClient(**kwargs)


# --------------------------------------------------------------------------- #
# 1. build_llm_client：协议选择与两条分支
# --------------------------------------------------------------------------- #


class BuildClientSelectionTests(unittest.TestCase):
    """协议判定 + 分支出路 + SDK 构造参数。"""

    def test_explicit_anthropic_short_circuits_to_compat_client(self) -> None:
        with _Transport() as requests:
            built = build_llm_client(api_key="a", base_url="https://anything.test",
                                     protocol="anthropic")
        self.assertIsInstance(built, AnthropicCompatClient)
        self.assertEqual(built.protocol, "anthropic")
        self.assertEqual(built.base_url, "https://anything.test")  # 不归一化
        self.assertEqual(requests.calls, [])

    def test_url_signal_routes_to_anthropic_without_explicit_protocol(self) -> None:
        with _Transport():
            built = build_llm_client(api_key="a", base_url="https://api.anthropic.com")
        self.assertIsInstance(built, AnthropicCompatClient)

    def test_openai_protocol_uses_the_sdk(self) -> None:
        with _Transport():
            built = build_llm_client(api_key="sk", base_url="https://api.openai.com/v1",
                                     protocol="openai")
        self.assertNotIsInstance(built, AnthropicCompatClient)
        self.assertEqual(built.protocol, "openai")
        self.assertEqual(built.api_key, "sk")
        self.assertEqual(built.base_url, "https://api.openai.com/v1")
        self.assertEqual(_FakeOpenAI.last.kwargs["timeout"], 60.0)
        self.assertEqual(_FakeOpenAI.last.kwargs["max_retries"], 0)

    def test_o_llama_branch_gets_its_placeholder_key_and_normalized_base(self) -> None:
        with _Transport():
            built = build_llm_client(api_key="  ", base_url="http://127.0.0.1:11434",
                                     protocol="ollama")
        self.assertEqual(_FakeOpenAI.last.kwargs["api_key"], "ollama")
        self.assertEqual(_FakeOpenAI.last.kwargs["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(built.protocol, "ollama")

    def test_empty_key_falls_back_to_not_configured(self) -> None:
        with _Transport():
            build_llm_client(api_key="", base_url="https://api.deepseek.com/v1")
        self.assertEqual(_FakeOpenAI.last.kwargs["api_key"], "not-configured")

    def test_none_key_is_treated_as_empty(self) -> None:
        with _Transport():
            build_llm_client(api_key=None, base_url="https://api.deepseek.com/v1")
        self.assertEqual(_FakeOpenAI.last.kwargs["api_key"], "not-configured")

    def test_key_is_stripped_before_use(self) -> None:
        with _Transport():
            build_llm_client(api_key="  sk-x  ", base_url="https://api.deepseek.com/v1")
        self.assertEqual(_FakeOpenAI.last.kwargs["api_key"], "sk-x")

    def test_timeout_and_retries_are_forwarded(self) -> None:
        with _Transport():
            build_llm_client(api_key="k", base_url="https://x.test", protocol="openai",
                             timeout=12.5, max_retries=4)
        self.assertEqual(_FakeOpenAI.last.kwargs["timeout"], 12.5)
        self.assertEqual(_FakeOpenAI.last.kwargs["max_retries"], 4)

    def test_unknown_protocol_falls_back_to_openai(self) -> None:
        with _Transport():
            built = build_llm_client(api_key="k", base_url="https://plain.test",
                                     protocol="bogus")
        self.assertEqual(built.protocol, "openai")
        self.assertEqual(_FakeOpenAI.last.kwargs["base_url"], "https://plain.test")

    def test_sdk_client_that_rejects_attributes_still_gets_built(self) -> None:
        class _Locked:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                for key, value in kwargs.items():
                    object.__setattr__(self, key, value)

            def __setattr__(self, name, value):
                raise AttributeError("locked")

        saved = client_module.OpenAI
        client_module.OpenAI = _Locked
        try:
            with _Transport():
                built = build_llm_client(api_key="k", base_url="https://x.test")
        finally:
            client_module.OpenAI = saved
        self.assertEqual(built.kwargs["base_url"], "https://x.test")


# --------------------------------------------------------------------------- #
# 2. AnthropicCompatClient 构造归一化
# --------------------------------------------------------------------------- #


class CompatClientConstructionTests(unittest.TestCase):

    def test_all_four_fields_are_normalized(self) -> None:
        obj = _client(api_key="  k  ", base_url="  https://a.test  ",
                      timeout=None, max_retries=None)
        self.assertEqual(obj.api_key, "k")
        self.assertEqual(obj.base_url, "https://a.test")
        self.assertEqual(obj.timeout, 60.0)
        self.assertEqual(obj.max_retries, 0)

    def test_zero_timeout_falls_back_to_default(self) -> None:
        self.assertEqual(_client(timeout=0).timeout, 60.0)
        self.assertEqual(_client(timeout=0.5).timeout, 0.5)

    def test_zero_retries_stay_zero(self) -> None:
        self.assertEqual(_client(max_retries=0).max_retries, 0)
        self.assertEqual(_client(max_retries=3).max_retries, 3)

    def test_protocol_is_always_anthropic(self) -> None:
        self.assertEqual(_client().protocol, "anthropic")

    def test_chat_surface_reaches_create(self) -> None:
        self.assertTrue(callable(_client().chat.completions.create))


# --------------------------------------------------------------------------- #
# 3. 端点拼装与请求头
# --------------------------------------------------------------------------- #


class EndpointAndHeaderTests(unittest.TestCase):

    def test_endpoint_shapes(self) -> None:
        cases = {
            "https://api.anthropic.com": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/": "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages": "https://api.anthropic.com/v1/messages",
            "https://proxy.test/anthropic": "https://proxy.test/anthropic/v1/messages",
            "": "/v1/messages",
        }
        for base, want in cases.items():
            with self.subTest(base=base):
                transport = _Transport(_FakeResponse(body=_ok_body()))
                with transport:
                    _client(base_url=base).chat.completions.create(model="m", messages=[])
                self.assertEqual(transport.calls[0]["url"], want)

    def test_headers_are_the_wire_contract(self) -> None:
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client(api_key="kk").chat.completions.create(model="m", messages=[])
        self.assertEqual(transport.calls[0]["headers"], {
            "content-type": "application/json",
            "x-api-key": "kk",
            "anthropic-version": "2023-06-01",
        })

    def test_request_body_is_sent_as_json(self) -> None:
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client().chat.completions.create(model="m", messages=[])
        call = transport.calls[0]
        self.assertIn("json", call)
        self.assertIn("timeout", call)
        self.assertFalse(call["stream"])


# --------------------------------------------------------------------------- #
# 4. 请求体改写
# --------------------------------------------------------------------------- #


class PayloadRewriteTests(unittest.TestCase):

    def _payload(self, **options) -> dict:
        options.setdefault("model", "m")
        options.setdefault("messages", [])
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client().chat.completions.create(**options)
        return transport.calls[0]["json"]

    def test_minimal_payload_shape(self) -> None:
        payload = self._payload()
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertNotIn("system", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("stop_sequences", payload)
        self.assertNotIn("stream", payload)

    def test_max_tokens_fallback_order(self) -> None:
        self.assertEqual(self._payload(max_tokens=0)["max_tokens"], 1024)
        self.assertEqual(self._payload(max_completion_tokens=77)["max_tokens"], 77)
        self.assertEqual(self._payload(max_tokens=5, max_completion_tokens=9)["max_tokens"], 5)
        self.assertEqual(self._payload(max_tokens=0, max_completion_tokens=0)["max_tokens"], 1024)

    def test_temperature_is_clamped_into_unit_interval(self) -> None:
        self.assertEqual(self._payload(temperature=-3)["temperature"], 0.0)
        self.assertEqual(self._payload(temperature=0.4)["temperature"], 0.4)
        self.assertEqual(self._payload(temperature=9)["temperature"], 1.0)
        self.assertNotIn("temperature", self._payload(temperature=None))

    def test_top_p_is_passed_through_when_present(self) -> None:
        self.assertEqual(self._payload(top_p=0.0)["top_p"], 0.0)
        self.assertEqual(self._payload(top_p=0.95)["top_p"], 0.95)
        self.assertNotIn("top_p", self._payload(top_p=None))

    def test_stop_string_becomes_single_element_list(self) -> None:
        self.assertEqual(self._payload(stop="END")["stop_sequences"], ["END"])
        self.assertNotIn("stop_sequences", self._payload(stop="   "))
        self.assertNotIn("stop_sequences", self._payload(stop=""))

    def test_stop_list_is_stringified_not_nulled(self) -> None:
        # None 在列表里会变成字符串 "None" 并保留——契约 §1.6（与系统块的取值语义不同）。
        self.assertEqual(self._payload(stop=["a", "  b  ", ""])["stop_sequences"], ["a", "b"])
        self.assertEqual(self._payload(stop=[None])["stop_sequences"], ["None"])
        self.assertNotIn("stop_sequences", self._payload(stop=[]))
        self.assertNotIn("stop_sequences", self._payload(stop=5))

    def test_stream_flag_only_written_when_key_present(self) -> None:
        self.assertNotIn("stream", self._payload())
        self.assertFalse(self._payload(stream=False)["stream"])
        self.assertTrue(self._payload(stream=1)["stream"])
        self.assertNotIn("stream", self._payload(stream=None))

    def test_extra_body_does_not_override_explicit_fields(self) -> None:
        payload = self._payload(extra_body={"top_k": 3, "model": "IGNORED", "max_tokens": 1})
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["top_k"], 3)

    def test_extra_body_non_dict_is_ignored(self) -> None:
        payload = self._payload(extra_body="not-a-dict")
        self.assertNotIn("top_k", payload)


class SystemBlocksTests(unittest.TestCase):

    def _payload(self, messages, **options) -> dict:
        options.setdefault("model", "m")
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client().chat.completions.create(messages=messages, **options)
        return transport.calls[0]["json"]

    def test_system_and_developer_roles_are_extracted(self) -> None:
        payload = self._payload([
            {"role": "system", "content": "sys-a"},
            {"role": "developer", "content": "dev-b"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ])
        self.assertEqual(payload["messages"],
                         [{"role": "user", "content": "u"},
                          {"role": "assistant", "content": "a"}])
        self.assertEqual(payload["system"], [
            {"type": "text", "text": "sys-a\ndev-b", "cache_control": {"type": "ephemeral"}},
        ])

    def test_system_blocks_from_extra_blocks_join_the_system_text(self) -> None:
        payload = self._payload(
            [{"role": "system", "content": "main"}],
            system_extra_blocks=[" extra-1 ", "", None, "extra-2"],
        )
        self.assertEqual([block["text"] for block in payload["system"]],
                         ["main", "extra-1", "extra-2"])

    def test_only_the_first_four_blocks_carry_cache_control(self) -> None:
        payload = self._payload(
            [{"role": "system", "content": "s"}],
            system_extra_blocks=["e1", "e2", "e3", "e4", "e5"],
        )
        flagged = [block for block in payload["system"] if "cache_control" in block]
        self.assertEqual(len(flagged), 4)
        self.assertNotIn("cache_control", payload["system"][4])

    def test_empty_system_text_is_dropped(self) -> None:
        payload = self._payload([{"role": "system", "content": ""}, {"role": "user", "content": "z"}])
        self.assertNotIn("system", payload)

    def test_system_list_content_keeps_only_text_blocks(self) -> None:
        payload = self._payload([
            {"role": "system", "content": [
                {"type": "text", "text": " t1 "},
                {"type": "image_url", "image_url": {"url": "https://x"}},
                {"type": "text", "text": "t2"},
                "not-a-dict",
            ]},
        ])
        self.assertEqual(payload["system"][0]["text"], "t1\nt2")

    def test_all_blank_system_messages_leave_no_system_field(self) -> None:
        payload = self._payload([
            {"role": "system", "content": []},
            {"role": "developer", "content": "   "},
        ])
        self.assertNotIn("system", payload)


class MessageRewriteTests(unittest.TestCase):

    def _messages(self, messages) -> list:
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client().chat.completions.create(model="m", messages=messages)
        return transport.calls[0]["json"]["messages"]

    def test_unknown_roles_become_user(self) -> None:
        self.assertEqual(self._messages([{"role": "tool", "content": "x"}]),
                         [{"role": "user", "content": "x"}])
        self.assertEqual(self._messages([{"role": "", "content": "y"}]),
                         [{"role": "user", "content": "y"}])
        self.assertEqual(self._messages([{"content": "no-role"}]),
                         [{"role": "user", "content": "no-role"}])

    def test_non_dict_items_are_skipped(self) -> None:
        self.assertEqual(self._messages(["junk", 5, {"role": "user", "content": "ok"}]),
                         [{"role": "user", "content": "ok"}])

    def test_role_is_normalized_to_lowercase(self) -> None:
        self.assertEqual(self._messages([{"role": "USER", "content": "x"}]),
                         [{"role": "user", "content": "x"}])

    def test_string_content_passes_through_untouched(self) -> None:
        self.assertEqual(self._messages([{"role": "user", "content": "  keep  "}]),
                         [{"role": "user", "content": "  keep  "}])

    def test_text_blocks_are_collected_without_stripping(self) -> None:
        self.assertEqual(
            self._messages([{"role": "user", "content": [
                {"type": "text", "text": " t1 "}, {"type": "text", "text": "t2"},
            ]}]),
            [{"role": "user", "content": [
                {"type": "text", "text": " t1 "}, {"type": "text", "text": "t2"},
            ]}],
        )

    def test_inline_data_image_becomes_base64_source(self) -> None:
        messages = self._messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]}])
        self.assertEqual(messages[0]["content"], [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png", "data": "AAA"}},
        ])

    def test_remote_image_becomes_url_source(self) -> None:
        messages = self._messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x.test/y.png"}},
        ]}])
        self.assertEqual(messages[0]["content"], [
            {"type": "image", "source": {"type": "url", "url": "https://x.test/y.png"}},
        ])

    def test_bare_image_url_payload_is_accepted(self) -> None:
        messages = self._messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": "https://bare.test/i.png"},
        ]}])
        self.assertEqual(messages[0]["content"][0]["source"]["url"], "https://bare.test/i.png")

    def test_empty_image_urls_are_dropped(self) -> None:
        messages = self._messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": ""},
            {"type": "image_url"},
            {"type": "image_url", "image_url": {"url": "  "}},
        ]}])
        # 一个块都没收到 ⇒ 退化为纯文本（内容压平后为空串）。
        self.assertEqual(messages[0]["content"], "")

    def test_non_text_non_image_blocks_fall_back_to_plain_text(self) -> None:
        messages = self._messages([{"role": "user", "content": [{"type": "weird"}]}])
        self.assertEqual(messages[0]["content"], "")

    def test_non_list_non_string_content_is_stringified(self) -> None:
        self.assertEqual(self._messages([{"role": "user", "content": 42}]),
                         [{"role": "user", "content": "42"}])
        self.assertEqual(self._messages([{"role": "user", "content": None}]),
                         [{"role": "user", "content": ""}])

    def test_mixed_blocks_stay_blocks(self) -> None:
        messages = self._messages([{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}])
        self.assertEqual(len(messages[0]["content"]), 2)
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "look"})


class RequestTimeoutTests(unittest.TestCase):

    def _call(self, **options) -> float:
        transport = _Transport(_FakeResponse(body=_ok_body()))
        with transport:
            _client(**options.pop("_client", {})).chat.completions.create(
                model="m", messages=[], **options)
        return transport.calls[0]["timeout"]

    def test_call_without_timeout_uses_client_timeout(self) -> None:
        self.assertEqual(self._call(_client={"timeout": 33.0}), 33.0)

    def test_call_with_timeout_wins_over_client_timeout(self) -> None:
        self.assertEqual(self._call(_client={"timeout": 33.0}, timeout=12.5), 12.5)

    def test_falsy_call_timeout_falls_back_to_client_timeout(self) -> None:
        self.assertEqual(self._call(_client={"timeout": 33.0}, timeout=0), 33.0)


# --------------------------------------------------------------------------- #
# 5. 非流式回装
# --------------------------------------------------------------------------- #


def _completion_view(obj) -> dict:
    usage = vars(obj.usage)
    return {
        # 生成 id 带随机 uuid hex：只比「是否生成」，非生成时比字面值。
        "id": None if obj.id.startswith("chatcmpl-") else obj.id,
        "object": obj.object,
        "created": obj.created,
        "model": obj.model,
        "choices": [(c.index, c.finish_reason, c.message.role,
                     c.message.content, list(c.message.tool_calls))
                    for c in obj.choices],
        "usage": usage,
    }


class NonStreamResponseTests(unittest.TestCase):

    def _create(self, body, **options):
        options.setdefault("model", "req-model")
        options.setdefault("messages", [])
        with _Transport(_FakeResponse(body=body)):
            return _client().chat.completions.create(**options)

    def test_full_shape_matches_the_contract(self) -> None:
        view = _completion_view(self._create(_ok_body()))
        self.assertEqual(view["id"], "msg_1")
        self.assertEqual(view["object"], "chat.completion")
        self.assertEqual(view["created"], 0)
        self.assertEqual(view["model"], "claude-x")
        self.assertEqual(view["choices"], [(0, "stop", "assistant", "hi", [])])
        self.assertEqual(view["usage"], {
            "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            "cache_read_input_tokens": 11, "cache_creation_input_tokens": 5,
        })

    def test_generated_id_when_body_has_no_id(self) -> None:
        body = _ok_body()
        body.pop("id")  # 键缺失才走生成；id="" 会原样返回空串
        view = _completion_view(self._create(body))
        self.assertIsNone(view["id"])  # 生成的 chatcmpl-<uuid> 前缀

    def test_model_falls_back_to_requested_model(self) -> None:
        view = _completion_view(self._create(_ok_body(model="")))
        self.assertEqual(view["model"], "req-model")

    def test_text_pieces_are_concatenated_and_stripped(self) -> None:
        view = _completion_view(self._create(_ok_body(
            content=[{"type": "text", "text": " a "}, {"type": "text", "text": "b"}])))
        self.assertEqual(view["choices"][0][3], "a b")

    def test_non_text_blocks_are_skipped(self) -> None:
        view = _completion_view(self._create(_ok_body(
            content=[{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "t"}])))
        self.assertEqual(view["choices"][0][3], "t")

    def test_empty_content_yields_empty_string(self) -> None:
        for content in ([], "not-a-list", None):
            with self.subTest(content=content):
                view = _completion_view(self._create({"content": content}))
                self.assertEqual(view["choices"][0][3], "")

    def test_missing_blocks_get_defaults(self) -> None:
        view = _completion_view(self._create({}))
        self.assertIsNone(view["choices"][0][1])
        self.assertEqual(view["usage"], {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        })

    def test_finish_reason_mapping_table(self) -> None:
        table = {"end_turn": "stop", "max_tokens": "length",
                 "stop_sequence": "stop", "tool_use": "tool_calls",
                 "weird": None, "": None, None: None}
        for stop_reason, want in table.items():
            with self.subTest(stop_reason=stop_reason):
                view = _completion_view(self._create(_ok_body(stop_reason=stop_reason)))
                self.assertEqual(view["choices"][0][1], want)

    def test_stringy_usage_numbers_survive(self) -> None:
        view = _completion_view(self._create(_ok_body(
            usage={"input_tokens": "5", "output_tokens": None})))
        self.assertEqual(view["usage"]["prompt_tokens"], 5)
        self.assertEqual(view["usage"]["completion_tokens"], 0)
        self.assertEqual(view["usage"]["total_tokens"], 5)

    def test_total_is_the_sum_not_the_server_value(self) -> None:
        view = _completion_view(self._create(_ok_body(
            usage={"input_tokens": 2, "output_tokens": 3, "total": 999})))
        self.assertEqual(view["usage"]["total_tokens"], 5)


# --------------------------------------------------------------------------- #
# 6. 流式 SSE
# --------------------------------------------------------------------------- #


class StreamEventTests(unittest.TestCase):

    def _stream(self, lines, **options):
        options.setdefault("model", "m")
        options.setdefault("messages", [])
        options["stream"] = True
        transport = _Transport(_FakeResponse(lines=lines))
        with transport:
            stream = _client().chat.completions.create(**options)
            chunks = list(stream)
        return stream, chunks, transport.calls[0]

    def _views(self, chunks) -> list:
        return [(c.object, c.model, c.choices[0].index,
                 c.choices[0].delta.content, c.choices[0].finish_reason)
                for c in chunks]

    def test_post_is_called_with_stream_true(self) -> None:
        _stream, _chunks, call = self._stream(
            _sse(("content_block_delta", {"delta": {"text": "x"}})))
        self.assertTrue(call["stream"])

    def test_full_event_sequence_is_translated(self) -> None:
        lines = _sse(
            ("message_start", {"message": {"usage": {
                "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2}}}),
            ("content_block_start", {"content_block": {"type": "text", "text": "He"}}),
            ("content_block_delta", {"delta": {"type": "text_delta", "text": "llo"}}),
            ("content_block_delta", {"delta": {"type": "text_delta", "text": " world"}}),
            ("message_delta", {"delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {"type": "message_stop"}),
        )
        stream, chunks, _ = self._stream(lines)
        self.assertEqual(self._views(chunks), [
            ("chat.completion.chunk", "m", 0, "He", None),
            ("chat.completion.chunk", "m", 0, "llo", None),
            ("chat.completion.chunk", "m", 0, " world", None),
            ("chat.completion.chunk", "m", 0, "", "stop"),
        ])
        self.assertEqual(vars(stream.usage), {
            "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2})

    def test_chunk_ids_are_generated_with_the_prefix(self) -> None:
        _stream, chunks, _ = self._stream(
            _sse(("content_block_delta", {"delta": {"text": "x"}})))
        self.assertTrue(chunks[0].id.startswith("chatcmpl-"))
        self.assertEqual(chunks[0].created, 0)
        self.assertEqual(len(chunks[0].choices), 1)

    def test_empty_text_events_do_not_emit(self) -> None:
        lines = _sse(
            ("content_block_start", {"content_block": {"text": ""}}),
            ("content_block_delta", {"delta": {"text": ""}}),
            ("ping", {"type": "ping"}),
        )
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual(chunks, [])

    def test_done_sentinel_and_empty_data_do_not_emit(self) -> None:
        lines = [b"event: content_block_delta", b"data: [DONE]", b"",
                 b"event: content_block_delta", b"data:", b"",
                 b"event: content_block_delta", b'data: {"delta": {"text": "C"}}', b""]
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], ["C"])

    def test_trailing_data_without_blank_line_still_emits(self) -> None:
        lines = [b"event: content_block_delta", b'data: {"delta": {"text": "tail"}}']
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], ["tail"])

    def test_string_and_none_lines_are_accepted(self) -> None:
        # None 行被跳过但不是事件边界；无事件名的合法 data 不出块。
        lines = ["event: content_block_delta", 'data: {"delta": {"text": "S"}}', "",
                 None, 'data: {"delta": {"text": "N"}}', ""]
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], ["S"])

    def test_multi_line_data_is_joined_by_newline(self) -> None:
        lines = [b"event: content_block_delta", b'data: {"delta": {"text":',
                 b'data: "B"}}', b""]
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], ["B"])

    def test_event_name_after_data_is_honoured(self) -> None:
        # data 行先出现、event 名后出现：出块发生在空行边界，此时两者都已就位。
        lines = [b'data: {"delta": {"text": "A"}}', b"event: content_block_delta", b""]
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], ["A"])

    def test_unrelated_lines_are_ignored(self) -> None:
        lines = [b": keep-alive comment", b"id: 7", b"event: ping",
                 b'data: {"delta": {"text": "k"}}', b""]
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual([c.choices[0].delta.content for c in chunks], [])

    def test_message_delta_maps_stop_reason(self) -> None:
        lines = _sse(("message_delta", {"delta": {"stop_reason": "max_tokens"}}))
        _stream, chunks, _ = self._stream(lines)
        self.assertEqual(chunks[0].choices[0].finish_reason, "length")
        self.assertEqual(chunks[0].choices[0].delta.content, "")

    def test_usage_is_none_without_message_start(self) -> None:
        _stream, _chunks, _ = self._stream(_sse(
            ("content_block_delta", {"delta": {"text": "x"}})))
        stream = _stream
        self.assertIsNone(stream.usage)

    def test_message_start_usage_extraction_semantics(self) -> None:
        # 能解析的 message_start 一律落 usage（缺字段补 0）；json 解析失败才保持 None。
        cases = [
            ({"message": {}}, {"cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}),
            ({"message": {"usage": {}}}, {"cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}),
            ({"message": {"usage": {"cache_read_input_tokens": 4,
                                    "cache_creation_input_tokens": 2}}},
             {"cache_read_input_tokens": 4, "cache_creation_input_tokens": 2}),
            ({"message": "not-a-dict"}, None),   # .get 失败 → 静默跳过
            ({}, {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}),
        ]
        for payload, want in cases:
            with self.subTest(payload=payload):
                lines = _sse(("message_start", payload),
                             ("content_block_delta", {"delta": {"text": "x"}}))
                stream, chunks, _ = self._stream(lines)
                if want is None:
                    self.assertIsNone(stream.usage)
                else:
                    self.assertEqual(vars(stream.usage), want)
                self.assertEqual(len(chunks), 1)

    def test_connection_is_closed_after_iteration(self) -> None:
        lines = _sse(("content_block_delta", {"delta": {"text": "x"}}))
        with _Transport(_FakeResponse(lines=lines)):
            stream = _client().chat.completions.create(model="m", messages=[], stream=True)
            self.assertFalse(stream._reply.closed)
            list(stream)
            self.assertTrue(stream._reply.closed)

    def test_connection_is_closed_when_iteration_breaks_early(self) -> None:
        lines = _sse(("content_block_delta", {"delta": {"text": "x"}}),
                     ("content_block_delta", {"delta": {"text": "y"}}))
        with _Transport(_FakeResponse(lines=lines)):
            stream = _client().chat.completions.create(model="m", messages=[], stream=True)
            for _ in stream:
                break
            import gc
            gc.collect()
            self.assertTrue(stream._reply.closed)


# --------------------------------------------------------------------------- #
# 7. 失败路径
# --------------------------------------------------------------------------- #


class ErrorPathTests(unittest.TestCase):

    def _error(self, *, status=500, body=_NOT_JSON, text=""):
        with _Transport(_FakeResponse(status=status, body=body, text=text)):
            try:
                _client().chat.completions.create(model="m", messages=[])
            except Exception as exc:  # noqa: BLE001
                return exc
        self.fail("expected an exception")

    def test_error_dict_message_is_used(self) -> None:
        exc = self._error(body={"error": {"message": "boom"}})
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(str(exc), "boom")

    def test_plain_error_value_is_stringified(self) -> None:
        exc = self._error(body={"error": "plain-error"})
        self.assertEqual(str(exc), "plain-error")

    def test_blank_message_gives_generic_error(self) -> None:
        # json 能解析但 message 为空 ⇒ 不走 text 兜底（兜底只在 json() 本身抛错时触发）。
        exc = self._error(status=400, body={"error": {"message": "   "}}, text="raw text body")
        self.assertEqual(str(exc), "Anthropic request failed: HTTP 400")

    def test_non_json_body_falls_back_to_text(self) -> None:
        exc = self._error(status=503, text="  spaced text  ")
        self.assertEqual(str(exc), "spaced text")

    def test_everything_blank_gives_the_generic_message(self) -> None:
        exc = self._error(status=502, text="")
        self.assertEqual(str(exc), "Anthropic request failed: HTTP 502")

    def test_error_null_or_empty_falls_back(self) -> None:
        self.assertEqual(str(self._error(body={"error": None})),
                         "Anthropic request failed: HTTP 500")
        self.assertEqual(str(self._error(body={"error": {}})),
                         "Anthropic request failed: HTTP 500")
        self.assertEqual(str(self._error(body={"unrelated": 1})),
                         "Anthropic request failed: HTTP 500")

    def test_json_non_dict_body_gives_generic_error(self) -> None:
        # json 能解析但不是 dict ⇒ 取不到 error 键 ⇒ 通用消息（text 兜底不触发）。
        exc = self._error(body="not-a-dict", text="texty")
        self.assertEqual(str(exc), "Anthropic request failed: HTTP 500")

    def test_ok_statuses_do_not_raise(self) -> None:
        for status in (200, 204, 299):
            with self.subTest(status=status):
                with _Transport(_FakeResponse(status=status, body=_ok_body())):
                    _client().chat.completions.create(model="m", messages=[])


# --------------------------------------------------------------------------- #
# 8. 边界（B1 起每批都有的口径）
# --------------------------------------------------------------------------- #


class C2_2BoundaryTests(unittest.TestCase):
    """「表达层独立」的直接证据 + 本批的两处有意分叉被钉住。"""

    C2_2_FILES = (
        SHINKU_PKG / "llm" / "__init__.py",
        SHINKU_PKG / "llm" / "client.py",
    )

    FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane")

    #: 旧私有名并集（对照物 ∪ Shinku 旧实现，32 个）。
    LEGACY_INTERNAL_NAMES = (
        # 上游 code_shared/llm_client.py
        "_anthropic_messages_endpoint",
        "_build_anthropic_payload",
        "_build_openai_style_response",
        "_build_stream_chunk",
        "_build_system_text_block",
        "_capture_stream_usage",
        "_chunk_from_sse_event",
        "_convert_content_blocks",
        "_convert_image_url_block",
        "_convert_messages",
        "_extract_anthropic_text",
        "_flatten_content_to_text",
        "_map_finish_reason",
        "_raise_for_status",
        # 两侧共有的实例成员名
        "_client", "_model", "_response",
        # Shinku 旧侧 services/llm_client.py 特有
        "_anthropic_request",
        "_as_openai_response",
        "_chunk",
        "_content_blocks",
        "_content_text",
        "_finish_reason",
        "_image_block",
        "_messages_url",
        "_nonempty_strings",
        "_raise_http_error",
        "_read_usage",
        "_response_text",
        "_stream_chunk",
        "_system_blocks",
        "_translate_messages",
    )

    LEGACY_PROSE = (
        "Provider-neutral LLM client construction and Anthropic compatibility.",
        "Project wrappers may add a legacy project-specific protocol attribute for old",
        "Shinku's provider client boundary.",
        "The application talks to providers through a small OpenAI-shaped surface.",
        "Small OpenAI-shaped facade over the Anthropic Messages endpoint.",
        "Build one client with the protocol inferred from explicit config.",
        "Compatibility helper retained for offline payload tests.",
    )

    def _sources(self):
        for path in self.C2_2_FILES:
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
        self.assertEqual(len(self.LEGACY_INTERNAL_NAMES), 32)
        self.assertEqual(len(set(self.LEGACY_INTERNAL_NAMES)),
                         len(self.LEGACY_INTERNAL_NAMES))

    def test_no_legacy_prose_sentence_survives(self) -> None:
        for path, text in self._sources():
            for sentence in self.LEGACY_PROSE:
                with self.subTest(module=path.name, sentence=sentence[:24]):
                    self.assertNotIn(sentence, text)

    def test_no_legacy_environment_prefix(self) -> None:
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("COMPANION_", text)
                self.assertNotIn("SHINKU_SERVER_", text)

    def test_the_legacy_flat_module_name_was_not_recreated(self) -> None:
        self.assertFalse((SHINKU_PKG / "llm_client.py").exists())
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("shinku.llm_client")

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        for forbidden in ("companion_v01", "code_shared", "akane"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_transport_module_is_the_only_network_import(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "llm" / "client.py")
        self.assertIn("requests", modules)
        self.assertIn("openai", modules)
        for forbidden in ("socket", "subprocess", "urllib", "http"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, modules)

    def test_client_reaches_for_the_provider_vocabulary(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "llm" / "client.py")
        self.assertIn("..providers.config", modules)

    def test_the_shinku_protocol_alias_is_not_provided(self) -> None:
        # 契约 §0.4 决定 2：兼容标记不携带。
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("_shinku_protocol", text)
                self.assertNotIn("_akane_protocol", text)

    def test_the_legacy_payload_shim_is_not_provided(self) -> None:
        # 契约 §0.4 决定 3：私有垫片与归一化函数重导出不保留。
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("_build_anthropic_payload", _identifiers(path))
        llm_all = set(importlib.import_module("shinku.llm").__all__)
        self.assertEqual(llm_all,
                         {"AnthropicCompatClient", "LlmCircuitBreaker",
                          "build_llm_client", "get_llm_circuit_breaker"})

    def test_the_client_surface_is_what_the_contract_says(self) -> None:
        self.assertEqual(set(client_module.__all__),
                         {"AnthropicCompatClient", "build_llm_client"})


class C2_2ContractLiteralTests(unittest.TestCase):
    """契约里「改坏了也不会让别处变红」的字面量，逐条钉住。"""

    def test_the_api_version_header_value(self) -> None:
        self.assertEqual(client_module._ANTHROPIC_API_VERSION, "2023-06-01")

    def test_the_default_max_tokens(self) -> None:
        self.assertEqual(client_module._DEFAULT_MAX_TOKENS, 1024)

    def test_the_system_cache_slot_count(self) -> None:
        self.assertEqual(client_module._SYSTEM_CACHE_SLOTS, 4)

    def test_the_stop_reason_map(self) -> None:
        self.assertEqual(client_module._STOP_REASON_MAP, {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        })

    def test_the_key_placeholders(self) -> None:
        self.assertEqual(client_module._KEY_PLACEHOLDERS, {"ollama": "ollama"})
        self.assertEqual(client_module._KEY_FALLBACK, "not-configured")

    def test_the_inline_image_pattern(self) -> None:
        self.assertEqual(client_module._INLINE_IMAGE_RE.pattern,
                         r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$")
        self.assertTrue(client_module._INLINE_IMAGE_RE.flags & re.IGNORECASE)

    def test_the_generic_error_message_template(self) -> None:
        with _Transport(_FakeResponse(status=599, text="")):
            try:
                _client().chat.completions.create(model="m", messages=[])
                self.fail("expected RuntimeError")
            except RuntimeError as exc:
                self.assertEqual(str(exc), "Anthropic request failed: HTTP 599")


if __name__ == "__main__":
    unittest.main()

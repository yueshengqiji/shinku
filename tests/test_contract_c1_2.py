"""C1-2 共享基础模块的测试。

这一批按来源记录 §5.9 的**路径 A 第二类**处理：模块含逻辑与分支，属**洁净室重写**。
因此断言的对象全部是**外部可观察行为**（契约 `docs/contracts/c1_2_shared_modules.md`），
不涉及任何实现结构；失败测试与行为测试同等重要——契约的一半价值在它拒绝什么。

两处刻意的地方：

1. **`request_context` 的异常路径**（契约 §2.7 最后一行、§7.2）。这是旧测试的缺口，
   也是本批唯一需要"直接驱动 `dispatch`"才能真验的一条：``TestClient`` 外层读
   `current_correlation_id()` 恒为空串，用它断言"已重置"是假通过。
   这里在**同一个任务内**前后取值，并额外断言调用期间**确实**处于绑定态——
   否则"重置"断言会因为"从没绑过"而空过。
2. **边界扫描**沿用 C1-1 的做法，但把旧项目的**内部符号名**也纳入：旧模块名、
   旧常量名、旧私有函数名一个都不该出现在本批新文件里。这是"表达层独立"最直接的证据。
"""

from __future__ import annotations

import asyncio
import re
import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from shinku import names
from shinku.api import correlation, sessions
from shinku.api.correlation import (
    CorrelationIdMiddleware,
    correlation_headers,
    current_correlation_id,
    normalize_correlation_id,
)
from shinku.tasks import artifacts
from shinku.tools import invocation

HEADER = names.CORRELATION_ID_HEADER
HEX32 = r"^[0-9a-f]{32}$"
SHINKU_PKG = Path(correlation.__file__).resolve().parents[1]


def _noop_app(scope, receive, send):  # pragma: no cover - 只作为占位，不应被调用
    raise AssertionError("the wrapped ASGI app must not be called when dispatch is driven directly")


class _FakeRequest:
    """只暴露中间件用到的那两个入口：``headers`` 与 ``state``。"""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = dict(headers or {})
        self.state = SimpleNamespace()


class _FakeResponse:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# §2 request_context —— 归一化
# --------------------------------------------------------------------------- #


class NormalizeCorrelationIdTests(unittest.TestCase):
    def test_conforming_values_pass_through_unchanged(self) -> None:
        for value in ("test-correlation-123", "AbC0._:-xyz", "a" * 8, "a" * 128):
            with self.subTest(value=value):
                self.assertEqual(normalize_correlation_id(value), value)

    def test_surrounding_whitespace_is_stripped_before_the_shape_check(self) -> None:
        self.assertEqual(normalize_correlation_id("  abcdefgh  "), "abcdefgh")

    def test_a_numeric_input_is_stringified_then_checked(self) -> None:
        # 转成字符串后长度 8、全数字 —— 恰好落在边界内，应当被采纳。
        self.assertEqual(normalize_correlation_id(12345678), "12345678")

    def test_only_the_boundary_lengths_are_accepted(self) -> None:
        self.assertEqual(normalize_correlation_id("b" * 8), "b" * 8)
        self.assertEqual(normalize_correlation_id("b" * 128), "b" * 128)
        self.assertRegex(normalize_correlation_id("b" * 7), HEX32)
        self.assertRegex(normalize_correlation_id("b" * 129), HEX32)

    def test_malformed_values_are_replaced_never_echoed(self) -> None:
        for value in (
            "abc defgh",
            "abc$defgh",
            "abc\ndefgh",
            "abc\tdefgh",
            "abc/defgh",
            "abc,defgh",
            "abc+defgh",
        ):
            with self.subTest(value=value):
                self.assertRegex(normalize_correlation_id(value), HEX32)
                self.assertNotEqual(normalize_correlation_id(value), value)

    def test_falsy_input_counts_as_not_provided(self) -> None:
        for value in (None, "", "   ", "\t", 0, False, [], {}):
            with self.subTest(value=value):
                self.assertRegex(normalize_correlation_id(value), HEX32)

    def test_an_out_of_range_number_is_not_padded_into_shape(self) -> None:
        self.assertRegex(normalize_correlation_id(12), HEX32)

    def test_generated_values_are_32_lowercase_hex_without_hyphens(self) -> None:
        self.assertRegex(normalize_correlation_id(None), HEX32)

    def test_two_generated_values_differ(self) -> None:
        self.assertNotEqual(normalize_correlation_id(None), normalize_correlation_id(None))

    def test_normalizing_a_generated_value_is_idempotent(self) -> None:
        generated = normalize_correlation_id("not usable")
        self.assertEqual(normalize_correlation_id(generated), generated)


class CorrelationContextTests(unittest.TestCase):
    def test_the_context_is_empty_outside_a_request(self) -> None:
        self.assertEqual(current_correlation_id(), "")

    def test_normalizing_does_not_bind_the_context(self) -> None:
        normalize_correlation_id(None)
        self.assertEqual(current_correlation_id(), "")


class CorrelationHeadersTests(unittest.TestCase):
    def test_the_header_uses_the_protocol_name_and_holds_one_entry(self) -> None:
        headers = correlation_headers()
        self.assertEqual(set(headers), {HEADER})
        self.assertRegex(headers[HEADER], HEX32)

    def test_each_call_without_a_context_mints_a_fresh_value(self) -> None:
        self.assertNotEqual(correlation_headers()[HEADER], correlation_headers()[HEADER])

    def test_building_headers_does_not_bind_the_context(self) -> None:
        correlation_headers()
        self.assertEqual(current_correlation_id(), "")


# --------------------------------------------------------------------------- #
# §2 request_context —— 中间件（直接驱动 dispatch）
# --------------------------------------------------------------------------- #


class MiddlewareDispatchTests(unittest.TestCase):
    """直接驱动 ``dispatch``——只有在同一任务内读上下文，断言才有意义（契约 §7.2）。"""

    def _dispatch(self, request: _FakeRequest, call_next):
        middleware = CorrelationIdMiddleware(app=_noop_app)
        return asyncio.run(middleware.dispatch(request, call_next))

    def test_a_conforming_inbound_header_is_bound_and_echoed(self) -> None:
        seen: dict[str, object] = {}

        async def call_next(request):
            seen["context"] = current_correlation_id()
            seen["state"] = request.state.correlation_id
            return _FakeResponse()

        response = self._dispatch(_FakeRequest({HEADER: "test-correlation-123"}), call_next)

        self.assertEqual(seen["context"], "test-correlation-123")
        self.assertEqual(seen["state"], "test-correlation-123")
        self.assertEqual(response.headers[HEADER], "test-correlation-123")

    def test_a_missing_header_mints_one_value_shared_by_handler_and_response(self) -> None:
        seen: dict[str, object] = {}

        async def call_next(request):
            seen["context"] = current_correlation_id()
            return _FakeResponse()

        response = self._dispatch(_FakeRequest(), call_next)

        self.assertRegex(str(seen["context"]), HEX32)
        self.assertEqual(response.headers[HEADER], seen["context"])

    def test_a_malformed_header_is_replaced_and_the_original_is_dropped(self) -> None:
        seen: dict[str, object] = {}

        async def call_next(request):
            seen["context"] = current_correlation_id()
            return _FakeResponse()

        response = self._dispatch(_FakeRequest({HEADER: "short"}), call_next)

        self.assertNotEqual(response.headers[HEADER], "short")
        self.assertEqual(response.headers[HEADER], seen["context"])
        self.assertRegex(response.headers[HEADER], HEX32)

    def test_two_consecutive_requests_get_distinct_ids(self) -> None:
        async def call_next(request):
            return _FakeResponse()

        first = self._dispatch(_FakeRequest(), call_next).headers[HEADER]
        second = self._dispatch(_FakeRequest(), call_next).headers[HEADER]
        self.assertNotEqual(first, second)

    def test_the_context_is_restored_after_a_successful_request(self) -> None:
        async def scenario() -> str:
            middleware = CorrelationIdMiddleware(app=_noop_app)

            async def call_next(request):
                return _FakeResponse()

            await middleware.dispatch(_FakeRequest({HEADER: "ok-correlation-1"}), call_next)
            return current_correlation_id()

        self.assertEqual(asyncio.run(scenario()), "")

    def test_the_context_is_bound_during_the_call_and_restored_when_downstream_raises(self) -> None:
        """契约 §2.7 最后一行，旧测试的缺口。

        两句断言缺一不可：``during`` 证明绑定真的发生过（否则 ``after`` 是空过），
        ``after`` 证明异常路径上绑定被撤掉了。
        """

        async def scenario() -> tuple[str, str, str]:
            before = current_correlation_id()
            during: dict[str, str] = {}
            middleware = CorrelationIdMiddleware(app=_noop_app)

            async def failing_call_next(request):
                during["value"] = current_correlation_id()
                during["state"] = request.state.correlation_id
                raise RuntimeError("downstream failed")

            with self.assertRaises(RuntimeError):
                await middleware.dispatch(
                    _FakeRequest({HEADER: "fail-correlation-1"}), failing_call_next
                )

            return before, during["value"], current_correlation_id()

        before, during, after = asyncio.run(scenario())

        self.assertEqual(before, "")
        self.assertEqual(during, "fail-correlation-1")
        self.assertEqual(after, before)


class MiddlewareHttpTests(unittest.TestCase):
    """挂到真实应用上，证明它是可用的 Starlette 中间件而不只是一个可调用对象。"""

    def setUp(self) -> None:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)

        @app.get("/echo")
        async def echo(request: Request) -> dict[str, str]:
            return {
                "correlation_id": current_correlation_id(),
                "state": request.state.correlation_id,
            }

        self.client = TestClient(app)

    def test_a_conforming_header_is_adopted_end_to_end(self) -> None:
        response = self.client.get("/echo", headers={HEADER: "test-correlation-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["correlation_id"], "test-correlation-123")
        self.assertEqual(response.headers[HEADER], "test-correlation-123")

    def test_the_handler_state_and_the_response_agree_when_nothing_was_supplied(self) -> None:
        response = self.client.get("/echo")
        body = response.json()
        self.assertRegex(body["correlation_id"], HEX32)
        self.assertEqual(body["state"], body["correlation_id"])
        self.assertEqual(response.headers[HEADER], body["correlation_id"])

    def test_two_requests_do_not_share_an_id(self) -> None:
        first = self.client.get("/echo").json()["correlation_id"]
        second = self.client.get("/echo").json()["correlation_id"]
        self.assertNotEqual(first, second)


class CorrelationExportTests(unittest.TestCase):
    def test_the_export_surface_is_exactly_four_names(self) -> None:
        self.assertEqual(
            set(correlation.__all__),
            {
                "CorrelationIdMiddleware",
                "correlation_headers",
                "current_correlation_id",
                "normalize_correlation_id",
            },
        )

    def test_every_exported_name_resolves(self) -> None:
        for name in correlation.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(correlation, name))

    def test_the_header_name_is_not_re_exported(self) -> None:
        self.assertNotIn("CORRELATION_ID_HEADER", correlation.__all__)

    def test_the_header_name_has_a_single_definition_site(self) -> None:
        # 头名是线路协议；本仓库规定它只在 names 里定义一次。
        self.assertEqual(correlation.CORRELATION_ID_HEADER, names.CORRELATION_ID_HEADER)
        source = Path(correlation.__file__).read_text(encoding="utf-8")
        self.assertNotIn("CORRELATION_ID_HEADER =", source)


# --------------------------------------------------------------------------- #
# §3 routes/sessions —— 会话路由
# --------------------------------------------------------------------------- #

_PACK_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")

#: 会话读取面的最小替身。记录调用、返回预设值，并按需抛异常。
class _FakeStore:
    def __init__(self, **overrides) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.session = overrides.get("session", {"id": "s-1"})
        self.sessions = overrides.get("sessions", [{"id": "s-1"}])
        self.messages = overrides.get("messages", [{"id": "m-1"}])
        self.latest_turn = overrides.get("latest_turn", {"final_json": {"ok": 1}})
        self.rename_result = overrides.get("rename_result", {"id": "s-1", "title": "new"})
        self.raise_on_list = overrides.get("raise_on_list", False)
        self.raise_on_ensure = overrides.get("raise_on_ensure", False)

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))

    def call(self, name: str) -> dict:
        """返回某个方法的最后一次调用参数；方法未被调用则测试失败。"""
        matches = [kwargs for recorded, kwargs in self.calls if recorded == name]
        if not matches:
            raise AssertionError(f"{name} was never called; recorded={[c[0] for c in self.calls]}")
        return matches[-1]

    def ensure_session(
        self, *, profile_user_id, session_id, character_pack_id="", display_title=None
    ):
        self._record(
            "ensure_session",
            profile_user_id=profile_user_id,
            session_id=session_id,
            character_pack_id=character_pack_id,
            display_title=display_title,
        )
        if self.raise_on_ensure:
            raise RuntimeError("store unavailable")
        return self.session

    def get_session(self, profile_user_id, session_id):
        self._record("get_session", profile_user_id=profile_user_id, session_id=session_id)
        return None

    def get_character_session(self, *, profile_user_id, session_id, character_pack_id=""):
        self._record(
            "get_character_session",
            profile_user_id=profile_user_id,
            session_id=session_id,
            character_pack_id=character_pack_id,
        )
        return None

    def list_sessions(self, profile_user_id, limit=50, character_pack_id=None):
        self._record(
            "list_sessions",
            profile_user_id=profile_user_id,
            limit=limit,
            character_pack_id=character_pack_id,
        )
        if self.raise_on_list:
            raise RuntimeError("list failed")
        return list(self.sessions)

    def rename_session(self, *, profile_user_id, session_id, display_title):
        self._record(
            "rename_session",
            profile_user_id=profile_user_id,
            session_id=session_id,
            display_title=display_title,
        )
        return self.rename_result

    def get_session_messages(
        self, *, profile_user_id, session_id, character_pack_id=None, limit=120
    ):
        self._record(
            "get_session_messages",
            profile_user_id=profile_user_id,
            session_id=session_id,
            character_pack_id=character_pack_id,
            limit=limit,
        )
        return list(self.messages)

    def get_latest_eval_turn_for_session(
        self, *, profile_user_id, session_id, character_pack_id=None
    ):
        self._record(
            "get_latest_eval_turn_for_session",
            profile_user_id=profile_user_id,
            session_id=session_id,
            character_pack_id=character_pack_id,
        )
        return self.latest_turn


class _FakeMetrics:
    def __init__(self) -> None:
        self.observed: list[tuple[str, float, bool]] = []

    def observe_request(self, name: str, *, duration_ms: float, ok: bool) -> None:
        self.observed.append((name, duration_ms, ok))


class _FakeLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, **fields) -> None:
        self.events.append((event, fields))


def _error_envelope(*, error=None, message=None, retryable=False, details=None) -> dict:
    """契约 §3.7 的信封形状；这里作为**注入方**实现，验证路由真的用了注入的那一个。"""

    payload = {
        "ok": False,
        "status": "error",
        "contract_version": "desktop_pet.v0.1",
        "error": error or "unknown_error",
        "message": message or "请求失败。",
        "retryable": bool(retryable),
    }
    if details:
        payload["details"] = details
    return payload


def _resolve_identity(source) -> tuple[str, str]:
    if isinstance(source, Request):
        session_id = source.query_params.get("session_id") or "s-default"
        profile = source.query_params.get("profile_user_id") or "u-default"
        return str(session_id), str(profile)
    return str(source.get("session_id") or "s-default"), str(source.get("profile_user_id") or "u-default")


def _normalize_pack_id(value) -> str:
    text = str(value or "").strip()
    return text if text and _PACK_ID_PATTERN.fullmatch(text) else ""


def _sessions_client(**overrides):
    store = overrides.get("store") or _FakeStore()
    metrics = overrides.get("metrics") or _FakeMetrics()
    log = overrides.get("log") if "log" in overrides else _FakeLog()
    app = FastAPI()
    app.include_router(
        sessions.build_sessions_router(
            sessions=store,
            metrics=metrics,
            log_event=log,
            resolve_identity=overrides.get("resolve_identity") or _resolve_identity,
            normalize_pack_id=overrides.get("normalize_pack_id") or _normalize_pack_id,
            build_error_payload=overrides.get("build_error_payload") or _error_envelope,
        )
    )
    return TestClient(app), store, metrics, log


class SessionsRouterShapeTests(unittest.TestCase):
    def test_the_documented_limits_are_module_constants(self) -> None:
        self.assertEqual(sessions.SESSION_LIST_LIMIT, 50)
        self.assertEqual(sessions.MESSAGE_LIMIT, 120)

    def test_only_the_factory_is_exported(self) -> None:
        self.assertEqual(sessions.__all__, ["build_sessions_router"])

    def test_the_router_carries_no_prefix_and_exposes_three_paths(self) -> None:
        router = sessions.build_sessions_router(
            sessions=_FakeStore(),
            metrics=_FakeMetrics(),
            log_event=_FakeLog(),
            resolve_identity=_resolve_identity,
            normalize_pack_id=_normalize_pack_id,
            build_error_payload=_error_envelope,
        )
        self.assertEqual(router.prefix, "")
        self.assertEqual(
            {(route.path, tuple(sorted(route.methods))) for route in router.routes},
            {
                ("/sessions", ("GET",)),
                ("/sessions/ensure", ("POST",)),
                ("/sessions/rename", ("POST",)),
            },
        )

    def test_the_factory_takes_keyword_arguments_only(self) -> None:
        with self.assertRaises(TypeError):
            sessions.build_sessions_router(  # type: ignore[misc]
                _FakeStore(), _FakeMetrics(), _FakeLog(), _resolve_identity,
                _normalize_pack_id, _error_envelope,
            )


class SessionsListTests(unittest.TestCase):
    def test_limits_can_be_overridden_without_changing_module_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "SHINKU_SESSION_LIST_LIMIT": "7",
                "SHINKU_SESSION_MESSAGE_LIMIT": "33",
            },
        ):
            client, store, *_ = _sessions_client()
            client.get("/sessions")
            client.post("/sessions/ensure", json={"session_id": "s-9"})
        self.assertEqual(store.call("list_sessions")["limit"], 7)
        self.assertEqual(store.call("get_session_messages")["limit"], 33)

    def test_it_returns_the_list_and_the_current_session(self) -> None:
        client, *_ = _sessions_client()
        response = client.get("/sessions", params={"session_id": "s-9"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"sessions": [{"id": "s-1"}], "current_session_id": "s-9"}
        )

    def test_it_asks_for_at_most_fifty_sessions(self) -> None:
        client, store, *_ = _sessions_client()
        client.get("/sessions")
        self.assertEqual(store.call("list_sessions")["limit"], 50)

    def test_it_passes_the_profile_from_the_query(self) -> None:
        client, store, *_ = _sessions_client()
        client.get("/sessions", params={"profile_user_id": "u-7"})
        self.assertEqual(store.call("list_sessions")["profile_user_id"], "u-7")

    def test_the_pack_id_key_order_is_fixed(self) -> None:
        client, store, *_ = _sessions_client()
        client.get(
            "/sessions",
            params={"characterPackId": "secondary", "character_pack_id": "primary"},
        )
        self.assertEqual(store.call("list_sessions")["character_pack_id"], "primary")

    def test_the_pack_id_runs_through_the_injected_normalizer(self) -> None:
        client, store, *_ = _sessions_client()
        client.get("/sessions", params={"character_pack_id": "bad/id"})
        self.assertEqual(store.call("list_sessions")["character_pack_id"], "")

    def test_an_absent_pack_id_stays_none_rather_than_empty(self) -> None:
        client, store, *_ = _sessions_client()
        client.get("/sessions")
        self.assertIsNone(store.call("list_sessions")["character_pack_id"])

    def test_a_store_failure_is_measured_and_logged_then_reraised(self) -> None:
        store = _FakeStore(raise_on_list=True)
        client, _, metrics, log = _sessions_client(store=store)

        with self.assertRaises(RuntimeError):
            client.get("/sessions")

        self.assertEqual(len(metrics.observed), 1)
        name, duration_ms, ok = metrics.observed[0]
        self.assertEqual(name, "sessions_list")
        self.assertIsInstance(duration_ms, float)
        self.assertIs(ok, False)
        self.assertEqual(log.events[0][0], "sessions_list_error")
        self.assertEqual(log.events[0][1]["session_id"], "s-default")
        self.assertIn("list failed", log.events[0][1]["message"])

    def test_a_successful_list_is_measured_as_ok(self) -> None:
        client, _, metrics, log = _sessions_client()
        client.get("/sessions")
        self.assertEqual(len(metrics.observed), 1)
        self.assertEqual(metrics.observed[0][0], "sessions_list")
        self.assertIs(metrics.observed[0][2], True)
        self.assertEqual(log.events, [])


class SessionsEnsureTests(unittest.TestCase):
    def test_unreadable_json_is_a_400_carrying_no_store(self) -> None:
        client, _, metrics, _ = _sessions_client()
        response = client.post(
            "/sessions/ensure",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"], "invalid_json")
        self.assertFalse(body["retryable"])
        self.assertEqual(body["ok"], False)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIs(metrics.observed[-1][2], False)

    def test_the_echoed_reason_is_truncated_to_the_configured_bound(self) -> None:
        client, *_ = _sessions_client()
        with mock.patch.object(sessions, "_REASON_CHARS", 5):
            body = client.post(
                "/sessions/ensure",
                content=b"{not json",
                headers={"Content-Type": "application/json"},
            ).json()
        prefix = "无法读取会话请求："
        self.assertTrue(body["message"].startswith(prefix), body["message"])
        self.assertEqual(len(body["message"]) - len(prefix), 5)

    def test_a_payload_that_is_not_an_object_is_a_400(self) -> None:
        client, _, metrics, _ = _sessions_client()
        response = client.post("/sessions/ensure", json=[1, 2, 3])
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"], "invalid_payload")
        self.assertFalse(body["retryable"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIs(metrics.observed[-1][2], False)

    def test_an_empty_body_is_a_400_not_a_crash(self) -> None:
        client, *_ = _sessions_client()
        response = client.post("/sessions/ensure", content=b"")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_json")

    def test_the_success_body_is_the_four_part_session_state(self) -> None:
        client, store, metrics, _ = _sessions_client()
        response = client.post("/sessions/ensure", json={"session_id": "s-1"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body), {"session", "sessions", "messages", "latest_final_json"}
        )
        self.assertEqual(body["session"], {"id": "s-1"})
        self.assertEqual(body["latest_final_json"], {"ok": 1})
        self.assertEqual(store.call("get_session_messages")["limit"], 120)
        self.assertEqual(store.call("list_sessions")["limit"], 50)
        self.assertIs(metrics.observed[-1][2], True)

    def test_the_display_title_is_stripped_and_blank_becomes_none(self) -> None:
        client, store, *_ = _sessions_client()
        client.post("/sessions/ensure", json={"session_id": "s-1", "display_title": "  新会话  "})
        self.assertEqual(store.call("ensure_session")["display_title"], "新会话")

        client.post("/sessions/ensure", json={"session_id": "s-1", "display_title": "   "})
        self.assertIsNone(store.call("ensure_session")["display_title"])

    def test_identity_is_resolved_from_the_payload(self) -> None:
        client, store, *_ = _sessions_client()
        client.post("/sessions/ensure", json={"session_id": "s-42", "profile_user_id": "u-42"})
        call = store.call("ensure_session")
        self.assertEqual(call["session_id"], "s-42")
        self.assertEqual(call["profile_user_id"], "u-42")

    def test_the_pack_id_comes_from_the_payload_top_level_first(self) -> None:
        client, store, *_ = _sessions_client()
        client.post(
            "/sessions/ensure",
            json={
                "session_id": "s-1",
                "character_pack_id": "top",
                "current_visual": {"character_pack_id": "nested"},
            },
        )
        self.assertEqual(store.call("ensure_session")["character_pack_id"], "top")

    def test_the_pack_id_falls_back_to_the_nested_visual_block(self) -> None:
        client, store, *_ = _sessions_client()
        client.post(
            "/sessions/ensure",
            json={"session_id": "s-1", "current_visual": {"characterPackId": "nested"}},
        )
        self.assertEqual(store.call("ensure_session")["character_pack_id"], "nested")

    def test_a_present_but_empty_top_level_key_stops_the_nested_lookup(self) -> None:
        # "键存在"与"键有值"不是一回事：存在即命中，命中即返回。
        client, store, *_ = _sessions_client()
        client.post(
            "/sessions/ensure",
            json={
                "session_id": "s-1",
                "character_pack_id": None,
                "current_visual": {"character_pack_id": "nested"},
            },
        )
        self.assertEqual(store.call("ensure_session")["character_pack_id"], "")

    def test_the_pack_id_is_absent_only_when_no_key_exists_anywhere(self) -> None:
        client, store, *_ = _sessions_client()
        client.post("/sessions/ensure", json={"session_id": "s-1", "current_visual": "not a mapping"})
        self.assertEqual(store.call("ensure_session")["character_pack_id"], "")

    def test_latest_final_json_is_none_unless_the_turn_carries_a_mapping(self) -> None:
        for turn in (None, {}, {"final_json": None}, {"final_json": ["not", "a", "dict"]}):
            with self.subTest(turn=turn):
                client, *_ = _sessions_client(store=_FakeStore(latest_turn=turn))
                body = client.post("/sessions/ensure", json={"session_id": "s-1"}).json()
                self.assertIsNone(body["latest_final_json"])

    def test_a_store_failure_is_measured_and_logged_then_reraised(self) -> None:
        client, _, metrics, log = _sessions_client(store=_FakeStore(raise_on_ensure=True))

        with self.assertRaises(RuntimeError):
            client.post("/sessions/ensure", json={"session_id": "s-1"})

        self.assertEqual(len(metrics.observed), 1)
        self.assertEqual(metrics.observed[0][0], "sessions_ensure")
        self.assertIs(metrics.observed[0][2], False)
        self.assertEqual(log.events[0][0], "sessions_ensure_error")
        self.assertEqual(log.events[0][1]["session_id"], "s-1")


class SessionsRenameTests(unittest.TestCase):
    def test_a_blank_title_is_rejected(self) -> None:
        for title in ("", "   ", None):
            with self.subTest(title=title):
                client, store, metrics, _ = _sessions_client()
                response = client.post(
                    "/sessions/rename", json={"session_id": "s-1", "display_title": title}
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"], "display_title is required")
                self.assertIs(metrics.observed[-1][2], False)
                self.assertNotIn("rename_session", [name for name, _ in store.calls])

    def test_a_missing_session_is_a_404(self) -> None:
        client, *_ = _sessions_client(store=_FakeStore(rename_result=None))
        response = client.post(
            "/sessions/rename", json={"session_id": "s-1", "display_title": "标题"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "session not found")

    def test_a_successful_rename_returns_the_session_and_the_list(self) -> None:
        client, store, metrics, _ = _sessions_client()
        response = client.post(
            "/sessions/rename", json={"session_id": "s-1", "display_title": "  标题  "}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"session", "sessions"})
        self.assertEqual(body["session"], {"id": "s-1", "title": "new"})
        self.assertEqual(store.call("rename_session")["display_title"], "标题")
        self.assertEqual(store.call("list_sessions")["limit"], 50)
        self.assertIs(metrics.observed[-1][2], True)

    def test_the_rename_list_uses_the_payload_pack_id(self) -> None:
        client, store, *_ = _sessions_client()
        client.post(
            "/sessions/rename",
            json={"session_id": "s-1", "display_title": "标题", "character_pack": "p-1"},
        )
        self.assertEqual(store.call("list_sessions")["character_pack_id"], "p-1")


# --------------------------------------------------------------------------- #
# §4 task_artifacts —— 工作区产物
# --------------------------------------------------------------------------- #


class GeneratedFileArtifactTests(unittest.TestCase):
    def _build(self, generated, *, tool_type="media", send_to_user=False):
        return artifacts.artifact_from_generated_file(
            generated=generated, tool_type=tool_type, send_to_user=send_to_user
        )

    def test_a_non_mapping_input_yields_nothing(self) -> None:
        for bad in (None, [], "text", 3, object()):
            with self.subTest(bad=bad):
                self.assertIsNone(self._build(bad))

    def test_an_identity_is_required(self) -> None:
        for missing in ({}, {"generated_handle": "   "}, {"generated_id": ""}, {"output_title": "x"}):
            with self.subTest(missing=missing):
                self.assertIsNone(self._build(missing))

    def test_the_handle_wins_over_the_id(self) -> None:
        artifact = self._build({"generated_handle": " h ", "generated_id": " g-1 "})
        self.assertEqual(artifact["id"], "h")
        self.assertEqual(artifact["generated_handle"], "h")
        self.assertEqual(artifact["generated_id"], "g-1")

    def test_the_identity_falls_back_to_the_id(self) -> None:
        artifact = self._build({"generated_id": "g-1"})
        self.assertEqual(artifact["id"], "g-1")
        self.assertNotIn("generated_handle", artifact)

    def test_the_fixed_field_surface(self) -> None:
        artifact = self._build(
            {"generated_handle": "h", "output_format": "  PNG ", "output_title": " 报告 "},
            tool_type="media",
            send_to_user=True,
        )
        self.assertEqual(artifact["kind"], "png")
        self.assertEqual(artifact["title"], "报告")
        self.assertEqual(artifact["status"], "ready")
        self.assertEqual(artifact["source"], "generated_file")
        self.assertEqual(artifact["tool"], "media")
        self.assertIs(artifact["send_to_user"], True)
        self.assertEqual(artifact["delivery_role"], "requested_output")

    def test_send_to_user_decides_the_delivery_role(self) -> None:
        for flag, role in ((True, "requested_output"), (False, "workspace_material")):
            with self.subTest(flag=flag):
                artifact = self._build({"generated_handle": "h"}, send_to_user=flag)
                self.assertEqual(artifact["delivery_role"], role)

    def test_send_to_user_is_coerced_to_a_real_bool(self) -> None:
        artifact = self._build({"generated_handle": "h"}, send_to_user="yes")
        self.assertIs(artifact["send_to_user"], True)
        self.assertIs(self._build({"generated_handle": "h"})["send_to_user"], False)

    def test_the_kind_falls_back_to_the_extension_then_to_file(self) -> None:
        self.assertEqual(self._build({"generated_handle": "h", "file_ext": "DOCX"})["kind"], "docx")
        self.assertEqual(self._build({"generated_handle": "h"})["kind"], "file")
        self.assertEqual(self._build({"generated_handle": "h", "output_format": "  "})["kind"], "file")

    def test_the_title_falls_back_then_truncates(self) -> None:
        self.assertEqual(self._build({"generated_handle": "h"})["title"], "h")
        self.assertEqual(self._build({"generated_id": "g"})["title"], "生成文件")
        self.assertEqual(
            self._build({"generated_handle": "h", "output_title": "x" * 200})["title"], "x" * 120
        )

    def test_a_whitespace_only_status_falls_back_to_ready(self) -> None:
        self.assertEqual(self._build({"generated_handle": "h", "status": "   "})["status"], "ready")
        self.assertEqual(self._build({"generated_handle": "h", "status": " done "})["status"], "done")

    def test_the_stem_role_is_read_from_the_content_card_and_truncated(self) -> None:
        artifact = self._build(
            {
                "generated_handle": "h",
                "content_card": {"separation": {"stem_role": "r" * 60}},
            }
        )
        self.assertEqual(artifact["stem_role"], "r" * 40)

    def test_a_malformed_content_card_does_not_produce_a_stem_role(self) -> None:
        for card in (None, "text", {}, {"separation": "text"}, {"separation": {}}):
            with self.subTest(card=card):
                artifact = self._build({"generated_handle": "h", "content_card": card})
                self.assertNotIn("stem_role", artifact)

    def test_passthrough_fields_are_copied_verbatim(self) -> None:
        artifact = self._build(
            {
                "generated_handle": "h",
                "file_ext": "docx",
                "file_size": 2048,
                "created_by_tool": "media_agent",
                "version_of_generated_id": "g-0",
                "version_no": 3,
            }
        )
        self.assertEqual(artifact["file_size"], 2048)
        self.assertEqual(artifact["version_no"], 3)
        self.assertEqual(artifact["created_by_tool"], "media_agent")
        self.assertEqual(artifact["version_of_generated_id"], "g-0")

    def test_absent_passthrough_values_are_dropped(self) -> None:
        for absent in (None, "", [], {}):
            with self.subTest(absent=absent):
                artifact = self._build({"generated_handle": "h", "file_size": absent})
                self.assertNotIn("file_size", artifact)

    def test_zero_and_false_passthrough_values_survive(self) -> None:
        # "没给"的判定是空值集合，不是 falsy —— 0 与 False 是有效载荷。
        artifact = self._build({"generated_handle": "h", "file_size": 0, "version_no": False})
        self.assertEqual(artifact["file_size"], 0)
        self.assertIs(artifact["version_no"], False)


class AttachmentArtifactTests(unittest.TestCase):
    def _build(self, item, *, tool_type="media"):
        return artifacts.artifact_from_attachment_item(item=item, tool_type=tool_type)

    def test_a_non_mapping_input_yields_nothing(self) -> None:
        for bad in (None, [], "text", 3):
            with self.subTest(bad=bad):
                self.assertIsNone(self._build(bad))

    def test_an_identity_is_required(self) -> None:
        for missing in ({}, {"attachment_handle": "  "}, {"attachment_id": ""}):
            with self.subTest(missing=missing):
                self.assertIsNone(self._build(missing))

    def test_the_handle_wins_over_the_id(self) -> None:
        artifact = self._build({"attachment_handle": " h ", "attachment_id": " a-1 "})
        self.assertEqual(artifact["id"], "h")
        self.assertEqual(artifact["attachment_id"], "a-1")

    def test_the_fixed_field_surface(self) -> None:
        artifact = self._build(
            {
                "attachment_id": "a-1",
                "kind": " IMAGE ",
                "summary_title": " 封面 ",
                "status": "ready",
                "source": " clipboard ",
            }
        )
        self.assertEqual(artifact["kind"], "image")
        self.assertEqual(artifact["title"], "封面")
        self.assertEqual(artifact["source"], "attachment_inbox")
        self.assertEqual(artifact["source_type"], "clipboard")
        self.assertEqual(artifact["delivery_role"], "workspace_material")
        self.assertEqual(artifact["tool"], "media")

    def test_attachments_are_never_requested_output(self) -> None:
        artifact = self._build({"attachment_id": "a-1"})
        self.assertEqual(artifact["delivery_role"], "workspace_material")
        self.assertNotIn("send_to_user", artifact)

    def test_the_title_fallback_chain(self) -> None:
        self.assertEqual(
            self._build({"attachment_id": "a", "origin_name": "origin"})["title"], "origin"
        )
        self.assertEqual(self._build({"attachment_handle": "h"})["title"], "h")
        self.assertEqual(self._build({"attachment_id": "a"})["title"], "临时素材")

    def test_the_source_type_is_empty_when_the_item_carries_no_source(self) -> None:
        self.assertEqual(self._build({"attachment_id": "a"})["source_type"], "")

    def test_passthrough_fields_are_copied_and_absent_ones_dropped(self) -> None:
        artifact = self._build(
            {"attachment_id": "a", "origin_name": "o", "file_ext": "png", "file_size": 10, "mime_type": "image/png"}
        )
        self.assertEqual(artifact["mime_type"], "image/png")
        self.assertEqual(artifact["file_ext"], "png")
        sparse = self._build({"attachment_id": "a", "origin_name": "", "file_size": None})
        for key in ("origin_name", "file_size", "mime_type"):
            self.assertNotIn(key, sparse)


class ExtractArtifactsTests(unittest.TestCase):
    def test_only_the_two_known_event_types_produce_artifacts(self) -> None:
        events = [
            "not a mapping",
            None,
            {},
            {"type": "progress"},
            {"type": "generated_file_ready"},
            {"type": "attachment_remote_media_ready"},
        ]
        self.assertEqual(
            artifacts.extract_artifacts_from_tool_events(tool_type="t", stream_events=events), []
        )

    def test_the_result_keeps_the_stream_order(self) -> None:
        events = [
            {"type": "attachment_remote_media_ready", "item": {"attachment_id": "a-2"}},
            {"type": "noise"},
            {"type": "generated_file_ready", "generated_file": {"generated_id": "g-1"}},
        ]
        found = artifacts.extract_artifacts_from_tool_events(tool_type="t", stream_events=events)
        self.assertEqual([item["id"] for item in found], ["a-2", "g-1"])

    def test_the_send_to_user_flag_is_read_from_the_event(self) -> None:
        events = [
            {
                "type": "generated_file_ready",
                "generated_file": {"generated_id": "g-1"},
                "send_to_user": True,
            }
        ]
        found = artifacts.extract_artifacts_from_tool_events(tool_type="t", stream_events=events)
        self.assertIs(found[0]["send_to_user"], True)
        self.assertEqual(found[0]["delivery_role"], "requested_output")

    def test_the_tool_type_reaches_every_artifact(self) -> None:
        events = [
            {"type": "generated_file_ready", "generated_file": {"generated_id": "g-1"}},
            {"type": "attachment_remote_media_ready", "item": {"attachment_id": "a-1"}},
        ]
        found = artifacts.extract_artifacts_from_tool_events(tool_type="media", stream_events=events)
        self.assertEqual([item["tool"] for item in found], ["media", "media"])

    def test_an_empty_stream_yields_an_empty_list(self) -> None:
        self.assertEqual(
            artifacts.extract_artifacts_from_tool_events(tool_type="t", stream_events=[]), []
        )


class ArtifactIdentityTests(unittest.TestCase):
    def test_the_key_priority_is_fixed(self) -> None:
        artifact = {
            "id": "1",
            "generated_handle": "2",
            "generated_id": "3",
            "attachment_handle": "4",
            "attachment_id": "5",
        }
        for key, expected in (
            ("id", "1"),
            ("generated_handle", "2"),
            ("generated_id", "3"),
            ("attachment_handle", "4"),
            ("attachment_id", "5"),
        ):
            with self.subTest(leading_key=key):
                self.assertEqual(artifacts.artifact_identity(artifact), expected)
                del artifact[key]

    def test_values_are_stripped_before_being_used(self) -> None:
        self.assertEqual(artifacts.artifact_identity({"id": "  a-1  "}), "a-1")

    def test_whitespace_only_values_do_not_count(self) -> None:
        self.assertEqual(artifacts.artifact_identity({"id": "   ", "title": " t "}), "title::t")

    def test_the_title_fallback_includes_the_kind(self) -> None:
        self.assertEqual(artifacts.artifact_identity({"title": "报告", "kind": "docx"}), "title:docx:报告")

    def test_a_missing_kind_participates_as_an_empty_string(self) -> None:
        self.assertEqual(artifacts.artifact_identity({"title": "报告"}), "title::报告")

    def test_nothing_identifiable_is_an_empty_string(self) -> None:
        for artifact in ({}, {"id": ""}, {"title": "   "}, {"title": "", "kind": "docx"}):
            with self.subTest(artifact=artifact):
                self.assertEqual(artifacts.artifact_identity(artifact), "")


class MergeArtifactsTests(unittest.TestCase):
    def test_duplicates_inside_existing_are_left_alone(self) -> None:
        merged, added = artifacts.merge_artifacts(
            existing=[{"id": "a"}, {"id": "a"}], additions=[]
        )
        self.assertEqual([item["id"] for item in merged], ["a", "a"])
        self.assertEqual(added, [])

    def test_additions_are_deduplicated_against_existing_and_each_other(self) -> None:
        merged, added = artifacts.merge_artifacts(
            existing=[{"id": "a"}], additions=[{"id": "a"}, {"id": "b"}, {"id": "b"}, {"id": "c"}]
        )
        self.assertEqual([item["id"] for item in added], ["b", "c"])
        self.assertEqual([item["id"] for item in merged], ["a", "b", "c"])

    def test_undedupable_entries_are_dropped(self) -> None:
        merged, added = artifacts.merge_artifacts(
            existing=[], additions=[{}, {"title": ""}, {"id": "a"}]
        )
        self.assertEqual([item["id"] for item in added], ["a"])

    def test_non_mapping_entries_are_skipped_on_both_sides(self) -> None:
        merged, added = artifacts.merge_artifacts(
            existing=["x", {"id": "a"}], additions=[None, {"id": "b"}]
        )
        self.assertEqual([item["id"] for item in merged], ["a", "b"])
        self.assertEqual([item["id"] for item in added], ["b"])

    def test_existing_entries_are_copied_not_aliased(self) -> None:
        original = {"id": "a", "title": "t"}
        merged, _ = artifacts.merge_artifacts(existing=[original], additions=[])
        self.assertIsNot(merged[0], original)
        self.assertEqual(merged[0], original)

    def test_titles_can_act_as_the_identity(self) -> None:
        merged, added = artifacts.merge_artifacts(
            existing=[{"title": "t", "kind": "docx"}],
            additions=[{"title": "t", "kind": "docx"}, {"title": "t", "kind": "png"}],
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["kind"], "png")
        self.assertEqual(len(merged), 2)

    def test_an_added_entry_is_the_same_object_in_both_results(self) -> None:
        """既有语义：``added`` 与 ``merged`` 里是同一个字典，不是各存一份拷贝。"""
        merged, added = artifacts.merge_artifacts(
            existing=[], additions=[{"id": "a", "title": "t"}]
        )
        self.assertIs(merged[0], added[0])

        added[0]["title"] = "changed through added"
        self.assertEqual(merged[0]["title"], "changed through added")

        merged[0]["status"] = "changed through merged"
        self.assertEqual(added[0]["status"], "changed through merged")

    def test_existing_entries_are_not_shared_with_the_input_list(self) -> None:
        merged, added = artifacts.merge_artifacts(existing=[{"id": "a"}], additions=[])
        self.assertEqual(merged[0], {"id": "a"})
        self.assertEqual(added, [])


class CompactWorkspaceTests(unittest.TestCase):
    def test_the_surface_is_four_scalars(self) -> None:
        body = artifacts.compact_workspace_for_event(
            {"task_id": "t-1", "status": "running", "normalized_goal": "goal", "artifacts": [1, 2]}
        )
        self.assertEqual(
            body,
            {
                "task_id": "t-1",
                "status": "running",
                "normalized_goal": "goal",
                "artifact_count": 2,
            },
        )

    def test_values_are_reported_as_given_without_stripping(self) -> None:
        body = artifacts.compact_workspace_for_event(
            {"task_id": "  t-1  ", "status": " ok ", "normalized_goal": "  goal  "}
        )
        self.assertEqual(body["task_id"], "  t-1  ")
        self.assertEqual(body["status"], " ok ")
        self.assertEqual(body["normalized_goal"], "  goal  ")

    def test_the_goal_is_truncated_to_200_characters(self) -> None:
        body = artifacts.compact_workspace_for_event({"normalized_goal": "g" * 250})
        self.assertEqual(len(body["normalized_goal"]), 200)

    def test_missing_values_become_empty_strings(self) -> None:
        self.assertEqual(
            artifacts.compact_workspace_for_event({}),
            {"task_id": "", "status": "", "normalized_goal": "", "artifact_count": 0},
        )

    def test_artifact_count_is_a_length_not_the_list(self) -> None:
        for value, expected in ((None, 0), ([], 0), ([{"id": "a"}], 1), ([{"id": "a"}] * 5, 5)):
            with self.subTest(value=value):
                body = artifacts.compact_workspace_for_event({"artifacts": value})
                self.assertEqual(body["artifact_count"], expected)
                self.assertIsInstance(body["artifact_count"], int)

    def test_falsy_inputs_are_reported_as_empty_strings(self) -> None:
        body = artifacts.compact_workspace_for_event(
            {"task_id": None, "status": 0, "normalized_goal": False}
        )
        self.assertEqual(body["task_id"], "")
        self.assertEqual(body["status"], "")
        self.assertEqual(body["normalized_goal"], "")


class ArtifactsExportTests(unittest.TestCase):
    def test_the_export_surface_is_exactly_six_functions(self) -> None:
        self.assertEqual(
            set(artifacts.__all__),
            {
                "artifact_from_attachment_item",
                "artifact_from_generated_file",
                "artifact_identity",
                "compact_workspace_for_event",
                "extract_artifacts_from_tool_events",
                "merge_artifacts",
            },
        )

    def test_every_exported_name_resolves(self) -> None:
        for name in artifacts.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(artifacts, name))


# --------------------------------------------------------------------------- #
# §5 tool_invocation —— 工具调用协议
# --------------------------------------------------------------------------- #


class InvocationRecordTests(unittest.TestCase):
    def test_the_protocol_constants_are_frozen(self) -> None:
        self.assertEqual(invocation.LEGACY_JSON, "legacy_json")
        self.assertEqual(invocation.NATIVE_OPENAI, "native_openai")
        self.assertEqual(invocation.NATIVE_ANTHROPIC, "native_anthropic")
        self.assertEqual(invocation.TOOL_SOURCE_FIELD, "_tool_source")
        self.assertEqual(invocation.TOOL_INVOCATION_ID_FIELD, "_tool_invocation_id")
        self.assertEqual(invocation.NATIVE_TOOL_CALL_FIELD, "_native_tool_call")

    def test_the_invocation_defaults(self) -> None:
        item = invocation.ToolInvocation("search")
        self.assertEqual(item.arguments, {})
        self.assertEqual(item.source, invocation.LEGACY_JSON)

    def test_an_absent_id_is_generated(self) -> None:
        self.assertRegex(invocation.ToolInvocation("search").id, r"^call_[0-9a-f]{16}$")

    def test_a_blank_id_is_replaced(self) -> None:
        self.assertRegex(invocation.ToolInvocation("search", id="   ").id, r"^call_[0-9a-f]{16}$")

    def test_a_supplied_id_is_kept_verbatim(self) -> None:
        self.assertEqual(invocation.ToolInvocation("search", id="call_keepme").id, "call_keepme")

    def test_the_records_are_mutable(self) -> None:
        item = invocation.ToolInvocation("search")
        item.name = "renamed"
        self.assertEqual(item.name, "renamed")

        result = invocation.ValidationResult.success()
        result.ok = False
        self.assertIs(result.ok, False)

    def test_collection_defaults_are_not_shared_between_instances(self) -> None:
        first, second = invocation.ToolInvocation("a"), invocation.ToolInvocation("b")
        first.arguments["x"] = 1
        self.assertEqual(second.arguments, {})

        left = invocation.ToolResultEnvelope("i-1", "ok", "feedback")
        right = invocation.ToolResultEnvelope("i-2", "ok", "feedback")
        left.events.append({"type": "progress"})
        self.assertEqual(right.events, [])

    def test_the_envelope_requires_its_three_mandatory_fields(self) -> None:
        with self.assertRaises(TypeError):
            invocation.ToolResultEnvelope("i-1")  # type: ignore[call-arg]
        envelope = invocation.ToolResultEnvelope("i-1", "ok", "feedback")
        self.assertIsNone(envelope.data)

    def test_validation_success_and_failure_shapes(self) -> None:
        ok = invocation.ValidationResult.success()
        self.assertIs(ok.ok, True)
        self.assertEqual(ok.message, "")
        self.assertEqual(ok.code, "")

        bad = invocation.ValidationResult.fail("bad_code", "explanation")
        self.assertIs(bad.ok, False)
        self.assertEqual(bad.code, "bad_code")
        self.assertEqual(bad.message, "explanation")

    def test_fail_coerces_absent_values_to_empty_strings(self) -> None:
        bad = invocation.ValidationResult.fail(None, None)  # type: ignore[arg-type]
        self.assertEqual(bad.code, "")
        self.assertEqual(bad.message, "")


class LegacyToolCallConversionTests(unittest.TestCase):
    def test_a_non_mapping_input_is_rejected(self) -> None:
        for bad in (None, [], "text", 5):
            with self.subTest(bad=bad):
                self.assertIsNone(invocation.legacy_tool_call_to_invocation(bad))

    def test_a_missing_tool_name_is_rejected(self) -> None:
        for bad in ({}, {"type": ""}, {"type": "   "}, {"argument": 1}):
            with self.subTest(bad=bad):
                self.assertIsNone(invocation.legacy_tool_call_to_invocation(bad))

    def test_the_name_is_stripped(self) -> None:
        item = invocation.legacy_tool_call_to_invocation({"type": "  search  "})
        self.assertEqual(item.name, "search")

    def test_type_and_metadata_keys_stay_out_of_the_arguments(self) -> None:
        item = invocation.legacy_tool_call_to_invocation(
            {
                "type": "search",
                "query": "x",
                "_tool_source": "native_openai",
                "_tool_invocation_id": "call_1",
                "_tool_custom": 1,
                "_toolish": 2,
            }
        )
        self.assertEqual(item.arguments, {"query": "x", "_toolish": 2})

    def test_embedded_metadata_wins_over_the_caller_supplied_values(self) -> None:
        item = invocation.legacy_tool_call_to_invocation(
            {"type": "t", "_tool_source": "native_anthropic", "_tool_invocation_id": "call_embedded"},
            source="native_openai",
            invocation_id="call_argument",
        )
        self.assertEqual(item.source, "native_anthropic")
        self.assertEqual(item.id, "call_embedded")

    def test_caller_values_apply_when_the_payload_is_silent(self) -> None:
        item = invocation.legacy_tool_call_to_invocation(
            {"type": "t"}, source="native_openai", invocation_id="call_argument"
        )
        self.assertEqual(item.source, "native_openai")
        self.assertEqual(item.id, "call_argument")

    def test_an_id_is_generated_when_nobody_supplied_one(self) -> None:
        item = invocation.legacy_tool_call_to_invocation({"type": "t"})
        self.assertRegex(item.id, r"^call_[0-9a-f]{16}$")
        self.assertEqual(item.source, invocation.LEGACY_JSON)

    def test_the_conversion_options_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            invocation.legacy_tool_call_to_invocation({"type": "t"}, "native_openai")  # type: ignore[misc]


class InvocationToLegacyTests(unittest.TestCase):
    def test_the_base_payload_is_the_type_plus_the_arguments(self) -> None:
        item = invocation.ToolInvocation("search", {"query": "x"})
        self.assertEqual(
            invocation.invocation_to_legacy_tool_call(item), {"type": "search", "query": "x"}
        )

    def test_an_argument_named_type_shadows_the_tool_name(self) -> None:
        item = invocation.ToolInvocation("search", {"type": "shadow"})
        self.assertEqual(invocation.invocation_to_legacy_tool_call(item)["type"], "shadow")

    def test_metadata_is_omitted_by_default(self) -> None:
        item = invocation.ToolInvocation("t", source="native_openai")
        self.assertEqual(set(invocation.invocation_to_legacy_tool_call(item)), {"type"})

    def test_metadata_is_omitted_for_legacy_sources_even_when_requested(self) -> None:
        item = invocation.ToolInvocation("t")
        self.assertEqual(
            set(invocation.invocation_to_legacy_tool_call(item, include_metadata=True)), {"type"}
        )

    def test_metadata_is_added_for_native_sources_when_requested(self) -> None:
        item = invocation.ToolInvocation("t", source="native_anthropic", id="call_9")
        payload = invocation.invocation_to_legacy_tool_call(item, include_metadata=True)
        self.assertEqual(payload["_tool_source"], "native_anthropic")
        self.assertEqual(payload["_tool_invocation_id"], "call_9")

    def test_the_metadata_flag_is_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            invocation.invocation_to_legacy_tool_call(invocation.ToolInvocation("t"), True)  # type: ignore[misc]


class RoundTripTests(unittest.TestCase):
    CASES = (
        ({"type": "t", "a": 1}, {"type": "t", "a": 1}),
        ({"type": "t", "a": 1, "_tool_source": "native_openai"}, {"type": "t", "a": 1}),
        ({"type": "t", "_tool_invocation_id": "call_x", "b": 2}, {"type": "t", "b": 2}),
        ({"type": "t"}, {"type": "t"}),
        ({"a": 1}, None),
        ("not a mapping", None),
        (None, None),
        ({"type": "  "}, None),
    )

    def test_the_documented_cases(self) -> None:
        for given, expected in self.CASES:
            with self.subTest(given=given):
                self.assertEqual(invocation.round_trip_legacy_tool_call(given), expected)

    def test_metadata_is_always_normalized_away(self) -> None:
        given = {
            "type": "t",
            "a": 1,
            "_tool_source": "native_anthropic",
            "_tool_invocation_id": "call_1",
        }
        result = invocation.round_trip_legacy_tool_call(given)
        self.assertEqual(result, {"type": "t", "a": 1})
        for key in (invocation.TOOL_SOURCE_FIELD, invocation.TOOL_INVOCATION_ID_FIELD):
            self.assertNotIn(key, result)

    def test_the_metadata_flag_is_not_available_here(self) -> None:
        with self.assertRaises(TypeError):
            invocation.round_trip_legacy_tool_call({"type": "t"}, True)  # type: ignore[misc]


class InvocationExportTests(unittest.TestCase):
    def test_the_export_surface_is_twelve_distinct_names(self) -> None:
        self.assertEqual(len(invocation.__all__), 12)
        self.assertEqual(len(set(invocation.__all__)), 12)

    def test_every_exported_name_resolves(self) -> None:
        for name in invocation.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(invocation, name))

    def test_all_six_constants_are_exported(self) -> None:
        self.assertTrue(
            {
                "LEGACY_JSON",
                "NATIVE_OPENAI",
                "NATIVE_ANTHROPIC",
                "TOOL_SOURCE_FIELD",
                "TOOL_INVOCATION_ID_FIELD",
                "NATIVE_TOOL_CALL_FIELD",
            }.issubset(set(invocation.__all__))
        )


# --------------------------------------------------------------------------- #
# 边界：本批新文件与旧项目完全脱钩
# --------------------------------------------------------------------------- #


class C1_2BoundaryTests(unittest.TestCase):
    """B1 边界在新代码上的延续，同时是"表达层独立"的直接证据。"""

    C1_2_FILES = (
        Path(correlation.__file__),
        Path(sessions.__file__),
        Path(artifacts.__file__),
        Path(invocation.__file__),
        SHINKU_PKG / "tasks" / "__init__.py",
        SHINKU_PKG / "tools" / "__init__.py",
    )

    #: 旧项目的包名与代号。
    FOREIGN_TOKENS = ("companion_v01", "code_shared")

    #: 旧项目里这些模块曾经依赖、但本批不得再提及的实现符号。
    FOREIGN_SYMBOLS = (
        "normalize_character_pack_id",
        "build_desktop_pet_error_payload",
        "desktop_pet_contract",
    )

    #: 旧模块的内部命名。它们出现在新文件里就等于表达层被复制。
    LEGACY_INTERNAL_NAMES = (
        "MIN_ID_LENGTH",
        "MAX_ID_LENGTH",
        "ID_ALPHABET",
        "_is_well_formed",
        "_new_correlation_id",
        "task_workspace_",
        "pack_from_payload",
        "ResolveQueryIdentity",
        "_observe",
    )

    def _sources(self) -> list[tuple[Path, str]]:
        for path in self.C1_2_FILES:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"expected {path} to exist")
            yield path, path.read_text(encoding="utf-8")

    def test_no_foreign_project_is_mentioned(self) -> None:
        for path, text in self._sources():
            for token in self.FOREIGN_TOKENS:
                with self.subTest(module=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_no_symbol_from_the_modules_this_batch_replaces_is_mentioned(self) -> None:
        for path, text in self._sources():
            for symbol in self.FOREIGN_SYMBOLS:
                with self.subTest(module=path.name, symbol=symbol):
                    self.assertNotIn(symbol, text)

    def test_no_legacy_internal_name_survives(self) -> None:
        for path, text in self._sources():
            for legacy in self.LEGACY_INTERNAL_NAMES:
                with self.subTest(module=path.name, legacy=legacy):
                    self.assertNotIn(legacy, text)

    def test_no_legacy_environment_prefix(self) -> None:
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("COMPANION_", text)

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        # 本模块顶部已导入四个新模块，因此这里的检查是有意义的。
        for forbidden in ("companion_v01", "code_shared"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_two_new_package_markers_re_export_nothing(self) -> None:
        from shinku import tasks, tools

        self.assertEqual(tasks.__all__, [])
        self.assertEqual(tools.__all__, [])


if __name__ == "__main__":
    unittest.main()

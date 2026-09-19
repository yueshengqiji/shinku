"""C2-4 契约测试：模型服务控制中心路由。

对象是 ``shinku.api.model_services`` 的**外部可观察行为**
（契约文档 ``docs/contracts/c2_4_model_services.md``）。路由是真的挂到 FastAPI app 上跑的，
但：注册表读写一律落临时目录，模型探测用注入的假件，时钟换假件
（``lastModelProbeAt`` 要可比），engine / 度量 / 日志都是记录用的假件。
全程不发起任何真实网络请求。

边界类（``C2_4BoundaryTests``）沿用每批都有的口径：外来标识 / 旧私有名 / 旧整句 /
旧环境变量前缀不得复现；本批另钉住两件"有意不搬"的东西——
那 4 个单服务端点不在路由表里，且本模块只 import 本仓模块 + 标准库 + fastapi。
"""

from __future__ import annotations

import ast
import tempfile
import tokenize as py_tokenize
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shinku.api import model_services as routes
from shinku.providers.catalog import ModelServiceConfigStore
from shinku.providers.service import ModelServiceSettings

SHINKU_PKG = Path(routes.__file__).resolve().parents[1]

#: 四个路由（方法 + 路径）。顺序自由，集合必须正好是这四条。
EXPECTED_ROUTES = {
    ("GET", "/control-center/model-services"),
    ("POST", "/control-center/model-services/{provider_id}/config"),
    ("POST", "/control-center/model-services/{provider_id}/models"),
    ("POST", "/control-center/model-services/select"),
}

PAGE_PATH = "/control-center/model-services"
CONFIG_PATH = "/control-center/model-services/glm/config"
MODELS_PATH = "/control-center/model-services/glm/models"
SELECT_PATH = "/control-center/model-services/select"


class Clock:
    """假时钟：``time()`` 固定，``perf_counter()`` 恒 0（度量时长不进断言）。"""

    FIXED = 1_700_000_000

    def time(self) -> int:
        return self.FIXED

    def perf_counter(self) -> float:
        return 0.0


class Engine:
    """记录重载次数，并把次数塞进返回值——"到底重载了几次"也能被断言。"""

    def __init__(self) -> None:
        self.count = 0

    def reload_model_services(self) -> dict:
        self.count += 1
        return {"status": "reloaded", "count": self.count}


class Probe:
    """模型清单探测假件；``boom`` 非空时抛错。"""

    def __init__(self, models: list[str] | None = None, boom: str = "") -> None:
        self.models = list(models or [])
        self.boom = boom

    def __call__(self, settings) -> list[str]:
        if self.boom:
            raise RuntimeError(self.boom)
        return list(self.models)


class Recorder:
    """度量与结构化日志共用的记录件：``observe_request`` 记度量，``__call__`` 记日志。"""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, bool]] = []
        self.events: list[tuple[str, tuple[str, ...]]] = []

    def observe_request(self, name: str, *, duration_ms: float, ok: bool) -> None:
        self.metrics.append((name, ok))

    def __call__(self, event: str, **fields) -> None:
        self.events.append((event, tuple(sorted(fields))))


def config_module() -> SimpleNamespace:
    """环境变量视角的项目配置。四个通道要齐，少一个就折不出"当前生效的是哪家"。"""

    return SimpleNamespace(
        TEXT_API_KEY="env-secret",
        TEXT_BASE_URL="https://api.deepseek.com/v1",
        TEXT_MODEL_NAME="deepseek-chat",
        TEXT_API_PROTOCOL="openai",
        AUX_API_KEY="env-secret",
        AUX_BASE_URL="https://api.deepseek.com/v1",
        AUX_MODEL_NAME="deepseek-chat",
        AUX_API_PROTOCOL="openai",
        CHAT_API_KEY="env-secret",
        CHAT_BASE_URL="https://api.deepseek.com/v1",
        CHAT_MODEL_NAME="deepseek-chat",
        CHAT_API_PROTOCOL="openai",
        VISION_API_KEY="",
        VISION_BASE_URL="",
        VISION_MODEL_NAME="",
        VISION_API_PROTOCOL="openai",
    )


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


class Harness(unittest.TestCase):
    """搭一个只有本批路由的 app。每个用例都拿一套干净的临时目录。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ModelServiceConfigStore(Path(self.tmp.name) / "model_service.json")
        self.config = config_module()
        self.engine = Engine()
        self.probe = Probe(["glm-4.5-air", "glm-4.5"])
        self.recorder = Recorder()
        self.client = self.build()

    def build(self, **overrides) -> TestClient:
        kwargs = dict(
            store=self.store,
            config_module=self.config,
            engine=self.engine,
            runtime_metrics=self.recorder,
            log_event=self.recorder,
            model_probe=self.probe,
        )
        kwargs.update(overrides)
        app = FastAPI()
        app.include_router(routes.build_model_services_router(**kwargs))
        return TestClient(app)

    def remote(self, host: str = "192.0.2.10") -> TestClient:
        return TestClient(self.client.app, client=(host, 12345))

    def save_glm(self, **extra) -> dict:
        body = {"apiKey": "glm-secret", "chatModel": "glm-4.5-air"}
        body.update(extra)
        return self.client.post(CONFIG_PATH, json=body).json()


# --------------------------------------------------------------------------- #
# 1. 路由面
# --------------------------------------------------------------------------- #


class RouteSurfaceTests(Harness):
    def test_the_router_carries_exactly_four_routes(self) -> None:
        router = routes.build_model_services_router(
            store=self.store,
            config_module=self.config,
            engine=self.engine,
            model_probe=self.probe,
        )
        surface = {
            (method, route.path)
            for route in router.routes
            for method in getattr(route, "methods", ())
        }
        self.assertEqual(surface, EXPECTED_ROUTES)

    def test_the_legacy_single_service_routes_are_gone(self) -> None:
        """单数那四个端点不搬——真红仓内零消费者（契约决定 2）。"""

        router = routes.build_model_services_router(
            store=self.store,
            config_module=self.config,
            engine=self.engine,
            model_probe=self.probe,
        )
        paths = {route.path for route in router.routes}
        for stale in (
            "/control-center/model-service",
            "/control-center/model-service/models",
            "/control-center/model-service/test",
        ):
            with self.subTest(path=stale):
                self.assertNotIn(stale, paths)

    def test_the_page_route_is_the_only_one_open_to_anyone(self) -> None:
        """GET 不设本地闸门，四个 POST 都设——逐个试一遍。"""

        client = self.remote()
        self.assertEqual(client.get(PAGE_PATH).status_code, 200)
        for path in (CONFIG_PATH, MODELS_PATH, SELECT_PATH):
            with self.subTest(path=path):
                self.assertEqual(client.post(path, json={}).status_code, 403)

    def test_every_response_carries_the_no_store_policy(self) -> None:
        for response in (
            self.client.get(PAGE_PATH),
            self.client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m"}),
            self.remote().post(CONFIG_PATH, json={}),
        ):
            with self.subTest(url=response.url.path):
                self.assertEqual(response.headers["cache-control"], "no-store")


# --------------------------------------------------------------------------- #
# 2. 本地请求闸门
# --------------------------------------------------------------------------- #


class LocalCallerGateTests(Harness):
    def test_a_remote_caller_is_rejected_with_the_shared_envelope(self) -> None:
        for path in (CONFIG_PATH, MODELS_PATH, SELECT_PATH):
            with self.subTest(path=path):
                response = self.remote().post(path, json={})
                self.assertEqual(response.status_code, 403)
                self.assertEqual(
                    response.json(),
                    {"ok": False, "status": "forbidden", "reason": "local_request_required"},
                )

    def test_every_addressable_loopback_host_is_accepted(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost", "TESTCLIENT", " Localhost "):
            with self.subTest(host=host):
                client = TestClient(self.client.app, client=(host, 12345))
                response = client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m"})
                self.assertNotEqual(response.status_code, 403)

    def test_a_request_without_a_client_is_treated_as_remote(self) -> None:
        """取不到来源一律当"不是本机"——宁可挡住，不放行。"""

        request = mock.Mock(spec=[])
        self.assertFalse(routes._caller_is_local(request))

    def test_the_test_client_default_host_is_on_the_allowlist(self) -> None:
        """这一条掉了，所有测试都会被闸门挡在外面。"""

        self.assertIn("testclient", routes._LOCAL_CALLER_HOSTS)


# --------------------------------------------------------------------------- #
# 3. 整页快照
# --------------------------------------------------------------------------- #


class ReadPageTests(Harness):
    def test_the_page_lists_the_whole_catalog_without_keys(self) -> None:
        payload = self.client.get(PAGE_PATH).json()
        self.assertTrue(payload["ok"])
        self.assertIn("providers", payload)
        self.assertIn("activeProviderId", payload)
        self.assertNotIn("glm-secret", str(payload))

    def test_the_page_does_not_require_a_local_caller(self) -> None:
        self.assertEqual(self.remote().get(PAGE_PATH).status_code, 200)

    def test_a_broken_snapshot_becomes_a_500_with_a_redacted_reason(self) -> None:
        with mock.patch.object(
            routes, "public_model_services_snapshot", side_effect=RuntimeError("boom")
        ):
            response = self.client.get(PAGE_PATH)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"ok": False, "status": "invalid_config", "reason": "boom"},
        )

    def test_the_failure_is_recorded_as_a_failed_metric(self) -> None:
        self.recorder.metrics.clear()
        with mock.patch.object(
            routes, "public_model_services_snapshot", side_effect=RuntimeError("boom")
        ):
            self.client.get(PAGE_PATH)
        self.assertEqual(self.recorder.metrics, [("model_services.read", False)])

    def test_a_successful_read_is_recorded_as_ok(self) -> None:
        self.recorder.metrics.clear()
        self.client.get(PAGE_PATH)
        self.assertEqual(self.recorder.metrics, [("model_services.read", True)])


# --------------------------------------------------------------------------- #
# 4. 保存某家配置
# --------------------------------------------------------------------------- #


class SaveConfigTests(Harness):
    def test_saving_without_activating_does_not_reload_the_engine(self) -> None:
        payload = self.save_glm()
        self.assertTrue(payload["ok"])
        self.assertEqual(self.engine.count, 0)

    def test_saving_without_activating_leaves_runtime_empty(self) -> None:
        """config 侧**不兜底**——`runtime` 就是 `null`（与 select 侧不同）。"""

        with mock.patch.object(routes, "time", Clock()):
            payload = self.save_glm()
        self.assertTrue(payload["refresh"])
        self.assertIsNone(payload["runtime"])

    def test_activating_reloads_the_engine_and_reports_it(self) -> None:
        with mock.patch.object(routes, "time", Clock()):
            payload = self.save_glm(activate=True)
        self.assertEqual(self.engine.count, 1)
        self.assertEqual(payload["runtime"], {"status": "reloaded", "count": 1})

    def test_activating_writes_the_provider_into_the_project_config(self) -> None:
        self.save_glm(activate=True)
        self.assertEqual(self.config.CHAT_BASE_URL, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(self.config.CHAT_MODEL_NAME, "glm-4.5-air")

    def test_the_path_provider_id_wins_over_the_body_one(self) -> None:
        """`providerId` 是**覆盖**进载荷的，不是 `setdefault`。"""

        self.client.post(
            "/control-center/model-services/glm/config",
            json={"providerId": "deepseek", "apiKey": "k", "chatModel": "m"},
        )
        self.assertIsNotNone(self.store.get_provider("glm"))

    def test_the_key_is_inherited_from_the_saved_provider(self) -> None:
        self.save_glm()
        self.client.post(
            CONFIG_PATH, json={"chatModel": "glm-4.5"}   # 不带 apiKey
        )
        self.assertEqual(self.store.get_provider("glm").api_key, "glm-secret")

    def test_switching_provider_does_not_carry_the_old_key_over(self) -> None:
        """换供应商时密钥不继承——否则 A 家的密钥会糊到 B 家身上。"""

        self.save_glm()
        self.client.post(
            "/control-center/model-services/deepseek/config",
            json={"chatModel": "deepseek-chat"},
        )
        saved = self.store.get_provider("deepseek")
        self.assertIsNotNone(saved)
        self.assertNotEqual(saved.api_key, "glm-secret")

    def test_activating_without_a_model_is_refused(self) -> None:
        response = self.client.post(CONFIG_PATH, json={"apiKey": "k", "activate": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"ok": False, "status": "invalid_config", "reason": "model_service_model_missing"},
        )
        self.assertEqual(self.engine.count, 0)

    def test_a_blank_model_counts_as_missing(self) -> None:
        response = self.client.post(
            CONFIG_PATH, json={"apiKey": "k", "chatModel": "   ", "activate": True}
        )
        self.assertFalse(response.json()["ok"])
        self.assertEqual(response.json()["reason"], "model_service_model_missing")

    def test_the_saved_page_is_returned_with_a_refresh_flag(self) -> None:
        with mock.patch.object(routes, "time", Clock()):
            payload = self.save_glm()
        self.assertTrue(payload["refresh"])
        self.assertEqual(payload["providerId"], "glm")

    def test_the_save_event_carries_the_provider_shape(self) -> None:
        self.recorder.events.clear()
        self.save_glm(activate=True)
        self.assertEqual(
            self.recorder.events,
            [
                (
                    "model_services_save",
                    ("active", "chat_model", "protocol", "provider_id", "status"),
                )
            ],
        )

    def test_a_failed_save_is_logged_with_the_path_provider_id(self) -> None:
        self.recorder.events.clear()
        self.client.post(CONFIG_PATH, json={"activate": True})
        self.assertEqual(
            self.recorder.events,
            [("model_services_save", ("provider_id", "status"))],
        )
        self.assertEqual(self.recorder.metrics, [("model_services.save", False)])


# --------------------------------------------------------------------------- #
# 5. 真值词
# --------------------------------------------------------------------------- #


class TruthyWordTests(Harness):
    def test_affirmative_words_activate(self) -> None:
        """整数 `1` 也算——`str(value or "")` 之后就是 `"1"`。"""

        for word in ("true", "True", " TRUE ", "1", "yes", "on", "enabled", 1):
            with self.subTest(word=word):
                client, engine = self._fresh()
                client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m",
                                               "activate": word})
                self.assertEqual(engine.count, 1)

    def test_other_words_do_not_activate(self) -> None:
        """整数 `0` 不算——`str(0 or "")` 是空串。"""

        for word in ("false", "0", "no", "off", "", "   ", "maybe", None, 0, 2):
            with self.subTest(word=word):
                client, engine = self._fresh()
                client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m",
                                               "activate": word})
                self.assertEqual(engine.count, 0)

    def test_a_real_boolean_is_passed_through_untouched(self) -> None:
        client, engine = self._fresh()
        client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m", "activate": True})
        self.assertEqual(engine.count, 1)

    def test_setactive_is_the_second_key_name(self) -> None:
        client, engine = self._fresh()
        client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m", "setActive": "yes"})
        self.assertEqual(engine.count, 1)

    def test_activating_a_provider_marks_it_active_in_the_registry(self) -> None:
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m", "activate": True})
        self.assertEqual(self.store.load_registry()["activeProviderId"], "glm")
        # 用 activate=True 把激活项切到另一家；set_active 被写死 False 时这次保存
        # 不会改写 activeProviderId，注册表仍停在 glm（load_registry 的兜底会把它填回
        # 唯一供应商，单供应商场景看不出差异——必须两家对照）。
        client.post(
            "/control-center/model-services/deepseek/config",
            json={"apiKey": "k", "chatModel": "m", "activate": True},
        )
        self.assertEqual(self.store.load_registry()["activeProviderId"], "deepseek")

    def test_activate_wins_when_both_keys_are_present(self) -> None:
        client, engine = self._fresh()
        client.post(
            CONFIG_PATH,
            json={"apiKey": "k", "chatModel": "m", "setActive": True, "activate": False},
        )
        self.assertEqual(engine.count, 0)

    def test_the_word_table_is_exactly_the_documented_one(self) -> None:
        self.assertEqual(
            routes._AFFIRMATIVE_WORDS, {"1", "true", "yes", "on", "enabled"}
        )

    def _fresh(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ModelServiceConfigStore(Path(tmp.name) / "model_service.json")
        engine = Engine()
        app = FastAPI()
        app.include_router(
            routes.build_model_services_router(
                store=store,
                config_module=config_module(),
                engine=engine,
                model_probe=self.probe,
            )
        )
        return TestClient(app), engine


# --------------------------------------------------------------------------- #
# 6. 模型清单刷新
# --------------------------------------------------------------------------- #


class ModelRefreshTests(Harness):
    def test_an_unknown_provider_asks_for_a_save_first(self) -> None:
        response = self.client.post(MODELS_PATH, json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "status": "provider_not_configured",
                "reason": "先保存该供应方配置，再刷新模型列表。",
            },
        )

    def test_the_discovered_ids_are_deduped_and_sorted_case_insensitively(self) -> None:
        self.probe = Probe(["Beta", "alpha", "alpha", "  ", "Gamma"])
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        with mock.patch.object(routes, "time", Clock()):
            payload = client.post(MODELS_PATH, json={}).json()
        self.assertEqual(payload["models"], ["alpha", "Beta", "Gamma"])
        self.assertEqual(payload["count"], 3)

    def test_the_success_envelope_names_the_provider_and_the_stamp(self) -> None:
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        with mock.patch.object(routes, "time", Clock()):
            payload = client.post(MODELS_PATH, json={}).json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["status"], "available")
        self.assertEqual(payload["providerId"], "glm")
        self.assertEqual(payload["lastModelProbeAt"], Clock.FIXED)

    def test_stored_fields_fill_the_blanks_of_the_refresh_payload(self) -> None:
        """密钥永远取已存的：刷新不改变"在用哪家服务"。"""

        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        with mock.patch.object(routes, "settings_from_mapping", wraps=routes.settings_from_mapping) as spy:
            client.post(MODELS_PATH, json={})
        payload = spy.call_args[0][0]
        # 密钥**不进载荷**——它只能经由 `existing_api_key` 从已存的那份继承。
        self.assertNotIn("apiKey", payload)
        self.assertEqual(spy.call_args.kwargs["existing_api_key"], "glm-secret")
        self.assertEqual(payload["providerId"], "glm")
        self.assertEqual(payload["baseUrl"], "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(payload["chatModel"], "glm-4.5-air")
        self.assertEqual(payload["timeoutSeconds"], 120)

    def test_a_missing_endpoint_is_reported_as_a_failed_request(self) -> None:
        """防御性分支：落盘的 provider 也可能导致端点不齐（`endpoint_configured` 为假）。"""

        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        hollow = ModelServiceSettings(
            provider_id="glm",
            protocol="openai",
            base_url="",
            api_key="glm-secret",
            chat_model="m",
            use_for_vision=False,
            vision_model="",
            timeout_seconds=120,
        )
        with mock.patch.object(routes, "settings_from_mapping", return_value=hollow):
            response = client.post(MODELS_PATH, json={})
        self.assertEqual(
            response.json(),
            {"ok": False, "status": "request_failed", "reason": "model_service_config_incomplete"},
        )

    def test_the_api_key_is_redacted_out_of_the_probe_error(self) -> None:
        self.probe = Probe([], boom="boom glm-secret here")
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        payload = client.post(MODELS_PATH, json={}).json()
        self.assertEqual(payload["status"], "request_failed")
        self.assertEqual(payload["reason"], "boom <redacted> here")
        self.assertNotIn("glm-secret", payload["reason"])

    def test_a_failed_probe_is_recorded_in_the_registry(self) -> None:
        self.probe = Probe([], boom="boom")
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        client.post(MODELS_PATH, json={})
        entry = self.store.load_registry()["providers"]["glm"]
        self.assertEqual(entry["modelProbeError"], "boom")
        self.assertEqual(entry["discoveredModels"], [])

    def test_a_successful_probe_is_recorded_in_the_registry(self) -> None:
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        with mock.patch.object(routes, "time", Clock()):
            client.post(MODELS_PATH, json={})
        entry = self.store.load_registry()["providers"]["glm"]
        self.assertEqual(entry["discoveredModels"], ["glm-4.5", "glm-4.5-air"])
        self.assertEqual(entry["lastModelProbeAt"], Clock.FIXED)

    def test_a_failed_probe_still_writes_even_if_the_registry_is_broken(self) -> None:
        """失败路径里那次落盘自己抛错也要吞掉——不能带崩响应。"""

        self.probe = Probe([], boom="boom")
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        with mock.patch.object(
            self.store, "save_model_probe", side_effect=RuntimeError("disk gone")
        ):
            response = client.post(MODELS_PATH, json={})
        self.assertEqual(response.json()["status"], "request_failed")

    def test_the_models_event_counts_the_ids(self) -> None:
        client = self.build()
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        self.recorder.events.clear()
        with mock.patch.object(routes, "time", Clock()):
            client.post(MODELS_PATH, json={})
        self.assertEqual(
            self.recorder.events,
            [("model_services_models", ("count", "provider_id", "status"))],
        )


# --------------------------------------------------------------------------- #
# 7. 切换激活项
# --------------------------------------------------------------------------- #


class SelectTests(Harness):
    def setUp(self) -> None:
        super().setUp()
        self.save_glm()

    def test_selecting_reloads_and_reports_the_active_pair(self) -> None:
        payload = self.client.post(
            SELECT_PATH, json={"providerId": "glm", "modelId": "glm-4.5"}
        ).json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["activeProviderId"], "glm")
        self.assertEqual(payload["activeModel"], "glm-4.5")
        self.assertEqual(self.engine.count, 1)

    def test_a_missing_reload_result_falls_back_to_a_reloaded_marker(self) -> None:
        """select 侧**有**兜底（config 侧没有）——两边不许统一。"""

        class Quiet:
            def reload_model_services(self) -> dict:
                return {}

        client = self.build(engine=Quiet())
        client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "glm-4.5-air"})
        payload = client.post(SELECT_PATH, json={"providerId": "glm", "modelId": "m"}).json()
        self.assertEqual(payload["runtime"], {"status": "reloaded"})

    def test_the_underscore_spelling_is_accepted(self) -> None:
        payload = self.client.post(
            SELECT_PATH, json={"provider_id": "glm", "model_id": "glm-4.5"}
        ).json()
        self.assertEqual(payload["activeModel"], "glm-4.5")

    def test_chatmodel_and_model_are_both_fallback_keys_for_the_model(self) -> None:
        for key in ("modelId", "chatModel", "model"):
            with self.subTest(key=key):
                client = self.build()
                client.post(CONFIG_PATH, json={"apiKey": "glm-secret", "chatModel": "m"})
                payload = client.post(SELECT_PATH, json={"providerId": "glm", key: "glm-4.5"}).json()
                self.assertEqual(payload["activeModel"], "glm-4.5")

    def test_a_missing_provider_is_reported_as_select_failed(self) -> None:
        response = self.client.post(SELECT_PATH, json={"providerId": "nope", "modelId": "m"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ok"], False)
        self.assertEqual(response.json()["status"], "select_failed")

    def test_an_empty_payload_is_reported_as_select_failed(self) -> None:
        payload = self.client.post(SELECT_PATH, json={}).json()
        self.assertEqual(payload["status"], "select_failed")
        self.assertEqual(payload["reason"], "model_service_provider_missing")

    def test_selecting_writes_the_project_config(self) -> None:
        self.client.post(SELECT_PATH, json={"providerId": "glm", "modelId": "glm-4.5"})
        self.assertEqual(self.config.CHAT_MODEL_NAME, "glm-4.5")

    def test_the_select_event_carries_the_chosen_pair(self) -> None:
        self.recorder.events.clear()
        self.client.post(SELECT_PATH, json={"providerId": "glm", "modelId": "glm-4.5"})
        self.assertEqual(
            self.recorder.events,
            [("model_services_select", ("chat_model", "provider_id", "status"))],
        )


# --------------------------------------------------------------------------- #
# 8. 请求体形状
# --------------------------------------------------------------------------- #


class PayloadShapeTests(Harness):
    def test_a_non_object_body_falls_back_to_an_empty_payload(self) -> None:
        for body in ([1, 2, 3], "string", None, 42):
            with self.subTest(body=body):
                response = self.client.post(CONFIG_PATH, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "invalid_config")

    def test_unparsable_json_falls_back_to_an_empty_payload(self) -> None:
        response = self.client.post(CONFIG_PATH, content=b"not-json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "invalid_config")

    def test_an_empty_payload_asks_for_the_api_key(self) -> None:
        response = self.client.post(CONFIG_PATH, json={})
        self.assertEqual(response.json()["reason"], "model_service_api_key_missing")


# --------------------------------------------------------------------------- #
# 9. 观测设施
# --------------------------------------------------------------------------- #


class ObservationTests(Harness):
    def test_a_metrics_object_without_observe_request_is_ignored(self) -> None:
        client = self.build(runtime_metrics=object())
        response = client.get(PAGE_PATH)
        self.assertEqual(response.status_code, 200)

    def test_a_metrics_object_that_explodes_does_not_break_the_route(self) -> None:
        class Broken:
            def observe_request(self, name, *, duration_ms: float, ok: bool) -> None:
                raise RuntimeError("metrics gone")

        client = self.build(runtime_metrics=Broken())
        response = client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_a_log_sink_that_explodes_does_not_break_the_route(self) -> None:
        class Exploding:
            def __call__(self, *args, **kwargs) -> None:
                raise RuntimeError("log gone")

        client = self.build(log_event=Exploding())
        response = client.post(CONFIG_PATH, json={"apiKey": "k", "chatModel": "m"})
        self.assertTrue(response.json()["ok"])

    def test_metrics_and_logs_default_to_off(self) -> None:
        client = self.build(runtime_metrics=None, log_event=None)
        self.assertEqual(client.get(PAGE_PATH).status_code, 200)

    def test_the_metric_name_is_derived_from_the_route(self) -> None:
        expectations = [
            ("GET", PAGE_PATH, None, "model_services.read"),
            ("POST", CONFIG_PATH, {"apiKey": "k", "chatModel": "m"}, "model_services.save"),
            ("POST", MODELS_PATH, {}, "model_services.models"),
            ("POST", SELECT_PATH, {"providerId": "glm"}, "model_services.select"),
        ]
        for method, path, body, name in expectations:
            with self.subTest(name=name):
                self.recorder.metrics.clear()
                if body is None:
                    self.client.get(path)
                else:
                    self.client.post(path, json=body)
                observed = {entry[0] for entry in self.recorder.metrics}
                self.assertEqual(observed, {name})


# --------------------------------------------------------------------------- #
# 10. 线路字面量（改坏了也不会让别处变红的那批字符串）
# --------------------------------------------------------------------------- #


class C2_4ContractLiteralTests(unittest.TestCase):
    def test_the_route_paths_are_pinned(self) -> None:
        self.assertEqual(
            sorted(EXPECTED_ROUTES),
            [
                ("GET", "/control-center/model-services"),
                ("POST", "/control-center/model-services/select"),
                ("POST", "/control-center/model-services/{provider_id}/config"),
                ("POST", "/control-center/model-services/{provider_id}/models"),
            ],
        )

    def test_the_local_caller_allowlist_is_pinned(self) -> None:
        self.assertEqual(
            routes._LOCAL_CALLER_HOSTS, {"127.0.0.1", "::1", "localhost", "testclient"}
        )

    def test_the_metric_names_are_pinned(self) -> None:
        self.assertEqual(routes._METRIC_PAGE, "model_services.read")
        self.assertEqual(routes._METRIC_CONFIG, "model_services.save")
        self.assertEqual(routes._METRIC_MODELS, "model_services.models")
        self.assertEqual(routes._METRIC_SELECT, "model_services.select")

    def test_the_log_event_names_are_pinned(self) -> None:
        self.assertEqual(routes._EVENT_CONFIG, "model_services_save")
        self.assertEqual(routes._EVENT_MODELS, "model_services_models")
        self.assertEqual(routes._EVENT_SELECT, "model_services_select")

    def test_the_unsaved_provider_hint_is_pinned(self) -> None:
        self.assertEqual(routes._UNSAVED_PROVIDER_HINT, "先保存该供应方配置，再刷新模型列表。")

    def test_the_factory_takes_exactly_the_documented_keyword_arguments(self) -> None:
        import inspect

        signature = inspect.signature(routes.build_model_services_router)
        self.assertEqual(
            list(signature.parameters),
            ["store", "config_module", "engine", "runtime_metrics", "log_event", "model_probe"],
        )
        for name, parameter in signature.parameters.items():
            with self.subTest(name=name):
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


# --------------------------------------------------------------------------- #
# 11. 边界：表达层独立 + 本批的有意分叉
# --------------------------------------------------------------------------- #


class C2_4BoundaryTests(unittest.TestCase):
    """「表达层独立」的直接证据 + 本批的有意分叉被钉住。"""

    C2_4_FILE = SHINKU_PKG / "api" / "model_services.py"

    FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane")

    #: 旧私有名并集（Akane 侧 7 个 ∪ Shinku 旧侧多出的 2 个，规矩二十一）。
    LEGACY_INTERNAL_NAMES = (
        "_bool",                     # 旧侧 model_services
        "_is_local_request",         # 两侧都有
        "_load_effective_settings",  # 两侧都有
        "_load_existing_secret",     # 两侧都有
        "_load_provider_secret",     # 旧侧 model_services
        "_log",                      # 两侧都有
        "_observe",                  # 两侧都有
        "_request_mapping",          # 两侧都有
        "_run_candidate_action",     # 两侧都有
    )

    #: 1 句 = Shinku 旧侧 `build_model_services_router` 的整句 docstring。
    LEGACY_PROSE = (
        "Expose provider-aware model configuration while keeping v1 routes alive.",
    )

    def _source(self) -> str:
        self.assertTrue(self.C2_4_FILE.is_file(), f"expected {self.C2_4_FILE} to exist")
        return self.C2_4_FILE.read_text(encoding="utf-8")

    def test_no_foreign_project_is_mentioned(self) -> None:
        text = self._source().lower()
        for token in self.FOREIGN_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

    def test_no_legacy_internal_name_survives(self) -> None:
        identifiers = _identifiers(self.C2_4_FILE)
        for legacy in self.LEGACY_INTERNAL_NAMES:
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, identifiers)

    def test_the_legacy_name_list_covers_the_whole_union(self) -> None:
        self.assertEqual(len(self.LEGACY_INTERNAL_NAMES), 9)
        self.assertEqual(len(set(self.LEGACY_INTERNAL_NAMES)), len(self.LEGACY_INTERNAL_NAMES))

    def test_no_legacy_prose_sentence_survives(self) -> None:
        text = self._source()
        for sentence in self.LEGACY_PROSE:
            with self.subTest(sentence=sentence):
                self.assertNotIn(sentence, text)

    def test_the_legacy_prose_list_has_one_entry(self) -> None:
        self.assertEqual(len(self.LEGACY_PROSE), 1)

    def test_no_retired_environment_prefix(self) -> None:
        self.assertNotIn("COMPANION_", self._source())

    def test_the_module_only_imports_this_repository_and_stdlib(self) -> None:
        modules = _imported_modules(self.C2_4_FILE)
        unexpected = {
            name
            for name in modules
            if not name.startswith(".")
            and name.split(".")[0]
            not in {"__future__", "asyncio", "time", "collections", "typing", "fastapi"}
        }
        self.assertEqual(unexpected, set())

    def test_the_relative_imports_stay_inside_shinku(self) -> None:
        modules = _imported_modules(self.C2_4_FILE)
        self.assertIn("..contracts.http", modules)
        self.assertIn("..providers.catalog", modules)
        self.assertIn("..providers.service", modules)


if __name__ == "__main__":
    unittest.main()

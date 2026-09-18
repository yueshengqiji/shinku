"""命名契约的测试。

这里验证的是**契约**（默认值、覆盖规则、不继承的旧命名），不是实现细节。
所以断言只看"解析结果是什么"，不看用了什么写法去解析。
"""

from __future__ import annotations

import sys
import unittest

from shinku import names


class DefaultsTests(unittest.TestCase):
    def test_default_ports_match_the_declared_contract(self) -> None:
        self.assertEqual(
            names.DEFAULT_PORTS,
            {
                "backend": 9998,
                "response-host": 9995,
                "time-manager": 9996,
                "browser-host": 9997,
                "agent": 9100,
            },
        )

    def test_every_service_has_a_port_and_a_bind_key(self) -> None:
        for service in names.SERVICE_NAMES:
            with self.subTest(service=service):
                self.assertIn(service, names.DEFAULT_PORTS)
                self.assertTrue(names.env_key(service).startswith(names.ENV_PREFIX))
                self.assertTrue(names.bind_env_key(service).startswith(names.ENV_PREFIX))

    def test_python_services_are_a_subset_of_all_services(self) -> None:
        self.assertTrue(set(names.PYTHON_SERVICE_NAMES) <= set(names.SERVICE_NAMES))
        self.assertNotIn("agent", names.PYTHON_SERVICE_NAMES)

    def test_full_environment_resolves_to_contract_defaults(self) -> None:
        for service in names.SERVICE_NAMES:
            with self.subTest(service=service):
                self.assertEqual(names.service_port(service, {}), names.DEFAULT_PORTS[service])


class OverrideTests(unittest.TestCase):
    def test_port_can_be_overridden_by_environment(self) -> None:
        env = {names.env_key("backend"): "18080"}
        self.assertEqual(names.service_port("backend", env), 18080)
        self.assertEqual(names.service_port("response-host", env), 9995)

    def test_non_numeric_port_falls_back_to_default(self) -> None:
        env = {names.env_key("backend"): "not-a-port"}
        self.assertEqual(names.service_port("backend", env), 9998)

    def test_out_of_range_port_falls_back_to_default(self) -> None:
        for bad in ("80", "0", "70000", "-1"):
            with self.subTest(bad=bad):
                env = {names.env_key("backend"): bad}
                self.assertEqual(names.service_port("backend", env), 9998)

    def test_unknown_service_is_rejected_rather_than_guessed(self) -> None:
        with self.assertRaises(KeyError):
            names.service_port("frontend", {})
        with self.assertRaises(KeyError):
            names.log_path("frontend", environ={})


class RetiredNamingTests(unittest.TestCase):
    """旧项目的命名必须**不被读取**——这是"新项目不继承旧键名"的可执行证据。"""

    def test_companion_keys_do_not_affect_anything(self) -> None:
        env = {"COMPANION_HOST": "0.0.0.0", "COMPANION_PORT": "1234"}
        self.assertEqual(names.service_port("backend", env), names.DEFAULT_PORTS["backend"])
        self.assertEqual(names.service_bind("backend", env), names.DEFAULT_BIND)

    def test_shinku_server_keys_are_not_the_agent_port_source(self) -> None:
        # 旧项目里 SHINKU_SERVER_PORT 指 Agent(Java) 宿主；新项目不继承这个歧义。
        env = {"SHINKU_SERVER_PORT": "1234"}
        self.assertEqual(names.service_port("agent", env), 9100)

    def test_the_retired_key_list_is_declared(self) -> None:
        for key in ("COMPANION_HOST", "COMPANION_PORT", "SHINKU_SERVER_HOST", "SHINKU_SERVER_PORT"):
            with self.subTest(key=key):
                self.assertIn(key, names.RETIRED_ENV_KEYS)
        self.assertNotIn("", names.RETIRED_ENV_KEYS)


class BindTests(unittest.TestCase):
    def test_default_bind_is_loopback_only(self) -> None:
        for service in names.SERVICE_NAMES:
            with self.subTest(service=service):
                self.assertEqual(names.service_bind(service, {}), "127.0.0.1")

    def test_allow_lan_is_off_unless_explicitly_enabled(self) -> None:
        for raw in ("", "0", "false", "no", "off", "maybe"):
            with self.subTest(raw=raw):
                self.assertFalse(names.allow_lan({names.ALLOW_LAN_ENV: raw}))

    def test_allow_lan_accepts_explicit_true_values(self) -> None:
        for raw in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(raw=raw):
                self.assertTrue(names.allow_lan({names.ALLOW_LAN_ENV: raw}))

    def test_allow_lan_opens_the_default_bind(self) -> None:
        env = {names.ALLOW_LAN_ENV: "true"}
        self.assertEqual(names.service_bind("backend", env), "0.0.0.0")

    def test_explicit_bind_wins_over_allow_lan(self) -> None:
        env = {names.ALLOW_LAN_ENV: "true", names.bind_env_key("backend"): "10.0.0.5"}
        self.assertEqual(names.service_bind("backend", env), "10.0.0.5")


class PathTests(unittest.TestCase):
    def test_explicit_roots_win(self) -> None:
        env = {
            names.DATA_ROOT_ENV: "/tmp/shinku-data",
            names.CONFIG_ROOT_ENV: "/tmp/shinku-config",
            names.LOG_ROOT_ENV: "/tmp/shinku-logs",
        }
        self.assertEqual(names.data_root(env).name, "shinku-data")
        self.assertEqual(names.config_root(env).name, "shinku-config")
        self.assertEqual(names.log_root(env).name, "shinku-logs")

    def test_platform_default_lives_under_the_platform_state_root(self) -> None:
        # 注入 LOCALAPPDATA，避免测试依赖真实机器环境。
        # 非 Windows 平台下 _platform_state_root 走 XDG/家目录，注入的键不参与，
        # 所以这里按平台分支断言"落在平台状态根下的 Shinku 目录里"。
        env = {"LOCALAPPDATA": "/tmp/fake-localappdata"}
        for root in (names.data_root(env), names.config_root(env), names.log_root(env)):
            with self.subTest(root=str(root)):
                # root 本身是 .../<Shinku>/data|config|logs，所以状态根是它的父目录。
                self.assertEqual(root.parent.name.lower(), "shinku")
                if sys.platform == "win32":
                    self.assertEqual(root.parent.name, "Shinku")
                    self.assertEqual(root.parent.parent.name, "fake-localappdata")

    def test_data_default_is_not_inside_the_repository(self) -> None:
        # 旧项目默认把数据放仓库内 users_data/；新项目按平台约定放用户目录。
        root = names.data_root({"LOCALAPPDATA": "/tmp/fake-localappdata"})
        self.assertNotIn("users_data", str(root))

    def test_log_name_is_derived_from_the_service_name(self) -> None:
        env = {names.LOG_ROOT_ENV: "/tmp/shinku-logs"}
        self.assertEqual(names.log_path("backend", environ=env).name, "backend.log")
        self.assertEqual(names.log_path("backend", error=True, environ=env).name, "backend.err.log")
        self.assertEqual(names.log_path("time-manager", environ=env).name, "time-manager.log")

    def test_service_url_targets_loopback_even_when_bind_is_open(self) -> None:
        env = {names.ALLOW_LAN_ENV: "true"}
        self.assertEqual(names.service_url("backend", env), "http://127.0.0.1:9998")


class HeaderContractTests(unittest.TestCase):
    def test_wire_headers_are_pinned(self) -> None:
        # 这两个头名是线路协议，改它们等于改协议——所以单独钉住。
        self.assertEqual(names.CORRELATION_ID_HEADER, "X-Correlation-ID")
        self.assertEqual(names.SERVICE_TOKEN_HEADER, "X-Shinku-Service-Token")


if __name__ == "__main__":
    unittest.main()

"""后端应用工厂的测试。

B1 的健康探针不是"占位符"——它的职责是**证明新树能在没有旧项目的情况下起来**，
所以测试要验证：进程能构造应用、能回答、回显的是**注入的**那套配置（不是碰巧的默认值）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from shinku import __version__
from shinku import names
from shinku.api import create_app
from shinku.config import load_settings


def settings_for(root: Path):
    """构造一份指向临时目录的配置，绝不碰真实用户目录。"""

    return load_settings(
        {
            names.DATA_ROOT_ENV: str(root / "data"),
            names.CONFIG_ROOT_ENV: str(root / "config"),
            names.LOG_ROOT_ENV: str(root / "logs"),
            names.env_key("backend"): "18080",
        }
    )


class HealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.settings = settings_for(self.root)
        self.client = TestClient(create_app(self.settings))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_health_reports_ok(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_health_echoes_the_injected_configuration(self) -> None:
        body = self.client.get("/health").json()
        self.assertEqual(body["port"], 18080)
        self.assertEqual(body["bind"], "127.0.0.1")
        self.assertEqual(body["allow_lan"], False)
        self.assertEqual(body["version"], __version__)
        self.assertEqual(body["service"], "backend")
        self.assertEqual(body["pid"], __import__("os").getpid())
        self.assertIn("python", body)
        self.assertIn("yt_dlp", body)
        self.assertEqual(Path(body["data_root"]), self.root / "data")
        self.assertEqual(Path(body["log_file"]), self.root / "logs" / "backend.log")

    def test_ready_creates_and_probes_the_data_root(self) -> None:
        body = self.client.get("/health/ready").json()
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["detail"], "ok")
        self.assertTrue((self.root / "data").is_dir())

    def test_ready_does_not_leave_probe_files_behind(self) -> None:
        self.client.get("/health/ready")
        leftovers = list((self.root / "data").iterdir())
        self.assertEqual(leftovers, [], f"probe left files behind: {leftovers}")

    def test_the_app_does_not_expose_docs_or_schema(self) -> None:
        # 内网服务不必要地暴露 schema 只会扩大面；B1 就把它关掉。
        self.assertEqual(self.client.get("/docs").status_code, 404)
        self.assertEqual(self.client.get("/openapi.json").status_code, 404)

    def test_unknown_path_is_404_not_a_hollow_200(self) -> None:
        self.assertEqual(self.client.get("/capabilities").status_code, 404)


class FactoryTests(unittest.TestCase):
    def test_two_apps_do_not_share_state(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = TestClient(create_app(settings_for(Path(a)))).get("/health").json()
            second = TestClient(create_app(settings_for(Path(b)))).get("/health").json()
            self.assertNotEqual(first["data_root"], second["data_root"])

    def test_factory_falls_back_to_environment_when_no_settings_given(self) -> None:
        # 不传 settings 时读环境；这里只验证它不炸，且端口来自契约默认值。
        app = create_app()
        body = TestClient(app).get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["port"], int)


if __name__ == "__main__":
    unittest.main()

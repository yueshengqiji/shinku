"""命令行入口的测试。

重点在两条**负面**契约：
1. 未实现的服务必须报错退出，不能静默起一个空进程；
2. 旧命名（``COMPANION_*``）即使被设置也要被忽略并给出警告。
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from shinku import names
from shinku.cli import main


def isolated_env(root: Path) -> dict[str, str]:
    return {
        names.DATA_ROOT_ENV: str(root / "data"),
        names.CONFIG_ROOT_ENV: str(root / "config"),
        names.LOG_ROOT_ENV: str(root / "logs"),
        "SHINKU_ENV_FILE": str(root / "missing.env"),
    }


class DoctorTests(unittest.TestCase):
    def test_doctor_prints_the_resolved_configuration_and_creates_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.StringIO()
            with mock.patch.dict(os.environ, isolated_env(root), clear=True):
                with redirect_stdout(buf):
                    code = main(["doctor"])
            out = buf.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("project      : shinku", out)
            self.assertIn(str(root / "data"), out)
            self.assertIn("backend", out)
            self.assertTrue((root / "data").is_dir())
            self.assertTrue((root / "config").is_dir())
            self.assertTrue((root / "logs").is_dir())

    def test_doctor_warns_about_retired_env_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**isolated_env(Path(tmp)), "COMPANION_PORT": "1234"}
            buf = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True):
                with redirect_stdout(buf):
                    main(["doctor"])
            out = buf.getvalue()
            self.assertIn("retired env keys", out)
            self.assertIn("COMPANION_PORT", out)
            # 关键：警告了，但**没有**把 1234 当成后端端口。
            self.assertIn("9998", out)
            self.assertNotIn("1234", out)

    def test_doctor_marks_unimplemented_services(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with mock.patch.dict(os.environ, isolated_env(Path(tmp)), clear=True):
                with redirect_stdout(buf):
                    main(["doctor"])
            out = buf.getvalue()
            self.assertIn("[not implemented in B1]", out)
            # backend 是 B1 唯一实现的，不该被打标。
            backend_line = next(l for l in out.splitlines() if l.strip().startswith("backend"))
            self.assertNotIn("not implemented", backend_line)


class ServeTests(unittest.TestCase):
    def test_serve_refuses_an_unimplemented_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with mock.patch.dict(os.environ, isolated_env(Path(tmp)), clear=True):
                with redirect_stderr(err):
                    code = main(["serve", "--service", "agent"])
            self.assertEqual(code, 2)
            self.assertIn("not implemented", err.getvalue())

    def test_serve_reaches_uvicorn_for_the_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, isolated_env(Path(tmp)), clear=True):
                with mock.patch("uvicorn.run") as run:
                    code = main(["serve", "--service", "backend"])
            self.assertEqual(code, 0)
            self.assertTrue(run.called)
            _, kwargs = run.call_args
            self.assertEqual(kwargs["host"], "127.0.0.1")
            self.assertEqual(kwargs["port"], 9998)

    def test_serve_uses_the_overridden_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**isolated_env(Path(tmp)), names.env_key("backend"): "18080"}
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("uvicorn.run") as run:
                    main(["serve", "--service", "backend"])
            _, kwargs = run.call_args
            self.assertEqual(kwargs["port"], 18080)


class ParserTests(unittest.TestCase):
    def test_unknown_service_is_rejected_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit):
            main(["serve", "--service", "frontend"])

    def test_missing_subcommand_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()

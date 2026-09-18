"""C1-5 契约测试：进程健康载荷原语。

断言对象分三层：

1. **契约事实**（键集合、键序、四个取值语义）——契约事实是迁移来的，所以测试要
   逐条钉住它们，而不是只钉「返回了一个字典」；
2. **必须报错**——契约的价值一半在于它**拒绝**什么：这个原语不接受任何参数，
   传什么都必须是 `TypeError`，不是静默忽略；
3. **活性**——`pid` 必须是**造出这份载荷的那个进程**的进程号。
   这一点把本原语与「返回写死字典」区分开，也只有跨进程调用才能证明。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest

from shinku import health

CONTRACT_KEYS = ["status", "pid", "python", "yt_dlp"]


class HealthPayloadContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = health.build_basic_health_payload()

    def test_the_module_exports_exactly_one_name(self) -> None:
        self.assertEqual(health.__all__, ["build_basic_health_payload"])

    def test_the_key_set_is_exactly_the_contract(self) -> None:
        self.assertEqual(sorted(self.payload), sorted(CONTRACT_KEYS))
        self.assertEqual(len(self.payload), len(CONTRACT_KEYS))

    def test_the_keys_keep_the_contract_order(self) -> None:
        self.assertEqual(list(self.payload), CONTRACT_KEYS)

    def test_status_is_the_contract_literal(self) -> None:
        self.assertEqual(self.payload["status"], "ok")
        self.assertIsInstance(self.payload["status"], str)

    def test_pid_is_this_process(self) -> None:
        self.assertEqual(self.payload["pid"], os.getpid())
        self.assertIsInstance(self.payload["pid"], int)

    def test_python_is_the_running_interpreter(self) -> None:
        self.assertEqual(self.payload["python"], sys.executable)
        self.assertIsInstance(self.payload["python"], str)

    def test_yt_dlp_is_an_availability_flag(self) -> None:
        expected = importlib.util.find_spec("yt_dlp") is not None
        self.assertEqual(self.payload["yt_dlp"], expected)
        self.assertIsInstance(self.payload["yt_dlp"], bool)

    def test_the_probe_does_not_import_the_dependency(self) -> None:
        self.assertNotIn("yt_dlp", sys.modules, "用例前提：本环境不应已导入 yt_dlp")
        health.build_basic_health_payload()
        self.assertNotIn("yt_dlp", sys.modules, "探针只许查元数据，不许真导入")

    def test_each_call_returns_a_fresh_dict(self) -> None:
        first = health.build_basic_health_payload()
        second = health.build_basic_health_payload()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        for key in CONTRACT_KEYS:
            with self.subTest(key=key):
                first[key] = "poisoned"
        self.assertNotIn("poisoned", second.values(), "写坏的返回值污染了另一次调用")

    def test_the_payload_is_stable_across_calls_in_one_process(self) -> None:
        seen = {tuple(health.build_basic_health_payload().items()) for _ in range(5)}
        self.assertEqual(len(seen), 1)

    def test_the_payload_describes_the_process_that_built_it(self) -> None:
        # 同一进程里比对没有信息量（两侧都读同一个 pid）；必须换个进程才算验过。
        script = (
            "import json, shinku.health\n"
            "print(json.dumps(shinku.health.build_basic_health_payload()))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        child = json.loads(proc.stdout)
        self.assertEqual(list(child), CONTRACT_KEYS)
        self.assertEqual(child["status"], "ok")
        self.assertEqual(child["python"], sys.executable)
        self.assertNotEqual(child["pid"], os.getpid(), "子进程报的是父进程的 pid")
        self.assertGreater(child["pid"], 0)


class HealthPayloadFailureTests(unittest.TestCase):
    def test_a_positional_argument_is_rejected(self) -> None:
        # 这两条用例故意传错参数：契约的价值一半在于它拒绝什么，不是静默忽略什么。
        # 刻意不写 `# type: ignore`——工具指令会被散文口径当注释捞进来（SKILL 规矩十二），
        # 而本仓当前不跑静态检查，加了只是噪声。
        with self.assertRaises(TypeError):
            health.build_basic_health_payload("extra")

    def test_a_keyword_argument_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            health.build_basic_health_payload(reason="extra")


if __name__ == "__main__":
    unittest.main()

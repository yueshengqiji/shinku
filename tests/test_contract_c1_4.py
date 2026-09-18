"""C1-4 公共守卫、证据门、供应商词汇与原生工具 schema 的测试。

这一批按来源记录 §5.9 的**路径 A 第二类**处理：四个模块都含逻辑、分支或数据处理，属**洁净室重写**。
断言对象全部是**外部可观察行为**（契约 `docs/contracts/c1_4_guards_provider_tool_schema.md`），
不涉及任何实现结构。

四处刻意的地方：

1. **输入语料在本测试内组织。** 句子、base_url、handler 形态全部现写；不引用旧项目的
   测试文件或夹具目录。覆盖点与旧测试重合是行为等价的要求，但文本是自撰的。
2. **契约里那些"不会让任何既有测试变红"的表达式**在这里被逐条钉住。典型是 provider 的
   **三张信号词表必须分开**：把 ``_INFER_SIGNALS`` 与 ``_MATCH_SIGNALS`` 合成一张，
   ``https://deepseek.com/x`` 就会从"推不出"变成"deepseek"，而没有任何既有测试会因此变红。
3. **边界扫描用 `tokenize` 取精确标识符**，不用子串匹配（理由同 C1-3：本批的旧私有名里有
   ``_day``、``_lock``、``_units`` 这种短词，子串匹配会误判）。
4. **`guards/evidence.py` 的判定边界用两张表钉死**（契约 §3.3）：必须删的句式与必须保留的
   句式各一份，逐条断言。这是本批唯一一个旧实现与上游**逐字节相同**的模块，行为钉得最密。
"""

from __future__ import annotations

import ast
import inspect
import importlib
import tokenize
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from shinku.guards import evidence
from shinku.guards import public_think
from shinku.guards.evidence import (
    CHECK_AGAIN_LINE,
    apply_evidence_guard,
    strip_provider_speculation,
    strip_unsupported_claims,
)
from shinku.guards.public_think import GuardDecision, PublicThinkGuard
from shinku.providers import config as provider_config
from shinku.providers.config import (
    COMMON_PROVIDER_PRESETS,
    DEFAULT_PROVIDER_ID,
    SUPPORTED_PROTOCOLS,
    ProviderPreset,
    base_url_matches_provider,
    canonical_provider_id,
    infer_provider_id,
    normalize_api_protocol,
    normalize_base_url,
    normalize_configured_base_url,
    preset_index,
    provider_presets_payload,
)
from shinku.tools import native_schema
from shinku.tools.native_schema import (
    NATIVE_TOOL_DESCRIPTION_MAX_CHARS,
    NATIVE_TOOL_NAME_RE,
    build_openai_native_tool_specs,
)

SHINKU_PKG = Path(public_think.__file__).resolve().parents[1]


def _identifiers(path: Path) -> set[str]:
    """文件里出现过的所有 Python 标识符（精确 token，不是子串）。"""

    with path.open(encoding="utf-8") as handle:
        return {
            token.string
            for token in tokenize.generate_tokens(handle.readline)
            if token.type == tokenize.NAME
        }


def _imported_modules(path: Path) -> set[str]:
    """文件里所有 import 目标的字面名（相对导入保留前导点）。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def _guard(**overrides) -> PublicThinkGuard:
    values = dict(
        enabled=True,
        max_concurrent_thinks=2,
        daily_think_limit=10,
        busy_message="busy",
        daily_limit_message="limit",
    )
    values.update(overrides)
    return PublicThinkGuard(**values)


# --------------------------------------------------------------------------- #
# guards/public_think.py
# --------------------------------------------------------------------------- #


class PublicThinkSurfaceTests(unittest.TestCase):
    """对外面：签名、字段、导出。"""

    def test_exported_names_are_exactly_the_documented_two(self) -> None:
        self.assertEqual(public_think.__all__, ["GuardDecision", "PublicThinkGuard"])

    def test_guard_decision_has_the_documented_fields_and_default(self) -> None:
        decision = GuardDecision(True, True, "ok")
        self.assertEqual(decision.message, "")
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.acquired)
        self.assertEqual(decision.reason, "ok")

    def test_guard_decision_is_frozen(self) -> None:
        with self.assertRaises(Exception):
            GuardDecision(True, True, "ok").reason = "x"  # type: ignore[misc]

    def test_constructor_takes_keyword_only_arguments(self) -> None:
        parameters = inspect.signature(PublicThinkGuard.__init__).parameters
        for name in (
            "enabled",
            "max_concurrent_thinks",
            "daily_think_limit",
            "busy_message",
            "daily_limit_message",
        ):
            with self.subTest(name=name):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["timezone_name"].default, "Asia/Shanghai")

    def test_snapshot_exposes_exactly_the_six_documented_keys(self) -> None:
        snapshot = _guard().snapshot()
        self.assertEqual(
            set(snapshot),
            {
                "enabled",
                "max_concurrent_thinks",
                "daily_think_limit",
                "active_thinks",
                "used_today",
                "day_key",
            },
        )


class PublicThinkNormalizationTests(unittest.TestCase):
    """参数归一化：负数归零、空消息取默认。"""

    def test_negative_limits_are_floored_at_zero(self) -> None:
        guard = _guard(max_concurrent_thinks=-3, daily_think_limit=-1)
        self.assertEqual(guard.max_concurrent_thinks, 0)
        self.assertEqual(guard.daily_think_limit, 0)
        # 归零表示"不限"：仍能放行
        self.assertTrue(guard.try_acquire().allowed)

    def test_blank_messages_fall_back_to_the_documented_defaults(self) -> None:
        for blank in ("", "   ", None):
            with self.subTest(value=repr(blank)):
                guard = _guard(busy_message=blank, daily_limit_message=blank)
                self.assertEqual(guard.busy_message, "当前体验人数较多，请稍后再试。")
                self.assertEqual(guard.daily_limit_message, "今日体验名额已满，明天再来看看吧。")

    def test_given_messages_are_stripped_but_kept(self) -> None:
        guard = _guard(busy_message="  忙  ", daily_limit_message=" 满 ")
        self.assertEqual(guard.busy_message, "忙")
        self.assertEqual(guard.daily_limit_message, "满")


class PublicThinkAdmissionTests(unittest.TestCase):
    """准入判定的三条路径与优先级。"""

    def test_disabled_guard_allows_without_acquiring(self) -> None:
        guard = _guard(enabled=False)
        decision = guard.try_acquire()
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.acquired)
        self.assertEqual(decision.reason, "disabled")
        self.assertEqual(decision.message, "")

    def test_disabled_guard_does_not_count_quota(self) -> None:
        guard = _guard(enabled=False, daily_think_limit=1)
        for _ in range(5):
            self.assertTrue(guard.try_acquire().allowed)
        self.assertEqual(guard.snapshot()["used_today"], 0)

    def test_disabled_release_is_a_no_op(self) -> None:
        guard = _guard(enabled=False)
        guard.release()
        self.assertEqual(guard.snapshot()["active_thinks"], 0)

    def test_acquire_returns_ok_and_counts(self) -> None:
        guard = _guard()
        decision = guard.try_acquire()
        self.assertEqual((decision.allowed, decision.acquired, decision.reason), (True, True, "ok"))
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["active_thinks"], 1)
        self.assertEqual(snapshot["used_today"], 1)

    def test_concurrency_limit_blocks_and_carries_the_busy_message(self) -> None:
        guard = _guard(max_concurrent_thinks=1)
        self.assertTrue(guard.try_acquire().acquired)
        blocked = guard.try_acquire()
        self.assertEqual((blocked.allowed, blocked.acquired, blocked.reason), (False, False, "busy"))
        self.assertEqual(blocked.message, "busy")

    def test_daily_limit_blocks_and_carries_the_daily_message(self) -> None:
        guard = _guard(daily_think_limit=1)
        self.assertTrue(guard.try_acquire().acquired)
        blocked = guard.try_acquire()
        self.assertEqual(blocked.reason, "daily_limit")
        self.assertEqual(blocked.message, "limit")

    def test_daily_limit_wins_over_concurrency_limit(self) -> None:
        # 两个上限同时触发时，每日额度优先（契约 §2.2 第 3 条）。
        guard = _guard(max_concurrent_thinks=0, daily_think_limit=1)
        self.assertTrue(guard.try_acquire().acquired)
        self.assertEqual(guard.try_acquire().reason, "daily_limit")

    def test_release_does_not_return_the_daily_quota(self) -> None:
        guard = _guard(max_concurrent_thinks=2, daily_think_limit=1)
        self.assertTrue(guard.try_acquire().acquired)
        guard.release()
        blocked = guard.try_acquire()
        self.assertEqual(blocked.reason, "daily_limit")

    def test_release_never_makes_the_active_count_negative(self) -> None:
        guard = _guard()
        for _ in range(4):
            guard.release()
        self.assertEqual(guard.snapshot()["active_thinks"], 0)

    def test_release_frees_a_concurrency_slot(self) -> None:
        guard = _guard(max_concurrent_thinks=1)
        self.assertTrue(guard.try_acquire().acquired)
        self.assertFalse(guard.try_acquire().allowed)
        guard.release()
        self.assertTrue(guard.try_acquire().acquired)


class PublicThinkRolloverTests(unittest.TestCase):
    """跨日复位：只归零当日已用，不动进行中的条数。"""

    def _at_next_day(self):
        zone = ZoneInfo("Asia/Shanghai")
        fake = mock.Mock(wraps=datetime)
        fake.now.return_value = datetime.now(zone) + timedelta(days=1)
        return mock.patch.object(public_think, "datetime", fake)

    def test_a_new_day_clears_the_daily_counter(self) -> None:
        guard = _guard(daily_think_limit=1)
        self.assertTrue(guard.try_acquire().acquired)
        self.assertFalse(guard.try_acquire().allowed)
        with self._at_next_day():
            self.assertTrue(guard.try_acquire().allowed)

    def test_a_new_day_leaves_the_active_count_alone(self) -> None:
        guard = _guard(max_concurrent_thinks=5)
        guard.try_acquire()
        with self._at_next_day():
            self.assertEqual(guard.snapshot()["active_thinks"], 1)

    def test_the_day_key_advances_with_the_clock(self) -> None:
        guard = _guard()
        today = guard.snapshot()["day_key"]
        with self._at_next_day():
            self.assertNotEqual(guard.snapshot()["day_key"], today)


# --------------------------------------------------------------------------- #
# guards/evidence.py
# --------------------------------------------------------------------------- #

#: 无据时必须删除的句式（契约 §3.3 第一张表）。
CLAIM_SENTENCES = (
    "我查了日志，报的是端口占用。",
    "我已经查过了这件事。",
    "我刚才核对过一遍。",
    "我读过了那份说明。",
    "我确认过了。",
    "查到了，端口没被占用。",
    "读到了，直接说结论。",
    "翻到了那一页。",
    "查完了，结论是端口没起。",
    "核对过了。",
    "我手里有结果。",
    "我这边有数据。",
    "文档写着温度默认 0.7。",
    "README说了这件事。",
    "说明里写着怎么改。",
    "日志里写着 connection refused。",
    "日志报的是超时。",
    "配置文件里有这一项。",
    "代码里写着默认值。",
    "按读到的说，端口是 9998。",
    "按我刚读到的结论。",
)

#: 不得误删的句式（契约 §3.3 第二张表）。
SURVIVING_SENTENCES = (
    "我去查一下日志。",
    "要不要我去看一眼？",
    "先看看是不是端口问题。",
    "我看到你贴的日志了，是连接被拒。",
    "看你发的截图，端口没起来。",
    "我查完就告诉你。",
    "端口被占着通常是上一次没退干净。",
    "现在去看一眼也行。",
    "说明我还没看，写的是什么？",
)

#: 供应方内部推测，任何情况下都要删（契约 §3.3 第三张表）。
SPECULATION_SENTENCES = (
    "按常理推，服务端过滤不太可能改措辞。",
    "服务端确实做了截断或替换。",
    "也许是平台可能拦截了这一段。",
    "供应方内部做了替换。",
    "供应商层面进行了删改。",
    "这可能是模型权重导致的偏差。",
    "那条规则在内部那一层。",
    "这不在你手里。",
    "也许是隐藏审查拦下来了。",
    "隐藏的审查也可能。",
    "隐藏输出审查也许在起作用。",
    "规则写在内置规则里。",
    "内置提示可能改了它。",
    "内置审查拦了。",
    "也许是投递链路里某一步改了文本。",
    "投递链路中某处截断了。",
)

#: 关于供应方的诚实表述，不得删（契约 §3.3 第四张表）。
HONEST_SENTENCES = (
    "我看不到供应方内部做了什么，这不在我能查的范围里。",
    "服务端返回 502，先看网关那边的日志。",
    "平台已经下线了那个接口。",
    "供应方给出的错误码是 429。",
    "服务端确实返回了两个头。",
    "投递链路完好无损。",
)


class EvidenceClaimTests(unittest.TestCase):
    """无据查证句式：没有证据时删，有证据时不动。"""

    def test_claims_are_removed_without_evidence(self) -> None:
        for text in CLAIM_SENTENCES:
            with self.subTest(text=text):
                cleaned, reasons = strip_unsupported_claims(text, has_evidence=False)
                self.assertIn("unsupported_claim", reasons)
                self.assertEqual(cleaned, "")

    def test_claims_are_kept_with_evidence(self) -> None:
        for text in CLAIM_SENTENCES:
            with self.subTest(text=text):
                cleaned, reasons = strip_unsupported_claims(text, has_evidence=True)
                self.assertEqual(cleaned, text)
                self.assertEqual(reasons, [])

    def test_surviving_sentences_are_never_touched(self) -> None:
        for text in SURVIVING_SENTENCES:
            for has_evidence in (False, True):
                with self.subTest(text=text, evidence=has_evidence):
                    cleaned, reasons = strip_unsupported_claims(text, has_evidence=has_evidence)
                    self.assertEqual(cleaned, text)
                    self.assertEqual(reasons, [])

    def test_only_the_offending_sentence_is_dropped(self) -> None:
        cleaned, reasons = strip_unsupported_claims(
            "读完了，直接说结论。当前模型是 deepseek-v4-flash。", has_evidence=False
        )
        self.assertEqual(reasons, ["unsupported_claim"])
        self.assertEqual(cleaned, "当前模型是 deepseek-v4-flash。")
        self.assertNotIn("读完了", cleaned)

    def test_a_sentence_without_a_hit_returns_the_original_text(self) -> None:
        text = "端口被占着通常是上一次没退干净。"
        self.assertEqual(strip_unsupported_claims(text, has_evidence=False), (text, []))

    def test_blank_input_is_returned_unchanged(self) -> None:
        for text in ("", "   ", "\n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(strip_unsupported_claims(text, has_evidence=False), (text, []))

    def test_non_string_input_is_coerced(self) -> None:
        self.assertEqual(strip_unsupported_claims(None, has_evidence=False), ("", []))


class EvidenceSpeculationTests(unittest.TestCase):
    """供应方内部推测：任何情况下都删。"""

    def test_speculation_is_removed(self) -> None:
        for text in SPECULATION_SENTENCES:
            with self.subTest(text=text):
                cleaned, reasons = strip_provider_speculation(text)
                self.assertIn("provider_speculation", reasons)
                self.assertEqual(cleaned, "")

    def test_honest_statements_survive(self) -> None:
        for text in HONEST_SENTENCES:
            with self.subTest(text=text):
                cleaned, reasons = strip_provider_speculation(text)
                self.assertEqual(cleaned, text)
                self.assertEqual(reasons, [])

    def test_speculation_is_not_excused_by_evidence(self) -> None:
        speech, segments, reasons = apply_evidence_guard(
            speech="服务端过滤可能把它改掉了。\n我查了日志，是连接被拒。",
            speech_segments=[],
            has_evidence=True,
        )
        self.assertIn("provider_speculation", reasons)
        self.assertNotIn("服务端过滤", speech)
        self.assertIn("我查了日志", speech)

    def test_blank_input_is_returned_unchanged(self) -> None:
        self.assertEqual(strip_provider_speculation("  "), ("  ", []))


class EvidenceApplyTests(unittest.TestCase):
    """apply_evidence_guard 的编排：分段优先、整段清空时换核查话术。"""

    def test_everything_removed_becomes_the_check_line(self) -> None:
        speech, segments, reasons = apply_evidence_guard(
            speech="我查完了，文档里写着答案。",
            speech_segments=["我查完了，文档里写着答案。"],
            has_evidence=False,
        )
        self.assertEqual(speech, CHECK_AGAIN_LINE)
        self.assertEqual(segments, [CHECK_AGAIN_LINE])
        self.assertIn("unsupported_claim", reasons)

    def test_segments_are_cleaned_individually(self) -> None:
        speech, segments, reasons = apply_evidence_guard(
            speech="读完了。\n端口是 9998。",
            speech_segments=["读完了。", "端口是 9998。"],
            has_evidence=False,
        )
        self.assertEqual(segments, ["端口是 9998。"])
        self.assertEqual(speech, "端口是 9998。")
        self.assertIn("unsupported_claim", reasons)

    def test_a_clean_reply_passes_through(self) -> None:
        text = "先看谁占着端口。\n再用 PID 对进程名。"
        speech, segments, reasons = apply_evidence_guard(
            speech=text,
            speech_segments=["先看谁占着端口。", "再用 PID 对进程名。"],
            has_evidence=False,
        )
        self.assertEqual(speech, text)
        self.assertEqual(segments, ["先看谁占着端口。", "再用 PID 对进程名。"])
        self.assertEqual(reasons, [])

    def test_segments_win_over_the_joined_speech(self) -> None:
        speech, segments, reasons = apply_evidence_guard(
            speech="整段文本。\n第二段。",
            speech_segments=["整段文本。", "第二段。"],
            has_evidence=False,
        )
        self.assertEqual(speech, "整段文本。\n第二段。")
        self.assertEqual(segments, ["整段文本。", "第二段。"])
        self.assertEqual(reasons, [])

    def test_non_list_segments_are_treated_as_empty(self) -> None:
        for value in (None, "读完了。", ("读完了。",), 123):
            with self.subTest(value=repr(value)):
                speech, segments, _ = apply_evidence_guard(
                    speech="先看端口。", speech_segments=value, has_evidence=False
                )
                self.assertEqual(speech, "先看端口。")
                self.assertEqual(segments, [])

    def test_reasons_are_ordered_claims_before_speculation(self) -> None:
        _, _, reasons = apply_evidence_guard(
            speech="我查了日志。服务端过滤可能改了它。",
            speech_segments=[],
            has_evidence=False,
        )
        self.assertEqual(reasons, ["unsupported_claim", "provider_speculation"])

    def test_the_check_line_is_the_documented_sentence(self) -> None:
        self.assertEqual(CHECK_AGAIN_LINE, "这个我得先去看一眼，不能凭印象说。")


class EvidenceSurfaceTests(unittest.TestCase):
    """签名与导出面。"""

    def test_exported_names_are_exactly_the_documented_four(self) -> None:
        self.assertEqual(
            evidence.__all__,
            [
                "CHECK_AGAIN_LINE",
                "apply_evidence_guard",
                "strip_provider_speculation",
                "strip_unsupported_claims",
            ],
        )

    def test_strip_unsupported_claims_takes_evidence_as_keyword_only(self) -> None:
        parameters = inspect.signature(strip_unsupported_claims).parameters
        self.assertEqual(parameters["has_evidence"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["text"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_apply_takes_three_keyword_only_arguments_in_order(self) -> None:
        parameters = inspect.signature(apply_evidence_guard).parameters
        names = [name for name in parameters if name != "self"]
        self.assertEqual(names, ["speech", "speech_segments", "has_evidence"])
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_apply_returns_a_three_tuple(self) -> None:
        result = apply_evidence_guard(speech="x", speech_segments=[], has_evidence=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class EvidenceSentenceSplitTests(unittest.TestCase):
    """句子切分：标点留在句末，换行只作分隔。"""

    def test_a_removed_sentence_does_not_glue_its_neighbours_together(self) -> None:
        cleaned, _ = strip_unsupported_claims(
            "读完了。端口是 9998。", has_evidence=False
        )
        self.assertEqual(cleaned, "端口是 9998。")

    def test_newlines_are_treated_as_separators(self) -> None:
        cleaned, reasons = strip_unsupported_claims(
            "读完了。\n端口是 9998。", has_evidence=False
        )
        self.assertEqual(cleaned, "端口是 9998。")
        self.assertEqual(reasons, ["unsupported_claim"])

    def test_a_trailing_sentence_without_punctuation_is_kept(self) -> None:
        cleaned, _ = strip_unsupported_claims("我查了。最后一句", has_evidence=False)
        self.assertEqual(cleaned, "最后一句")

    def test_english_punctuation_splits_too(self) -> None:
        cleaned, reasons = strip_unsupported_claims(
            "I checked it. port is 9998.", has_evidence=False
        )
        # 英文句不受中文动词表影响，原样保留。
        self.assertEqual(cleaned, "I checked it. port is 9998.")
        self.assertEqual(reasons, [])


# --------------------------------------------------------------------------- #
# providers/config.py
# --------------------------------------------------------------------------- #


class ProviderVocabularyTests(unittest.TestCase):
    """常量、预置与载荷形状。"""

    def test_supported_protocols_and_default_id(self) -> None:
        self.assertEqual(SUPPORTED_PROTOCOLS, ("openai", "anthropic", "ollama"))
        self.assertEqual(DEFAULT_PROVIDER_ID, "openai_compatible")

    def test_preset_fields_and_default_capabilities(self) -> None:
        self.assertEqual(
            list(ProviderPreset.__dataclass_fields__),
            ["id", "label", "protocol", "base_url", "api_key_required", "description", "capabilities"],
        )
        bare = ProviderPreset("a", "b", "c", "d", True, "e")
        self.assertEqual(bare.capabilities, ("stream",))

    def test_the_seven_presets_are_intact(self) -> None:
        self.assertEqual(
            [(p.id, p.label, p.protocol, p.base_url, p.api_key_required, p.description, p.capabilities)
             for p in COMMON_PROVIDER_PRESETS],
            [
                ("openai", "OpenAI", "openai", "https://api.openai.com/v1", True, "OpenAI 官方接口。", ("stream", "vision", "native_tools")),
                ("deepseek", "DeepSeek", "openai", "https://api.deepseek.com/v1", True, "DeepSeek 官方 OpenAI 兼容接口。", ("stream", "native_tools")),
                ("pinaic", "Pinaic", "openai", "https://api.pinaic.com/v1", True, "Pinaic 的 OpenAI 兼容接口。", ("stream",)),
                ("gemini", "Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai", True, "Google AI Studio 的 OpenAI 兼容接口。", ("stream", "vision")),
                ("anthropic", "Anthropic Claude", "anthropic", "https://api.anthropic.com", True, "Anthropic 官方 Messages API。", ("stream", "vision", "native_tools")),
                ("ollama", "Ollama 本地模型", "ollama", "http://127.0.0.1:11434", False, "本机 Ollama 的 OpenAI 兼容接口。", ("stream", "vision")),
                ("openai_compatible", "其他 OpenAI 兼容服务", "openai", "", True, "中转站、自部署网关或其他兼容 /v1 的服务。", ("stream", "vision", "native_tools")),
            ],
        )

    def test_every_preset_uses_a_supported_protocol(self) -> None:
        for preset in COMMON_PROVIDER_PRESETS:
            with self.subTest(preset=preset.id):
                self.assertIn(preset.protocol, SUPPORTED_PROTOCOLS)

    def test_preset_index_is_keyed_by_id_and_last_wins(self) -> None:
        first = ProviderPreset("x", "first", "openai", "", True, "one")
        second = ProviderPreset("x", "second", "openai", "", True, "two")
        index = preset_index([first, second])
        self.assertEqual(list(index), ["x"])
        self.assertEqual(index["x"].label, "second")

    def test_payload_uses_camel_case_keys_and_omits_capabilities_by_default(self) -> None:
        row = provider_presets_payload([COMMON_PROVIDER_PRESETS[0]])[0]
        self.assertEqual(
            list(row),
            ["id", "label", "protocol", "baseUrl", "apiKeyRequired", "description"],
        )

    def test_payload_includes_capabilities_as_a_list_when_asked(self) -> None:
        row = provider_presets_payload([COMMON_PROVIDER_PRESETS[1]], include_capabilities=True)[0]
        self.assertEqual(row["capabilities"], ["stream", "native_tools"])
        self.assertIsInstance(row["capabilities"], list)

    def test_exported_names_are_exactly_the_documented_twelve(self) -> None:
        self.assertEqual(len(provider_config.__all__), 12)
        self.assertEqual(set(provider_config.__all__), {
            "SUPPORTED_PROTOCOLS", "DEFAULT_PROVIDER_ID", "ProviderPreset", "COMMON_PROVIDER_PRESETS",
            "preset_index", "normalize_api_protocol", "normalize_base_url", "normalize_configured_base_url",
            "canonical_provider_id", "infer_provider_id", "base_url_matches_provider", "provider_presets_payload",
        })


class ProviderNormalizeTests(unittest.TestCase):
    """协议与端点归一化。"""

    def test_an_explicit_supported_protocol_wins_over_the_url(self) -> None:
        self.assertEqual(normalize_api_protocol("anthropic", "http://127.0.0.1:11434"), "anthropic")
        self.assertEqual(normalize_api_protocol("OLLAMA", "https://api.openai.com/v1"), "ollama")

    def test_the_protocol_is_guessed_from_the_url_when_unset(self) -> None:
        self.assertEqual(normalize_api_protocol("auto", "http://127.0.0.1:11434"), "ollama")
        self.assertEqual(normalize_api_protocol("", "https://x.com/claude"), "anthropic")
        self.assertEqual(normalize_api_protocol("", "https://api.openai.com/v1"), "openai")

    def test_the_protocol_guess_is_case_insensitive(self) -> None:
        self.assertEqual(normalize_api_protocol("", "HTTP://127.0.0.1:11434"), "ollama")

    def test_ollama_gains_the_default_endpoint_and_v1(self) -> None:
        self.assertEqual(
            normalize_base_url(protocol="ollama", base_url="http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1",
        )
        self.assertEqual(normalize_base_url(protocol="ollama", base_url=""), "http://127.0.0.1:11434/v1")
        self.assertEqual(
            normalize_base_url(protocol="ollama", base_url="http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1",
        )

    def test_other_protocols_only_lose_their_trailing_slash(self) -> None:
        self.assertEqual(normalize_base_url(protocol="openai", base_url="https://x.com/v1/"), "https://x.com/v1")
        self.assertEqual(normalize_base_url(protocol="openai", base_url=""), "")

    def test_configured_urls_drop_the_endpoint_suffixes(self) -> None:
        self.assertEqual(
            normalize_configured_base_url("https://x.com/v1/chat/completions", protocol="openai"),
            "https://x.com/v1",
        )
        self.assertEqual(normalize_configured_base_url("https://x.com/v1/models", protocol="openai"), "https://x.com/v1")

    def test_the_suffix_strip_happens_once_per_suffix_not_in_a_loop(self) -> None:
        # 契约 §4.3 第 4 条：两个尾缀按顺序各检查一次，第二次检查的是剥完后的值。
        # `.../v1/models/chat/completions` 先被剥成 `.../v1/models`，再被第二个尾缀剥成 `.../v1`。
        self.assertEqual(
            normalize_configured_base_url("https://x.com/v1/models/chat/completions", protocol="openai"),
            "https://x.com/v1",
        )
        # 同一尾缀不循环剥离：`.../models/models` 只掉一层。
        self.assertEqual(
            normalize_configured_base_url("https://x.com/v1/models/models", protocol="openai"),
            "https://x.com/v1/models",
        )

    def test_anthropic_messages_path_loses_only_the_messages_part(self) -> None:
        self.assertEqual(
            normalize_configured_base_url("https://api.anthropic.com/v1/messages", protocol="anthropic"),
            "https://api.anthropic.com/v1",
        )
        # 非 anthropic 协议不动它
        self.assertEqual(
            normalize_configured_base_url("https://api.anthropic.com/v1/messages", protocol="openai"),
            "https://api.anthropic.com/v1/messages",
        )

    def test_canonical_id_folds_aliases(self) -> None:
        self.assertEqual(canonical_provider_id("Claude", aliases={"claude": "anthropic"}), "anthropic")
        self.assertEqual(canonical_provider_id("  OPENAI ", aliases={"openai": "x"}), "x")
        self.assertEqual(canonical_provider_id(None), "")
        self.assertEqual(canonical_provider_id("Pinaic"), "pinaic")


class ProviderInferenceTests(unittest.TestCase):
    """从 URL 反推供应商，以及 URL 与供应商的匹配。"""

    def test_priority_order_is_ollama_then_anthropic_then_the_rest(self) -> None:
        self.assertEqual(infer_provider_id(protocol="", base_url="http://127.0.0.1:11434"), "ollama")
        self.assertEqual(infer_provider_id(protocol="anthropic", base_url=""), "anthropic")
        self.assertEqual(infer_provider_id(protocol="", base_url="https://open.bigmodel.cn/api/paas/v4"), "glm")
        self.assertEqual(infer_provider_id(protocol="", base_url="https://api.openai.com/v1"), "openai")
        self.assertEqual(infer_provider_id(protocol="", base_url="https://api.deepseek.com/v1"), "deepseek")
        self.assertEqual(infer_provider_id(protocol="", base_url="https://api.pinaic.com/v1"), "pinaic")
        self.assertEqual(infer_provider_id(protocol="", base_url="https://generativelanguage.googleapis.com/x"), "gemini")

    def test_an_unknown_url_falls_back_to_the_default_id(self) -> None:
        self.assertEqual(infer_provider_id(protocol="", base_url="https://example.com/v1"), DEFAULT_PROVIDER_ID)
        self.assertEqual(infer_provider_id(protocol="", base_url=""), DEFAULT_PROVIDER_ID)

    def test_only_the_official_host_counts_for_inference(self) -> None:
        # 契约 §1.3 的三张表必须分开：推断只认 api.deepseek.com，匹配才认 deepseek.com。
        self.assertEqual(infer_provider_id(protocol="", base_url="https://deepseek.com/x"), DEFAULT_PROVIDER_ID)
        self.assertTrue(base_url_matches_provider("https://deepseek.com/x", "deepseek"))

    def test_the_candidate_set_restricts_inference(self) -> None:
        url = "https://api.pinaic.com/v1"
        self.assertEqual(infer_provider_id(protocol="", base_url=url, provider_ids=["openai"]), DEFAULT_PROVIDER_ID)
        self.assertEqual(infer_provider_id(protocol="", base_url=url, provider_ids=["pinaic"]), "pinaic")
        self.assertEqual(infer_provider_id(protocol="", base_url=url, provider_ids=[]), "pinaic")

    def test_a_protocol_only_match_still_respects_the_candidate_set(self) -> None:
        self.assertEqual(
            infer_provider_id(protocol="ollama", base_url="", provider_ids=["openai"]),
            DEFAULT_PROVIDER_ID,
        )

    def test_inference_is_case_insensitive_on_the_url(self) -> None:
        self.assertEqual(infer_provider_id(protocol="", base_url="HTTPS://API.OPENAI.COM/V1"), "openai")

    def test_matching_unknown_providers_is_false(self) -> None:
        for provider_id in ("", "nope", None):
            with self.subTest(provider_id=repr(provider_id)):
                self.assertFalse(base_url_matches_provider("https://api.openai.com/v1", provider_id or ""))

    def test_matching_uses_domain_roots(self) -> None:
        self.assertTrue(base_url_matches_provider("https://x.open.bigmodel.cn/y", "glm"))
        self.assertTrue(base_url_matches_provider("https://x.com/claude", "anthropic"))
        self.assertTrue(base_url_matches_provider("http://localhost:11434", "ollama"))
        self.assertFalse(base_url_matches_provider("https://api.openai.com/v1", "anthropic"))


# --------------------------------------------------------------------------- #
# tools/native_schema.py
# --------------------------------------------------------------------------- #


class _Handler:
    def __init__(self, tool_type="web_search", instruction="- web_search：搜索公开网页。"):
        self.tool_type = tool_type
        self._instruction = instruction

    def build_prompt_instruction(self):
        return self._instruction


class _MetadataHandler(_Handler):
    class _Meta:
        input_schema = {
            "type": "object",
            "description": "retrieve long-term memory",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    def tool_metadata(self):
        return self._Meta()


class NativeSchemaSurfaceTests(unittest.TestCase):
    """常量与导出面。"""

    def test_name_pattern_and_length_cap(self) -> None:
        self.assertEqual(NATIVE_TOOL_NAME_RE.pattern, r"^[A-Za-z0-9_.:-]{1,80}$")
        self.assertEqual(NATIVE_TOOL_DESCRIPTION_MAX_CHARS, 900)

    def test_exported_names(self) -> None:
        self.assertEqual(
            native_schema.__all__,
            ["NATIVE_TOOL_DESCRIPTION_MAX_CHARS", "NATIVE_TOOL_NAME_RE", "build_openai_native_tool_specs"],
        )

    def test_the_name_pattern_accepts_and_rejects_the_documented_shapes(self) -> None:
        for name in ("web_search", "a.b", "meta:colon", "a-b", "x" * 80):
            with self.subTest(name=name):
                self.assertIsNotNone(NATIVE_TOOL_NAME_RE.fullmatch(name))
        for name in ("a b", "x" * 81, "", "名字", "a/b"):
            with self.subTest(name=name):
                self.assertIsNone(NATIVE_TOOL_NAME_RE.fullmatch(name))


class NativeSchemaBuildTests(unittest.TestCase):
    """spec 构建。"""

    def test_a_non_mapping_input_yields_nothing(self) -> None:
        for value in (None, [], "x", 123):
            with self.subTest(value=repr(value)):
                self.assertEqual(build_openai_native_tool_specs(value), [])

    def test_a_handler_becomes_a_function_spec(self) -> None:
        specs = build_openai_native_tool_specs({"web_search": _Handler()})
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["type"], "function")
        self.assertEqual(specs[0]["function"]["name"], "web_search")
        self.assertIn("搜索公开网页", specs[0]["function"]["description"])
        self.assertEqual(specs[0]["function"]["parameters"]["type"], "object")

    def test_results_are_sorted_by_the_raw_key(self) -> None:
        specs = build_openai_native_tool_specs({
            "z": _Handler("z_tool"),
            "a": _Handler("a_tool"),
            "m": _Handler("m_tool"),
        })
        self.assertEqual([s["function"]["name"] for s in specs], ["a_tool", "m_tool", "z_tool"])

    def test_the_tool_type_wins_over_the_mapping_key(self) -> None:
        specs = build_openai_native_tool_specs({"key_name": _Handler("declared_name")})
        self.assertEqual(specs[0]["function"]["name"], "declared_name")

    def test_duplicate_names_are_kept_once(self) -> None:
        specs = build_openai_native_tool_specs({"b": _Handler("same"), "a": _Handler("same")})
        self.assertEqual([s["function"]["name"] for s in specs], ["same"])

    def test_invalid_and_empty_names_are_dropped(self) -> None:
        specs = build_openai_native_tool_specs({
            "ok": _Handler("ok_tool"),
            "bad": _Handler("bad tool"),
            "": _Handler(""),
        })
        self.assertEqual([s["function"]["name"] for s in specs], ["ok_tool"])

    def test_an_empty_tool_type_falls_back_to_the_mapping_key(self) -> None:
        # 契约 §5.2 第 3 条：`tool_type or raw_key` —— tool_type 为空时用键名。
        specs = build_openai_native_tool_specs({"from_key": _Handler("")})
        self.assertEqual([s["function"]["name"] for s in specs], ["from_key"])

    def test_the_allowed_set_filters(self) -> None:
        specs = build_openai_native_tool_specs(
            {"a": _Handler("one"), "b": _Handler("two")},
            allowed_tool_names={"two"},
        )
        self.assertEqual([s["function"]["name"] for s in specs], ["two"])

    def test_an_empty_allowed_set_means_no_filter(self) -> None:
        specs = build_openai_native_tool_specs(
            {"a": _Handler("one"), "b": _Handler("two")},
            allowed_tool_names=set(),
        )
        self.assertEqual([s["function"]["name"] for s in specs], ["one", "two"])

    def test_metadata_description_beats_the_prompt_instruction(self) -> None:
        specs = build_openai_native_tool_specs({"x": _MetadataHandler("retrieve_memory")})
        function = specs[0]["function"]
        self.assertEqual(function["description"], "retrieve long-term memory")
        self.assertEqual(function["parameters"]["additionalProperties"], False)
        self.assertIn("query", function["parameters"]["required"])
        self.assertNotIn("description", function["parameters"])

    def test_a_raising_metadata_reader_falls_back_to_the_prompt(self) -> None:
        class Raiser(_Handler):
            def tool_metadata(self):
                raise RuntimeError("boom")

        specs = build_openai_native_tool_specs({"x": Raiser("fallback_tool", "fallback text")})
        self.assertEqual(specs[0]["function"]["description"], "fallback text")

    def test_a_raising_prompt_builder_falls_back_to_the_default_line(self) -> None:
        class Raiser(_Handler):
            def build_prompt_instruction(self):
                raise RuntimeError("boom")

        specs = build_openai_native_tool_specs({"x": Raiser("quiet_tool")})
        self.assertEqual(specs[0]["function"]["description"], "Call Shinku tool quiet_tool.")

    def test_a_handler_without_a_prompt_builder_uses_the_default_line(self) -> None:
        class Bare:
            tool_type = "bare_tool"

        specs = build_openai_native_tool_specs({"x": Bare()})
        self.assertEqual(specs[0]["function"]["description"], "Call Shinku tool bare_tool.")

    def test_the_description_is_capped(self) -> None:
        long_handler = _Handler("long_tool", "长句。" * 400)
        specs = build_openai_native_tool_specs({"x": long_handler})
        self.assertEqual(len(specs[0]["function"]["description"]), NATIVE_TOOL_DESCRIPTION_MAX_CHARS)

    def test_a_non_object_schema_is_forced_to_object(self) -> None:
        class Stringy(_Handler):
            class _Meta:
                input_schema = {"type": "string", "properties": {}}

            def tool_metadata(self):
                return self._Meta()

        specs = build_openai_native_tool_specs({"x": Stringy("s")})
        self.assertEqual(specs[0]["function"]["parameters"]["type"], "object")

    def test_a_handler_without_metadata_gets_a_permissive_object(self) -> None:
        specs = build_openai_native_tool_specs({"x": _Handler("plain")})
        self.assertEqual(
            specs[0]["function"]["parameters"],
            {"type": "object", "additionalProperties": True},
        )


class NativeSchemaEnvelopeTests(unittest.TestCase):
    """老式信封子句的剔除。"""

    def test_envelope_clauses_are_dropped_sentence_by_sentence(self) -> None:
        handler = _Handler("env", '第一句正常。调用格式为 {"type": "function"}。第三句正常。')
        specs = build_openai_native_tool_specs({"x": handler})
        description = specs[0]["function"]["description"]
        self.assertIn("第一句正常", description)
        self.assertIn("第三句正常", description)
        self.assertNotIn("调用格式", description)

    def test_every_marker_drops_its_sentence(self) -> None:
        for marker in ("格式为", "调用格式", "tool_call", '{"type"'):
            with self.subTest(marker=marker):
                handler = _Handler("env", f"正文。这句含 {marker} 的字样。")
                specs = build_openai_native_tool_specs({"x": handler})
                self.assertNotIn(marker, specs[0]["function"]["description"])

    def test_dropping_everything_falls_back_to_the_original(self) -> None:
        handler = _Handler("envonly", "调用格式为 tool_call 那套。")
        specs = build_openai_native_tool_specs({"x": handler})
        self.assertEqual(specs[0]["function"]["description"], "调用格式为 tool_call 那套。")


# --------------------------------------------------------------------------- #
# 边界扫描与契约表达式
# --------------------------------------------------------------------------- #


class C1_4BoundaryTests(unittest.TestCase):
    """B1 边界在新代码上的延续，同时是"表达层独立"的直接证据。"""

    C1_4_FILES = (
        SHINKU_PKG / "guards" / "__init__.py",
        SHINKU_PKG / "guards" / "public_think.py",
        SHINKU_PKG / "guards" / "evidence.py",
        SHINKU_PKG / "providers" / "__init__.py",
        SHINKU_PKG / "providers" / "config.py",
        SHINKU_PKG / "tools" / "native_schema.py",
    )

    #: 旧项目的包名与代号（子串匹配，这些串足够独特）。
    FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane")

    #: 旧模块的内部命名。用**精确标识符 token** 比对，理由同 C1-3。
    LEGACY_INTERNAL_NAMES = (
        # public_guard.py
        "_active_thinks",
        "_day",
        "_day_key",
        "_lock",
        "_reset_if_new_day",
        "_timezone",
        "_used_today",
        # evidence_guard.py
        "_UNSUPPORTED_CLAIM_RE",
        "_PROVIDER_SPECULATION_RE",
        "_SENTENCE_RE",
        "_units",
        # native_tool_schema.py
        "_LEGACY_MARKERS",
        "_handler_description",
        "_handler_schema",
        "_metadata_description",
        "_metadata_parameters",
        "_strip_legacy_envelope_clauses",
    )

    #: 旧模块 docstring 里的整句。它们不该以任何形式出现在新文件里。
    LEGACY_PROSE = (
        "Shinku's bounded public-think admission guard.",
        "Coordinate concurrent and daily public thinking requests.",
        "Shinku provider vocabulary and endpoint normalization.",
        "Build conservative provider-native tool schemas for Shinku handlers.",
        "P0-06 事实与工具证据门：不许声称查过、不许拿供应方内部当解释。",
    )

    def _sources(self):
        for path in self.C1_4_FILES:
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

    def test_no_legacy_prose_sentence_survives(self) -> None:
        for path, text in self._sources():
            for sentence in self.LEGACY_PROSE:
                with self.subTest(module=path.name, sentence=sentence[:32]):
                    self.assertNotIn(sentence, text)

    def test_no_legacy_environment_prefix(self) -> None:
        for path, text in self._sources():
            with self.subTest(module=path.name):
                self.assertNotIn("COMPANION_", text)

    def test_the_flat_legacy_module_names_were_not_recreated(self) -> None:
        for name in ("public_guard.py", "evidence_guard.py", "provider_config.py"):
            with self.subTest(name=name):
                self.assertFalse((SHINKU_PKG / name).exists())
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("shinku.public_guard")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("shinku.evidence_guard")

    def test_the_package_markers_do_not_aggregate_submodules(self) -> None:
        guards_pkg = importlib.import_module("shinku.guards")
        self.assertEqual(guards_pkg.__all__, [])
        providers_pkg = importlib.import_module("shinku.providers")
        self.assertEqual(set(providers_pkg.__all__), set(provider_config.__all__))

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        import sys

        for forbidden in ("companion_v01", "code_shared", "akane"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_new_modules_do_not_import_each_other(self) -> None:
        # 守卫、供应商、工具三条线互不依赖。
        self.assertNotIn("shinku.providers", _imported_modules(SHINKU_PKG / "guards" / "public_think.py"))
        self.assertNotIn("shinku.guards", _imported_modules(SHINKU_PKG / "providers" / "config.py"))
        self.assertNotIn("shinku.guards", _imported_modules(SHINKU_PKG / "tools" / "native_schema.py"))

    def test_the_guards_do_not_reach_for_network_or_process_modules(self) -> None:
        for name in ("public_think.py", "evidence.py"):
            modules = _imported_modules(SHINKU_PKG / "guards" / name)
            for forbidden in ("socket", "subprocess", "requests", "urllib"):
                with self.subTest(module=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, modules)

    def test_the_evidence_module_has_no_single_mega_regex(self) -> None:
        # 旧实现是一个含 8 个并列分支的超长正则；新实现改为词表 + 分层判定。
        text = (SHINKU_PKG / "guards" / "evidence.py").read_text(encoding="utf-8")
        longest = max((len(line) for line in text.splitlines()), default=0)
        self.assertLess(longest, 120)


class ContractExpressionTests(unittest.TestCase):
    """契约里那些"不会让既有测试变红"的表达式，逐条定点钉住。"""

    def _flatten(self, path: Path) -> str:
        return " ".join(path.read_text(encoding="utf-8").split())

    def test_the_three_signal_tables_are_kept_separate(self) -> None:
        text = self._flatten(SHINKU_PKG / "providers" / "config.py")
        self.assertIn("_PROTOCOL_SIGNALS", text)
        self.assertIn("_INFER_SIGNALS", text)
        self.assertIn("_MATCH_SIGNALS", text)

    def test_the_inference_table_uses_the_api_host_for_deepseek(self) -> None:
        self.assertIn("api.deepseek.com", self._flatten(SHINKU_PKG / "providers" / "config.py"))
        # 匹配表用的是域名根
        self.assertIn('"deepseek": ("deepseek.com",)', self._flatten(SHINKU_PKG / "providers" / "config.py"))

    def test_the_protocol_signal_for_anthropic_is_the_loose_one(self) -> None:
        # 协议推断用 "anthropic"（宽松），provider 推断用 "anthropic.com"（严格）。
        text = self._flatten(SHINKU_PKG / "providers" / "config.py")
        self.assertIn('"anthropic": ("anthropic", "/claude")', text)

    def test_the_guard_reads_messages_with_the_or_fallback_reader(self) -> None:
        text = self._flatten(SHINKU_PKG / "guards" / "public_think.py")
        self.assertIn("str(given or fallback).strip()", text)

    def test_the_guard_release_uses_max_zero(self) -> None:
        text = self._flatten(SHINKU_PKG / "guards" / "public_think.py")
        self.assertIn("max(0, self._in_flight - 1)", text)

    def test_the_daily_quota_is_checked_before_concurrency(self) -> None:
        text = self._flatten(SHINKU_PKG / "guards" / "public_think.py")
        self.assertLess(text.index("_quota_spent()"), text.index("_saturated()"))

    def test_the_completion_marks_do_not_include_a_bare_wan(self) -> None:
        # 「我查完就告诉你」不算声称查过：完成标记里没有单独的「完」。
        text = self._flatten(SHINKU_PKG / "guards" / "evidence.py")
        self.assertIn('_COMPLETION_MARKS = ("了", "过", "完了", "了一遍", "过了")', text)

    def test_the_envelope_markers_are_preserved_verbatim(self) -> None:
        text = self._flatten(SHINKU_PKG / "tools" / "native_schema.py")
        self.assertIn('"格式为", "调用格式", "tool_call", \'{"type"\'', text)

    def test_the_tool_description_cap_is_900(self) -> None:
        self.assertEqual(NATIVE_TOOL_DESCRIPTION_MAX_CHARS, 900)

    def test_the_payload_keys_are_camel_case_in_source(self) -> None:
        text = self._flatten(SHINKU_PKG / "providers" / "config.py")
        self.assertIn('"baseUrl": preset.base_url', text)
        self.assertIn('"apiKeyRequired": preset.api_key_required', text)


if __name__ == "__main__":
    unittest.main()

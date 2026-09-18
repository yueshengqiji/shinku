"""C2-1 上游静默开关、文本切词与嵌入供应商的契约测试。

本批按来源记录 §5.9 的**路径 A 第二类**处理：四个模块都含逻辑、分支或算法，属**洁净室重写**。
断言对象全部是**外部可观察行为**（契约 `docs/contracts/c2_1_provider_layer.md`），
不涉及任何实现结构。

五处刻意的地方：

1. **输入语料在本测试内组织。** 文本、模型名、环境变量预设全部现写；
   不引用旧项目的测试文件或夹具。
2. **时钟靠注入，不靠等待。** 熔断器的三个读数全走 `time.time()`，真实冷却 60s 起，
   没法等，所以整段行为测试在 `mock.patch("time.time", …)` 里跑。
3. **`sentence_transformers` 用假模块。** 真的 `__init__` 会 import 并 encode。
   假 encoder 只收 `**kwargs`，于是「实现到底传了哪几个关键字」是可观测的；
   「签名里有没有 `local_files_only`」用 `__signature__` 单独控制，两者互不干扰。
4. **契约里那些"不会让任何既有测试变红"的表达式**在这里被逐条钉住。
   典型是哈希向量的常数（sha256／`digest[:4]` 大端取模／`digest[4]` 奇偶定号／
   权重 `1 + len/10`／L2 归一化）——它们改一个字节，向量就整体不同，
   但没有任何别的地方会因此报错。
5. **边界扫描用 `tokenize` 取精确标识符**，不用子串匹配（理由同 C1-3／C1-4：
   本批旧私有名里有 `_get`／`_put`／`_model` 这种短词，子串匹配会误判）。
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import math
import os
import sys
import tokenize as py_tokenize
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from shinku.llm import circuit_breaker as breaker_module
from shinku.llm.circuit_breaker import LlmCircuitBreaker, get_llm_circuit_breaker
from shinku.providers import embedding as embedding_module
from shinku.providers import huggingface as huggingface_module
from shinku.providers.embedding import (
    BaseEmbeddingProvider,
    CachedEmbeddingProvider,
    HashedEmbeddingProvider,
)
from shinku.providers.huggingface import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    HuggingFaceEmbeddingProvider,
)
from shinku.text import tokenizer as tokenizer_module
from shinku.text.tokenizer import STOPWORDS, TOKEN_RE, normalize_text, tokenize

SHINKU_PKG = Path(embedding_module.__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _identifiers(path: Path) -> set[str]:
    """文件里出现过的所有 Python 标识符（精确 token，不是子串）。

    注意用的是标准库的 ``tokenize``（这里别名成 ``py_tokenize``），
    不是本批被测的那个同名函数。
    """

    with path.open(encoding="utf-8") as handle:
        return {
            item.string
            for item in py_tokenize.generate_tokens(handle.readline)
            if item.type == py_tokenize.NAME
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


class _Clock:
    """可控时钟。冷却动辄 60 秒起，只能靠它把时间拨过去。"""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


@contextmanager
def _frozen_clock(start: float = 1_000.0):
    clock = _Clock(start)
    with mock.patch("time.time", new=clock):
        yield clock


def _fake_sentence_transformers(*, accepts_local_files_only: bool = True,
                                dimension: int | None = 768):
    """假 sentence_transformers 模块 + 调用日志。

    ``accepts_local_files_only`` 只改 `__signature__`，不改 `__init__` 的收参方式——
    这样「实现传了哪些关键字」与「签名里有没有那个参数」两件事可以分开断言。
    """

    log: dict[str, list] = {"builds": [], "encodes": []}

    class SentenceTransformer:
        def __init__(self, model_name=None, **kwargs):
            self.model_name = model_name
            self.kwargs = dict(kwargs)
            log["builds"].append({
                "model_name": model_name,
                "kwargs": dict(kwargs),
                "env": (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("HF_ENDPOINT")),
            })

        def get_sentence_embedding_dimension(self):
            return dimension

        def encode(self, texts, **kwargs):
            values = [str(text) for text in texts]
            log["encodes"].append({"texts": values, "kwargs": dict(kwargs)})
            return [[float(len(text)), 0.5, 0.25] for text in values]

    if accepts_local_files_only:
        SentenceTransformer.__signature__ = inspect.Signature([
            inspect.Parameter("model_name", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None),
            inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("cache_folder", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("local_files_only", inspect.Parameter.KEYWORD_ONLY,
                              default=False),
            inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
        ])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    return module, log


@contextmanager
def _with_fake_hf(**kwargs):
    module, log = _fake_sentence_transformers(**kwargs)
    with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
        yield log


class _EchoProvider(BaseEmbeddingProvider):
    """只回报文本长度的最小 provider，用来观测基类的批量语义。"""

    def embed_text(self, text: str) -> list[float]:
        return [float(len(str(text or "")))]


class _CountingProvider(BaseEmbeddingProvider):
    """记录每一次取数请求，用请求序列反推缓存命中与淘汰。"""

    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self.requests: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.requests.append(text)
        if not text:
            return []
        return [float(len(text)), 1.0]


# --------------------------------------------------------------------------- #
# llm/circuit_breaker.py
# --------------------------------------------------------------------------- #


class CircuitBreakerSurfaceTests(unittest.TestCase):
    """对外面：签名、公开属性、快照结构、导出。"""

    def test_exports_are_exactly_the_documented_two(self) -> None:
        self.assertEqual(sorted(breaker_module.__all__),
                         ["LlmCircuitBreaker", "get_llm_circuit_breaker"])

    def test_constructor_takes_three_keyword_only_arguments(self) -> None:
        parameters = inspect.signature(LlmCircuitBreaker).parameters
        self.assertEqual(list(parameters), [
            "failure_threshold", "base_cooldown_seconds", "max_cooldown_seconds",
        ])
        for name in parameters:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["failure_threshold"].default, 3)
        self.assertEqual(parameters["base_cooldown_seconds"].default, 60.0)
        self.assertEqual(parameters["max_cooldown_seconds"].default, 600.0)

    def test_methods_take_only_self(self) -> None:
        for name in ("record_success", "record_failure", "is_open",
                     "remaining_seconds", "snapshot"):
            with self.subTest(method=name):
                self.assertEqual(list(inspect.signature(
                    getattr(LlmCircuitBreaker, name)).parameters), ["self"])

    def test_normalized_values_are_public_attributes(self) -> None:
        breaker = LlmCircuitBreaker(failure_threshold=4, base_cooldown_seconds=30.0,
                                    max_cooldown_seconds=90.0)
        self.assertEqual(breaker.failure_threshold, 4)
        self.assertEqual(breaker.base_cooldown_seconds, 30.0)
        self.assertEqual(breaker.max_cooldown_seconds, 90.0)

    def test_snapshot_exposes_exactly_the_six_documented_keys(self) -> None:
        with _frozen_clock():
            snapshot = LlmCircuitBreaker().snapshot()
        self.assertEqual(sorted(snapshot), [
            "consecutiveFailures", "lastFailureAt", "lastSuccessAt",
            "open", "openCount", "remainingSeconds",
        ])
        self.assertIsInstance(snapshot["open"], bool)
        self.assertIsInstance(snapshot["consecutiveFailures"], int)
        self.assertIsInstance(snapshot["openCount"], int)
        self.assertIsInstance(snapshot["remainingSeconds"], float)
        self.assertIsInstance(snapshot["lastFailureAt"], float)
        self.assertIsInstance(snapshot["lastSuccessAt"], float)

    def test_the_module_does_not_touch_time_io_or_the_environment(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "llm" / "circuit_breaker.py")
        for forbidden in ("socket", "subprocess", "requests", "urllib", "os"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, modules)
        self.assertEqual(modules & {"threading", "time"}, {"threading", "time"})


class CircuitBreakerNormalizationTests(unittest.TestCase):
    """构造参数就地归一化，边界是**闭区间**。"""

    def test_threshold_is_floored_at_one(self) -> None:
        for raw, expected in ((0, 1), (-3, 1), (1, 1), (5, 5), (3.9, 3)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    LlmCircuitBreaker(failure_threshold=raw).failure_threshold, expected)

    def test_base_cooldown_is_floored_at_one_second(self) -> None:
        for raw, expected in ((0, 1.0), (0.0, 1.0), (-9.0, 1.0), (0.5, 1.0), (60, 60.0)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    LlmCircuitBreaker(base_cooldown_seconds=raw).base_cooldown_seconds,
                    expected)

    def test_max_cooldown_is_raised_to_the_base(self) -> None:
        breaker = LlmCircuitBreaker(base_cooldown_seconds=60.0, max_cooldown_seconds=10.0)
        self.assertEqual(breaker.max_cooldown_seconds, 60.0)
        keeper = LlmCircuitBreaker(base_cooldown_seconds=60.0, max_cooldown_seconds=600.0)
        self.assertEqual(keeper.max_cooldown_seconds, 600.0)

    def test_the_initial_state_is_all_zero(self) -> None:
        with _frozen_clock(777.0):
            breaker = LlmCircuitBreaker()
            self.assertEqual(breaker.snapshot(), {
                "open": False, "consecutiveFailures": 0, "openCount": 0,
                "remainingSeconds": 0.0, "lastFailureAt": 0.0, "lastSuccessAt": 0.0,
            })
            self.assertFalse(breaker.is_open())
            self.assertEqual(breaker.remaining_seconds(), 0.0)


class CircuitBreakerBehaviourTests(unittest.TestCase):
    """注入时钟后的完整行为。"""

    def test_failures_below_the_threshold_only_count(self) -> None:
        with _frozen_clock() as clock:
            breaker = LlmCircuitBreaker(failure_threshold=3)
            breaker.record_failure()
            breaker.record_failure()
            clock.advance(5.0)
            snapshot = breaker.snapshot()
        self.assertEqual(snapshot["consecutiveFailures"], 2)
        self.assertEqual(snapshot["openCount"], 0)
        self.assertFalse(snapshot["open"])
        self.assertEqual(snapshot["lastFailureAt"], 1_000.0)

    def test_reaching_the_threshold_opens_for_the_base_cooldown(self) -> None:
        with _frozen_clock() as clock:
            breaker = LlmCircuitBreaker(failure_threshold=3)
            for _ in range(3):
                breaker.record_failure()
            self.assertTrue(breaker.is_open())
            self.assertEqual(breaker.remaining_seconds(), 60.0)
            clock.advance(60.0)
            self.assertFalse(breaker.is_open())
            self.assertEqual(breaker.remaining_seconds(), 0.0)

    def test_every_extra_failure_trips_again_with_exponential_backoff(self) -> None:
        with _frozen_clock():
            breaker = LlmCircuitBreaker(failure_threshold=3, base_cooldown_seconds=60.0,
                                        max_cooldown_seconds=600.0)
            observed = []
            for _ in range(8):
                breaker.record_failure()
                observed.append((breaker.snapshot()["openCount"],
                                 breaker.remaining_seconds()))
        self.assertEqual(observed, [
            (0, 0.0), (0, 0.0), (1, 60.0), (2, 120.0), (3, 240.0),
            (4, 480.0), (5, 600.0), (6, 600.0),
        ])

    def test_success_clears_every_counter(self) -> None:
        with _frozen_clock() as clock:
            breaker = LlmCircuitBreaker(failure_threshold=2)
            for _ in range(6):
                breaker.record_failure()
            clock.advance(1.0)
            breaker.record_success()
            snapshot = breaker.snapshot()
        self.assertEqual(snapshot["consecutiveFailures"], 0)
        self.assertEqual(snapshot["openCount"], 0)
        self.assertEqual(snapshot["lastFailureAt"], 1_000.0)
        self.assertEqual(snapshot["lastSuccessAt"], 1_001.0)
        self.assertFalse(snapshot["open"])

    def test_success_stamps_the_time_even_without_any_failure(self) -> None:
        with _frozen_clock(2_048.5):
            breaker = LlmCircuitBreaker()
            breaker.record_success()
            snapshot = breaker.snapshot()
        self.assertEqual(snapshot["lastSuccessAt"], 2_048.5)
        self.assertEqual(snapshot["lastFailureAt"], 0.0)

    def test_the_two_timestamps_are_rounded_to_milliseconds(self) -> None:
        with _frozen_clock(1_000.123456):
            breaker = LlmCircuitBreaker()
            breaker.record_failure()
            breaker.record_success()
            snapshot = breaker.snapshot()
        self.assertEqual(snapshot["lastFailureAt"], round(1_000.123456, 3))
        self.assertEqual(snapshot["lastSuccessAt"], round(1_000.123456, 3))

    def test_remaining_seconds_is_rounded_to_one_decimal(self) -> None:
        with _frozen_clock(500.0) as clock:
            breaker = LlmCircuitBreaker(failure_threshold=1, base_cooldown_seconds=3.14)
            breaker.record_failure()
            clock.advance(0.06)
            snapshot = breaker.snapshot()
        self.assertEqual(snapshot["remainingSeconds"], round(3.14 - 0.06, 1))

    def test_remaining_seconds_never_goes_negative(self) -> None:
        with _frozen_clock() as clock:
            breaker = LlmCircuitBreaker(failure_threshold=1)
            breaker.record_failure()
            clock.advance(10_000.0)
            self.assertEqual(breaker.remaining_seconds(), 0.0)


class CircuitBreakerSingletonTests(unittest.TestCase):
    """进程级单例：调用层与事件层看到同一份状态。"""

    def test_the_accessor_returns_one_shared_instance(self) -> None:
        self.assertIs(get_llm_circuit_breaker(), get_llm_circuit_breaker())
        self.assertIsInstance(get_llm_circuit_breaker(), LlmCircuitBreaker)

    def test_the_shared_instance_is_an_ordinary_breaker(self) -> None:
        with _frozen_clock():
            shared = get_llm_circuit_breaker()
            shared.record_success()
            self.assertEqual(shared.snapshot()["consecutiveFailures"], 0)
            self.assertIsInstance(shared.failure_threshold, int)


# --------------------------------------------------------------------------- #
# text/tokenizer.py
# --------------------------------------------------------------------------- #


class TokenizerSurfaceTests(unittest.TestCase):
    """对外面：正则、词表、导出。"""

    def test_exports_are_exactly_the_documented_four(self) -> None:
        self.assertEqual(sorted(tokenizer_module.__all__),
                         ["STOPWORDS", "TOKEN_RE", "normalize_text", "tokenize"])

    def test_token_pattern_is_the_documented_one(self) -> None:
        self.assertEqual(TOKEN_RE.pattern, r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")

    def test_stopwords_are_exactly_the_documented_twenty_three(self) -> None:
        self.assertEqual(STOPWORDS, {
            "的", "了", "呢", "啊", "呀", "吗", "吧", "哦", "喵", "我", "你", "他", "她", "它",
            "我们", "你们", "他们", "然后", "就是", "这个", "那个", "现在", "一下",
        })
        self.assertEqual(len(STOPWORDS), 23)

    def test_signatures(self) -> None:
        self.assertEqual(list(inspect.signature(normalize_text).parameters), ["text"])
        self.assertEqual(list(inspect.signature(tokenize).parameters), ["text"])


class TokenizerNormalizationTests(unittest.TestCase):
    def test_blank_and_missing_input_become_empty(self) -> None:
        for raw in (None, "", "   ", "\t\n  "):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), "")

    def test_whitespace_runs_collapse_to_one_space(self) -> None:
        self.assertEqual(normalize_text("  两个   空白\t分隔  "), "两个 空白 分隔")

    def test_non_string_input_is_coerced(self) -> None:
        self.assertEqual(normalize_text(123), "123")
        self.assertEqual(normalize_text(12.5), "12.5")

    def test_blank_input_tokenizes_to_nothing(self) -> None:
        for raw in (None, "", "   ", "\t\n"):
            with self.subTest(raw=raw):
                self.assertEqual(tokenize(raw), [])


class TokenizerRuleTests(unittest.TestCase):
    """切词规则：小写化在前、汉字整串 + 滑窗、不过滤重复。"""

    def test_latin_is_lowercased_before_matching(self) -> None:
        self.assertEqual(tokenize("HELLO World"), ["hello", "world"])
        self.assertEqual(tokenize("ABC_123"), ["abc_123"])

    def test_digits_and_underscores_survive_whole(self) -> None:
        self.assertEqual(tokenize("a_b 12 3.5"), ["a_b", "12", "3", "5"])

    def test_short_han_runs_are_kept_whole(self) -> None:
        self.assertEqual(tokenize("你好"), ["你好"])
        self.assertEqual(tokenize("三四五"), ["三四五"])

    def test_a_three_character_run_produces_no_sliding_pieces(self) -> None:
        self.assertEqual(tokenize("三四五"), ["三四五"])

    def test_a_four_character_run_produces_three_bigrams_and_two_trigrams(self) -> None:
        self.assertEqual(tokenize("三四五六"),
                         ["三四五六", "三四", "四五", "五六", "三四五", "四五六"])

    def test_a_five_character_run_produces_four_bigrams_and_three_trigrams(self) -> None:
        pieces = tokenize("三四五六七")
        self.assertEqual(pieces, ["三四五六七",
                                  "三四", "四五", "五六", "六七",
                                  "三四五", "四五六", "五六七"])
        self.assertEqual(len(pieces), 8)

    def test_stopwords_are_dropped_after_matching(self) -> None:
        self.assertEqual(tokenize("的 了 呢"), [])

    def test_a_three_character_run_that_is_not_a_stopword_survives_whole(self) -> None:
        # 「的了呢」是三个停用词连写，但它整串不在词表里，也没有滑窗片段，
        # 于是原样留下——过滤按 token 比对，不做「字字都停用就丢掉」的二次判断。
        self.assertEqual(tokenize("的了呢"), ["的了呢"])

    def test_sliding_pieces_are_filtered_by_the_same_table(self) -> None:
        # 长串里切出来的片段要与词表比对：「就是」是停用词，作为片段也要被丢掉，
        # 而它的邻居「二就」「是三」照旧留下。
        pieces = tokenize("一二就是三四")
        self.assertEqual(pieces, ["一二就是三四", "一二", "二就", "是三", "三四",
                                  "一二就", "二就是", "就是三", "是三四"])
        self.assertNotIn("就是", pieces)

    def test_duplicates_are_not_collapsed(self) -> None:
        self.assertEqual(tokenize("一二三 一二三"), ["一二三", "一二三"])

    def test_mixed_scripts_are_split_at_script_boundaries(self) -> None:
        # 汉字与拉丁之间断开，但连续的汉字仍是一整段——「与汉字」不会被拆开。
        self.assertEqual(tokenize("混排English与汉字"),
                         ["混排", "english", "与汉字"])

    def test_emoji_and_punctuation_are_dropped(self) -> None:
        self.assertEqual(tokenize("😀 测试！"), ["测试"])


# --------------------------------------------------------------------------- #
# providers/embedding.py
# --------------------------------------------------------------------------- #


class EmbeddingBaseSurfaceTests(unittest.TestCase):
    """基类：抽象面、命名、集合键。"""

    def test_exports_are_exactly_the_documented_three(self) -> None:
        self.assertEqual(sorted(embedding_module.__all__),
                         ["BaseEmbeddingProvider", "CachedEmbeddingProvider",
                          "HashedEmbeddingProvider"])

    def test_the_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseEmbeddingProvider(dimension=8)
        with self.assertRaises(TypeError):
            type("NoEmbed", (BaseEmbeddingProvider,), {})(dimension=8)

    def test_constructor_takes_dimension_as_keyword_only(self) -> None:
        parameters = inspect.signature(BaseEmbeddingProvider.__init__).parameters
        self.assertEqual(list(parameters), ["self", "dimension"])
        self.assertEqual(parameters["dimension"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_dimension_is_floored_at_one(self) -> None:
        for raw, expected in ((-5, 1), (0, 1), (1, 1), (128, 128), (7.9, 7)):
            with self.subTest(raw=raw):
                self.assertEqual(_EchoProvider(dimension=raw).dimension, expected)

    def test_name_falls_back_to_base(self) -> None:
        for raw, expected in (("base", "base"), ("", "base"), ("   ", "base"),
                              ("My Provider", "My Provider")):
            provider = type("Probe", (_EchoProvider,), {"provider_name": raw})(dimension=1)
            with self.subTest(raw=raw):
                self.assertEqual(provider.name, expected)

    def test_the_base_class_has_no_legacy_collection_name(self) -> None:
        self.assertIsNone(_EchoProvider(dimension=8).legacy_collection_name)

    def test_collection_key_is_a_lowercased_slug(self) -> None:
        provider = type("Probe", (_EchoProvider,),
                        {"provider_name": "Hashed", "version": "V1"})(dimension=128)
        self.assertEqual(provider.collection_key(), "hashed_v1_128")

    def test_collection_key_squeezes_illegal_runs_and_strips_edges(self) -> None:
        provider = type("Probe", (_EchoProvider,),
                        {"provider_name": "  a...b  ", "version": "v2"})(dimension=8)
        self.assertEqual(provider.collection_key(), "a_b_v2_8")

    def test_the_dimension_suffix_always_survives_even_for_illegal_names(self) -> None:
        # 名字与版本全是非法字符时，非法段被压成下划线再削掉，
        # 但维度贡献的数字一定还在——所以集合键永远非空，
        # 契约里那句「为空则取 embedding」在公开路径上取不到（实施如实登记）。
        provider = type("Probe", (_EchoProvider,),
                        {"provider_name": "。", "version": "。"})(dimension=1)
        self.assertEqual(provider.collection_key(), "1")

    def test_batch_entry_point_neither_deduplicates_nor_filters(self) -> None:
        self.assertEqual(_EchoProvider(dimension=1).embed_texts(["a", "bb", "a", ""]),
                         [[1.0], [2.0], [1.0], [0.0]])

    def test_the_embed_text_error_is_not_swallowed(self) -> None:
        with self.assertRaises(NotImplementedError):
            BaseEmbeddingProvider.embed_text(_EchoProvider(dimension=1), "x")


class HashedEmbeddingProviderTests(unittest.TestCase):
    """哈希词袋：默认值、旧集合名门控、向量算法。"""

    def test_defaults_are_the_documented_ones(self) -> None:
        parameters = inspect.signature(HashedEmbeddingProvider.__init__).parameters
        self.assertEqual(parameters["dimension"].default, 128)
        self.assertEqual(parameters["legacy_collection_name"].default, "shinku_memory_v01")
        provider = HashedEmbeddingProvider()
        self.assertEqual(provider.dimension, 128)
        self.assertEqual((provider.name, provider.version), ("hashed", "v1"))

    def test_the_legacy_name_is_blank_normalized(self) -> None:
        for raw, expected in ((None, None), ("", None), ("   ", None),
                              ("  legacy  ", "legacy")):
            with self.subTest(raw=raw):
                provider = HashedEmbeddingProvider(legacy_collection_name=raw)
                self.assertEqual(provider.legacy_collection_name, expected)

    def test_the_legacy_name_is_only_offered_at_the_original_shape(self) -> None:
        self.assertEqual(HashedEmbeddingProvider(dimension=128).legacy_collection_name,
                         "shinku_memory_v01")
        self.assertIsNone(HashedEmbeddingProvider(dimension=64).legacy_collection_name)
        self.assertIsNone(HashedEmbeddingProvider(dimension=129).legacy_collection_name)

    def test_the_legacy_name_is_gated_on_the_version_too(self) -> None:
        provider = type("Probe", (HashedEmbeddingProvider,), {"version": "v2"})(
            dimension=128, legacy_collection_name="shinku_memory_v01")
        self.assertIsNone(provider.legacy_collection_name)
        self.assertEqual(provider.collection_key(), "hashed_v2_128")

    def test_vectors_are_l2_normalized(self) -> None:
        provider = HashedEmbeddingProvider(dimension=32)
        vector = provider.embed_text("今天天气不错，我们一起去公园散步吧")
        self.assertEqual(len(vector), 32)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0,
                               places=12)

    def test_a_text_without_any_surviving_token_gives_the_all_zero_vector(self) -> None:
        provider = HashedEmbeddingProvider(dimension=16)
        self.assertEqual(provider.embed_text("的 了 呢"), [0.0] * 16)
        self.assertEqual(provider.embed_text(""), [0.0] * 16)

    def test_the_algorithm_matches_an_independent_reimplementation(self) -> None:
        dimension = 24
        provider = HashedEmbeddingProvider(dimension=dimension)

        def expected(text: str) -> list[float]:
            vector = [0.0] * dimension
            for piece in tokenize(text):
                digest = hashlib.sha256(piece.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:4], "big") % dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[slot] += sign * (1.0 + len(piece) / 10.0)
            norm = math.sqrt(sum(value * value for value in vector))
            return [value / norm for value in vector] if norm else vector

        for text in ("你好世界", "混排 English 与汉字 123", "一二三四五六七八九十",
                     "__x__ a_b", "今天天气不错，我们一起去公园散步吧！"):
            with self.subTest(text=text):
                self.assertEqual(provider.embed_text(text), expected(text))

    def test_the_vector_width_follows_the_dimension(self) -> None:
        for dimension in (1, 2, 128, 300):
            with self.subTest(dimension=dimension):
                self.assertEqual(len(HashedEmbeddingProvider(dimension=dimension)
                                     .embed_text("你好")), dimension)

    def test_the_same_text_always_gives_the_same_vector(self) -> None:
        # 无随机盐、无残留状态：同一次会话里两次调用必须逐位相同。
        provider = HashedEmbeddingProvider(dimension=16)
        self.assertEqual(provider.embed_text("你好世界"), provider.embed_text("你好世界"))
        self.assertEqual(HashedEmbeddingProvider(dimension=16).embed_text("你好世界"),
                         provider.embed_text("你好世界"))


class CachedEmbeddingProviderTests(unittest.TestCase):
    """有界 LRU：转发面、直通、淘汰顺序、副本安全。"""

    def test_inner_and_max_entries_are_public(self) -> None:
        inner = _CountingProvider(dimension=4)
        provider = CachedEmbeddingProvider(inner, max_entries=16)
        self.assertIs(provider.inner, inner)
        self.assertEqual(provider.max_entries, 16)

    def test_max_entries_is_floored_at_zero(self) -> None:
        for raw, expected in ((-3, 0), (0, 0), (7, 7), (2.9, 2)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    CachedEmbeddingProvider(_CountingProvider(), max_entries=raw).max_entries,
                    expected)

    def test_the_wrapper_forwards_the_inner_face(self) -> None:
        inner = HashedEmbeddingProvider(dimension=64, legacy_collection_name="legacy")
        provider = CachedEmbeddingProvider(inner, max_entries=4)
        self.assertEqual(provider.name, inner.name)
        self.assertEqual(provider.version, inner.version)
        self.assertEqual(provider.dimension, inner.dimension)
        self.assertEqual(provider.legacy_collection_name, inner.legacy_collection_name)
        self.assertEqual(provider.collection_key(), inner.collection_key())

    def test_version_is_a_property_on_the_wrapper_and_a_class_attribute_on_the_base(self) -> None:
        # 这是一处真实的接口不对称：换名字会让 `cached.version` 静态化，
        # 于是内层换模型时集合键就不再跟着变。
        self.assertIsInstance(CachedEmbeddingProvider.__dict__["version"], property)
        self.assertIsInstance(BaseEmbeddingProvider.__dict__["version"], str)
        self.assertIsInstance(HashedEmbeddingProvider.__dict__["version"], str)

    def test_a_hit_does_not_ask_the_inner_provider_again(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=4)
        first = provider.embed_text("abc")
        second = provider.embed_text("abc")
        self.assertEqual(first, second)
        self.assertEqual(inner.requests, ["abc"])

    def test_results_are_copies(self) -> None:
        provider = CachedEmbeddingProvider(_CountingProvider(), max_entries=4)
        first = provider.embed_text("abc")
        first.append(99.0)
        self.assertEqual(provider.embed_text("abc"), [3.0, 1.0])

    def test_zero_capacity_passes_everything_through(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=0)
        provider.embed_text("a")
        provider.embed_text("a")
        self.assertEqual(provider.embed_texts(["a", "a"]), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(inner.requests, ["a", "a", "a", "a"])

    def test_the_least_recently_used_entry_is_evicted_first(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=2)
        provider.embed_text("a")
        provider.embed_text("b")
        provider.embed_text("a")      # 命中，把 a 挪到队尾；此时顺序是 b, a
        provider.embed_text("c")      # 写入 c，挤掉队头的 b
        provider.embed_text("b")      # b 已不在，必须重新算
        self.assertEqual(inner.requests, ["a", "b", "c", "b"])

    def test_a_batch_fills_every_position_and_reuses_duplicates(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=8)
        result = provider.embed_texts(["x", "y", "x", "z"])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], result[2])
        self.assertEqual(inner.requests, ["x", "y", "z"])
        self.assertNotIn(None, result)

    def test_the_batch_entry_point_passes_the_deduplicated_texts_in_first_seen_order(self) -> None:
        seen: list[list[str]] = []

        class Recorder(_CountingProvider):
            def embed_texts(self, texts):
                values = list(texts)
                seen.append(values)
                return [self.embed_text(value) for value in values]

        provider = CachedEmbeddingProvider(Recorder(), max_entries=8)
        provider.embed_texts(["b", "a", "b", "c", "a"])
        self.assertEqual(seen, [["b", "a", "c"]])

    def test_an_empty_inner_vector_is_cached_like_any_other(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=4)
        self.assertEqual(provider.embed_text(""), [])
        self.assertEqual(provider.embed_text(""), [])
        self.assertEqual(inner.requests, [""])

    def test_a_batch_of_nothing_asks_for_nothing(self) -> None:
        inner = _CountingProvider()
        provider = CachedEmbeddingProvider(inner, max_entries=4)
        self.assertEqual(provider.embed_texts([]), [])
        self.assertEqual(inner.requests, [])


# --------------------------------------------------------------------------- #
# providers/huggingface.py
# --------------------------------------------------------------------------- #


class HuggingFaceSurfaceTests(unittest.TestCase):
    def test_exports_are_exactly_the_documented_two(self) -> None:
        self.assertEqual(sorted(huggingface_module.__all__),
                         ["DEFAULT_HUGGINGFACE_EMBEDDING_MODEL",
                          "HuggingFaceEmbeddingProvider"])

    def test_the_default_model_is_the_documented_one(self) -> None:
        self.assertEqual(DEFAULT_HUGGINGFACE_EMBEDDING_MODEL, "BAAI/bge-m3")

    def test_constructor_takes_six_keyword_only_arguments_with_defaults(self) -> None:
        parameters = inspect.signature(HuggingFaceEmbeddingProvider.__init__).parameters
        self.assertEqual(list(parameters),
                         ["self", "model_name", "device", "local_files_only",
                          "cache_folder", "hf_endpoint", "normalize_embeddings"])
        for name in ("model_name", "device", "local_files_only", "cache_folder",
                     "hf_endpoint", "normalize_embeddings"):
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["model_name"].default, DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
        self.assertIsNone(parameters["device"].default)
        self.assertFalse(parameters["local_files_only"].default)
        self.assertIsNone(parameters["cache_folder"].default)
        self.assertIsNone(parameters["hf_endpoint"].default)
        self.assertTrue(parameters["normalize_embeddings"].default)

    def test_provider_name_is_huggingface(self) -> None:
        self.assertEqual(HuggingFaceEmbeddingProvider.provider_name, "huggingface")

    def test_importing_the_module_does_not_import_the_optional_dependency(self) -> None:
        # 没装 sentence-transformers 时也能 import 本模块——重活必须推迟到构造期。
        self.assertNotIn("sentence_transformers", sys.modules)


class HuggingFaceConstructionTests(unittest.TestCase):
    def test_parameters_are_normalized(self) -> None:
        with _with_fake_hf(dimension=8) as log:
            provider = HuggingFaceEmbeddingProvider(
                model_name="  ", device="  ", local_files_only=1,
                cache_folder="   ", hf_endpoint="  https://mirror.example/  ",
                normalize_embeddings=0)
        self.assertEqual(provider.model_name, DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
        self.assertIsNone(provider.device)
        self.assertIs(provider.local_files_only, True)
        self.assertIsNone(provider.cache_folder)
        self.assertEqual(provider.hf_endpoint, "https://mirror.example")
        self.assertIs(provider.normalize_embeddings, False)
        self.assertEqual(log["builds"][0]["model_name"], DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)

    def test_version_is_the_model_name(self) -> None:
        with _with_fake_hf():
            provider = HuggingFaceEmbeddingProvider(model_name="  some/model  ")
        self.assertEqual(provider.version, "some/model")
        self.assertEqual(provider.collection_key(), "huggingface_some_model_768")

    def test_dimension_is_asked_from_the_encoder(self) -> None:
        with _with_fake_hf(dimension=384) as log:
            provider = HuggingFaceEmbeddingProvider()
        self.assertEqual(provider.dimension, 384)
        self.assertEqual(log["encodes"], [])

    def test_a_falsy_dimension_falls_back_to_a_probe_run(self) -> None:
        with _with_fake_hf(dimension=None) as log:
            provider = HuggingFaceEmbeddingProvider()
        self.assertEqual(provider.dimension, 3)
        self.assertEqual(len(log["encodes"]), 1)
        self.assertEqual(log["encodes"][0]["texts"], ["探针"])
        self.assertEqual(log["encodes"][0]["kwargs"], {
            "normalize_embeddings": True, "convert_to_numpy": True,
            "show_progress_bar": False,
        })

    def test_device_is_always_passed_and_cache_folder_only_when_set(self) -> None:
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(device="cpu")
        self.assertEqual(log["builds"][0]["kwargs"], {"device": "cpu",
                                                     "local_files_only": False})
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(cache_folder="/tmp/emb")
        self.assertEqual(log["builds"][0]["kwargs"], {"device": None, "cache_folder": "/tmp/emb",
                                                      "local_files_only": False})

    def test_local_files_only_is_dropped_when_the_signature_cannot_take_it(self) -> None:
        with _with_fake_hf(accepts_local_files_only=False) as log:
            HuggingFaceEmbeddingProvider(local_files_only=True)
        self.assertEqual(log["builds"][0]["kwargs"], {"device": None})

    def test_an_unreadable_signature_is_treated_as_accepting_local_files_only(self) -> None:
        module, log = _fake_sentence_transformers(accepts_local_files_only=False)
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            with mock.patch("inspect.signature", side_effect=ValueError("nope")):
                HuggingFaceEmbeddingProvider(local_files_only=True)
        self.assertEqual(log["builds"][0]["kwargs"],
                         {"device": None, "local_files_only": True})

    def test_the_provider_is_a_base_embedding_provider(self) -> None:
        with _with_fake_hf():
            provider = HuggingFaceEmbeddingProvider()
        self.assertIsInstance(provider, BaseEmbeddingProvider)
        self.assertIsNone(provider.legacy_collection_name)


class HuggingFaceEncodingTests(unittest.TestCase):
    def test_embed_texts_normalizes_and_converts_to_plain_floats(self) -> None:
        with _with_fake_hf(dimension=4) as log:
            provider = HuggingFaceEmbeddingProvider()
            vectors = provider.embed_texts(["你好", "abc", 7])
        self.assertEqual(vectors, [[2.0, 0.5, 0.25], [3.0, 0.5, 0.25], [1.0, 0.5, 0.25]])
        self.assertEqual(log["encodes"][0]["texts"], ["你好", "abc", "7"])
        self.assertEqual(log["encodes"][0]["kwargs"], {
            "normalize_embeddings": True, "convert_to_numpy": True,
            "show_progress_bar": False,
        })

    def test_embed_text_is_the_single_item_batch(self) -> None:
        with _with_fake_hf(dimension=4) as log:
            provider = HuggingFaceEmbeddingProvider()
            single = provider.embed_text("你好")
            batch = provider.embed_texts(["你好"])[0]
        self.assertEqual(single, batch)
        self.assertEqual(len(log["encodes"]), 2)

    def test_an_empty_batch_returns_empty_without_calling_the_encoder(self) -> None:
        with _with_fake_hf(dimension=4) as log:
            provider = HuggingFaceEmbeddingProvider()
            self.assertEqual(provider.embed_texts([]), [])
        self.assertEqual(log["encodes"], [])

    def test_normalize_embeddings_is_forwarded(self) -> None:
        with _with_fake_hf(dimension=4) as log:
            provider = HuggingFaceEmbeddingProvider(normalize_embeddings=False)
            provider.embed_texts(["x"])
        self.assertIs(log["encodes"][0]["kwargs"]["normalize_embeddings"], False)


class HuggingFaceEnvironmentTests(unittest.TestCase):
    """载入期的两个环境开关必须写完就还，包括「本来不存在」的那一种。"""

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key)
                       for key in ("HF_HUB_OFFLINE", "HF_ENDPOINT")}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_offline_mode_sets_the_flag_only_while_loading(self) -> None:
        os.environ.pop("HF_HUB_OFFLINE", None)
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(local_files_only=True)
        self.assertEqual(log["builds"][0]["env"], ("1", None))
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_a_pre_existing_offline_flag_is_restored(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "keep-me"
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(local_files_only=True)
        self.assertEqual(log["builds"][0]["env"], ("1", None))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "keep-me")

    def test_the_endpoint_loses_its_trailing_slash_before_being_written(self) -> None:
        os.environ.pop("HF_ENDPOINT", None)
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(hf_endpoint="https://mirror.example/")
        self.assertEqual(log["builds"][0]["env"], (None, "https://mirror.example"))
        self.assertNotIn("HF_ENDPOINT", os.environ)

    def test_a_pre_existing_endpoint_is_restored(self) -> None:
        os.environ["HF_ENDPOINT"] = "https://old.example"
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(hf_endpoint="https://mirror.example")
        self.assertEqual(log["builds"][0]["env"], (None, "https://mirror.example"))
        self.assertEqual(os.environ["HF_ENDPOINT"], "https://old.example")

    def test_a_blank_endpoint_touches_nothing(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "keep-me"
        os.environ.pop("HF_ENDPOINT", None)
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider(hf_endpoint="   ")
        self.assertEqual(log["builds"][0]["env"], ("keep-me", None))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "keep-me")
        self.assertNotIn("HF_ENDPOINT", os.environ)

    def test_neither_switch_means_the_environment_is_left_alone(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "untouched"
        os.environ["HF_ENDPOINT"] = "https://untouched.example"
        with _with_fake_hf() as log:
            HuggingFaceEmbeddingProvider()
        self.assertEqual(log["builds"][0]["env"],
                         ("untouched", "https://untouched.example"))


class HuggingFaceDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key)
                       for key in ("HF_HUB_OFFLINE", "HF_ENDPOINT")}
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_ENDPOINT", None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_a_missing_dependency_becomes_an_actionable_runtime_error(self) -> None:
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError) as caught:
                HuggingFaceEmbeddingProvider(local_files_only=True,
                                             hf_endpoint="https://mirror.example")
        self.assertEqual(
            str(caught.exception),
            "sentence-transformers is not installed; install requirements-ml.txt "
            "to enable HuggingFace embeddings.",
        )
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_the_environment_is_restored_even_when_loading_fails(self) -> None:
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError):
                HuggingFaceEmbeddingProvider(local_files_only=True,
                                            hf_endpoint="https://mirror.example")
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)
        self.assertNotIn("HF_ENDPOINT", os.environ)


# --------------------------------------------------------------------------- #
# 边界扫描与契约字面量
# --------------------------------------------------------------------------- #


class C2_1BoundaryTests(unittest.TestCase):
    """B1 边界在新代码上的延续，同时是「表达层独立」的直接证据。"""

    C2_1_FILES = (
        SHINKU_PKG / "llm" / "__init__.py",
        SHINKU_PKG / "llm" / "circuit_breaker.py",
        SHINKU_PKG / "text" / "__init__.py",
        SHINKU_PKG / "text" / "tokenizer.py",
        SHINKU_PKG / "providers" / "__init__.py",
        SHINKU_PKG / "providers" / "embedding.py",
        SHINKU_PKG / "providers" / "huggingface.py",
    )

    #: 旧项目的包名与代号（子串匹配，这些串足够独特）。
    FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane")

    #: 旧模块的内部命名，取「对照物 ∪ Shinku 旧实现」的并集（34 个）。
    #: 用**精确标识符 token** 比对，理由同 C1-3／C1-4。
    LEGACY_INTERNAL_NAMES = (
        # llm_circuit_breaker.py（对照物与旧侧同一份）
        "_BREAKER",
        "_consecutive_failures", "_open_until", "_open_count",
        "_last_failure_at", "_last_success_at", "_lock",
        # text_utils.py —— 本切片相关
        "_HASHED_EMBEDDING_PROVIDERS",
        # text_utils.py —— 同文件其余切片（属 C5），本批更不该出现
        "_TOPIC_WORD_RE", "_TOPIC_TAIL_CHARS", "_TOPIC_STOPWORDS",
        "_normalize_date_label", "_normalize_time_of_day", "_normalize_hour",
        "_normalize_minute", "_normalize_positive_int", "_extract_clock_time",
        "_contains_ambiguous_reminder_time", "_infer_relative_date_label",
        "_default_hour_for_time_of_day", "_apply_cn_time_period_bias",
        "_resolve_relative_offset_seconds", "_parse_human_number",
        # embedding_provider.py
        "_COLLECTION_COMPONENT_RE", "_cache_get", "_cache_put", "_get", "_put",
        "_dimension", "_legacy_collection_name", "_cache",
        # huggingface_provider.py
        "_load_model", "_temporary_hf_load_env", "_model",
    )

    #: 旧模块 docstring 与注释里的整句。它们不该以任何形式出现在新文件里。
    LEGACY_PROSE = (
        "LLM 上游熔断器：连续失败时短暂静默，避免刷屏。",
        "背景（2026-09-08 事故）：DeepSeek API 间歇性连接失败（APIConnectionError）期间，",
        "被同一群连续发出 8 次，群友直接说「掉线了」。",
        "熔断只影响\"发起新的 LLM 回合\"，不影响命令处理、旁观入库等本地能力。",
        "每轮 LLM 调用都失败 → persona 兜底话术（当前为角色内的短句）",
        "一次失败调用：累计到阈值后打开/延长熔断（指数退避）。",
        "进程级单例：LLM 调用层与 QQ 事件层共用同一个熔断状态。",
        "Project-neutral tokenizer used by shared embedding primitives.",
        "Project-neutral embedding provider primitives.",
        "Local embedding primitives for Shinku memory retrieval.",
        "Optional HuggingFace sentence-transformer embedding provider.",
        "Optional HuggingFace sentence-transformer provider for Shinku memory.",
        "没有分词库，所以用\"跨消息重复\"来筛真话题：只在多条消息里都出现的词才算话题，",
        "旁观原文并进上下文会让模型学走别人的比喻和措辞（2026-09-10 \"跟翻书似的\"事故）；",
        "这里只往外给话题词，模型仍知道群里在聊什么，但拿不到别人的句子。",
    )

    def _sources(self):
        for path in self.C2_1_FILES:
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
        # 名单是「对照物 ∪ Shinku 旧实现」的并集，这里钉住条数，
        # 免得以后有人补名字时顺手删掉一个（C1-4 就漏过 6 个）。
        self.assertEqual(len(self.LEGACY_INTERNAL_NAMES), 34)
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

    def test_the_flat_legacy_module_names_were_not_recreated(self) -> None:
        for name in ("llm_circuit_breaker.py", "text_utils.py", "text_tokenizer.py",
                     "embedding_provider.py", "huggingface_provider.py"):
            with self.subTest(name=name):
                self.assertFalse((SHINKU_PKG / name).exists())
        for dotted in ("shinku.llm_circuit_breaker", "shinku.text_utils",
                       "shinku.embedding_provider", "shinku.huggingface_provider"):
            with self.subTest(dotted=dotted):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(dotted)

    def test_importing_the_batch_does_not_pull_the_old_project_in(self) -> None:
        for forbidden in ("companion_v01", "code_shared", "akane"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, sys.modules)

    def test_the_new_packages_aggregate_their_own_slice_only(self) -> None:
        llm = importlib.import_module("shinku.llm")
        text = importlib.import_module("shinku.text")
        self.assertEqual(set(llm.__all__), {"LlmCircuitBreaker", "get_llm_circuit_breaker"})
        self.assertEqual(set(text.__all__),
                         {"STOPWORDS", "TOKEN_RE", "normalize_text", "tokenize"})
        # providers 是从 C1-4 的扁平模块迁过来的，__all__ 保持兼容面不扩大。
        providers = importlib.import_module("shinku.providers")
        self.assertEqual(set(providers.__all__),
                         set(importlib.import_module("shinku.providers.config").__all__))

    def test_the_llm_line_does_not_depend_on_the_provider_line(self) -> None:
        self.assertNotIn("shinku.providers",
                         _imported_modules(SHINKU_PKG / "llm" / "circuit_breaker.py"))
        self.assertNotIn("shinku.llm",
                         _imported_modules(SHINKU_PKG / "providers" / "embedding.py"))

    def test_the_embedding_module_reaches_for_the_text_slice(self) -> None:
        modules = _imported_modules(SHINKU_PKG / "providers" / "embedding.py")
        self.assertIn("..text.tokenizer", modules)

    def test_no_network_module_is_imported_by_this_batch(self) -> None:
        for name in ("llm/circuit_breaker.py", "text/tokenizer.py",
                     "providers/embedding.py", "providers/huggingface.py"):
            modules = _imported_modules(SHINKU_PKG / name)
            for forbidden in ("socket", "subprocess", "requests", "urllib", "http"):
                with self.subTest(module=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, modules)


class C2_1ContractLiteralTests(unittest.TestCase):
    """契约里那些「改坏了也不会让别处变红」的字面量，逐条钉住。"""

    def test_the_token_pattern_literal(self) -> None:
        self.assertEqual(TOKEN_RE.pattern, r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")

    def test_the_stopword_table_literal(self) -> None:
        self.assertEqual(sorted(STOPWORDS), sorted([
            "的", "了", "呢", "啊", "呀", "吗", "吧", "哦", "喵", "我", "你", "他", "她", "它",
            "我们", "你们", "他们", "然后", "就是", "这个", "那个", "现在", "一下",
        ]))

    def test_the_snapshot_keys_are_camel_case_literals(self) -> None:
        with _frozen_clock():
            snapshot = LlmCircuitBreaker().snapshot()
        self.assertEqual(set(snapshot), {
            "open", "consecutiveFailures", "openCount",
            "remainingSeconds", "lastFailureAt", "lastSuccessAt",
        })

    def test_the_legacy_collection_name_is_the_documented_literal(self) -> None:
        self.assertEqual(HashedEmbeddingProvider(dimension=128).legacy_collection_name,
                         "shinku_memory_v01")

    def test_the_huggingface_environment_keys_are_the_documented_literals(self) -> None:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_ENDPOINT", None)
        try:
            with _with_fake_hf() as log:
                HuggingFaceEmbeddingProvider(local_files_only=True,
                                             hf_endpoint="https://mirror.example")
            self.assertEqual(log["builds"][0]["env"], ("1", "https://mirror.example"))
            source = (SHINKU_PKG / "providers" / "huggingface.py").read_text(encoding="utf-8")
            self.assertIn('"HF_HUB_OFFLINE"', source)
            self.assertIn('"HF_ENDPOINT"', source)
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("HF_ENDPOINT", None)

    def test_the_probe_text_is_the_documented_literal(self) -> None:
        with _with_fake_hf(dimension=0) as log:
            HuggingFaceEmbeddingProvider()
        self.assertEqual(log["encodes"][0]["texts"], ["探针"])

    def test_the_install_hint_is_the_documented_literal(self) -> None:
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError) as caught:
                HuggingFaceEmbeddingProvider()
        self.assertEqual(
            str(caught.exception),
            "sentence-transformers is not installed; install requirements-ml.txt "
            "to enable HuggingFace embeddings.")


if __name__ == "__main__":
    unittest.main()

"""C1-1 数据类型契约层的测试。

这一批按来源记录 §5.9 的**路径 A** 处理：契约事实（类名、字段名、类型、默认值、
协议常量）允许按事实迁移，**表达层必须独立撰写**。所以这里断言的对象是
**外部可观察行为**——字段面、不可变语义、默认值的独立性、协议常量的取值——
而不是任何实现结构。

其中"失败测试"是刻意的：契约的价值一半在于它**拒绝**什么。
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from shinku import contracts
from shinku.contracts import (
    AGENT_ALIASES,
    AGENT_ALLOWED_TOOLS,
    CapabilityManifestError,
    CapabilityProtocolError,
    InvocationContext,
    RetrievalPipelineResult,
    WorkerDelegation,
    WorkerRunSummary,
    json_response,
)

CONTRACTS_DIR = Path(contracts.__file__).resolve().parent


class FrozenRecordMixin:
    """契约事实：这些记录实例创建后不可赋值修改。"""

    def assert_frozen(self, instance: object, field_name: str) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, "changed")


class DelegationRecordTests(FrozenRecordMixin, unittest.TestCase):
    def test_required_fields_have_no_default(self) -> None:
        with self.assertRaises(TypeError):
            WorkerDelegation()  # type: ignore[call-arg]

    def test_optional_fields_carry_documented_defaults(self) -> None:
        item = WorkerDelegation("t-1", "media_agent", "convert it")
        self.assertEqual(item.handle_id, "")
        self.assertFalse(item.started)

    def test_instance_cannot_be_mutated(self) -> None:
        item = WorkerDelegation("t-1", "media_agent", "convert it")
        self.assert_frozen(item, "started")

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            WorkerDelegation("t-1", "media_agent", "x", missing=True)  # type: ignore[call-arg]


class RunSummaryTests(unittest.TestCase):
    def test_summary_is_mutable(self) -> None:
        summary = WorkerRunSummary("t-1", "media_agent", "done")
        summary.rounds = 3
        self.assertEqual(summary.rounds, 3)

    def test_defaults_are_stable(self) -> None:
        summary = WorkerRunSummary("t-1", "media_agent", "done")
        self.assertEqual(summary.rounds, 0)
        self.assertEqual(summary.messages, [])
        self.assertEqual(summary.tool_results, [])

    def test_collections_are_not_shared_between_instances(self) -> None:
        first = WorkerRunSummary("t-1", "media_agent", "done")
        second = WorkerRunSummary("t-2", "media_agent", "done")
        first.messages.append("only the first")
        first.tool_results.append("only the first")
        self.assertEqual(second.messages, [])
        self.assertEqual(second.tool_results, [])


class AgentVocabularyTests(unittest.TestCase):
    def test_canonical_agents_are_exactly_four(self) -> None:
        self.assertEqual(
            set(AGENT_ALLOWED_TOOLS),
            {"document_agent", "media_agent", "speech_agent", "resource_agent"},
        )

    def test_every_allow_list_is_a_set(self) -> None:
        for agent, allowed in AGENT_ALLOWED_TOOLS.items():
            with self.subTest(agent=agent):
                self.assertIsInstance(allowed, set)
                self.assertTrue(allowed)

    def test_every_alias_targets_a_known_agent(self) -> None:
        for alias, target in AGENT_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIn(target, AGENT_ALLOWED_TOOLS)

    def test_alias_never_shadows_a_canonical_name(self) -> None:
        self.assertEqual(set(AGENT_ALIASES) & set(AGENT_ALLOWED_TOOLS), set())

    def test_tool_names_are_shared_where_the_work_overlaps(self) -> None:
        # 附件工作区同步对四个专员都可用，是这张表的核心约定之一。
        for agent, allowed in AGENT_ALLOWED_TOOLS.items():
            with self.subTest(agent=agent):
                self.assertIn("sync_attachment_workspace", allowed)


class RetrievalContractTests(FrozenRecordMixin, unittest.TestCase):
    FIELDS = (
        "used_retrieval",
        "confirmed_snippets",
        "router_output",
        "router_timing",
        "retrieval_result",
        "verifier_output",
        "verifier_timing",
    )

    def build(self) -> RetrievalPipelineResult:
        return RetrievalPipelineResult(
            used_retrieval=True,
            confirmed_snippets=["a"],
            router_output={},
            router_timing={},
            retrieval_result={},
            verifier_output={},
            verifier_timing={},
        )

    def test_field_surface_is_fixed(self) -> None:
        self.assertEqual(
            tuple(f.name for f in dataclasses.fields(RetrievalPipelineResult)), self.FIELDS
        )

    def test_no_field_is_optional(self) -> None:
        for spec in dataclasses.fields(RetrievalPipelineResult):
            with self.subTest(field=spec.name):
                self.assertIs(spec.default, dataclasses.MISSING)
                self.assertIs(spec.default_factory, dataclasses.MISSING)

    def test_instance_cannot_be_mutated(self) -> None:
        self.assert_frozen(self.build(), "used_retrieval")


class JsonResponseTests(unittest.TestCase):
    def test_default_cache_directive_is_no_store(self) -> None:
        response = json_response({"ok": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_status_code_is_coerced_to_int(self) -> None:
        response = json_response({}, "201")  # type: ignore[arg-type]
        self.assertEqual(response.status_code, 201)

    def test_cache_directive_can_be_overridden(self) -> None:
        response = json_response({}, cache_control="private")
        self.assertEqual(response.headers["cache-control"], "private")

    def test_caller_headers_are_merged_and_coerced_to_str(self) -> None:
        response = json_response({}, headers={"X-Trace": 77})  # type: ignore[dict-item]
        self.assertEqual(response.headers["x-trace"], "77")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_explicit_cache_header_wins_over_the_default(self) -> None:
        response = json_response({}, headers={"Cache-Control": "public, max-age=60"})
        self.assertEqual(response.headers["cache-control"], "public, max-age=60")

    def test_empty_header_mapping_leaves_the_directive_alone(self) -> None:
        response = json_response({}, headers={})
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_cache_options_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            json_response({}, "200", "private")  # type: ignore[misc]


class CapabilityContractTests(FrozenRecordMixin, unittest.TestCase):
    def test_export_surface_is_exactly_sixteen_names(self) -> None:
        self.assertEqual(len(contracts.capability.__all__), 16)
        self.assertEqual(len(set(contracts.capability.__all__)), 16)

    def test_every_exported_name_resolves(self) -> None:
        for name in contracts.capability.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(contracts.capability, name))

    def test_invocation_context_is_constructible_with_no_arguments(self) -> None:
        context = InvocationContext()
        self.assertEqual(context.session_id, "")
        self.assertEqual(context.master_qq, "")

    def test_invocation_context_is_frozen(self) -> None:
        self.assert_frozen(InvocationContext(), "session_id")

    def test_error_bases_match_what_callers_catch(self) -> None:
        self.assertTrue(issubclass(CapabilityManifestError, ValueError))
        self.assertTrue(issubclass(CapabilityProtocolError, RuntimeError))

    def test_manifest_error_is_not_a_runtime_error(self) -> None:
        self.assertFalse(issubclass(CapabilityManifestError, RuntimeError))


class ContractBoundaryTests(unittest.TestCase):
    """契约层不得把旧项目拉进来——这是 B1 边界在新代码上的延续。"""

    FORBIDDEN = ("companion_v01", "code_shared", "akane")

    def test_no_contract_module_mentions_a_foreign_project(self) -> None:
        sources = sorted(CONTRACTS_DIR.glob("*.py"))
        self.assertTrue(sources, "contracts 包下应有源文件")
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN:
                with self.subTest(module=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_package_re_exports_every_public_contract(self) -> None:
        expected = {
            "AGENT_ALIASES",
            "AGENT_ALLOWED_TOOLS",
            "RetrievalPipelineResult",
            "WorkerDelegation",
            "WorkerRunSummary",
            "json_response",
        }
        self.assertTrue(expected.issubset(set(contracts.__all__)))


if __name__ == "__main__":
    unittest.main()

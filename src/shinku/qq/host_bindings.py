"""Explicit in-process bindings supplied by a trusted launcher, never a turn.

The launcher owns provider selection and client lifetime. No environment loader,
dynamic import, credentials, tool grants, or implicit production opt-in live here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shinku.agent_runtime.execution import ExecutionProfile
from shinku.agent_runtime.materials import MaterialPolicy
from shinku.result_expression import ExpressionProfile
from .interaction_tools import ChatReadCapability
from .query_continuation import ForegroundQueryPolicy


@dataclass(frozen=True)
class QQAgentHostBindings:
    """Host-approved dependencies for the existing task-mode assembly.

    Permission mode, allowed execution tools, sending, task budgets, and storage
    remain owned by the existing assembly/configuration. Empty material roots
    and read capabilities grant no extra local roots or chat tools.
    """

    chat_runtime: Any = field(repr=False)
    execution_profile: ExecutionProfile | None = field(default=None, repr=False)
    expression_profile: ExpressionProfile | None = field(default=None, repr=False)
    material_policy: MaterialPolicy = field(default_factory=MaterialPolicy)
    material_roots: tuple[str | Path, ...] = ()
    chat_read_capabilities: Mapping[str, ChatReadCapability] = field(default_factory=dict)
    foreground_query_policy: ForegroundQueryPolicy = field(default_factory=ForegroundQueryPolicy)
    # 新可选连接放在原字段之后，保留旧调用方的位置参数含义。
    delegation: Any = field(default=None, repr=False)
    # 可信启动器的后端资源；返回 context manager，生命周期由 CLI ExitStack 持有。
    backend_setup: Any = field(default=None, repr=False)
    # 插件由可信启动器拥有，装配仅借用；不从回合输入或环境自动加载。
    runtime_extension: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.runtime_extension is not None:
            from shinku.extensions import RuntimeExtension
            if not isinstance(self.runtime_extension, RuntimeExtension):
                raise ValueError("host_bindings_runtime_extension_required")
        if self.backend_setup is not None and not callable(self.backend_setup):
            raise ValueError("host_backend_setup_required")
        from .delegation import DelegationConnection
        if self.delegation is not None and not isinstance(self.delegation, DelegationConnection):
            raise ValueError("host_bindings_delegation_required")
        if not callable(getattr(self.chat_runtime, "call_chat_json", None)):
            raise ValueError("host_bindings_runtime_required")
        if self.delegation is None or self.execution_profile is not None:
            if (not isinstance(self.execution_profile, ExecutionProfile)
                    or not callable(getattr(self.execution_profile.runtime, "call_chat_json", None))):
                raise ValueError("host_bindings_runtime_required")
            if self.execution_profile.runtime is self.chat_runtime:
                raise ValueError("host_bindings_distinct_execution_runtime_required")
        if self.expression_profile is not None and not isinstance(self.expression_profile, ExpressionProfile):
            raise ValueError("host_bindings_expression_profile_required")
        if (not isinstance(self.material_policy, MaterialPolicy)
                or not isinstance(self.foreground_query_policy, ForegroundQueryPolicy)):
            raise ValueError("host_bindings_policy_required")
        if (not isinstance(self.material_roots, tuple)
                or any(not isinstance(p, (str, Path)) or not str(p).strip() for p in self.material_roots)):
            raise ValueError("host_bindings_material_roots_invalid")
        if (not isinstance(self.chat_read_capabilities, Mapping)
                or any(not isinstance(value, ChatReadCapability) or key != value.name
                       for key, value in self.chat_read_capabilities.items())):
            raise ValueError("host_bindings_read_capabilities_invalid")
        object.__setattr__(self, "chat_read_capabilities", MappingProxyType(dict(self.chat_read_capabilities)))

    def assembly_options(self) -> dict[str, Any]:
        options = dict(runtime=self.chat_runtime, execution_profile=self.execution_profile, delegation=self.delegation,
                    expression_profile=self.expression_profile, material_policy=self.material_policy,
                    material_roots=self.material_roots, tool_decision_mode="task",
                    chat_read_capabilities=dict(self.chat_read_capabilities),
                    foreground_query_policy=self.foreground_query_policy)
        if self.runtime_extension is not None:
            options["runtime_extension"] = self.runtime_extension
        return options

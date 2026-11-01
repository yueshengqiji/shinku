"""插件接线的故障隔离、资源归属和误伤回归；不访问模型、生产数据或外部平台。"""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import ModuleType, SimpleNamespace

import pytest

from shinku.extensions import RuntimeExtension
from shinku.plugins import PluginContext, PluginContributions, PluginDefinition, PluginManager, PluginManifest, PluginUnavailable
from shinku.plugins.config import load_plugin_definitions
from shinku.tools.registry import ToolRegistry


class Echo:
    tool_type = "fixture_echo"
    agent_permission_class = "read"
    agent_progress_class = "observe"

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        return {"call": call, "context": context}


class Resource:
    def __init__(self, events=None, name="sample", tools=None):
        self.events, self.name = events if events is not None else [], name
        self.tools = tools if tools is not None else {Echo.tool_type: Echo()}
        self.closed = 0

    def activate(self, context):
        self.context = context
        self.events.append("start:" + self.name)
        return PluginContributions(tools=self.tools)

    def close(self):
        self.closed += 1
        self.events.append("close:" + self.name)


def definition(name="sample", factory=Resource, **kwargs):
    keys = {key: kwargs.pop(key) for key in tuple(kwargs) if key in {"version", "api_version", "hosts", "depends_on", "metadata"}}
    return PluginDefinition(PluginManifest(name, **keys), factory, **kwargs)


def states(manager):
    return {row["id"]: row for row in manager.diagnostics()["plugins"]}


def test_disabled_or_other_host_never_imports_implementation(monkeypatch):
    monkeypatch.setattr("shinku.plugins.manager.importlib.import_module", lambda *_: pytest.fail("unexpected import"))
    with PluginManager(PluginContext("shinku"), [
        definition("disabled", "missing:factory", enabled=False),
        definition("other", "missing:factory", hosts=("angelos",)),
    ]) as manager:
        assert states(manager)["disabled"]["status"] == "disabled"
        assert states(manager)["other"]["status"] == "inactive"


def test_optional_failure_preserves_healthy_plugin_and_sanitizes_exception():
    resource = Resource()
    class Broken(Resource):
        def activate(self, context):
            raise RuntimeError("private credential must not appear")
    broken = Broken()
    with PluginManager(PluginContext("shinku"), [definition("broken", lambda: broken), definition(factory=lambda: resource)]) as manager:
        registry = ToolRegistry()
        manager.register_tools(registry)
        assert registry.get(Echo.tool_type).execute(call={"text": "正常请求"}, context=None)["call"]["text"] == "正常请求"
        assert broken.closed == 1
        assert states(manager)["broken"]["reason"] == "plugin_activate_failed:RuntimeError"
        assert "private credential" not in json.dumps(manager.diagnostics())
    assert resource.closed == broken.closed == 1


def test_dependency_order_is_independent_of_declaration_order_and_closes_reverse():
    events = []
    child, base = Resource(events, "child", {}), Resource(events, "base", {})
    with PluginManager(PluginContext("angelos"), [
        definition("child", lambda: child, depends_on=("base",)), definition("base", lambda: base),
    ]) as manager:
        assert events == ["start:base", "start:child"]
    assert events == ["start:base", "start:child", "close:child", "close:base"]


def test_missing_dependency_cycle_or_api_problem_does_not_stop_unrelated_plugin():
    with PluginManager(PluginContext("shinku"), [
        definition("missing", depends_on=("absent",)),
        definition("cycle-a", depends_on=("cycle-b",)), definition("cycle-b", depends_on=("cycle-a",)),
        definition("future", api_version=2), definition(),
    ]) as manager:
        status = states(manager)
        assert status["sample"]["status"] == "active"
        assert status["future"]["reason"] == "plugin_api_incompatible"
        assert status["missing"]["reason"] == "plugin_dependency_unavailable"
        assert status["cycle-a"]["reason"] == status["cycle-b"]["reason"] == "plugin_dependency_cycle"


def test_registration_conflict_withdraws_partial_plugin_and_dependents_only():
    original = Echo()
    bad = Resource(tools={"new_tool": Echo(), "existing": Echo()})
    child = Resource(tools={"child_tool": Echo()})
    healthy = Resource(tools={"healthy_tool": Echo()})
    registry = ToolRegistry({"existing": original})
    with PluginManager(PluginContext("shinku"), [
        definition("bad", lambda: bad), definition("child", lambda: child, depends_on=("bad",)),
        definition("healthy", lambda: healthy),
    ]) as manager:
        manager.register_tools(registry)
        assert registry.names() == ("existing", "healthy_tool")
        assert registry.get("existing") is original
        assert states(manager)["bad"]["status"] == "failed"
        assert states(manager)["child"]["status"] == "disabled"
    assert registry.names() == ("existing",)


@pytest.mark.parametrize("text", [
    "把这段聊天改写一下，里面提到写文件", "没有找到异常", "不要取消正在运行的任务", '请解释引用中的“取消”',
    "需要查三十份资料并比较很多步骤", "名称带空格和中文，不要求额外审批", "", "x" * 20000,
])
def test_wrapper_preserves_request_arguments_and_context_without_keyword_or_size_checks(text):
    call, context = {"text": text, "nested": {"mode": "cancel"}}, object()
    with PluginManager(PluginContext("shinku"), [definition(version="自用实验版 / nightly", metadata={"extra": "unknown"})]) as manager:
        registry = ToolRegistry()
        manager.register_tools(registry)
        handler = registry.get(Echo.tool_type)
        assert handler.normalize_call(call) is call
        result = handler.execute(call=call, context=context)
        assert result["call"] is call and result["context"] is context
        assert handler.agent_permission_class == "read"


def test_stale_handler_cannot_start_after_disable_and_replacement_is_preserved():
    resource, registry = Resource(), ToolRegistry()
    manager = PluginManager(PluginContext("shinku"), [definition(factory=lambda: resource)])
    manager.register_tools(registry)
    old = registry.get(Echo.tool_type)
    replacement = Echo()
    registry.register(replacement, replace=True)
    manager.disable("sample")
    assert registry.get(Echo.tool_type) is replacement
    with pytest.raises(PluginUnavailable, match="plugin_disabled"):
        old.execute(call={}, context=None)
    manager.close()
    assert resource.closed == 1


def test_disable_waits_for_inflight_dependent_before_closing_either_resource():
    entered, release = threading.Event(), threading.Event()
    class Blocking(Echo):
        def execute(self, *, call, context):
            entered.set()
            assert release.wait(5)
            return "finished"
    events = []
    child = Resource(events, "child", {"child_tool": Blocking()})
    base = Resource(events, "base", {})
    manager = PluginManager(PluginContext("angelos"), [
        definition("child", lambda: child, depends_on=("base",)), definition("base", lambda: base),
    ])
    registry = ToolRegistry()
    manager.register_tools(registry)
    handler = registry.get("child_tool")
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(handler.execute, call={}, context=None)
        try:
            assert entered.wait(5)
            manager.disable("base")
            assert registry.names() == () and child.closed == base.closed == 0
            assert states(manager)["child"]["status"] == states(manager)["base"]["status"] == "draining"
            with pytest.raises(PluginUnavailable):
                handler.execute(call={}, context=None)
        finally:
            release.set()
        assert future.result(5) == "finished"
    assert events[-2:] == ["close:child", "close:base"]
    manager.close()
    assert child.closed == base.closed == 1


def test_close_failure_is_diagnostic_and_does_not_skip_other_resources():
    class Broken(Resource):
        def close(self):
            super().close()
            raise ValueError("private path must not appear")
    healthy, broken = Resource(tools={}), Broken(tools={})
    manager = PluginManager(PluginContext("shinku"), [definition("healthy", lambda: healthy), definition("broken", lambda: broken)])
    manager.start(); manager.start(); manager.close(); manager.close()
    assert healthy.closed == broken.closed == 1
    assert states(manager)["broken"]["cleanup_error"] == "plugin_close_failed:ValueError"
    assert "private path" not in json.dumps(manager.diagnostics())


def test_managers_do_not_share_plugin_state_and_keep_instance_config_separate():
    with PluginManager(PluginContext("shinku", "one"), [definition(config={"label": "a"})]) as one:
        with PluginManager(PluginContext("shinku", "two"), [definition(config={"label": "b"})]) as two:
            a, b = one._by_id["sample"].instance, two._by_id["sample"].instance
            assert a is not b
            assert a.context.instance_id == "one" and b.context.instance_id == "two"
            assert a.context.plugin_config["label"] == "a" and b.context.plugin_config["label"] == "b"


def test_legacy_prompt_commands_binding_and_tools_are_preserved_through_adapter():
    legacy = ModuleType("fixture_legacy")
    calls = []
    command = object()
    legacy.register_tools = lambda registry, config: registry.register(Echo(), name="legacy_tool")
    legacy.bind_runtime = lambda **kwargs: calls.append(kwargs)
    legacy.augment_system_prompt = lambda prompt, config: prompt + "\n原私用人设"
    legacy.build_command_interpreter = lambda **kwargs: command
    with PluginManager(PluginContext("shinku"), [definition()]) as manager:
        extension = manager.runtime_extension(legacy=RuntimeExtension("fixture_legacy", legacy))
        registry = ToolRegistry()
        extension.register_tools(registry=registry, config=None)
        extension.bind_runtime(marker="unchanged")
        assert extension.augment_system_prompt(prompt="原文", config=None) == "原文\n原私用人设"
        assert extension.build_command_interpreter() is command
        assert registry.names() == (Echo.tool_type, "legacy_tool")
        assert calls == [{"marker": "unchanged"}]
        assert extension.diagnostics()["plugins"][0]["status"] == "active"
    assert registry.names() == ("legacy_tool",)


def test_json_config_preserves_unknown_fields_and_isolates_a_bad_row(tmp_path):
    path = tmp_path / "插件配置.json"
    path.write_text(json.dumps({"plugins": [
        {"id": "bad-shape", "hosts": 123, "factory": "never_import:create"},
        {"id": "example", "factory": "shinku.plugins.examples.local_clock:create", "version": "试用版", "future_field": True},
    ]}), encoding="utf-8")
    definitions = load_plugin_definitions(path)
    assert definitions[1].manifest.metadata["future_field"]
    with PluginManager(PluginContext("shinku"), definitions) as manager:
        registry = ToolRegistry()
        manager.register_tools(registry)
        assert states(manager)["bad-shape"]["status"] == "failed"
        assert states(manager)["example"]["status"] == "active"
        result = registry.get("local_clock").execute(call={"timezone": "Asia/Shanghai"}, context=None)
        assert result["ok"] and result["state_updates"]["iso"].endswith("+08:00")


def test_entrypoint_discovery_loads_only_the_explicit_name(monkeypatch):
    requested = []
    def entries(**kwargs):
        requested.append(kwargs)
        return [SimpleNamespace(load=lambda: Resource)]
    monkeypatch.setattr("shinku.plugins.manager.importlib.metadata.entry_points", entries)
    with PluginManager(PluginContext("angelos"), [definition(factory="entrypoint:sample")]) as manager:
        assert states(manager)["sample"]["status"] == "active"
    assert requested == [{"group": "shinku.plugins", "name": "sample"}]


@pytest.mark.parametrize("body", [None, "not JSON", '{"plugins":{}}'])
def test_optional_config_problem_is_a_diagnostic_not_a_global_start_failure(tmp_path, body):
    path = tmp_path / "optional.json"
    if body is not None:
        path.write_text(body)
    with PluginManager(PluginContext("shinku"), load_plugin_definitions(path)) as manager:
        registry = ToolRegistry()
        manager.register_tools(registry)
        assert registry.names() == ()
        problem = states(manager)["plugin-configuration"]
        assert problem["status"] == "failed" and problem["reason"].startswith("plugin_config_")


def test_read_conflict_removes_previously_collected_dependent_reads():
    from shinku.qq.interaction_tools import ChatReadCapability
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    host = ChatReadCapability("host_read", "host", schema)
    dependent = ChatReadCapability("dependent_read", "dependent", schema)
    class Reads(Resource):
        def __init__(self, read):
            super().__init__(tools={})
            self.read = read
        def activate(self, context):
            return PluginContributions(chat_read_capabilities={self.read.name: self.read})
    with PluginManager(PluginContext("shinku"), [
        definition("child", lambda: Reads(dependent), depends_on=("base",)),
        definition("base", lambda: Reads(host)),
    ]) as manager:
        assert manager.read_capabilities({host.name: host}) == {host.name: host}
        assert states(manager)["child"]["status"] == "disabled"

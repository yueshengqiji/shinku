"""不同宿主的供应商展示目录不能改变同一份注册表的模型选择。"""

import json
import os
from types import SimpleNamespace

from shinku.api.model_runtime import RuntimeModelServiceConfig, RuntimeModelServiceController
from shinku.providers.catalog import ModelServiceConfigStore, public_model_services_snapshot


def _entry(provider, model):
    """所有地址和凭据均为合成数据，不调用供应商。"""
    return {
        "providerId": provider,
        "protocol": "openai",
        "baseUrl": f"https://{provider}.example/v1",
        "apiKey": f"fixture-{provider}",
        "chatModel": model,
        "visionModel": model,
    }


def _registry(path):
    data = {
        "schemaVersion": 2,
        "activeProviderId": "gateway-a",
        "providers": {
            "openai": _entry("openai", "old-model"),
            "gateway-a": _entry("gateway-a", "model-a"),
            "gateway-b": _entry("gateway-b", "model-b"),
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_hosts_with_different_catalogs_read_the_same_active_connection(tmp_path):
    path = tmp_path / "models.json"
    data = _registry(path)
    main_env = {"SHINKU_PROVIDER_PRESETS_JSON": json.dumps([
        {"id": p, "protocol": "openai", "baseUrl": v["baseUrl"]}
        for p, v in data["providers"].items() if p.startswith("gateway-")
    ])}
    main = ModelServiceConfigStore(path, environ=main_env)
    background = ModelServiceConfigStore(path, environ={})
    before = path.read_bytes()

    assert background.load() == main.load()
    assert background.load().provider_id == "gateway-a"
    assert set(background.load_registry()["providers"]) == set(data["providers"])
    assert path.read_bytes() == before


def test_existing_controller_follows_later_provider_and_model_changes(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    _registry(path)
    # 模拟后台宿主早于新供应商启动；它在每次生成前重新读取主回复注册表。
    for key in list(os.environ):
        if key.startswith("SHINKU_"):
            monkeypatch.delenv(key)
    for prefix in ("TEXT", "AUX", "CHAT", "VISION"):
        for field in ("API_KEY", "BASE_URL", "MODEL_NAME", "API_PROTOCOL"):
            monkeypatch.setenv(f"SHINKU_{prefix}_{field}", "")
    for key in ("ENABLE_NATIVE_TOOL_DECISION", "NATIVE_TOOL_PROVIDER_ALLOWLIST"):
        monkeypatch.setenv(f"SHINKU_{key}", "")
    store = ModelServiceConfigStore(path, environ={})
    config = RuntimeModelServiceConfig.from_agent_config(SimpleNamespace(), environ={})
    controller = RuntimeModelServiceController(store=store, config_module=config)

    for provider, model in (("gateway-a", "model-a"), ("gateway-a", "model-a-new"),
                            ("gateway-b", "model-b-new")):
        store.activate(provider, model)
        selected = controller.load_saved()
        assert selected.provider_id == provider
        assert config.CHAT_MODEL_NAME == os.environ["SHINKU_CHAT_MODEL_NAME"] == model
        assert config.CHAT_BASE_URL == f"https://{provider}.example/v1"
        assert config.CHAT_API_KEY == f"fixture-{provider}"


def test_catalogless_write_keeps_other_custom_providers_and_probe_metadata(tmp_path):
    path = tmp_path / "models.json"
    data = _registry(path)
    data["providers"]["gateway-b"]["discoveredModels"] = ["model-b"]
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ModelServiceConfigStore(path, environ={})
    store.activate("gateway-a", "model-a-new")
    store.save_model_probe("gateway-a", ["model-a-new"], timestamp=100)

    saved = json.loads(path.read_text())
    assert set(saved["providers"]) == set(data["providers"])
    assert saved["activeProviderId"] == "gateway-a"
    assert saved["providers"]["gateway-b"]["discoveredModels"] == ["model-b"]
    assert store.get_provider("gateway-b").chat_model == "model-b"


def test_public_snapshot_retains_custom_identity_without_exposing_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINKU_PROVIDER_PRESETS_JSON", raising=False)
    path = tmp_path / "models.json"
    _registry(path)
    store = ModelServiceConfigStore(path, environ={})
    snapshot = public_model_services_snapshot(store, SimpleNamespace())
    assert snapshot["activeProviderId"] == "gateway-a"
    assert snapshot["activeModel"] == "model-a"
    custom = [row for row in snapshot["providers"] if row["id"].startswith("gateway-")]
    assert {row["id"] for row in custom} == {"gateway-a", "gateway-b"}
    assert "fixture-" not in json.dumps(snapshot)

"""Recovery startup must preserve source data and enforce the no-send profile."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux recovery entrypoint")

SPEC = importlib.util.spec_from_file_location(
    "linux_recovery_launcher", Path(__file__).resolve().parents[1] / "scripts/run_linux_recovery.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def profile(tmp_path):
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    source.mkdir()
    recovery.mkdir()
    env = {"SHINKU_LINUX_SOURCE_DATA_ROOT": str(source), "SHINKU_LINUX_RECOVERY_ROOT": str(recovery)}
    for key, name in zip(launcher.WRITE_ROOT_KEYS, ("data", "config", "logs")):
        directory = recovery / name
        directory.mkdir()
        env[key] = str(directory)
    for key in launcher.WRITE_FILE_KEYS:
        env[key] = str(recovery / "data" / key.lower())
    env["SHINKU_PERSONA_FILE"] = str(recovery / "config/persona.md")
    for key in ("SHINKU_MEMORY_DB_PATH", "SHINKU_MODEL_SERVICES_PATH", "SHINKU_PERSONA_FILE"):
        Path(env[key]).write_text("fixture", encoding="utf-8")
    return env


def save_profile(env):
    path = Path(env["SHINKU_CONFIG_ROOT"]) / ".env.linux"
    path.write_text("\n".join(f"{key}={value}" for key, value in env.items()), encoding="utf-8")
    return path


def test_inherited_and_file_flags_cannot_enable_sending_or_extensions(profile):
    profile.update({key: "true" for key in launcher.DISABLED_FLAGS})
    profile["SHINKU_RUNTIME_EXTENSION_MODULE"] = "unmigrated_extension"
    inherited = {"SHINKU_MEMORY_DB_PATH": "/original.sqlite3", "SHINKU_QQ_AGENT_SEND_ENABLED": "true",
                 "LLM_PROMPT_AUDIT_ENABLED": "1", "PATH": os.environ.get("PATH", "")}
    resolved = launcher.recovery_environment(save_profile(profile), inherited)
    assert resolved["SHINKU_MEMORY_DB_PATH"] == profile["SHINKU_MEMORY_DB_PATH"]
    assert all(resolved[key] == "false" for key in launcher.DISABLED_FLAGS)
    assert "SHINKU_RUNTIME_EXTENSION_MODULE" not in resolved
    assert resolved["SHINKU_MEMORY_LIFECYCLE_MODE"] == "off"
    assert resolved["SHINKU_BACKEND_HOST"] == "127.0.0.1"
    assert resolved["SHINKU_ENV_FILE"] == os.devnull
    assert resolved["LLM_PROMPT_AUDIT_ENABLED"] == "0"
    assert resolved["PATH"] == inherited["PATH"]


@pytest.mark.parametrize("key", ["SHINKU_MEMORY_DB_PATH", "SHINKU_QQ_AGENT_STATE_DB_PATH", "SHINKU_LOG_ROOT"])
def test_original_and_outside_write_paths_are_rejected(profile, key):
    profile[key] = str(Path(profile["SHINKU_LINUX_SOURCE_DATA_ROOT"]) / "must-not-write")
    with pytest.raises(ValueError, match="write_path_outside_recovery_tree"):
        launcher.recovery_environment(save_profile(profile), {})


def test_recovery_tree_cannot_be_nested_in_original_data(profile):
    profile["SHINKU_LINUX_RECOVERY_ROOT"] = str(Path(profile["SHINKU_LINUX_SOURCE_DATA_ROOT"]) / "recovery")
    with pytest.raises(ValueError, match="separate_from_source_data"):
        launcher.validate_recovery_paths(profile)


def test_symlink_to_original_memory_is_rejected(profile):
    path = Path(profile["SHINKU_MEMORY_DB_PATH"])
    original = Path(profile["SHINKU_LINUX_SOURCE_DATA_ROOT"]) / "memory.sqlite3"
    original.write_text("original")
    path.unlink()
    path.symlink_to(original)
    with pytest.raises(ValueError, match="write_path_outside_recovery_tree"):
        launcher.validate_recovery_paths(profile)
    assert original.read_text() == "original"


def test_hardlink_to_original_memory_is_rejected(profile):
    path = Path(profile["SHINKU_MEMORY_DB_PATH"])
    original = Path(profile["SHINKU_LINUX_SOURCE_DATA_ROOT"]) / path.name
    original.write_text("original")
    path.unlink()
    path.hardlink_to(original)
    with pytest.raises(ValueError, match="recovery_file_is_original_hardlink"):
        launcher.validate_recovery_paths(profile)
    assert original.read_text() == "original"


def test_required_input_must_exist(profile):
    Path(profile["SHINKU_PERSONA_FILE"]).unlink()
    with pytest.raises(ValueError, match="recovery_input_missing:SHINKU_PERSONA_FILE"):
        launcher.validate_recovery_paths(profile)


def test_blocked_connections_never_reach_a_real_listener_and_guard_restores():
    original = socket.socket.connect
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.02)
        with launcher.block_ip_egress():
            for method in ("connect", "connect_ex"):
                with socket.socket() as client:
                    with pytest.raises(PermissionError, match="linux_recovery_ip_egress_disabled"):
                        getattr(client, method)(listener.getsockname())
            with socket.socket(socket.AF_INET6) as client:
                with pytest.raises(PermissionError):
                    client.connect(("::1", 3000))
            left, right = socket.socketpair()
            with left, right:
                left.sendall(b"unix-ok")
                assert right.recv(7) == b"unix-ok"
        with pytest.raises(TimeoutError):
            listener.accept()
    assert socket.socket.connect is original


def test_cli_reload_cannot_undo_recovery_profile(profile):
    from shinku.config import load_project_env
    profile["SHINKU_QQ_AGENT_SEND_ENABLED"] = "true"
    profile["SHINKU_RUNTIME_EXTENSION_MODULE"] = "unmigrated_extension"
    resolved = launcher.recovery_environment(save_profile(profile), {})
    load_project_env(resolved)
    assert resolved["SHINKU_QQ_AGENT_SEND_ENABLED"] == "false"
    assert "SHINKU_RUNTIME_EXTENSION_MODULE" not in resolved


def test_local_semantic_is_explicit_offline_cpu_and_preserves_all_send_guards(profile):
    cache = Path(profile["SHINKU_DATA_ROOT"]) / "models/copied-model"
    cache.mkdir(parents=True)
    profile["SHINKU_MEMORY_EMBEDDING_CACHE_DIR"] = str(cache)
    profile["SHINKU_MEMORY_EMBEDDING_DEVICE"] = "cuda"
    profile["SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY"] = "false"
    profile.update({key: "true" for key in launcher.DISABLED_FLAGS})
    path = save_profile(profile)
    assert launcher.recovery_environment(path, {})["SHINKU_MEMORY_SEMANTIC_ENABLED"] == "false"
    env = launcher.recovery_environment(path, {"HF_HUB_OFFLINE": "0"}, local_semantic=True)
    assert env["SHINKU_MEMORY_SEMANTIC_ENABLED"] == "true"
    assert all(env[key] == "false" for key in launcher.DISABLED_FLAGS
               if key != "SHINKU_MEMORY_SEMANTIC_ENABLED")
    assert env["SHINKU_MEMORY_EMBEDDING_DEVICE"] == "cpu"
    assert env["SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY"] == "true"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(env["HF_HOME"]).is_relative_to(Path(profile["SHINKU_DATA_ROOT"]))


@pytest.mark.parametrize("kind", ["missing", "source", "directory_symlink", "file_symlink"])
def test_local_semantic_rejects_missing_or_original_cache(profile, kind):
    cache = Path(profile["SHINKU_DATA_ROOT"]) / "models"
    source = Path(profile["SHINKU_LINUX_SOURCE_DATA_ROOT"])
    if kind == "source":
        cache = source
    elif kind == "directory_symlink":
        cache.symlink_to(source, target_is_directory=True)
    elif kind == "file_symlink":
        cache.mkdir()
        (source / "model").write_text("original")
        (cache / "model").symlink_to(source / "model")
    profile["SHINKU_MEMORY_EMBEDDING_CACHE_DIR"] = str(cache)
    with pytest.raises(ValueError, match="local_embedding_cache"):
        launcher.recovery_environment(save_profile(profile), {}, local_semantic=True)

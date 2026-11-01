"""Run the existing backend against a separate recovery tree, with IP egress off.

This entrypoint is deliberately restricted to recovery. It neither activates
runtime extensions nor sends platform messages. Normal deployment uses the
project's ordinary launchers after migration acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinku.config import load_project_env

WRITE_ROOT_KEYS = ("SHINKU_DATA_ROOT", "SHINKU_CONFIG_ROOT", "SHINKU_LOG_ROOT")
WRITE_FILE_KEYS = (
    "SHINKU_MEMORY_DB_PATH", "SHINKU_MODEL_SERVICES_PATH", "SHINKU_QQ_COMMAND_STATE_PATH",
    "SHINKU_QQ_AGENT_TASKS_DB_PATH", "SHINKU_QQ_AGENT_STATE_DB_PATH",
    "SHINKU_QQ_CONVERSATION_STATE_DB_PATH", "SHINKU_QQ_TRACE_STORE_PATH",
    "SHINKU_CHAT_METRICS_PATH", "SHINKU_QQ_AGENT_APPROVAL_STORE_PATH",
)
DISABLED_FLAGS = (
    "SHINKU_QQ_AGENT_SEND_ENABLED", "SHINKU_QQ_AUTONOMOUS_REPLY_ENABLED",
    "SHINKU_QQ_AGENT_ASYNC_ENABLED", "SHINKU_PRIVATE_GREETING_ENABLED",
    "SHINKU_PRIVATE_SHELL_ENABLED", "SHINKU_PRIVATE_BROWSER_ENABLED",
    "SHINKU_VOICE_ENABLED", "SHINKU_MEMORY_SEMANTIC_ENABLED",
    "SHINKU_MEMORY_JUDGE_ENABLED", "SHINKU_MEMORY_LIFECYCLE_SEMANTIC_REVIEW_ENABLED",
    "SHINKU_CONTEXT_PROJECTION_ENABLED", "SHINKU_CONTEXT_PROJECTION_SEMANTIC_ENABLED",
    "SHINKU_CHANGE_AWARENESS_ENABLED", "SHINKU_ALLOW_LAN",
)


def _absolute(env: Mapping[str, str], key: str) -> Path:
    value = env.get(key, "").strip()
    if not value or not Path(value).is_absolute():
        raise ValueError(f"absolute_recovery_path_required:{key}")
    return Path(value).resolve()


def validate_recovery_paths(env: Mapping[str, str]) -> None:
    recovery = _absolute(env, "SHINKU_LINUX_RECOVERY_ROOT")
    source = _absolute(env, "SHINKU_LINUX_SOURCE_DATA_ROOT")
    if recovery == source or recovery.is_relative_to(source) or source.is_relative_to(recovery):
        raise ValueError("recovery_tree_must_be_separate_from_source_data")
    if not recovery.is_dir() or not source.is_dir():
        raise ValueError("recovery_and_source_directories_required")
    data = _absolute(env, "SHINKU_DATA_ROOT")
    for key in (*WRITE_ROOT_KEYS, *WRITE_FILE_KEYS):
        path = _absolute(env, key)
        if not path.is_relative_to(recovery):
            raise ValueError(f"write_path_outside_recovery_tree:{key}")
        if key in WRITE_ROOT_KEYS and not path.is_dir():
            raise ValueError(f"recovery_directory_missing:{key}")
        if key in WRITE_FILE_KEYS and not path.is_relative_to(data):
            raise ValueError(f"state_file_outside_recovery_data:{key}")
        if key in WRITE_FILE_KEYS and path.is_file():
            original = source / path.relative_to(data)
            if original.is_file() and path.samefile(original):
                raise ValueError(f"recovery_file_is_original_hardlink:{key}")
    for key in ("SHINKU_MEMORY_DB_PATH", "SHINKU_MODEL_SERVICES_PATH", "SHINKU_PERSONA_FILE"):
        if not _absolute(env, key).is_file():
            raise ValueError(f"recovery_input_missing:{key}")


def recovery_environment(
    config_path: Path, inherited: Mapping[str, str], *, local_semantic: bool = False,
) -> dict[str, str]:
    if not config_path.is_file():
        raise ValueError("recovery_config_missing")
    env = {key: value for key, value in inherited.items()
           if not key.startswith(("SHINKU_", "LLM_"))}
    load_project_env(env, path=config_path)
    validate_recovery_paths(env)
    for key in DISABLED_FLAGS:
        env[key] = "false"
    env.pop("SHINKU_RUNTIME_EXTENSION_MODULE", None)
    env["SHINKU_PRIVATE_DATA_ROOT"] = env["SHINKU_DATA_ROOT"]
    env["SHINKU_MEMORY_LIFECYCLE_MODE"] = "off"
    env["SHINKU_BACKEND_HOST"] = "127.0.0.1"
    # The CLI must not reload a profile and undo the enforced recovery flags.
    env["SHINKU_ENV_FILE"] = os.devnull
    env["LLM_PROMPT_AUDIT_ENABLED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if local_semantic:
        cache = _absolute(env, "SHINKU_MEMORY_EMBEDDING_CACHE_DIR")
        recovery = _absolute(env, "SHINKU_LINUX_RECOVERY_ROOT")
        if not cache.is_dir() or not cache.is_relative_to(recovery):
            raise ValueError("local_embedding_cache_must_be_inside_recovery_tree")
        if any(not path.resolve().is_relative_to(cache) for path in cache.rglob("*")):
            raise ValueError("local_embedding_cache_contains_external_symlink")
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            env.pop(key, None)
        env.update({
            "SHINKU_MEMORY_SEMANTIC_ENABLED": "true",
            "SHINKU_MEMORY_SEMANTIC_PROVIDER": "huggingface",
            "SHINKU_MEMORY_EMBEDDING_DEVICE": "cpu",
            "SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY": "true",
            "HF_HOME": str(_absolute(env, "SHINKU_DATA_ROOT") / "models/hf-runtime"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        })
    return env


@contextmanager
def block_ip_egress():
    """Block Python IP connects; listeners and local Unix sockets still work.

    No execution extensions are loaded in this mode. This guard applies to this
    Python process, and is not a general sandbox for arbitrary subprocesses.
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError("linux_recovery_ip_egress_disabled")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError("linux_recovery_ip_egress_disabled")
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--command", choices=("doctor", "serve"), default="serve")
    parser.add_argument("--local-semantic", action="store_true",
                        help="Enable offline CPU embeddings from a cache copied into the recovery tree")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("Linux is required")
    try:
        env = recovery_environment(args.config.resolve(), os.environ,
                                   local_semantic=args.local_semantic)
    except (OSError, ValueError) as exc:
        # Only the local fixed error category is exposed, never config values.
        category = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Linux recovery rejected: {category}", file=sys.stderr)
        return 2
    os.environ.clear()
    os.environ.update(env)
    print(json.dumps({"mode": "linux_recovery", "qq_send_enabled": False,
                      "ip_egress_enabled": False, "runtime_extension_enabled": False,
                      "local_semantic_requested": args.local_semantic,
                      "data_root": env["SHINKU_DATA_ROOT"]}), flush=True)
    from shinku.cli import main as cli_main
    command = ["doctor"] if args.command == "doctor" else ["serve", "--service", "backend"]
    with block_ip_egress():
        return int(cli_main(command))


if __name__ == "__main__":
    raise SystemExit(main())

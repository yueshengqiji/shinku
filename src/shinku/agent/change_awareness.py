"""运行时变更感知。

这个模块只记录“当前运行实例看到了什么变化”，不把源码正文放进上下文：

* 启动时把代码文件的摘要和已注册工具保存成快照；
* 与上一次启动快照比较，形成“本次启动加载了哪些变更”；
* ``read_self_state`` 被调用时再次扫描，发现进程启动后尚未重启的改动。

默认关闭。私用启动器显式打开后，公共核心也能复用同一套只读能力。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

__all__ = ["RuntimeChangeTracker"]


_DEFAULT_EXTENSIONS = frozenset({
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
})
_SKIP_DIRS = frozenset({
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
})
_SKIP_NAMES = frozenset({".env", ".env.local", ".env.private"})
_MAX_REPORTED_FILES = 100


class RuntimeChangeTracker:
    """记录代码/MCP 运行快照，并提供小而安全的诊断结果。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        scan_roots: Sequence[str | Path] = (),
        snapshot_path: str | Path | None = None,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.scan_roots = tuple(self._resolve_root(item) for item in scan_roots if str(item or "").strip())
        self.snapshot_path = Path(snapshot_path).expanduser().resolve() if snapshot_path else None
        self.extensions = frozenset(
            self._normalize_extension(item)
            for item in (extensions or _DEFAULT_EXTENSIONS)
            if self._normalize_extension(item)
        ) or _DEFAULT_EXTENSIONS
        self._lock = RLock()
        self._started = False
        self._baseline_files: dict[str, str] = {}
        self._current_files: dict[str, str] = {}
        self._baseline_tools: tuple[str, ...] = ()
        self._startup_changes: dict[str, Any] = self._empty_changes()
        self._scan_errors: list[str] = []

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeChangeTracker":
        env = os.environ if environ is None else environ
        enabled = str(env.get("SHINKU_CHANGE_AWARENESS_ENABLED", "false") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        roots_raw = str(env.get("SHINKU_CHANGE_SCAN_ROOTS", "") or "").strip()
        roots = tuple(item for item in roots_raw.split(os.pathsep) if item.strip())
        snapshot_raw = str(env.get("SHINKU_CHANGE_SNAPSHOT_PATH", "") or "").strip()
        if not snapshot_raw:
            data_root = Path(str(env.get("SHINKU_DATA_ROOT", "data") or "data"))
            snapshot_raw = str(data_root / "runtime" / "change_snapshot.json")
        extensions_raw = str(env.get("SHINKU_CHANGE_SCAN_EXTENSIONS", "") or "").strip()
        extensions = tuple(item for item in extensions_raw.split(",") if item.strip()) or None
        return cls(
            enabled=enabled,
            scan_roots=roots,
            snapshot_path=snapshot_raw,
            extensions=extensions,
        )

    @staticmethod
    def _resolve_root(value: str | Path) -> Path:
        return Path(str(value)).expanduser().resolve(strict=False)

    @staticmethod
    def _normalize_extension(value: object) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        return raw if raw.startswith(".") else "." + raw

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _empty_changes() -> dict[str, Any]:
        return {
            "added_files": [],
            "removed_files": [],
            "modified_files": [],
            "tools_added": [],
            "tools_removed": [],
            "file_change_count": 0,
            "tool_change_count": 0,
            "restart_required": False,
            "applied_at_startup": False,
        }

    def start(self, *, registered_tools: Sequence[str] = ()) -> None:
        """记录本次启动基线，并保存与上次启动的差异。"""

        if not self.enabled:
            return
        with self._lock:
            current_files = self._scan_files()
            current_tools = self._normalize_tools(registered_tools)
            previous = self._read_snapshot()
            previous_files = previous.get("files", {}) if isinstance(previous, Mapping) else {}
            previous_tools = previous.get("tools", []) if isinstance(previous, Mapping) else []
            if not isinstance(previous_files, Mapping):
                previous_files = {}
            startup_changes = self._compare(
                {str(key): str(value) for key, value in previous_files.items()},
                current_files,
                self._normalize_tools(previous_tools),
                current_tools,
            )
            if previous is None:
                startup_changes["status"] = "baseline_created"
            else:
                startup_changes["status"] = "changes_detected" if self._has_changes(startup_changes) else "unchanged"
            # 这组差异已经被本次启动加载，不应继续提示用户重启；只有
            # ``since_startup`` 里的新差异才表示当前进程需要重启。
            startup_changes["restart_required"] = False
            startup_changes["applied_at_startup"] = previous is not None and self._has_changes(startup_changes)
            self._baseline_files = current_files
            self._current_files = current_files
            self._baseline_tools = current_tools
            self._startup_changes = startup_changes
            self._started = True
            self._write_snapshot(
                {
                    "schema_version": 1,
                    "updated_at": self._now(),
                    "files": current_files,
                    "tools": list(current_tools),
                    "startup_changes": startup_changes,
                }
            )

    def diagnostics(
        self,
        *,
        registered_tools: Sequence[str] = (),
        refresh: bool = True,
    ) -> dict[str, Any]:
        """返回可注入 Agent 的摘要，不返回源文件内容。"""

        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        with self._lock:
            if refresh or not self._started:
                self._current_files = self._scan_files()
            current_files = self._current_files
            current_tools = self._normalize_tools(registered_tools)
            if not self._started:
                return {
                    "enabled": True,
                    "status": "not_started",
                    "snapshot_path": str(self.snapshot_path or ""),
                    "scan_roots": [str(item) for item in self.scan_roots],
                    "scan_errors": list(self._scan_errors),
                }
            since_startup = self._compare(
                self._baseline_files,
                current_files,
                self._baseline_tools,
                current_tools,
            )
            has_live_changes = self._has_changes(since_startup)
            startup_status = str(self._startup_changes.get("status") or "unchanged")
            if has_live_changes and startup_status == "changes_detected":
                status = "changed_at_startup_and_since_startup"
            elif has_live_changes:
                status = "changed_since_startup"
            elif startup_status == "changes_detected":
                status = "changed_at_startup"
            else:
                status = "clean"
            return {
                "enabled": True,
                "status": status,
                "summary": self._summary(status, self._startup_changes, since_startup),
                "last_startup": self._public_changes(self._startup_changes),
                "since_startup": self._public_changes(since_startup),
                "current": {
                    "file_count": len(current_files),
                    "registered_tools": list(current_tools),
                },
                "snapshot_path": str(self.snapshot_path or ""),
                "scan_roots": [str(item) for item in self.scan_roots],
                "scan_errors": list(self._scan_errors),
            }

    def _scan_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        errors: list[str] = []
        common_root = self._common_root()
        for root in self.scan_roots:
            if not root.is_dir():
                errors.append(f"missing_root:{root}")
                continue
            try:
                candidates = root.rglob("*")
                for path in candidates:
                    if not path.is_file() or path.suffix.lower() not in self.extensions:
                        continue
                    if any(part in _SKIP_DIRS for part in path.parts) or path.name in _SKIP_NAMES:
                        continue
                    try:
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                    except OSError as exc:
                        errors.append(f"unreadable:{path}:{type(exc).__name__}")
                        continue
                    try:
                        label = path.relative_to(common_root).as_posix()
                    except ValueError:
                        label = path.as_posix()
                    files[label] = digest
            except OSError as exc:
                errors.append(f"scan_failed:{root}:{type(exc).__name__}")
        self._scan_errors = errors[:20]
        return dict(sorted(files.items()))

    def _common_root(self) -> Path:
        if not self.scan_roots:
            return Path.cwd().resolve()
        try:
            common = os.path.commonpath([str(item) for item in self.scan_roots])
            return Path(common).resolve()
        except (OSError, ValueError):
            return self.scan_roots[0]

    @staticmethod
    def _normalize_tools(values: Sequence[str] | object) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            return ()
        return tuple(sorted({str(item or "").strip() for item in values if str(item or "").strip()}))

    @classmethod
    def _compare(
        cls,
        old_files: Mapping[str, str],
        new_files: Mapping[str, str],
        old_tools: Sequence[str],
        new_tools: Sequence[str],
    ) -> dict[str, Any]:
        old_keys = set(old_files)
        new_keys = set(new_files)
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        modified = sorted(key for key in old_keys & new_keys if old_files[key] != new_files[key])
        old_tool_set = set(old_tools)
        new_tool_set = set(new_tools)
        result = {
            "added_files": added[:_MAX_REPORTED_FILES],
            "removed_files": removed[:_MAX_REPORTED_FILES],
            "modified_files": modified[:_MAX_REPORTED_FILES],
            "tools_added": sorted(new_tool_set - old_tool_set),
            "tools_removed": sorted(old_tool_set - new_tool_set),
            "file_change_count": len(added) + len(removed) + len(modified),
            "tool_change_count": len(new_tool_set - old_tool_set) + len(old_tool_set - new_tool_set),
            "restart_required": bool(added or removed or modified),
        }
        return result

    @staticmethod
    def _has_changes(changes: Mapping[str, Any]) -> bool:
        return bool(
            changes.get("file_change_count")
            or changes.get("tool_change_count")
            or changes.get("tools_added")
            or changes.get("tools_removed")
        )

    @classmethod
    def _public_changes(cls, changes: Mapping[str, Any]) -> dict[str, Any]:
        result = cls._empty_changes()
        result.update({key: changes.get(key, default) for key, default in result.items()})
        if "status" in changes:
            result["status"] = str(changes.get("status") or "")
        return result

    @staticmethod
    def _summary(status: str, startup: Mapping[str, Any], live: Mapping[str, Any]) -> str:
        if status == "clean":
            return "本次运行期间没有检测到代码或工具注册变化。"
        if status == "changed_since_startup":
            return "进程启动后检测到代码文件变化；相关改动需要重启后才会进入当前运行实例。"
        if status == "changed_at_startup_and_since_startup":
            return "启动时加载过一批变更，进程启动后又发现新的代码变化；后者需要重启。"
        if status == "changed_at_startup":
            return "本次启动已经加载了上次运行以来的代码或工具变化。"
        return "已建立本次运行的变更基线。"

    def _read_snapshot(self) -> Mapping[str, Any] | None:
        if self.snapshot_path is None or not self.snapshot_path.is_file():
            return None
        try:
            decoded = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return decoded if isinstance(decoded, Mapping) else None

    def _write_snapshot(self, payload: Mapping[str, Any]) -> None:
        if self.snapshot_path is None:
            return
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.snapshot_path.parent,
                prefix=self.snapshot_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, self.snapshot_path)
        except OSError:
            # 自身状态诊断不能阻断 Agent 启动；下次仍会重新建立基线。
            return

"""Desktop pet process inspection and control."""
from __future__ import annotations
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from fastapi.responses import JSONResponse

def _find_electron_pids() -> list[int]:
    project_root = Path(__file__).resolve().parents[3]
    runtime_file = project_root / "private" / "desktop-pet" / "pet_runtime.json"

    live_electron_pids: list[int] = []
    try:
        res = subprocess.run(
            ["tasklist", "/fi", "imagename eq electron.exe", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for row in csv.reader(io.StringIO(res.stdout)):
            if row and len(row) > 1 and row[0].lower() == "electron.exe":
                try:
                    live_electron_pids.append(int(row[1]))
                except ValueError:
                    pass
    except Exception:
        pass

    if runtime_file.is_file():
        try:
            with runtime_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            recorded_pid = int(data.get("pid", 0))
            if recorded_pid and recorded_pid in live_electron_pids:
                return [recorded_pid] + [p for p in live_electron_pids if p != recorded_pid]
        except Exception:
            pass

    return live_electron_pids


def pet_snapshot(
    settings: Any,
    assembly: Any,
    *,
    find_pids: Callable[..., Any],
    trace_snapshot: Callable[..., Any],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    pet_dir = project_root / "private" / "desktop-pet"
    settings_file = pet_dir / "settings.json"
    live2d_dir = project_root / "private" / "assets" / "live2d" / "shinku"

    settings_data = {
        "scale": 1.0,
        "opacity": 1.0,
        "alwaysOnTop": True,
        "soundEnabled": True,
    }
    if settings_file.is_file():
        try:
            with settings_file.open("r", encoding="utf-8") as f:
                settings_data.update(json.load(f))
        except Exception:
            pass

    live2d_model_found = False
    live2d_model_name = ""
    if live2d_dir.is_dir():
        for mf in live2d_dir.rglob("*.model*.json"):
            live2d_model_found = True
            live2d_model_name = mf.name
            break

    pids = find_pids()
    is_running = len(pids) > 0

    traces = trace_snapshot(assembly, limit=50)
    last_turn = None
    for trace in traces:
        if trace.get("conversation_type") == "desktop":
            time_str = ""
            ts = trace.get("finished_at") or trace.get("started_at")
            if ts:
                try:
                    time_str = datetime.fromtimestamp(ts, timezone.utc).astimezone().strftime("%H:%M:%S")
                except Exception:
                    pass
            last_turn = {
                "trace_id": trace.get("trace_id", ""),
                "elapsed_ms": trace.get("elapsed_ms", 0),
                "input_chars": trace.get("input_chars", 0),
                "reply_chars": trace.get("reply_chars", 0),
                "status": trace.get("status", ""),
                "time": time_str,
            }
            break

    return {
        "available": True,
        "status": "online" if is_running else "offline",
        "pid": pids[0] if pids else None,
        "all_pids": pids,
        "character_name": "二阶堂真红",
        "worldview": "风之滨 (风之浜) / 伴侣",
        "window": {"width": 340, "height": 560},
        "live2d": {"loaded": live2d_model_found, "model": live2d_model_name},
        "settings": settings_data,
        "last_turn": last_turn,
        "launcher_available": (pet_dir / "start_pet.cmd").is_file(),
    }


async def control_pet(
    action: str,
    *,
    settings: Any,
    assembly: Any,
    find_pids: Callable[..., Any],
    snapshot: Callable[..., Any],
) -> JSONResponse:
    project_root = Path(__file__).resolve().parents[3]
    pet_dir = project_root / "private" / "desktop-pet"
    start_cmd = pet_dir / "start_pet.cmd"

    if action == "launch":
        pids = find_pids()
        if pids:
            return JSONResponse(
                {'ok': True, 'status': 'already_running', 'pids': pids, 'pet': snapshot(settings, assembly)},
            )
        if not start_cmd.is_file():
            return JSONResponse({"ok": False, "status": "launcher_not_found"}, status_code=404)
        flags = 0x00000008 | 0x00000200
        subprocess.Popen(
            ["cmd.exe", "/c", str(start_cmd)],
            cwd=str(pet_dir),
            creationflags=flags,
            close_fds=True,
        )
        return JSONResponse({"ok": True, "status": "launched", "pet": snapshot(settings, assembly)})

    elif action in {"stop", "kill", "quit"}:
        pids = find_pids()
        if not pids:
            return JSONResponse({"ok": True, "status": "not_running", "pet": snapshot(settings, assembly)})
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True, timeout=3)
            except Exception:
                pass
        runtime_file = pet_dir / "pet_runtime.json"
        if runtime_file.is_file():
            try:
                runtime_file.unlink()
            except Exception:
                pass
        return JSONResponse({"ok": True, "status": "stopped", "pet": snapshot(settings, assembly)})

    elif action == "restart":
        pids = find_pids()
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True, timeout=3)
            except Exception:
                pass
        import asyncio
        await asyncio.sleep(0.5)
        flags = 0x00000008 | 0x00000200
        subprocess.Popen(
            ["cmd.exe", "/c", str(start_cmd)],
            cwd=str(pet_dir),
            creationflags=flags,
            close_fds=True,
        )
        return JSONResponse({"ok": True, "status": "restarted", "pet": snapshot(settings, assembly)})

    elif action in {"status", ""}:
        return JSONResponse({"ok": True, "status": "status", "pet": snapshot(settings, assembly)})

    else:
        return JSONResponse(
            {"ok": False, "status": "invalid_action", "reason": "use launch, stop, restart, or status"},
            status_code=400,
        )

"""本机 Agent 后台任务查询与控制路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..tasks.coordinator import AgentTaskCoordinator

__all__ = ["build_agent_work_router"]


def _local_only(request: Request) -> bool:
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "").strip().lower() in {"127.0.0.1", "::1", "localhost", "testclient"}


def build_agent_work_router(coordinator: AgentTaskCoordinator) -> APIRouter:
    if not isinstance(coordinator, AgentTaskCoordinator):
        raise TypeError("coordinator must be an AgentTaskCoordinator")
    router = APIRouter()

    @router.get("/agent-work/tasks", response_model=None)
    def list_tasks(request: Request, profile_user_id: str = "", limit: int = 50) -> JSONResponse:
        if not _local_only(request):
            return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
        return JSONResponse(
            {"ok": True, "tasks": coordinator.list(profile_user_id=profile_user_id, limit=limit)},
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/agent-work/tasks/{task_id}", response_model=None)
    def get_task(task_id: str, request: Request, profile_user_id: str = "") -> JSONResponse:
        if not _local_only(request):
            return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
        task = coordinator.get(task_id, profile_user_id=profile_user_id)
        if task is None:
            return JSONResponse({"ok": False, "reason": "task_not_found"}, status_code=404)
        return JSONResponse({"ok": True, "task": task}, headers={"Cache-Control": "no-store"})

    async def control_task(task_id: str, request: Request, action: str) -> JSONResponse:
        if not _local_only(request):
            return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        profile_user_id = str(payload.get("profile_user_id") or "") if isinstance(payload, dict) else ""
        result = coordinator.control(action, task_id, profile_user_id)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409, headers={"Cache-Control": "no-store"})

    @router.post("/agent-work/tasks/{task_id}/cancel", response_model=None)
    async def cancel_task(task_id: str, request: Request) -> JSONResponse:
        return await control_task(task_id, request, "cancel")

    @router.post("/agent-work/tasks/{task_id}/retry", response_model=None)
    async def retry_task(task_id: str, request: Request) -> JSONResponse:
        return await control_task(task_id, request, "retry")

    return router

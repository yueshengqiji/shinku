"""私用桌面宠物专属 API 路由。

负责为独立桌宠提供：
1. 状态与角色信息查询（/api/pet/status，包含 Live2D 模型识别）
2. Live2D 资产安全托管（/api/pet/live2d/{asset_path}）
3. 本机桌面会话交互（/api/pet/chat），直接复用后端的 AgentLoop、工具集与长期记忆，
   并为桌宠返回 Live2D 表情（expression）与动作（motion）指令。
"""

from __future__ import annotations

import asyncio
import json
import logging
import hashlib
from pathlib import Path
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from ..qq.message import IncomingMessage, MessageSegment
from ..character import SHINKU_CHARACTER
from ..qq.turns import QQTurn

logger = logging.getLogger(__name__)

__all__ = ["build_pet_router"]

# 定位工程根目录与私用资源目录
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PRIVATE_ROOT = _PROJECT_ROOT / "private"
_CHARACTER_JSON_PATH = _PRIVATE_ROOT / "profile" / "shinku" / "character.json"
_LIVE2D_DIR = _PRIVATE_ROOT / "assets" / "live2d" / "shinku"


class _DesktopChatResponse(JSONResponse):
    """成功写出后执行送达回调；异常立即终结本轮，避免留下运行中的假状态。"""
    def __init__(self, *args, delivery, **kwargs):
        super().__init__(*args, **kwargs)
        self.delivery = delivery

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            self.delivery.fail()
            raise


def _guard(request: Request) -> JSONResponse | None:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse({"ok": False, "reason": "local_request_required"}, status_code=403)
    return None


def _load_character_data() -> dict[str, Any]:
    if _CHARACTER_JSON_PATH.is_file():
        try:
            return json.loads(_CHARACTER_JSON_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load character.json: %s", exc)
    return {
        "identity": {
            "name": "真红",
            "relationship": "你叫真红，全名二阶堂真红，生活在风之滨。主人是你的伴侣。",
        },
        "dialogue": {
            "input_placeholder": "和真红说点什么……",
            "local_click_lines": [
                {"emotion": "normal", "text": "嗯？怎么了。"},
                {"emotion": "normal", "text": "别戳了，我在看书呢。"},
                {"emotion": "happy", "text": "风之滨今天天气还不错。"},
                {"emotion": "thinking", "text": "……嗯？有什么事要跟我说吗？"},
            ],
        },
        "layout": {
            "outfits": {
                "default": {
                    "window": {"width": 340, "height": 560},
                    "bubble": {"anchor_x": 0.5, "anchor_y": 0.12, "max_width": 300},
                }
            }
        },
    }


def _detect_live2d_model() -> dict[str, Any]:
    """检测本地是否有可用的 Live2D 模型文件。"""
    if not _LIVE2D_DIR.is_dir():
        return {"loaded": False, "version": None, "model_url": None, "filename": None}

    # 优先检测 Cubism 4/5 (.model3.json)
    for model_file in _LIVE2D_DIR.rglob("*.model3.json"):
        rel_path = model_file.relative_to(_LIVE2D_DIR).as_posix()
        return {
            "loaded": True,
            "version": "cubism4",
            "model_url": f"/api/pet/live2d/{rel_path}",
            "filename": model_file.name,
        }

    # 其次检测 Cubism 2 (.model.json)
    for model_file in _LIVE2D_DIR.rglob("*.model.json"):
        rel_path = model_file.relative_to(_LIVE2D_DIR).as_posix()
        return {
            "loaded": True,
            "version": "cubism2",
            "model_url": f"/api/pet/live2d/{rel_path}",
            "filename": model_file.name,
        }

    return {"loaded": False, "version": None, "model_url": None, "filename": None}


def _resolve_live2d_state(text: str) -> dict[str, str]:
    """根据回复内容推断 Live2D 情绪表情（expression）与动作提示（motion）。"""
    normalized = str(text or "").lower()

    if any(k in normalized for k in ("早八", "困", "睡", "累", "歇着")):
        return {"expression": "sleepy", "motion": "Tired"}
    if any(k in normalized for k in ("幸福", "喜欢", "真好", "温暖", "抱")):
        return {"expression": "happy", "motion": "Joy"}
    if any(k in normalized for k in ("笑", "哈哈", "噗", "有趣")):
        return {"expression": "smile", "motion": "Smile"}
    if any(k in normalized for k in ("做梦", "恶心", "离我远点", "闭嘴", "少贫嘴")):
        return {"expression": "annoyed", "motion": "Displeased"}
    if any(k in normalized for k in ("笨蛋", "别扭", "想得美", "不行就是不行")):
        return {"expression": "shy", "motion": "Pout"}
    if any(k in normalized for k in ("没办法", "耸肩", "摊手", "老实待着")):
        return {"expression": "helpless", "motion": "Shrug"}
    if any(k in normalized for k in ("？", "吗", "为什么", "怎么", "思考", "看看")):
        return {"expression": "thinking", "motion": "Thinking"}
    if any(k in normalized for k in ("无聊", "装傻", "糊弄")):
        return {"expression": "blank", "motion": "Idle"}

    return {"expression": "neutral", "motion": "Talk"}


def build_pet_router(
    settings: Any = None,
    *,
    assembly: Any = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/pet")
    character = getattr(getattr(assembly, "bridge", None), "character", SHINKU_CHARACTER)
    # 桌面聊天仍复用本实例桥接；没有配置角色资产时不读默认真红的私用文件。
    explicit = not character.legacy_audience

    @router.options("/{rest_of_path:path}")
    def options_handler(rest_of_path: str) -> Response:
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    @router.get("/status")
    def pet_status(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied

        char_data = ({"identity": {"name": character.name}, "dialogue": {
            "input_placeholder": f"和{character.name}说点什么……", "local_click_lines": []}}
            if explicit else _load_character_data())
        bridge = getattr(assembly, "bridge", None)
        model_info = {"loaded": False, "version": None, "model_url": None, "filename": None} if explicit else _detect_live2d_model()

        layout_outfit = char_data.get("layout", {}).get("outfits", {}).get("default", {})
        window_cfg = layout_outfit.get("window", {"width": 340, "height": 560})
        bubble_cfg = layout_outfit.get("bubble", {"anchor_x": 0.5, "anchor_y": 0.12, "max_width": 300})

        dialogue = char_data.get("dialogue", {})
        local_lines = dialogue.get("local_click_lines", [
            {"emotion": "normal", "text": "嗯？怎么了。"},
            {"emotion": "normal", "text": "别戳了，我在看书呢。"},
            {"emotion": "happy", "text": "风之滨今天天气还不错。"},
            {"emotion": "thinking", "text": "……嗯？有什么事要跟我说吗？"},
        ])

        return JSONResponse(
            {
                "ok": True,
                "status": "online",
                "agent_ready": bridge is not None,
                "character": {
                    "name": char_data.get("identity", {}).get("name", "真红"),
                    "relationship": char_data.get("identity", {}).get("relationship", ""),
                    "dialogue": {
                        "input_placeholder": dialogue.get("input_placeholder", "和真红说点什么……"),
                        "local_click_lines": local_lines,
                    },
                    "window": window_cfg,
                    "bubble": bubble_cfg,
                    "live2d": model_info,
                },
            },
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
        )

    @router.get("/live2d/{asset_path:path}")
    def get_live2d_asset(asset_path: str, request: Request) -> Response:
        denied = _guard(request)
        if denied is not None:
            return denied

        if explicit:
            return JSONResponse({"ok": False, "reason": "instance_pet_not_configured"}, status_code=404)
        clean_path = asset_path.strip().lstrip("/\\")
        target_file = (_LIVE2D_DIR / clean_path).resolve()

        try:
            target_file.relative_to(_LIVE2D_DIR.resolve())
        except ValueError:
            return JSONResponse({"ok": False, "reason": "access_denied"}, status_code=403)

        if not target_file.is_file():
            return JSONResponse({"ok": False, "reason": "asset_not_found"}, status_code=404)

        return FileResponse(
            target_file,
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600"},
        )

    @router.post("/chat")
    async def pet_chat(request: Request) -> JSONResponse:
        denied = _guard(request)
        if denied is not None:
            return denied

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "reason": "invalid_json_body"}, status_code=400)

        text = str(body.get("text", "")).strip()
        conversation_id = str(body.get("conversation_id", "desktop_pet")).strip() or "desktop_pet"

        if not text:
            return JSONResponse({"ok": False, "reason": "empty_text"}, status_code=400)

        bridge = getattr(assembly, "bridge", None)
        if bridge is None:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "agent_not_ready",
                    "reply": "……我现在还没完全连上，稍等我一下哦。",
                    "expression": "blank",
                    "motion": "Idle",
                    "emotion": "normal",
                },
                headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
            )

        owners = tuple(getattr(bridge, "master_user_ids", ()))
        if len(owners) != 1:
            return JSONResponse({"ok": False, "reason": "desktop_owner_identity_ambiguous"}, status_code=409)
        master_id = owners[0]
        # 客户端只能选择桌面会话定位符，不能命名 QQ 私聊、群聊或其他主人的范围。
        conversation_id = "desktop:" + hashlib.sha256(
            (master_id + "\0" + conversation_id).encode("utf-8")).hexdigest()[:32]
        bot_ids = tuple(getattr(bridge, "bot_ids", ()))
        self_id = bot_ids[0] if len(bot_ids) == 1 else "shinku:desktop"
        from ..qq.local_delivery import DesktopReplyDelivery
        local_delivery = DesktopReplyDelivery(bridge, requester=master_id)

        turn_id = "desktop_" + uuid.uuid4().hex
        incoming_message = IncomingMessage(
            message_id=turn_id,
            sender_id=master_id,
            conversation_id=conversation_id,
            conversation_type="desktop",
            segments=(MessageSegment(kind="text", text=text),),
            timestamp=time.time(),
            raw_event={"self_id": self_id},
        )
        turn = QQTurn(
            conversation_id=conversation_id,
            conversation_type="desktop",
            primary_message_id=turn_id,
            message_ids=(turn_id,),
            combined_text=text,
            messages=(incoming_message,),
            local_delivery=local_delivery.prepare,
        )

        try:
            reply = await asyncio.to_thread(bridge.handle_turn, turn)
        except Exception as exc:
            logger.exception("Desktop pet turn failed: %s", exc)
            return JSONResponse(
                {
                    "ok": False,
                    "reason": f"bridge_error:{type(exc).__name__}",
                    "reply": "……刚才有点走神了，你再说一遍？",
                    "expression": "thinking",
                    "motion": "Thinking",
                    "emotion": "confused",
                },
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
            )

        final_text = str(getattr(reply, "final_text", "") or "").strip()
        if not final_text:
            final_text = "……" if getattr(reply, "agent_status", "") == "completed" else "……刚才这句没接上。"

        state = _resolve_live2d_state(final_text)

        return _DesktopChatResponse(
            {
                "ok": True,
                "reply": final_text,
                "agent_status": getattr(reply, "agent_status", "completed"),
                "expression": state["expression"],
                "motion": state["motion"],
                "emotion": state["expression"],
                "trace_id": turn_id,
            },
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
            background=BackgroundTask(local_delivery.confirm, final_text),
            delivery=local_delivery,
        )

    return router

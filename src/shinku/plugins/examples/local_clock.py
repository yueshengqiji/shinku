"""小型只读时间插件：演示现有工具、聊天只读目录和管理器的完整接线。

时间来自本机时钟，不访问网络和 QQ；时区使用系统 zoneinfo，不另建地点白名单。
插件不持有模型、记忆或送达句柄，也不改变用户请求。
"""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from shinku.plugins import PluginContributions


class LocalClockTool:
    tool_type = "local_clock"
    agent_permission_class = "read"
    agent_progress_class = "observe"
    agent_fast_perception = True
    agent_observation_required_fields = ("iso",)
    agent_observation_receipt_fields = ("iso",)

    def tool_metadata(self):
        return SimpleNamespace(description="查看本机当前时间，可指定 IANA 时区。", input_schema={
            # 沿用现有 ChatReadCapability 的对象/字符串 schema 约定，无地点枚举。
            "type": "object", "properties": {"timezone": {"type": "string", "maxLength": 4000}},
            "additionalProperties": False,
        })

    def normalize_call(self, call):
        if not isinstance(call, dict):
            return None
        timezone = call.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            return None
        return dict(call)

    def execute(self, *, call, context):
        timezone = call.get("timezone")
        now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
        return {"tool_type": self.tool_type, "ok": True, "followup_context": now.isoformat(),
                "state_updates": {"iso": now.isoformat(), "timezone": timezone or str(now.tzinfo)}}


class LocalClockPlugin:
    def activate(self, context):
        from shinku.qq.interaction_tools import ChatReadCapability
        tool = LocalClockTool()
        metadata = tool.tool_metadata()
        read = ChatReadCapability(tool.tool_type, metadata.description, metadata.input_schema)
        return PluginContributions(tools={tool.tool_type: tool}, chat_read_capabilities={read.name: read})

    def close(self):
        """本示例不创建后台资源；真实插件在此关闭自身 ExitStack。"""


def create():
    return LocalClockPlugin()

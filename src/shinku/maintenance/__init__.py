"""显式启用的自动维护：先看 host，再看 queue、agent 和 workspace。

故障来源、维护授权和模型实例均由可信宿主提供；本包不会自行加载凭据、
重放业务任务、发送消息或部署候选。维护状态独立于原任务和聊天记忆。
"""
from .queue import FailureEvent, RepairQueue, classify_failure
from .host import MaintenanceHost
from .agent import RepairAgent
from .workspace import RepairPolicy

__all__ = ['FailureEvent', 'RepairQueue', 'classify_failure', 'MaintenanceHost', 'RepairAgent', 'RepairPolicy']

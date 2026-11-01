"""内核的数据边界。身份和事件由受信本地主机传入，模型仅能提交 Decision。

阅读顺序：Identity / Event -> Decision -> store -> kernel。来源、作用域、尝试、
送达状态均不属于模型输出。这里的验证用于程序边界，不判断自然语言陈述是否属实。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Protocol


class KernelError(ValueError):
    """只暴露稳定错误码；供应商异常正文可能含凭据或私用输入，不直接输出。"""


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def text(value: Any, limit: int = 1000, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise KernelError("invalid_text")
    return value


def identifier(value: Any) -> str:
    value = text(value, 128)
    if any(ord(char) < 33 for char in value):
        raise KernelError("invalid_identifier")
    return value


def fields(value: Any, expected: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise KernelError("invalid_fields")
    return value


def timestamp(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KernelError("invalid_timestamp")
    try:
        result = float(value)
    except OverflowError:
        raise KernelError("invalid_timestamp") from None
    if not math.isfinite(result) or result < 0:
        raise KernelError("invalid_timestamp")
    return result


@dataclass(frozen=True)
class Identity:
    """用户维护的稳定身份；不内置性格。绑定数据库后不允许模型或新文件静默替换。"""

    agent_id: str
    name: str
    foundations: tuple[str, ...] = ()
    relationships: tuple[tuple[str, str], ...] = ()

    def payload(self) -> dict:
        identifier(self.agent_id)
        text(self.name, 128)
        if len(self.foundations) > 16 or len(self.relationships) > 16:
            raise KernelError("identity_too_large")
        foundations = [text(item, 500) for item in self.foundations]
        relationships = {identifier(actor): text(relation, 256) for actor, relation in self.relationships}
        if len(relationships) != len(self.relationships):
            raise KernelError("duplicate_relationship")
        return {"agent_id": self.agent_id, "name": self.name,
                "foundations": foundations, "relationships": relationships}

    @classmethod
    def parse(cls, value: Any) -> Identity:
        fields(value, {"agent_id", "name", "foundations", "relationships"})
        if not isinstance(value["foundations"], list) or not isinstance(value["relationships"], dict):
            raise KernelError("invalid_identity")
        result = cls(value["agent_id"], value["name"], tuple(value["foundations"]),
                     tuple(value["relationships"].items()))
        result.payload()
        return result


@dataclass(frozen=True)
class Event:
    """受信入口提供的事件。message 是转述；observation 必须由宿主核实来源。

    tick 也是显式宿主事件，不自行启动定时器。occurred_at 是来源时间，不用于获得权限。
    身份与作用域不可由消息正文、模型或未来网络请求自行声明。
    """

    event_id: str
    scope: str
    actor: str
    kind: str
    content: str
    occurred_at: float

    def payload(self) -> dict:
        for value in (self.event_id, self.scope, self.actor):
            identifier(value)
        if not isinstance(self.kind, str) or self.kind not in {"message", "observation", "tick"}:
            raise KernelError("invalid_event_kind")
        text(self.content, 4000)
        return {**asdict(self), "occurred_at": timestamp(self.occurred_at)}

    @classmethod
    def parse(cls, value: Any) -> Event:
        fields(value, {"event_id", "scope", "actor", "kind", "content", "occurred_at"})
        result = cls(**value)
        result.payload()
        return result


@dataclass(frozen=True)
class Decision:
    """模型只能选择表达/等待并建议状态更新，不能制造观察、身份、权限或送达回执。"""

    action: str
    reply: str
    attention: tuple[dict, ...] = ()
    understandings: tuple[dict, ...] = ()

    def payload(self) -> dict:
        return {"action": self.action, "reply": self.reply,
                "attention": list(self.attention), "understandings": list(self.understandings)}

    @classmethod
    def parse(cls, value: Any) -> Decision:
        fields(value, {"action", "reply", "attention", "understandings"})
        if not isinstance(value["action"], str) or value["action"] not in {"reply", "ask", "wait"}:
            raise KernelError("invalid_action")
        text(value["reply"], 4000, empty=value["action"] == "wait")
        if value["action"] == "wait" and value["reply"]:
            raise KernelError("wait_has_reply")
        for kind in ("attention", "understandings"):
            rows = value[kind]
            if not isinstance(rows, list) or len(rows) > 8:
                raise KernelError("too_many_changes")
            seen = set()
            for row in rows:
                fields(row, {"key", "text", "version", "evidence", "status"})
                key = identifier(row["key"])
                if key in seen:
                    raise KernelError("duplicate_change")
                seen.add(key)
                text(row["text"])
                if type(row["version"]) is not int or row["version"] < 0:
                    raise KernelError("invalid_version")
                evidence = row["evidence"]
                if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
                    raise KernelError("evidence_required")
                for event_id in evidence:
                    identifier(event_id)
                allowed = {"open", "completed", "dropped"} if kind == "attention" else {"tentative"}
                if not isinstance(row["status"], str) or row["status"] not in allowed:
                    raise KernelError("invalid_state_status")
        # 冻结独立副本，防止调用方在校验后改动嵌套列表。
        value = json.loads(encode(value))
        return cls(value["action"], value["reply"], tuple(value["attention"]), tuple(value["understandings"]))


class Decider(Protocol):
    """同步的一次有界决策；网络实现必须自行设置超时，不在协议层自动重试。"""

    def decide(self, context: dict) -> Decision: ...

"""Narrow, evidence-bound recall of the bot's own outgoing group activity.

This intent never grants access to third-party speech or to another private
conversation. No matching delivery receipt means unknown, not did-not-happen.
"""
from __future__ import annotations

import re
import time

from shinku.character import SHINKU_CHARACTER


def needs_group_activity_context(text: str) -> bool:
    """Relevance hint for bounded evidence, never a conclusion about the actor.

    Casual questions often omit 'you' or ask about enjoyment rather than speech.
    Reading our own receipts is useful there, but cannot prove watching, feelings,
    other people's actions, or permission to read any additional conversation.
    """
    text = str(text or "").strip()[:1000]
    return bool(re.search(r"群里|群聊|群中|群友|哪个群|群\s*\d+", text)
                and re.search(r"今天|今日|昨[天晚夜日]|前天|刚才|刚刚|之前|先前|上次|最近|那天|\d+[月/-]\d+", text))


def is_self_group_recall(text: str, *, aliases=SHINKU_CHARACTER.aliases) -> bool:
    text = str(text or "").strip()[:240]
    # 仅用于自身指代候选识别；别名匹配不授予身份或读取权限。
    actor = "(?:你|" + "|".join(re.escape(name) for name in aliases) + ")"
    if re.search(r"^(?:请|帮我|帮忙|麻烦)|" + actor + r"(?:现在)?(?:去|帮|发一条)", text):
        return False
    if re.search(actor + r".{0,4}(?:觉得|认为|知道|听说)", text):
        return False
    historical = re.search(r"刚才|刚刚|刚|之前|先前|上次|今天|昨日|昨天|前天|昨晚|最近|说过|聊过|回复过|发过|\d+[月/-]\d+", text)
    if not historical:
        return False
    if not re.search(r"谁|什么|哪些|哪个|怎么|是否|有没有|说没说|吗|么|[？?]|回忆|核对", text):
        return False
    group = r"(?:群里|群聊|群友|哪个群|群中|群\s*\d+)"
    speech = r"(?:说(?:话|了|过|什么)|聊|回复|发言|发.{0,4}消息|讲)"
    return bool(re.search(actor + r".{0,22}" + group + r".{0,18}" + speech, text)
                or re.search(group + r".{0,18}" + actor + r".{0,22}" + speech, text)
                or re.search(actor + r".{0,18}" + speech + r".{0,12}" + group, text))


def activity_window(text: str, *, now: int, date_window=None) -> tuple[int, int]:
    # Calendar requests stay bounded to that day. 'Just now' never scans all
    # retained history or treats a reply from yesterday as current activity.
    if date_window is not None:
        return date_window[0], min(now, date_window[1] - 1)
    minutes = re.search(r"最近\s*(\d{1,3})\s*分钟", text)
    age = max(60, min(86400, int(minutes.group(1)) * 60)) if minutes else 1800
    return now - age, now


def activity_answer(result: dict, *, query: str) -> str:
    """Host-authored factual answer; free prose cannot invent/deny an event."""
    if result.get("access") == "denied":
        return "……我不能把别处的群聊内容带到这里。"
    records = result.get("records") or []
    if not records:
        return "……这会儿我没法确认那段时间在群里和谁说过话，不敢乱说。"
    item = records[0]
    meta = item["metadata"]
    occurred = time.localtime(meta["delivered_at"])
    current = time.localtime(result.get("observed_at", time.time()))
    stamp = time.strftime("%H:%M" if occurred[:3] == current[:3] else "%m月%d日 %H:%M", occurred)
    group = item["scope"].removeprefix("group:")
    requester = meta.get("requester_name") or (f"群友（QQ {meta['requester_id']}）" if meta.get("requester_id") else "")
    target = f"那条回复对应的提问者是{requester}" if requester else "具体回复对象我没法确认"
    partial = meta.get("delivery_status") == "partial"
    if re.search(r"有没有|是否|说没说|说过话吗|发言了吗", query):
        return f"……说过，{stamp}在群里发出了{'部分' if partial else '一条'}回复。"
    location = f"群{group}" if "哪个群" in query or re.search(r"群\s*\d+", query) else "群里"
    if partial:
        return f"……{stamp}我在{location}发出了部分回复，{target}；完整内容不能确认。"
    text = item["text"]
    # Oversized/raw reclaimed or sensitive content is never revived by this
    # path; retained activity still proves occurrence without quoting it.
    if not text:
        return f"……{stamp}我在{location}发过回复，{target}；具体内容这会儿不能确认。"
    if re.search(r"谁|哪个人", query) and not re.search(r"说了什么|聊了什么|内容|原话", query):
        return f"……最近一次能确认的是，{stamp}我在{location}回过话，{target}。"
    return f"……{stamp}我在{location}回过话，{target}。我说的是：{text}"

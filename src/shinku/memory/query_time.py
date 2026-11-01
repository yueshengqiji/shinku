"""Declared record dates, separate from the semantic subject of a query."""
from collections.abc import Mapping
from datetime import date, datetime, time


def recorded_time_schema():
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "start": {"type": "string", "maxLength": 10,
                      "description": "包含的起始本地日期，YYYY-MM-DD；无下界填空字符串。"},
            "end": {"type": "string", "maxLength": 10,
                    "description": "不包含的截止本地日期，YYYY-MM-DD；无上界填空字符串。"},
        },
        "required": ["start", "end"],
        "description": "原话保存时间的筛选范围，必须显式填写。不限保存时间时start/end均为空。"
        "事件发生时间和寒暄日期留在query中，不自动限制原话保存时间。"
        "仅明确查询某段时间保存的记录时，根据本轮宿主日期填写完整日期。"
        "recent、activity及task_id精确读取时start/end均为空。",
    }


def normalize_recorded_time(value):
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise ValueError("invalid_recorded_time")
    result = {}
    for name in ("start", "end"):
        text = value[name]
        if not isinstance(text, str) or len(text) > 10:
            raise ValueError("invalid_recorded_time")
        if text:
            try:
                parsed = date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError("invalid_recorded_time") from exc
            if parsed.isoformat() != text:
                raise ValueError("invalid_recorded_time")
        result[name] = text
    if result["start"] and result["end"] and result["start"] >= result["end"]:
        raise ValueError("invalid_recorded_time_order")
    return result


def recorded_time_window(value):
    """Use the host's local calendar, as existing source timestamps do.

    Convert both midnights independently; a local day need not be 86400
    seconds across a DST transition. Model arguments cannot set a timezone.
    """
    normalized = normalize_recorded_time(value)
    bounds = []
    for name in ("start", "end"):
        text = normalized[name]
        if not text:
            bounds.append(None)
            continue
        try:
            local = datetime.combine(date.fromisoformat(text), time.min).astimezone()
            bounds.append(int(local.timestamp()))
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError("recorded_time_out_of_range") from exc
    return tuple(bounds)

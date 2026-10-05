"""Bounded evidence planning, literal coverage and source-anchored time.

All fields are derived from retained sources delivered in this request. Coverage
does not certify statements are objectively true or that all history was scanned.
Unknown syntax stays unresolved; temporal order never supplies a missing reason.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_PRICE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|人民币|USD|美元)")
_LATENCY = re.compile(r"\d+(?:\.\d+)?\s*(?:毫秒|ms|秒)(?!钟)", re.I)
_RANGE = re.compile(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})(?:日|号)?\s*(?:至|到|—|~|～|-)\s*"
                    r"(?:(20\d{2})年)?(?:(\d{1,2})月)?(\d{1,2})(?:日|号)?")


def validity_query(query: str) -> bool:
    text = str(query)
    if re.search(r"只查|那天.*(?:记录|聊天)|(?:记录|聊天).*那天|当天.*(?:说|聊|记录)", text):
        return False
    return bool(re.search(r"(?:今天|现在|目前|\d{1,2}月\d{1,2}).*(?:还|仍|是否|不在)", text)
                and re.search(r"限制期|限制|有效|适用|生效|能喝|能吃|可以喝", text))


def event_time_query(query: str) -> bool:
    if re.search(r"只查|那天.*(?:记录|聊天|说|聊)|(?:记录|聊天).*那天|说过|聊过", query):
        return False
    return bool(re.search(r"(?:搬家|看展|发生|参加|完成|出发|抵达).{0,14}(?:几月|哪一天|哪天|几号|什么时候)", query)
                or re.search(r"区分.*事件日期|按.*来源.*日期.*(?:解析|倒推)", query))


def project_for(query: str) -> str:
    text = str(query)
    for filler in ("你还记得", "你记得", "请比较", "帮我比较", "帮我检查", "请检查", "检查", "比较", "当时", "之前", "关于"):
        text = text.replace(filler, " ")
    found = re.search(r"([\u4e00-\u9fffA-Za-z0-9_-]{2,10}?项目)", text)
    if not found:
        return ""
    value = re.sub(r"^(?:我们的|我的|这个|那个|我)", "", found[1])
    return value if value not in {"项目", "刚才项目"} else ""


def _target_pattern(target: str) -> str:
    value = re.escape(target)
    return rf"(?:方案|服务)\s*{value}(?![A-Za-z])|(?<![A-Za-z]){value}(?:方案|服务)|(?<![A-Za-z]){value}(?![A-Za-z])"


def targets_for(query: str) -> list[str]:
    matches = [(match.start(), value) for value in (*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "甲", "乙")
               if (match := re.search(_target_pattern(value), query)) is not None]
    return [value for _, value in sorted(matches)]


def needs_for(query: str) -> list[dict[str, str]]:
    project = project_for(query)
    targets = targets_for(query) or [""]
    facets = []
    if re.search(r"费用|花费|成本|月费|价格|收费|预算", query):
        facets.append("cost")
    if re.search(r"延迟|耗时|响应|性能|速度|毫秒", query):
        facets.append("latency")
    result = [{"key": target + "." + facet if target else facet,
               "target": target, "facet": facet, "project": project} for target in targets for facet in facets]
    if re.search(r"为什么|为何|理由|原因", query):
        result.append({"key": "explicit_reason", "target": targets[0], "facet": "reason", "project": project})
    if validity_query(query):
        result.append({"key": "validity", "target": "", "facet": "validity", "project": project})
    elif re.search(r"几月|哪一天|几号|什么时候|哪天", query):
        result.append({"key": "event_time", "target": "", "facet": "event_time", "project": project})
    if "生日" in query:
        result.append({"key": "birthday", "target": "", "facet": "birthday", "project": project})
    if re.search(r"原话|逐字|原文", query):
        result.append({"key": "original_source", "target": "", "facet": "original", "project": project})
    return result or [{"key": "relevant_source", "target": "", "facet": "source", "project": project}]


def _zone(value: str):
    try:
        return ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        if value == "Asia/Shanghai":
            return timezone(timedelta(hours=8), "Asia/Shanghai")
        if value in {"UTC", "+00:00"}:
            return timezone.utc
        raise ValueError("unknown_source_timezone")


def source_time(record: Any, *, now: int, default_timezone: str = "Asia/Shanghai") -> dict[str, Any]:
    zone_name = str(record.metadata.get("source_timezone") or default_timezone)
    result = {"recorded_at": None, "timezone": zone_name,
              "timezone_assumed": not bool(record.metadata.get("source_timezone")),
              "event_times": [], "validity": None, "time_state": "not_specified"}
    try:
        zone = _zone(zone_name)
        recorded = datetime.fromtimestamp(record.created_at, zone)
        asof = datetime.fromtimestamp(now, zone).date()
        result["recorded_at"] = recorded.isoformat()
        body = record.text
        if (record.metadata.get("reading_representation") == "summary"
                or "summary" in record.layer or record.metadata.get("raw_detail_unavailable")):
            # A generated summary's date is not the utterance date. Parent
            # sources may be read separately, but this summary can't supply
            # a relative-time anchor or silently reconstruct recovered raw text.
            if re.search(r"上个月|上周五|昨天|这周|本周|\d{1,2}月\d{1,2}", body):
                result["time_state"] = "unresolved"
            return result
        span = _RANGE.search(body)
        if span is not None:
            year, month, day, end_year, end_month, end_day = span.groups()
            start = date(int(year or recorded.year), int(month), int(day))
            finish = date(int(end_year or year or recorded.year), int(end_month or month), int(end_day))
            if finish < start:
                raise ValueError("ambiguous_cross_year_range")
            result["validity"] = {"start": start.isoformat(), "end_exclusive": (finish + timedelta(days=1)).isoformat(),
                                  "raw_expression": span[0], "year_assumed_from_record": not bool(year),
                                  "applies_at_request": start <= asof <= finish}
            result["time_state"] = "resolved_from_source"
        if "上个月" in body:
            first = recorded.date().replace(day=1)
            start = (first - timedelta(days=1)).replace(day=1)
            result["event_times"].append({"raw_expression": "上个月", "start": start.isoformat(),
                                         "end_exclusive": first.isoformat(), "precision": "month"})
        if "上周五" in body:
            start = recorded.date() - timedelta(days=recorded.weekday() + 3)
            result["event_times"].append({"raw_expression": "上周五", "start": start.isoformat(),
                                         "end_exclusive": (start + timedelta(days=1)).isoformat(), "precision": "day"})
        if "昨天" in body:
            start = recorded.date() - timedelta(days=1)
            result["event_times"].append({"raw_expression": "昨天", "start": start.isoformat(),
                                         "end_exclusive": recorded.date().isoformat(), "precision": "day"})
        if not span and re.search(r"这周|本周", body):
            start = recorded.date() - timedelta(days=recorded.weekday())
            result["validity"] = {"start": start.isoformat(), "end_exclusive": (start + timedelta(days=7)).isoformat(),
                                  "raw_expression": "这周" if "这周" in body else "本周",
                                  "year_assumed_from_record": False, "applies_at_request": start <= asof < start + timedelta(days=7)}
        if result["event_times"] or result["validity"]:
            result["time_state"] = "resolved_from_source"
    except (ValueError, OverflowError, OSError):
        result["event_times"] = []
        result["validity"] = None
        result["time_state"] = "unresolved"
    return result


def _section(body: str, target: str) -> str:
    if not target:
        return body
    found = re.search(_target_pattern(target), body)
    if not found:
        return ""
    tail = body[found.start():]
    other_starts = [match.start() for other in (*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "甲", "乙") if other != target
                    for match in re.finditer(_target_pattern(other), tail)]
    return tail[:min(other_starts)] if other_starts else tail


def build_evidence_package(*, query: str, records, now: int, allowed_scopes: tuple[str, ...],
                           requester_id: str, checked_sources: int, budget_exhausted: bool = False,
                           default_timezone: str = "Asia/Shanghai") -> dict[str, Any]:
    selected = {}
    for record in records:
        if record.scope not in allowed_scopes or not record.metadata.get("source_version"):
            continue
        selected.setdefault((record.scope, record.record_id), record)
    sources = [{"record_id": row.record_id, "source_id": row.source_id, "scope": row.scope,
                "role": row.metadata.get("role", "unknown"), "author": row.metadata.get("sender_id", ""),
                "source_version": row.metadata["source_version"],
                "representation": row.metadata.get("reading_representation", "source"),
                "time": source_time(row, now=now, default_timezone=default_timezone)} for row in selected.values()]
    coverage = []
    for need in needs_for(query):
        supports = []
        quotes = []
        resolved_dates = []
        ambiguous = False
        for record in selected.values():
            if need["project"] and (need["project"] not in record.text or re.search(
                    r"(?:不是|不属于|并非|无关.{0,3}|不在)\s*" + re.escape(need["project"]), record.text)):
                continue
            section = _section(record.text, need["target"])
            if not section:
                continue
            facet = need["facet"]
            found = []
            if facet in {"cost", "latency"}:
                found = ( _PRICE if facet == "cost" else _LATENCY).findall(section)
                ambiguous |= len(set(found)) > 1
            elif facet == "reason":
                if (record.metadata.get("role") == "user" and record.metadata.get("sender_id") == requester_id
                        and not re.search(r"猜|可能|听说|他说|假设|建议", section)):
                    match = re.search(r"(?:因为|理由是|原因是)([^。；;]{1,160})", section)
                    if match:
                        found = [match[0]]
            elif facet in {"validity", "event_time"}:
                temporal = source_time(record, now=now, default_timezone=default_timezone)
                if facet == "validity" and temporal["validity"]:
                    found = [temporal["validity"]["raw_expression"]]
                    resolved_dates.append((temporal["validity"]["start"], temporal["validity"]["end_exclusive"]))
                if facet == "event_time":
                    found = [row["raw_expression"] for row in temporal["event_times"]]
                    resolved_dates.extend((row["start"], row["end_exclusive"]) for row in temporal["event_times"])
            elif facet == "birthday":
                if (record.metadata.get("role") == "user" and record.metadata.get("sender_id") == requester_id
                        and not re.search(r"猜|可能|听说|他说|她说|假设|建议|朋友|别人|引用", section)):
                    match = re.search(r"(?:我的?生日|我出生)[^。；;]{0,12}?\d{1,2}月\d{1,2}(?:日|号)", section)
                    if match:
                        found = [match[0]]
            elif facet == "original":
                if record.metadata.get("reading_representation") != "summary" and not record.metadata.get("raw_detail_unavailable"):
                    found = [record.text]
            elif facet == "source":
                # A generic source isn't a proof of answering an unclassified
                # question. Leave coverage not_assessed, show the source only.
                pass
            if found:
                supports.append(record.source_id)
                quotes.extend(found)
        if need["facet"] in {"cost", "latency"}:
            ambiguous |= len(set(quotes)) > 1
        if need["facet"] in {"validity", "event_time"}:
            ambiguous |= len(set(resolved_dates)) > 1
        state = ("not_assessed" if need["facet"] == "source" else "ambiguous_in_checked_sources" if ambiguous
                 else "covered" if supports else "not_found_in_checked_sources")
        coverage.append({**need, "state": state, "source_ids": list(dict.fromkeys(supports)),
                         "quotes": list(dict.fromkeys(quotes))[:12]})
    return {"request": query, "requested_at": now, "allowed_scopes": list(allowed_scopes),
            "sources": sources, "coverage": coverage,
            "missing": [row["key"] for row in coverage if row["state"] != "covered"],
            "budget_exhausted": budget_exhausted, "partial_scan": True,
            "checked_sources": max(0, int(checked_sources)), "read_state": "bounded_sources_only",
            "causal_inference": False, "internal_reasoning": False, "generator_version": "bounded-evidence-v1"}


def render_evidence_brief(package: dict[str, Any], *, max_chars: int = 500) -> str:
    lines = ["【证据覆盖与缺口】只检查本次有界来源，不代表检查全部历史；时间先后不证明原因。"]
    for row in package["coverage"]:
        state = {"covered": "已有明确来源", "not_found_in_checked_sources": "本次未找到证据",
                 "ambiguous_in_checked_sources": "来源内容或时间不一致，需核对", "not_assessed": "此问题未自动判定覆盖"}[row["state"]]
        lines.append(row["key"] + "：" + state)
    for source in package["sources"]:
        temporal = source["time"]
        for event in temporal["event_times"]:
            lines.append(f"事件时间（按来源日期）：{event['raw_expression']}→{event['start']}（{event['precision']}精度）")
        span = temporal["validity"]
        if span:
            active = "仍在范围内" if span["applies_at_request"] else "不在范围内"
            lines.append(f"来源有效期：{span['start']}—{span['end_exclusive']}（右端不含）；请求日{active}")
        if temporal["time_state"] == "unresolved":
            lines.append("某条来源时间无法确定，不能补造日期。")
    if package["budget_exhausted"]:
        lines.append("已达到本轮阅读预算，缺口不等于全库没有记录。")
    selected = []
    for line in dict.fromkeys(lines):
        if len("\n".join(selected + [line])) <= max_chars:
            selected.append(line)
    return "\n".join(selected)

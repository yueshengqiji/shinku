"""完成核验的文本与目标解析；目标判定回调由 verification 门面注入。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address
from typing import Any, Callable
from urllib.parse import urlsplit

from shinku.paths import (
    absolute_file_path,
    file_selector_matches,
    file_targets_match,
    normalize_file_path,
)

from .loop import AgentState
from .request_intent import assertion_text, execution_request_text, is_reply_composition
from .verification_rules import (
    _ACTION_SUCCESS,
    _DELETE_CLAIM_PATTERNS,
    _FILE_TARGET,
    _PROCESS_STOP_CLAIM_PATTERNS,
    _QUOTED_TARGET,
    _REMINDER_CLAIM_PATTERNS,
    _SEND_CLAIM_PATTERNS,
    _URL_TARGET,
    _WRITE_CLAIM_PATTERNS,
)


def selector_matches(selector, path) -> bool:
    if not isinstance(selector, Mapping) or not isinstance(path, str) or "://" in path:
        return False
    return file_selector_matches(selector, path)


def claims_success(text: str) -> bool:
    text = assertion_text(text)
    return bool(_ACTION_SUCCESS.search(text)) or any(
        pattern.search(text)
        for pattern in (*_WRITE_CLAIM_PATTERNS, *_DELETE_CLAIM_PATTERNS, *_SEND_CLAIM_PATTERNS, *_REMINDER_CLAIM_PATTERNS, *_PROCESS_STOP_CLAIM_PATTERNS)
    )


def receipt(call: Mapping[str, Any]) -> Mapping[str, Any]:
    data = call.get("data", {})
    updates = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
    return updates if isinstance(updates, Mapping) else {}


def user_requested_action(user_goal: str, keywords: Sequence[str]) -> bool:
    if is_reply_composition(user_goal):
        return False
    goal = execution_request_text(user_goal).strip()
    if not goal or not any(keyword in goal for keyword in keywords):
        return False
    return True


def extract_user_messages(state: AgentState) -> list[str]:
    from .request_scope import RequestScope

    context = state.context if isinstance(state.context, Mapping) else {}
    if "agent_request_scope" in context:
        scope = context["agent_request_scope"]
        # No model-authored dict or restored arbitrary JSON is an intent
        # grant. Hosts rebuild this value from validated message/task state.
        return scope.user_messages() if isinstance(scope, RequestScope) else []
    messages: list[str] = []
    for item in state.transcript:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "assistant" and (item.get("tool_calls") or str(item.get("kind") or "").lower() == "tool"):
            break
        if role == "user":
            content = str(item.get("content") or "").strip()
            if content:
                messages.append(content)
    return messages


def is_file_mention(source: str, start: int, end: int, value: str, *, quoted: bool=False) -> bool:
    if not (_FILE_TARGET.fullmatch(value) or absolute_file_path(value)):
        return False
    prefix, following = source[max(0, start - 80):start], source[end:]
    file_label = re.search(r"(?:文件|路径|file|path|read|write|读取|读入|保存|输出|输入|删除|发送)\s*[:：=]?\s*$", prefix, re.I)
    if any(c in value for c in "\\/") or absolute_file_path(value):
        return True
    if file_label:
        return True
    # A network command glued to Chinese prose can be consumed by the broad
    # filename regexp ("你ping一下221.214.140.66"). It is an address, not a
    # missing file deliverable. Explicit quoted names/paths stay authoritative.
    if not quoted:
        address = re.search(r"(?<![0-9.])([0-9]+(?:\.[0-9]+){3})$", value)
        if address and re.search(r"(?:ping(?:\.exe)?|IP|地址|主机|连接|探测)[^A-Za-z0-9.\\/]*$",
                                 value[:address.start()], re.I):
            try:
                IPv4Address(address.group(1))
                return False
            except ValueError:
                pass
    if not quoted and ("=" in value or re.fullmatch(r"\d+(?:\.\d+)+", value)):
        return False
    if not file_label and (re.match(r"\s*[:=]", following)
            or re.search(r"(?:字段|属性|键|field|property|attribute|key)\s*$", prefix, re.I)):
        return False
    # 只消解绑定在前一文件下的类限定成员；一般点分名字仍可能是嵌套文件。
    # 不按扩展名白名单猜测，显式文件标签/路径/普通引号字面文件保留。
    owner = re.search(r"(?:[`\"'“（(]([^`\"'”）)\r\n]+)[`\"'”）)]|([^\s，。；;、`\"'“”（）()]+))"
                      r"\s*(?:的|中(?:的)?|内(?:的)?|'s\s+)\s*$", prefix, re.I)
    if (owner and (not quoted or source[start:start + 1] == '`')
            and re.fullmatch(r"[A-Z]\w*(?:\.[A-Za-z_]\w*)+", value)
            and (_FILE_TARGET.fullmatch(owner.group(1) or owner.group(2))
                 or absolute_file_path(owner.group(1) or owner.group(2)))):
        return False
    return True


def quoted_target_spans(source):
    """行内代码名与原有引号同样处理；不拆开引号内的合法反引号文件名。"""
    spans = [(m.start(), m.end(), m.group(1)) for m in _QUOTED_TARGET.finditer(source)]
    for match in re.finditer(r"(?<!`)`([^`\r\n]{2,4096})`(?!`)", source):
        if not any(start < match.end() and match.start() < end for start, end, _ in spans):
            spans.append((match.start(), match.end(), match.group(1)))
    return sorted(spans)


def mask_target_spans(source, spans):
    """仅遮罩已提取部分，保留原偏移供成员绑定与明确操作角色判断。"""
    masked = list(source)
    for start, end, _ in spans:
        masked[start:end] = ' ' * (end - start)
    return ''.join(masked)


def extract_targets(
    text: str,
    *,
    file_mention: Callable[..., Any],
    http_target: Callable[..., Any],
    normalize: Callable[..., Any],
) -> list[str]:
    source = str(text or "")
    quoted_mentions = quoted_target_spans(source)
    quoted = [value for start, end, value in quoted_mentions
              if file_mention(source, start, end, value, quoted=True)]
    unquoted_source = mask_target_spans(source, quoted_mentions)
    urls = [value for _, _, value in quoted_mentions if re.match(r"https?://", value, re.IGNORECASE)]
    for match in _URL_TARGET.finditer(unquoted_source):
        url = match.group(0).rstrip(".,;")
        # Markdown/prose closing parentheses are not part of the URL, but
        # balanced path parentheses are. Preserve queries and fragments.
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        urls.append(url)
    urls = [url for url in urls if http_target(url)]
    unquoted_source = _URL_TARGET.sub(lambda m: " " * len(m.group(0)), unquoted_source)
    def unwrap_file_mention(match):
        value = (match.group(1) or match.group(2)).strip()
        prefix = re.split(r"[\s，。；：！？、]", unquoted_source[:match.start()])[-1]
        following = unquoted_source[match.end():]
        # Natural labels such as 文件（a.pdf） must not become a filename
        # containing the description. Literal path components and suffixes
        # such as C:\dir(a.pdf)\b.pdf or 报告（正式）.pdf stay intact;
        # quoted literal filenames bypass this prose normalization entirely.
        if (not any(c in prefix for c in "\\/:")
                and (not re.match(r"[\w\\/]|\.[A-Za-z0-9]", following)
                     or re.match(r"(?:中|内)的(?:\s|[`\"'“]|[A-Za-z_])", following))
                and _FILE_TARGET.fullmatch(value)):
            # 保持原偏移，后续消歧必须能读取原文中的带引号所属文件。
            return " " * (len(match.group(0)) - len(value) - 1) + value + " "
        return match.group(0)
    unquoted_source = re.sub(r"\(([^()（）\r\n]+)\)|（([^()（）\r\n]+)）",
                             unwrap_file_mention, unquoted_source)
    # Quoted property values/prose are not extra file deliverables. Avoid
    # extracting a second, truncated filename from inside a quoted path.
    # Bare versions/decimal quantities are not file paths. Explicit quoted
    # numeric filenames and numeric filenames inside paths remain targets.
    unquoted = [m.group(0) for m in _FILE_TARGET.finditer(unquoted_source)
                if file_mention(source, m.start(), m.end(), m.group(0))]
    raw_targets = [*urls, *unquoted,
                   *(value for value in quoted if "://" not in value
                     and (_FILE_TARGET.fullmatch(value) or absolute_file_path(value)))]
    targets: list[str] = []
    seen: set[str] = set()
    for raw in raw_targets:
        for normalized in normalize(raw):
            if normalized and normalized not in seen:
                seen.add(normalized)
                targets.append(normalized)
    return targets


def valid_http_target(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def normalize_target_values(value: Any) -> list[str]:
    if not isinstance(value, (str, int, float)):
        return []
    target = str(value).strip().strip("\"'“”「」『』《》")
    if re.match(r"https?://", target, re.IGNORECASE):
        return [target.rstrip("/")]
    target = target.strip(" \t\r\n\"'“”「」『』《》,，。；;：:！？!?)）]}>")
    if not target:
        return []
    if "://" in target:
        return [target.rstrip("/")]
    normalized = normalize_file_path(target)
    return [normalized]


def targets_match(expected: str, actual: str) -> bool:
    if not expected or not actual:
        return False
    if "://" in expected or "://" in actual:
        return expected == actual
    # A basename request can match an absolute path; an explicit path cannot.
    return file_targets_match(expected, actual)


def missing_effect_reason(effect: str, requested_targets: Sequence[str]) -> str:
    target_note = f"，且成功调用的工具没有作用于请求目标 {', '.join(requested_targets)}" if requested_targets else ""
    messages = {
        "write": "本轮未实际执行任何写入/保存类工具，或成功工具结果没有作用于请求目标",
        "delete": "本轮没有与请求目标匹配的成功删除工具结果；写入不能充当删除证据",
        "send": "本轮没有与请求目标匹配的成功发送工具结果",
        "reminder": "本轮没有与请求内容匹配的成功提醒设置结果",
        "process_stop": "本轮没有与请求目标匹配的成功进程停止结果",
    }
    return (
        f"完成核验未通过：{messages.get(effect, '没有对应的成功工具结果')}{target_note}。"
        "请先调用并核验对应工具；若无法完成，就如实说明阻塞原因。"
    )

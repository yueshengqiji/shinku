"""Completion verifier facade; contract checks, effect receipts and target parsing are separate responsibilities: verification_targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .request_intent import execution_request_text, positive_request_text
from .verification_rules import _FILE_TARGET, _URL_TARGET
from .verification_text import mask_target_spans, quoted_target_spans


def bind_literal_paths(source, candidates):
    """仅为原文逐字出现的完整空格路径补解析引号，不改变原请求或授予文件权限。

    已有显式引文保持。候选不能吞并两个文件名、第二个绝对路径或换行；
    模型增加一个原文没有的目标不会影响原目标的覆盖要求。
    """
    from shinku.paths import absolute_file_path
    source = str(source)
    spans = []
    quoted = quoted_target_spans(source)
    for value in sorted(set(candidates), key=len, reverse=True):
        if (not isinstance(value, str) or not absolute_file_path(value) or not re.search(r'\s', value)
                or any(c in value for c in '\r\n"\'`') or re.search(r'\s[\\/]|[A-Za-z]:[\\/].*[A-Za-z]:[\\/]', value)):
            continue
        name = value.replace('\\', '/').rsplit('/', 1)[-1]
        if not re.search(r'\.[A-Za-z0-9]{1,10}$', name) or len(list(_FILE_TARGET.finditer(name))) != 1:
            continue
        for match in re.finditer(re.escape(value), source):
            start, end = match.span()
            if (any(a < end and start < b for a, b, *_ in quoted + spans)
                    or start and not re.match(r'[\s:：=（(]', source[start - 1])
                    or end < len(source) and not re.match(r'[\s，。；;！？!?)）]', source[end])):
                continue
            spans.append((start, end, value))
    for start, end, value in sorted(spans, reverse=True):
        source = source[:start] + '"' + value + '"' + source[end:]
    return source


def file_delivery_requested(messages, path=''):
    """原请求的明确文件交付下限；不推断收件人、不授予发送权限。

    后一分句“把文件作为附件发送到当前QQ会话”没有再写路径，仍不能从契约中
    丢掉send条件。否定、引文、讨论和条件假设不产生这个下限。
    """
    from .request_intent import assertion_text
    texts = [str(v) for v in messages]
    if texts and re.search(r'(?:不要|不用|禁止|不得|别再)\s*(?:发送(?:文件|附件)?|发给我|传给我)', assertion_text(texts[-1])):
        return False
    for text in texts:
        source = assertion_text(execution_request_text(text))
        # 普通“把这句话发给我”可以在对话中回答，不因此要求文件发送工具。
        # 执行侧已有文件契约时允许“把报告发给我”；全局契约下限只认明确文件。
        file_context = bool(_FILE_TARGET.search(source) or re.search(r'文件|附件|文档', source)
                            or path and re.search(r'报告', source))
        if path:
            source = source.replace(path, ' ')
        if re.search(r'如果|假设|假如|举例|例如|我会|我自己|我打算', source):
            continue
        if file_context and re.search(r'发给我|发送(?:文件|附件)?给我|传给我|'
                     r'(?:文件|附件).{0,24}(?:发送|发|传)(?:到|至)(?:当前)?(?:QQ)?(?:会话|群)', source, re.I):
            return True
    return False


@dataclass(frozen=True)
class VerificationTargetsDependencies:
    """Bound callbacks and live getters for this responsibility."""
    is_file_mention: Callable[..., Any]
    normalize_target_values: Callable[..., Any]


def explicit_target_roles(user_messages, *, deps: VerificationTargetsDependencies):
    roles = {}
    prefixes = {
        "read": r"(?:原文(?:文件|路径)?|源文件|输入(?:文件|路径)?|从(?:本地)?文件|读取(?:本地)?(?:文件)?|读入(?:文件)?|(?:source|input)(?:\s+(?:file|path))?)\s*[:：=]?\s*$",
        "write": r"(?:输出(?:文件|路径)?|写入(?:唯一)?(?:输出|文件|路径)?|保存(?:到|至)?|创建(?:文件)?|新建(?:文件)?|修改(?:文件)?|更新(?:文件)?|覆盖(?:文件)?|(?:output|write|save|create)(?:\s+(?:file|to|path))?)\s*[:：=]?\s*$",
        "delete": r"(?:删除(?:文件)?|移除(?:文件)?|delete(?:\s+file)?)\s*[:：=]?\s*$",
        "send": r"(?:(?:发送|推送)(?:(?:已有|现有|这个|该)?(?:文件|附件))?|send(?:\s+file)?)\s*[:：=]?\s*$",
    }
    for message in user_messages:
        source = str(message)
        directives = positive_request_text(source)
        effects = execution_request_text(source)
        quoted = quoted_target_spans(source)
        mentions = [(start, end, value, True) for start, end, value in quoted]
        unquoted = mask_target_spans(source, quoted)
        unquoted = _URL_TARGET.sub(lambda m: " " * len(m.group(0)), unquoted)
        mentions.extend((m.start(), m.end(), m.group(0), False) for m in _FILE_TARGET.finditer(unquoted))
        for start, end, value, quoted in mentions:
            if "://" in value or not deps.is_file_mention(source, start, end, value, quoted=quoted):
                continue
            prefix = re.split(r"[，。；;！？!?\r\n]", directives[max(0, start - 80):start])[-1]
            for effect, pattern in prefixes.items():
                # 讨论中的文件仍可作为读取来源，但不因此成为写入/发送目标。
                effect_prefix = prefix if effect == "read" else re.split(
                    r"[，。；;！？!?\r\n]", effects[max(0, start - 80):start])[-1]
                if re.search(pattern, effect_prefix, re.I):
                    for target in deps.normalize_target_values(value):
                        roles.setdefault(target, set()).add(effect)
            # “生成a.txt并作为文件发给我”仍要求a的发送回执。只识别紧邻该目标的
            # 有界后置指令；否定/讨论已由effects遮罩，不能将别的文件或引文算到它。
            following = effects[end:end + 100]
            postfix = (r'^\s*(?:并且|并|然后|随后|再|，然后|,\s*then)?\s*'
                r'(?:(?:作为|以)(?:一个)?(?:文件|附件)(?:形式)?)?\s*'
                r'(?:发送(?:文件|附件)?给我|发给我|传给我|发送到当前(?:会话|群)|发到当前(?:会话|群)'
                r'|(?:and\s+)?send\s+(?:it|the\s+file)\s+to\s+me)')
            if re.search(postfix, following, re.I):
                for target in deps.normalize_target_values(value):
                    roles.setdefault(target, set()).add('send')
    return roles

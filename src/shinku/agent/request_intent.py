"""Bounded prose/source parsing for legacy completion checks.

These helpers neither authorize effects nor prove semantic correctness. Actual
operations still require host-scoped requests and matching, current receipts.
"""
from __future__ import annotations

import re


_QUOTED = re.compile(
    r'```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~|'
    r'`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\'|“[^”]+”|「[^」]+」|『[^』]+』'
)
_POLITE = r"(?:(?:请|帮我|替我|麻烦|你|先|直接)\s*)*"
_TEXT_OBJECT = r"(?:(?:下面|以下)的?)?(?:这|那)?(?:句话|句子|段话|段文字|段文本|文字|文本|回复)"
_TRANSFORM = r"(?:改写|修改|改成|改为|换成|编辑|润色|重写|调整)"
_REPLY_HEAD = re.compile(
    rf"{_POLITE}(?:把?{_TEXT_OBJECT}\s*{_TRANSFORM}|{_TRANSFORM}\s*{_TEXT_OBJECT})(?P<style>[^：:\n]*)"
)


def assertion_text(text: str) -> str:
    """Hide quoted examples inside prose, preserving standalone quoted reports.

    Keep offsets/newlines and never use this view to extract paths. In particular,
    putting the entire success report in quotes must not evade evidence checks.
    """
    source = str(text or "")
    masked = _QUOTED.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    return masked if masked.strip(" \t\r\n，。；;：:！!？?") else source


def positive_request_text(text: str) -> str:
    """保留原偏移，仅遮蔽本分句中被明确否定的要求，不遮蔽下一项肯定操作。

    旧实现只要整段出现一次“不要”，就把所有动作都排除，同时又漏掉“不修改”。
    这里统一处理极性范围，供意图判断、目标角色和普通分析选型共用；不改原文、
    不改变权限。双重否定有歧义时保留原句，继续严格验收而非猜成无副作用。
    """
    from .verification_rules import (
        _WRITE_KEYWORDS, _DELETE_KEYWORDS, _SEND_KEYWORDS,
        _REMINDER_KEYWORDS, _PROCESS_STOP_KEYWORDS,
    )
    source = str(text or "")
    # 引号里的否定例子不改变外部要求，但不能把原动作词一起删掉：
    # “请对文件进行‘修改’”仍须执行回执。只用去引文视图定位否定范围。
    scope = assertion_text(source)
    verbs = '|'.join(re.escape(v) for v in sorted(set((*_WRITE_KEYWORDS, *_DELETE_KEYWORDS,
        *_SEND_KEYWORDS, *_REMINDER_KEYWORDS, *_PROCESS_STOP_KEYWORDS, '写', '执行', '运行')), key=len, reverse=True))
    # 标点/转折开启新的要求；“不写文件或发送消息”中的“或”仍在否定范围内。
    boundary = re.compile(r'[，,。；;！？!?\r\n]|但是|然而|而是|然后|随后|接着|同时|但|'
        rf'(?:并且|并|且|后|还要|(?<!不)需要|必须|请)(?=\s*(?:{verbs}))|\b(?:but|then|however)\b', re.I)
    # “能不能帮我修改”中的正反问不是“不能修改”的否定，后者仍受范围遮蔽。
    negative = re.compile(r'不要|不必|无需|不需要|不用|禁止|不希望|不想|不让你|'
        rf'别(?:再|去|擅自|直接)?(?=(?:{verbs}))|不得(?!不)|(?<!能)不能(?!不)|'
        rf'不(?:再|擅自|直接|实际)?(?=(?:{verbs}))|\b(?:do\s+not|don\x27t|never)\b', re.I)
    ambiguous = re.compile(r'不能不|不得不|不要不|不是.{0,6}(?:不|禁止)|并非.{0,6}(?:不|禁止)|\bnot\s+(?:never|prohibited)\b', re.I)
    chars = list(source)
    starts = [0]
    spans = []
    for match in boundary.finditer(scope):
        spans.append((starts[-1], match.start()))
        starts.append(match.end())
    spans.append((starts[-1], len(source)))
    for start, end in spans:
        clause = scope[start:end]
        if ambiguous.search(clause):
            continue
        match = negative.search(clause)
        if match:
            chars[start + match.start():end] = ' ' * (end - start - match.start())
    return ''.join(chars)


def execution_request_text(text: str) -> str:
    """为旧动作验收投影本轮要求；讨论的宾语不自动变成执行要求。

    原文和宿主 RequestScope 不变，本视图只供动作预检、效果目标和角色核验。
    先沿用否定范围，再按分句及带动作谓词的连接词隔离要求；信息请求与
    用户自述只遮蔽自己支配的内容，不能吞掉后续“然后保存”或“请修改”。
    无法识别的表达保留原有严格检查；本解析既不授权工具，也不取消契约
    条件或模型成功声明所需的真实回执。
    """
    from .verification_rules import (
        _WRITE_KEYWORDS, _DELETE_KEYWORDS, _SEND_KEYWORDS,
        _REMINDER_KEYWORDS, _PROCESS_STOP_KEYWORDS,
        _INFORMATION_REQUEST, _INFORMATIONAL_SHAPE, _EXPLICIT_ACTION_REQUEST, _ACTION_REQUESTER,
    )
    source = positive_request_text(text)
    if is_reply_composition(source):
        return re.sub(r'[^\n]', ' ', source)
    # 引文里的标点/连接词不是外层要求边界；仍保留原偏移以匹配目标路径。
    scope = assertion_text(source)
    verbs = '|'.join(re.escape(v) for v in sorted(set((
        *_WRITE_KEYWORDS, *_DELETE_KEYWORDS, *_SEND_KEYWORDS,
        *_REMINDER_KEYWORDS, *_PROCESS_STOP_KEYWORDS, '执行', '运行',
    )), key=len, reverse=True))
    modifiers = r'(?:(?:请|帮我|替我|麻烦|先|再|直接|实际|也)\s*)*'
    boundary = re.compile(
        r'[，,。；;！？!?\r\n]|'
        rf'(?:然后|随后|接着|同时|但是|而是|但|并且|并|且|后|再|就|完|还要)'
        rf'(?=\s*{modifiers}(?:{verbs}|把|对|在|将))|'
        rf'(?:请|帮我|替我|麻烦)(?=\s*{modifiers}(?:{verbs}|把|对|在|将))', re.I,
    )
    chars = list(source)
    start = 0
    spans = []
    for match in boundary.finditer(scope):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(source)))
    for start, end in spans:
        raw_clause = scope[start:end]
        leading = len(raw_clause) - len(raw_clause.lstrip())
        clause = raw_clause.strip()
        # “帮我讨论修改流程”的最外层谓词仍是讨论；不能让礼貌前缀使
        # 内层“修改”越过语法范围。独立行动已在上面切分，继续严格核验。
        polite_end = re.match(_POLITE, clause).end()
        head_start = polite_end if _INFORMATION_REQUEST.match(clause[polite_end:]) else 0
        information = _INFORMATION_REQUEST.match(clause[head_start:])
        mask_end = end
        if information:
            # 复用原明确请求语法定位内层独立要求，不让“分析……需要你修改”
            # 被讨论范围吞掉；只保留执行部分，讨论中的另一文件不绑定为写目标。
            operand_start = head_start + information.end()
            direct = next((m for m in _EXPLICIT_ACTION_REQUEST.finditer(clause[operand_start:])
                           if re.match(_ACTION_REQUESTER, m.group())), None)
            if direct:
                mask_end = start + leading + operand_start + direct.start()
        if (information or _self_reported_action(clause, verbs) or _reported_text_reference(clause)
                or (_INFORMATIONAL_SHAPE.search(clause)
                            and not _EXPLICIT_ACTION_REQUEST.search(clause))):
            chars[start:mask_end] = re.sub(r'[^\n]', ' ', source[start:mask_end])
    return ''.join(chars)


def _reported_text_reference(clause: str) -> bool:
    """文本中出现动作词属于内容描述，不能自动成为本轮外部操作。

    只识别以聊天/文本为主语、以出现/提到等为谓语的分句；独立的后续操作由
    execution_request_text 分句保留，内层明确请求也继续核验。此处不改原请求、
    不授权工具，也不影响已声明的宿主验收契约。
    """
    from .verification_rules import _EXPLICIT_ACTION_REQUEST

    return bool(re.match(
        r"(?:聊天|对话|回复|文字|文本|这段话|这句话|这段文字|这段文本)"
        r"[^，,。；;！？!?\r\n]{0,40}?(?:中|里)(?:出现|提到|包含|写着)(?:了)?", clause,
    )) and not (_EXPLICIT_ACTION_REQUEST.search(clause)
                or re.search(r"需要?|必须|应当|应该", clause))


def _self_reported_action(clause: str, verbs: str) -> bool:
    """识别本分句中用户自己的既有操作，避免把“我保存得很好”当成指令。

    只接受明确第一人称、单个动作以及完成/持续体或程度补语；相对从句、
    愿望、明确请求和多动作歧义仍保留原检查。分句边界由上游持有，后续
    操作独立核验。本投影既不修改原请求，也不遮蔽模型声明或宿主契约。
    """
    from .verification_rules import _EXPLICIT_ACTION_REQUEST

    if _EXPLICIT_ACTION_REQUEST.search(clause):
        return False
    actions = list(re.finditer(verbs, clause))
    if len(actions) != 1:
        return False
    action = actions[0]
    subject = re.search(
        r'(?:我|我们)(?:(?:可是|确实|已经|早就|刚刚|刚才|之前|曾经|一直|也|都|可|亲手)\s*)*'
        r'(?:(?:把|将)[^，,。；;！？!?\r\n]{1,80}?)?\s*$', clause[:action.start()],
    )
    if subject is None:
        return False
    return bool(re.fullmatch(
        r'(?:了|过|着).{0,120}|[得的](?:很|挺|非常|相当|特别|足够).{1,40}',
        clause[action.end():],
    ))


def is_reply_composition(text: str) -> bool:
    """Recognize a transformation of an explicit in-chat text operand.

    A colon introduces source text, not executable instructions. A quoted source
    ends at its closing delimiter; trailing operations remain outside this form.
    Anything outside this narrow syntax uses the existing conservative checks.
    """
    # Import locally so rules remain the single inventory of legacy effect words.
    from .verification_rules import (
        _DELETE_KEYWORDS, _FILE_TARGET, _PROCESS_STOP_KEYWORDS,
        _REMINDER_KEYWORDS, _SEND_KEYWORDS, _WRITE_KEYWORDS,
    )

    source = str(text or "").strip()
    parts = re.split(r"[:：]", source, maxsplit=1)
    head = _REPLY_HEAD.fullmatch(parts[0].strip())
    if head is None:
        return False
    style = head.group("style")
    effects = (*_WRITE_KEYWORDS, *_DELETE_KEYWORDS, *_SEND_KEYWORDS,
               *_REMINDER_KEYWORDS, *_PROCESS_STOP_KEYWORDS)
    if (_FILE_TARGET.search(style) or re.search(r"[。；;！？!?\n]", style)
            or any(word in style for word in effects)):
        return False
    if len(parts) == 2:
        payload = parts[1].strip()
        if not payload:
            return False
        quoted = _QUOTED.match(payload)
        if quoted and payload[quoted.end():].strip(" \t\r\n，。；;！!"):
            return False
    return True

"""发送前的证据门：削掉无据的自述与拿供应方内部当解释的句子。

为什么要单独一层（而不是写进口径）：口径只是引导，模型偶尔仍会写「读完了」「我看了日志」，
而这一轮**根本没有工具事件**——这是事实性错误，不是语气问题，所以除口径外还要一层确定性的
发送前兜底。

这层只做**减法**：把没有证据支撑的那一句去掉、把拿供应方内部当解释的那一句去掉，其余原样保留；
不重写、不扩写、不发明事实。整段都只剩这类句子时，才换成一句简短的核查话术。

两个原因码：``unsupported_claim``（声称查过但本轮无据）、``provider_speculation``
（把偏差解释成供应方内部）。后者任何情况下都删，前者在有证据时不动。

判定用词表与分层查找实现，不采用单个巨型正则——判定边界见行为契约 §3.3 的两张表。
"""

from __future__ import annotations

from typing import Any

#: 句末与分隔标点。切句时把它们留在句子末尾；换行只作分隔、不保留。
_SENTENCE_ENDINGS = frozenset("。！？!?；;")

#: 「看/查」类动词，用于识别第一人称的完成态自述。
_LOOKUP_VERBS = ("查", "看", "读", "翻", "核对", "确认")
#: 跟在动词后面的完成标记。注意不含单独的「完」——「查完就告诉你」不算。
_COMPLETION_MARKS = ("了", "过", "完了", "了一遍", "过了")
#: 第一人称后可能出现的时间副词（长词在前，避免「刚刚」被「刚」截断）。
_TIME_HINTS = ("已经", "刚才", "刚刚", "刚", "这边", "手里")

#: 「……到了」形态的无主语完成态。
_ARRIVAL_VERBS = ("查", "看", "读", "翻")
_ARRIVAL_TAIL = "到了"

#: 「……完/过了」形态的无主语完成态。
_FINISH_VERBS = ("查", "看", "读", "翻", "核对")
_FINISH_MARKS = ("完", "过")

#: 「我手里/这边有……结果/数据」形态的持有自述。
_HOLDING_HEADS = ("我手里", "我这边")
_HELD_NOUNS = ("结果", "数据")

#: 「引用了某份资料」的句式：名词（可带「里」）+ 动词。
_CITATION_RULES = (
    (("文档", "README", "说明"), ("写", "说")),
    (("日志",), ("写", "说", "报")),
    (("配置", "文件", "代码"), ("写", "有")),
)

#: 供应方侧干预的动词。两类主体的模态/动作词略有不同，故分列。
_INTERFERENCE_VERBS_SERVER = ("截断", "替换", "改写", "过滤", "审查", "拦截")
_INTERFERENCE_VERBS_VENDOR = ("截断", "替换", "改写", "过滤", "审查", "拦截", "删")
_MODAL_HINTS_SERVER = ("确实", "可能", "大概", "会")
_MODAL_HINTS_VENDOR = ("可能", "确实", "会")
_ACTION_HINTS_SERVER = ("进行了", "做了", "进行", "做")
_ACTION_HINTS_VENDOR = ("进行了", "做了", "做")

#: 不可观测的内部概念，出现即命中。
_OPAQUE_CONCEPTS = (
    "模型权重",
    "内部那一层",
    "内置规则",
    "内置提示",
    "内置审查",
    "不在你手里",
)
#: 「隐藏（的）（输出）审查」的各种写法。
_HIDDEN_REVIEW_FORMS = ("隐藏审查", "隐藏的审查", "隐藏输出审查", "隐藏的输出审查")
#: 「投递链路……改/截断/替换/吞」的中间可容字符数。
_DELIVERY_SPAN = 12
_DELIVERY_TOKENS = ("截断", "替换", "改", "吞")

#: 整段只剩无据内容时替换成的短句：说明还没查，不冒充查过。属面向用户的输出，逐字保留。
CHECK_AGAIN_LINE = "这个我得先去看一眼，不能凭印象说。"


def _split_sentences(text: str) -> list[str]:
    """按句末标点与换行切句，去空白后保留非空段落。"""

    units: list[str] = []
    pending: list[str] = []
    for char in text:
        if char == "\n":
            _take(pending, units)
        else:
            pending.append(char)
            if char in _SENTENCE_ENDINGS:
                _take(pending, units)
    _take(pending, units)
    return units


def _take(pending: list[str], units: list[str]) -> None:
    piece = "".join(pending).strip()
    pending.clear()
    if piece:
        units.append(piece)


def _self_reporting_lookup(sentence: str) -> bool:
    """「我（已经）查了」这一类：第一人称 + 查看动词 + 完成标记。"""

    cursor = 0
    while True:
        head = sentence.find("我", cursor)
        if head < 0:
            return False
        cursor = head + 1
        index = head + 1
        for hint in _TIME_HINTS:
            if sentence.startswith(hint, index):
                index += len(hint)
                break
        if sentence.startswith("去", index):
            index += 1
        for verb in _LOOKUP_VERBS:
            if sentence.startswith(verb, index):
                after = index + len(verb)
                if any(sentence.startswith(mark, after) for mark in _COMPLETION_MARKS):
                    return True
                break


def _reported_arrival(sentence: str) -> bool:
    """「查到了」这类无主语的完成态。"""

    return any(verb + _ARRIVAL_TAIL in sentence for verb in _ARRIVAL_VERBS)


def _reported_completion(sentence: str) -> bool:
    """「查完了」这类无主语的完成态。"""

    return any(
        verb + mark + "了" in sentence
        for verb in _FINISH_VERBS
        for mark in _FINISH_MARKS
    )


def _claims_possession(sentence: str) -> bool:
    """「我手里有结果」这类持有自述——也是在暗示已经查过。"""

    return any(
        head + "有" + noun in sentence
        for head in _HOLDING_HEADS
        for noun in _HELD_NOUNS
    )


def _cites_source(sentence: str) -> bool:
    """引用了文档/日志/配置文件——同样是在声称已经查过。"""

    return any(
        _noun_then_verb(sentence, noun, verb)
        for nouns, verbs in _CITATION_RULES
        for noun in nouns
        for verb in verbs
    )


def _noun_then_verb(sentence: str, noun: str, verb: str) -> bool:
    """名词（可带「里」）后面紧跟动词即命中。"""

    cursor = 0
    while True:
        head = sentence.find(noun, cursor)
        if head < 0:
            return False
        cursor = head + 1
        index = head + len(noun)
        if sentence.startswith("里", index):
            index += 1
        if sentence.startswith(verb, index):
            return True


def _per_reading(sentence: str) -> bool:
    """「按（我）（刚才）读到的」——把引用说成已完成。"""

    cursor = 0
    while True:
        head = sentence.find("按", cursor)
        if head < 0:
            return False
        cursor = head + 1
        index = head + 1
        if sentence.startswith("我", index):
            index += 1
        for hint in ("刚才", "刚"):
            if sentence.startswith(hint, index):
                index += len(hint)
                break
        if sentence.startswith("读到的", index):
            return True


def _claims_past_lookup(sentence: str) -> bool:
    """句子是否在声称「已经查过」。"""

    return (
        _self_reporting_lookup(sentence)
        or _reported_arrival(sentence)
        or _reported_completion(sentence)
        or _claims_possession(sentence)
        or _cites_source(sentence)
        or _per_reading(sentence)
    )


def _actor_interference(
    sentence: str,
    actors: tuple[str, ...],
    verbs: tuple[str, ...],
    modals: tuple[str, ...],
    actions: tuple[str, ...],
    *,
    with_scope: bool,
) -> bool:
    """「某主体 + （内部/层面）+ （模态）+ （动作）+ 干预动词」这种推测句。"""

    for actor in actors:
        cursor = 0
        while True:
            head = sentence.find(actor, cursor)
            if head < 0:
                break
            cursor = head + 1
            index = head + len(actor)
            if with_scope:
                for scope in ("内部", "层面"):
                    if sentence.startswith(scope, index):
                        index += len(scope)
                        break
            for hint in modals:
                if sentence.startswith(hint, index):
                    index += len(hint)
                    break
            for hint in actions:
                if sentence.startswith(hint, index):
                    index += len(hint)
                    break
            if any(sentence.startswith(verb, index) for verb in verbs):
                return True
    return False


def _delivery_link_edit(sentence: str) -> bool:
    """「投递链路……改了/截断/替换/吞」这种把改动归给链路的句子。"""

    cursor = 0
    while True:
        head = sentence.find("投递链路", cursor)
        if head < 0:
            return False
        cursor = head + 1
        start = head + len("投递链路")
        for offset in range(_DELIVERY_SPAN + 1):
            index = start + offset
            if any(sentence.startswith(token, index) for token in _DELIVERY_TOKENS):
                return True


def _speculates_provider(sentence: str) -> bool:
    """句子是否把行为解释成供应方内部。"""

    return (
        _actor_interference(
            sentence,
            ("服务端",),
            _INTERFERENCE_VERBS_SERVER,
            _MODAL_HINTS_SERVER,
            _ACTION_HINTS_SERVER,
            with_scope=False,
        )
        or _actor_interference(
            sentence,
            ("供应方", "供应商", "平台"),
            _INTERFERENCE_VERBS_VENDOR,
            _MODAL_HINTS_VENDOR,
            _ACTION_HINTS_VENDOR,
            with_scope=True,
        )
        or any(concept in sentence for concept in _OPAQUE_CONCEPTS)
        or any(form in sentence for form in _HIDDEN_REVIEW_FORMS)
        or _delivery_link_edit(sentence)
    )


def strip_unsupported_claims(text: Any, *, has_evidence: bool) -> tuple[str, list[str]]:
    """把「自称已经查看过」而本轮又拿不出依据的句子删掉。

    返回清理后的文本与命中的原因码列表；没有命中时原样返回输入。
    """

    value = str(text or "")
    if not value.strip() or has_evidence:
        return value, []
    kept: list[str] = []
    reasons: list[str] = []
    for unit in _split_sentences(value):
        if _claims_past_lookup(unit):
            reasons.append("unsupported_claim")
            continue
        kept.append(unit)
    if not reasons:
        return value, []
    return "".join(kept).strip(), reasons


def strip_provider_speculation(text: Any) -> tuple[str, list[str]]:
    """删掉那些把现象归因到供应方不可观测内部的句子。"""

    value = str(text or "")
    if not value.strip():
        return value, []
    kept: list[str] = []
    reasons: list[str] = []
    for unit in _split_sentences(value):
        if _speculates_provider(unit):
            reasons.append("provider_speculation")
            continue
        kept.append(unit)
    if not reasons:
        return value, []
    return "".join(kept).strip(), reasons


def apply_evidence_guard(
    *,
    speech: Any,
    speech_segments: Any,
    has_evidence: bool,
) -> tuple[str, list[str], list[str]]:
    """发送前对最终台词跑一遍两道判定。

    本轮确有工具事件时，只做第二道——把归因到供应方的句子削掉，第一道放过，
    那时「查过」是站得住的。否则两道都做。
    """

    raw_text = str(speech or "")
    raw_segments = [str(item) for item in speech_segments] if isinstance(speech_segments, list) else []
    reasons: list[str] = []

    def scrub(value: str) -> str:
        staged, claim_reasons = strip_unsupported_claims(value, has_evidence=has_evidence)
        staged, speculation_reasons = strip_provider_speculation(staged)
        reasons.extend(claim_reasons)
        reasons.extend(speculation_reasons)
        return staged

    text = scrub(raw_text)
    surviving = [part for part in (scrub(part) for part in raw_segments) if part.strip()]
    if surviving:
        text = "\n".join(surviving)
    elif not text.strip() and reasons:
        # 整段都被清空且确实有命中：换成核查话术，不把原句发出去。
        text = CHECK_AGAIN_LINE
        surviving = [CHECK_AGAIN_LINE]
    return text, surviving, reasons


__all__ = [
    "CHECK_AGAIN_LINE",
    "apply_evidence_guard",
    "strip_provider_speculation",
    "strip_unsupported_claims",
]

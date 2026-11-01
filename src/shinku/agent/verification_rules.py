"""Agent 交付核验门禁 (Action Completion Verification Gate)。

用于防范大模型在多步执行任务中的“假性完成与行动脱节”幻觉：
即用户要求了修改、写入、发送或设置操作，模型读取信息后倾向于直接在文本中
声称“好的，已经帮你做好了”，但实际上根本没有调用对应的操作类工具。
"""

from __future__ import annotations

import re


# 操作意图关键词匹配
_WRITE_KEYWORDS = (
    "修改", "改写", "改成", "改为", "写入", "写进", "保存", "创建", "新建", "覆盖", "更新", "保存到",
    "编辑", "追加", "替换", "换成", "设为", "设成",
)
_DELETE_KEYWORDS = ("删除", "移除", "清理", "清空")
_DELETE_CLAIM_PATTERNS = [re.compile(r"已(?:经|成功)?.*?(?:删除|删掉|删了|移除|清理|清空)")]

_WRITE_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(保存|修改|更新|写入|创建|替换|覆盖)"),
    re.compile(r"已(经)?.*?(将|把).*?(保存|写入|修改|更新|替换|覆盖|改成|改为)"),
    re.compile(r"已(成功)?保存(到|至)"),
    re.compile(r"已(成功)?(新建|创建)了?"),
    re.compile(r"(参数|配置|文件|内容).*?(已|已经|成功)(修改|变更为|更改为|更新为|设置为)"),
]

# Compatibility for legacy text-only final decisions. This does not select tools.
_ACTION_SUCCESS = re.compile(
    '(?:^\\s*(?:好了|完成了|搞定了)|(?:已(?:经)?|成功).{0,24}(?:完成|修好|打开|运行)|正常(?:打开|运行|访问)|修好了)',
)

_SEND_KEYWORDS = (
    "发送", "发到群", "发给我", "点播", "推送到", "发这首", "发这篇", "发文件", "发图", "发截图", "分享这首",
)

_SEND_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(发送|点播|推送|分享)"),
    re.compile(r"卡片已(发送|送达)"),
]

_REMINDER_KEYWORDS = ("提醒我", "定个闹钟", "设置提醒", "设个提醒", "添加提醒")
_REMINDER_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(设置|定好|添加)(了)?提醒"),
    re.compile(r"已(成功)?创建提醒"),
]

_PROCESS_STOP_KEYWORDS = ("停止进程", "杀掉进程", "关闭进程", "结束进程", "kill进程")
_PROCESS_STOP_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(停止|杀掉|关闭|结束)(了)?进程"),
]

_INFORMATION_REQUEST = re.compile(
    r"^\s*(?:(?:请)?(?:告诉我|解释|说明|介绍|分析|评估|讲讲|讨论|探讨)|我(?:想知道|想了解|想问|想请教|想确认)|"
    r"为什么|怎么(?:做|才能|去)?|如何|能不能|是否|是不是|什么是|请问|你觉得|"
    r"给我(?:一个)?(?:方案|建议)|列出|有哪些)"
)
_INFORMATIONAL_SHAPE = re.compile(
    r"(?:工作原理|怎么工作|如何工作|是什么|什么意思|有什么作用|有什么区别|有哪些风险|"
    r"怎么实现|如何实现|原理|教程|操作步骤|为什么|是否|能否|可不可以|行不行)"
)
# 共用明确请求者前缀，避免分句投影与旧动作预检维护两套意图词表。
_ACTION_REQUESTER = r"(?:帮我|替我|帮忙|我想|我要|需要你)"
_EXPLICIT_ACTION_REQUEST = re.compile(
    _ACTION_REQUESTER + r".{0,40}(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
    r"|^\s*(?:请)?\s*(?:直接)?\s*(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
)
_NEGATED_ACTION_REQUEST = re.compile(
    r"(?:不要|别(?!人)|不必|无需|不需要|不用|禁止|不希望|不想|不让你).{0,20}"
    r"(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
)

# Deliberately narrow compatibility guard: informational translation questions
# and plain in-chat translations remain unchanged. This NEVER grants a tool.
_PDF_TRANSLATION = re.compile(
    '(?:翻译|汉化|译成|英译中).{0,200}(?:PDF|pdf)|(?:PDF|pdf).{0,200}(?:翻译|汉化|译成|英译中)',
    re.S,
)
_FULL_DOCUMENT = re.compile(r"全文|完整|全部|所有|整篇|全篇|(?:第\s*)?1\s*[-—到至]\s*\d+\s*页|(?<![第\d])\d+\s*页")

# 明确的工具能力登记：核验不再通过工具名中是否“包含 write/send”等片段猜测。
# 自定义 handler 可声明相同属性；未登记的工具不会被当成副作用证据。
_DEFAULT_TOOL_CAPABILITIES: dict[str, dict[str, tuple[str, ...]]] = {
    "private_file_read": {"effects": ("read", "tool:private_file_read"), "target_arguments": ("path",)},
    "file_write": {"effects": ("write",), "target_arguments": ("path",)},
    "write_file": {"effects": ("write",), "target_arguments": ("path",)},
    "private_file_write": {"effects": ("write",), "target_arguments": ("path",)},
    "file_save": {"effects": ("write",), "target_arguments": ("path",)},
    "private_file_delete": {"effects": ("delete",), "target_arguments": ("path",)},
    "file_delete": {"effects": ("delete",), "target_arguments": ("path",)},
    "private_browser_open": {"effects": ("browser_access",), "target_arguments": ("url",)},
    "private_qq_send_file": {"effects": ("send",), "target_arguments": ("path",)},
    "private_qq_send_netease_music": {"effects": ("send",), "target_arguments": ()},
    "private_reminder_set": {"effects": ("reminder",), "target_arguments": ("message",)},
    "private_process_stop": {"effects": ("process_stop",), "target_arguments": ("name", "pid")},
}

_FILE_TARGET = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/])?(?:[^\s\"'<>|，。；：！？、]+[\\/])*"
    r"[^\s\"'<>|，。；：！？、\\/]+\.[A-Za-z0-9]{1,10}(?![\w])"
)
_QUOTED_TARGET = re.compile(r"[\"“「『《]([^\"”」』》]{2,4096})[\"”」』》]")
# Bare RFC-3986 URLs end before adjacent Chinese prose. Unicode URL components
# remain literal when quoted (or percent-encoded); never parse their host/path
# again as a local filename. This extracts targets, not permissions or tools.
_URL_TARGET = re.compile(
    r"https?://(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._~!$&()*+,;=%:@-]+)(?::\d+)?"
    r"(?:[/?#][A-Za-z0-9._~:/?#@!$&()*+,;=%-]*)?", re.IGNORECASE,
)
_UNCHANGED_INPUTS = re.compile(
    r"(?:原文件|源文件|输入文件|原文).{0,12}(?:保持不变|不得修改|不能修改|不要修改|不修改)"
    r"|(?:不要|禁止|不得).{0,12}(?:修改|覆盖|更新).{0,6}(?:原文件|源文件|输入文件|原文)"
)

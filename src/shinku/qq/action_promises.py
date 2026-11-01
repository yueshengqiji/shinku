"""A narrow guard for unstarted external actions in direct QQ chat replies.

Tool results and background acknowledgements use their own receipt gates. This
guard applies only before a plain reply has initiated any action. It is a text
compatibility check, not evidence that an operation did or did not succeed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_QUOTES = re.compile(r"```[\s\S]*?```|`[^`\n]*`|“[^”]*”|「[^」]*」|『[^』]*』|\"[^\"\n]*\"")
_CONDITION = re.compile(r"^(?:如果|假如|假设|要是|等(?:你|主人|到)|待|获准后|你同意后|你确认后|到时候)")
_ACTION = re.compile(
    r"^(?:(?:嗯|好|好的|行|那|那么|收到|了解|放心)[…\s]*)?"
    r"我(?:这就|马上|立刻|现在|正在|已经在|接下来(?:就)?|这会儿|会|去)"
    r"(?:(?:去|来|帮你|替你|给你|先|就|再|亲自)[\s]*){0,3}"
    r"(?:查(?:询|看|证|找)?|检查|核对|检测|排查|测试|诊断|修改|修复|调整|重启|运行|执行|"
    r"安装|下载|上传|发送|删除|读取|打开|搜索|联网)(?P<target>[^。！？!?；;\n]{0,90})"
)
_EXTERNAL_TARGET = re.compile(
    r"分流|网络|代理|服务|后台|日志|消息|QQ|系统|工具|脚本|配置|文件|程序|接口|端口|"
    r"网站|网页|资料|新闻|天气|版本|数据库|进程|目录|磁盘|权限|验证码|https?://|[A-Z]:[\\/]",
    re.IGNORECASE,
)
_EMPTY_TARGET = re.compile(r"^(?:一下|下|一遍|一会儿|了|啦|哦|啊|呀|吧|呢|哈|的|看看|[\s，,：:…])*$")


def has_unstarted_action_promise(text: str) -> bool:
    """Exclude quoted examples, conditional plans and ordinary in-chat work."""
    value = _QUOTES.sub("", str(text or ""))
    for sentence in re.split(r"[。！？!?；;\n]+", value):
        sentence = sentence.strip(" \t…~～")
        if not sentence or sentence.startswith(">") or _CONDITION.search(sentence):
            continue
        for clause in re.split(r"[，,：:]+", sentence):
            match = _ACTION.match(clause.strip(" \t…~～"))
            if match and (_EXTERNAL_TARGET.search(match["target"]) or _EMPTY_TARGET.fullmatch(match["target"])):
                return True
    return False


def unstarted_action_reply(context: Mapping) -> str:
    audience = context.get("audience_context")
    if isinstance(audience, Mapping) and audience.get("role") == "other":
        return "这件事我还没有实际执行；需要主人发起，我才能调用工具。"
    return "这件事我还没有实际执行，暂时不能确认结果。"

"""Agent 独立检索策略：精确标识符与中英文有界词项，不调用 Bot Router。

任务连续性先按真实任务 ID 读取，此检索仅补长期陈述/经验。FTS 索引是
导航；命中不代表来源有效，更不代表当前事实或授权。没有向量模型调用。
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AgentMemoryPolicy:
    max_results: int = 4
    candidate_limit: int = 32
    max_query_chars: int = 512
    max_history_turns: int = 6
    max_records: int = 4096

    def __post_init__(self):
        for key, low, high in [('max_results', 1, 8), ('candidate_limit', 8, 128),
                ('max_query_chars', 32, 1024), ('max_history_turns', 2, 12), ('max_records', 16, 16384)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('agent_memory_policy_invalid')
        if self.max_results > self.candidate_limit:
            raise ValueError('agent_memory_policy_invalid')


def terms(text):
    """保留版本/路径等标识符的组成段；中文用二元字串，不推断别名或同一实体。"""
    result = []
    for part in re.findall(r'[A-Za-z0-9_]+|[\u4e00-\u9fff]+', text.casefold()):
        if part.isascii():
            result.append(part)
        elif len(part) == 1:
            result.append(part)
        else:
            result.extend(part[i:i+2] for i in range(len(part)-1))
    return list(dict.fromkeys(result))


def match_query(query, policy):
    if not isinstance(query, str) or not query.strip() or len(query) > policy.max_query_chars:
        raise ValueError('agent_memory_query_invalid')
    # 不把用户字符串当 FTS 运算符；词项经过字符白名单，再逐个引号包裹。
    return ' OR '.join('"' + t + '"' for t in terms(query)[:32])

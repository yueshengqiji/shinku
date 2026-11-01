"""纯成稿的有界格式恢复；不接收工具、契约或其他内部执行元数据。

只兼容完整对象的text字符串内未转义控制换行。宿主仍核验正文与完成状态；
原始供应商消息留在原生历史和输出档案，不用修复对象覆盖原文。
"""
import json


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_final_field')
        result[key] = value
    return result


def _only_text_controls(raw):
    """词法定位根对象text值；不把旧契约里的坏换行顺带修好。"""
    depth, start, escaped, text_value, previous_key = 0, None, False, False, None
    for i, char in enumerate(raw):
        if start is not None:
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
            elif char == '"':
                if depth == 1 and raw[i + 1:].lstrip().startswith(':'):
                    previous_key = json.loads(raw[start:i + 1])
                start = None
            elif ord(char) < 32 and (not text_value or char not in '\n\r\t'):
                return False
        elif char == '"':
            text_value = depth == 1 and previous_key == 'text' and raw[:i].rstrip().endswith(':')
            start = i
        elif char in '{[':
            depth += 1
        elif char in '}]':
            depth -= 1
    return True


def recover_final_text_json(content, *, discard_task_metadata=False):
    """返回候选或None；不修截断、引号、逗号、嵌套对象、重复键和调用参数。"""
    if not isinstance(content, str) or len(content.encode('utf-8')) > 256 * 1024:
        return None
    raw = content.strip()
    try:
        json.loads(raw, object_pairs_hook=_unique)
        return None  # 合法JSON沿用原解析路径，不重复声明恢复。
    except json.JSONDecodeError as exc:
        if not exc.msg.startswith('Invalid control character'):
            return None
    except (ValueError, RecursionError):
        return None
    try:
        if not _only_text_controls(raw):
            return None
        value = json.loads(raw, strict=False, object_pairs_hook=_unique)
    except (ValueError, TypeError, RecursionError):
        return None
    # 已有执行任务允许保留成稿，但忽略模型夹带的旧契约/关联字段。
    # 不能恢复工具参数、权限字段或截断对象；删除元数据不是接纳新契约。
    if discard_task_metadata and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in {'task_contract', 'task_link', 'task_notes'}}
    if (not isinstance(value, dict) or set(value) - {'kind', 'text', 'task_outcome'}
            or value.get('kind') != 'final' or not isinstance(value.get('text'), str)
            or not value['text'].strip()
            or not isinstance(value.get('task_outcome', 'partial'), str)
            or value.get('task_outcome', 'partial') not in {'completed', 'partial', 'failed', 'blocked'}
            or any(ord(c) < 32 and c not in '\n\r\t' for c in value['text'])):
        return None
    return value

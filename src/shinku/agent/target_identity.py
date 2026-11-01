"""由工具宿主解析文件身份；不根据回执后缀猜目录，不改变模型契约原文。"""
from shinku.paths import absolute_file_path, normalize_file_path


def resolve_target(target, effect, handlers, context=None):
    """相对目录路径只接受唯一宿主解析；歧义、越界及无解析器保持原值。"""
    if (not isinstance(target, str) or '://' in target or absolute_file_path(target)
            or not any(c in target for c in '/\\')):
        return target
    resolved = set()
    for handler in handlers.values():
        if effect not in getattr(handler, 'agent_completion_effects', ()):
            continue
        resolver = getattr(handler, 'agent_resolve_completion_target', None)
        if not callable(resolver):
            continue
        try:
            value = resolver(target=target, context=context or {})
        except Exception:
            continue
        if isinstance(value, str) and '://' not in value and absolute_file_path(value):
            resolved.add(normalize_file_path(value))
    return next(iter(resolved)) if len(resolved) == 1 else target

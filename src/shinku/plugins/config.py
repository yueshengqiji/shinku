"""显式插件清单读取；只解析数据，导入和资源创建由管理器的 start 执行。

未知字段作为描述元数据保留，不因额外说明或非语义版本标签拒绝插件。单条清单
错误留给对应插件诊断；文件本身无法读取或顶层不是插件列表时产生配置故障项，
宿主的普通聊天和原有能力仍能启动，不把可选插件配置问题变成全局拒绝。
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import PluginDefinition, PluginManifest


def _names(value):
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError("plugin_names_required")
    return tuple(value)


def _invalid_entry():
    raise TypeError("plugin_config_entry_invalid")


def _configuration_failure(reason):
    return (PluginDefinition(PluginManifest("plugin-configuration"), "", load_error=reason),)


def load_plugin_definitions(path: str | Path) -> tuple[PluginDefinition, ...]:
    """读取 JSON 的 plugins 数组；不读取密钥值、不扫描目录、不隐式启用其他插件。"""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return _configuration_failure(f"plugin_config_unreadable:{type(exc).__name__}")
    rows = value.get("plugins") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        return _configuration_failure("plugin_config_list_required")
    result = []
    for index, row in enumerate(rows):
        # 无效条目产生固定失败记录，后面的健康插件仍可启动。
        if not isinstance(row, dict):
            row = {"id": f"invalid-entry-{index}", "factory": None}
        try:
            manifest = PluginManifest(id=row.get("id", ""), version=row.get("version", ""),
                api_version=row.get("api_version", 1), hosts=_names(row.get("hosts")),
                depends_on=_names(row.get("depends_on")), metadata=dict(row))
            result.append(PluginDefinition(manifest, row.get("factory", ""),
                enabled=row.get("enabled", True), config=row.get("config") or {}))
        except (TypeError, ValueError):
            name = row.get("id") if isinstance(row.get("id"), str) else f"invalid-entry-{index}"
            result.append(PluginDefinition(PluginManifest(name), _invalid_entry,
                enabled=row.get("enabled", True)))
    return tuple(result)

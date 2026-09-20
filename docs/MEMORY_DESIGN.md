# Shinku 独立记忆系统

## 目标

独立版不把历史记忆固定塞进每轮提示词，也不把模型生成的摘要直接当成事实。
记忆分为原始回合、短期事件和用户明确确认的事实；只有当前回合需要时才建立
`MemoryContext`，并把来源一并交给模型。

## 当前流程

```text
QQTurn
  -> MemoryRouter
       明确回忆: 直接判定
       普通闲聊/当前任务: 不检索
       模糊请求: 轻量结构化模型判断
  -> MemoryStore 按会话作用域检索
  -> MemoryContext 带来源注入 Agent user prompt
  -> 回合写入原始层/短期事件层
```

模型路由器只回答“是否需要查、查什么”，不负责确认事实，不负责修改记忆。
这避免了把路由模型的推测变成长期人格或用户事实。

## 可调策略

检索和注入边界集中在 `shinku.memory.MemoryPolicy`。默认值保持当前行为，部署时可在
`.env` 覆盖：

- `SHINKU_MEMORY_RETRIEVAL_LIMIT`：一次最多注入多少条记录；
- `SHINKU_MEMORY_MAX_QUERY_TERMS`：查询拆分的最大词项数；
- `SHINKU_MEMORY_EVENT_SCAN_LIMIT` / `SHINKU_MEMORY_FACT_SCAN_LIMIT`：数据库候选扫描范围；
- `SHINKU_MEMORY_EVENT_TTL_SECONDS`：短期事件的保留时间；设为 `0` 表示不自动过期；
- `SHINKU_MEMORY_DETERMINISTIC_GUARDS`：是否启用明确回忆/短闲聊等确定性捷径。关闭后，
  已配置的记忆路由模型会统一判断，适合灰度比较“规则优先”和“模型优先”两种策略。

这些参数只改变检索边界，不会把长期事实自动降级，也不会让普通聊天默认注入全部历史。

## 生命周期

- 原始回合：保留在 `memory_turns`，不默认注入。
- 短期事件：默认 30 天后过期，按时间衰减排序。
- 明确事实：只有“请记住……”这类明确表达进入 `memory_facts`；默认不自动过期，
  但用户说“忘记/清除/不要记”时会标记为不可检索。
- 记忆检索：只在回忆路由通过时执行，并限制当前会话作用域。
- 遗忘日志：记录撤销动作，避免已撤销内容重新被恢复。

## 旧库迁移

```powershell
python scripts/migrate_shinku_memory.py `
  "C:\Users\as233\AppData\Local\Shinku\users_data\shinku_memory_v01\shinku_memory_v01.db" `
  "$env:LOCALAPPDATA\Shinku\data\memory\shinku_memory.sqlite3" `
  --dry-run
```

确认统计后去掉 `--dry-run`。迁移规则如下：

- 原始聊天 -> 低优先级 `legacy_raw` 事件；
- 普通摘要 -> `legacy_summary` 事件；
- 语义摘要 -> 低优先级 `legacy_semantic` 事件；其中的 stable facts 只保存在来源元数据，
  不会直接升级成长期事实；
- Lessons -> `legacy_lesson` 事件；
- 临时状态 -> 带原过期时间的 `legacy_state` 事件；
- Chroma 向量索引不直接复制，清洗后再重新建立。

源数据库始终以只读方式打开。迁移是幂等的，重复运行不会重复插入相同来源。

迁移后可以运行只读灰测：

```powershell
python scripts/check_shinku_memory.py `
  "$env:LOCALAPPDATA\Shinku\standalone-gray\data\memory\shinku_memory.sqlite3"
```

它应当同时满足：普通聊天不检索、明确回忆触发检索、检索结果带来源。

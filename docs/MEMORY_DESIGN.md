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
- `SHINKU_MEMORY_MASTER_USER_IDS`：私人版主人 QQ 号列表；启用后会建立独立的
  `private:master` 主人记忆层，只有主人本人发起、且真红实际参与的对话才会复制进去；
- `SHINKU_MEMORY_DETERMINISTIC_GUARDS`：是否启用明确回忆/短闲聊等确定性捷径。关闭后，
  已配置的记忆路由模型会统一判断，适合灰度比较“规则优先”和“模型优先”两种策略。
- `SHINKU_MEMORY_SEMANTIC_ENABLED`：是否启用私人版本地语义索引；公共版默认关闭；
- `SHINKU_MEMORY_SEMANTIC_PROVIDER`：默认 `huggingface`，使用本地 `BAAI/bge-m3`；
- `SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY`：运行期是否禁止模型联网加载；私人版建议保持开启；
- `SHINKU_MEMORY_SEMANTIC_BACKFILL_BATCH_SIZE` / `SHINKU_MEMORY_SEMANTIC_INDEX_INTERVAL_SECONDS`：后台回填批量和间隔；
- `SHINKU_MEMORY_SEMANTIC_MIN_SCORE`：语义候选的最低余弦相似度。

这些参数只改变检索边界，不会把长期事实自动降级，也不会让普通聊天默认注入全部历史。

## 生命周期

- 原始回合：保留在 `memory_turns`，不默认注入。
- 短期事件：默认 30 天后过期，按时间衰减排序。
- 明确事实：只有“请记住……”这类明确表达进入 `memory_facts`；默认不自动过期，
  但用户说“忘记/清除/不要记”时会标记为不可检索。
- 记忆检索：只在回忆路由通过时执行，并限制当前会话作用域。
- 主人记忆：`private:master` 是跨私聊/群聊的逻辑独立库。群聊旁观记录仍留在原群作用域；
  只有主人本人请求回忆时才会把主人库加入 `retrieve`/`summary` 检索，其他成员永远不能读取。
- 主人库过滤：主人参与但真红没有介入的群聊不会进入主人库；其他成员之间的旁观聊天也不会进入。
- 遗忘日志：记录撤销动作，避免已撤销内容重新被恢复。

## 私人版语义检索

私人版现在可以在同一个 SQLite 数据库中维护 `memory_embeddings` 表。SQLite 仍然是唯一
事实源；向量只是可重建的检索副本，不单独引入 Chroma 或常驻向量数据库服务。

- 公共适配器默认保留 `BAAI/bge-m3`，私人版当前选择已缓存的
  `BAAI/bge-small-zh-v1.5`：中文 QQ 记忆足够、CPU 负担小，后续可通过配置切换回 BGE-M3；
  两者都通过可选的 `sentence-transformers` 加载，聊天内容不会发给 embedding 服务；
- 事件和事实先入库，后台线程按模型集合键和内容哈希批量建立向量，避免被动 QQ 入库等待模型；
- 普通聊天仍不检索；进入 `retrieve` 时才合并关键词候选和语义候选；精确日期查询继续走
  时间窗与词项重合保护，避免语义相近的旧消息越过日期边界；
- 混合结果保留 `retrieval_method`、`semantic_similarity` 和 `lexical_score`，调试中可以看出
  一条记忆为什么被选中；模型加载或向量计算失败时自动退回原关键词检索。

首次准备私人版模型（不应在后端启动阶段自动下载）：

```powershell
& C:\Users\as233\Desktop\ai_life\_c1_cleanenv\Scripts\python.exe -m pip install -r requirements-ml.txt
& C:\Users\as233\Desktop\ai_life\_c1_cleanenv\Scripts\python.exe scripts\warm_shinku_embedding_model.py `
  --model BAAI/bge-small-zh-v1.5 `
  --cache-dir private\data\memory\models\bge-small-zh-v1.5
```

模型缓存成功后，私人版配置保持 `SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`，重启后
由后台索引器逐批处理已有记忆。若机器没有可用的 ML 依赖，状态会明确显示为
`fallback_hashed`，这只是可复现的粗排，不应当当作语义模型命中。

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
- 旧版 Chroma 向量索引不直接复制；独立私人版在清洗后的 SQLite 记录上重新建立
  `memory_embeddings`，模型版本变化时按新的集合键重建。

源数据库始终以只读方式打开。迁移是幂等的，重复运行不会重复插入相同来源。

迁移后可以运行只读灰测：

```powershell
python scripts/check_shinku_memory.py `
  "$env:LOCALAPPDATA\Shinku\standalone-gray\data\memory\shinku_memory.sqlite3"
```

它应当同时满足：普通聊天不检索、明确回忆触发检索、检索结果带来源。

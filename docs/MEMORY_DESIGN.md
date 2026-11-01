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
- `SHINKU_MEMORY_AGENT_TASK_DETAIL_ENABLED`：是否保存主人 Agent 任务的有界过程档案；
- `SHINKU_MEMORY_AGENT_TASK_DETAIL_TTL_SECONDS` / `SHINKU_MEMORY_AGENT_TASK_DETAIL_MAX_RECORDS`：
  过程档案默认保留 30 天、最多保留 500 条；超过期限或容量的旧详情会停止检索，摘要卡片不受影响；
- `SHINKU_MEMORY_AGENT_TASK_DETAIL_MAX_STEPS`：每个任务最多保留多少个工具步骤，默认 32；
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
- Agent 任务记忆：主人任务摘要长期保留；过程详情仅保存工具顺序、工具名、执行状态和轮数，不保存原始参数、原始工具输出或模型思维过程；详情只对主人开放，并按 TTL/容量清理。显式遗忘/重置同样作用于主人库。
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

## 分层生命周期（2026-10-01 源码能力，非线上启用声明）

实现入口为 `memory/lifecycle.py`、`lifecycle_policy.py`、`lifecycle_review.py`。
`memory_lifecycle` 是 SQLite 旁表；事件/事实仍是唯一内容来源，检索访问时间
与有效强化证据分离。完整参数、风险映射、实测报告与上线门槛见
实施与验收报告（历史/本机资料，未随此快照提供）。

- 新受管事件 L1 保存原文；有效重提/已送达回答的实际采用进入 L2。
- 时间跨度、UTC 不同日、每日上限与请求去重共同控制长期晋级；默认
  6 次、至少 4 天、跨度至少 14 天才提出 L3，不是同一轮连说 6 次。
- L3/明确保护不自然过期；仍可主动撤销、更正，事实有效期不被晋级取消。
- 等级和原文/摘要形态独立。冷却先降级，摘要只逐句抽取并校验数字、
  否定/条件片段；校验失败保留原文，不自动用模型修补。摘要回收后标记
  `raw_detail_unavailable`，再次晋级不能恢复已经移除的细节。
- `shadow` 默认保持原等级、原文和旧 TTL，仅记录证据与方案；
  `active` 接管受管事件的生命周期，普通扫描仍不删除索引原文。
  回收还必须另给精确作用域和独立确认，不能只打开 active 就清旧库。
- 自然改写/实质采用核验为独立 opt-in，使用现有 AUX 服务，来源、用户
  和最终答案片段必须逐字绑定。仅检索、未送达答案、助手复述、第三方
  引用、改口或无法确认作者的旧资料不累计。语义判断不提高事实可信度。
- 同源的历史回合仍由原有审计保留策略管理；只回收记忆索引文本/向量
  不会清除聊天记录或备份，也不等于 SQLite 文件缩小或磁盘安全擦除。

所有字段可用 `.env.example` 中的 `SHINKU_MEMORY_LIFECYCLE_*` 配置。
保留期是有效时间常数，不是保证在第 7/30 个自然日删除；空闲时间按
0.08 加权，有交互后的一小时按全速累计，并要求至少 12 次后续交互。
阈值修改开启新迁移宽限期，但不抹掉既有记忆年龄或冒充重复使用。

本地管理接口（只允许本机，写请求需同源 JSON）：

| 路径 | 边界 |
| --- | --- |
| `GET /memory/lifecycle?scope=private:...` | 元数据、模式、计数、预算/错误；不返回原文 |
| `POST /memory/lifecycle/preview` | `{scope}`；事务回滚预览，无等级/年龄持久变更 |
| `POST /memory/lifecycle/scan` | `{scope,confirm:true}`；不回收 |
| 同上，授权回收 | 再需 `allow_reclaim:true`、`approved_scope` 等于 scope、`reclamation_confirmation:"RECLAIM_MEMORY_INDEX_RAW"` 和 active |
| `POST /memory/lifecycle/protect` | 精确 `scope,record_id,pinned,confirm:true`；active 才能保护/解除，拒绝永久保护凭据 |
| `POST /memory/lifecycle/correct-fact` | `scope,record_id,expected_hash,replacement,confirm:true`；仅当前事实，内容哈希 CAS，旧版撤销，历史原话不改写 |
| `POST /memory/lifecycle/revoke-record` | `scope,record_id,confirm:true`；原文已回收、不能按旧词搜索时仍可明确撤销；不涉及日志删除 |

恢复旧备份前需应用独立导出的撤销清单。辅助脚本 `gray_memory_lifecycle.py`
支持只读基线、隔离回放及带确认的撤销导入；未经作用域确认不修改恢复库。
`gray_memory_models.py --bge` 使用已缓存 CPU 模型、禁止下载；
`--live-review` 单独使用现有模型核验虚构语料，不发送真实 QQ 记录。

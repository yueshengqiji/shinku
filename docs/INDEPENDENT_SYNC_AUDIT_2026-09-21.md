# 独立版与历史运行版同步盘点

这是一份内部核验记录，不是公开项目的功能承诺。核对对象是本机退役的历史参考树与当前独立树 `src/shinku`。

判断标准不是“文件名是否相同”，而是历史运行版曾经提供的行为是否在独立版仍有等价入口。独立版已经重写的模块不按文件名强行搬运。

## 本轮已经补齐

### QQ 戳一戳事件（P0，已完成）

历史参考树在 `qq_gateway.py` 和 `routes/qq.py` 中专门处理了 poke；独立版原先的 `NapCatEventAdapter` 只接受 `post_type=message`，所以所有 notice/notify 事件在进入路由前就返回 400。

当前独立版已完成：

- 支持 `notice/poke`、`notice/group_poke` 和 `notify/notify/poke` 形状；
- 归一化发送者、群组、目标机器人和事件 ID；
- 只在 `target_id` 是当前机器人时进入 `qq_poke` 路由；
- 戳一戳与普通文字互不合并，避免把轻互动混入同一轮；
- 不把 poke 写入旁观记忆、不触发历史检索、不要求工具调用；
- 仍然拒绝未支持的 notice（例如 `group_upload`）。

对应实现：

- `src/shinku/qq/message.py`
- `src/shinku/qq/napcat.py`
- `src/shinku/qq/routing.py`
- `src/shinku/qq/attention.py`
- `src/shinku/qq/agent_bridge.py`

### P1 入站与 Agent 稳定性（已完成本轮实现，待运行实例重启后灰测）

- 引用消息：`IncomingMessage` 保留 reply id；`NapCatHost` 通过显式 `get_msg` 回取引用正文，引用图片继续进入视觉输入，不再退化成纯文本 `[图片]`。
- 事件准入：`IngressPolicy` 在路由前执行重复指纹、过期、自消息和忽略群/用户过滤；poke 事件不因旧 notice 时间误伤。
- 群聊交接：`GroupContextTracker` 为群聊保留带 TTL/字符上限的本地话题快照，并给 Agent 注入发送者标签与引用块。
- 自身状态：`read_self_state` 作为低风险只读工具注册到 Agent；只返回模型、配置摘要、工具清单和人设摘要，不返回密钥或系统提示词原文。
- 后台任务：`AgentTaskCoordinator` 提供任务记录、协作式取消、失败重试和状态查询；QQ `/agent cancel|retry` 与本机 `/agent-work/tasks/*` 共用同一协调器。

对应实现和回归：

- `src/shinku/qq/ingress_guard.py`
- `src/shinku/qq/group_context.py`
- `src/shinku/tasks/coordinator.py`
- `src/shinku/agent/self_state.py`
- `src/shinku/api/agent_work.py`
- `tests/test_p1_ingress_and_context.py`
- `tests/test_p1_self_state_and_tasks.py`

### P2 管理与诊断入口（第一批已完成，待运行实例重启后灰测）

- 存活与指标：`/health/live`、`/metrics`、`/metrics/recent`；指标面只允许本机访问，不包含密钥。
- QQ 诊断：`/api/qq/napcat/status`、`/api/qq/self-check`；只检查本地装配、准入守卫、引用回取和调度器，不主动访问 NapCat。
- 记忆管理：`/memory/stats`、`/memory/timeline`、`/memory/facts`、`/memory/rollup`、`/memory/maintenance` 和带显式确认的 `/memory/forget`。
- 能力与授权查询：`/capabilities`、`/capabilities/{tool_name}`、`/capabilities/approvals/{conversation_id}`；只读展示工具元数据和待授权请求，不增加执行权限。

对应实现和回归：

- `src/shinku/api/observability.py`
- `src/shinku/api/qq_diagnostics.py`
- `src/shinku/api/memory_admin.py`
- `src/shinku/api/capabilities.py`
- `tests/test_p2_management_api.py`

### P2 外围行为与生命周期（第二批已完成，默认关闭）

- 入群事件：`group_increase` 现在可以被独立适配器识别，但不会进入 Agent、旁观记忆或话题快照。
- 入群欢迎：`GroupWelcomePolicy` 只使用固定模板；必须同时打开 `SHINKU_QQ_GROUP_WELCOME_ENABLED=true`
  和 `SHINKU_QQ_AGENT_SEND_ENABLED=true`，并可用群号白名单与冷却时间限制范围。默认不发消息。
- 独立监督：`shinku supervise --service backend` 只启动并管理自己创建的后端子进程，以 `/health/live`
  做存活检查，重启次数有上限；不会扫描、接管或按名称终止其他进程。

对应实现和回归：

- `src/shinku/qq/welcome.py`
- `src/shinku/supervisor.py`
- `tests/test_p2_welcome_supervisor.py`

## 确认存在、但还没有同步的行为

这些不是“独立化后必须把旧文件搬过来”，而是当前独立版确实还没有等价行为。

| 优先级 | 缺口 | 历史参考实现 | 独立版现状 | 影响 |
|---|---|---|---|---|
| P1 | 引用消息展开与引用图片/文件回取 | 历史参考版的 quoted context、`get_msg` 回取逻辑 | 已由 `NapCatHost` 的显式引用回取和视觉桥接覆盖 | 仍需在真实 NapCat 上灰测不同版本的 `get_msg` 返回形状 |
| P1 | 入站事件去重、过期过滤、自消息/忽略对象过滤 | 历史参考版的 duplicate/stale/self/ignored 判断 | 已由 `IngressPolicy` 在旁观入库和回复路由前统一执行 | 过滤阈值可通过 `SHINKU_QQ_*` 环境变量调整 |
| P1 | 群聊发送者标签与话题指代 | 历史参考版的 sender label/topic handoff | 已由 `GroupContextTracker` 和 Agent transcript 注入覆盖 | 仍需真实群聊灰测多人并发话题 |
| P1 | Agent 可读的自身状态/运行配置工具 | 历史参考版的只读状态工具 | 已注册 `read_self_state`，敏感字段只返回脱敏摘要 | 工具仍受 `allowed_tool_names` 约束 |
| P1 | 后台任务协调、取消、重试和任务记录 | 历史参考版的任务协调与 HTTP 路由 | 已由 `AgentTaskCoordinator`、`/agent-work/tasks/*` 和 QQ 指令覆盖；记录镜像到本地 SQLite，最多保留最近 500 条 | 重启后未完成任务标记失败、不自动重放；handler 不落盘，需重新发起任务 |
| P2 | 入群欢迎 | `qq_group_welcome.py`、`routes/qq.py` 的 `group_increase` 分支 | 已由独立 `GroupWelcomePolicy` 覆盖；默认关闭且不经过模型/记忆 | 仍需在真实 NapCat 上以白名单群做一次受控灰测 |
| P2 | 日期时间线/摘要的独立 HTTP 管理入口 | 历史参考版的 memory route | 已由 `/memory/timeline`、`/memory/facts`、`/memory/rollup`、`/memory/maintenance` 和显式确认的 `/memory/forget` 覆盖 | 候选 backfill 仍不开放，避免把旧管理面整段搬回 |

## 外围 HTTP 与运维入口的差异

这部分不是核心对话链缺失，而是旧运行版曾经暴露的外围接口。它们需要单独决定是否补，不能因为路径名字相同就把旧路由整段搬回独立版。

| 类别 | 历史参考树有 | 独立版现状 | 处理判断 |
|---|---|---|---|
| 观测与重置 | `/health/live`、`/metrics`、`/metrics/recent`、`/reset` | 已补齐存活与本机指标；没有通用 reset | reset 涉及破坏性状态操作，继续保持显式管理接口，不默认迁移 |
| QQ 诊断 | `/api/qq/napcat/status`、`/api/qq/self-check` | 已补只读 QQ 状态/自检适配层 | 不主动探测外部 NapCat，真实连通性由灰测确认 |
| 直接思考接口 | `/think`、`/think_once` | 请求统一经过 Agent/LLM 运行时 | 当前属于架构差异；除非需要外部调试，不建议恢复绕过 Agent 的直通接口 |
| 能力目录与审批 | `/capabilities/*`，其中混有 MCP、语音、工作流和本地环境控制 | 已补安全的只读能力清单和待授权查询 | 高危工作流、语音和本地环境操作仍不迁移 |
| 服务启动与监督 | 多个 `launch_*`、`manage_shinku_services.py`、时间/浏览器/回复宿主启动脚本 | 已提供 `shinku supervise --service backend`；只管理自己创建的后端子进程 | 仍需在真实运行环境做一次人工启停灰测，不复用旧启动器路径和进程模型 |

## 已经同步，或已被独立架构替代

| 领域 | 结论 |
|---|---|
| 记忆与上下文 | 已由 `src/shinku/memory/` 独立重写；有被动入库、四级历史模式、摘要、长期事实、遗忘和来源审计，不应再搬回旧 `context_manager.py`。 |
| Agent 主循环与人工授权 | 已由 `src/shinku/agent/`、`src/shinku/hosts/tool_host.py` 和 `src/shinku/tools/` 组成新的边界；旧 `engine.py` 不应回迁。 |
| 供应方与模型运行时 | 已由 `src/shinku/providers/`、`src/shinku/llm/` 和控制中枢 API 替代；当前模型切换以控制中枢为事实来源。 |
| QQ 消息、视觉桥接、回复断句 | 已有 `message.py`、`napcat.py`、`delivery.py`、`reply_host.py` 和 `turns.py`；其中引用回取仍是上表缺口。 |
| 固定斜杠指令 | 已由 `commands.py` 独立实现，且只在消息开头的 `/` 进入解释器；旧版散落在网关中的自然语言兼容解析不应重新混入普通聊天。 |

## 明确没有同步、且目前属于发布边界内的有意排除

下面这些在历史运行版存在，但当前独立公开树没有随意带回：

- 桌宠前端、角色资源、好感度、喂食和商店；
- 旧版 voice/persona 示例、声线守卫、游戏台词和角色素材；
- ASR/TTS 和 QQ 语音投递；
- 本地 CLI、Shell、PowerShell、文件删除、浏览器 Cookie、内网控制等高危 MCP；
- 浏览器/桌面截图宿主及其自动化执行器；
- 提醒、时间管理和主动后台宿主；
- 生成文件、音频处理、媒体工作台、贴纸包等旧版项目业务工具。
- 旧版全量 `capabilities` 工作流、直接 `/think` 旁路和破坏性 `/reset` 语义。

这些不是漏同步，而是当前独立项目的发布范围选择。若以后重新接入，应以外部宿主 + capability manifest + 风险授权的方式接入，不应复制历史 `engine.py` 或把高危能力默认注册到 Agent。

## 建议的后续顺序

1. P1 与 P2 两批代码实现已经完成，下一步只做真实独立实例的灰度回归，并核对新管理入口是否由正确进程提供。
2. P2 剩余工作变成真实环境人工验收：用白名单群验证一次欢迎语、用临时端口验证监督器的子进程重启；默认配置仍不产生主动外发。

本次盘点结论：P1 的五项代码边界和 P2 两批管理、诊断、入群事件与生命周期能力已经在独立版落地，当前剩余工作是重启真实运行实例并做受控灰度验收；没有把历史路由或启动器原样搬回。

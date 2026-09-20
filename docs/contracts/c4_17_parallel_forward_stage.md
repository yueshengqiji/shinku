# C4-17：NapCat 并行 forward 灰度准备

## 已完成

- 保留旧的 `legacy-event`：`http://127.0.0.1:9998/api/qq/napcat/event`；
- 追加独立版接收端：`shinku-standalone-gray` → `http://127.0.0.1:19998/events/napcat`；
- 修改前已备份 OneBot 配置文件；
- 独立版已用独立运行目录启动在 `19998`；
- 旧后端、NapCat 登录状态和旧事件入口没有停止或覆盖。

## 当前状态

配置已经写入磁盘，并已在 2026-09-20 重启 NapCat 使其生效。独立版后端当前以临时
`SHINKU_QQ_AGENT_SEND_ENABLED=false` 运行，因此真实事件即使进入独立版，也只会经过
入站、记忆和 Agent 处理，不会向 QQ 出站。旧的 `9998` 上报入口保持不变。

## 验收证据（2026-09-19）

| 项目 | 结果 |
| --- | --- |
| 旧 `9998/health/ready` | 通过；database、LLM、vector store、QQ 均 ready |
| 新 `19998/health/ready` | 通过 |
| NapCat `get_status` | 通过；在线且状态良好 |
| OneBot 配置 JSON | 通过解析；旧客户端仍在，新客户端已追加 |
| 旧配置备份 | 已生成 `onebot11_3599477026.json.before-shinku-gray-20260919.bak` |
| NapCat reload/restart | 已完成；日志显示两个 HTTP 上报服务均已启动 |
| 独立后端出站 | 临时关闭；`.env` 未修改 |

## 下一步门槛

1. 观察一条真实 QQ 入站事件是否同时到达独立版，并核对独立日志与记忆记录；
2. 继续保持独立版出站关闭，确认无重复回复、无旧入口串线；
3. 真实出站回归必须另行明确授权；
4. 任何异常都可恢复备份，并停止 `19998`，旧 `9998` 不受影响。

## 首条灰测结果与修复（2026-09-20）

首条真实测试消息已被 NapCat 接收，旧 `9998` 正常回复，但独立记忆库没有新增回合。
原因是 OneBot 真实事件使用 `self_id=3599477026` 表示当前机器人，而独立路由只检查
显式 `bot_ids` 或测试用 `at_bot` 标志，导致真实 `at` 被判为未定向群消息。

`src/shinku/qq/routing.py` 已增加 `self_id` 回退识别，新增路由回归覆盖；本地 webhook
端到端验证已确认带 `self_id` 的事件返回 `scheduled=true`，并成功写入独立记忆库。
因此需要再发一条真实 QQ 测试消息完成最终确认。

## 第二条真实复测结果（2026-09-20）

第二条真实消息已进入独立库，作用域为 `group:872732158`，说明 `self_id` 路由修复
已经生效。独立版当前进程的 QQ 出站仍关闭，因此 QQ 中看到的回复来自旧 `9998`，不能
作为独立版回复链路的通过证据。

本次检查终端输出时一度出现乱码，但用 Unicode 转义和码点复核后确认，独立
`memory_turns` 保存的是正确的 `灰度测试`；问题只发生在 PowerShell 展示 Python
UTF-8 输出的显示层，不是项目或 NapCat 编码问题。C5-3 的入站路由、中文入库和作用域
链路均通过。

## C5-4 受控出站灰度（2026-09-20）

随后使用一条合成事件直接 POST 到独立版 `19998/events/napcat`，临时打开独立版
出站开关验证 NapCat action server。独立库成功写入输入回合，NapCat 日志也确认
独立版消息已经实际发出；验证结束后已停止出站进程并恢复
`SHINKU_QQ_AGENT_SEND_ENABLED=false`，因此当前不会因并行 forward 产生独立版重复回复。

首轮出站的内容是 Agent 兜底句，不是传输失败。进一步诊断确认供应商忽略 JSON mode，
返回了普通文本，旧运行时把它判成 JSON 失败。已在 `runtime_core.py` 增加仅针对
Agent fallback 的纯文本恢复，并用回归覆盖；详情见 `c5_4_controlled_egress.md`。
修复后的单条灰测已通过，NapCat 收到正常回复“嗯，我在这里。”；验证后已恢复
`SHINKU_QQ_AGENT_SEND_ENABLED=false`，当前安全模式不变。

# C5-4：受控出站灰度与 Agent 纯文本恢复

## 范围

本批只验证独立版从 `19998` 接收事件、调用 Agent、通过 NapCat action server
发送消息的最小出站闭环。测试使用一条直接 POST 到独立 webhook 的合成群消息，
不经过旧 `9998`，不替换旧服务，也不把普通真实群聊切到独立版。

## 灰度结果

| 环节 | 结果 |
| --- | --- |
| `19998/events/napcat` 入站 | 通过 |
| 独立库保存输入回合 | 通过，作用域 `group:872732158` |
| NapCat action server 出站 | 通过，NapCat 日志出现独立版发出的消息 |
| 旧 `9998` 串线 | 本次合成事件未使用旧入口 |
| 出站开关恢复 | 已恢复 `SHINKU_QQ_AGENT_SEND_ENABLED=false` |

首轮出站虽然到达 NapCat，但 Agent 回复了兜底文本。诊断不是网络、密钥或
action server 失败，而是供应商在要求 JSON 时返回了普通文本 `嗯，收到。`，
旧逻辑把它当成 JSON 解析失败并继续使用 Agent fallback `暂时无法完成这一步。`。

## 修复

`src/shinku/llm/runtime_core.py` 现在只对带有 Agent `kind` 的 fallback 做窄范围
恢复：当网关返回非空普通文本，且 fallback 的 `kind` 为 `final`、`wait` 或
`waiting_user` 时，将该文本包装回 Agent 结构；记忆路由等没有 Agent `kind` 的
结构化调用仍然保持严格 JSON 失败处理。新增指标
`chat_json_plain_text_recoveries`，用于观察网关兼容性，不吞掉真正的 JSON 错误。

## 回归

使用独立 clean environment 执行：

```text
82 tests passed
compileall passed
git diff --check passed
127.0.0.1:19998/health passed
```

其中新增回归覆盖“网关忽略 JSON mode、Agent 返回普通文本”的情形。

## 修复后最终确认

2026-09-20 09:16 使用唯一标记的单条合成事件再次打开出站开关，webhook 返回
`scheduled=true`，NapCat 日志确认独立版发出正常回复“嗯，我在这里。”，没有再
出现 `暂时无法完成这一步。`。验证结束后已停止出站进程，并以
`SHINKU_QQ_AGENT_SEND_ENABLED=false` 重启 19998；当前独立版仍保持安全模式。

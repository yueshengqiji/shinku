# C4-17：NapCat 并行 forward 灰度准备

## 已完成

- 保留旧的 `akane-event`：`http://127.0.0.1:9998/api/qq/napcat/event`；
- 追加独立版接收端：`shinku-standalone-gray` → `http://127.0.0.1:19998/events/napcat`；
- 修改前已备份 OneBot 配置文件；
- 独立版已用独立运行目录启动在 `19998`；
- 旧后端、NapCat 登录状态和旧事件入口没有停止或覆盖。

## 当前状态

配置已经写入磁盘，但 NapCat 当前进程仍使用启动时加载的旧配置，因此第二个 forward
客户端还没有生效。这样可以避免在独立版 Agent/LLM 出站链路尚未装配完成时，把真实 QQ
消息送入一个只完成入站调度的服务。

## 验收证据（2026-09-19）

| 项目 | 结果 |
| --- | --- |
| 旧 `9998/health/ready` | 通过；database、LLM、vector store、QQ 均 ready |
| 新 `19998/health/ready` | 通过 |
| NapCat `get_status` | 通过；在线且状态良好 |
| OneBot 配置 JSON | 通过解析；旧客户端仍在，新客户端已追加 |
| 旧配置备份 | 已生成 `onebot11_3599477026.json.before-shinku-gray-20260919.bak` |

## 下一步门槛

1. 先在独立版补齐 Agent/LLM 的入站到出站处理器，并继续保持默认不自动发消息；
2. 用户确认窗口后重启或 reload NapCat，使第二个 forward 客户端生效；
3. 先观察独立版收到真实事件但不出站，再单独做一条明确授权的受控回复回归；
4. 任何异常都可恢复备份，并停止 `19998`，旧 `9998` 不受影响。

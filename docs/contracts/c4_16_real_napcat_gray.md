# C4-16：真实 NapCat 灰度联调记录

## 目的

验证独立 Shinku 仓库在不覆盖旧后端、不过度修改 NapCat 配置的前提下，能够：

1. 通过本仓库的 `NapCatHttpActionCaller` 调用现有 NapCat HTTP action；
2. 启动独立后端并挂载 `/events/napcat` 入站 webhook；
3. 保留 OneBot 图片 segment，并把可直接消费的 `data:image/*` 输入标记为 ready；
4. 不因这次灰测发送 QQ 消息，也不关闭或重启现有 `9998` 服务。

## 灰度边界

- 现有 `127.0.0.1:9998` 由旧项目后端占用，进程和配置保持不动；
- 独立版临时使用回环地址 `127.0.0.1:19998`，运行期目录位于系统临时目录；
- 只读 action 为 `get_login_info`；
- webhook 使用本地合成的 OneBot `private message` 事件，不向 QQ 出站；
- 本批没有改写 NapCat 的 forward webhook 配置，因此尚未声称“真实 QQ 事件已自动转发到独立版”。

## 结果（2026-09-19）

| 检查项 | 结果 |
| --- | --- |
| 独立版 `/health` | 通过，`project=shinku`，端口 `19998` |
| 独立版 `/health/ready` | 通过，运行目录可写 |
| 独立版调用 `get_login_info` | 通过，返回当前 QQ 登录信息 |
| 文本 webhook 入站 | 通过，`scheduled=true` |
| 图片 segment 保留 | 通过，`visual_ready=true`、`visual_pending=false` |
| 旧 `9998` 服务 | 未修改、未停止 |
| 独立版临时进程 | 已用 Ctrl+C 正常停止 |
| 全量回归 | `1149 passed, 1 warning, 2286 subtests passed` |
| 工作区 | clean |

## 未覆盖项

真实 NapCat forward webhook 的持久化配置、从真实 QQ 事件进入独立版后的 Agent/LLM 处理、
以及真实出站 `send_private_msg`/`send_group_msg`，必须在用户确认切换入口后另行做受控灰度。

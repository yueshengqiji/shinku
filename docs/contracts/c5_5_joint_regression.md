# C5-5：独立版联合回归与切换准备

## 目标

将 C5-2、C5-3、C5-4 的单项证据合成一条可重复的离线联合契约：真实 OneBot 事件
形状进入独立路由，经短窗口调度、记忆入库、Agent 桥接，最后到达投递边界。

## 契约

1. 没有显式 `bot_ids` 时，OneBot `self_id` 仍能识别 @ 真红。
2. 事件进入回合后，用户文本写入当前群作用域的独立记忆库。
3. 发送开关打开且提供 sender 时，只产生一个投递结果。
4. 发送开关关闭时，仍完成 Agent dry-run 和记忆入库，但 sender 不得被调用。

## 当前处理

本批新增 `tests/test_contract_c5_5.py`，覆盖真实 OneBot 字段、路由、回合调度、
记忆和投递边界。独立本机 `.env` 的 `SHINKU_QQ_AGENT_SEND_ENABLED` 已固化为
`false`；需要真实出站时必须通过明确的临时进程环境覆盖，不修改安全默认值。

联合回归及全量回归已通过：新增测试 **2 passed / 0 failed**，全量测试
**1171 passed / 0 failed**，compileall 和 `git diff --check` 通过。当前不切换旧
`9998` 主入口。

下一步处理准入自动检查和 `/health` 双轨收口。

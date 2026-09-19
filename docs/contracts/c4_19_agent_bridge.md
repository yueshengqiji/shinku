# C4-19：QQ Agent/LLM 桥接边界

## 目标

把已经合并的 `QQTurn` 接到独立的 `AgentLoop`、`ToolHost` 和 `LLMRuntime`，并把最终文本
转换成一个统一的 QQ 出站消息。供应商、主人设、工具注册表和发送器都由调用方注入。

## 契约

- QQ 文本进入 Agent transcript；图片以 `user_images` 进入 LLMRuntime，不只留下 `[图片]`；
- 工具 schema 和执行仍由 `ToolHost` 控制，AgentLoop 负责有限轮次与失败恢复；
- 模型返回完成/等待用户且有文本时才构造回复；失败、空文本或异常不伪造成功；
- 没有注入 sender 时为 dry-run，不发送 QQ；
- sender 只接受和返回统一的 `OutgoingMessage`/`DeliveryResult` 边界；
- 本批不读取旧项目配置，不加载旧 persona，也不启用真实 NapCat 出站。

## 验收

专项测试覆盖：主人设与 transcript 传递、图片输入传递、统一出站消息、dry-run 和失败不发送。

# C4-18：QQ 短窗口回合调度器

## 目标

把 NapCat 入站事件和 Agent/LLM 调用之间隔开：webhook 只接收并快速返回，调度器负责
等待注意力窗口、合并同一会话的已调度消息，并把整轮输入交给调用方注入的 handler。

## 契约

- 没有注入 handler 时，现有 `NapCatHost` 行为不变；
- handler 不在 webhook 请求线程中等待模型，timer 到期后再处理；
- 相邻同会话消息按 `AttentionBatcher` 的窗口合并；
- 每张图片按原消息 id 保留为 `NapCatVisualInput`，不折叠成 `[图片]`；
- handler 异常被记录为 `handler_error:<Type>`，不会把已接收的 webhook 改成 500；
- 本批不认识 LLM、persona、工具或出站协议，后续 Agent bridge 通过 handler 接入；
- 默认没有启用调度器，因此不会改变当前旧服务或独立灰度服务的行为。

## 验收

专项测试覆盖：两条相邻消息合并、图片按消息保留、timer 触发、handler 失败隔离。

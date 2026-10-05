# C3-4：Agent 规划-工具-结果循环

## 本批交付

`shinku.agent.loop.AgentLoop` 是一个不绑定模型供应商的 Agent 循环内核：

```text
planner(state) -> final / wait / tool
                         ↓
                 execute_invocation
                         ↓
                 envelope 回灌 state
                         ↺
```

它提供：

- 最大轮数限制，防止无界调用；
- 工具结果回灌给下一轮 planner；
- 空决定、未知工具、参数错误、handler 异常的可恢复 envelope；
- 相同工具调用的重复检测和最终阻断；
- 可选完成验收门，避免模型只说“好了”就宣告完成；
- 可选地把工具结果事件写入 C3-2 工作区服务。

## 后台跨段协议（2026-10-02）

同步调用仍以 max_rounds 为终点；ToolHost 的显式 `sliced=True` 则把它作为量子。
AgentRunResult 的 `yielded/slice_complete` 是宿主内部调度信号，不是模型可选终态，
不投递、不写长期完成记忆。可执行 AgentRunCheckpoint 保留同一 AgentState、工具回执、
ResearchLedger、ProgressLedger、调用签名/失败/重复计数、变更纪元和停滞保护。
续段仅重绑当前宿主上下文，不能用 JSON/模型摘要伪造检查点。
协调器 running->queued 是同一 attempt 的让出执行权，不是 retry；cancel 与权限检查
仍在规划/工具/终态检查点生效。任务预算位于 host ContextVar，实际 SDK 请求（含
重试/契约修复/已接入工具内部调用）共享账本；浏览器线程显式复制上下文。
审批可在同一进程绑定原检查点/预算；重启缺失检查点不执行旧审批。

## 接入边界

模型供应商只需要适配 `Planner`，工具宿主只需要实现 `ToolHandler`。本批不把
`LLMRuntime` 硬编码进循环，也不处理 QQ、浏览器、截图或后台线程。这样后续接入
真实模型时，可以单独判断慢点发生在模型调用、工具执行还是循环策略。

## 交付状态（2026-09-30，批次 A）

`final` 表示模型提出本轮结束，不再等同于业务目标成功。模型可附带
`task_outcome`：`completed`、`partial`、`failed`、`blocked` 或
`waiting_background`。默认 ToolHost 使用结构化 `CompletionVerdict` 验收，
实际终态和已满足/缺失条件保存在 `AgentRunResult.completion`，QQ 会正常发送
失败、部分完成和后台处理中汇报。

旧 bool/string/tuple 门禁仍兼容；结构化门禁和模型结果的非法状态、未知门禁
返回格式不能放行成功。宿主上下文可提供 `agent_completion_contract`，其中
`requirements` 为 `id/effect/targets` 条件列表；工具输出不能覆盖此可信契约。
当前只接入契约核验接口，模型生成契约与跨消息持久化由后续批次开发。

浏览器访问证据要求执行器签发 `verified_effects=["browser_access"]`；HTTP 403
只有导航/错误页证据，不能签发访问成功证据。2xx/3xx 且可读页面是基础访问
证据，不保证登录完成、内容正确或业务操作成功，后续还需具体目标条件验收。

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

## 接入边界

模型供应商只需要适配 `Planner`，工具宿主只需要实现 `ToolHandler`。本批不把
`LLMRuntime` 硬编码进循环，也不处理 QQ、浏览器、截图或后台线程。这样后续接入
真实模型时，可以单独判断慢点发生在模型调用、工具执行还是循环策略。

# C3-9：LLMRuntime native tool call 回灌

## 问题

`LLMRuntime.call_chat_json()` 在识别供应商原生 `tool_calls` 后，会统一输出线路字段
`_native_tool_call`。此前 `LLMPlanner` 只消费测试桩直接返回的 `tool_calls`，没有消费
runtime 的统一字段，因此真实 runtime 会被 AgentLoop 判为 `invalid_decision`。

## 修复边界

`LLMPlanner` 复用 `tools.invocation.legacy_tool_call_to_invocation()`，把
`_native_tool_call` 转成 provider-neutral 的 `ToolInvocation`，再交给已有的
`AgentDecision.tool()`、工具执行器和宿主。供应商响应解析仍留在 `LLMRuntime`，planner
不复制 OpenAI/Anthropic 的响应结构。

## 验收口径

使用真实独立仓库的 `LLMRuntime` 类和本地 fake OpenAI client：第一轮返回 OpenAI 形状的
native tool call，第二轮返回最终 JSON。fake client 不发网络请求，因此结果证明的是
runtime→planner→host→handler 的内部线路，不代表任何真实供应商、QQ 或浏览器端口已经
联调成功。

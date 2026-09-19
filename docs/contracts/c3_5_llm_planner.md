# C3-5：LLM planner 适配

## 本批交付

`shinku.agent.planner.LLMPlanner` 把现有 `LLMRuntime.call_chat_json` 适配成 C3-4
需要的 planner：

- 固定的 Agent system prompt 与可缓存的 prompt key；
- 可注入的用户 prompt 构造器；
- 结构化 `native_tools` / `native_tool_choice` 透传；
- 历史回合的有限窗口透传；
- final / wait / legacy tool_call / OpenAI native tool_calls 的统一解析。

## 重要边界

工具 schema 走 runtime 的结构化字段，不把每个 handler 的实现注释拼进普通 user prompt。
模型返回形状无法识别时返回 `None`，交由 AgentLoop 统一记为 `invalid_decision`；不在
planner 内部偷偷执行工具，也不在这里决定失败重试次数。

# 上下文预算观察与裁剪

默认 `SHINKU_LLM_CONTEXT_BUDGET_MODE=observe`。每次实际组装请求时，在
`SHINKU_LOG_ROOT/llm_context_budget/YYYY-MM-DD.jsonl` 写入元数据；如果显式设置了
`SHINKU_LOG_DIR`，则优先使用它。记录包括模型、估算的
system / 历史 / 当前请求 / 工具 schema / 图片 / 消息开销、预算和预计需移除的历史条数。
不写提示词、图片数据或工具参数。每个请求有一条 `kind=estimate`，完成时再写一条
`kind=usage`，两者用 `request_id` 对应；`channel` 区分 chat / aux，
`estimate_to_actual_ratio` 帮助对照估算。供应商未报告用量时 `usage_available=false`。
开启裁剪后，比值使用实际发出的请求估算值。`history_turns_removed` 是完整移除的
对话轮次，`history_messages_removed` 是移除的消息数；预算无法满足时原请求保持不变，
`projected_after` 仅记录尝试裁剪后的模拟大小。
`snapshot_metrics()` 仍保留原有的 input/output/cache token 汇总和预算状态计数。

估算器不是模型 tokenizer。它对非 ASCII 字符（包括日文、韩文和 emoji）采用保守估算；
图片和不同网关的隐藏包装仍会有误差，不能把它当成精确 token 账单。
未知模型窗口时记录 `window_unconfigured`，预留输出后没有正数输入预算时
记录 `input_budget_exhausted`；两种情况都不会猜一个窗口值并截断。

确认供应商的上下文窗口后，在本地配置 `SHINKU_LLM_MODEL_CONTEXT_WINDOWS_JSON`，
例如 `{"model-id":32768}`。单模型也可使用 `SHINKU_LLM_CONTEXT_WINDOW_TOKENS`。
可用输入预算 = 窗口 - `SHINKU_LLM_OUTPUT_RESERVE_TOKENS` -
`SHINKU_LLM_CONTEXT_MARGIN_TOKENS`。在 `observe` 下先看多轮真实请求的日志；
`would_trim_history` 表示如果开启执行会裁掉较早的聊天轮次，当前并未改动请求。

确认阈值后设 `SHINKU_LLM_CONTEXT_BUDGET_MODE=enforce`：仅从最旧的 user/assistant
历史轮次开始移除；工具反馈阶段会保留当前 Agent 任务目标。固定 system、当前用户请求、
图片和工具 schema 保持原样。
Agent 模式中的工具反馈只在此模式下限制单条与累计字符数，优先保留最新结果的开头和
结尾，完整工具执行记录不修改。如果仅靠移除历史仍无法满足估算预算，记录
`unresolvable_preserved`，保留原请求，不静默删掉核心内容。

上线次序：观察多轮私聊、群聊和 Agent 工具任务 → 对比估算和实际用量 → 配模型窗口
→ 小流量开启 enforce → 检查 `trimmed_history`、`unresolvable_preserved`、答复连续性
及缓存用量。`off` 可完全关闭这一层。

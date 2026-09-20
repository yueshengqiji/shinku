# C6：模型决策与语义授权

## 目标

工具是否需要使用由模型依据当前任务、上下文和已注入的 native tool schema 判断，
不再由消息关键词直接路由。关键词只能作为外围宿主的提示或兜底，不能绕过 Agent
规划器，也不能直接触发副作用。

## 执行边界

1. `LLMPlanner` 使用 `native_tool_choice=auto`。模型可以选择最终回复、等待用户，
   或提出一个具体的 `ToolInvocation` 批次；同一轮的多个 native `tool_calls` 会被
   统一归一化并按顺序执行。
2. `AgentLoop` 先对整批调用做工具名、参数和执行策略校验；通过校验后交给宿主的
   `approval_gate`。授权未完成时只返回 `waiting_user`，工具 handler 不会执行。
3. QQ 宿主按会话顺序保存待授权调用批次。下一条同一会话消息交给无工具权限的
   `LLMConsentResolver`，由模型判断其是否在语义上同意这一次具体调用。
4. 只有 `approve` 才恢复原始 `ToolInvocation` 批次；不会重新规划一次工具，避免重复或
   换错调用。`deny` 清除待处理调用，`unclear` 继续等待，不执行任何副作用。
5. 授权判断器的异常、非法返回和无法确认都按 `unclear` 处理，默认安全失败。

## 风险策略

工具授权不是只有一个写死的开关。`ExecutionPolicy` 为每个工具提供 `low`、`medium`
或 `high` 风险级别：

1. 未登记的工具默认按 `medium` 处理，新增工具不会因为漏配而自动执行；
2. 只有当策略明确把调用判定为不需要授权的风险级别时，AgentLoop 才会直接执行；
3. 默认需要授权的级别是 `medium,high`，因此 QQ 的默认行为与此前一致；
4. 同一批调用只要含有一个需要授权的调用，就整批等待，避免先执行一部分再询问用户。

QQ 装配可以通过环境变量覆盖默认风险策略：

- `SHINKU_TOOL_DEFAULT_RISK=low|medium|high`
- `SHINKU_TOOL_APPROVAL_RISKS=low,medium,high`
- `SHINKU_TOOL_RISK_OVERRIDES_JSON={"tool_name":"low"}`

这些变量只改变授权门槛，不会绕过 `allowed` / `blocked` 工具白名单，也不会改变
工具参数校验。默认配置仍然是安全失败。

## 设计取舍

- 不维护“同意词表”，因此“可以，你按刚才说的做”“行，查吧”等自然表达可以由
  模型结合上下文判断；答非所问或模糊闲聊不会被当成授权。
- 当前待授权状态默认保存在 QQ Agent 进程内存中，默认五分钟过期。它不把待执行工具
  写入长期记忆，也不在进程重启后自动恢复。需要跨重启恢复时，可显式设置
  `SHINKU_QQ_AGENT_APPROVAL_STORE_PATH` 使用独立 JSON store；同一会话的多条待授权请求
  按先入先出处理。该文件可能包含原始
  工具参数，应按敏感本地状态保护，不能放进发布包或提交到仓库。
- 直接使用 `AgentLoop`/`ToolHost` 的离线调用仍保持兼容：只有注入
  `approval_gate` 时才启用授权拦截。面向 QQ 的桥接默认启用授权。
- `SHINKU_QQ_AGENT_HISTORY_TURNS`、`SHINKU_QQ_AGENT_RECENT_CONTEXT_MESSAGES` 和
  `SHINKU_QQ_AGENT_APPROVAL_TTL_SECONDS` 把原先散落在桥接代码里的三个运行参数收口
  到配置层；未设置时保持原来的 12 / 8 / 300 秒行为。
- 内置能力表仍采用保守默认值；已经验证的新 OpenAI 兼容网关可以用
  `SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST=host:model`，临时试探未知模型则必须显式打开
  `SHINKU_NATIVE_TOOL_ALLOW_UNKNOWN_MODEL=true`。

## 验收

- `tests/test_contract_c6_1.py` 覆盖：首次拦截、精确恢复、批量调用、语义同意、拒绝/不明确不
  执行，以及 QQ 两轮桥接。
- 全量回归必须通过，工具调用 schema 仍然以 native tools 传给模型。

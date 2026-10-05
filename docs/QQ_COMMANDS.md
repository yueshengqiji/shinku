# QQ 斜杠指令

私聊消息在第一个字符就是 ASCII `/` 时进入指令解释器。群聊消息必须同时满足
“消息正文以 `/` 开头，并且消息明确 @ 真红”；普通群聊里单独发送 `/stream`、
`/memory` 等内容不会修改状态，也不会进入指令解释器。前导空格也不会触发指令分支。

指令在 Agent、记忆和主人设之前处理，不会调用模型，也不会写入聊天记忆。群聊
里没有 @ 真红的斜杠消息仍可按普通群消息继续走路由（是否回复由群聊注意力策略决定）。
未知的斜杠指令只返回帮助，不会落到普通聊天模型。

## 开关与运行控制

```text
/help
/status
/stream on|off|status
/at reply on|off|status [群号]
/at reply default [群号]
/onlooker on|off|status
/stop
/start
/agent on|off|status
/agent tasks
/agent status task-id
/agent permission status|traditional|readonly|confirm|full|ultra
/agent cancel task-id
/agent retry task-id
```

`/agent permission` 只允许主人在私聊，或在明确 @ 真红的群消息中使用；五档设置保存在
`command_state.json`，重启后继续生效。默认是 `traditional`，沿用原有风险分级；`readonly`
只开放明确声明为只读的工具；`confirm` 对每次工具调用都要求主人逐次授权。`full` 自动放开
已注册的常规能力，并允许文件工具访问整台电脑的路径、自动执行高影响操作；`ultra` 额外开放
任意本机 PowerShell 和被配置关闭的私用浏览器能力，并解除 Shinku 执行策略的工具过滤。

`full` / `ultra` 不会提升 Windows 账户权限、绕过 UAC，也不会凭空安装或注册不存在的工具；
它们只能使用当前后端进程已注册且操作系统账户本身可访问的能力。控制中枢启用高权限档位时
会要求再次确认；两种入口都会写入私用审计日志。可随时执行 `/agent permission traditional`
收回。

高权限档位不跳过任务条件和完成核验。模型误把删除等副作用写成 `tool:工具名`、或生成
可恢复的内部格式错误时，宿主会在任何工具执行前尝试一次无工具元数据修复；已选调用、
参数、权限、旧目标和旧条件不会因此被改写。修复仍不合法就停止，不会要求重复授权或
把查找成功算成删除成功。尚未发现路径时可使用准确文件名/名称的 `target_selector`；
只有与该筛选条件匹配的真实操作回执、当前结果核验通过，才可报告完成。明确的完整路径
仍须精确覆盖；筛选条件不是新增操作权限。

启用 `SHINKU_QQ_AGENT_ASYNC_ENABLED=true` 后，模型选定的通用 Agent 工具链在后台
队列运行；普通聊天和轻量只读感知保持现有路径，首次聊天/工具选择请求仍在前台。
默认一个后台工具工作者、最多 8 项排队/活动任务（含当前任务，可用
`SHINKU_QQ_AGENT_ASYNC_QUEUE_LIMIT` 设置）。不发送固定开始气泡或逐步进度；完成、失败
或需要授权时才回复。任务号保存在移交 API 结果；QQ 使用 `/agent tasks` 查看当前会话
的任务号，再用 `/agent status task-id` 查询真实状态、轮数/调用数和交付状态。
群里仍须明确 @ 真红；QQ 查询/控制按请求者与私聊/群聊会话隔离。

`/agent cancel` 会发出协作式取消信号；正在执行的任务需要在下一个宿主检查点结束，
状态仍为运行中并注明“已请求取消”，不能当作瞬间停止或撤销在途操作。关闭 Agent、
暂停回复或切换权限会撤销旧后台任务；快速关闭再开启也不会复活旧执行。
通用 QQ Agent 的旧闭包禁止 `/agent retry` 盲目重放；请重新说明继续目标，模型根据
原任务证据重新规划。只有明确允许安全重试的其他本机任务 handler 保留原 retry 行为。任务记录
也可从本机控制面查询：`GET /agent-work/tasks`、`GET /agent-work/tasks/{task-id}`，
以及对应的 `/cancel` 和 `/retry` POST 端点。
正式 CLI 启动时，任务记录默认保存在数据目录下的 `runtime/agent_tasks.sqlite3`
（可用 `SHINKU_QQ_AGENT_TASKS_DB_PATH` 覆盖），最多保留最近 500 条终态记录，
活动和待授权/阻塞记录不会因较旧而被历史上限挤掉。重启不会自动
重放未完成任务：它们会被标成失败并注明中断原因；执行函数不落盘，因此这类记录不能
跨重启直接 retry，需要重新发起任务。

后台任务现在以现有 `SHINKU_QQ_AGENT_MAX_ROUNDS`（默认 6）作为单段轮数，
未结束时在队尾自动续跑，同一任务号/尝试和完整内存执行状态保持，不需反复说“继续”。
整项任务默认累计上限：120 次 SDK 请求、600000 Token 记账额度、128 次工具调用、
1800 活动秒，分别使用 `SHINKU_QQ_AGENT_TASK_MAX_MODEL_REQUESTS/MAX_TOKENS/MAX_TOOLS/MAX_ACTIVE_SECONDS`
配置。路由、元数据修复、HTTP/协议重试及工具内部模型调用都计入；
无 usage/流式请求保守预留输入 UTF-8 字节数与最大输出（缺省限制 32768），不是精确账单。
排队/等待审批不计活动时间；在途请求不能抢断，可能略超时后停。
预算耗尽保留证据并阻塞，不自动重试；审批和同任务语义承接不刷新旧预算。
可执行检查点不落盘；重启后即使仍有旧审批也不能直接重放，需用户重新说明目标。
取消、停滞、防重复与原验收继续有效，不引入子代理；
长工具可使用上下文 `agent_task_interrupt()` 在内部检查点协作停止，任意 Shell/HTTP
不保证能在返回前抢断。后台部分完成/阻塞/等待授权与成功分别记录，执行成功也不
等于最终回复发送成功。

`/at reply` 在群里不带群号时只修改当前群；附群号时修改指定群。私聊不带群号时修改
全局默认，已单独设置的群不受影响。`status` 可查看当前群、指定群或全局默认；`default`
可清除某个群的单独覆盖，使其重新跟随全局默认。群级覆盖保存在 `command_state.json`，
重启后继续生效。关闭自主回复后仍然响应明确的 @ 或直接呼叫。

`/onlooker` 只影响发送该指令的群；关闭后仍然响应明确的 @ 或直接呼叫，
只是停止该群的被动观察入库。`/stop` 会暂停普通回复，但 `/start` 仍可进入，
以便恢复服务。

## 兼容入口

```text
/reply mode text|voice|both|auto
/model
/model list
/model model-id
/provider
/provider list
/provider provider-id
/character
/memory recent|retrieve|summary [query]
/timeline [query]
/mface config
```

模型、供应方、角色和 mface 指令会被正确识别并阻止它们进入 LLM。模型和供应方的
实际保存、模型探测与激活现在由本机控制中枢完成；QQ 命令层仍只返回识别结果，
不会直接伪造热切换。角色热切换和语音投递仍未接入独立 QQ 宿主。

## 权限与状态

私用版以 `SHINKU_MEMORY_MASTER_USER_IDS` 作为指令权限的唯一主人列表；旧的
`SHINKU_QQ_COMMAND_USER_IDS` 仅在没有主人列表时作为兼容回退。私用版没有配置
主人 ID 时会拒绝所有指令，不会默认放行。状态默认保存在
`SHINKU_DATA_ROOT/qq/command_state.json`，也可以用
`SHINKU_QQ_COMMAND_STATE_PATH` 指定路径。

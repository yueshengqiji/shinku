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
/at reply on|off|status
/onlooker on|off|status
/stop
/start
/agent on|off|status
/agent cancel task-id
/agent retry task-id
```

`/agent cancel` 会发出协作式取消信号；正在执行的任务需要在下一个宿主检查点结束，
不会被强行杀线程。失败或取消的任务可以用 `/agent retry task-id` 重新排队。任务记录
也可从本机控制面查询：`GET /agent-work/tasks`、`GET /agent-work/tasks/{task-id}`，
以及对应的 `/cancel` 和 `/retry` POST 端点。
正式 CLI 启动时，任务记录默认保存在数据目录下的 `runtime/agent_tasks.sqlite3`
（可用 `SHINKU_QQ_AGENT_TASKS_DB_PATH` 覆盖），最多保留最近 500 条。重启不会自动
重放未完成任务：它们会被标成失败并注明中断原因；执行函数不落盘，因此这类记录不能
跨重启直接 retry，需要重新发起任务。

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

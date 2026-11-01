# 真红自主委托独立 Angelos

真红主对话模型选择现有 `agent_task_submit/status/result/cancel/resume/supplement` 接口；宿主不根据请求文字、任务类别、来源数或步骤数改变委托决定。动态能力是主对话输入的数据，不是第二个意图裁判，也不是聊天可以直接调用的执行工具。

## 显式接线

服务端默认不开放 `/angelos/host/v1`。可信工厂可以传入 `ServiceHost(..., delegation_principals=...)`，也可以在原 Angelos 命令追加 `--delegation-hosts /absolute/path/hosts.json`。只监听回环地址。配置与令牌文件必须属于当前用户且权限为 `0600`；不进入公共 Git。

配置结构如下，示例不含真实身份或凭据：

```json
{
  "schema_version": 1,
  "hosts": [{
    "id": "shinku",
    "token_file": "shinku-host.token",
    "requesters": ["example-owner"],
    "sessions": ["private:", "desktop:"],
    "allowed_tools": ["example_installed_tool"],
    "budget_limits": {"max_requests": 12, "max_tokens": 600000, "max_tools": 8, "max_active_seconds": 300}
  }]
}
```

`allowed_tools` 必须是实际已安装工具，不自动导入、安装或增加能力。文件附件仍受服务端现有 `TaskMaterialStore.approved_roots` 及逐次来源检查约束，连接配置不能扩大文件根目录。令牌至少32字符，调用者使用同一专用令牌；不得使用网页bootstrap令牌或模型密钥代替。

调用端由可信启动器传入：

```python
from shinku.qq.delegation import DelegationConnection
from shinku.qq.host_bindings import QQAgentHostBindings

bindings = QQAgentHostBindings(
    chat_runtime=existing_shinku_chat_runtime,
    execution_profile=None,
    expression_profile=existing_gemini_expression_profile,
    delegation=DelegationConnection(
        "http://127.0.0.1:8770", "shinku", dedicated_host_secret,
    ),
    # 原有审核过的只读能力、资料根目录等宿主配置继续显式传入。
)
```

将该绑定交给原有 `agent_bindings` 上下文工厂；启动器拥有连接凭据与模型客户端生命周期；普通公共绑定应在宿主等待工作结束后关闭 `bridge.delegation.close()`，或显式传入启动器管理的会话连接池。私用模型客户端工厂同样支持显式 `delegation` 参数；启用时不创建本地执行客户端，不在断连时降级到本地执行。

## 状态、权限和资料

AgentService继续拥有执行任务、材料库、累计预算和真实回执。真红本地 `agent_remote_references` 仅保存请求与远端任务的绑定，不复制执行权、不连接Agent记忆或共享Bot记忆检索。原始请求和宿主选中的上下文/附件传给远端，模型提供的goal不能替换原文或增加授权。

来源宿主与服务端权限/工具/预算取交集。Angelos的ZAYIN/HE/WAW/ALEPH继续生效；需要批准的实际参数在Angelos原审批面板确认，模型没有批准接口。调用端按短周期续租，表达/送达期间也续租；撤权或取消等待会撤回该任务授权。连接丢失到租约到期期间可能仍有已授权调用正在运行；到期后禁止后续步骤，不能宣称已经回滚副作用。

提交、取消、恢复和补料都先按原消息持久登记；消息重放或响应未知不会触发第二次执行。

同一原消息仅创建一个远端任务；提交响应未知保留确定任务号，不自动创建新任务。状态/结果查询不重新执行或重新润色；补料与续接使旧结果过期时明确标为上一执行版本，不把旧表达当作新结果。新补充附件继续保留在原任务材料库。结果查询重新校验完整材料；表达缓存缺失时用完整事实模板保留原文和条件，不只返回短报告，也不新增模型调用。恢复必须显式调用原任务且通过原累计预算、版本和当前授权检查；重启不自动重放没有检查点的动作。

## Gemini结果表达

真红侧复用ExpressionArchive/ExpressionService，委托链显式使用 `format=2 / detail=2 / body=5 / semantic=1`。规范结果和完整资料投影在远端当前作用域/来源版本下恢复和验证；长原文由宿主完整插入，关键值和逐句覆盖保留。表达模型只组织结果，不执行或改变验收状态。失败使用现有完整事实回退；权限撤销、来源改变或结果过期时不发布旧正文。

源码与离线测试通过不代表正式接线已加载，也不代表真实Gemini/DeepSeek模型已完成意图、口吻、延迟与费用验收。生产启动、模型调用和真实消息发送按明确授权分别执行。


## 任务会话与分层交付（SA02）

宿主协议增加 `messages(task_id, after)`、`handoff(task_id)`、`details(task_id, result_hash)` 和 `revise`。
消息类型由宿主生成，包含任务号、轮次、输入版本、来源、原文、幂等键及内容摘要；
用户原文中的标签不产生控制权。原任务SQLite保存消息与待应用补充，不连接两边的记忆检索。
`messages`以游标分页，每页最多50条，读取仍核验连接身份、请求者和会话。
进度记录来自宿主观察到的任务状态，不保证收录每个未被轮询观察到的中间步骤。

`handoff`复用Angelos最终成稿，不新增摘要模型：完整答复正文、实质事实、限制、失败/未知回执和产物位置保留；
成功机器回执、预算和重复审计信息折叠到完整结果。需要完整分析时不把正文压成固定字数摘要。
Gemini仍使用原有表达与独立覆盖检查；本次没有移除已有检查调用。
`agent_task_result`可选 `view=full`，在当前身份和结果版本下读取完整执行报告及证据。
旧 `result`接口保留兼容；旧版本结果可保留为历史证据，不能重新发布为当前答案。

中途补充使用原消息键，先持久保存，再请求旧执行停止。宿主后续状态查询推进已登记的交接；
worker仍活动时不修改执行输入；停止后重验授权并应用，始终不自动继续执行。
暂时失联时补充留在收件箱，下次获准访问可继续交接，不后台自行启动。
完成后的修订将任务标为部分完成并保留旧结果绑定；需要续跑时仍走原resume、累计预算和副作用恢复检查。
一次只允许一个待应用补充，避免多份尚未应用的输入互相覆盖；同键不同内容拒绝，同键重放不重复应用。

只读成稿的格式恢复允许text内未转义换行，旧task_contract/task_link/task_notes字段丢弃，
原供应商响应继续保存在原生历史。截断、坏引号、重复键、工具参数、权限字段和元数据中的坏控制字符不恢复；
正文候选不能创建契约、扩大权限或改变独立执行验收。表达失败回退保留执行报告正文，
执行完成、正文是否可用、表达回退与实际送达分别保留其状态。

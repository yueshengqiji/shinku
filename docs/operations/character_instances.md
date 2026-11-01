# 共用代码、独立角色实例

入口是 `python -m shinku.instance`，复用 `QQAgentBridge → MemoryService / AgentExecutor`
的公共实现。默认真红入口不变；显式实例必须提供独立资料、账号和目录。
`src/aozora` 仍保留为早期原型，不是新入口的运行依赖。

## 配置与状态的归属

| 内容 | 来源与所有者 |
| --- | --- |
| 名字、别名、父母等关系 | `character.json`，按宿主发送者 ID 精确绑定 |
| 说话风格与角色背景 | 单独的人设文件；显式角色不加载旧伴侣提示 |
| 执行权限 | 既有 owner gate、命令状态、批准与工具策略，亲属关系不授予权限 |
| 记忆、审批、任务、会话、回执、指标、模型注册表 | 该实例专属 `state_root` |
| 工具 | 显式 `extension_module` 注册；不自动加载已有私用扩展 |
| 模型客户端 | 实例子进程持有聊天/执行两个客户端；退出时关闭 |

历史正文和作者保持原始来源，不对消息、记忆或人设做全局姓名替换。
人物关系不会从模型自称、引用、昵称或记忆摘要中自动更新。

角色文件使用严格版本 1 JSON，例如：

```json
{
  "version": 1,
  "id": "example-child",
  "name": "小星",
  "aliases": ["小星", "星星"],
  "relationships": {
    "local:owner": {"kind": "father", "address": "爸爸"}
  }
}
```

没有账号绑定的家庭成员可以保留在人设背景中，不能据此认定聊天中的某个发送者就是她。
关系影响称呼与相处方式，不能证明两人已经说过话或共享某段经历。

实例声明示例（文件路径相对这份声明）：

```json
{
  "version": 1,
  "character_file": "character.json",
  "persona_file": "persona.md",
  "state_root": "state",
  "settings_file": "settings.json",
  "owner_id": "local:owner",
  "bot_id": "local:example-child",
  "channel": "terminal",
  "backend_port": 18088
}
```

可选 `extension_module` 和 `module_paths` 必须显式填写。扩展代码是受信任的本机代码，
需要单独审查其目录、网络权限和后台副作用；这个入口不是 OS 沙箱。
默认只注册公共记忆查询和运行状态工具，不能将“已接入执行器”说成“已开放全部工具”。

`settings.json` 在 Linux 上须为 0600，严格只允许模型、模型调用预算参数和 QQ 连接参数。
聊天通道目前要求 OpenAI 兼容协议；聊天与辅助通道都需显式提供 `SHINKU_CHAT_*` / `SHINKU_AUX_*` 的 `BASE_URL`、
`MODEL_NAME`、`API_PROTOCOL`，非 Ollama 还需 `API_KEY`。还必须提供
`SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST`，值为聊天端点的 `host:port:model`（无端口时
为 `host:model`），仅声明当前这一个模型，不允许通配符。该字段是宿主的协议能力声明，
不是本轮真实模型工具调用验收；供应商是否正确执行仍需独立实测。
启动时会核对真实客户端的能力配置，失败不降级成没有执行能力的入口。
不会继承 shell 中的真红模型、
`.env.private`、旧注册表或 Python 扩展路径。调用超时和输出限制可设置
`SHINKU_LLM_CHAT_TIMEOUT_SECONDS`、`SHINKU_LLM_AUX_TIMEOUT_SECONDS`、
`SHINKU_LLM_DEFAULT_MAX_TOKENS`。网络/流式重试固定为一次尝试。

## 检查与本地终端

在仓库根，使用该项目 Python 环境：

```sh
PYTHONPATH=src ../.venv-linux/bin/python -B -m shinku.instance check /absolute/path/instance.json
PYTHONPATH=src ../.venv-linux/bin/python -B -m shinku.instance terminal /absolute/path/instance.json
```

`check` 只读取配置与目录归属，不导入工具扩展、不装配客户端、不创建数据库、不调用模型。
它验证配置，不代表模型可用或人格验收通过。`terminal` 会使用配置的模型，可能产生费用；
启动后每行是一个输入，`/exit` 退出。首次运行才创建独立状态根和归属标记。
终端成功写入并 flush 后才记录 `local_terminal` 回执；这不代表用户已阅读。

状态根不能嵌套到另一实例中，不能含符号链接/硬链接，也不能包含输入配置。
既有非空、未绑定目录拒绝接管；更改角色 ID、owner、bot 或渠道也拒绝接管旧目录。
同一实例通过 Linux 文件锁只允许一个入口进程，锁覆盖子进程生命周期。
修改人设文本或关系绑定后，下次启动读取新快照；没有自动搬运旧原型数据库的步骤。

## QQ 接入准备

QQ 使用单独的实例声明和空状态根，`channel` 为 `qq`，owner/bot 必须是明确的 QQ 账号。
关系中的 ID 同时改为真实宿主账号，不能沿用本地 `local:owner` 冒充 QQ 身份。
`settings.json` 还必须提供 `SHINKU_NAPCAT_BASE_URL`、`SHINKU_NAPCAT_ACCESS_TOKEN`、
`SHINKU_NAPCAT_EVENT_TOKEN`、`SHINKU_SERVICE_TOKEN`，并指定独立后端端口。
NapCat 登录、账号归属与 webhook 线路需要单独接入验证，声明有效不等于 QQ 可送达。

```sh
PYTHONPATH=src ../.venv-linux/bin/python -B -m shinku.instance serve /absolute/path/qq-instance.json
# 明确启用 QQ 发送：
PYTHONPATH=src ../.venv-linux/bin/python -B -m shinku.instance serve /absolute/path/qq-instance.json --send
```

两个实例复用执行代码，但各自装配执行器和任务库。尚不支持将两个角色直接接到同一个在线
Angelos 任务空间；那需要扩展远端角色作用域。语音、主动问候、浏览器、下载和订阅也不会
因为新角色存在而自动启用。正式服务启动、真实模型验证和 QQ 发送是分别验收的步骤。

## 控制台与连接状态

`serve` 复用本机 `/debug/hub` 控制台；端口来自实例声明。页面名称、聊天对象、模型、
记忆与任务读取当前后端。网页聊天通过 `/api/pet/chat` 使用宿主绑定的 owner 身份，
写入 `desktop_http` 本机送达回执；点击网页发送不会额外向 QQ 发送消息。
显式实例尚未配置桌宠资产时，桌宠入口显示未配置，不读取或控制默认角色的桌宠。

`/api/qq/self-check` 只检查本地装配。`/api/qq/connection` 另外调用 NapCat 的
`get_status` 和 `get_login_info`，核对在线状态和当前 bot 账号，单实例缓存十秒；
不请求模型或发送消息。页面的消息接入/发送开关是装配状态，收到事件数是当前去重窗口
的记录数，均不证明某条消息成功回复。真实回复需核对回合 trace 和对应渠道送达回执。

## 显式聊天只读能力

执行器注册工具与聊天层直接读取是两个入口。受信扩展可以实现
`build_chat_read_capabilities(*, config)`，返回以工具名为键、`ChatReadCapability`
为值的映射。独立启动器将它交给 `QQAgentHostBindings` 做类型和键名校验；未实现时
返回空映射，不从 handler 上的只读标记自动推导授权。

声明中的工具仍须真实注册，并通过宿主账号门禁、当前工具集合、审批策略和只读适配器
审核。限定参数 schema、单轮次数、单次耗时和结果大小同样生效。文件写入等动作不能
通过这个声明变成聊天只读调用，继续走已有正式任务及授权流程。
传统模式的空工具 allowlist 沿用原有“全部当前注册工具”含义，在桥接处展开为明确集合；
非主人不会因空集合获得权限，显式非空集合继续限定工具范围。

# 青空独立事件内核

这是独立 `aozora` 包的本地使用说明。包位于同一 Git 仓库便于快照，但不导入
`shinku`，不使用真红的 persona 装配、聊天路由、记忆服务或运行配置。仅依赖 Python
3.11+ 标准库。正式人设由用户编写；仓库只含匿名合成夹具。

## 先看离线演示

在仓库根目录运行，`--directory` 必须是尚不存在的新目录：

```sh
PYTHONPATH=src ../.venv-linux/bin/python -B -m aozora demo --directory tmp/aozora-demo
```

演示接纳一条转述、生成一个关注事项，重新打开数据库，再接纳合成观察并修正理解。
目录中生成 `identity.example.json`、`state.sqlite3` 和 `report.json`。回复保持 prepared，
没有发送到 QQ。演示决策是明确写死的两步夹具，不调用模型，也不证明模型自主判断或口吻。
可以在任何仅有标准 Python 的目录复制 `src/aozora` 后运行同一入口。

## 状态归属与阅读顺序

1. `contracts.py` 定义用户身份、受信宿主事件和模型决策的不同权限。
2. `store.py` 持有独立 SQLite，保存不可改写事件、尝试、版本化理解/关注和待发回复。
3. `kernel.py` 运行事件循环：接纳并占用 -> 当前状态投影 -> 一次决策 -> 原子提交。
4. `decider.py` 将显式配置的模型响应转成严格 Decision；不控制数据库或发送端。
5. `__main__.py` 是本地受信入口；`demo.py` 是不联网的合成场景。
6. `terminal.py` 将显式私用配置装配为固定身份的日常终端入口，复用内核和终端回执。

| 数据 | 写入者 | 含义 |
| --- | --- | --- |
| identity | 用户通过显式文件初始化 | 固定身份资料；模型不能更换，重新打开库时逐项核对 |
| message / reported | 已校验身份和作用域的宿主 | 某人说过这段话；不证明内容属实，不变成亲历 |
| observation / host_observation | 已核实来源的宿主 | 宿主提供的观察；接入通道须先验证来源 |
| tick / host_tick | 宿主显式触发 | 不是自行启动的后台定时器 |
| understandings / tentative | 模型建议、宿主校验后提交 | 有原始事件来源的可修正理解；不等于事实认定 |
| attention | 模型建议、宿主校验后提交 | open / completed / dropped；结束关注不等于完成工具任务 |
| outbox | 内核与受信发送端 | prepared / unknown / delivered / failed；模型无权签发送达 |

每个数据库只属于一个 identity；scope 负责会话资料可见范围。第一版不跨 scope 共享
记忆，即使同一人出现在不同会话也不能自动读到另一会话的经历。身份资料对本个体所有
scope 可见，应避免把会话隐私塞进 identity。模型可见最近 8 条事件、最多各 16 项有效
关注和理解，以及它们依赖的旧原文；完整投影上限 256 KiB，超限明确拒绝。
历史事件、旧版本决策和失败尝试仍留库，不默默丢弃。暂不提供自动归档、向量检索或遗忘。

证据 ID 必须来自当前可见同作用域事件；更新需匹配 version。任一更新失败会回滚整个
决策，不留下部分理解、关注或回复。来源检查只保证引用和归属，不是自然语言语义核验器；
真实模型仍可能产生错误推断或不合适的表达，需另作真实模型验收。

## 身份和事件文件

以下只是结构示例，不是正式人设。真实文件放私用目录，不提交公共 Git：

```json
{
  "agent_id": "your-stable-agent-id",
  "name": "由用户填写",
  "foundations": [],
  "relationships": {}
}
```

`foundations` 是用户写的稳定原则和基础资料；`relationships` 将宿主核实的 actor ID
映射到关系说明。更改身份后直接打开旧库会得到 `identity_mismatch`。首版不提供隐式身份
迁移，不应绕过错误去覆盖旧库；正式身份确定前使用演示库即可。

宿主事件结构：

```json
{
  "event_id": "source-message-unique-id",
  "scope": "local-private-conversation",
  "actor": "host-verified-user-id",
  "kind": "message",
  "content": "本轮输入内容",
  "occurred_at": 1792684800
}
```

时间为来源提供的 Unix 秒值。相同 event_id/相同内容重放原结果，改变任何字段会拒绝。
不得把外部消息中的 JSON 原样当成受信 Event；后续 QQ/网络入口必须从已鉴权的通道
生成 actor、scope、kind。现在的 CLI 仅供本地操作者使用，不是网络鉴权层。

## 显式模型接入与本地聊天

需要支持 Chat Completions 的完整 URL、模型名和 JSON object 输出。没有默认供应商、
自动模型发现或自动回退。下面命令是运行方式，不表示本轮已经调用或验收供应商：

```sh
PYTHONPATH=src ../.venv-linux/bin/python -B -m aozora chat \
  --identity private/aozora/identity.json \
  --db private/aozora/state.sqlite3 \
  --scope local-private-conversation --actor local-user \
  --endpoint https://YOUR-HOST/v1/chat/completions \
  --model YOUR-MODEL --api-key-env AOZORA_API_KEY
```

密钥仅从显式指定的环境变量读取，省略 `--api-key-env` 表示不带认证头；适合本地匿名服务。
HTTP 仅允许 loopback，远程要求 HTTPS，禁止重定向转发凭据。`--timeout` 默认 30 秒、
范围 (0,120]，是 socket 阻塞 I/O 超时，不保证整轮严格墙钟上限。
`--max-tokens` 默认 2048、范围 128–8192；输入和响应字节另有上限。
第一版不提供费用账本或供应商 usage 核验，不将尝试次数或 max_tokens 当作实际计费值。

提供商即使收到 JSON object 请求，也可能将完整决策包进一个 Markdown JSON 代码块。
适配器接受原始 JSON，或整段恰好为一个标记 `json` 的三反引号代码块；只去掉外围包装，
不搜索混合文字中的 JSON、不补括号或字段。重复字段、NaN/Infinity、代码块外的说明、
多段输出、截断与工具调用仍拒绝；解析后继续执行原有身份、证据、作用域和版本校验。
错误码区分 `provider_envelope_invalid`（外层 JSON/choices 结构）、
`provider_content_invalid`（正文缺失或类型错误）和 `provider_decision_json_invalid`
（决策正文不是完整、无歧义的 JSON）。日常入口不把响应原文写入日志。

本地 chat 每行输入触发一次调用，成功回复写到终端并 flush 后才记录 local-terminal 回执。
Ctrl-D 结束。生成失败停止本地聊天，不编造兜底回复，不自动再请求。

单轮处理可使用 `turn --identity ... --db ... --endpoint ... --model ... --event ...`，
结果输出 JSON，回复仅 prepared；`deliver --identity ... --db ... --scope ... --delivery-id ...`
才会实际打印本地终端并记回执。它不发送 QQ。`inspect --identity ... --db ... --scope ...`
只查看当前投影，不调用模型；首次初始化显式指定的新库仍会写入身份。

## 配置式日常终端

固定配置可避免每次输入 actor、scope、模型和数据库路径。所有文件路径以配置文件所在
目录为准；入口不读取真红的环境变量或模型目录，也不自动选择备用提供商。
以下仅为匿名结构示例；实际配置、凭据和关系说明保留在私用目录：

```json
{
  "version": 1,
  "identity": "identity.json",
  "database": "terminal.sqlite3",
  "actor": "local:operator",
  "scope": "local:conversation",
  "model": {
    "endpoint": "https://YOUR-HOST/v1/chat/completions",
    "name": "YOUR-MODEL",
    "api_key_file": "model.key",
    "timeout": 120,
    "max_tokens": 4096
  }
}
```

identity.relationships 必须包含配置的 actor；关系由本机操作者确认，不能从输入正文
推断。这个绑定只适用于受信本机终端，不是 QQ 账号认证，也不能证明操作电脑的是谁。
单独的 UTF-8 密钥文件仅含密钥一行，POSIX 权限须为 `0600` 或更严格；匿名服务需明确
设 `api_key_file: null`。缺失或不安全的凭据直接拒绝，不回退到环境中的真红凭据。

```sh
PYTHONPATH=src ../.venv-linux/bin/python -B -m aozora.terminal \
  --config private/aozora/terminal.json check
PYTHONPATH=src ../.venv-linux/bin/python -B -m aozora.terminal \
  --config private/aozora/terminal.json chat
```

`check` 只读配置、身份和凭据，校验请求参数；输出不含凭据和基础人设。它不打开或创建
数据库、不探测端口、不调用模型，因此 `configuration_valid` 仅代表配置可解析，
已有数据库的身份匹配要等 `chat` 打开时验证，服务可用性也未由检查证明。

`chat` 会创建或打开独立数据库，使用固定 actor/scope 为每行创建 `message` 事件。
每行只调用一次配置模型，空行忽略，Ctrl-D 退出；生成失败立即停止并给出失败事件 ID，
不继续消费后续输入。终端输出仍经过持久发送占用、实际 flush 和回执。关闭后再次用同一
配置启动，会恢复该库中的事件、关注和理解；不要为了改人设删库或静默换成新库。
TTY 下的姓名提示和“本轮没有生成回复”是宿主界面提示，不记成角色发言。

## 失败、并发与恢复

- 一库只允许一个运行中决策。模型调用在事务外，其他事件得到 decision_busy；同事件
  重复请求读取 running/终态，不发生第二次调用。
- 普通失败持久保存 failed 和稳定错误码，事件原文保留，关注/理解/回复不部分写入。
  供应商错误正文不写日志或抛给用户，以免泄露凭据；不保留原始模型响应。
- 进程硬中断后 running 不会自行过期。先用 `inspect` 和
  `status --identity ... --db ... --scope ... --event-id ...` 核对，
  再显式 `cancel --identity ... --db ... --scope ... --event-id ...` 撤销提交资格。
  取消不保证中止已经发出的 HTTP，也不撤回费用。
- 重试同一事件必须显式给 `turn --retry`。旧尝试号迟到不能写入新状态；旧中断也不能
  取消后继尝试。重试会重新投影当前状态，可能再次计费；没有无限自动恢复。
- 发送前先持久 claim，状态改为 unknown，再输出/flush。输出后写回执前崩溃也保留
  unknown，禁止自动重发。只有受信宿主查证后才可调用 `resolve_delivery` 确认结果。
  回执证明指定通道完成该动作，不证明人类已经阅读。
- 如果模型思考期间旧回复的送达状态改变，拒绝沿旧上下文提交，保留失败供核对。

## 验证范围

合成回归覆盖状态延续、转述/观察区分、可修正理解、历史来源回取、跨 scope 隔离、
身份绑定、原子提交、容量、并发防重、取消与迟到、失败显式重试、终端送达中断、
实际 loopback HTTP 请求，以及独立复制包在禁止 shinku 导入和禁止网络下的演示。

```sh
../.venv-linux/bin/python -B scripts/run_desktop_batch.py -- \
  ../.venv-linux/bin/python -B -m pytest \
  tests/test_aozora_kernel.py tests/test_aozora_http.py tests/test_aozora_cli.py \
  tests/test_aozora_terminal.py
```

QQ 与真红之间的消息适配、外部动作执行、主动活动调度、正式人设和真实模型效果均未由
这些离线检查验收。阶段证据和源码快照见 `docs/plans/AOZORA_KERNEL_2026-10-23.md`。

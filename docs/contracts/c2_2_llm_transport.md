# C2-2 行为契约：LLM 传输客户端

本文件是 **C2-2 洁净室重写**的验收依据。实现可以换写法，**本文件写下的外部可观察行为不能换**。

---

## 0. 判据与口径

### 0.1 契约事实 vs 表达层（沿用来源记录 §5.9「路径 A」）

| 层 | 内容 | 归属 |
| --- | --- | --- |
| **契约事实** | 公开函数名与签名、参数名与默认值、类名、实例公开属性名、请求头名与取值、端点拼装规则、请求体字段名与取值语义、返回结构的键名、停止原因映射表、异常类型 | 必须逐字保留，属接口信息 |
| **表达层** | docstring、注释、内部组织与文件切分、私有命名风格、实现步骤与算法写法 | 必须 Shinku 独立撰写 |

**推论：** 对照物（上游 `code_shared/llm_client.py`）与 Shinku 旧侧
（`services/llm_client.py`）里的**私有名**属表达层，新实现**不得复用**。
本批登记 **32 个旧私有名**（二者并集，见 `_kit/batches/c2_2.py` 的 `LEGACY_NAMES`），
新实现每一处都换名——`_combine`→`_prepare_request`、`_raise_http_error`→`_reject_bad_status`、
`_finish_reason`→`_canonical_finish_reason`、`_chunk`→`_delta_chunk` 等。
其中 **17 个来自上游**（`_build_anthropic_payload`／`_convert_messages`／`_map_finish_reason` …），
**15 个为旧侧特有**（`_anthropic_request`／`_translate_messages`／`_system_blocks` …），
另有 `_client`／`_model`／`_response` 三个**两侧共有的实例成员名**，本批整组改掉
（新实现用 `_owner`／`_label`／`_reply`）。

### 0.2 字符串字面量不在「散文筛子」口径内

协议头名（`x-api-key`／`anthropic-version`／`content-type`）、版本值 `2023-06-01`、
错误消息模板 `Anthropic request failed: HTTP {status_code}`、SSE 事件名
（`message_start`／`content_block_start`／`content_block_delta`／`message_delta`）、
停止原因字面量、`chatcmpl-` 前缀、`data:image/...;base64,` 正则，全是**契约事实**
（外部可观察 + 线路协议），必须逐字保留。散文筛子只覆盖 docstring 与注释（SKILL 规矩十二）。

### 0.3 本批的模块与来源

| 新落点 | 对照物（上游真实现） | 对照物字节 | 行为基准 |
| --- | --- | ---: | --- |
| `llm/client.py` | `code_shared/code_shared/llm_client.py` | 13,660 | 对照物本身（上游） |

**对照物判定（SKILL 规矩六）：** legacy 侧 `services/llm_client.py` 只有 **839 B**，
主体是 `from code_shared.llm_client import *` 再加一个 `_legacy_protocol` 标记 ——
**转发壳，不能当对照物**。Shinku 旧侧 `services/llm_client.py`（13,167 B）是独立改写
（body 相似度 0.3115），**只进 `LEGACY_NAMES` 并集，不作来源**。

**为什么 C2-2 是 `llm_client`：** 上游 `model_service.py` 第 16 行在模块级
`from .llm_client import build_llm_client`，而 `build_llm_client` 的 anthropic 分支
直接返回整个 `AnthropicCompatClient`（388 行里约 365 行）。切片≈整份文件，
按「传递依赖随行」口径（C2-1 先例），传输层必须先落。

### 0.4 与 Shinku 旧侧的三处有意分叉（各配一条理由）

本批**以对照物（上游）为准**，旧侧在三处偏离上游，均**不继承**：

| # | 旧侧行为 | 上游行为（本批采用） | 理由 |
| --- | --- | --- | --- |
| 1 | anthropic 请求超时**硬编码 60.0**（`options.get("timeout", 60.0)`），`client.timeout` 对 anthropic 请求实际失效 | 超时回落到 `client.timeout`（`options.get("timeout", client.timeout) or client.timeout`） | 上游是行为基准；旧侧那处让 `AnthropicCompatClient.timeout` 成了死字段 |
| 2 | `build_llm_client` 在**两条分支上都**挂 `client._shinku_protocol = selected` | 只挂规范属性 `protocol`（anthropic 分支由 `__init__` 设置，openai 分支 `setattr`） | `_shinku_protocol` 是消费侧兼容读取（旧仓 `llm_runtime.py` 读 9 次），不是与上游的契约；本仓运行时同批重写，统一读 `protocol` |
| 3 | `__all__` 里重导出 `normalize_api_protocol` / `normalize_base_url`，并保留私有垫片 `_build_anthropic_payload` 供旧测试 import | `__all__` 只列本模块真正的两个公开名；两个归一化函数由 `shinku.providers` 提供，不在此重复导出 | 旧侧的重导出与垫片都是**旧侧便利**，不是契约；新侧测试走自己的缝 |

> 决定 2、3 属「兼容门面与旧命名」，C7（`清理兼容门面与旧命名`）会再确认一遍是否已无残留。
> 若后续重写 `llm_runtime` 时发现确有外部消费者需要一个别名，则**改在运行时侧**适配，
> 不回头给本模块加回私有标记。

---

## 1. `llm/client.py`

### 1.1 公开面

```python
def build_llm_client(*, api_key: str, base_url: str,
                     timeout: float = 60.0, max_retries: int = 0,
                     protocol: str = "auto")

class AnthropicCompatClient:
    def __init__(self, *, api_key: str, base_url: str,
                 timeout: float = 60.0, max_retries: int = 0)

__all__ = ["AnthropicCompatClient", "build_llm_client"]
```

`AnthropicCompatClient` 的**公开属性**（构造时归一化后就地存放）：
`api_key`、`base_url`、`timeout`、`max_retries`、`protocol`（恒为字符串 `"anthropic"`）、
`chat`。`chat.completions.create(**options)` 是唯一实现的调用入口。

### 1.2 `build_llm_client` 的协议判定与两条分支

1. `resolved = normalize_api_protocol(protocol=protocol, base_url=base_url)`
   —— 显式协议优先，其次按 `base_url` 猜（anthropic / ollama 信号），兜底 `"openai"`。
2. **`resolved == "anthropic"`** ⇒ 返回 `AnthropicCompatClient(
   api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)`。
   （`protocol` 由该类 `__init__` 设为 `"anthropic"`；`base_url` **原样**传入，不归一化。）
3. **其余协议** ⇒
   `OpenAI(api_key=..., base_url=normalize_base_url(protocol=resolved, base_url=base_url),
   timeout=timeout, max_retries=max_retries)`，再 `setattr(client, "protocol", resolved)`。
   - `api_key` 归一化：`str(api_key or "").strip()`；为空时用占位串 ——
     `resolved == "ollama"` 用 `"ollama"`，否则 `"not-configured"`。
     （SDK 构造期就校验密钥，占位串让它先立起来；真鉴权失败留给第一次请求报。）
   - `setattr` 包在 `try/except` 里：SDK 客户端是否允许加属性由它决定，加不上也不让构造失败。

### 1.3 `AnthropicCompatClient` 构造归一化

| 参数 | 规则 |
| --- | --- |
| `api_key` | `str(api_key or "").strip()` |
| `base_url` | `str(base_url or "").strip()` |
| `timeout` | `float(timeout or 60.0)`（`None`／`0` 都落回 60.0） |
| `max_retries` | `int(max_retries or 0)` |

### 1.4 端点拼装（`_messages_endpoint`）

先 `str(base_url or "").strip().rstrip("/")`，再：

| 归一后的 base | 结果 |
| --- | --- |
| 以 `/v1/messages` 结尾 | 原样 |
| 以 `/v1` 结尾 | 追加 `/messages` |
| 其他 | 追加 `/v1/messages` |

### 1.5 请求头（逐字保留）

```
content-type:      application/json
x-api-key:         <client.api_key>
anthropic-version: 2023-06-01
```

### 1.6 请求体改写规则（OpenAI 形状的 `**options` → Messages 请求体）

| options 字段 | 规则 |
| --- | --- |
| `model` | 原样放入；缺省 `""` |
| `messages` | 见下方「消息改写」 |
| `max_tokens` | `int(max_tokens or max_completion_tokens or 1024)`（**默认 1024**） |
| `system_extra_blocks` | 逐个 `str(x or "").strip()`，去掉空串；与 `messages` 里抽出的系统文本**合并**成 `system` 块列表 |
| `system`（请求体字段） | 仅在至少一块非空时出现；**前 4 块**带 `cache_control: {"type": "ephemeral"}`，之后的不带 |
| `temperature` | 仅在**键存在且非 None** 时写入，并夹到 `[0.0, 1.0]`（闭区间） |
| `top_p` | 仅在非 None 时写入，`float()` 原样 |
| `stop` | 字符串且去空白后非空 ⇒ `stop_sequences: [去空白后的串]`；列表 ⇒ 取各元素 `str().strip()` 后的非空项，**非空才写入** ；其他类型忽略 |
| `stream` | 仅在**键存在且非 None** 时写入，`bool()` 化 |
| `extra_body` | 字典时逐项写入，**已存在的键不覆盖**（相当于 `setdefault`） |
| `timeout` | 只影响本次 HTTP 超时，不进请求体；取值见 §1.9 |

**消息改写（`messages` 列表）：**

- 非字典项跳过。
- `role` 归一：`str(item.get("role", "user") or "user").strip().lower()`。
- `role ∈ {system, developer}` ⇒ 抽成**系统文本**（不留在 messages 里）：
  内容压平见下，非空才收；多段以 `"\n"` 连接，最后整体 `strip()`。
- `role ∉ {user, assistant}` ⇒ 强改为 `"user"`。
- 每条转成 `{"role": <归一 role>, "content": <块或纯文本>}`。

**内容压平（`_as_plain_text`）：** 字符串 ⇒ `strip()`；列表 ⇒ 收其中
`type == "text"` 的项，各取 `text` 去空白、滤掉空串、以 `"\n"` 连接再 `strip()`；
其他 ⇒ `str(content or "").strip()`。

**内容转块（`_as_content_blocks`）：** 字符串 ⇒ **原样返回**（不切块）；
非列表 ⇒ 退化为纯文本；列表 ⇒ 逐项：`type == "text"` ⇒ `{"type": "text", "text": str(item["text"])}`
（**不再 `strip`**）；`type == "image_url"` ⇒ 转 image 块（见下）；
**收到过块才返回块列表**，一个都没有则退化为纯文本。

**图片块（`_as_image_source`）：** 取 `image_url`（字典取 `url` 字段，否则 `str(value)`）
去空白；空 ⇒ `None`（整块丢弃）。匹配 `^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$`
（`re.IGNORECASE`）⇒ `{"type":"image","source":{"type":"base64","media_type":<g1>,"data":<g2>}}`；
否则 ⇒ `{"type":"image","source":{"type":"url","url":<url>}}`。

### 1.7 非流式响应的回装

`_as_chat_completion(body, requested_model)` 返回 `SimpleNamespace`：

```python
id        = body.get("id", f"chatcmpl-{uuid4().hex}")
object    = "chat.completion"
created   = 0
model     = body.get("model") or requested_model
choices   = [SimpleNamespace(index=0,
                             finish_reason=<映射>,
                             message=SimpleNamespace(role="assistant",
                                                     content=<拼接正文>,
                                                     tool_calls=[]))]
usage     = SimpleNamespace(prompt_tokens, completion_tokens, total_tokens,
                            cache_read_input_tokens, cache_creation_input_tokens)
```

- 正文：把 `body["content"]` 里所有 `type == "text"` 项以 `""` **直接拼接**后 `strip()`。
- `prompt_tokens = int(usage.get("input_tokens", 0) or 0)`，
  `completion_tokens = int(usage.get("output_tokens", 0) or 0)`，
  `total_tokens` 为二者之和（**不是**读服务端返回的 total）。
- 两个缓存计数：`int(usage.get(<字段>, 0) or 0)`，字段名 `cache_read_input_tokens` /
  `cache_creation_input_tokens`。

**停止原因映射（逐字保留）：**

| Anthropic `stop_reason` | `finish_reason` |
| --- | --- |
| `end_turn` | `stop` |
| `max_tokens` | `length` |
| `stop_sequence` | `stop` |
| `tool_use` | `tool_calls` |
| 其他 / 空 | `None` |

### 1.8 流式：SSE 事件翻译

`create(stream=True)` 返回一个可迭代对象，迭代产出的块形状为
`SimpleNamespace(id=f"chatcmpl-{uuid4().hex}", object="chat.completion.chunk",
created=0, model=<model>, choices=[SimpleNamespace(index=0,
delta=SimpleNamespace(content=<text>), finish_reason=<...>)])`。

解析规则：

| 输入行 | 处理 |
| --- | --- |
| `None` 行 | 跳过 |
| 字节行 | `decode("utf-8", errors="replace")` 后 `strip()`；字符串行直接 `str().strip()` |
| 空行 | **事件边界**：先按已攒下的 `event` / `data` 出块，再清空 `event`、`data` |
| `event:` 前缀 | `event = 冒号后去空白` |
| `data:` 前缀 | 把冒号后去空白的内容**追加**进 `data` 列表（可多行） |
| 其他行 | **忽略**（不出块、不清状态） |

`data` 以 `"\n"` 连接并 `strip()`；空串或 `"[DONE]"` ⇒ 不出块。
否则 `json.loads` 后按事件名分发（逐字保留事件名）：

| 事件名 | 出块条件 |
| --- | --- |
| `content_block_start` | 取 `payload.content_block.text`，非空才出块 |
| `content_block_delta` | 取 `payload.delta.text`，非空才出块 |
| `message_delta` | **总是**出块（正文为空串），`finish_reason` 取 `payload.delta.stop_reason` 的映射 |
| 其他 | 不出块 |

**两处收尾行为：**

- 遍历结束后若 `data` 还有残留（流尾没有空行收尾），**再翻一个块**出去。
- 无论正常结束还是中途异常，**`finally` 里调用 `response.close()`**。
- 遇到 `event == "message_start"` 的空行边界时，先尝试从该事件抓 usage：
  `payload.message.usage` 的 `cache_read_input_tokens` / `cache_creation_input_tokens`
  （各 `int(x or 0)`）存到迭代器的 `usage` 属性；**取不到就静默跳过**（不影响正文）。

### 1.9 失败模式

| 情形 | 结果 |
| --- | --- |
| HTTP 非 2xx（`response.ok` 为假） | 抛 `RuntimeError`。消息取 `body["error"]["message"]`（字典）或 `str(body["error"])`（非字典）；取不到则退回 `response.text.strip()`；仍为空则 `f"Anthropic request failed: HTTP {response.status_code}"` |
| 响应体不是 JSON | 走上面的退回路径（`except Exception` 分支），用 `response.text` |
| 本次超时 | `float(options.get("timeout", client.timeout) or client.timeout)` —— 传了用传的，传空或没传都用 `client.timeout` |
| 上游连接异常 | 由 `requests` 原样抛出，本模块不捕获 |
| `build_llm_client` 里 `setattr` 失败 | 静默忽略（`protocol` 只是给调用层看的） |

**本模块不做的事：** 不读环境变量、不读配置文件、不写盘、不重试
（`max_retries` 只是转交给 SDK）、不记录日志。

---

## 2. 本批的验收口径

洁净室重写的判据是「有契约文档 + 独立测试 + 实现独立」，**不看相似度**。六项证据：

1. 本文件（只记录外部可观察行为，是本批唯一的需求输入）。
2. 独立测试 `tests/test_contract_c2_2.py`（传输层两侧都替换：假 `requests.post` 与假 `openai.OpenAI`）。
3. **实现独立**：旧侧 32 个私有名（「对照物 ∪ Shinku 旧实现」并集）在新文件里**零命中**，
   用 `tokenize` 取**精确标识符 token** 比对；`kit.py namegap` 报覆盖完整、0 缺口。
4. **散文独立**：docstring + 注释、连续 ≥12 字符的逐字片段 **0 处**。
5. **行为等价性对拍 `_c2_2_parity.py`**：对照物（上游）vs 新实现，替换两侧传输层后逐项比对
   端点 / 头 / 请求体 / 非流式回装 / SSE 分块序列 / 异常消息。
6. **变异测试**：注入典型缺陷，全部被抓住、0 漏，恢复后逐文件 sha256 与初始值一致。

## 3. 本批的两个决定

见 §0.4 的决定 2 与决定 3：**不携带 `_shinku_protocol` 兼容标记**、
**不保留 `_build_anthropic_payload` 私有垫片与两个归一化函数的重导出**。
两处都是旧侧便利而非上游契约；C7 复查时要确认没有残留消费者。

# C1-2 行为契约：共享基础模块（洁净室重写）

**制定日期：** 2026-09-18
**处理类型：** 洁净室重写（来源记录 §5.9 路径 A 的第二类）
**依据：** `SOURCE_PROVENANCE.md` §5.9「C1 批次划分」；旧项目 `docs/shinku_independent_c1_2_request_context_preflight_20260918.md`

---

## 0. 本契约的性质

C1-2 的四个模块都**含逻辑、分支或数据处理**，因此按路径 A 属**洁净室重写**：

> 依行为契约独立实现。契约事实可迁移，**表达层（docstring、注释、内部组织与文件切分、
> 内部命名风格、实现步骤与算法写法）必须 Shinku 单独撰写**。

本文件**只记录外部可观察行为**，作为新实现的唯一需求输入。它记录三件事：

1. **必须成立的行为**——新实现与旧行为等价的部分；
2. **本仓库自有的决定**——文件切分、命名、依赖注入方式等表达层选择；
3. **刻意不做的行为**——明确排除的部分。

**关于旧实现的说明：** 本契约的行为面来自对旧模块的读取与既有行为测试。
表达层（本仓库的模块划分、命名、实现步骤）全部是本次决定，不复用旧的组织方式。

---

## 1. 来源模块 → 本仓库模块映射

| 旧模块 | 字节 | 本仓库路径 | 类型 |
| --- | --: | --- | --- |
| `companion_v01/request_context.py` | 2,667 | `src/shinku/api/correlation.py` | 洁净室重写 |
| `companion_v01/routes/sessions.py` | 6,785 | `src/shinku/api/sessions.py` | 洁净室重写 |
| `companion_v01/task_artifacts.py` | 5,563 | `src/shinku/tasks/artifacts.py` | 洁净室重写 |
| `companion_v01/tool_invocation.py` | 3,107 | `src/shinku/tools/invocation.py` | 洁净室重写 |
| — | — | `src/shinku/tasks/__init__.py`、`src/shinku/tools/__init__.py` | 包标记 |

合计来源字节 **18,122**（与计划登记值一致）。

### 1.1 文件切分决定（表达层，Shinku 自有）

- **不沿用旧模块名。** 旧名 `request_context` / `task_artifacts` / `tool_invocation` 是 Akane 时代的
  扁平命名；本仓库按**领域**组织：HTTP 面进 `api/`，任务产物进 `tasks/`，工具协议进 `tools/`。
- **`api/correlation.py` 而非 `http/correlation.py`**：`api/app.py` 是应用工厂，中间件与路由都由它装配，
  放在同一层可以减少一次跨包跳转。等 HTTP 相关模块多起来再拆 `http/`。
- **新建 `tasks/`、`tools/` 两个包**：它们是后续 C3（Agent 任务循环、工具注入、工具结果回传）的落点，
  本批先把包建起来、放进去第一个模块，避免后续批次再改路径。
- **不为每模块单独建包**：四个模块分属三个包，不做一模块一包。

---

## 2. `api/correlation.py` —— 请求关联 ID

### 2.1 协议常量

| 项 | 值 | 性质 |
| --- | --- | --- |
| 请求/响应头名 | `X-Correlation-ID` | **线路协议**，跨服务日志对齐用，固定不变 |

**本仓库决定：** 该常量的**唯一定义处是 `src/shinku/names.py`**（B1 已建立，含注释说明它是线路协议）。
`api/correlation.py` 从 `names` 导入使用，**不再定义第二份**。

理由：旧项目把常量定义在 `request_context.py` 里（旧项目没有 `names.py`）；本仓库已有命名模块，
再定义一份会产生两个事实源。这是本仓库的组织决定，不是行为变更。

### 2.2 取值形状（调用方给的关联 ID 只有在满足以下形状时才被接受）

| 项 | 规定 |
| --- | --- |
| 长度 | 闭区间 **8 ≤ len ≤ 128** |
| 字符集 | 仅限 `A–Z`、`a–z`、`0–9`，以及 `.` `_` `:` `-` 四种分隔符 |
| 不接受的 | 空格、换行、制表符、`$`、`,`、`/` 等一切其他字符 |
| 长度按 **字符数** 计 | 不做字节数换算 |

### 2.3 归一化 `normalize_correlation_id(value: Any) -> str`

顺序固定，不可交换：

1. `str(value or "")` —— 非字符串先转字符串；**假值（`None`、`""`、`0`、`False`、`[]`）一律视为"没给"**；
2. `.strip()` —— 去掉首尾空白；
3. 形状合规 → 返回去空白后的原值；不合规 → **返回新生成的值**。

必须成立的行为：

| 输入 | 输出 |
| --- | --- |
| `"test-correlation-123"` | 原值 |
| `"AbC0._:-xyz"` | 原值（四种分隔符全在其中） |
| `"  abcdefgh  "` | `"abcdefgh"`（先 strip 再判形状） |
| `"a" * 8` / `"a" * 128` | 原值（边界含） |
| `"a" * 7` / `"a" * 129` | 新生成的值 |
| `"abc defgh"`、`"abc$defgh"`、`"abc\ndefgh"` | 新生成的值 |
| `12345678`（int） | `"12345678"`（转字符串后长度 8、全数字 → 接受） |
| `12`（int） | 新生成的值（长度不足） |
| `None` / `""` / `"   "` / `"\t"` / `0` / `False` | 新生成的值 |

### 2.4 生成值的形状

**32 位小写十六进制，无连字符**（`uuid4().hex`）。这是可观察契约：日志与测试都按此形状识别"生成态"。

### 2.5 上下文

- `current_correlation_id() -> str`
  - 在当前请求上下文内 → 本次请求的关联 ID；
  - 不在请求上下文内（包括尚未进入请求的普通代码路径）→ **空串 `""`**。
- 上下文**不跨任务泄漏**：在请求处理之外读取恒为空串。这条同时是测试时必须注意的坑
  （见 §7.2）。

### 2.6 出站头 `correlation_headers() -> dict[str, str]`

- 返回**恰好一个键**：`{CORRELATION_ID_HEADER: <值>}`；
- 值取当前上下文；上下文为空时**生成一个新值**；
- **每次调用独立**：连续两次调用（上下文为空时）必须得到不同值；
- 不抛异常。

### 2.7 中间件 `CorrelationIdMiddleware`

必须是 Starlette/FastAPI 兼容的 HTTP 中间件（`BaseHTTPMiddleware` 子类）。

单次请求的处理链，顺序固定：

1. 读入站头 `request.headers.get(HEADER)`；
2. 按 §2.3 归一化；
3. 把该值绑定到上下文变量（保存 reset token）；
4. 写入 `request.state.correlation_id`，供下游处理函数直接读取；
5. 调用下游；
6. **响应上回显同一个值**（响应头 `X-Correlation-ID` = 第 2 步的归一化结果）；
7. **无论如何都把上下文变量重置**（含下游抛异常的情况）。

必须成立的行为：

| 场景 | 结果 |
| --- | --- |
| 入站头为合法值 | 处理函数读到该值；响应头回显该值 |
| 缺入站头 | 为整次请求生成**一个**值；处理函数读到的与响应头回显的**是同一个** |
| 入站头畸形（如 `"short"`） | 替换为新生成的值，**不回显原畸形值** |
| 两个连续请求 | 各自得到不同的 ID |
| 下游抛异常 | 响应头不设置；**上下文变量必须已重置** |

### 2.8 导出面

`__all__` = `CorrelationIdMiddleware`、`correlation_headers`、`current_correlation_id`、
`normalize_correlation_id`。

**本仓库决定：不导出头名常量。** 旧模块把它放进 `__all__`，但旧项目里无任何生产消费方从该模块导入它
（只在测试里用过）；本仓库的常量唯一处是 `names.py`，需要它的代码直接从 `names` 取。
这属于导出面收窄，不影响任何消费方。

---

## 3. `api/sessions.py` —— 会话 HTTP 路由

### 3.1 依赖注入面

工厂函数（**全部关键字参数**）：

```
build_sessions_router(
    *,
    sessions: SessionReader,          # 会话存储读取面
    metrics: MetricsRecorder,         # 度量记录面
    log_event: LogEvent,              # 结构化日志
    resolve_identity: IdentityResolver,   # 从请求查询参数解析 (session_id, profile_user_id)
    normalize_pack_id: PackIdNormalizer,  # 角色包 ID 归一化
    build_error_payload: ErrorPayloadBuilder,  # 错误信封构造
) -> APIRouter
```

**与旧签名的差异（必须记录，属依赖倒置而非行为变更）：**

| 旧 | 本仓库 | 理由 |
| --- | --- | --- |
| `engine`（大对象，取 `engine.store`） | `sessions`（只注入读取面） | 旧模块通过 `engine.store` 触达整体存储；本仓库只声明真正用到的方法，避免在 C1-2 阶段引入整个引擎 |
| `runtime_metrics` | `metrics` | 同义，改名为本仓库风格 |
| `resolve_identity_from_query` + `resolve_identity_from_payload` | 合并为 `resolve_identity`，两处调用同一个 | 行为上两者都是"从某处解析出 `(session_id, profile_user_id)`"；旧项目拆两个是因为实现不同，本仓库把差异留给调用方 |
| （无） | `normalize_pack_id` | 旧模块直接调 `store.normalize_character_pack_id`；该函数所属模块不在 C1-2 范围，改为注入 |
| （无） | `build_error_payload` | 旧模块直接调 `desktop_pet_contract.build_desktop_pet_error_payload`；该模块不在 C1-2 范围，改为注入 |

**注入面的类型别名**（本仓库命名）：

```
IdentityResolver     = Callable[[Request | Mapping[str, Any]], tuple[str, str]]
PackIdNormalizer     = Callable[[Any], str]
ErrorPayloadBuilder  = Callable[..., dict[str, Any]]
LogEvent             = Callable[..., None]
MetricsRecorder      = Protocol: observe_request(name: str, *, duration_ms: float, ok: bool) -> None
```

`(session_id, profile_user_id)` 的**元组顺序不可改**——旧消费方按此顺序解包。

### 3.2 `GET /sessions`

- 身份来源：请求查询参数；
- 可选的 `pack_id`：从查询参数按 §3.5 的键序取第一个存在的键，交 `normalize_pack_id`；
- 成功响应：`{"sessions": [...], "current_session_id": <session_id>}`，列表上限 **50**；
- 存储抛异常：先记度量（`ok=False`）、再记一条结构化日志（事件名 `sessions_list_error`，含
  `session_id`／`profile_user_id`／`message`），**然后把异常继续抛出**（不吞）。

### 3.3 `POST /sessions/ensure`

- 身份来源：请求体；
- 请求体读取失败（非法 JSON）→ **400**，错误码 `invalid_json`，消息含截断到 **160** 字符的原因，`retryable=False`；
- 请求体不是对象 → **400**，错误码 `invalid_payload`，`retryable=False`；
- 正常路径：**确保会话存在**（不存在则创建，`display_title` 取 `str(payload.get("display_title") or "").strip() or None`）；
- 成功响应 = §3.4 的会话状态形状；
- 所有 400 响应 **必须带 `Cache-Control: no-store`**。

### 3.4 会话状态形状

```
{
  "session":            <会话对象>,
  "sessions":           <列表，上限 50>,
  "messages":           <列表，上限 120>,
  "latest_final_json":  <dict | None>,
}
```

`latest_final_json` 的取法：取该会话最新一条评测回合（`character_pack_id` 参与筛选），
取其 `final_json` 字段，**仅当它是 `dict` 时返回，否则 `None`**。

会话不存在且未要求创建 → **404**，`detail = "session not found"`。

### 3.5 `pack_id` 的取法

**键序固定**，取第一个存在的键：`character_pack_id` → `characterPackId` → `character_pack`。

- 请求体路径：先看顶层；顶层没有任何一个键时，看嵌套的 `current_visual` 字典（同样三键序）；
- 查询参数路径：只看查询参数，不做嵌套查找；
- 取到的值一律交 `normalize_pack_id`；
- 三键都不存在 → `None`（表示"不按角色包筛选"，与归一化后得到空串语义不同）。

`normalize_pack_id` 的契约（事实来源 `companion_v01/store.py` 的 `normalize_character_pack_id`）：
`str(value or "").strip()` → 空则 `""`；否则必须整体匹配 `^[A-Za-z0-9_.-]+$`，不匹配返回 `""`。

### 3.6 `POST /sessions/rename`

- `display_title` 取 `str(payload.get("display_title") or "").strip()`；
- 为空 → **400**，`detail = "display_title is required"`；
- 重命名返回 `None` → **404**，`detail = "session not found"`；
- 成功响应：`{"session": <重命名后的会话>, "sessions": <列表，上限 50>}`。

### 3.7 错误信封（契约事实，来源 `companion_v01/desktop_pet_contract.py`）

```
{
  "ok": false,
  "status": "error",
  "contract_version": "desktop_pet.v0.1",
  "error": "<error 或 'unknown_error'>",
  "message": "<message 或 '请求失败。'>",
  "retryable": <bool>,
  # details: 仅当传入非空时才出现
}
```

### 3.8 度量与日志

- 每次请求（成功与失败**都**）调 `metrics.observe_request(name, duration_ms=<毫秒浮点>, ok=<bool>)`；
- 度量名固定为：`sessions_list`、`sessions_ensure`、`sessions_rename`；
- 计时用单调时钟（`perf_counter` 语义），不用挂钟；
- 结构化日志只在这两处：`sessions_list_error`、`sessions_ensure_error`。

### 3.9 路由前缀

工厂返回的 `APIRouter` **不带前缀**，路径为 `/sessions`、`/sessions/ensure`、`/sessions/rename`。
挂载前缀由装配方决定。

---

## 4. `tasks/artifacts.py` —— 任务工作区产物

六条纯函数，**无 I/O、无全局状态**，输入输出都是普通字典/列表。

### 4.1 `artifact_from_generated_file(*, generated, tool_type, send_to_user) -> dict | None`

- `generated` 非字典 → `None`；
- 取 `handle = str(generated["generated_handle"] or "").strip()`、
  `generated_id = str(generated["generated_id"] or "").strip()`；
- `id = handle or generated_id`；为空 → `None`；
- 固定字段：

| 字段 | 值 |
| --- | --- |
| `id` | 见上 |
| `kind` | `str(output_format or file_ext or "file").strip().lower() or "file"` |
| `title` | `str(output_title or handle or "生成文件").strip()[:120]` |
| `status` | `str(status or "ready").strip() or "ready"` |
| `source` | 固定 `"generated_file"` |
| `tool` | `tool_type` 原样 |
| `send_to_user` | `bool(send_to_user)` |
| `delivery_role` | `send_to_user` 为真 → `"requested_output"`；否则 `"workspace_material"` |

- 条件字段：`generated_id`（非空才加）、`generated_handle`（非空才加）；
- `stem_role`：从 `content_card.separation.stem_role` 取（两级都必须是字典），非空才加，**截断到 40**；
- 透传字段：`file_ext`、`file_size`、`created_by_tool`、`version_of_generated_id`、`version_no` ——
  **仅当值不在 `(None, "", [], {})` 内时**加入，值原样（不做 str 转换）。

### 4.2 `artifact_from_attachment_item(*, item, tool_type) -> dict | None`

- `item` 非字典 → `None`；
- `handle = str(item["attachment_handle"] or "").strip()`、`attachment_id = str(item["attachment_id"] or "").strip()`；
- `id = handle or attachment_id`；为空 → `None`；
- 固定字段：

| 字段 | 值 |
| --- | --- |
| `id` | 见上 |
| `kind` | `str(kind or file_ext or "file").strip().lower() or "file"` |
| `title` | `str(summary_title or origin_name or handle or "临时素材").strip()[:120]` |
| `status` | `str(status or "ready").strip() or "ready"` |
| `source` | 固定 `"attachment_inbox"` |
| `tool` | `tool_type` 原样 |
| `source_type` | `str(item["source"] or "").strip()`（注意：源字段是 `source`，落在产物上叫 `source_type`） |
| `delivery_role` | 固定 `"workspace_material"` |

- 条件字段：`attachment_id`、`attachment_handle`（非空才加）；
- 透传字段：`origin_name`、`file_ext`、`file_size`、`mime_type`（同 §4.1 的排除规则）。

### 4.3 `extract_artifacts_from_tool_events(*, tool_type, stream_events) -> list[dict]`

遍历事件流，按 `type` 分派：

| `type` | 处理 |
| --- | --- |
| `"generated_file_ready"` | 用事件里的 `generated_file` 调 §4.1，`send_to_user=bool(event["send_to_user"])` |
| `"attachment_remote_media_ready"` | 用事件里的 `item` 调 §4.2 |
| 其他 | 跳过 |

非字典事件跳过；构造结果为假值（`None` 或空字典）时不加入结果。返回顺序与事件顺序一致。

### 4.4 `artifact_identity(artifact) -> str`

按**固定键序**取第一个非空值：`id` → `generated_handle` → `generated_id` →
`attachment_handle` → `attachment_id`。

全部为空时回退：`title` 非空 → `f"title:{kind}:{title}"`；否则 `""`。

（`kind` 缺失时按空串参与拼接。）

### 4.5 `merge_artifacts(*, existing, additions) -> tuple[list[dict], list[dict]]`

- `merged`：`existing` 中每个字典元素的**浅拷贝**；
- `seen`：`merged` 中所有非空身份串的集合；
- 遍历 `additions`：非字典跳过；身份串为空、或已存在于 `seen` → 跳过；
- 否则把该元素的**浅拷贝**同时追加到 `merged` 和 `added`，并把身份串加入 `seen`。

**可观察契约（必须保留）：** 追加进 `merged` 与 `added` 的是**同一个字典对象**——
通过返回的 `added` 修改某元素，`merged` 里对应元素同步可见；反之亦然。
这不是实现细节的偶然，是本函数的既有语义，消费方依赖它。

返回 `(merged, added)`，`added` 只含本次新增、按 `additions` 原序。

### 4.6 `compact_workspace_for_event(task) -> dict`

```
{
  "task_id":         str(task["task_id"] or ""),
  "status":          str(task["status"] or ""),
  "normalized_goal": str(task["normalized_goal"] or "")[:200],
  "artifact_count":  len(list(task["artifacts"] or [])),
}
```

`artifact_count` 是**长度**，不是列表本身；`artifacts` 缺失或为假值 → `0`。

### 4.7 导出面

`__all__` 六个名字，与 §4.1–§4.6 一一对应（本仓库命名见 §1.1，去掉 `task_workspace_` 冗余前缀）。

---

## 5. `tools/invocation.py` —— 工具调用协议

### 5.1 协议常量

| 常量 | 值 | 性质 |
| --- | --- | --- |
| `LEGACY_JSON` | `"legacy_json"` | 来源标记 |
| `NATIVE_OPENAI` | `"native_openai"` | 来源标记 |
| `NATIVE_ANTHROPIC` | `"native_anthropic"` | 来源标记 |
| `TOOL_SOURCE_FIELD` | `"_tool_source"` | **嵌入在工具调用里的元数据键**，属线路约定 |
| `TOOL_INVOCATION_ID_FIELD` | `"_tool_invocation_id"` | 同上 |
| `NATIVE_TOOL_CALL_FIELD` | `"_native_tool_call"` | 同上 |

三个下划线开头的键是**跨模块串接用的字段名**，多个模块按字符串匹配它们——
属线路协议，**不随命名风格改动**。

### 5.2 记录类型（三个都是**可变** dataclass）

| 类型 | 必填 | 可选（默认值） |
| --- | --- | --- |
| `ToolInvocation` | `name: str` | `arguments: dict = {}`、`source: str = LEGACY_JSON`、`id: str = ""` |
| `ValidationResult` | `ok: bool` | `message: str = ""`、`code: str = ""` |
| `ToolResultEnvelope` | `invocation_id: str`、`status: str`、`model_feedback: str` | `data: dict \| None = None`、`events: list = []` |

- 三个类型都**可赋值修改**（不是 frozen）；
- 集合类默认值**不共享**（每个实例独立）；
- `ToolInvocation` 在构造后：若 `id` 去空白后为空 → 置为 `"call_" + <16 位小写十六进制>`；
  非空则**保持原值不动**；
- `ValidationResult.success() -> ValidationResult`：`ok=True`，`message`/`code` 为默认空串；
- `ValidationResult.fail(code, message) -> ValidationResult`：`ok=False`，两字段都做 `str(... or "")`。

### 5.3 转换函数

**`legacy_tool_call_to_invocation(tool_call, *, source=LEGACY_JSON, invocation_id="") -> ToolInvocation | None`**

- 非字典 → `None`；
- `name = str(tool_call["type"] or "").strip()`；为空 → `None`；
- `embedded_source = str(tool_call[TOOL_SOURCE_FIELD] or "").strip()`、
  `embedded_id = str(tool_call[TOOL_INVOCATION_ID_FIELD] or "").strip()`；
- `arguments` = **除 `"type"` 之外、且键名不以 `"_tool_"` 开头的所有键值对**（值原样）；
- `source = embedded_source or source`；`id = embedded_id or invocation_id`（可能仍为空串 → 触发 §5.2 的生成）。

**`invocation_to_legacy_tool_call(invocation, *, include_metadata=False) -> dict`**

- 基底 `{"type": invocation.name, **invocation.arguments}`（**参数键在后**，同名时覆盖 `type`）；
- `include_metadata` 为真 **且** `str(invocation.source or LEGACY_JSON) != LEGACY_JSON` 时，
  追加 `TOOL_SOURCE_FIELD` 与 `TOOL_INVOCATION_ID_FIELD`（都做 `str(... or "")`）；
- 否则只返回基底。

**`round_trip_legacy_tool_call(tool_call) -> dict | None`**

- 等价于"转成 invocation 再转回 legacy"，**不带 `include_metadata`**；
- 输入非法 → `None`。

### 5.4 `round_trip` 的可观察语义（必须成立）

| 输入 | 输出 |
| --- | --- |
| `{"type": "t", "a": 1}` | `{"type": "t", "a": 1}` |
| `{"type": "t", "a": 1, "_tool_source": "native_openai"}` | `{"type": "t", "a": 1}`（元数据被剥掉） |
| `{"type": "t", "_tool_invocation_id": "call_x", "b": 2}` | `{"type": "t", "b": 2}` |
| `{"a": 1}`（无 type） | `None` |
| 非字典 | `None` |
| `{"type": "  "}` | `None` |

即 **round trip 会把 `_tool_*` 元数据规范化掉**——这是"归一化到 provider 无关形态"的既有含义。

### 5.5 导出面

`__all__` = 六个常量 + 三个记录类型 + 三个转换函数，共 **12 个名字**。

---

## 6. 本批不做的事

- **不迁移** `store.py` 的 `normalize_character_pack_id` 与 `desktop_pet_contract.py` 的
  `build_desktop_pet_error_payload` 本体——只把它们的**行为与信封形状**作为契约事实记录（§3.5／§3.7），
  实现由 `api/sessions.py` 的注入面提供；
- **不装配路由与中间件**：本批只产出可被装配的工厂与中间件类。
  把 `CorrelationIdMiddleware` 挂到 `api/app.py`、把 sessions 路由注册进应用，属 C4/C6 的装配工作；
- **不引入 `engine`**：`api/sessions.py` 只声明它真正调用的存储方法；
- **不做 `task_workspace`、`tool_orchestration` 的其余部分**（C3 范围）。

---

## 7. 验收

### 7.1 必须通过

- [ ] 四个模块在新仓库可导入，且**不导入旧项目任何模块**（`companion_v01` / `code_shared` / `akane` 零命中）；
- [ ] 本批新增文件里不出现旧项目路径字符串与旧环境变量前缀；
- [ ] 行为测试覆盖 §2、§3、§4、§5 的每一条"必须成立的行为"；
- [ ] **失败测试**覆盖各模块的拒绝面（畸形形状、非法输入、缺失必填字段、越界长度）；
- [ ] `request_context` 的**异常路径**有对应用例（§2.7 最后一行）——这是旧测试的缺口；
- [ ] 干净环境全量回归零失败（B1 38 + C1-1 30 + 本批）；
- [ ] 旧项目源码零改动；
- [ ] `docs/SOURCE_RECORD.md` 按 §5.9 分别登记「契约事实来源」与「表达层撰写情况」。

### 7.2 测试时必须避开的假通过

`current_correlation_id()` 在 `TestClient` **外层**读取恒为空串——上下文变量不跨任务泄漏。
**用它在请求外断言"已重置"是假通过**（默认值本来就是空串）。

要真正验证 §2.7 的最后一行，必须**直接驱动 `dispatch`**：自行构造一个会抛异常的 `call_next`，
在**同一个任务内**于调用前后读取上下文值，断言调用后已回到调用前的状态。

---

## 8. 验收结果

见 `docs/shinku_independent_c1_2_shared_modules_20260918.md`（批次记录）。

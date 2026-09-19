# C2-5 契约文档：LLM 调用引擎（`llm_runtime` → `shinku.llm.runtime`）

- 批次：C2-5a（本批）/ C2-5b（能力协商）/ C2-5c（审计与用量度量）
- 日期：2026-09-19
- 类型：**洁净室重写**（含逻辑 / 分支 / 算法 / 数据处理，C 阶段判据 B 类）
- 事实来源：`AkaneCompanionLab_Shinku/companion_v01/llm_runtime.py`（98,409 B / 2,344 行）

## 1. 对照物形态（规矩二十五第三例「上游无对应物」）

上游 `code_shared` 的 59 个模块里没有 `llm_runtime.py`，也没有任何等价的
调用编排层——它是 Shinku 项目侧架在共享原语（client / 熔断 / 模型服务配置）
之上的自研引擎。因此：

| 侧 | 路径 | 字节 | 用途 |
| --- | --- | --- | --- |
| 上游 | 不存在 | — | — |
| Shinku 旧侧 | `companion_v01/llm_runtime.py` | 98,409 B / 2,344 行 | **行为基线**（唯一对照物） |

Akane 侧 `companion_v01/llm_runtime.py` 是 `code_shared` 兼容转发壳（同路径
文件转发共享包），不构成独立证据，不参与对照。

`provider_probe.py`（诊断 CLI）与本批无关：旧仓零 import、纯 dev 工具，
暂缓迁移（届时上游 `code_shared/provider_probe.py` 有现成契约可对照）。

## 2. 切批方案（依赖图驱动，2026-09-19 使用者拍板）

旧实现是单类巨石（`LLMRuntime` 84 方法），但方法簇边界清晰，按簇拆
三个文件、三个批次，每批都有可跑可测的增量面：

| 批次 | 落点 | 内容 | 状态 |
| --- | --- | --- | --- |
| C2-5a | `runtime_core.py` + `runtime.py` | 传输骨架：bundle 构建/热切换、三通道（JSON/NDJSON/流式）、payload 组装、瞬时重试与熔断记账、`StreamTap` 流式解析、JSON 修复/截断/兜底、度量计数与 last-error | 本批 |
| C2-5b | `runtime_capabilities.py` | 原生工具（tools/tool_choice/tool_calls）、(host, model) 画像与 allowlist、thinking 控制、图片项 | **已完成** |
| C2-5c | `runtime_audit.py` | prompt 审计、token/缓存用量度量（双口径）、缓存提示组装 | **已完成** |

组合根 `runtime.py`：`class LLMRuntime(RuntimeCapabilitiesMixin,
RuntimeAuditMixin, RuntimeCore)`。三片通过 `self.<方法>` 互调（core 调
capabilities/audit 的桥接点在 `_build_payload` / `_run_json_call` /
`_stream_chat` / `_run_ndjson_call`），MRO 与混入顺序保证解析到正确实现。

**桩边界（C2-5a）**：capabilities/audit 两文件先放「空输入 / 默认环境」
路径的最小等价实现（如 `_coerce_tools(None) → []`、`_record_cache(...) →
False`、`_maybe_record_audit(...) → no-op`）。即在「不启用原生工具、
不开审计、不带缓存提示、非 DeepSeek」的路径上，C2-5a 的 LLMRuntime
行为与旧实现一致；这些路径之外的行为由 C2-5b/C2-5c 替换桩体后补齐。
对拍与变异在 C2-5a 先覆盖桩等价路径；C2-5b/c 收口后已扩到能力与审计全量矩阵。

## 3. 公开契约面（后续批次 engine / memory_compaction_service 依赖）

类与函数（名字、签名、参数顺序不变）：

- `LLMRuntime(*, metrics_path: Path | None = None)`
- `LLMRuntime.reload_from_config() -> dict[str, str]`（键 `status`/`auxModel`/`chatModel`）
- `LLMRuntime.call_aux_json(*, system_prompt, user_prompt, fallback, temperature=0.2, prompt_cache_key="")`
- `LLMRuntime.call_chat_json(*, system_prompt, user_prompt, fallback, temperature=0.7, prompt_cache_key="", user_images=None, native_tools=None, native_tool_choice="", system_extra_blocks=None, history_turns=None, prompt_audit_sections=None)`
- `LLMRuntime.chat_supports_native_tools() -> bool`
- `LLMRuntime.record_metric(key: str, amount: int = 1) -> None`
- `LLMRuntime.call_aux_ndjson(*, system_prompt, user_prompt, on_event=None, temperature=0.2, prompt_cache_key="") -> NDJSONCallResult`
- `LLMRuntime.stream_chat_json(*, system_prompt, user_prompt, fallback, temperature=0.7, early_tool_call_validator=None, prompt_cache_key="", user_images=None, native_tools=None, native_tool_choice="", system_extra_blocks=None, history_turns=None, prompt_audit_sections=None) -> Generator[dict, None, ChatJSONStreamResult]`
- `LLMRuntime.snapshot_metrics() -> dict[str, int]`
- `LLMRuntime.recent_metric_days(days: int = 7) -> list[dict[str, Any]]`
- `LLMRuntime.metrics_persistence_metadata() -> dict[str, Any]`
- `LLMRuntime.close() -> None`
- `LLMRuntime.snapshot_last_error() -> dict[str, str]`
- `normalize_reply_medium(value) -> str`（别名表含中文键，取值 `text`/`voice`/`both`/`""`）
- 数据类：`ModelBundle(client, model)`；`ProviderToolProfile`（frozen，6 字段：
  `supports_native_tools=False`、`native_tools_coexist_with_forced_json=False`、
  `native_call_shape="openai_tool_calls"`、`verified=False`、
  `force_response_json=True`、`notes=""`）；`NDJSONCallResult`（7 字段）；
  `ChatJSONStreamResult`（11 字段，含 `stopped_early`/`early_tool_call`/
  `rescue_attempted`/`rescue_succeeded`）
- `StreamTap`（流式顶层 JSON 增量抽取器；`.feed(text) -> list[事件]`，
  `latest_emotion` / `latest_speech` / `latest_reply_medium` 属性）
- 实例属性 `aux` / `chat`（`ModelBundle`）
- 模块常量：`DEFAULT_PROVIDER_TOOL_PROFILE`、`CONFIG_ALLOWLISTED_PROVIDER_TOOL_PROFILE`、
  `PROVIDER_TOOL_PROFILES`（含两条 deepseek 条目，notes 原文保留）

## 4. 契约事实（字面量逐字保留）

- 度量键全集（41 个构造初始键 + 运行期追加键 `llm_http_retry` /
  `chat_stream_retries` / `response_truncated` 等）——`snapshot_metrics()`
  的可观察面。
- `StreamTap` 事件类型字符串：`speech_chunk` / `ui` / `delivery_hint` /
  `speech_segment`；`_ensure_json_hint` 插入的中文提示
  `"（本轮只输出一个合法的 JSON object，不要输出多余文字。）"`。
- last-error 通道的 `type` 字符串：`ResponseTruncated`、`ChatJSONFallback`。
- 重试口径：`_HTTP_ATTEMPTS=2`、退避 0.8s×attempt、可重试状态码
  `{408,409,425,429,500,502,503,504,529}`、可重试 token 列表（14 项）。
- `SECRET_PATTERNS` 三条打码正则；`REPLY_MEDIUM_ALIASES` 别名表。
- 工具字段常量（与 C1-2 `tools/invocation.py` 一致）：`NATIVE_OPENAI`、
  `TOOL_SOURCE_FIELD`、`TOOL_INVOCATION_ID_FIELD`、`NATIVE_TOOL_CALL_FIELD`。
- rescue 哨兵键 `__shinku_stream_rescue_fallback__`；rescue 失败缺省消息
  `stream_json_rescue_failed`。
- `reload_from_config` 返回键 `status="reloaded"` / `auxModel` / `chatModel`。
- 诊断日志通道 `shinku.llm_debug`（logger 名保留）。
- `_chat_reply_needs_rescue` 的「有意沉默」白名单：`status in
  {skip, silent, decline}` 不触发 rescue。

## 5. 配置访问决定（本批新增）

新仓 `Settings` 不含 LLM 供应商字段；旧实现 `config.X` 的读取点全部改为
`SHINKU_X` 环境变量（`RuntimeCore._env`），键名沿用旧配置属性名（契约事实，
待 C4/C6 复查）：

| 旧 `config.*` | 新环境变量 | 默认 |
| --- | --- | --- |
| `AUX_API_KEY` / `AUX_BASE_URL` / `AUX_API_PROTOCOL` / `AUX_MODEL_NAME` | `SHINKU_AUX_*` | `""`/`""`/`auto`/`""` |
| `CHAT_API_KEY` / `CHAT_BASE_URL` / `CHAT_API_PROTOCOL` / `CHAT_MODEL_NAME` | `SHINKU_CHAT_*` | `""`/`""`/`auto`/`""` |
| `LLM_PROMPT_AUDIT_ENABLED` / `LLM_PROMPT_AUDIT_INCLUDE_AUX` | `SHINKU_LLM_PROMPT_AUDIT_*` | 关（C2-5c 生效） |
| `LOG_DIR` | `SHINKU_LOG_DIR` | `logs`（C2-5c 生效） |
| `NATIVE_TOOL_PROVIDER_ALLOWLIST` | `SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST` | `""`（C2-5b 生效） |
| `LLM_THINKING_MODE` | `SHINKU_LLM_THINKING_MODE` | `disabled` |
| `PROMPT_CACHE_HINTS_ENABLED` / `PROMPT_CACHE_HINTS_FORCE` / `PROMPT_CACHE_RETENTION` / `PROMPT_CACHE_NAMESPACE` | `SHINKU_PROMPT_CACHE_*` | `True`/`False`/`""`/`shinku`（C2-5c 生效） |

`PersistentCounterStore` 在新仓不存在：可选导入，缺失时 `_counter_store=None`
退回内存计数。`metrics_path=None` 时两侧同无持久化，对拍安全。

## 6. 表达层决定（Shinku 撰写）

- 全部 `_*` 私有名整组另起（63 项重命名映射，核心如
  `_call_json→_run_json_call`、`_record_metric→_add_metric`、
  `_TopLevelJSONStreamTap→StreamTap`、`_RetryableLLMError→TransientLLMError`、
  实例属性 `_metrics→_counters` / `_last_error→_error_state` 等）。
- docstring / 注释全部 Shinku 自写；协议字面量与 notes 数据原样保留
  （规矩十二：输出文案与协议字面量不进散文口径）。
- 文件切分按簇三片；`import logging as _log` 改为 `import logging`。

## 7. 行为保证（对拍与变异的靶子）

1. 非流式 JSON：解析链 `json.loads → _first_json_object → JSON_RE →
   _heal_json`；解析不出走 `_recover_partial`（半截 speech/emotion 恢复），
   再不行记 `ChatJSONFallback` 并返回 `dict(fallback)`。
2. 瞬时重试：`TransientLLMError` 循环重试一次；非瞬时错误进熔断记账
   （`get_llm_circuit_breaker().record_failure()`）。
3. `TypeError`（网关不认 prompt_cache 参数）→ 剥提示重发一次。
4. 流式：连接层空产出重试一次（`chat_stream_retries`）；早停探测
   `_scan_stream_tool_call` 返回 `pending/none/null/object` 四态；
   rescue 只跑一次且非流式结果权威（tap 状态清空）。
5. NDJSON：行缓冲 `_drain_buffer` + 尾行补解析；`on_event` 返回真值即
   早停；`completed_stream = not stopped_early and not error`。
6. 度量：`_add_metric` 持久化优先、内存兜底；`record_metric` 是公开面。

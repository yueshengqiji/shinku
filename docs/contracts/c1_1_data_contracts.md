# C1-1 行为契约：数据类型契约层

**批次：** C1-1（真红独立计划 C 阶段）
**日期：** 2026-09-18
**处理类型：** **契约事实迁移（来源记录 §5.9 路径 A）**
**本批 5 个来源模块 → 本仓库 4 个文件**

---

## 0. 本契约的性质

C1-1 的 5 个来源模块经 A2-9 取证均为**纯契约或近纯契约**（无分支、无算法），
因此按路径 A 走**契约事实迁移**，而不是形式化洁净室重写。

本文件记录的是**外部行为规格**，用于：

1. 固定本仓库必须提供的公开名字与行为；
2. 让实现可以被独立测试验证；
3. 在来源台账里区分「契约事实来源」与「表达层撰写」。

**本文件只写"要有哪些名字、要满足什么行为"，不描述旧实现怎么组织、怎么命名、怎么注释。**

---

## 1. 来源模块 → 本仓库模块映射

| 来源模块（旧项目） | 字节 | 本仓库模块 | 说明 |
| --- | --: | --- | --- |
| `companion_v01/task_types.py` | 574 | `src/shinku/contracts/tasks.py` | 与下一行合并为同一模块（同属"委派工作"域） |
| `companion_v01/task_worker_contract.py` | 1,282 | `src/shinku/contracts/tasks.py` | 同上 |
| `companion_v01/retrieval_types.py` | 1,050 | `src/shinku/contracts/retrieval.py` | |
| `companion_v01/http_response.py` | 647 | `src/shinku/contracts/http.py` | |
| `companion_v01/capability_adapters/types.py` | 3,354 | `src/shinku/contracts/capability.py` | |

**文件切分与来源切分不一致是有意的**：来源侧把"任务记录"与"工具词表"拆成两个文件，
本仓库按域合并为一个。文件组织属表达层，不受契约事实约束。

---

## 2. `contracts/tasks.py`

### 2.1 委派记录

**契约：** 需要两个记录形状，描述"交给专员的委派"与"专员跑完后的回报"。

**`WorkerDelegation`（不可变）**

| 字段 | 类型 | 必填 | 默认 |
| --- | --- | :-: | --- |
| `task_id` | `str` | 是 | — |
| `assigned_agent` | `str` | 是 | — |
| `brief` | `str` | 是 | — |
| `handle_id` | `str` | 否 | `""` |
| `started` | `bool` | 否 | `False` |

行为要求：实例创建后**不可赋值修改**；创建时对未知字段应报 `TypeError`。

**`WorkerRunSummary`（可变）**

| 字段 | 类型 | 必填 | 默认 |
| --- | --- | :-: | --- |
| `task_id` | `str` | 是 | — |
| `assigned_agent` | `str` | 是 | — |
| `status` | `str` | 是 | — |
| `rounds` | `int` | 否 | `0` |
| `messages` | `list[str]` | 否 | 新建空列表 |
| `tool_results` | `list[str]` | 否 | 新建空列表 |

行为要求：`messages` 与 `tool_results` 的默认值**必须是每个实例独立的新列表**，
两个实例之间不得共享同一个列表对象（否则一个实例的 append 会污染另一个）。

### 2.2 专员工具词表

**契约：** 需要两张表——「每个专员被允许调用的工具集合」与「可用的短名到规范专员的映射」。

**`AGENT_ALLOWED_TOOLS`**：`dict[str, set[str]]`。

规范专员与各自被允许的工具（工具名是协议常量，必须逐字一致）：

| 专员 | 被允许的工具 |
| --- | --- |
| `document_agent` | `sync_attachment_workspace`、`inspect_attachment`、`read_attachment_section`、`compose_file`、`revise_generated_file`、`apply_style_to_existing_file`、`inspect_generated_file` |
| `media_agent` | `fetch_media_from_url`、`sync_attachment_workspace`、`inspect_attachment`、`inspect_media_info`、`convert_media_file`、`separate_audio_stems`、`clean_voice_track`、`transcribe_media`、`prepare_voice_dataset`、`inspect_generated_file`、`compose_file` |
| `speech_agent` | `sync_attachment_workspace`、`inspect_attachment`、`inspect_media_info`、`separate_audio_stems`、`clean_voice_track`、`transcribe_media`、`prepare_voice_dataset`、`compose_file`、`revise_generated_file`、`inspect_generated_file` |
| `resource_agent` | `sync_attachment_workspace`、`inspect_attachment`、`read_attachment_section`、`inspect_generated_file` |

行为要求：
- 值必须是 `set`（不是 `list` / `tuple`），因为调用方依赖成员测试；
- 键集合必须**恰好**是上表 4 个规范名。

**`AGENT_ALIASES`**：`dict[str, str]`，短名 → 规范专员名。

| 短名 | 指向 |
| --- | --- |
| `auto` | `media_agent` |
| `worker` | `media_agent` |
| `video_agent` | `media_agent` |
| `audio_agent` | `media_agent` |
| `file_agent` | `document_agent` |
| `doc_agent` | `document_agent` |

行为要求：
- 每个别名**必须指向 `AGENT_ALLOWED_TOOLS` 里存在的键**（否则别名会解析到未定义的专员）；
- 别名本身**不得**同时是规范名（避免解析歧义）。

---

## 3. `contracts/retrieval.py`

**契约：** 需要**一个**不可变信封，承载"一次检索决策 + 随后的验证"的完整结果。

**`RetrievalPipelineResult`（不可变）**

| 字段 | 类型 |
| --- | --- |
| `used_retrieval` | `bool` |
| `confirmed_snippets` | `list[str]` |
| `router_output` | `dict[str, Any]` |
| `router_timing` | `dict[str, Any]` |
| `retrieval_result` | `dict[str, Any]` |
| `verifier_output` | `dict[str, Any]` |
| `verifier_timing` | `dict[str, Any]` |

行为要求：
- 全部 7 个字段**必填、无默认值**（漏传任一字段应报 `TypeError`）；
- 实例创建后**不可赋值修改**；
- 三个 `*_timing` / `*_output` / `retrieval_result` 字段的类型是 JSON 形状的 `dict`，
  本契约**不**依赖数据库、嵌入库或模型 SDK。

---

## 4. `contracts/http.py`

**契约：** 需要一个函数，把任意负载包成 JSON 响应，并强制带缓存指令。

**`json_response(payload, status_code=200, *, cache_control="no-store", headers=None) -> JSONResponse`**

行为要求：

1. 返回值是 FastAPI 的 `JSONResponse`，正文为 `payload` 的 JSON 序列化；
2. `status_code` 经 `int()` 归一化后写入响应（传 `"201"` 应得 201）；
3. 响应**始终**带 `Cache-Control` 头，默认值 `"no-store"`；
4. `cache_control` 经 `str()` 归一化后写入；
5. `headers` 为 `None` 时不额外加头；
6. `headers` 里的键与值都经 `str()` 归一化后合并；**合并后的字典允许覆盖 `Cache-Control`**
   （即调用方显式传 `{"Cache-Control": "..."}` 时应生效）——这是有意保留的逃生口；
7. `cache_control` 与 `headers` 都是**仅关键字**参数。

---

## 5. `contracts/capability.py`

**契约：** 需要描述"能力提供方"的一组不可变值对象，覆盖：提供方身份、端点、健康检查、
档位、触发条件、能力描述、输入输出槽位、调用上下文，以及两种错误类型。

### 5.1 类型别名

| 名字 | 取值集合 |
| --- | --- |
| `SourceLayer` | `"builtin"`、`"profile"` |
| `RiskLevel` | `"low"`、`"medium"`、`"high"` |
| `ConfirmPolicy` | `"never"`、`"first_time"`、`"always"` |

### 5.2 值对象（全部不可变）

| 名字 | 字段（类型 = 默认值） |
| --- | --- |
| `EndpointConfig` | `url: str` · `loopback_only: bool = False` · `raw: Mapping[str, Any] \| None = None` |
| `HealthConfig` | `method: str` · `path: str` · `timeout_seconds: float` · `expect_status: tuple[int, ...]` · `raw: Mapping[str, Any] \| None = None` |
| `TierConfig` | `id: str` · `label: str` · `preset: Mapping[str, Any]` · `raw: Mapping[str, Any] \| None = None` |
| `TriggerConfig` | `kind: str` · `raw: Mapping[str, Any]` |
| `CapabilityIOSlot` | `name: str` · `kind: str` · `required: bool = False` · `max_bytes: int \| None = None` · `delivery: str = ""` · `raw: Mapping[str, Any] \| None = None` |
| `CapabilityDescriptor` | `id: str` · `display_name: str` · `short_hint: str` · `visible_in: tuple[str, ...]` · `prompt_exposed: bool` · `risk: RiskLevel` · `confirm: ConfirmPolicy` · `effects: tuple[str, ...]` · `trigger: TriggerConfig \| None` · `inputs: tuple[CapabilityIOSlot, ...]` · `outputs: tuple[CapabilityIOSlot, ...]` · `raw: Mapping[str, Any]` |
| `CapabilityManifest` | `schema: str` · `provider_id: str` · `provider_type: str` · `display_name: str` · `endpoint: EndpointConfig \| None` · `health: HealthConfig \| None` · `tiers: tuple[TierConfig, ...]` · `capabilities: tuple[CapabilityDescriptor, ...]` · `secrets: tuple[str, ...]` · `source_path: Path` · `source_layer: SourceLayer` · `raw: Mapping[str, Any]` |
| `InvalidManifest` | `source_path: Path` · `source_layer: SourceLayer` · `reason: str` · `detail: str = ""` · `provider_id: str = ""` |
| `HealthStatus` | `ok: bool` · `status: str` · `reason: str = ""` |
| `CapabilityResult` | `is_error: bool` · `content: Any = None` · `status: str = ""` · `reason: str = ""` |
| `InvocationContext` | `profile_user_id: str = ""` · `session_id: str = ""` · `client_mode: str = ""` · `qq_user_id: str = ""` · `master_qq: str = ""` |

行为要求：全部不可变；`InvocationContext` 的 5 个字段**全部有默认值**（可零参构造）。

### 5.3 错误类型

| 名字 | 基类 |
| --- | --- |
| `CapabilityManifestError` | `ValueError` |
| `CapabilityProtocolError` | `RuntimeError` |

行为要求：基类必须准确——调用方依赖 `except ValueError` / `except RuntimeError` 可捕获。

### 5.4 导出面

`__all__` 必须包含上述 3 个别名 + 11 个值对象 + 2 个错误类型 = **16 个名字**。

---

## 6. 本批不做的事

- ❌ 不搬运 `capability_adapters/manifest.py` 的解析逻辑（属 C1-3，洁净室重写）；
- ❌ 不搬运 `manifest_loader.py` 的转发壳（与 `manifest` 同批处理）；
- ❌ 不引入任何对旧项目的运行时依赖。

---

## 7. 验收

| 项 | 要求 |
| --- | --- |
| 实现 | `src/shinku/contracts/` 下 4 个模块 + 包标记 |
| 测试 | `tests/test_contract_c1_1.py`，含**行为测试**与**失败测试** |
| 发布记录 | 公开契约文档登记契约事实与表达层 |
| 边界 | import graph 不含旧项目；不含 `code_shared` |
| 回归 | 新仓库全量 pytest 通过（B1 的 38 项 + 本批新增） |

---

## 8. 验收结果（2026-09-18 实测）

| 项 | 结果 |
| --- | --- |
| 交付文件 | `src/shinku/contracts/{__init__,tasks,retrieval,http,capability}.py`（5）+ `tests/test_contract_c1_1.py`（1） |
| 本批测试数 | **30 个**（7 个测试类），另含 89 个 subtest |
| 全仓库回归 | **68 passed / 0 failed / 0 error / 0 skipped**，89 subtests passed，0.69s |
| 其中 B1 基线 | 38 passed（`test_names` 22 + `test_health` 8 + `test_cli` 8） |
| 边界扫描 | `docs/evidence/20260918/c1_1_boundary_scan.json`：禁止导入 0 项 |
| 测试内建边界断言 | `ContractBoundaryTests` 逐文件扫描 `companion_v01` / `code_shared` / `legacy` 字样，全部未出现 |

**本批暴露的问题：** 契约测试首版有一处变量名笔误（`field` 未定义，应为遍历
`dataclasses.fields(...)` 得到的 `spec`），导致 7 个 subtest 失败。已修正并复跑通过。
记录在案是因为「失败测试确实跑起来了」这件事本身是证据——说明这些断言不是装饰性的。

**本批未改动：** 旧项目任何源码。旧项目本批只新增文档与证据。

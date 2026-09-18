# C1-3 行为契约：能力清单与资源清单（洁净室重写）

**制定日期：** 2026-09-18
**处理类型：** 洁净室重写（来源记录 §5.9 路径 A 的第二类）
**依据：** 计划 §5「C1 批次划分」；`SOURCE_RECORD.md` §4.2「C 阶段起」

---

## 0. 本契约的性质

C1-3 的三个来源模块都**含逻辑、分支、算法或数据处理**，因此按路径 A 属**洁净室重写**：

> 依行为契约独立实现。契约事实可迁移，**表达层（docstring、注释、内部组织与文件切分、
> 内部命名风格、实现步骤与算法写法）必须 Shinku 单独撰写**。

本文件**只记录外部可观察行为**，作为新实现的唯一需求输入。它记录四件事：

1. **必须成立的行为**——新实现与旧行为等价的部分；
2. **契约事实**——常量名/值、字段名/类型、签名与参数顺序、错误原因码、输出文案。这些是
   「接口信息」（路径 A 允许按事实迁移），本仓库照录，但**表达方式必须自撰**；
3. **本仓库自有的决定**——文件切分、内部类划分、命名、控制流组织等表达层选择；
4. **刻意不做的行为**——明确排除的部分。

**关于旧实现的说明：** 本契约的行为面来自对旧模块的读取与既有行为测试
（`tests/test_capability_adapter_manifest_loader.py`、`tests/test_capability_manifest_shared_contract.py`、
`tests/test_resource_manifest.py`），以及旧模块自身的实现事实。表达层（本仓库的模块划分、
内部类划分、命名、实现步骤）全部是本次决定。

### 0.1 「输出文案」不算表达层

本批两个模块都会产生**面向用户或模型的字符串**（提示词段落、标签、默认名）。这些字符串是
**外部可观察输出**，属契约事实，必须逐字保留。它们**不是**散文——散文筛子（docstring + 注释，
连续 ≥12 字符）不覆盖字符串字面量，这一区分是本批的判定前提。

---

## 1. 来源模块 → 本仓库模块映射

| 旧模块 | 字节 | 本仓库路径 | 类型 |
| --- | --: | --- | --- |
| `companion_v01/capability_adapters/manifest.py` | 13,024 | `src/shinku/capabilities/manifest.py` | 洁净室重写 |
| `companion_v01/capability_adapters/manifest_loader.py` | 135 | 折入上者（不单独建模块） | 转发壳消解 |
| `companion_v01/resource_manifest.py` | 30,352 | `src/shinku/resources/manifest.py` | 洁净室重写 |
| `companion_v01/capability_adapters/safety.py` | 211 | 折入下者（不单独建模块） | 转发壳消解 |
| `companion_v01/capability_safety.py` | 371 | `src/shinku/capabilities/safety.py` | 洁净室重写（传递依赖） |

**字节核对**（逐字节实测，见 `_c1_3_numbers.py` §2）：

| 口径 | 合计 | 覆盖 |
| --- | --: | --- |
| 计划登记的 C1-3 三个模块 | **43,511** | `manifest` 13,024 + `manifest_loader` 135 + `resource_manifest` 30,352 |
| 本批实际触碰的五个模块 | **44,093** | 上面三个 + `capability_adapters/safety.py` 211 + `capability_safety.py` 371 |

两个数都要写出来，因为它们答的不是同一个问题。计划值 **43,511** 只覆盖**计划点名的三个模块**；
另外两个是 `manifest.py` 的**传递依赖与它的转发壳**——`manifest.py` 不导入它们就无法加载，
所以它们必须随本批落地。**本批不把 44,093 说成"计划登记值"**：那是本次实测的汇总口径，
与计划条目不是同一个集合。

### 1.1 对照物判定（规矩六：先确认真实现，再谈相似度）

| 旧模块 | Akane 同路径 | `code_shared` 同路径 | 判定 |
| --- | --- | --- | --- |
| `capability_adapters/manifest.py` | 无副本 | 无副本 | **旧项目自有实现**（266 有效行），无双侧可对照的实现 |
| `capability_adapters/manifest_loader.py` | 184 B / 2 有效行，**转发壳** | 无副本 | 转发壳，不单独建模块 |
| `resource_manifest.py` | 137 B / 1 有效行，**转发壳** | 60,732 B / 1,342 有效行，**真实现** | 对照物取 `code_shared` |
| `capability_safety.py` | 无副本 | 371 B，**真实现**（仅 docstring 不同） | 对照物取 `code_shared`，属「复制 + 改头」 |
| `capability_adapters/safety.py` | 无副本 | 无副本 | 转发壳，指向 `capability_safety`，一并消解 |

### 1.2 「砍版本 vs 独立实现」判定（规矩七的应用）

`resource_manifest.py` 旧 520 行 / `code_shared` 1,505 行，整文件比值 **0.1057**——单看比值无法
区分「干净重写」与「把上游砍到 520 行」。因此改算**共有方法的函数体相似度**：

- 共有方法 26 个（旧项目 47 个方法 / `code_shared` 62 个方法，API 面覆盖率 26/47 = **55.3%**）；
- 函数体相似度 **均值 0.1476、中位 0.0984**，仅 1 个 ≥0.80（`refresh`，3 行 vs 3 行，
  两语句赋值 + 返回，任何实现都会逐字相同）；
- 旧项目独有方法 22 个、`code_shared` 独有方法 36 个 —— 两侧是**不同功能集**。

**结论：旧项目的 `resource_manifest.py` 是独立的小实现，不是上游的砍版本。**
证据脚本 `_c1_3_recon2.py`（输出存 `_c1_3_recon2.txt`）。

### 1.3 文件切分决定（表达层，Shinku 自有）

- **不沿用旧的扁平模块名。** `capability_adapters/` 是 Akane 时代的实现细节目录名，
  `resource_manifest` 是扁平命名。本仓库按**领域**组织：能力面进 `capabilities/`，
  视觉/音频资源面进 `resources/`。
- **两个转发壳直接消解，不建对应模块。**
  - `manifest_loader.py` 的全部内容是 `from .manifest import *` + `__all__` 转发。
    新仓库里 `load_manifest` 的直接导入路径就是 `shinku.capabilities.manifest`，
    保留一个只做转发的中转模块等于把「旧仓库的目录形状」固化进新仓库。
  - `capability_adapters/safety.py` 同理，只转发 `..capability_safety` 的三个名字。
  - 因此 C1-3 净增 **2 个实现模块 + 1 个传递依赖模块**，而不是 3 + 2 个模块。
- **`capability_safety.py` 一并处理。** 它是 `manifest.py` 的传递依赖，三个常量都是纯契约事实
  （frozenset / tuple / 编译后的正则），归属批次原本待定；本批必须落地它，否则 `manifest.py`
  无法导入。它也是 A2 列入重写清单的 3 个高相似模块之一（0.800，仅 docstring 差异 ⇒「复制 + 改头」），
  在本批一并解决，**并在台账里单独登记**（它是纯契约，与两个洁净室重写模块判据不同）。

---

## 2. `capabilities/safety.py` —— 能力面安全常量

### 2.1 契约事实（常量名与值）

| 常量 | 值 | 性质 |
| --- | --- | --- |
| `LOOPBACK_HOSTS` | `{"127.0.0.1", "localhost", "::1"}` | 普通集合；用于回环校验 |
| `MCP_SECRET_MARKERS` | `("api_key", "password", "secret", "token")` | 普通元组；用于拒绝过于泛化的密钥名 |
| `MCP_SAFE_TYPE_RE` | `re.compile(r"^[A-Za-z0-9_.-]{1,40}$")` | 编译后的正则；用于校验 provider type 字符集 |

顺序按 `__all__` 为 `["LOOPBACK_HOSTS", "MCP_SECRET_MARKERS", "MCP_SAFE_TYPE_RE"]`。

### 2.2 本仓库决定

- **不改名、不改值、不改类型。** 这三个是「常量表」，属契约事实；且 `LOOPBACK_HOSTS` 被
  旧项目测试按名断言（`assertIn("127.0.0.1", LOOPBACK_HOSTS)`）。
- **不引入 `re` 之外的任何依赖**，模块 import 必须是无副作用的（旧注释强调 import-safe）。
- **表达层：** 模块 docstring、常量分组注释、`__all__` 的书写顺序说明全部自撰。

---

## 3. `capabilities/manifest.py` —— 能力提供方清单的加载与校验

### 3.1 对外面（契约事实）

```
SCHEMA               = "capability_adapter/v1"
ALLOWED_ADAPTER_TYPES= frozenset({"mcp_stdio", "comfyui", "openai_compat_tts",
                                 "openai_compat_asr", "python_plugin"})
VISIBLE_IN_VALUES    = frozenset({"base", "web", "desktop", "qq"})
RISK_VALUES          = frozenset({"low", "medium", "high"})
CONFIRM_VALUES       = frozenset({"never", "first_time", "always"})
EFFECT_VALUES        = frozenset({"file_read", "file_write", "command_exec", "network_outbound",
                                  "browser_action", "media_generation", "state_mutation"})
PROVIDER_ID_RE       = re.compile(r"^[A-Za-z0-9_.-]+$")
SECRET_VALUE_RE      = re.compile(r"(?i)(\bbearer\s+\S+|\bsk-[A-Za-z0-9]|\bghp_[A-Za-z0-9]|"
                                  r"\bxox[baprs]-[A-Za-z0-9]|=|:)")

load_manifest(path: Path, *, source_layer: SourceLayer) -> CapabilityManifest | InvalidManifest
```

`load_manifest` 的第二个参数是**仅关键字**参数。

### 3.2 加载管线（行为，顺序即优先级）

```
读文件（utf-8-sig） → safe_load → 顶层必须是映射 → schema 必须是 SCHEMA
  → provider 必须是映射 → provider.id 非空且合 PROVIDER_ID_RE
  → provider.type 非空、合 MCP_SAFE_TYPE_RE、在白名单内
  → 解析 endpoint → 解析 secrets → 解析 tiers → 解析 capabilities
  → 构造 CapabilityManifest
```

**任一环节失败即返回 `InvalidManifest` 并终止，不抛异常**（唯一例外见 §3.6）。
上面的顺序决定了「同时有多个问题时先报哪一个」，属可观察行为。

### 3.3 成功路径的字段规则

| 字段 | 规则 |
| --- | --- |
| `schema` | 恒为常量 `SCHEMA`（不是原样回填） |
| `provider_id` / `provider_type` | 去空白后的字符串 |
| `display_name` | `provider.display_name` 去空白；空则回退为 `provider_id` |
| `endpoint` | 见 §3.4；`None` 或空串 ⇒ `None` |
| `health` | 见 §3.5；非映射 ⇒ `None`（**不报错**） |
| `tiers` | 见 §3.4；`None`/空串 ⇒ 空元组 |
| `capabilities` | 取自**顶层** `capabilities`（不是 `provider.capabilities`）；`None`/空串 ⇒ 空元组 |
| `secrets` | 见 §3.4；`None`/空串 ⇒ 空元组 |
| `source_path` / `source_layer` | 原样保留入参 |
| `raw` | 顶层映射原样保留（同一对象，不做深拷贝） |

### 3.4 各子结构的解析规则

**endpoint**
- `None` 或 `""` ⇒ `None`
- 非映射 ⇒ 报 `endpoint_must_be_mapping`
- `url` 去空白；`loopback_only = bool(raw.get("loopback_only"))`
- 若 `loopback_only` 为真：取 `urlparse(url).hostname` 去空白转小写，不在 `LOOPBACK_HOSTS` 中
  ⇒ 报 `endpoint_not_loopback`，`detail=f"host={host or '<empty>'}"`
- 通过 ⇒ `EndpointConfig(url, loopback_only, raw=原映射)`

**health**（宽松，永不报错）
- 非映射 ⇒ `None`
- `expect_status`：逐个尝试 `int()`，失败者跳过；结果为空则回退 `(200,)`
- `timeout_seconds`：`float(raw.get("timeout_seconds") or 3)`，转换失败回退 `3.0`
- `method`：去空白 → 转大写 → 空则 `"GET"`
- `path`：去空白
- `raw=原映射`

**tiers**
- `None` 或 `""` ⇒ `()`
- 非列表 ⇒ 报 `tiers_must_be_list`
- 逐项：非映射 ⇒ 报 `tier_must_be_mapping`
- `id` 去空白；**空或重复** ⇒ 报 `tier_id_not_unique`，`detail=tier_id or "<empty>"`
- `preset`：是映射则用之，否则 `{}`
- `label`：去空白，空则回退 `tier_id`

**secrets**
- `None` 或 `""` ⇒ `()`
- 非列表 ⇒ 报 `secrets_must_be_key_names`（detail：`"provider.secrets must be a list of key names"`）
- 逐项：非字符串 ⇒ 同类报错（detail：`"secret entries must be strings"`）
- 值去空白后，满足任一条件 ⇒ 同类报错（detail：`"secret entry looks like a value"`）：
  空串 / 长度 > 120 / 含任意空白 / 命中 `SECRET_VALUE_RE`
- 值小写后在 `MCP_SECRET_MARKERS` 中 ⇒ 同类报错（detail：`"secret entry is too generic"`）
- 注意 `SECRET_VALUE_RE` 含 `=` 与 `:` 两个裸字符分支，因此**任何含 `=` 或 `:` 的密钥名都会被拒**

**capabilities**（顶层列表）
- `None` 或 `""` ⇒ `()`；非列表 ⇒ 报 `capabilities_must_be_list`
- 逐项非映射 ⇒ `capability_must_be_mapping`
- `id` 去空白，空 ⇒ `missing_capability_id`
- `visible_in`：**只接受列表**（非列表 ⇒ 空元组）；逐项 `str(item or "").strip()` 后丢弃空串；
  任一值不在 `VISIBLE_IN_VALUES` ⇒ 报 `visible_in_invalid`，`detail=",".join(值)`
- `risk` / `confirm`：见 §3.5 的归一化
- `effects`：同 `visible_in` 的取串口径；任一值不在 `EFFECT_VALUES` ⇒ 报 `effects_invalid`，
  `detail=",".join(未识别值)`
- 顺序：**`visible_in` 的校验先于 `effects`**
- 通过后施加 §3.5 的效果提升，再构造 `CapabilityDescriptor`

**trigger**（宽松）：非映射 ⇒ `None`；`kind` 去空白，空 ⇒ `None`；否则 `TriggerConfig(kind, raw=原映射)`

**inputs` / `outputs`**（宽松，逐项过滤而非报错）
- 非列表 ⇒ `()`；非映射项 ⇒ **跳过**
- `name` 或 `kind` 去空白后为空 ⇒ **跳过**
- `max_bytes`：`None`/`""` ⇒ `None`；否则 `int(str(value).replace("_", ""))`，失败 ⇒ `None`
- `required = bool(raw.get("required"))`；`delivery` 去空白
- `raw=原映射`

### 3.5 归一化与效果提升（纯函数行为）

**`risk` / `confirm` 归一化**
- `risk`：`str(raw.get("risk") or "medium").strip().lower()`；不在 `RISK_VALUES` ⇒ `"medium"`
- `confirm`：`str(raw.get("confirm") or "first_time").strip().lower()`；不在 `CONFIRM_VALUES` ⇒ `"first_time"`

**效果提升**（在归一化之后，按下列顺序短路判断）

| 条件 | 结果 |
| --- | --- |
| `effects` 与 `{command_exec, browser_action}` 有交集 | `("high", "always")` |
| `effects` 与 `{file_write, network_outbound}` 有交集 **且** `risk == "low"` | `("medium", "first_time" if confirm == "never" else confirm)` |
| `risk == "high"` | `("high", "always")` |
| 其余 | 原样返回 |

### 3.6 失败原因码总表（契约事实）

`InvalidManifest(source_path, source_layer, reason, detail, provider_id)`。

| `reason` | `detail` | `provider_id` |
| --- | --- | --- |
| `yaml_parse_error` | 解析异常文本 | `""` |
| `read_error` | `OSError` 文本 | `""` |
| `manifest_must_be_mapping` | `top-level yaml must be a mapping` | `""` |
| `schema_mismatch` | `expected capability_adapter/v1` | `""` |
| `missing_provider` | `provider must be a mapping` | `""` |
| `missing_provider_id` | `provider.id is required` | `""` |
| `invalid_provider_id` | `provider.id contains unsupported characters` | `provider_id` |
| `missing_provider_type` | `provider.type is required` | `provider_id` |
| `provider_type_invalid` | `provider.type contains unsupported characters` | `provider_id` |
| `provider_type_not_allowed` | **`provider_type` 本身** | `provider_id` |
| `endpoint_must_be_mapping` | `provider.endpoint must be a mapping` | `provider_id` |
| `endpoint_not_loopback` | `host=<host>`（空主机时为 `host=<empty>`） | `provider_id` |
| `secrets_must_be_key_names` | 见 §3.4 的三种 detail | `provider_id` |
| `tiers_must_be_list` | `provider.tiers must be a list` | `provider_id` |
| `tier_must_be_mapping` | `tier entry must be a mapping` | `provider_id` |
| `tier_id_not_unique` | `tier_id` 或 `<empty>` | `provider_id` |
| `capabilities_must_be_list` | `capabilities must be a list` | `provider_id` |
| `capability_must_be_mapping` | `capability entry must be a mapping` | `provider_id` |
| `missing_capability_id` | `capability.id is required` | `provider_id` |
| `visible_in_invalid` | 逗号连接的非法值 | `provider_id` |
| `effects_invalid` | 逗号连接的未识别值 | `provider_id` |

**已知边界（本批不改行为）：** 读文件时若字节不是合法 UTF-8，`UnicodeDecodeError` 会**逃逸**
`load_manifest`（旧实现只捕 `yaml.YAMLError` 与 `OSError`）。本批保持等价行为并在测试里钉住，
不做「顺手修好」——修它属于行为变更，需单独决定。

### 3.7 本仓库决定（表达层）

- **控制流改为「私有异常短路 + 边界统一转换」。** 旧实现让每个解析函数返回
  `X | InvalidManifest` 哨兵值，调用方逐个 `isinstance` 检查；新实现让解析层直接抛私有异常
  `_ManifestRejected`，只有 `load_manifest` 一处 `except` 并转成 `InvalidManifest`。
  可观察行为完全一致（首个失败即终止），但分支组织与旧实现不同。
- **不保留任何旧的私有函数名。** 旧名 `_invalid` / `_parse_*` / `_string_tuple` /
  `_risk_and_confirm` / `_promote_for_effects` 在新模块中零命中，由测试 `C1_3BoundaryTests` 钉住。
- **类型契约来自 C1-1。** `CapabilityManifest`、`CapabilityDescriptor`、`EndpointConfig`、
  `HealthConfig`、`TierConfig`、`TriggerConfig`、`CapabilityIOSlot`、`InvalidManifest`、
  `SourceLayer`、`RiskLevel`、`ConfirmPolicy` 全部从 `shinku.contracts.capability` 导入，
  **本模块不再定义任何数据类型**。
- **安全常量来自 `shinku.capabilities.safety`**（本批新建），不从旧路径转发。
- **`manifest_loader` 不建模块。** 新仓库的直接导入路径是
  `shinku.capabilities.manifest.load_manifest`。
- **新依赖：`pyyaml`。** 本模块是本仓库第一个真正用到 YAML 的模块，按仓库既定口径
  「每接入一个模块，只加它真正用到的那一个依赖」加入 `pyproject.toml`。

---

## 4. `resources/manifest.py` —— 视觉/音频资源清单

### 4.1 对外面（契约事实）

```
IMAGE_EXTS            = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS            = {".mp3", ".ogg", ".wav", ".m4a", ".flac"}
NOTE_FILENAMES        = ("ai.md", "ai.txt", "prompt.md", "prompt.txt", "说明.md", "说明.txt",
                         "readme.md", "readme.txt", "notes.md", "notes.txt")
SIDECAR_NOTE_SUFFIXES = (".md", ".txt", ".note.md", ".note.txt")
META_NOTE_KEYS        = ("notes", "note", "prompt", "ai_hint", "ai_prompt", "usage")
PROMPT_NOTE_LIMIT     = 240
EMOTION_ALIASES       = {"normal": "normal", "shy": "shy", "smug": "smug", "sumg": "smug", "cry": "cry"}
EMOTION_FALLBACK_CANDIDATES = { ... 21 条，见旧实现逐字表 ... }
BACKGROUND_ALIASES    = { ... 23 条中英对照 ... }
BACKGROUND_LABELS     = {"morning": "清晨", "afternoon": "午后", "evening": "黄昏", "night": "深夜"}
EMOTION_LABELS        = {"normal": "平静", "shy": "害羞", "smug": "得意", "cry": "哭哭"}
BACKGROUND_PRIORITY   = {"evening": 0, "morning": 1, "afternoon": 2, "night": 3}

class ResourceManifest:
    def __init__(self, assets_dir: Path, public_prefix: str = "/assets", *,
                 emotion_aliases: dict[str, list[str]] | None = None)
```

**`emotion_aliases` 是仅关键字参数。**

**`__all__`**（契约事实，顺序即字面量）：
`["AUDIO_EXTS", "BACKGROUND_ALIASES", "EMOTION_ALIASES", "IMAGE_EXTS", "PROMPT_NOTE_LIMIT", "ResourceManifest"]`
——注意它**只导出这 6 个名字**，`BACKGROUND_LABELS` / `EMOTION_LABELS` / `BACKGROUND_PRIORITY` /
`NOTE_FILENAMES` / `SIDECAR_NOTE_SUFFIXES` / `META_NOTE_KEYS` / `EMOTION_FALLBACK_CANDIDATES`
虽为模块级名，但**不在 `__all__` 里**（仍可按下划线外的名字直接导入）。

### 4.2 公开方法（12 个被外部真实调用，签名即契约）

| 方法 | 签名要点 |
| --- | --- |
| `refresh()` | 无参，重建索引并返回清单字典 |
| `get_manifest()` | 无参，无缓存则先 `refresh()` |
| `build_runtime_manifest(...)` | 仅关键字 `extra_bgm_tracks` / `extra_scene_groups` / `extra_character_outfits`，默认 `None` |
| `build_prompt_context(...)` | 同上三个仅关键字参数 → `str` |
| `build_character_prompt_context(...)` | 仅关键字 `extra_character_outfits` → `str` |
| `build_emotion_prompt_context()` | 无参 → `str` |
| `normalize_emotion_id(value, *, preferred_outfit="")` | → `str` |
| `normalize_emotion_output(result)` | → `dict` |
| `normalize_visual_output(result, *, 三个 extra)` | → `dict` |
| `resolve_visual_bundle(result, **kwargs)` | → `dict`（8 个键，见 §4.5） |
| `describe_visual_state(result, **kwargs)` | → `str`，`；` 连接 5 段 |
| `describe_character_visual_state(result, *, extra_character_outfits=None)` | → `str` |

**另外 35 个下划线方法属实现细节**，本仓库不保证同名同形（表达层）。

### 4.3 清单字典形状（契约事实）

```
{
  "schema_version": 2,
  "scenes":     {"majors": [major, ...]},
  "characters": {"outfits": [outfit, ...]},
  "defaults":   {"major": str, "minor": str, "background": str,
                 "bgm": str, "outfit": str, "emotion": str}
}
major   = {id, name, aliases, description, notes, minors:[minor,...], default_minor}
minor   = {id, name, aliases, description, notes, default_background,
           default_bgm, backgrounds:[asset,...], bgm_tracks:[asset,...]}
outfit  = {id, name, aliases, description, notes, default_emotion,
           allowed_emotions:[str,...], emotions:[asset,...]}
asset   = {id, name, aliases, description, notes, path}
```

### 4.4 扫描规则（行为）

**统一的小工具口径**（这些是行为，不是命名）：

| 行为 | 口径 |
| --- | --- |
| 读 JSON 元数据 | `utf-8` 读 + `json.loads`；失败或非 dict ⇒ `{}` |
| 读文本备注 | `utf-8` 读 + `.strip()`；仅在 `is_file()` 时；失败 ⇒ `""` |
| 归一化 key | `str(value or "").strip().lower()`，`-`→`_`，空格→`_` |
| 取字符串列表 | 列表 ⇒ 逐项 `str().strip()` 留非空；字符串 ⇒ 按 `,` 切分去空；其余 ⇒ `[]` |
| 压缩文本 | `" ".join(str(value or "").split())`；超 240 字符则取前 239 字符 `rstrip()` + `…` |
| 拼接文本 | 逐个压缩，非空且不重复才收集，空格连接 |
| 忽略目录名 | `name.lower() in {"backup", "backups", "备份", "__pycache__"}` |
| 别名去重 | `dict.fromkeys(取字符串列表(meta["aliases"]))`，保序 |
| 条目匹配 | 条目空或值为 `None`/`""` ⇒ `False`；否则把值 `str().strip().lower()`，与条目的 `id` / `name` / `aliases` 逐项小写比较 |

**场景扫描（`<assets>/scenes`）**
- 目录不存在 ⇒ 空
- 主场景目录：`sorted(是目录 且 未被忽略, key=name)`；元数据取该目录 `meta.json`
- 该主目录下的**子目录**（同样排序、过滤忽略）：子目录内 `IMAGE_EXTS` 文件各成一条背景资产；
  有背景才成为一个 minor，minor id 即子目录名
- 该主目录下的**直接图片文件**：按 `legacy_split(stem)` 得到 `(minor 名, 背景 id)` 分组；
  分组内背景资产带 `explicit_id`；minor 的元数据取 `<主目录>/<minor名>.meta.json`
- 有 minor 才产出该主场景：`id = 元数据 id 或 目录名`，`name = 元数据 name 或 id`，
  `notes = 主目录的目录备注`，`minors` 按 `default_minor` 排序（默认项在首，其后按 id）
- **`legacy_split(stem)`**：无 `_` ⇒ `("default", stem)`；否则从右侧切一次；
  后缀的归一化 key 在 `BACKGROUND_ALIASES` 中 ⇒ `(前缀, canon(后缀))`；否则 `("default", stem)`

**扁平背景（`<assets>/backgrounds`）**
- 目录不存在或没有图片 ⇒ 空
- 有图片 ⇒ 单个主场景：`id="default"`、`name="默认"`、`aliases=[]`、`description=""`、`notes=""`，
  只有一个 minor：`("default", {}, 背景列表)`

**主场景合并**：`scenes(嵌套) + scenes(扁平)` 后按 id 合并；结果为空 ⇒ 用内建默认场景

**服装扫描（`<assets>/characters`）**
- 目录不存在 ⇒ 空；否则每个子目录（排序、过滤忽略）成一个 outfit，元数据取 `meta.json`
- 该目录内 `IMAGE_EXTS` 文件各成一条表情资产；**没有表情的目录不产出 outfit**
- `allowed_emotions` 非空时过滤：保留「与允许项匹配」或「其 id 等于 `canon(允许项)`」的表情；
  **过滤结果为空则回退为未过滤的全量**
- `default_emotion = canon(元数据 default_emotion)`；表情排序：默认项在前，其余按 id
- `allowed_emotions` 字段 = 排序后的表情 id 列表

**BGM 挂载（`<assets>/bgm`，递归）**
- 目录不存在 ⇒ 直接返回（不挂任何 BGM）
- 每个音频文件按相对路径定位：路径深度 ≥3 时取 `(倒数第 3 段, 倒数第 2 段)`，否则 `("default","default")`
- 主场景/minor 取自己 key 下的曲目；**为空则回退为该目录下 `("default","default")` 的曲目**
- 每个 minor 的 `bgm_tracks` 先按 id 合并，再按 `(BACKGROUND_PRIORITY.get(id, 10), id)` 排序

**背景与表情的排序**：minor 的背景先合并，再按 `(BACKGROUND_PRIORITY.get(id, 10), id)` 排序

**资产条目的 id / name / 别名口径**

| 类别 | id | name 回退链 |
| --- | --- | --- |
| 背景 | `元数据 id` → `explicit_id（legacy 后缀）` → `canon_background(stem)` | `元数据 name` → `BACKGROUND_LABELS[id]` → `id` |
| 表情 | `canon_emotion(元数据 id 或 stem)` | `元数据 name` → `EMOTION_LABELS[id]` → `id` |
| BGM | `元数据 id 或 stem` | `元数据 name` → **`EMOTION_LABELS[id]`** → `id` |

**BGM 走表情标签表是来源事实，不是笔误。** 旧实现的回退链写的是
「`kind == "background"` 用场景标签表，**否则**用表情标签表」，BGM 落在「否则」一侧。
后果是：一首 id 恰好在 `EMOTION_LABELS` 里的 BGM（例如 `shy.ogg`）会被显示成「害羞」。
本批**照录不改**——修它属于行为变更，需单独决定；改动会在测试
`test_the_emotion_label_table_also_applies_to_tracks` 处立刻变红。

别名：`元数据 aliases`，若 `stem` 既不在别名里、又不等于 id，则追加 `stem`。
备注：`META_NOTE_KEYS` 各键值 + sidecar 备注（`SIDECAR_NOTE_SUFFIXES` 中第一个有内容的文件）
一起拼接压缩。`description` 取压缩后的元数据 `description`。
`path`：见 §4.6。

**目录备注**：`NOTE_FILENAMES` 中第一个有内容的文件，压缩后返回；都没有 ⇒ `""`

**`emotion_aliases` 构造**：对入参逐项 `_key(k)` 归一化，值经 `取字符串列表` 去重；
**key 归一化后为空、或值列表为空的项目被丢弃**

**`public_prefix` 归一化**：`str(value or "/assets").strip()`，`\`→`/`，两侧去 `/` 后前面补一个 `/`；
结果为空 ⇒ `/assets`

### 4.5 运行时合并与默认值（行为）

**`build_runtime_manifest`**
1. `deepcopy(get_manifest())`
2. 合并 `extra_scene_groups`
3. 合并 `extra_character_outfits`
4. 若 `extra_bgm_tracks` 为真：对**每个主场景的每个 minor**，
   `bgm_tracks = 合并(minor.bgm_tracks + [其中是 dict 的 extra 项])`
5. 重算 `defaults`，返回

**合并口径**
- 场景组/服装：逐项要求是 dict 且有 `id`；已存在同 id（按条目匹配）则 `update`（**场景组排除 `minors`**、
  **服装排除 `emotions`**）并把两者按 id 合并；不存在则 `deepcopy` 追加；最后整体按 id 合并一次
- 通用条目合并：按 `str(id)` 去重；**先出现者为基底**，后者 `update`（**排除 `aliases`**），
  `aliases` 取并集去重

**默认值重算**：`major = 第 1 个主场景`、`minor = 第 1 个 minor`、`outfit = 第 1 套服装`；
`background = minor 的第 1 条背景`；`bgm = minor 的第 1 条曲目（无则 ""）`；
`emotion = outfit 的第 1 个表情`

**`normalize_visual_output`**（顺序即优先级）
1. 构造运行时清单，取 `defaults`
2. `major = 匹配(scene.major)`，否则 `匹配(defaults.major)`
3. `minor = 匹配(major, scene.minor)`
4. 若 `minor` 为空：先在 `major` 的**所有** minor 里跨场景找 `scene.background`（找到则同时确定 minor 与背景）；
   找不到则 `minor = 匹配(major, defaults.minor) 或 major 的第 1 个 minor`，背景置空
5. 否则 `background = 匹配(minor, scene.background)`
6. `background = background or 匹配(minor, defaults.background) or minor 的第 1 条背景`
7. `outfit = 匹配(character.outfit) or 匹配(defaults.outfit) or 第 1 套服装`
8. `emotion = 匹配(outfit, value.emotion) or 匹配(outfit, defaults.emotion) or outfit 的第 1 个表情`
9. `bgm = 匹配(minor, scene.bgm) or 匹配(minor, background.id) or 匹配(minor, defaults.bgm)`
10. 覆写 `emotion`（id）、`character = {"outfit": id}`、`scene = {major: id, minor: id, background: id, bgm: id 或 ""}`

**`normalize_emotion_id(value, preferred_outfit="")`**
- `outfit = 匹配(preferred_outfit) or 匹配(defaults.outfit) or 第 1 套服装`
- `requested = canon_emotion(value) or defaults.emotion`
- 先在该 outfit 找；找不到则遍历**所有** outfit 找
- 返回 `str((找到的 或 该 outfit 第 1 个表情)["id"])`

**表情候选阶梯**：`[canon_emotion(raw)] + EMOTION_FALLBACK_CANDIDATES.get(key(raw), []) +
self.emotion_aliases.get(key(raw), [])`，去重去空；逐个按条目匹配或 id 相等查找

**`resolve_visual_bundle`**：先把 `result` 做 JSON 往返（`ensure_ascii=False`）——即**深拷贝 + 剔除不可序列化对象**
——再走 `normalize_visual_output`；随后用**只取 kwargs 中出现的三个 extra 键**重建清单；
返回 8 键：

```
{"normalized", "manifest", "major", "minor", "background", "outfit", "emotion", "bgm"}
```
后 4 个查找均在 `major`/`minor`/`outfit` 为空时返回 `None`。

**`describe_visual_state`**：`；` 连接 5 段，依次是
`地点: <主/次标签 或 normalized.scene.major>`、`背景: <标签 或 normalized.scene.background>`、
`服装: <标签 或 normalized.character.outfit>`、`表情: <标签 或 normalized.emotion>`、
`BGM: <标签 或 normalized.scene.bgm 或 "未设置">`

**`describe_character_visual_state`**：`；` 连接；前两段是服装与表情；
若 outfit 有表情，追加第 3 段 `当前服装可用表情: ` + 标签逗号连接

### 4.6 公开路径（`path` 字段）

- `relative = path.resolve().relative_to(assets_dir.resolve()).as_posix()`
- 成功 ⇒ `f"{public_prefix}/{relative}"`；`OSError`/`ValueError`（不在 assets 目录下）⇒ `""`

### 4.7 提示词构造（输出文案 = 契约事实）

**`build_prompt_context`** 依次拼装（有内容才追加该段）：

| 段 | 文案 |
| --- | --- |
| 全局备注 | 首行 `资源说明：`，其后每行 `- {note}` |
| 场景规则 | `场景输出规则：scene.major/minor/background 使用清单列出的名称或 id，不要把未列出的文件名自行拆成新场景。` |
| 场景标题 | `可用场景与背景：` |
| 每场景 | `- {主标签}/{次标签} -> 背景: {背景标签逗号连接 或 (无)}` |
| 服装标题 | `可用服装与表情：` |
| 每服装 | `- {服装标签} -> 表情: {表情标签逗号连接 或 (无)}` |
| BGM 标题 | `可用 BGM：`（仅当存在曲目） |
| 每 BGM | `- {主标签}/{次标签} -> BGM: {曲目标签逗号连接}` |

用 `\n` 连接；整体为空 ⇒ `当前没有额外的视觉资源。`。
提示词里**不含** `schema_version`、路径或 id 之外的信息。

**`build_character_prompt_context`** 固定两行表头：
`桌宠模式只渲染角色服装立绘和表情，不渲染场景、背景或 BGM。` 与
`输出时优先沿用当前 character.outfit，只在同一套服装下选择可用 emotion；确实需要换衣服时才切换 character.outfit。`
其后每套服装一行 `- {标签} -> 表情: {标签逗号连接 或 (无)}`；
行数不超过 2 ⇒ `当前没有额外的角色视觉资源。`

**`build_emotion_prompt_context`** 固定两行表头：
`emotion 与桌宠共用当前角色包的表情图片变量。` 与
`即使当前客户端不渲染立绘，也必须直接使用图片文件名去掉扩展名后的稳定 id；不要编造或改写 emotion。`
其后每套服装一行 `- {outfit id}: {表情 id 逗号连接}`（无 id 的服装跳过）

**`normalize_emotion_output`**：浅拷贝入参；`character` 是 dict 则取之否则 `{}`；
`emotion` 覆写为 `normalize_emotion_id(value.emotion, preferred_outfit=str(character.get("outfit") or ""))`

### 4.8 本仓库决定（表达层）

- **内部拆成「索引构建」与「对外门面」两个类。** 旧实现是一个 47 方法的 god-object，
  目录扫描（`_scan*`）、资产条目构造（`_background`/`_emotion`/`_audio`/`_asset_entry`/`_minor`）、
  合并、查找、提示词构造全部挂在同一个类上。新实现把**「文件系统 → 清单字典」**这一步
  收进一个独立的内部类（模块私有），`ResourceManifest` 只保留缓存、合并、归一化、查找与提示词构造。
  可观察行为不变，但职责边界与旧实现不同。
- **合并类操作改为模块级纯函数。** 旧实现把 `_merge_entries` / `_merge_scene_groups` /
  `_merge_outfits` 做成实例方法，但它们只操作字典、不读实例状态；新实现改为无副作用的模块级函数。
- **不保留任何旧的私有名字**（模块级 11 个 + 类内 35 个，共 46 个），全部改用自有命名。
- **常量名与值不改**（契约事实）；`__all__` 的内容与顺序不改。
- **提示词文案不改**（外部可观察输出）。
- **全类 47 个方法里只实现「外部真实调用的 12 个」+ 必要的内部支撑。** 旧实现里有若干方法
  没有任何外部文件调用（`describe_character_visual_state` 也属此类——它只被旧测试调用）。
  本仓库**保留全部 12 个被外部调用的公开方法**，其余旧方法与旧行为（见 §5）不迁移。

---

## 5. 刻意不做的行为

1. **不做装配。** 本批不把能力清单接入任何注册表，也不把资源清单接进 `api/app.py`
   ——装配属 C4/C6。
2. **不迁移旧项目 `docs/fixtures/capability_adapter_m1/*.yaml`。** 夹具属旧仓库的测试资产；
   本仓库的测试在自己的临时目录里构造 YAML 与资源目录树。
3. **不迁移旧测试文本。** `tests/test_capability_adapter_manifest_loader.py`、
   `tests/test_resource_manifest.py` 里的断言可作为行为依据被读取，
   但测试代码本身必须新写。
4. **不修旧实现的行为缺陷**（§3.6 的 `UnicodeDecodeError` 逃逸只是其中一例）。
   本批口径是**行为等价迁移 + 表达层独立**，不是行为修正；修正需单独开条。
5. **不碰 `capability_adapters/protocol.py`、`registry.py`、`comfyui.py` 等其他能力模块**
   ——它们不在本批来源范围内。

---

## 6. 本批的验收口径

沿用 C1-2 定下的三条洁净室重写必跑检查，并按本批特点收窄：

| # | 检查 | 判据 |
| --- | --- | --- |
| 1 | **散文重叠** | docstring + 注释，连续 ≥12 字符 = **0**（脚本 `_c1_2_prose_overlap.py` 复用） |
| 2 | **旧项目内部命名命中** | 模块级 11 + 类内 35 + 3.7 的 11 个私有名，在本批新增文件里 = **0** |
| 3 | **契约字段表达式逐条对照** | §3.3／§3.4／§3.5／§4.3～§4.7 的每一条都能在实现里找到对应语句；差异必须写进批次记录 |

外加：

| # | 检查 | 判据 |
| --- | --- | --- |
| 4 | 禁止导入 / 弃用键读取 | **0**（与 B1／C1-1／C1-2 基线对照，非新增） |
| 5 | `src/` 内来源路径字样 | 新增命中逐条判读，**本批新增文件单列断言为 0** |
| 6 | 干净环境全量回归 | 新 venv 重装（含新依赖 `pyyaml`）后 `0 failed / 0 error / 0 skipped` |

**输出文案不经散文筛子**（§0.1）——提示词段落与标签是外部可观察输出，属契约事实。

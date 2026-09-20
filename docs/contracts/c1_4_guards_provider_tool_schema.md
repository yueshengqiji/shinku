# C1-4 行为契约：公共守卫、证据门、供应商词汇与原生工具 schema（洁净室重写）

**制定日期：** 2026-09-18
**处理类型：** 洁净室重写（来源记录 §5.9 路径 A 的第二类）
**依据：** 计划 §5「C1 批次划分」与本批公开契约记录。

---

## 0. 本契约的性质

C1-4 的四个来源模块都**含逻辑、分支、算法或数据处理**，因此按路径 A 属**洁净室重写**：

> 依行为契约独立实现。契约事实可迁移，**表达层（docstring、注释、内部组织与文件切分、
> 内部命名风格、实现步骤与算法写法）必须 Shinku 单独撰写**。

本文件**只记录外部可观察行为**，作为新实现的唯一需求输入。它记录四件事：

1. **必须成立的行为**——新实现与旧行为等价的部分；
2. **契约事实**——常量名/值、字段名/类型/默认值、签名与参数顺序、错误原因码、输出文案。这些是
   「接口信息」（路径 A 允许按事实迁移），本仓库照录，但**表达方式必须自撰**；
3. **本仓库自有的决定**——文件切分、内部组织、命名、控制流组织等表达层选择；
4. **刻意不做的行为**——明确排除的部分。

**关于旧实现的说明：** 本契约的行为面来自对旧模块的读取与既有行为测试
（`tests/test_public_guard.py`、`tests/test_evidence_guard.py`、
`tests/test_shared_provider_config.py`、`tests/test_native_tool_schema.py`），
以及旧模块自身的实现事实。表达层（本仓库的模块划分、内部组织、命名、实现步骤）全部是本次决定。

### 0.1 「哪些字符串算契约事实」

本批四个模块都产出**字符串**，但它们的性质不同，判定前必须分开：

| 类别 | 例子 | 性质 | 处置 |
| --- | --- | --- | --- |
| **面向用户的输出文案** | `CHECK_AGAIN_LINE`、`busy_message`/`daily_limit_message` 默认值、7 条 provider `description` | 外部可观察输出 | **逐字保留** |
| **线路/协议字面量** | `SUPPORTED_PROTOCOLS`、`DEFAULT_PROVIDER_ID`、payload 键 `baseUrl`/`apiKeyRequired`、tool spec 形状键 `type`/`function`/`parameters`、旧信封标记 `tool_call`/`{"type"` | 兼容性约定 | **逐字保留** |
| **实现步骤（正则模式、切分表达式、常量表内容）** | `_UNSUPPORTED_CLAIM_RE` 的模式、`_SENTENCE_RE`、`_LEGACY_MARKERS` 的成员 | **实现写法** | 由 Shinku 重写；其**行为效果**用测试钉住 |

第三类是本批最容易混的一类。正则是**实现**，不是契约；契约只承诺「哪类句子会被删、哪类会保留」，
并在 §3.3 / §5.3 用句子清单把这个边界写死。新实现可以用完全不同的写法（词表驱动、状态判定等），
只要句子清单上的行为逐条成立。

**散文筛子（docstring + 注释，连续 ≥12 字符）不覆盖字符串字面量。** 输出文案与协议字面量
是「被要求输出的内容」，不是散文，**不进散文口径**。

---

## 1. 来源模块 → 本仓库模块映射

| 旧模块 | 字节 | 本仓库路径 | 类型 |
| --- | --: | --- | --- |
| `companion_v01/public_guard.py` | 3,024 | `src/shinku/guards/public_think.py` | 洁净室重写 |
| `companion_v01/evidence_guard.py` | 5,693 | `src/shinku/guards/evidence.py` | 洁净室重写 |
| `companion_v01/provider_config.py` | 5,525 | `src/shinku/providers/config.py` | 洁净室重写 |
| `companion_v01/native_tool_schema.py` | 3,409 | `src/shinku/tools/native_schema.py` | 洁净室重写 |

**字节核对**（逐字节实测，见 `_c1_4_numbers.py` §1）：

| 口径 | 合计 | 覆盖 |
| --- | --: | --- |
| 计划登记的 C1-4 四个模块 | **17,651** | `public_guard` 3,024 + `evidence_guard` 5,693 + `provider_config` 5,525 + `native_tool_schema` 3,409 |

本批**只有这一个口径**——四个模块都是计划点名的，没有随行的传递依赖，也没有转发壳要消解。
计划值与本批实测值同为 **17,651** 字节。

### 1.1 对照物判定（规矩六：先确认真实现，再谈相似度）

| 旧模块 | legacy 同路径 | `code_shared` 同路径 | C1 取证分级 | 对照物取 |
| --- | --- | --- | --- | --- |
| `public_guard.py` | 183 B / 3 有效行，**转发壳**（`from code_shared.public_guard import *`） | 3,423 B / 85 有效行，**真实现** | T2 压缩改写 | `code_shared` |
| `evidence_guard.py` | 5,693 B，**与旧项目字节完全相同** | **无副本** | T3 直连同源（A2 盲区） | legacy 同路径 |
| `provider_config.py` | **无副本** | 6,848 B / 172 有效行，**真实现** | T2 压缩改写 | `code_shared` |
| `native_tool_schema.py` | 1,015 B / 30 有效行，**转发壳** | 4,430 B / 105 有效行，**真实现** | **T4 低相似待核** | `code_shared` |

**规矩六再次命中。** `public_guard.py` 与 `native_tool_schema.py` 在 legacy 侧都是转发壳
（只有几行 `from code_shared.xxx import *`），若拿 legacy 同路径当对照物，整文件比值会给出
0.0652 / 0.2941 这样的**假低值**，方向完全反。对照物一律取真实现。

### 1.2 「砍版本 vs 独立实现」判定（规矩七的应用）

整文件比值只能**证伪复制**，不能**证实独立**。对四个模块都算了**共有函数体的相似度**
（`_c1_4_numbers.py` §2）：

| 模块 | 整文件比值 | 共有方法 | 函数体相似度均值 | 中位 | ≥0.80 的个数 | 判定 |
| --- | --: | --: | --: | --: | --: | --- |
| `public_guard.py` | 0.6842 | 4 / 6 | 0.7218 | 0.9000 | **2**（`release` 1.0、`snapshot` 0.9） | 同一实现的**结构压缩** |
| `evidence_guard.py` | 1.0000 | 5 / 5 | 1.0000 | 1.0000 | **5** | **逐字保留，一字未改** |
| `provider_config.py` | 0.5354 | 8 / 8 | 0.6830 | 1.0000 | **5**（含 3 个 1.0） | **照抄 + 局部压缩** |
| `native_tool_schema.py` | 0.4775 | 3 / 6 | 0.2879 | 0.3636 | **0** | **独立的小实现，非上游砍版本** |

逐条说明：

- **`public_guard.py`（T2）**：旧实现把上游的 `_current_day_key()` + `_rollover_day_if_needed()`
  合并成 `_day()`/`_reset_if_new_day()`，并把 `GuardDecision(...)` 压成位置参数写法。
  `try_acquire` 的函数体相似度只有 0.2286（旧 11 行 vs 上游 24 行）——**是真实结构改动，
  但仍是同一份实现的压缩，不构成独立创作**，故重写。
- **`evidence_guard.py`（T3）**：与 legacy 侧**逐字节相同**（sha 均为 `6434e377080a`），
  5 个函数体全部 1.0000。它不经由 `code_shared`，因此 A2 的两轮取证（引用扫描 + 字节相同性）
  之间存在夹角而漏过它；C1 边界对账把它列为「T3 直连同源（A2 盲区）」。
  **这是本批来源最不洁净的一个模块**，重写时表达层与实现写法都必须彻底换掉。
- **`provider_config.py`（T2）**：8 个函数全部在两侧出现，其中 `canonical_provider_id`、
  `normalize_api_protocol`、`preset_index`、`provider_presets_payload` 的函数体**逐字相同**。
  这是「照抄 + 把几个归一化函数压缩」，故重写。
- **`native_tool_schema.py`（T4 → 本批核清）**：C1 对账把它列为 T4「低相似（<0.50）需人工核验」。
  本批完成核验：共有 3 个方法、函数体相似度均值 0.2879、中位 0.3636、**≥0.80 的一个都没有**；
  旧项目 6 个方法、上游 6 个方法，**两侧各有 3 个独有方法**（旧项目独有 `_handler_schema`、
  `_metadata_description`、`_metadata_parameters`；上游独有另外 3 个）——**两侧是不同功能集**。
  结论与 C1-3 的 `resource_manifest` 同型：**这是独立的小实现，不是上游的砍版本。**
  但按路径 A，只要含逻辑就必须有契约文档与独立测试；且它的 T4 身份意味着**来源未经取证**，
  因此**仍然照洁净室重写处理**，不按「已独立」放行。

**因此四个模块全部归入洁净室重写，无一按「证明独立」豁免。** `native_tool_schema.py` 的
核验结论（是独立实现）不改变它的处理类型，只改变它的风险评级：它是四个里**唯一无「必须重写」
硬证据**的一个，重写的目标是**补齐契约与测试**，而非清偿同源债务。

### 1.3 文件切分决定（表达层，Shinku 自有）

- **不沿用旧的扁平模块名。** `public_guard`、`evidence_guard`、`provider_config`、
  `native_tool_schema` 都是旧仓库的扁平命名。本仓库按**领域**组织：

  | 领域 | 新包/模块 | 收什么 |
  | --- | --- | --- |
  | 公共发言的准入与放行控制 | `src/shinku/guards/`（新建） | `public_think.py`、`evidence.py` |
  | 供应商词汇与端点归一化 | `src/shinku/providers/`（新建） | `config.py` |
  | 工具面 | `src/shinku/tools/`（已存在） | `native_schema.py` |

- **两个「门」合进一个包。** `public_guard` 管的是「一次公开发言**能不能进**」（并发配额、
  每日配额），`evidence_guard` 管的是「一次公开发言**发出去之前**要不要削」（无据句式、
  供应方内部推测）。一进一出，同属「对公开发言的确定性控制面」，合进 `guards/`。
  模块名去掉 `guard` 后缀（包名已表达），保留各自的核心词：`public_think`、`evidence`。
- **`providers/` 是为后续批次预留的。** 已列入 C 阶段重写清单的 `huggingface_provider.py`
  属同一领域，将来落进 `providers/`；本批只放 `config.py`（词汇表与端点归一化）。
- **`native_tool_schema.py` 去掉 `tool` 前缀**，改置 `tools/native_schema.py`——包名已表达
  「工具」，模块名保留「原生 schema」这个区分点。
- **不新建 `__init__.py` 之外的中转模块，不做转发壳。**

---

## 2. `guards/public_think.py` —— 公开发言的准入与配额

### 2.1 对外面（契约事实）

```
@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    acquired: bool
    reason: str
    message: str = ""

class PublicThinkGuard:
    def __init__(self, *, enabled: bool, max_concurrent_thinks: int,
                 daily_think_limit: int, busy_message: str,
                 daily_limit_message: str, timezone_name: str = "Asia/Shanghai") -> None: ...
    def try_acquire(self) -> GuardDecision: ...
    def release(self) -> None: ...
    def snapshot(self) -> dict[str, int | str | bool]: ...

__all__ = ["GuardDecision", "PublicThinkGuard"]
```

- 构造参数**全部为关键字参数**（`*` 之后的六个），参数顺序如上。
- `reason` 的值域是四个字面量：`"disabled"`、`"ok"`、`"busy"`、`"daily_limit"`。
- 两个默认文案（字面量，逐字保留）：
  - `busy_message` 空值时 → `"当前体验人数较多，请稍后再试。"`
  - `daily_limit_message` 空值时 → `"今日体验名额已满，明天再来看看吧。"`
- `snapshot()` 的键集合固定为六个：`enabled`、`max_concurrent_thinks`、`daily_think_limit`、
  `active_thinks`、`used_today`、`day_key`。

### 2.2 必须成立的行为

1. **参数归一化**：`max_concurrent_thinks` 与 `daily_think_limit` 都按 `max(0, int(x))` 处理
   （负数归 0）；两个 message 按 `str(x or 默认).strip() or 默认` 处理（空串、纯空白、None 都取默认）。
2. **禁用短路**：`enabled=False` 时 `try_acquire()` 返回
   `GuardDecision(allowed=True, acquired=False, reason="disabled")`（**不占配额**）；
   `release()` 直接返回（不改计数）。
3. **判定优先级**：启用时**先判每日限额，再判并发限额**。两者都触发时返回 `daily_limit`。
4. **每日限额**：`daily_think_limit` 非 0 且 `used_today >= 限额` → 返回
   `GuardDecision(False, False, "daily_limit", daily_limit_message)`。
   **`release()` 不回退 `used_today`**——用过一次就计一次，释放并发名额不减当日已用。
5. **并发限额**：`max_concurrent_thinks` 非 0 且 `active_thinks >= 限额` → 返回
   `GuardDecision(False, False, "busy", busy_message)`。
6. **放行**：以上都不触发时 `active_thinks += 1`、`used_today += 1`，返回
   `GuardDecision(True, True, "ok")`（`message` 取默认空串）。
7. **释放**：`release()` 里 `active_thinks = max(0, active_thinks - 1)`——**永不转负**，
   重复释放到 0 就停住。
8. **跨日复位**：`_day()` 取配置时区（默认 `Asia/Shanghai`）当天的 `"%Y-%m-%d"` 字符串；
    `try_acquire()` 与 `snapshot()` 进入时先检查日期键是否变化，变了就把 `used_today` 归 0
   （`active_thinks` 不动）。
9. **线程安全**：计数与日期判定在 `RLock` 下进行。
10. **只读快照**：`snapshot()` 返回新字典，外加当日已用与日期键（只读）。

### 2.3 本仓库决定（表达层）

- **模块名与包归置**：`guards/public_think.py`（§1.3）。
- **类名、字段名、方法名、reason 值、默认文案全部保留**——它们是契约事实，且被复用方按名调用。
- **内部组织自撰**：日期滚动、时区对象、锁的持有范围、`_day()` 与复位方法的切分方式由本次实现决定；
  旧实现把上游的 `_current_day_key()` + `_rollover_day_if_needed()` 合并成两个方法，
  本仓库不复用旧命名（`_day`/`_reset_if_new_day`/`_active_thinks`/`_used_today`/`_day_key`/`_timezone`
  均为旧私有名，边界扫描须零命中）。
- **不引入 `zoneinfo` 之外的依赖**；`ZoneInfo` 的非法时区名任其抛 `ZoneInfoNotFoundError`（见 §2.4）。

### 2.4 刻意不做的行为

- **不做「禁用时也不占配额」之外的旁路**：禁用不建任何状态。
- **不校验 `timezone_name`**：非法时区名照旧抛异常，本批不加兜底。
- **不把 guard 接进任何请求路径**：装配属 C4/C6，本批只产出可被装配的类。
- **不做指标上报**：旧模块不产生 metric，本批也不加。

---

## 3. `guards/evidence.py` —— 发送前的证据门

### 3.1 对外面（契约事实）

```
CHECK_AGAIN_LINE = "这个我得先去看一眼，不能凭印象说。"

def strip_unsupported_claims(text: Any, *, has_evidence: bool) -> tuple[str, list[str]]: ...
def strip_provider_speculation(text: Any) -> tuple[str, list[str]]: ...
def apply_evidence_guard(*, speech: Any, speech_segments: Any,
                         has_evidence: bool) -> tuple[str, list[str], list[str]]: ...

__all__ = ["CHECK_AGAIN_LINE", "apply_evidence_guard",
           "strip_provider_speculation", "strip_unsupported_claims"]
```

- 两个原因码字面量：`"unsupported_claim"`、`"provider_speculation"`。
- 签名要点：`strip_unsupported_claims` 的证据标志是**关键字专用**（`*`）；`apply_evidence_guard`
  三个参数**全部关键字专用**且顺序为 `speech`、`speech_segments`、`has_evidence`。
- 返回值形状：两个 strip 函数返回 `(清理后文本, 命中原因列表)`；`apply` 返回
  `(台词, 分段列表, 命中原因列表)`。
- 输入容错：`text`/`speech` 接受 `Any`，内部 `str(... or "")`；`speech_segments` 不是 `list` 时
  按空列表处理。

### 3.2 必须成立的行为

1. **减法，不重写**：只删句、不增句、不改写保留句的字面内容。**无命中时返回原文（原对象不加改动）**，
   原因列表为空。
2. **`has_evidence` 的语义**：`strip_unsupported_claims(text, has_evidence=True)` 直接返回原文与空原因
   ——本轮确有工具事件时，「我查了」是有依据的，不动它。
   **`strip_provider_speculation` 不受 `has_evidence` 影响**：供应方内部推测任何情况下都要删。
3. **句子切分**：按中文/英文句末与分隔符 `。！？!?；;` 及换行切分；保留句末符号。
4. **切分后逐句判定**：命中即整句删除；未命中的句子按原顺序拼接（**不做跨句合并、不补空格**）。
5. **原因列表**：命中的句子各追加一条对应原因码（同句多次命中不重复累积；不同原因分别追加）。
6. **`apply_evidence_guard` 的编排**：
   - 对 `speech` 与每个 segment 依次做 `strip_unsupported_claims` → `strip_provider_speculation`
     （**先 claim 后 speculation**，因此原因列表的顺序是「先所有 unsupported_claim、后所有
     provider_speculation」——按处理顺序追加）。
   - `speech_segments` 非空时，清理后的 segments 过滤掉空白项、用 `"\n"` 连接作为最终台词
     （**覆盖**由 `speech` 单独清理得到的文本）。
   - segments 为空（或全是空白）且 speech 清理后也空白，但**原因列表非空**时，最终台词与分段
     都置为 `CHECK_AGAIN_LINE`（一句「还没查」的短话术，**不假装查过**）。
   - 完全没有命中、且文本非空时，原样透传。

### 3.3 判定边界（本批的行为钉）

**「无据查证」句式必须删除**（`unsupported_claim`）——当**没有**证据时：

| 句式 | 例 |
| --- | --- |
| 第一人称 + 查看动词 + 完成标记 | 我查了…／我已经查过了／我看了…／我读过了／我刚才核对过／我确认过了 |
| 查看动词 + 「到了」 | 查到了／读到了／翻到了／看到了…（作「…到了」完成态时） |
| 查看动词 + 「完/过」+ 了 | 查完了／核对过了 |
| 第一人称 + 持有结果 | 我手里有结果／我这边有数据 |
| 引用文档 | 文档写着…／README 说了…／说明里写着… |
| 引用日志 | 日志里写着…／日志报… |
| 引用配置/文件/代码 | 配置文件里有…／代码里写… |
| 「按…读到的」 | 按读到的…／按刚才读到的… |

**这些句式必须保留**（不得误删）：

| 句式 | 例 |
| --- | --- |
| 将来时／意愿 | 我去查一下。／要不要我去看一眼？／先看看是不是端口问题。 |
| 现在时的观察（无「了」） | 我看到你贴的日志了，是连接被拒。／看你发的截图，端口没起来。 |

**「供应方内部推测」句式必须删除**（`provider_speculation`），**任何情况下**（含 `has_evidence=True`）：

| 句式 | 例 |
| --- | --- |
| 服务端 + 推测动作 + 干预动词 | 服务端过滤…／服务端确实做了截断… |
| 供应方/供应商/平台 + 推测 + 干预动词 | 平台可能拦截了这一段。／供应方层面做了替换 |
| 不可观测的内部概念 | 模型权重／隐藏审查／内部那一层／内置规则／不在你手里 |
| 投递链路 + 改动 | 投递链路里某一步改了文本。 |

**这些必须保留**：

| 句式 | 例 |
| --- | --- |
| 诚实承认看不到 | 我看不到供应方内部做了什么，这不在我能查的范围里。 |
| 正常的服务端事实 | 服务端返回 502，先看网关那边的日志。／平台已经下线了那个接口。 |
| 普通句子 | 端口被占着通常是上一次没退干净。 |

> **边界说明（为什么需要「推测动作」这一环）：** `服务端`/`平台`/`供应方内部` 这些词本身
> **不是**命中条件——只有当它们与「干预动词」（截断/替换/改写/过滤/审查/拦截/删）组合，
> 或与「不可观测概念」（模型权重/隐藏审查/内部那一层/内置规则/不在你手里）同现时才删。
> 否则会把「服务端返回 502」这种正常事实误删。这条边界由 §3.3 的两张表钉住。
>
> **另一条边界（引语类）：** 「文档/日志/配置/文件/代码」这些名词后面必须**紧跟**动词（中间
> 只许出现「里」），所以 `README 说了这件事。`（名词与动词之间有一个空格）**不**命中、
> 而 `README说了这件事。` 命中。这不是疏漏，是「名词短语 + 动词」的邻接判定；同理，
> 「投递链路」后面的窗口是 12 个字符，`投递链路完好无损，没有改动。` 里的「改」落在窗口内，
> **会被删**——这条也在测试里钉住。

### 3.4 本仓库决定（表达层）

- **模块名与包归置**：`guards/evidence.py`（§1.3）。
- **函数名、原因码、`CHECK_AGAIN_LINE` 文案、签名与参数顺序全部保留**——契约事实。
- **实现写法必须重写（本批最硬的一条）。** 旧实现由**一个超长正则**
  （`_UNSUPPORTED_CLAIM_RE`，含 8 个并列分支）+ **一个超长正则**（`_PROVIDER_SPECULATION_RE`，
  含 8 个并列分支）+ 一个切分正则（`_SENTENCE_RE`）构成。新实现**不沿用这种「一坨大正则」的组织**：
  改为**词表 + 分层判定**（先切句，再按「主语／动词／完成标记／引语头」等结构化特征判定），
  或任何等价但不逐字复制的写法。判定依据是 §3.3 的两张表，**不是**旧正则的模式串。
- **旧私有名零命中**：`_UNSUPPORTED_CLAIM_RE`、`_PROVIDER_SPECULATION_RE`、`_SENTENCE_RE`、
  `_units` 均为旧私有名，边界扫描须零命中。

### 3.5 刻意不做的行为

- **不做加法**：不生成替代文本（除整段清空时的一句 `CHECK_AGAIN_LINE`）。
- **不接计数通道**：旧 docstring 提到 `evidence_claim_trimmed` / `provider_speculation_trimmed`
  走引擎 metric 通道，但**模块本身不产生任何 metric**（只是文档说明消费方会怎么用）。本批只按
  返回值形状实现，不引入上报；计数接线属装配批次。
- **不接进发送链路**：装配属 C4/C6。
- **不做大小写/繁简的额外归一**：按字面匹配，与旧行为一致。

---

## 4. `providers/config.py` —— 供应商词汇与端点归一化

### 4.1 对外面（契约事实）

```
SUPPORTED_PROTOCOLS  = ("openai", "anthropic", "ollama")
DEFAULT_PROVIDER_ID  = "openai_compatible"

@dataclass(frozen=True)
class ProviderPreset:
    id: str
    label: str
    protocol: str
    base_url: str
    api_key_required: bool
    description: str
    capabilities: tuple[str, ...] = ("stream",)

COMMON_PROVIDER_PRESETS: tuple[ProviderPreset, ...]   # 7 条，见 §4.2
```

函数（参数顺序与关键字专用性逐字保留）：

```
preset_index(presets) -> dict[str, ProviderPreset]
normalize_api_protocol(protocol="", base_url="") -> str
normalize_base_url(*, protocol: str, base_url: str) -> str
normalize_configured_base_url(base_url: str, *, protocol: str) -> str
canonical_provider_id(value: object, *, aliases: Mapping[str, str] | None = None) -> str
infer_provider_id(*, protocol: str, base_url: str, provider_ids: Iterable[str] | None = None) -> str
base_url_matches_provider(base_url: str, provider_id: str) -> bool
provider_presets_payload(presets, *, include_capabilities: bool = False) -> list[dict[str, object]]

__all__ = ["SUPPORTED_PROTOCOLS", "DEFAULT_PROVIDER_ID", "ProviderPreset",
           "COMMON_PROVIDER_PRESETS", "preset_index", "normalize_api_protocol",
           "normalize_base_url", "normalize_configured_base_url", "canonical_provider_id",
           "infer_provider_id", "base_url_matches_provider", "provider_presets_payload"]
```

### 4.2 `COMMON_PROVIDER_PRESETS` 的七条（逐字保留）

| # | id | label | protocol | base_url | api_key_required | description | capabilities |
| :-: | --- | --- | --- | --- | :-: | --- | --- |
| 1 | `openai` | `OpenAI` | `openai` | `https://api.openai.com/v1` | True | OpenAI 官方接口。 | `("stream","vision","native_tools")` |
| 2 | `deepseek` | `DeepSeek` | `openai` | `https://api.deepseek.com/v1` | True | DeepSeek 官方 OpenAI 兼容接口。 | `("stream","native_tools")` |
| 3 | `pinaic` | `Pinaic` | `openai` | `https://api.pinaic.com/v1` | True | Pinaic 的 OpenAI 兼容接口。 | `("stream",)` |
| 4 | `gemini` | `Google Gemini` | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai` | True | Google AI Studio 的 OpenAI 兼容接口。 | `("stream","vision")` |
| 5 | `anthropic` | `Anthropic Claude` | `anthropic` | `https://api.anthropic.com` | True | Anthropic 官方 Messages API。 | `("stream","vision","native_tools")` |
| 6 | `ollama` | `Ollama 本地模型` | `ollama` | `http://127.0.0.1:11434` | False | 本机 Ollama 的 OpenAI 兼容接口。 | `("stream","vision")` |
| 7 | `openai_compatible` | `其他 OpenAI 兼容服务` | `openai` | `""` | True | 中转站、自部署网关或其他兼容 /v1 的服务。 | `("stream","vision","native_tools")` |

> `description` 是**前端展示文案**（契约事实，逐字保留）；`id`/`label`/`base_url` 是线路事实。

### 4.3 函数行为（必须成立）

1. **`preset_index`**：把可迭代的 preset 收成 `{preset.id: preset}`；后者覆盖前者（同 id 后者胜）。
2. **`normalize_api_protocol(protocol="", base_url="")`**：
   - `str(protocol or "").strip().lower()` 落在 `SUPPORTED_PROTOCOLS` 里 → 原样返回该值；
   - 否则看 `str(base_url or "").strip().lower()`：含 `"/claude"` 或 `"anthropic"` → `"anthropic"`；
     含 `"11434"` 或 `"ollama"` → `"ollama"`；
   - 否则 → `"openai"`。
3. **`normalize_base_url(*, protocol, base_url)`**：strip 尾部 `/`；`protocol == "ollama"` 时，
   空值补 `"http://127.0.0.1:11434"`，且不以 `/v1` 结尾就补 `/v1`；其他 protocol 只做 strip。
4. **`normalize_configured_base_url(base_url, *, protocol)`**：strip 尾部 `/`；按顺序对
   `"/chat/completions"` 与 `"/models"` 两个尾缀**各检查一次**（命中即剥掉、剥完再 strip 尾 `/`），
   第二次检查的是**第一次剥完后的值**（所以 `.../v1/models/chat/completions` 会先变成
   `.../v1/models`、再变成 `.../v1`）；同一尾缀不循环剥离（`.../models/models` 只掉一层）。
   随后 `protocol == "anthropic"` 且以 `"/v1/messages"` 结尾时，去掉 `"/messages"`。
5. **`canonical_provider_id(value, *, aliases=None)`**：
   `str(value or "").strip().lower()` 后查 `aliases`（缺省空 dict），命中返回别名值，否则返回自身。
6. **`infer_provider_id(*, protocol, base_url, provider_ids=None)`**：
   按**固定优先级**在 base_url（小写）里找信号，第一个命中且（`provider_ids` 为空 **或** 该 id 在
   允许集合内）的胜出：
   `ollama`（protocol==`ollama` 或含 `11434` 或含 `ollama`）→
   `anthropic`（protocol==`anthropic` 或含 `anthropic.com` 或含 `/claude`）→
   `glm`（含 `open.bigmodel.cn` 或 `bigmodel.cn`）→ `openai`（含 `api.openai.com`）→
   `deepseek`（含 `api.deepseek.com`）→ `pinaic`（含 `api.pinaic.com`）→
   `gemini`（含 `generativelanguage.googleapis.com`）。
   都不命中返回 `DEFAULT_PROVIDER_ID`（`"openai_compatible"`）。
7. **`base_url_matches_provider(base_url, provider_id)`**：按 provider id 查「信号词表」，
   任一词出现在 base_url（小写）里即 True；未知 id → False。
   信号词表：`glm`→(`bigmodel.cn`,)；`deepseek`→(`deepseek.com`,)；
   `gemini`→(`generativelanguage.googleapis.com`,)；`anthropic`→(`anthropic.com`, `/claude`)；
   `ollama`→(`11434`, `ollama`)；`openai`→(`api.openai.com`,)；`pinaic`→(`api.pinaic.com`,)。
8. **`provider_presets_payload(presets, *, include_capabilities=False)`**：逐条产出 dict，
   键为 `id`/`label`/`protocol`/`baseUrl`/`apiKeyRequired`/`description`；
   `include_capabilities=True` 时追加 `capabilities`（**转成 `list`**）。
   **键名是线路契约（camelCase），逐字保留。**

### 4.4 本仓库决定（表达层）

- **模块名与包归置**：`providers/config.py`（§1.3）。
- **所有公开名、常量、字段、payload 键名、7 条 preset 内容全部保留**——契约事实。
- **内部组织自撰**：把「信号词表」与「优先级序列」的**数据结构**重新设计（旧实现把优先级写成
  一个 `candidates` 元组、把 matches 表写成内联 dict），本批可拆成具名常量或独立辅助；
  归一化步骤的分解方式由本次决定。**旧实现 8 个函数全部保留同名同签名**，但函数体写法重写。
- **不引入第三方依赖**，只用标准库。

### 4.5 刻意不做的行为

- **不解析真实请求**：只做字符串归一化与词汇表，不发网络请求、不读环境变量。
- **不改 7 条 preset 的内容**（含 `description` 的中文措辞与 `glm` 的缺席——`glm` 属项目侧
  `model_service_config`，不在本模块词汇表里；本批不把它补进来）。
- **不把 `infer_provider_id` 的返回类型改成 `Literal`** 或加枚举：返回值就是 `str`。

---

## 5. `tools/native_schema.py` —— provider 原生工具 schema 的构建

### 5.1 对外面（契约事实）

```
NATIVE_TOOL_NAME_RE               = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
NATIVE_TOOL_DESCRIPTION_MAX_CHARS = 900

def build_openai_native_tool_specs(
    handlers: Mapping[str, Any] | None,
    *,
    allowed_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]: ...

__all__ = ["NATIVE_TOOL_DESCRIPTION_MAX_CHARS", "NATIVE_TOOL_NAME_RE",
           "build_openai_native_tool_specs"]
```

`NATIVE_TOOL_NAME_RE` 的字符集与长度（`[A-Za-z0-9_.:-]`，1–80）与上限常量 `900` 都是契约事实。

### 5.2 必须成立的行为

1. **入参校验**：`handlers` 不是 `Mapping` → 返回 `[]`。
2. **遍历顺序**：按**原始键**（`str(key or "")`）升序遍历，结果稳定。
3. **取名与去重**：`name = str(getattr(handler, "tool_type", "") or raw_name or "").strip()`；
   同名只取第一条（`seen` 去重）。
4. **跳过条件**（满足任一即跳过）：名字为空；名字已出现；`allowed_tool_names` 非空且名字不在其中；
   名字不匹配 `NATIVE_TOOL_NAME_RE`。
5. **description 取值优先级**：
   - 先取 `handler.tool_metadata().input_schema` 的 `description`（可退 `x_description`）；
     非空则用（`" ".join(text.split())` 压空白后截 900）；
   - 否则取 `handler.build_prompt_instruction()`（**调用异常时当空串**）；
   - 否则退回 `f"Call Shinku tool {name}."`；
   - 最后**去旧信封子句**（§5.3）并截 900。
6. **parameters 取值优先级**：
   - 先取 `handler.tool_metadata().input_schema` 去掉 `description`/`x_description` 后的 dict，
     并**强制 `type == "object"`**（若原值不是 `"object"` 就改写）；
   - 否则 `{"type": "object", "additionalProperties": True}`。
7. **spec 形状**（逐字保留）：
   `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`。

### 5.3 「去旧信封子句」的行为

- 把 description 按**句号 `。`** 切成片段（保留句号）。
- 删除**含任一旧信封标记**的片段；标记集合：`格式为`、`调用格式`、`tool_call`、`{"type"`。
- 若删完为空，**退回原文**（不做全删）。
- 这个「标记集合」是行为的一部分（决定删哪些句子），**成员逐字保留**；但检测方式（切分与过滤的
  写法）属实现，由 Shinku 重写。

### 5.4 本仓库决定（表达层）

- **模块名与包归置**：`tools/native_schema.py`（§1.3）。
- **公开名、常量、签名、spec 形状全部保留**——契约事实。
- **内部组织自撰**：旧实现用 6 个函数（`build_openai_native_tool_specs` +
  `_handler_description`/`_handler_schema`/`_metadata_description`/`_metadata_parameters`/
  `_strip_legacy_envelope_clauses`）。本批**可以改内部拆分**（例如合并元数据读取、把「取
  description/parameters」做成一次 `tool_metadata()` 调用返回两者），只要公开行为不变。
  **旧私有名 `_LEGACY_MARKERS`/`_handler_schema`/`_metadata_description`/`_metadata_parameters`/
  `_handler_description`/`_strip_legacy_envelope_clauses` 边界扫描须零命中。**
- **不引入第三方依赖**，只用标准库 `re` 与 `typing`。

### 5.5 刻意不做的行为

- **不调用真实 handler 之外的东西**：只读 `tool_type`、`tool_metadata`、`build_prompt_instruction`
  三个属性/方法，不做注册、不发请求。
- **不做 JSON Schema 校验或补全**：只透传 `input_schema` 的字段。
- **不接进工具编排引擎**：装配属 C3。
- **不定义工具元数据类型**：`TOOL_METADATA_BY_TYPE` 等属其它批次。

---

## 6. 刻意不做的行为（本批整体）

1. **不装配**：四个模块都不接进 `api/app.py`、不注册路由、不挂中间件。B1 的健康面与启动链
   行为不变，因此**本批不做进程探针**（与 C1-2/C1-3 同样的理由，记在批次记录里）。
2. **不改旧项目源码**：旧项目保持只读档案，0 改动。
3. **不搬运既有测试文本**：`tests/test_contract_c1_4.py` 全部自撰，断言对象是契约的外部可观察
   行为；允许与旧测试**覆盖相同的用例点**（因为行为等价），但**不复制测试文本或固定实现结构**。
4. **不引入新的第三方依赖**：四个模块都只用标准库（`dataclasses`、`datetime`、`threading`、
   `zoneinfo`、`re`、`typing`）。
5. **不清理旧项目的其余模块**：`model_service_config.py`、`tool_runtime.py`、
   `tool_orchestration_engine.py` 等消费方不在本批范围。

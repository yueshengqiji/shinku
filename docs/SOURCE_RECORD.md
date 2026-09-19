# 来源记录

## 1. 旧项目的角色变更

自 2026-09-18（B1）起，`AkaneCompanionLab_Shinku` 从"开发基线"**降级为只读内部来源档案**。

| 维度 | 变更前 | 变更后 |
| --- | --- | --- |
| 角色 | 开发基线 + 发布候选 | **只读来源档案** |
| 本仓库对它的依赖 | —— | **代码依赖：无。测试依赖：无。运行时依赖：无。** |
| 允许的操作 | 读写 | **只读**（可查、可对照、可引用路径，不复制表达性文本） |
| 是否被删除 | —— | **否，未被删除也未被修改** |

**为什么保留而不删除：** 它是 A 阶段全部来源取证的对象和证据载体
（`SOURCE_PROVENANCE.md`、`docs/shinku_independent_*` 批次记录、`docs/evidence/`）。
删掉它等于销毁证据链。计划 D 阶段也明确要求"不在确认完成前删除现有 `LICENSE`、
`NOTICE`、`UPSTREAM.md`"。

## 2. 本仓库不得做的事

- ❌ 从旧项目复制源码文本、注释、测试文本、文档段落；
- ❌ 把旧项目加入 `sys.path`、作为 `-e` 依赖、或以任何方式在运行期读取它；
- ❌ 要求旧项目的目录、环境变量、配置文件或进程存在才能启动；
- ❌ 继承旧项目里 Akane 时代的键名与路径（`COMPANION_*` 等）。

## 3. 可以做的事

- ✅ 读取旧项目的**接口事实**——端口号、HTTP 头名、文件名、命令名。这些是兼容性所需的
  约定，不是表达性内容（例如"关联 ID 头叫 `X-Correlation-ID`"是协议，不是创作）；
- ✅ 以**行为契约**的形式记录旧模块的对外行为，作为 C 阶段重写的需求输入；
- ✅ 在文档里引用旧项目的路径，用于追溯。

> 第 3 条第一点是本仓库 `src/shinku/names.py` 的依据：它包含的端口值、环境变量名和
> 头名来自旧项目的接口事实。**其中的解释性文字、结构组织和解析逻辑均为本次原创实现。**

## 4. 来源台账

每个进入本仓库的文件都要有一行。**行数应等于本仓库的业务文件数**（见 `ADMISSION.md` 第 4 节的欠账项）。

### 4.1 B1 骨架（2026-09-18 建仓时全部写入）

| 路径 | 分类 | 与旧项目的关系 | 依据 |
| --- | --- | --- | --- |
| `src/shinku/__init__.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/names.py` | `INDEPENDENT_KEEP` | 接口兼容（端口/头名沿用） | 本仓库新建；接口事实见 `README.md` 命名约定 |
| `src/shinku/config.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/cli.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/__main__.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/api/__init__.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/api/app.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `src/shinku/hosts/__init__.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建（仅包标记） |
| `src/shinku/py.typed` | `INDEPENDENT_KEEP` | 无 | 本仓库新建（PEP 561 标记，`pyproject.toml` 已声明） |
| `tests/test_names.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `tests/test_health.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |
| `tests/test_cli.py` | `INDEPENDENT_KEEP` | 无 | 本仓库新建 |

**没有从旧项目复制任何文件。** 这也意味着 B1 阶段本仓库只有骨架、
不构成可运行的完整产品——这是准入规则算出来的结果，不是遗漏。

### 4.2 C 阶段起（每批追加）

进库日期均为 2026-09-18。**C 阶段有两种处理类型，判定标准不同，台账分列登记、不得混用**
（来源记录 §5.9 路径 A）：

| 类型 | 判据 | 登记写法 | 合格标准 |
| --- | --- | --- | --- |
| **契约事实迁移** | 纯契约：无分支、无算法，只有 dataclass／常量表／类型别名／签名 | 「事实来源=`<旧路径>`｜表达层：Shinku 撰写」 | 台账**分列**了事实来源与表达层；**不看相似度** |
| **洁净室重写** | 含逻辑、分支、算法、数据处理 | 「契约=`<契约文档>`｜实现独立」 | 有契约文档 + 独立测试 + 实现独立 |

迁移批次的产物「与上游相似」是**预期结果，不是缺陷**；洁净室重写批次必须另有契约文档与独立测试。
两类**不得用同一句结论概括**——A2 用一句「已本地重写」概括了两类工作，结果漏掉 4 项（§5.8）。

#### 4.2.1 C1-1 数据类型契约层 —— 契约事实迁移

契约事实允许按事实迁移并须登记事实来源；docstring、注释、内部组织与文件切分、
内部命名风格、实现步骤必须 Shinku 独立撰写。

| 路径 | 进库日期 | 分类 | 契约事实来源 | 依据（重写记录 / 契约文档） |
| --- | --- | --- | --- | --- |
| `src/shinku/contracts/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 表达层：Shinku 撰写 |
| `src/shinku/contracts/tasks.py` | 2026-09-18 | **契约事实迁移** | `companion_v01/task_types.py`、`companion_v01/task_worker_contract.py` | 契约 `docs/contracts/c1_1_data_contracts.md` §2；表达层：Shinku 撰写（两个来源文件合并为一个模块，文件切分为本仓库自有组织） |
| `src/shinku/contracts/retrieval.py` | 2026-09-18 | **契约事实迁移** | `companion_v01/retrieval_types.py` | 同上 §3 |
| `src/shinku/contracts/http.py` | 2026-09-18 | **契约事实迁移** | `companion_v01/http_response.py` | 同上 §4 |
| `src/shinku/contracts/capability.py` | 2026-09-18 | **契约事实迁移** | `companion_v01/capability_adapters/types.py` | 同上 §5 |
| `tests/test_contract_c1_1.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；断言对象为契约的外部可观察行为，含失败测试 |

**C1-1 未做的事：** 未搬运 `capability_adapters/manifest.py` 的解析逻辑（属 C1-3，洁净室重写）；
未搬运 `manifest_loader.py` 的转发壳（与 `manifest` 同批处理）。

#### 4.2.2 C1-2 共享基础模块 —— 洁净室重写

四个来源模块都含逻辑、分支或数据处理，因此**不是**契约事实迁移，按洁净室重写处理。
来源侧合计 **18,122 字节**。登记写法「契约=`<契约文档>`｜实现独立」：

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/api/correlation.py` | 2026-09-18 | **洁净室重写** | `companion_v01/request_context.py`（2,667 B） | 契约 `docs/contracts/c1_2_shared_modules.md` §2；表达层：Shinku 撰写（按领域改置 `api/`；头名常量改由 `names.py` 单点定义、本模块不重定义；导出面收窄为 4 个名字，不再导出头名） |
| `src/shinku/api/sessions.py` | 2026-09-18 | **洁净室重写** | `companion_v01/routes/sessions.py`（6,785 B） | 同上 §3；表达层：Shinku 撰写（改为全注入——不引入 `engine`，只声明真正调用的存储方法；两个身份解析器合并为一个；错误信封与角色包归一化改由注入面提供） |
| `src/shinku/tasks/artifacts.py` | 2026-09-18 | **洁净室重写** | `companion_v01/task_artifacts.py`（5,563 B） | 同上 §4；表达层：Shinku 撰写（去掉 `task_workspace_` 冗余前缀；抽 `_first_present`／`_carry_over` 等具名辅助） |
| `src/shinku/tools/invocation.py` | 2026-09-18 | **洁净室重写** | `companion_v01/tool_invocation.py`（3,107 B） | 同上 §5；表达层：Shinku 撰写（线路字段常量集中声明；参数抽取抽为 `_arguments_from`） |
| `src/shinku/tasks/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包标记） | 本仓库新建（仅包标记，`__all__` 为空） |
| `src/shinku/tools/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包标记） | 本仓库新建（仅包标记，`__all__` 为空） |
| `tests/test_contract_c1_2.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；断言对象为契约的外部可观察行为，含失败测试、边界扫描与 `request_context` 异常路径 |

**C1-2 的合格判据不是相似度，而是「有契约文档 + 独立测试 + 实现独立」。** 三项证据：

1. 契约文档 `docs/contracts/c1_2_shared_modules.md`（只记录外部可观察行为，作为新实现的唯一需求输入）；
2. 独立测试 `tests/test_contract_c1_2.py`（143 个用例，其中异常路径用例经**反证**验证非空过）；
3. 实现独立：旧模块的内部命名（`MIN_ID_LENGTH`／`ID_ALPHABET`／`_is_well_formed`／
   `_new_correlation_id`／`task_workspace_` 前缀／`pack_from_payload`／`_observe` 等）
   在新文件中**零命中**，由测试 `C1_2BoundaryTests` 钉住。

**C1-2 未做的事：** 未装配——`CorrelationIdMiddleware` 未挂进 `api/app.py`，sessions 路由未注册，
两者均属 C4/C6 的装配工作；未迁移 `store.normalize_character_pack_id` 与
`desktop_pet_contract.build_desktop_pet_error_payload` 本体，只把行为与信封形状记为契约事实（§3.5／§3.7），
实现由注入面提供；未做 `task_workspace`、`tool_orchestration` 的其余部分（C3 范围）。

顺带说明：`docs/contracts/c1_1_data_contracts.md` 是本批的行为契约文档，
`docs/ADMISSION.md`、`README.md`、`NOTICE`、`env.example` 是仓库级文档，
均不在本台账的「代码 + 测试」计数口径内（与 B1 保持一致）。

#### 4.2.3 C1-3 能力清单与资源清单 —— 洁净室重写（含一项纯契约传递依赖）

来源侧三个模块都含逻辑、分支与数据处理，因此**不是**契约事实迁移。登记写法
「契约=`<契约文档>`｜实现独立」。计划登记的三个模块合计 **43,511 字节**；
本批另落地两个**必须随行**的模块（`manifest.py` 的传递依赖与其转发壳），
实测五者合计 **44,093 字节**——两个数不是同一集合，见契约 §1。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/capabilities/manifest.py` | 2026-09-18 | **洁净室重写** | `companion_v01/capability_adapters/manifest.py`（13,024 B） | 契约 `docs/contracts/c1_3_capability_and_resource_manifests.md` §3；表达层：Shinku 撰写（按领域改置 `capabilities/`；控制流改为「私有异常短路 + 边界统一转换」；不定义任何数据类型，全部取自 C1-1 的 `contracts/capability`） |
| `src/shinku/resources/manifest.py` | 2026-09-18 | **洁净室重写** | `companion_v01/resource_manifest.py`（30,352 B） | 同上 §4；表达层：Shinku 撰写（把「文件系统 → 清单字典」拆进独立的内部索引类；合并类操作改为模块级纯函数；来源侧是 47 方法的单类） |
| `src/shinku/capabilities/safety.py` | 2026-09-18 | **洁净室重写** | `companion_v01/capability_safety.py`（371 B） | 同上 §2；表达层：Shinku 撰写。**它是纯契约（3 个常量），本可按契约事实迁移登记**；但它是 A2 列入重写清单的高相似模块（0.800，与共享包仅 docstring 不同），且本批必须落地它，故归入洁净室重写并单独登记。这项登记**同时了结 A2 重写清单里的 `capability_safety.py`** |
| `src/shinku/capabilities/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 本仓库新建 |
| `src/shinku/resources/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 本仓库新建 |
| `tests/test_contract_c1_3.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；断言对象为契约的外部可观察行为，含失败测试、边界扫描与契约表达式定点断言 |

**两个转发壳没有对应的新模块，这是本批的表达层决定**（契约 §1.3）：

| 旧转发壳 | 字节 | 处置 | 理由 |
| --- | --: | --- | --- |
| `companion_v01/capability_adapters/manifest_loader.py` | 135 | 不建模块，`load_manifest` 的直接导入路径就是 `shinku.capabilities.manifest` | 内容只有 `from .manifest import *` + `__all__` 转发；保留中转模块等于把旧仓库的目录形状固化进新仓库 |
| `companion_v01/capability_adapters/safety.py` | 211 | 不建模块，常量由 `shinku.capabilities.safety` 直接提供 | 同上，只转发 `..capability_safety` 的三个名字 |

**C1-3 的合格判据不是相似度，而是「有契约文档 + 独立测试 + 实现独立」。** 四项证据：

1. 契约文档 `docs/contracts/c1_3_capability_and_resource_manifests.md`（只记录外部可观察行为）；
2. 独立测试 `tests/test_contract_c1_3.py`（**143 个用例**，含 **31 条拒绝路径用例**
   （覆盖契约 §3.6 的 **19 个原因码**，读失败类 `yaml_parse_error`／`read_error` 另由
   `ManifestReadFailureTests` 覆盖 ⇒ 21 个原因码全数覆盖）、8 条优先级顺序、
   边界扫描与契约表达式定点断言）；
3. 实现独立：旧模块的 **56 个私有名**（模块级 11 + 类内 35 + capabilities 侧 11 中重合部分）
   在本批五个新文件里**零命中**。这条用 `tokenize` 取**精确标识符 token** 比对，
   不用子串匹配——`_text` / `_key` / `_meta` / `_background` 这类短名字用子串匹配会把
   `_read_note_text` / `_background_record` 误判为命中（C1-2 用的是子串匹配，本批加强了口径）；
4. 散文独立：docstring + 注释、连续 ≥12 字符的逐字片段，从首版 6 处收敛到 **2 处**，
   且两处都只剩领域词 `capability`（12–13 字符）本身。**输出文案（提示词段落与标签）
   是外部可观察输出，属契约事实，不在散文口径内**（契约 §0.1）。

**C1-3 未做的事：** 未装配——`load_manifest` 未接进任何注册表，`ResourceManifest` 未接进
`api/app.py`，两者均属 C4/C6；未迁移旧项目 `docs/fixtures/capability_adapter_m1/*.yaml`
与本批测试夹具无关（测试在临时目录里现造 YAML 与目录树）；
未碰 `capability_adapters/` 下的 `protocol`、`registry`、`comfyui` 等其余能力模块。

**本批暴露的两个来源事实照录不改**（修它们属行为变更，需单独决定）：
`load_manifest` 对非 UTF-8 文件会逃逸 `UnicodeDecodeError`；BGM 条目的名称回退走
`EMOTION_LABELS` 而非「无标签表」。两者都在测试里被钉住。

#### 4.2.4 C1-4 公共守卫、证据门、供应商词汇与原生工具 schema —— 洁净室重写

来源侧四个模块都含逻辑、分支或数据处理，因此**不是**契约事实迁移。登记写法
「契约=`<契约文档>`｜实现独立」。计划登记的四个模块合计 **17,651 字节**；本批**只有这一个口径**
——没有随行的传递依赖，也没有转发壳要消解。计划值与本批实测值同为 **17,651**。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/guards/public_think.py` | 2026-09-18 | **洁净室重写** | `companion_v01/public_guard.py`（3,024 B） | 契约 `docs/contracts/c1_4_guards_provider_tool_schema.md` §2；表达层：Shinku 撰写（按领域改置 `guards/` 并去掉 `guard` 后缀；内部状态命名、跨日滚动的拆分、消息兜底都重写；`_day`／`_reset_if_new_day`／`_active_thinks`／`_used_today`／`_day_key`／`_timezone`／`_lock` 七个旧私有名零命中） |
| `src/shinku/guards/evidence.py` | 2026-09-18 | **洁净室重写** | `companion_v01/evidence_guard.py`（5,693 B） | 同上 §3；表达层：Shinku 撰写。**实现写法彻底更换**：旧版是「两个各含 8 个并列分支的超长正则 + 一个切分正则」，新版改为**词表 + 分层查找**（先切句，再按「主语／动词／完成标记／引语头」结构化判定），文件内最长行 < 120 字符并由测试钉住。散文重叠从首版 5 处收敛到 0 处 |
| `src/shinku/providers/config.py` | 2026-09-18 | **洁净室重写** | `companion_v01/provider_config.py`（5,525 B） | 同上 §4；表达层：Shinku 撰写（按领域改置 `providers/`；把「信号词」拆成三张**必须分开**的具名表 `_PROTOCOL_SIGNALS`／`_INFER_SIGNALS`／`_MATCH_SIGNALS` 并统一走 `_has_signal` 查表；八个函数体全部重写） |
| `src/shinku/tools/native_schema.py` | 2026-09-18 | **洁净室重写** | `companion_v01/native_tool_schema.py`（3,409 B） | 同上 §5；表达层：Shinku 撰写（旧版 6 个私有函数改为 4 个具名辅助；`_LEGACY_MARKERS` → `_ENVELOPE_MARKERS`；句号切分由正则 `re.split` 改为手写遍历；跳过条件收进 `_admits`） |
| `src/shinku/guards/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包标记） | 本仓库新建（仅包标记，`__all__` 为空） |
| `src/shinku/providers/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 本仓库新建 |
| `tests/test_contract_c1_4.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；断言对象为契约的外部可观察行为，含边界扫描与契约表达式定点断言 |

**对照物判定（规矩六在本批第二次命中）：** `public_guard.py` 与 `native_tool_schema.py` 在
Akane 侧都是**转发壳**（183 B / 1,015 B，正文只有 `from code_shared.xxx import *`），若拿它们当
对照物，整文件比值会给出 0.0652 / 0.2941 这样的**假低值**，方向完全反。对照物一律取
`code_shared` 的真实现。`evidence_guard.py` 在 `code_shared` **无副本**、与 Akane 侧
**逐字节相同**（sha 均为 `6434e377080a`，C1 对账列为「T3 直连同源，A2 盲区」），对照物取 Akane 同路径。

**`native_tool_schema.py` 的 T4 身份在本批核清：** C1 对账把它列为「T4 低相似（<0.50），需人工核验」。
本批完成核验——共有方法 3 个，函数体相似度均值 **0.2879**、中位 0.3636、**≥0.80 的一个都没有**；
旧项目 6 个方法、上游 6 个方法，**两侧各有 3 个独有方法**，是**不同功能集**。结论与 C1-3 的
`resource_manifest` 同型：**独立的小实现，不是上游砍版本**。但它仍按洁净室重写处理
（来源未经取证，本批补齐契约与测试），**不按「已证明独立」豁免**。

而另三个的相似度都够高、够不上独立：`evidence_guard.py` 逐字节相同（5/5 方法全 1.0）、
`provider_config.py` 8/8 方法共有且 3 个函数体逐字相同、`public_guard.py` 是同一实现的结构压缩
（`release` 1.0 / `snapshot` 0.9）。**四个模块全部归入洁净室重写，无一豁免。**

**C1-4 的合格判据不是相似度，而是「有契约文档 + 独立测试 + 实现独立」。** 五项证据：

1. 契约文档 `docs/contracts/c1_4_guards_provider_tool_schema.md`（只记录外部可观察行为）；
2. 独立测试 `tests/test_contract_c1_4.py`（**114 个用例 / 17 个测试类 / 24 个 subTest 点**）；
3. 实现独立：旧模块的 **17 个私有名**（`public_guard` 7 + `evidence_guard` 4 +
   `native_tool_schema` 6；`provider_config` 无私有名）在本批新文件里**零命中**，
   用 `tokenize` 取**精确标识符 token** 比对；
4. 散文独立：docstring + 注释、连续 ≥12 字符的逐字片段，从首版 **5 处**收敛到 **0 处**。
   5 处全部落在 `guards/evidence.py`——本批唯一「旧实现与上游逐字节相同」的模块，
   首版沿用了它的解释性措辞；**这是散文筛子在本批的实际战果**；
5. **行为等价性对拍 + 变异测试（本批新增的两条证据口径，对后续洁净室重写批次通用）**：
   - 对拍 `_c1_4_parity.py`：把旧实现与新实现放在同一批语料上逐项比返回值——
     `public_think` **1,512 组**、`evidence` **147,825 次**、`provider_config` **953 组**、
     `native_tool_schema` **60 组**，合计 **150,350 次比对，全部 0 差异**；
   - 变异 `_c1_4_mutation.py`：逐个注入 **20 个典型缺陷**（`release` 允许转负、每日额度优先级反转、
     `has_evidence` 语义反转、推断表与匹配表合并、payload 键名改 snake_case、去信封标记漏项、
     `tool_type` 不再优先于键名、去重失效……），确认测试变红——**20 个全部被抓住，0 漏**；
     恢复后逐文件 sha256 与初始值一致。首版有 **1 个漏**（「引语类名词必须紧跟动词」这条边界
     未被覆盖），补一条测试句后抓住。

> 第 5 项是本批对 SKILL 的贡献。相似度与散文只能证明**没有抄**；对拍证明**行为对齐**，
> 变异证明**测试有效**。三者合起来才构成洁净室重写的完整证据链。两项脚本在协作工作区
> （`_c1_4_parity.py` / `_c1_4_mutation.py`），后续批次可沿用。

**依赖变更（本批唯一一处 `pyproject.toml` 改动）：** `guards/public_think.py` 用
`zoneinfo.ZoneInfo` 按 `Asia/Shanghai` 做跨日判定，而 Windows 没有系统 tzdb，需要 `tzdata`。
按项目约定（「每接入一个模块，加它需要的那一个依赖」）在 `dependencies` 里加 `tzdata`
——与旧项目 `requirements.txt` 的无条件声明一致。干净环境原先缺 `tzdata`，
本批就地补装（PyPI 当时不可达，从旧 venv 复制纯数据包，等价于解包 wheel）。

**C1-4 未做的事：** 未装配——`PublicThinkGuard` 与证据门都没接进任何发送或请求路径，属 C4/C6；
未接计数通道（旧 docstring 提到的 `evidence_claim_trimmed` / `provider_speculation_trimmed`
走引擎 metric，模块本身不产生 metric）；未碰 `model_service_config.py`、`tool_runtime.py`、
`tool_orchestration_engine.py` 等消费方。

#### 4.2.5 C2-1 上游静默开关、文本切词与嵌入供应商 —— 洁净室重写（含一段随行切片）

来源侧四个模块都含逻辑、分支或算法，因此**不是**契约事实迁移。登记写法
「契约=`<契约文档>`｜实现独立」。四个对照物合计 **14,406 字节**；另有一段**必须随行**的
源——`text_utils.py`（570 行 / 18,415 B）里的 `tokenize` 切片，见契约 §2.3。

**切批依据不是目录，是依赖图**（`_c2_dep_graph.py` 的输出）：`huggingface_provider →
embedding_provider`、`embedding_provider → text_utils(tokenize)`。**传递依赖必须随行**，
与 C1-3 的 `capability_safety.py` 同一条规矩。本批也是 C2 里**唯一不碰网络**的一批
（四个模块一个 `requests`／`urllib` 都没有），所以不必替换传输层就能把对拍跑完——
这是它被排在 C2 第一位的理由。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/llm/circuit_breaker.py` | 2026-09-18 | **洁净室重写** | `companion_v01/llm_circuit_breaker.py`（3,573 B） | 契约 `docs/contracts/c2_1_provider_layer.md` §1；表达层：Shinku 撰写（按领域改置 `llm/`；内部状态命名整组更换——`_lock`／`_consecutive_failures`／`_open_until`／`_open_count`／`_last_failure_at`／`_last_success_at`／`_BREAKER` 七个旧私有名零命中；docstring 里 2026-09-08 事故背景那段整体重写） |
| `src/shinku/text/tokenizer.py` | 2026-09-18 | **洁净室重写** | `code_shared/text_tokenizer.py`（1,144 B）；行为基准取 Shinku 旧侧 `companion_v01/text_utils.py` 的 `tokenize` | 同上 §2；表达层：Shinku 撰写（**只迁本切片用到的四条顶层语句**，`text_utils.py` 其余 500 行属 C5；汉字串展开抽为 `_expanded`，判定用 `_CJK_ONLY.fullmatch`，窗宽表 `_SLIDING_WIDTHS` 具名；`TOKEN_RE`／`STOPWORDS`／`normalize_text` 是契约事实，逐字保留） |
| `src/shinku/providers/embedding.py` | 2026-09-18 | **洁净室重写** | `code_shared/embedding_provider.py`（5,384 B）；行为基准取 Shinku 旧侧 `companion_v01/embedding_provider.py` | 同上 §3；表达层：Shinku 撰写（`_COLLECTION_COMPONENT_RE`→`_NON_KEY_CHARS`、`_dimension`→`_width`、`_legacy_collection_name`→`_retired_name`、`_cache`→`_stored`、`_get`／`_put`→`_recall`／`_store`、`_lock`→`_guard`；哈希算法是契约事实，逐字保留） |
| `src/shinku/providers/huggingface.py` | 2026-09-18 | **洁净室重写** | `code_shared/huggingface_provider.py`（4,305 B）；行为基准取 Shinku 旧侧 `companion_v01/huggingface_provider.py` | 同上 §4；表达层：Shinku 撰写（`_load_model`→`_open_encoder`、`_model`→`_encoder`、`_temporary_hf_load_env`→`_hf_env_override`；`RuntimeError` 消息与两个环境变量名是契约事实，逐字保留） |
| `src/shinku/llm/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 本仓库新建 |
| `src/shinku/text/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | 本仓库新建 |
| `src/shinku/providers/__init__.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无（包导出面） | C1-4 新建，本批只改 docstring：说明新增的两个子模块**不进** `__all__`（`__all__` 是从扁平模块 `provider_config` 迁入时留下的兼容面，再往里塞会把聚合面变成「什么都有」的口袋；调用方按全路径导入，做法同 `shinku.tools`／`shinku.guards`） |
| `tests/test_contract_c2_1.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；断言对象为契约的外部可观察行为，含边界扫描与契约表达式定点断言 |

**对照物判定（规矩六在本批第三次命中）：** `embedding_provider.py` 在 Akane 侧是
**628 B / 20 行转发壳**、`huggingface_provider.py` 是 **279 B / 10 行转发壳**，若拿它们当对照物，
整文件比值会给出假低值、方向完全反（与 C1-4 的 `public_guard` 同型）。对照物一律取 `code_shared`
的真实现。`text_tokenizer.py` 在 Akane 侧**不存在**（它是 `code_shared` 从 `text_utils.py`
抽出来的公共件）；`llm_circuit_breaker.py` 则是**两侧逐字节相同**
（sha `2c70d263b115`，属 A2-6 那批「同路径完全相同文件」），因此并集就是这一个文件。

**C2-1 的合格判据不是相似度，而是「有契约文档 + 独立测试 + 实现独立」。** 六项证据：

1. 契约文档 `docs/contracts/c2_1_provider_layer.md`（只记录外部可观察行为，是本批唯一的需求输入）；
2. 独立测试 `tests/test_contract_c2_1.py`（**115 个用例 / 17 个测试类 / 21 个 subTest 点**）；
3. 实现独立：旧侧 **34 个私有名**（取「对照物 ∪ Shinku 旧实现」的并集，`kit.py namegap`
   核对覆盖完整、0 缺口）在本批新文件里**零命中**，用 `tokenize` 取**精确标识符 token** 比对；
4. 散文独立：docstring + 注释、连续 ≥12 字符的逐字片段，**0 处**。首版 1 处命中的是
   产品名（`sentence-transformers`／`HuggingFace`），按规矩十二「术语不算复制」本可登记豁免；
   仍改写掉以获得一个**不需要解释的 0**；
5. **行为等价性对拍 `_c2_1_parity.py`：2,278 次比对、0 处差异**。四组：
   `circuit_breaker` 45 组参数网格 × 24 步操作（注入可控时钟）、`tokenizer` 41 组语料、
   `embedding` 命名网格 48 组 + 哈希 48 组 + 缓存 48 组、`huggingface` 600 组构造网格；
6. **变异测试：注入 31 个典型缺陷，全部被抓住、0 漏**，恢复后逐文件 sha256 与初始值一致。

**本批登记的一处有意差异（不是行为差异）：** 新侧 `llm/circuit_breaker.py` 显式声明了 `__all__`，
旧侧没有（`__all__` 只影响 `from ... import *`，旧侧从未声明）。对拍脚本对 `__all__`
**只在两侧都声明时才比对**，并在报告里打印这次跳过——不把「旧侧没声明」伪装成「旧侧声明了空表」。

**来源侧的一个事实照录不改：** 契约 §3.2 的集合键兜底值 `"embedding"`（`squeezed or "embedding"`）
在公开路径上**不可达**——名字至少是 `"base"`、维度至少是 1，拼出来的键永远含数字。
它与旧实现一致，属契约事实，**保留以免改变行为**；本批把它记为「已知死分支」而不是删掉。

**依赖变更：无。** 四个模块只用标准库（`threading`／`time`／`re`／`hashlib`／`math`／
`collections`／`abc`／`typing`／`inspect`／`os`／`warnings`／`contextlib`），
`pyproject.toml` 的 `dependencies` 本批未动。

**C2-1 未做的事：** 未装配——熔断器没接进任何发送路径、`tokenizer`／`embedding` 没接进检索，
属 C4/C6；未碰 `model_service_config.py`（读代码后确认它不是自足实现，整整 389 行都在委托
`model_service.py`，两者必须同批 ⇒ 已并入 C2-2）；未迁 `text_utils.py` 的其余部分
（时间表达解析、话题抽取、聊天渲染属 C5）。

#### 4.2.6 C1-5 进程健康载荷原语 —— 契约事实迁移（C1 重写清单的收口批）

**为什么叫「收口批」：** 2026-09-18 的 C1 对账决定把 3 个未处理的高相似模块纳入重写清单
（`huggingface_provider.py` 0.887、`health.py` 0.905、`capability_safety.py` 0.800）。
前两项已分别由 C1-3 与 C2-1 随行落地（§4.2.3／§4.2.5），**本批了结最后一项**。
C1-1～C1-4 仍是 C1 的四个子批次，本批不改变它们的登记。

**类型判定：** 全文件 = 1 个模块 docstring + 1 个无参函数 + 1 条 `return` 字典字面量 + `__all__`，
**无分支、无算法、无状态** ⇒ 归「纯契约」类，登记写法「事实来源=…｜表达层：Shinku 撰写」。
契约文档 `docs/contracts/c1_5_health_primitive.md` 列了 10 条契约事实与 5 条外部可观察行为。

| 路径 | 进库日期 | 分类 | 事实来源 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/health.py` | 2026-09-18 | **契约事实迁移** | `code_shared/health.py`（上游真实现 507 B） | 契约 `docs/contracts/c1_5_health_primitive.md` §1；表达层：Shinku 撰写（模块与函数两层 docstring、函数体内两条注释全部新写；四键字典字面量与四个取值表达式按事实逐字迁移） |
| `tests/test_contract_c1_5.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；15 个用例 / 3 个测试类 / 2 个 subTest 点，含**必须报错**断言（位置参数与关键字参数均 `TypeError`）、**跨进程活性**断言，以及本仓每批都有的边界类（不得出现旧项目字样、只许标准库导入） |
| `docs/contracts/c1_5_health_primitive.md` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；10 条契约事实 + 5 条可观察行为 + 已知瑕疵登记（§5） |

**对照物判定（规矩六的第四种形态）。** 前三种是：Akane 侧是转发壳、Akane 侧逐字节相同、
Akane 侧存在同类实现。本批是第四种：

- Akane 侧 `companion_v01/health.py` **根本不存在**——上游的 `routes/core.py` 直接
  `from code_shared.health import ...`，所以「Akane 同路径文件」这个默认对照物拿不到；
- 上游真实现是外置包 `code_shared/code_shared/health.py`（507 B）；
- **Shinku 旧侧 `companion_v01/health.py`（428 B）不是转发壳**（没有 `from code_shared ...`），
  而是**完整复制**：`return` 语句逐字相同，差异只有「删掉函数 docstring + 改写模块 docstring」。
  ⇒ 这是 §5.8 那 4 项「本地重写声明不成立」的**同型第 5 例**，且 0.905 是全项目最高的一例。
  同侧还**缺少对应测试**（旧仓 `tests/test_health.py` 不存在，只有上游包自带一份 530 B 的）。

**本批判据不看相似度**（§5.6）。相似度与代码行 diff 仍照录：

| 对比 | 整文件相似度 | 代码行 diff（剔掉注释与 docstring 之后） |
| --- | --- | --- |
| 新实现 vs 上游 | 0.6255 | **除两处 docstring 外逐行相同**（导入、`def` 行、`return` 块、`__all__`） |
| 新实现 vs Shinku 旧侧 | 0.6473 | 同上；旧侧函数 docstring 已被删 |
| Shinku 旧侧 vs 上游 | 0.8171 | 只有两处 docstring 差异 |

**这张表本身是规矩七的样本：** 新实现的整文件相似度（0.6255）**低于**旧侧的（0.8171），
唯一原因是新 docstring 更长——整文件比值对「表达层有没有被重写」没有分辨力。

**为什么一个零逻辑的迁移批仍然做了对拍与变异（本批超出迁移批最低要求）：**
本模块虽无分支，却有**运行时读取**（`os.getpid()`／`sys.executable`／依赖探针），
也就是有可观察行为。不跑对拍，「值语义逐字迁移」就只是一句主张；不跑变异，
「测试有效」也一样。实测：对拍 5 组 / 21 次比对 / **0 差异**（含跨进程）；
变异注入 5 个 / 抓住 5 个 / **漏 0 个**。**21 次比对是个小数字**，因为本模块没有参数、
没有分支，语料无法拓宽——不拿次数冒充强度（对比 C2-1 的 2,278 次）。

**依赖变更：无。** 只用标准库（`importlib.util`／`os`／`sys`／`typing`）。
**`yt_dlp` 是探测对象，不是依赖**——本批明确不把它写进 `pyproject.toml`。
推论（已记入契约文档 §5）：在新仓接入附件下载链路（C4）之前，该字段**恒为 `False`**。

**本批未做的事：** 未装配——`api/app.py` 的 `/health`（B1 写就，12 个字段的配置回显）
**不改**；它与本原语的字段只有 `status`／`pid` 两项重叠，合并是一次架构决定，
不属于一个模块迁移批。接入点已记入契约文档 §6 与 §5 的未决项。

## 5. 当前未决

#### 4.2.7 C2-2 LLM 传输客户端 —— 洁净室重写（C2 的真正前置）

**为什么 C2-2 是 `llm_client` 而不是计划里的 `model_service`：** 读上游代码发现
`code_shared/model_service.py` 第 16 行在**模块级** `from .llm_client import build_llm_client`，
而 `build_llm_client` 的 anthropic 分支直接 `return AnthropicCompatClient(...)`——
`llm_client.py` 388 行里约 365 行都是这个类与它的私有助手，**切片≈整份文件**。
按「传递依赖随行」（C2-1 先例），传输层必须先落。C2 剩余批次就此重排：
C2-3 = `model_service` + `model_service_config`（一对，必须同批）、
C2-4 = `routes/model_services`、C2-5 = `provider_probe` + `llm_runtime`。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/llm/client.py` | 2026-09-19 | **洁净室重写** | `code_shared/llm_client.py`（上游真实现 13,660 B / 388 行） | 契约 `docs/contracts/c2_2_llm_transport.md`；表达层：Shinku 撰写（私有助手整组换名：`_convert_messages`→`_split_roles`、`_flatten_content_to_text`→`_as_plain_text`、`_convert_content_blocks`→`_as_content_blocks`、`_build_anthropic_payload`→`_prepare_request`（并抽成 `_PreparedRequest` 数据类）、`_raise_for_status`→`_reject_bad_status`、`_build_openai_style_response`→`_as_chat_completion`、`_AnthropicStream`→`_MessageEventStream`、`_capture_stream_usage`→`_absorb_usage`、`_chunk_from_sse_event`→`_event_to_chunk`、`_build_stream_chunk`→`_delta_chunk`、`_map_finish_reason`→`_canonical_finish_reason`；常量具名化：`_ANTHROPIC_API_VERSION`／`_DEFAULT_MAX_TOKENS`／`_SYSTEM_CACHE_SLOTS`／`_STOP_REASON_MAP`／`_INLINE_IMAGE_RE`／`_KEY_PLACEHOLDERS`） |
| `src/shinku/llm/__init__.py` | 2026-09-19 | `INDEPENDENT_KEEP` | 无（包导出面） | C2-1 新建；本批扩 `__all__` 收 `AnthropicCompatClient` 与 `build_llm_client` |
| `tests/test_contract_c2_2.py` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；传输层全换假件（`requests.post` 记录入参、`openai.OpenAI` 记录构造参数），不发起真实网络 |
| `docs/contracts/c2_2_llm_transport.md` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；含 §0.4 三处「与 Shinku 旧侧的有意分叉」及两个决定 |
| `pyproject.toml`（dependencies） | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本批新增 `openai` + `requests`——第一个真正碰网络传输的模块，逐项加、不整份搬旧 requirements |

**对照物判定（规矩六，本批是「两侧都有」的 B 组形态）：** Akane 侧
`services/llm_client.py` 仅 **839 B**，主体是 `from code_shared.llm_client import *`
再加一个 `_akane_protocol` 标记——**转发壳，不能当对照物**；对照物取 `code_shared`
真实现。Shinku 旧侧 `services/llm_client.py`（13,167 B）与上游 body 相似度仅
**0.3115**——旧仓迁移文档写它的设计意图是「兼容包装，只保留旧导入路径与
`_shinku_protocol` 标记」，但实测改写幅度远超设计意图（全量类型标注、私有助手整组
改名、请求组装抽出 `_AnthropicRequest`）；**不作来源**，只进 `LEGACY_NAMES` 并集。

**两处「旧侧独有物」不继承（契约 §0.4 决定 2／3）：** `client._shinku_protocol`
兼容标记（旧仓 `llm_runtime.py` 读它 9 次——那是**消费侧兼容**，不是与上游的契约；
上游 docstring 明说项目包装层"可以"加、非必须）与 `_build_anthropic_payload`
私有垫片（两仓旧测试 import 它；上游那份是真实实现，属表达层）。新仓运行时
同批重写，统一读规范属性 `protocol`。C7 复查时确认无残留消费者。

**六项证据：**

1. 契约文档 `docs/contracts/c2_2_llm_transport.md`（端点／请求头／请求体改写表／
   回装结构／SSE 事件表／失败模式，§0.4 三处有意分叉各配理由）；
2. 独立测试 `tests/test_contract_c2_2.py`（**100 个用例 / 12 个测试类 / 134 个 subtest**，
   全程假传输层）；
3. 实现独立：旧侧 **32 个私有名**（「对照物 ∪ Shinku 旧实现」并集，`kit.py namegap`
   核对覆盖完整、0 缺口）在新文件里**零命中**（精确标识符 token 比对）；
4. 散文独立：docstring + 注释、连续 ≥12 字符逐字片段 **0 处**；
5. **行为等价性对拍 `_c2_2_parity.py`：2,212 次比对、0 处差异**。六组：`build_llm_client`
   224 组参数网格、构造归一化 180 组、请求体改写 375 组（15 类消息 × 25 类覆盖）、
   非流式回装 11 组、流式 SSE 8 组事件流 × 2 种 base_url、失败路径 9 组。
   两侧传输层都换假件后比对（`requests.post` 全入参 + `openai.OpenAI` 构造参数）；
   生成 id 的随机 uuid hex 只比「是否生成」；
6. **变异测试：注入 32 个典型缺陷，全部被抓住、0 漏**，恢复后 sha256 与初始值一致。

**依赖变更：`+openai` `+requests`。** 旧侧把 `httpx` 一并带进来的做法**不学**——
本模块经 SDK 与直发 HTTP 实测只用到这两个。

**本批未做的事：** 未装配——`build_llm_client` 当前**无消费者**（`llm_runtime` 属 C2-5，
`model_service` 属 C2-3），属「可被装配的传输件」；未碰 `tts_client.py`
（Akane 侧也是壳，Shinku 旧侧 11,274 B 独立改写 sim=0.4754，排期属 C6 声音链路）。

#### 4.2.8 C2-3 模型服务配置原语 + 真红侧供应商目录 —— 洁净室重写（两个模块同批）

**为什么这两个必须同批：** Shinku 旧侧 `companion_v01/model_service_config.py` 第 19 行起
整段 `from .model_service import (...)`——它是**上游原语的项目侧入口**（目录、别名、默认模型、
应用策略留在项目，值对象、归一化、校验与注册表实现在上游）。两者是一对，拆批会让第二批的
对照物变成「已有落点的模块」，判定口径跟着歪。落点命名按 C2-2 定的词汇：
`providers/service.py`（与项目无关的原语）+ `providers/catalog.py`（项目侧目录与策略）。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/providers/service.py` | 2026-09-19 | **洁净室重写** | `code_shared/model_service.py`（上游真实现 23,836 B / 637 行） | 契约 `docs/contracts/c2_3_model_service.md`；表达层：Shinku 撰写（私有名整组另起：`_coalesce`／`_inherit`／`_write_registry`；公开面 `bool_value`／`bounded_int`／`safe_int`／`probe_metadata`／`public_provider_entry` 是契约事实，按原名迁移） |
| `src/shinku/providers/catalog.py` | 2026-09-19 | **洁净室重写** | **上游无对应物**（规矩二十五）；行为基线取 Shinku 旧侧 `companion_v01/model_service_config.py` 14,198 B / 389 行 | 契约同上 §1.7；表达层：Shinku 撰写（私有名 `_as_provider_id`／`_vision_model_fits_provider`／`_sync_native_tool_policy`） |
| `tests/test_contract_c2_3.py` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；探测与自检全程假传输层（`requests.get` 记录入参、`build_llm_client` 换假工厂），注册表读写落在临时目录 |
| `docs/contracts/c2_3_model_service.md` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；§0 三项决定（对拍基线 / 键序 / 配置模块属性名）各配理由 |

**对照物判定（本批是规矩二十五的两种形态，必须分开登记）：**

| 模块 | 上游 `code_shared` | Akane 侧 | Shinku 旧侧 | 形态 |
| --- | --- | --- | --- | --- |
| `model_service.py` | **23,836 B / 637 行（对照物）** | 无此文件 | 19,104 B / 348 行 | 上游有真实现 |
| `model_service_config.py` | **无此文件** | 4,062 B / 128 行（**接线核对用**） | 14,198 B / 389 行（**行为基线**） | 上游无对应物 |

`model_service_config` 属**项目侧策略**，上游共享包里本来就没有它——Akane 侧那份（4,062 B）
只有「哪些是共享契约、哪些是项目策略」这一个用途，**不拿来抄**，行为基线是 Shinku 旧侧
那份 389 行（GLM 预置、别名、默认模型、视觉守卫、原生工具联动都在里面）。
旧侧 `model_service.py`（348 行 vs 上游 637 行，调用形态是**上一代**：位置参数
`normalize_api_protocol(str(...), base)`、`build_llm_client` 从 `services.llm_client` 导入）
是比 C2-2 的 `services/llm_client.py` 更早期的一次改写，**仍只进 `LEGACY_NAMES` 并集**。

**三项契约决定（契约 §0）：** ① 对拍基线——`catalog` 侧对**真红旧实现**而非 Akane 壳，
因为 Akane 那份没有 GLM／别名／环境变量兜底／视觉守卫／原生工具联动；
② `model_service_settings_payload` 的**键序跟上游**（pop 式 ⇒ `protocol` 键在最前），
不跟旧侧（按字段顺序现搭 ⇒ `providerId` 在最前）——JSON 对象键序不是语义，
但「跟哪一侧」必须钉住，故对拍里单开一组比 `list(payload.keys())`；
③ 配置模块的属性名（`CHAT_*` / `VISION_*` / `ENABLE_NATIVE_TOOL_DECISION` 等）**按原名继承**，
属契约事实，C4/C6 装配时复查。

**六项证据：**

1. 契约文档 `docs/contracts/c2_3_model_service.md`（10,455 B / 187 行：常量、值对象字段、
   四个错误码、两条取值链的语义差别、函数签名与参数顺序、注册表文件格式、
   目录视图 20 个键、真红侧策略，§0 三项决定各配理由）；
2. 独立测试 `tests/test_contract_c2_3.py`（**167 个用例 / 17 个 TestCase 类 + 8 个假件类 /
   23 个 subTest 调用点**，76,599 B / 1,536 行）；
3. 实现独立：旧私有名并集 **9 个**（「对照物 ∪ Shinku 旧实现」，`kit.py namegap` 核对
   覆盖完整、0 缺口 0 多出）在两个新文件里**零命中**（精确标识符 token 比对）；
4. 散文独立：docstring + 注释、连续 ≥12 字符逐字片段 **0 处**；
5. **行为等价性对拍 `_c2_3_parity.py`：812 次比对、0 处差异**。八组：设置归一化 452、
   目录侧 112、校验 18、存取 50（含 7 组落盘内容，用 `json.loads` 比内容不比字节）、
   探测与自检 14、目录条目 48、整页与应用 14、纯函数助手 104；
6. **变异测试：注入 60 个典型缺陷，全部被抓住、0 漏**，恢复后 sha256 与初始值一致。

**全量回归：865 passed / 0 failed / 2,170 subtests**（C2-2 基线 698，本批 +167）。
`src/shinku` 由 38 个 .py / 193,258 B → **40 个 .py / 236,926 B**。

**本批未做的事：** 未装配——`ModelServiceConfigStore` / `public_model_services_snapshot` /
`apply_model_service_settings` 当前**无消费者**（`routes/model_services` 属 C2-4，
`provider_probe` 与 `llm_runtime` 属 C2-5），属「可被装配的配置件」。

#### 4.2.9 C2-4 模型服务控制中心路由 —— 洁净室重写（上游无路由层，规矩二十五第二例）

**为什么这一批是「上游无对应物」的第二例：** `code_shared` 共享包里**根本没有路由层**——
59 个模块既没有 `model_services.py`，连 `routes/` 子目录都不存在。路由是项目侧把原语
拼成 HTTP 面的那一层：上游只提供 `model_service`／`model_service_config` 原语（C2-3 已落），
不提供把它们挂到 `/control-center/model-services` 的代码。所以本批对照物**不能取上游**，
只能取 Akane 侧同路径文件（8,014 B / 242 行）做**接线核对**——分辨「哪些是共享契约、
哪些是 Shinku 项目策略」——行为基线则取 Shinku 旧侧同路径（16,818 B / 397 行），
因为 Akane 那份是**上一代单服务形态**（只有 `GET/POST /control-center/model-service` +
`/models` + `/test`，没有多供应商面），对拍它等于什么都没对。

| 路径 | 进库日期 | 分类 | 契约 | 依据（契约文档 / 表达层） |
| --- | --- | --- | --- | --- |
| `src/shinku/api/model_services.py` | 2026-09-19 | **洁净室重写** | **上游无路由层**（规矩二十五）；接线核对取 Akane 侧 `companion_v01/routes/model_services.py` 8,014 B / 242 行；行为基线取 Shinku 旧侧同路径 16,818 B / 397 行 | 契约 `docs/contracts/c2_4_model_services.md`；表达层：Shinku 撰写（按本仓 HTTP 层约定落 `api/`，沿用 `api/sessions.py` 的 `build_sessions_router` 先例；私有助手整组另起：`_caller_is_local`／`_json_body`／`_saved_api_key_for`／`_emit_metric`／`_emit_log`／`_truthy`／`_normalize_model_ids`；整页快照读取、按供应商保存、刷新模型、切换激活项、本地请求闸门、度量与结构化日志均为独立实现） |
| `tests/test_contract_c2_4.py` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；两侧各挂真 `TestClient` 打相同请求比对 status/body/cache-control/配置模块变更/注册表字节 |
| `docs/contracts/c2_4_model_services.md` | 2026-09-19 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；§0 三项决定（对拍基线 / 丢掉的单服务端点 / 配置模块属性名）各配理由 |

**对照物判定（规矩二十五第二例「上游无对应物」）：** 上游 `code_shared` 无路由层，
Akane 侧 `companion_v01/routes/model_services.py` 仅 242 行且是**上一代单服务形态**，
既不能当真实现也不能当行为基线，**只用于接线核对**（分辨共享契约 vs 项目策略 +
进 `LEGACY_NAMES` 并集）；真红的行为基线是旧侧 397 行（多供应商面、`/{provider_id}/config`、
`/{provider_id}/models`、`/select` 都在里面）。

**本批丢掉的 4 个向后兼容单服务端点（登记为契约决定，契约 §0.2）：** 旧侧 397 行里有
`# Backward-compatible single-service endpoints` 一组（`/control-center/model-service` 单数、
`/models`、`/test`）。全仓 grep `control-center/model-service`（单数）结果显示：**Shinku 仓内
除它自己与 `docs/PROJECT_MAP.md` 一行过时描述外零消费者**；唯一消费方是 **Akane 自己的**
`web/app.js` 与 `desktop_pet_next`——另一个产品。⇒ 本批不搬这 4 个端点，
`connection_tester`／`require_model` 参数随之不需要。这条有 grep 证据支撑，不是「顺手删」。

**六项证据：**

1. 契约文档 `docs/contracts/c2_4_model_services.md`（13,107 B / 273 行：端点清单、请求/响应
   字段、本地请求闸门、错误信封、度量与日志事件名、配置写回模块的属性名、§0 三项决定各配理由）；
2. 独立测试 `tests/test_contract_c2_4.py`（**72 个用例 / 16 个测试类（3 个 TestCase + 13 个
   假件类）/ 14 个 subTest 调用点**，37,905 B / 892 行）；
3. 实现独立：旧私有名并集 **9 个**（「对照物 ∪ Shinku 旧实现」，`kit.py namegap` 核对
   覆盖完整、0 缺口 0 多出）在新文件里**零命中**（精确标识符 token 比对）；
4. 散文独立：docstring + 注释、连续 ≥12 字符逐字片段 **0 处**；
5. **行为等价性对拍 `_c2_4_parity.py`：154 次比对、0 处差异**。两侧各挂真 `TestClient`、
   打相同请求比对 status/body/`cache-control`/配置模块变更/注册表字节；探测回调由双方各自
   注入假件，不替换传输层（本批碰 `store`／`engine` 注入面，不碰 `requests`）；
6. **变异测试：注入 59 个典型缺陷，全部被抓住、0 漏**，恢复后逐文件 sha256 与初始值一致。

**全量回归：937 passed / 0 failed / 2,235 subtests**（C2-3 基线 865，本批 +72）。
`src/shinku` 由 40 个 .py / 236,926 B → **41 个 .py / 252,566 B**。

**本批暴露的一个覆盖盲区（写进 §5 验收缺陷，供后续批次警惕）：** `set_active` 变异在
**单供应商**场景下被 `load_registry` 的兜底（`active not in providers` 时回退
`next(iter(providers))`）**掩盖**——变异把 `set_active=False` 写下的空 `activeProviderId`
被兜底填回唯一供应商，断言永远通过。必须用**两供应商对照**（先激活 glm，再用
`activate=True` 保存 deepseek，断言切到 `deepseek`）才能暴露。这是变异覆盖的真实盲区，
不是测试缺口，已用一条两供应商测试钉住。

**依赖变更：无。** `model_services.py` 只用 FastAPI `APIRouter`／`Request`／`TestClient`
（测试侧）、标准库与注入的 `store`／`engine`／`config_module` 面，未引入任何新依赖；
`declared_dependencies` 与 C2-3 基线一致（10 项）。

**本批未做的事：** 未装配——`build_model_services_router` 当前**未注册进 `api/app.py`**
（属 C4/C6 装配工作）；未碰 `provider_probe.py` 与 `llm_runtime.py`（C2-5）。

## 5. 当前未决

- **C1 四个子批次全部完成**（2026-09-18）：C1-1 契约事实迁移见 §4.2.1；C1-2／C1-3／C1-4
  洁净室重写见 §4.2.2／§4.2.3／§4.2.4。**C1 范围内已无待办模块。**
  C1-5（§4.2.6）是 C1 对账清单的**收口批**，不改变上列四个子批次的登记。
- **C2 起按依赖图切批**（2026-09-18；2026-09-19 依依赖图重排）：C2-1 已完成（§4.2.5）、
  **C2-2 已完成**（§4.2.7，`llm_client` 传输层——`model_service` 模块级依赖它，
  anthropic 分支返回整个 `AnthropicCompatClient`，切片≈整份文件，故先行）、
  **C2-3 已完成**（§4.2.8，`model_service` + `model_service_config` 一对同批）、
  **C2-4 已完成**（§4.2.9，`routes/model_services`，上游无路由层、对照物取 Akane 侧
  8,014 B 仅作接线核对）。下一步 **C2-5 = `provider_probe` + `llm_runtime`**
  （77,579 B 巨石、84 方法，需单独定切批方案）。
- ~~**C 阶段重写清单里尚未排批的，只剩 `health.py`。**~~ **已于 2026-09-18 由 C1-5 了结**
  （见 §4.2.6）——该清单三项全部落地，**清单已空**，无待排批模块。
  ~~C1-4（`public_guard`、`evidence_guard`、`provider_config`、`native_tool_schema`）~~
  **已于 2026-09-18 完成**，见 §4.2.4。
- **C1-2 / C1-3 / C1-4 的装配都未做**：`CorrelationIdMiddleware` 与 sessions 路由没接进
  `api/app.py`；`load_manifest` 没接进任何注册表，`ResourceManifest` 也没接进应用工厂；
  `PublicThinkGuard` 与证据门没接进任何发送路径。三批都只产出「可被装配的工厂与类」，
  **没有改到运行期服务面**，B1 的健康面与启动链行为不变。因此三批都不做进程探针，
  理由记在各自的批次记录里。
- **已纳入 C 阶段重写清单的 3 个模块 —— 三项全部了结**：`capability_safety.py`
  由 C1-3 随行落地（§4.2.3）、`huggingface_provider.py` 由 C2-1 随行落地（§4.2.5）、
  `health.py` 由 **C1-5** 了结（§4.2.6）。**该清单已空。**
- **`/health` 的「双轨」是未决项，不是遗漏**（2026-09-18 C1-5 记）：B1 的 `api/app.py`
  里那个 12 字段的 `/health`（配置回显）与本原语（进程事实）字段只重叠 `status`／`pid`。
  合并、并存还是让 B1 那个改成消费本原语，是一次**架构决定**，需要单独拍板；
  在拍板前两边并存，且**本原语当前无消费者**——这一点在台账里说清楚，
  免得后面读的人把「模块存在」误读成「已装配」。
- **`ADMISSION.md` §4 的自动化检查**仍为欠账（import graph 扫描、文本扫描规则重写、
  台账完整性、许可证清单）。
- 旧项目里的 36 个 `UNCONFIRMED` 与 106 个 `REWRITE_REQUIRED` 的处置方向
  **已定**：分类不动（旧项目 `SOURCE_PROVENANCE.md` §5.7），C 阶段范围按 §5.9 收缩。

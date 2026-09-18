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
| `tests/test_contract_c1_5.py` | 2026-09-18 | `INDEPENDENT_KEEP` | 无 | 本仓库新建；13 个用例 / 2 个测试类 / 1 个 subTest 点，含**必须报错**断言（位置参数与关键字参数均 `TypeError`）与**跨进程活性**断言 |
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

- **C1 四个子批次全部完成**（2026-09-18）：C1-1 契约事实迁移见 §4.2.1；C1-2／C1-3／C1-4
  洁净室重写见 §4.2.2／§4.2.3／§4.2.4。**C1 范围内已无待办模块。**
  C1-5（§4.2.6）是 C1 对账清单的**收口批**，不改变上列四个子批次的登记。
- **C2 起按依赖图切批**（2026-09-18）：C2-1 已完成（§4.2.5）。下一批 C2-2 的候选是
  `model_service_config.py` + `model_service.py`——读代码确认两者必须同批（前者的 389 行
  全部在委托后者），且属**传输层**，对拍时要替换 `requests`。
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

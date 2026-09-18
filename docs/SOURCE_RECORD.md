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

## 5. 当前未决

- **C1 剩余批次**：C1-3（`capability_adapters/manifest` + `manifest_loader`、`resource_manifest`）、
  C1-4（`public_guard`、`evidence_guard`、`provider_config`、`native_tool_schema`）——
  两者均为**洁净室重写**，不是契约事实迁移。
  ~~C1-2（`request_context`、`routes/sessions`、`task_artifacts`、`tool_invocation`）~~
  **已于 2026-09-18 完成**，见 §4.2.2。
- **C1-2 的装配未做**：`CorrelationIdMiddleware` 与 sessions 路由都没接进 `api/app.py`，
  本批只产出「可被装配的工厂与中间件类」。因此**本批没有改到任何运行期服务面**，
  B1 的健康面与启动链行为不变（见批次记录「验收」一节对此的说明）。
- **已纳入 C 阶段重写清单的 3 个模块**：`huggingface_provider.py`、`health.py`、
  `capability_safety.py`（见来源记录 §5.9 决定三），归属批次待定。
- **`ADMISSION.md` §4 的自动化检查**仍为欠账（import graph 扫描、文本扫描规则重写、
  台账完整性、许可证清单）。
- 旧项目里的 36 个 `UNCONFIRMED` 与 106 个 `REWRITE_REQUIRED` 的处置方向
  **已定**：分类不动（旧项目 `SOURCE_PROVENANCE.md` §5.7），C 阶段范围按 §5.9 收缩。

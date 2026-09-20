# C2-3 契约：模型服务配置与供应商目录

本批落两个模块：

| 新仓模块 | 对照物 | 关系 |
| --- | --- | --- |
| `shinku.providers.service` | `code_shared/model_service.py`（23,836 B / 637 行） | 上游真实现；**洁净室重写** |
| `shinku.providers.catalog` | 上游无对应物；Akane 侧 4,062 B 仅作接线核对；行为基线取旧侧 `companion_v01/model_service_config.py` | 真红侧策略；**洁净室重写** |

`service` 是**与项目无关**的原语：一份连接参数怎么折出来、怎么校验、怎么存、怎么探测。
`catalog` 是**真红自己的**半边：目录里有哪几家、别名、默认模型、把设置写回项目配置的方式。
上游共享包里没有 `catalog` 的对应物——目录属项目策略，这件事本身就是规矩二十五的
「上游无对应物」形态。

---

## 0. 本批的三个决定

### 决定 1：`catalog` 的对拍对照侧取**旧侧真红**，不是 Akane 侧

Akane 那份 4,062 B 只是共享契约的薄包装（没有 GLM 目录、没有别名、没有环境变量回退、
没有视觉模型守卫、没有原生工具开关），拿它对拍等于什么都没对。
真红的行为基线是旧侧 `companion_v01/model_service_config.py`。
Akane 那份在本批只用于**分辨共享契约与项目策略**（进 `LEGACY_NAMES` 并集）。

### 决定 2：`model_service_settings_payload` 的键序跟上游，不跟旧侧

两代的写法不同：

- 上游（本批对照物）是 **pop 式**：`asdict()` 后逐个 `pop` 并重新赋值 ⇒ `protocol` 排第一，
  `providerId` 等 camelCase 键追加在尾部；
- 旧侧真红（上一代）是**按字段顺序另建字典** ⇒ `providerId` 排第一。

JSON 对象的键序不参与语义（`json.loads` 后无差别），但**落盘字节**会不同。
本批选上游写法：`service` 的对照物就是上游，键序随它。
对拍的 D 组因此**只比解析后的内容**不比字节，键序另在 H 组单独核对（新侧 = 上游，逐键相同）。

### 决定 3：写回项目配置时用的属性名**按原样继承**，记待复查

`apply_model_service_settings` 会写 `TEXT_` / `AUX_` / `CHAT_` / `VISION_` 四个通道的
`_API_KEY` / `_BASE_URL` / `_MODEL_NAME` / `_API_PROTOCOL`，并读写
`ENABLE_NATIVE_TOOL_DECISION` / `NATIVE_TOOL_PROVIDER_ALLOWLIST` / `MODEL_PRESETS`。

这些名字属于**未来配置模块的对外面**，而新仓的配置模块还没落（属 C4/C6）。
现在另起一套名字会凭空造出第二个对不上的契约，所以按原样继承，并在此登记：
**C4/C6 建配置模块时若改了这些名字，本函数随改**；C7 复查是否与旧命名一并清理。

---

## 1. 契约事实（按事实迁移，改了就不是同一个东西）

### 1.1 常量

| 常量 | 值 | 说明 |
| --- | ---: | --- |
| `DEFAULT_TIMEOUT_SECONDS` | `120` | 缺省超时 |
| `MODEL_SERVICE_SCHEMA_VERSION` | `2` | 注册表文件结构版本 |
| `DEFAULT_PROVIDER_ID` | `"openai_compatible"` | 认不出供应商时的兜底 id（`providers.config` 已有，此处再导出） |
| 超时夹逼区间 | `[5, 600]` | 不可解析用缺省值，再夹逼 |
| 错误文案截断 | `600` 字符 | `redact_provider_error` 与 `modelProbeError` 同用 |
| Ollama 缺省端点 | `http://127.0.0.1:11434` | `ollama_tags_endpoint` 在 base_url 为空时用 |
| Anthropic 版本头 | `2023-06-01` | 线路协议，不随改名而改 |

### 1.2 值对象 `ModelServiceSettings`

frozen dataclass，**字段顺序即构造顺序**：

`provider_id` · `protocol` · `base_url` · `api_key` · `chat_model` ·
`use_for_vision = True` · `vision_model = ""` · `timeout_seconds = 120`

两个只读属性：

- `endpoint_configured` = 有 `base_url` 且（`protocol == "ollama"` 或有 `api_key`）；
- `configured` = `endpoint_configured` 且有 `chat_model`。

### 1.3 错误码（字符串即契约，调用方按它分支）

| 错误码 | 触发条件 |
| --- | --- |
| `model_service_protocol_invalid` | 协议不在 `openai` / `anthropic` / `ollama` |
| `model_service_base_url_invalid` | base_url 不以 `http://` / `https://` 开头 |
| `model_service_api_key_missing` | 非 ollama 且无密钥 |
| `model_service_model_missing` | `require_model=True` 且无聊天模型 |
| `model_service_config_invalid` | 注册表文件不是 JSON 对象 |
| `model_service_models_response_invalid` | 模型清单响应形状不对 |
| `model_service_provider_missing` | 激活/记探测结果时该供应商不在注册表里 |
| `model_service_response_invalid` | 自检响应取不到 `choices[0].message.content` |
| `model_service_http_<状态码>` | 对端非 2xx 且没给出 `error.message` |

前四个由 `validate_model_service_settings` 抛 `ValueError`，其余抛 `RuntimeError`。

### 1.4 载荷字段的两种取法（**必须分开**，这是最容易写错的一处）

同一个字段在前端是 camelCase、在后端是 snake_case，两边都收。但**两种链式语义不同**：

| 取法 | 语义 | 用在哪 |
| --- | --- | --- |
| 取第一个**非空**值 | `raw.get("apiKey") or raw.get("api_key") or ""` | `providerId` / `protocol` / `baseUrl` / `apiKey` / `chatModel` / `visionModel` |
| 取第一个**出现的键** | `raw.get("clearApiKey", raw.get("clear_api_key"))` | `clearApiKey` / `useForVision` / `timeoutSeconds` / `lastModelProbeAt` / `modelProbeError` |

第二组的理由是 `False` 与 `0` 是**明确的取值**而不是「没给」：
`clearApiKey=False` 是「别清空旧密钥」，不能因为它是假值就滑到 `clear_api_key` 上。
写成第一种会静默改变行为，且不会让任何既有测试变红。

### 1.5 函数签名（参数名与顺序即契约）

```python
build_model_service_settings(raw, *, presets, aliases=None, default_models=None,
                             existing_api_key="", default_timeout=120, require_model=True)
effective_model_service_settings(config_module, *, infer_provider_id, timeout_seconds=120)
build_public_model_service_snapshot(settings, *, source, providers, load_status="ok")
model_service_settings_payload(settings) -> dict
validate_model_service_settings(settings, *, require_model=True) -> None
redact_provider_error(error, *, api_key="") -> str
probe_model_ids(settings, *, use_ollama_tags=False) -> list[str]
test_model_service(settings) -> str
normalize_model_ids(models) -> list[str]          # 去重 + 去空 + casefold 排序
model_ids_from_payload(payload) -> list[str]
bounded_int(value, *, default, minimum, maximum) -> int
bool_value(value, default) -> bool               # 1/true/yes/on/enabled ↔ 0/false/no/off/disabled
raise_provider_error(response) -> None
anthropic_models_endpoint(base_url) -> str
ollama_tags_endpoint(base_url) -> str
load_and_apply_saved_model_service(*, store, config_module, apply, on_error=None)
safe_int(value) -> int
probe_metadata(raw) -> dict
public_provider_entry(*, preset, settings, metadata, active) -> dict
```

`ModelServiceConfigStore` 的方法面：
`settings_from_mapping` · `canonical_provider_id` · `load` · `save` · `load_registry` ·
`get_provider` · `save_provider` · `activate` · `save_model_probe`。

### 1.7 可选自定义供应商目录

内置 `PROVIDER_PRESETS` 仍是稳定的安全默认目录；部署者可以通过
`SHINKU_PROVIDER_PRESETS_JSON` 追加公开目录元数据，而不修改源码。每项至少包含
`id`、`protocol` 和兼容端点 `baseUrl`，可选 `label`、`description`、`apiKeyRequired`
与 `capabilities`。仅接受 `openai`、`anthropic`、`ollama` 三种协议，供应商 id 使用
小写字母、数字、`_`、`-`，且不能覆盖内置 id；非法项逐条忽略。这个 JSON 不接受密钥，
密钥仍从注册表或现有通道配置读取。

`SHINKU_PROVIDER_DEFAULT_MODELS_JSON` 可以为自定义 id 提供首次启动的默认模型名。
目录加载不改变 `PROVIDER_PRESETS` 常量，因此没有配置时既有目录顺序、载荷和契约保持不变。

### 1.6 注册表文件格式

```json
{ "schemaVersion": 2, "activeProviderId": "glm", "providers": { "<providerId>": { ...设置 + 探测元数据 } } }
```

- **读**：v1（单对象直接落文档根）**就地升级**成 v2 结构返回，不动文件；
  单个供应商记录解析失败就整条跳过；`activeProviderId` 指向空档位时回落到第一个。
- **写**：永远写 v2，原子替换（临时文件 + `os.replace`），且**保留已有的探测元数据**
  （`discoveredModels` / `lastModelProbeAt` / `modelProbeError`）——写设置不能把探测结果清掉。

### 1.7 目录视图（`public_provider_entry`）

16 个键，**不含 `apiKey`**，只报 `hasApiKey`。
`preset` 与 `settings` 可各自为 None（前者「目录里有、没配过」，后者「配过但不在目录里」），
都没有时给一组中性缺省（协议 `openai`、需密钥、能力 `stream/vision/native_tools`）。

### 1.8 真红侧策略（`catalog`）

- 目录 = GLM 在最前 + 通用预置；别名收 `gml` / `智谱` / `智谱ai` / `智谱glm` / `zhipu` / `zhipuai`；
- 默认模型：`glm → glm-4.5-air`、`openai → gpt-4o-mini`、`deepseek → deepseek-chat`、
  `gemini → gemini-2.5-flash`、`anthropic → claude-3-5-sonnet-latest`；
  **但** `settings_from_mapping` 不带默认模型（那是校验用户提交的表单，没填就空着），
  只有注册表读写走默认模型——两处口径不同，是刻意的；
- `probe_model_ids` 走 Ollama 原生清单端点（`use_ollama_tags=True`）；
- 视觉通道：换供应商后残留的跨供应商视觉模型名**不投递**给视觉端点，
  保留已载入的专用视觉配置并记一条 warning；已知供应商按前缀判
  （`glm-` / `deepseek-` / `gpt-,o1,o3,o4,chatgpt-` / `gemini-` / `claude-`），
  自定义与本地服务一律放行。

---

## 2. 表达层（Shinku 自撰，不进对照物比对）

模块 docstring 与全部注释、内部组织与文件切分、私有助手命名与算法写法。
本批新起的私有名：`_coalesce`（取第一个非空值）· `_inherit`（取第一个出现的键）·
`_write_registry` · `_as_provider_id` · `_vision_model_fits_provider` · `_sync_native_tool_policy`，
以及 `_TIMEOUT_FLOOR` / `_TIMEOUT_CEILING` / `_ERROR_DETAIL_LIMIT` / `_OLLAMA_FALLBACK_BASE` /
`_ANTHROPIC_VERSION_HEADER` / `_SPECIFIC_ENV_FIELDS` / `_CHANNEL_PREFIXES` / `_VISION_MODEL_PREFIXES`。

被换掉的旧私有名（两侧并集 9 个，零命中）：
`_save_registry` · `_bool_value` · `_bounded_int` · `_safe_int` · `_canonical_provider_id` ·
`_base_url_matches_provider` · `_public_provider_entry` · `_vision_model_matches_provider` ·
`_apply_native_tool_capability`。

---

## 3. 装配状态与未决

- **未装配**：两个模块都没接进任何调用路径（注册表路径、配置模块、路由都还没有），属 C4/C6。
- **待复查**：决定 3 里的配置模块属性名（C4/C6 建配置模块时确认，C7 一并清理旧命名）。
- 对拍覆盖 812 次比对（A 归一化 452 · B 目录侧 112 · C 校验 18 · D 存取 50 ·
  E 探测与自检 14 · F 目录条目 48 · G 整页与应用 14 · H 助手 104），0 差异。

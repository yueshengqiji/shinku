# C2-4 契约：模型服务控制中心路由

| 新仓模块 | 对照物 | 关系 |
| --- | --- | --- |
| `shinku.api.model_services` | legacy 侧 8,014 B / 242 行（**接线核对**）；行为基线 Shinku 旧侧 `companion_v01/routes/model_services.py` 16,818 B / 397 行 | **上游无对应物**；**洁净室重写** |

`code_shared` 里**没有路由层**——59 个上游模块里既没有 `model_services.py`，
连 `routes/` 目录都不存在。这是规矩二十五「上游无对应物」形态的第二例
（第一例是 C2-3 的 `providers/catalog`）。

legacy 侧那份是**上一代**：只有单服务的 `GET/POST /control-center/model-service`
+ `/models` + `/test`，没有多供应商面。它在本批只用于分辨「哪些是线路契约、
哪些是项目接线」，**不拿来抄**，也**不拿来当对拍基线**。

---

## 0. 本批的两个决定

### 决定 1：行为基线取**旧侧真红 397 行**，不取 legacy 侧 242 行

与 C2-3 `catalog` 同一处置。legacy 那份缺整个多供应商面（`/{provider_id}/config`、
`/{provider_id}/models`、`/select`），对它做对拍等于什么都没对。

### 决定 2：**不搬那 4 个「向后兼容的单服务端点」**

旧侧 397 行里有 4 个标着 `# Backward-compatible single-service endpoints used by
older control-center builds.` 的路由，做法是把同一份能力用单服务的路径再暴露一遍。

**全仓 grep `control-center/model-service`（单数）的证据：**

| 命中位置 | 说明 |
| --- | --- |
| `companion_v01/routes/model_services.py` | 它自己（旧侧） |
| `docs/PROJECT_MAP.md` 一行 | 过时描述：把整个模块摘要成单服务那条路由 |
| **legacy** `web/app.js`、`desktop_pet_next/src/control-center/data-sources.js` | **legacy 自己的前端**，另一个产品 |

**Shinku 侧零消费者**：真红自己的控制中心是 `companion_v01/routes/debug_hub.py`，
它只用多供应商那四个端点（`GET /control-center/model-services`、`/…/config`、
`/…/models`、`/select`）；真红仓库里没有 `web/`、`desktop_pet/`、`desktop_pet_next/`
任何一个前端目录。

⇒ 本批只落多供应商面。`connection_tester` 参数、`require_model=True` 的分支、
`_run_candidate_action` 那个共享助手随之不需要——它们只服务于被丢掉的那四个端点。

代价登记：若哪天要给旧版控制台做兼容，这四个端点得单独一批加回来，**不由本批预留**。

---

## 1. 契约事实（按事实迁移，改了就不是同一个东西）

### 1.1 装配签名

```python
build_model_services_router(
    *,
    store: ModelServiceConfigStore,
    config_module: Any,
    engine: Any,
    runtime_metrics: Any = None,
    log_event: Callable[..., Any] | None = None,
    model_probe: Callable[..., list[str]] = <目录侧 probe_model_ids>,
) -> APIRouter
```

六个参数**全部关键字传入**，顺序如上。`runtime_metrics` 与 `log_event` 可缺省
（缺省即不记度量、不写结构化日志）；其余四个是必填语义上的依赖项。

被依赖的两侧角色：

- `store` —— 注册表读写（`get_provider` / `save_provider` / `save_model_probe` / `activate`）；
- `config_module` —— 项目配置命名空间，`apply_model_service_settings` 与
  `effective_settings_from_config` 消费它；
- `engine` —— 只需要 `reload_model_services()`，返回 `dict`；
- `runtime_metrics` —— 只需要 `observe_request(name, *, duration_ms: float, ok: bool)`；
  两者都用 `hasattr` 探测，**没有 `observe_request` 的对象被当作"没配度量"**。

### 1.2 路由表

| 方法 | 路径 | 处理函数 |
| --- | --- | --- |
| `GET` | `/control-center/model-services` | 整页快照 |
| `POST` | `/control-center/model-services/{provider_id}/config` | 保存某家配置（可选同时激活） |
| `POST` | `/control-center/model-services/{provider_id}/models` | 刷新某家的模型清单 |
| `POST` | `/control-center/model-services/select` | 切换当前激活项 |

四个路由有**顺序要求**吗？没有：`/select` 是常量后缀，`/{provider_id}/…` 是两段，
二者不冲突，注册顺序自由（属表达层）。

### 1.3 本地请求闸门

```python
host = str(getattr(getattr(request, "client", None), "host", "") or "").strip().lower()
hole = host in {"127.0.0.1", "::1", "localhost", "testclient"}
```

非法来源一律 `403`：

```json
{"ok": false, "status": "forbidden", "reason": "local_request_required"}
```

**四个 POST 路由全部要求本地来源；GET 不要求。**
`testclient` 必须在集合里——`fastapi.testclient.TestClient` 默认客户端地址就是它，
没有这一项，所有测试都会被闸门挡住。

### 1.4 `GET /control-center/model-services`

1. 记下起始时刻 `time.perf_counter()`；
2. 调 `public_model_services_snapshot(store, config_module)`；
3. **异常** → 记度量（`ok=False`）+ `500`，body
   `{"ok": false, "status": "invalid_config", "reason": redact_provider_error(exc)}`
   （**不传 `api_key`**，因为没有要说要遮蔽的具体密钥）；
4. 成功 → 记度量（`ok=True`）+ `200`，body 就是快照本身（不额外包 `ok`/`status`）。

### 1.5 `POST /control-center/model-services/{provider_id}/config`

执行顺序（**顺序是契约的一部分**）：

1. 本地闸门；
2. 读请求体 → 非 dict 一律折成 `{}`，解析失败也折成 `{}`；
3. `existing = _load_provider_secret(store, config_module, provider_id)`
   —— 见 §1.6；
4. `payload["providerId"] = provider_id`（**覆盖**，不是 `setdefault`）；
5. `settings = settings_from_mapping(payload, existing_api_key=existing, require_model=False)`
   —— `require_model` 恒 `False`；
6. `activate = _truthy(payload.get("activate", payload.get("setActive", False)))`；
7. `if activate and not settings.chat_model: raise ValueError("model_service_model_missing")`；
8. `store.save_provider(settings, set_active=activate)`；
9. 仅当 `activate`：先 `apply_model_service_settings(config_module, settings)`，
   再 `await asyncio.to_thread(engine.reload_model_services)`；
10. 任一步抛异常 → 度量 `ok=False` + 日志 `status="failed"` + `200`
    （**不是 4xx**，body 自带 `ok=false`）：
    ```json
    {"ok": false, "status": "invalid_config",
     "reason": redact_provider_error(exc, api_key=existing)}
    ```
11. 成功 → `{**public_model_services_snapshot(store, config_module), "refresh": true,
    "runtime": reload_result}`。

**`"runtime"` 在未激活时是 `null`**——`reload_result` 初值是 `None`，
只有 `activate` 为真才会被赋值。这一条容易在迁移时被"顺手改成 `or {"status": ...}`"，
但 `/select`（§1.7）**确实**用了 `or` 兜底，两者不同，不得统一。

### 1.6 密钥继承（两家共用一套规则）

保存某家的配置时，表单里通常不带密钥，密钥要从「已存的那份」继承；
**换供应商时不能继承**——否则会把 A 家的密钥写进 B 家。

```python
try:
    saved = store.get_provider(provider_id)   # 注册表读失败 ⇒ 当作没存过
except Exception:
    saved = None
if saved is not None:
    return saved.api_key
effective = effective_settings_from_config(config_module)
return effective.api_key if effective.provider_id == str(provider_id or "").strip().lower() else ""
```

`store.get_provider` 抛异常被**吞掉当没存过**，这是有意的：注册表损坏不该让保存路由 500。

### 1.7 `POST /control-center/model-services/{provider_id}/models`

1. 本地闸门；
2. 读请求体（同上，非法折 `{}`）；
3. `stored = store.get_provider(provider_id)`，异常 → `invalid_config` + `200`；
4. `stored is None` → `{"ok": false, "status": "provider_not_configured",
   "reason": "先保存该供应方配置，再刷新模型列表。"}` + `200`；
5. 用已存的字段**补表单的空缺**（`setdefault`）：
   `protocol` / `baseUrl` / `chatModel` / `useForVision` / `visionModel` / `timeoutSeconds`
   分别取 `stored.protocol` / `.base_url` / `.chat_model` / `.use_for_vision` /
   `.vision_model` / `.timeout_seconds`；再 `payload["providerId"] = provider_id`（覆盖）；
6. `settings = settings_from_mapping(payload, existing_api_key=stored.api_key,
   require_model=False)`；
7. `if not settings.endpoint_configured: raise ValueError("model_service_config_incomplete")`
   —— 探测只需要端点 + 密钥，模型名本来就是这一步要发现的；
8. `models = await asyncio.to_thread(model_probe, settings)`；
9. 归一化：去空白、丢空串、去重、`key=str.casefold` 升序；
10. `now = int(time.time())`，
    `store.save_model_probe(settings.provider_id, normalized, timestamp=now)`；
11. 异常 → 先用 `redact_provider_error(exc, api_key=stored.api_key)` 算出文案，
    **尽力**写一次失败探测记录（`store.save_model_probe(stored.provider_id, [],
    error=..., timestamp=int(time.time()))`，这次写盘的异常要吞掉），
    再返回 `200` + `{"ok": false, "status": "request_failed", "reason": error}`；
12. 成功 → `200`：
    ```json
    {"ok": true, "status": "available", "providerId": <...>, "models": [...],
     "count": <len>, "lastModelProbeAt": <now>}
    ```

### 1.8 `POST /control-center/model-services/select`

1. 本地闸门；
2. 读请求体；
3. `provider_id` 取键序 `providerId` → `provider_id`，`str(...).strip()`；
4. `model_id` 取键序 `modelId` → `model_id` → `chatModel` → `model`，`str(...).strip()`；
5. `settings = store.activate(provider_id, model_id)`（同步调用，**不走 `to_thread`**）；
6. `apply_model_service_settings(config_module, settings)`；
7. `reload_result = await asyncio.to_thread(engine.reload_model_services)`；
8. 异常 → `200` + `{"ok": false, "status": "select_failed", "reason": redact_provider_error(exc)}`
   （**不传 `api_key`**）；
9. 成功 → `200`：
    ```json
    {**snapshot, "activeProviderId": settings.provider_id, "activeModel": settings.chat_model,
     "runtime": reload_result or {"status": "reloaded"}}
    ```

与 §1.5 的第 11 条对照：**这里 `runtime` 有兜底，那里没有。**

### 1.9 错误信封的状态字符串

| `status` | 出现位置 | HTTP 码 |
| --- | --- | --- |
| `forbidden` | 非本地来源（四个 POST） | 403 |
| `invalid_config` | GET 快照异常；config 保存异常；models 读注册表异常 | 500 / 200 / 200 |
| `provider_not_configured` | models 刷新时该家还没存过 | 200 |
| `request_failed` | models 探测失败 | 200 |
| `select_failed` | 切换激活项异常 | 200 |
| `available` | models 成功 | 200 |
| `ok` 字段成功侧 | config / models / select 成功响应 | 200 |

除 403 与 GET 的 500 之外，**所有业务成败都用 `200` + body 里的 `ok` 表达**。

### 1.10 度量名与结构化日志事件名

| 路由 | 度量名 | 日志事件名 |
| --- | --- | --- |
| GET 快照 | `model_services.read` | —（不写日志） |
| config 保存 | `model_services.save` | `model_services_save` |
| models 刷新 | `model_services.models` | `model_services_models` |
| select 切换 | `model_services.select` | `model_services_select` |

日志字段：

- `model_services_save` 成功：`status="saved"`，`provider_id`、`protocol`、`chat_model`、
  `active=<activate>`；失败：`status="failed"`，`provider_id=<路径里的 provider_id>`；
- `model_services_models` 成功：`status="ok"`，`provider_id=<settings.provider_id>`、
  `count=<len(normalized)>`；失败：`status="failed"`，`provider_id=<路径里的 provider_id>`；
- `model_services_select` 成功：`status="selected"`，`provider_id`、`chat_model`；
  失败：`status="failed"`，`provider_id=<请求体里解析出的 provider_id>`。

度量与日志的**回调自身抛异常必须被吞掉**——观测设施故障不该让业务路由失败。

### 1.11 真值词

`activate` / `setActive` 的取值判定：

```python
if isinstance(value, bool):
    return value
return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}
```

### 1.12 依赖面（本批不新增外部依赖）

`fastapi`（已在 `declared_dependencies`）+ 标准库 `asyncio` / `time`。
另外两处依赖是**本仓自己的模块**：

- `shinku.contracts.http.json_response` —— 统一的 `Cache-Control: no-store` JSON 响应信封；
- `shinku.providers.{catalog,service}` —— 上一批（C2-3）刚落地的 primitive。

只有「宿主侧」的东西（`store` / `config_module` / `engine` / 度量 / 日志 / 探测回调）
走注入；已经属于本仓的那些不注入——它们的形状本来就是本仓定的，
再注入一遍只会让工厂签名虚胖。

---

## 2. 表达层（本批 Shinku 自撰，不属契约）

- 文件切片与私有助手命名；
- `Protocol` 描述 `engine` / `runtime_metrics` 的鸭子类型（沿用本仓 `api/sessions.py` 的写法）；
- 每个路由里「先定 HTTP 成败与信封，再谈业务」的代码组织顺序。

"""模型服务配置原语：一份连接参数的归一化、校验、存取与探测。

一个「模型服务」在这里就是一组可持久化的连接参数：走哪家供应商、用什么协议、
端点在哪、密钥、聊天模型、视觉模型、超时。本模块只负责这一种值对象本身：
怎么从一份项目载荷把它折出来、怎么判它是否可用、怎么存文件、怎么探测对端有哪些模型。

**不负责**「本项目目录里有哪些供应商」——那是项目侧策略，在
:mod:`shinku.providers.catalog`；目录、别名、默认模型都由调用方从参数给进来，
这样「GLM 这类项目自有条目留在本项目」与「字段词表与校验行为一致」能同时成立。

存取两 format 的两条不变量：

- **读**：v1 结构（单个设置对象直接落在文档根）读出来就**就地升级**成 v2 注册表，
  只影响返回值，不动磁盘上的文件；
- **写**：永远写 v2，而且是一次原子替换（先写临时文件再 ``os.replace``），
  不会出现半截文件把注册表毁掉。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from ..llm.client import build_llm_client
from .config import (
    DEFAULT_PROVIDER_ID,
    ProviderPreset,
    canonical_provider_id,
    normalize_api_protocol,
    normalize_configured_base_url,
    preset_index,
)

__all__ = [
    "DEFAULT_PROVIDER_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "MODEL_SERVICE_SCHEMA_VERSION",
    "ModelServiceConfigStore",
    "ModelServiceSettings",
    "anthropic_models_endpoint",
    "bool_value",
    "bounded_int",
    "build_model_service_settings",
    "build_public_model_service_snapshot",
    "effective_model_service_settings",
    "load_and_apply_saved_model_service",
    "model_ids_from_payload",
    "model_service_settings_payload",
    "normalize_model_ids",
    "ollama_tags_endpoint",
    "probe_metadata",
    "probe_model_ids",
    "public_provider_entry",
    "raise_provider_error",
    "redact_provider_error",
    "safe_int",
    "test_model_service",
    "validate_model_service_settings",
]

#: 缺省超时。120 秒是「够跑完一次长回复」的经验值。
DEFAULT_TIMEOUT_SECONDS = 120

#: 注册表文件结构版本。v1 是单个设置对象直接落在文档根；v2 把多家供应商收进一张
#: 映射表，并单独记下「当前激活的是哪一家」。v1 仍可读，**永不写回 v1**。
MODEL_SERVICE_SCHEMA_VERSION = 2

#: 超时的夹逼区间。下界防止配成 0（几乎必然超时），上界防止配成几千秒把连接挂死。
_TIMEOUT_FLOOR = 5
_TIMEOUT_CEILING = 600

#: 错误文案的截断长度。供应商出错时可能返回整页 HTML 或巨型 JSON，
#: 截断后再进日志与注册表里的探测字段，免得把整页内容写进配置文件。
_ERROR_DETAIL_LIMIT = 600

#: Ollama 的本地默认地址：配置里 base_url 留空时用它。
_OLLAMA_FALLBACK_BASE = "http://127.0.0.1:11434"

#: Anthropic 模型列表端点要求的版本头（线路协议，不随本项目改名而改）。
_ANTHROPIC_VERSION_HEADER = "2023-06-01"


def _coalesce(raw: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """依次取第一个「非空」值，全空才用 ``default``。"""

    for key in keys:
        value = raw.get(key)
        if value:
            return value
    return default


def _inherit(raw: Mapping[str, Any], *keys: str) -> Any:
    """按**键是否出现**依次取，前一个键没出现才看后一个。

    与 :func:`_coalesce` 的差别是这里**不看值**：``clearApiKey=False`` 是一句明确的
    「不要清空旧密钥」，不能因为它是假值就滑到 ``clear_api_key`` 上去。
    """

    for key in keys[:-1]:
        if key in raw:
            return raw[key]
    return raw.get(keys[-1])


@dataclass(frozen=True)
class ModelServiceSettings:
    """一个模型服务的连接参数。字段顺序即构造顺序。"""

    provider_id: str
    protocol: str
    base_url: str
    api_key: str
    chat_model: str
    use_for_vision: bool = True
    vision_model: str = ""
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @property
    def endpoint_configured(self) -> bool:
        """端点层面是否齐了：有地址，且本地 Ollama 或已有密钥。"""

        if not self.base_url:
            return False
        if self.protocol == "ollama":
            return True
        return bool(self.api_key)

    @property
    def configured(self) -> bool:
        """是否能直接发一次普通聊天请求。"""

        return self.endpoint_configured and bool(self.chat_model)


def build_model_service_settings(
    raw: Mapping[str, Any],
    *,
    presets: Iterable[ProviderPreset],
    aliases: Mapping[str, str] | None = None,
    default_models: Mapping[str, str] | None = None,
    existing_api_key: str = "",
    default_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    require_model: bool = True,
) -> ModelServiceSettings:
    """把一份项目载荷折成设置对象。

    载荷里同一个字段可能用 camelCase（前端）或 snake_case（后端）两种写法，
    两者都收；「空值」与「键不存在」在这里是两回事（见 :func:`_inherit`）。
    """

    by_id = preset_index(tuple(presets))
    fallback = by_id.get(DEFAULT_PROVIDER_ID)
    if fallback is None:
        fallback = ProviderPreset(
            id=DEFAULT_PROVIDER_ID,
            label="OpenAI compatible",
            protocol="openai",
            base_url="",
            api_key_required=True,
            description="OpenAI-compatible service.",
        )

    provider_id = canonical_provider_id(
        _coalesce(raw, "providerId", "provider_id", default=DEFAULT_PROVIDER_ID),
        aliases=aliases,
    )
    preset = by_id.get(provider_id, fallback)
    declared_url = str(_coalesce(raw, "baseUrl", "base_url", default=preset.base_url))
    protocol = normalize_api_protocol(
        protocol=str(_coalesce(raw, "protocol", default=preset.protocol)),
        base_url=declared_url,
    )
    api_key = str(_coalesce(raw, "apiKey", "api_key", default="")).strip()
    if not api_key and not bool_value(_inherit(raw, "clearApiKey", "clear_api_key"), False):
        api_key = str(existing_api_key or "").strip()
    default_model = str((default_models or {}).get(provider_id) or "").strip()

    settings = ModelServiceSettings(
        provider_id=provider_id if provider_id in by_id else DEFAULT_PROVIDER_ID,
        protocol=protocol,
        base_url=normalize_configured_base_url(declared_url, protocol=protocol),
        api_key=api_key,
        chat_model=str(
            _coalesce(raw, "chatModel", "chat_model", "model", default=default_model)
        ).strip(),
        use_for_vision=bool_value(_inherit(raw, "useForVision", "use_for_vision"), True),
        vision_model=str(_coalesce(raw, "visionModel", "vision_model", default="")).strip(),
        timeout_seconds=bounded_int(
            _inherit(raw, "timeoutSeconds", "timeout_seconds"),
            default=default_timeout,
            minimum=_TIMEOUT_FLOOR,
            maximum=_TIMEOUT_CEILING,
        ),
    )
    validate_model_service_settings(settings, require_model=require_model)
    return settings


def effective_model_service_settings(
    config_module: Any,
    *,
    infer_provider_id: Callable[..., str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ModelServiceSettings:
    """从项目配置模块读「当前生效的聊天通道」。

    怎么把通道读出来由调用方给 ``infer_provider_id``，本函数只认
    ``CHAT_*`` / ``VISION_MODEL_NAME`` 这几个属性名，不 import 任何配置类型。
    """

    base_url = str(getattr(config_module, "CHAT_BASE_URL", "") or "").strip()
    protocol = normalize_api_protocol(
        protocol=str(getattr(config_module, "CHAT_API_PROTOCOL", "auto") or "auto"),
        base_url=base_url,
    )
    vision_model = str(getattr(config_module, "VISION_MODEL_NAME", "") or "").strip()
    return ModelServiceSettings(
        provider_id=infer_provider_id(protocol=protocol, base_url=base_url),
        protocol=protocol,
        base_url=base_url,
        api_key=str(getattr(config_module, "CHAT_API_KEY", "") or "").strip(),
        chat_model=str(getattr(config_module, "CHAT_MODEL_NAME", "") or "").strip(),
        use_for_vision=bool(vision_model),
        vision_model=vision_model,
        timeout_seconds=timeout_seconds,
    )


def build_public_model_service_snapshot(
    settings: ModelServiceSettings,
    *,
    source: str,
    providers: list[dict[str, Any]],
    load_status: str = "ok",
) -> dict[str, Any]:
    """给前端的快照：**不含密钥**，只报有没有。"""

    return {
        "ok": True,
        "status": "configured" if settings.configured else "missing_config",
        "loadStatus": load_status,
        "source": source,
        "providerId": settings.provider_id,
        "protocol": settings.protocol,
        "baseUrl": settings.base_url,
        "hasApiKey": bool(settings.api_key),
        "endpointConfigured": settings.endpoint_configured,
        "chatModel": settings.chat_model,
        "useForVision": settings.use_for_vision,
        "visionModel": settings.vision_model,
        "timeoutSeconds": settings.timeout_seconds,
        "providers": providers,
    }


def model_service_settings_payload(settings: ModelServiceSettings) -> dict[str, Any]:
    """落盘/出库用的 camelCase 形态（含 apiKey，只在服务端流转）。"""

    payload = asdict(settings)
    payload["providerId"] = payload.pop("provider_id")
    payload["baseUrl"] = payload.pop("base_url")
    payload["apiKey"] = payload.pop("api_key")
    payload["chatModel"] = payload.pop("chat_model")
    payload["useForVision"] = payload.pop("use_for_vision")
    payload["visionModel"] = payload.pop("vision_model")
    payload["timeoutSeconds"] = payload.pop("timeout_seconds")
    return payload


def validate_model_service_settings(
    settings: ModelServiceSettings,
    *,
    require_model: bool = True,
) -> None:
    """校验；不通过抛 :class:`ValueError`，消息是稳定的错误码。"""

    if settings.protocol not in {"openai", "anthropic", "ollama"}:
        raise ValueError("model_service_protocol_invalid")
    if not settings.base_url.startswith(("http://", "https://")):
        raise ValueError("model_service_base_url_invalid")
    if settings.protocol != "ollama" and not settings.api_key:
        raise ValueError("model_service_api_key_missing")
    if require_model and not settings.chat_model:
        raise ValueError("model_service_model_missing")


def redact_provider_error(error: Exception, *, api_key: str = "") -> str:
    """把供应商错误压成可展示的一行：去掉密钥，截断长度。"""

    text = str(error or "").strip() or error.__class__.__name__
    if api_key:
        text = text.replace(api_key, "<redacted>")
    return text[:_ERROR_DETAIL_LIMIT]


def probe_model_ids(
    settings: ModelServiceSettings,
    *,
    use_ollama_tags: bool = False,
) -> list[str]:
    """问对端「你有哪些模型」。

    Anthropic 与 Ollama 有各自的原生端点（``/v1/models``、``/api/tags``），
    其余走 SDK 的 ``models.list()``——三方返回的形状不同，这里统一折成排序后的 id 列表。
    """

    validate_model_service_settings(settings, require_model=False)
    if settings.protocol == "anthropic":
        response = requests.get(
            anthropic_models_endpoint(settings.base_url),
            headers={
                "x-api-key": settings.api_key,
                "anthropic-version": _ANTHROPIC_VERSION_HEADER,
            },
            timeout=settings.timeout_seconds,
        )
        raise_provider_error(response)
        return model_ids_from_payload(response.json())

    if settings.protocol == "ollama" and use_ollama_tags:
        response = requests.get(
            ollama_tags_endpoint(settings.base_url),
            timeout=settings.timeout_seconds,
        )
        raise_provider_error(response)
        payload = response.json()
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("model_service_models_response_invalid")
        return normalize_model_ids(
            [item.get("name") for item in models if isinstance(item, dict)]
        )

    client = build_llm_client(
        api_key=settings.api_key,
        base_url=settings.base_url,
        protocol=settings.protocol,
        timeout=float(settings.timeout_seconds),
        max_retries=0,
    )
    response = client.models.list()
    raw_models = getattr(response, "data", response)
    ids: list[str] = []
    for item in raw_models or []:
        model_id = getattr(item, "id", "")
        if not model_id and isinstance(item, dict):
            model_id = item.get("id", "")
        if model_id:
            ids.append(str(model_id))
    return normalize_model_ids(ids)


def test_model_service(settings: ModelServiceSettings) -> str:
    """做一次最小连通性自检：要一个八 token 的「OK」。"""

    validate_model_service_settings(settings)
    client = build_llm_client(
        api_key=settings.api_key,
        base_url=settings.base_url,
        protocol=settings.protocol,
        timeout=float(settings.timeout_seconds),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "Reply with only OK."}],
        temperature=0,
        max_tokens=8,
    )
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("model_service_response_invalid") from exc
    return str(text or "").strip() or "OK"


def normalize_model_ids(models: Any) -> list[str]:
    """去重 + 去空 + 按大小写无关排序。非序列一律当空。"""

    if not isinstance(models, (list, tuple, set)):
        return []
    values = {str(item or "").strip() for item in models if str(item or "").strip()}
    return sorted(values, key=str.casefold)


def model_ids_from_payload(payload: Any) -> list[str]:
    """从 ``{"data": [{"id": ...}]}`` 里取 id；形状不对抛错。"""

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("model_service_models_response_invalid")
    return normalize_model_ids(
        [item.get("id") for item in data if isinstance(item, dict)]
    )


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """取整后夹逼；不可解析用缺省值。"""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def bool_value(value: Any, default: bool) -> bool:
    """宽容地读布尔：认 ``1/true/yes/on/enabled`` 与 ``0/false/no/off/disabled``。

    认不出来的（含 ``None``）一律回 ``default``——配置项写错时不猜。
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def raise_provider_error(response: Any) -> None:
    """非 2xx 就抛 :class:`RuntimeError`，消息优先取对端给的 ``error.message``。"""

    if response.ok:
        return
    message = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "").strip()
            elif error:
                message = str(error).strip()
    except Exception:
        message = str(response.text or "").strip()
    raise RuntimeError(message or f"model_service_http_{response.status_code}")


def anthropic_models_endpoint(base_url: str) -> str:
    """Anthropic 的模型列表端点：``.../v1``（或 ``.../v1/messages``）→ ``.../v1/models``。"""

    clean = str(base_url or "").strip().rstrip("/")
    if clean.endswith("/v1/messages"):
        clean = clean[: -len("/messages")]
    if clean.endswith("/v1"):
        return f"{clean}/models"
    return f"{clean}/v1/models"


def ollama_tags_endpoint(base_url: str) -> str:
    """Ollama 的本地模型清单端点；base_url 为空时用本机默认地址。"""

    clean = str(base_url or "").strip().rstrip("/")
    if clean.endswith("/v1"):
        clean = clean[:-3].rstrip("/")
    return f"{clean or _OLLAMA_FALLBACK_BASE}/api/tags"


class ModelServiceConfigStore:
    """供应商注册表的读写入口。

    目录、别名、默认模型从构造参数进来，所以「同一份注册表实现」可以给不同项目用，
    各自的目录与别名互不干扰。
    """

    def __init__(
        self,
        path: Path,
        *,
        presets: Iterable[ProviderPreset],
        aliases: Mapping[str, str] | None = None,
        default_models: Mapping[str, str] | None = None,
        schema_version: int = MODEL_SERVICE_SCHEMA_VERSION,
    ) -> None:
        self.path = Path(path)
        self.presets: tuple[ProviderPreset, ...] = tuple(presets)
        self.aliases: dict[str, str] = dict(aliases or {})
        self.default_models: dict[str, str] = dict(default_models or {})
        self.schema_version = int(schema_version or MODEL_SERVICE_SCHEMA_VERSION)

    # --- 读写 ---
    def settings_from_mapping(
        self,
        raw: dict[str, Any],
        *,
        existing_api_key: str = "",
        require_model: bool = True,
    ) -> ModelServiceSettings:
        return build_model_service_settings(
            raw,
            presets=self.presets,
            aliases=self.aliases,
            default_models=self.default_models,
            existing_api_key=existing_api_key,
            default_timeout=DEFAULT_TIMEOUT_SECONDS,
            require_model=require_model,
        )

    def canonical_provider_id(self, value: Any) -> str:
        return canonical_provider_id(value, aliases=self.aliases)

    def load(self) -> ModelServiceSettings | None:
        """当前激活的那一家；没配过或指向空档位就返回 None。"""

        registry = self.load_registry()
        active_provider_id = str(registry.get("activeProviderId") or "").strip()
        providers = registry.get("providers")
        if not isinstance(providers, dict):
            return None
        raw = providers.get(active_provider_id)
        if not isinstance(raw, dict):
            return None
        return self.settings_from_mapping(raw, require_model=False)

    def save(self, settings: ModelServiceSettings) -> None:
        self.save_provider(settings, set_active=True)

    def load_registry(self) -> dict[str, Any]:
        """读整份注册表；v1 文件在这里被升级成 v2 结构返回。"""

        if not self.path.exists():
            return {
                "schemaVersion": self.schema_version,
                "activeProviderId": "",
                "providers": {},
            }
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("model_service_config_invalid")

        if isinstance(data.get("providers"), dict):
            providers: dict[str, dict[str, Any]] = {}
            for raw_provider_id, raw in data["providers"].items():
                if not isinstance(raw, dict):
                    continue
                candidate = dict(raw)
                candidate["providerId"] = self.canonical_provider_id(
                    raw.get("providerId") or raw_provider_id
                )
                try:
                    settings = self.settings_from_mapping(candidate, require_model=False)
                except Exception:
                    continue
                normalized = model_service_settings_payload(settings)
                normalized.update(probe_metadata(raw))
                providers[settings.provider_id] = normalized
            active = self.canonical_provider_id(
                data.get("activeProviderId") or data.get("active_provider_id")
            )
            if active not in providers:
                active = next(iter(providers), "")
            return {
                "schemaVersion": self.schema_version,
                "activeProviderId": active,
                "providers": providers,
            }

        settings = self.settings_from_mapping(data, require_model=False)
        return {
            "schemaVersion": self.schema_version,
            "activeProviderId": settings.provider_id,
            "providers": {settings.provider_id: model_service_settings_payload(settings)},
        }

    def get_provider(self, provider_id: str) -> ModelServiceSettings | None:
        provider_id = self.canonical_provider_id(provider_id)
        registry = self.load_registry()
        raw = (registry.get("providers") or {}).get(provider_id)
        if not isinstance(raw, dict):
            return None
        return self.settings_from_mapping(raw, require_model=False)

    def save_provider(
        self, settings: ModelServiceSettings, *, set_active: bool = False
    ) -> None:
        """写一家进去；已探测到的模型清单等元数据不会被这次写入清掉。"""

        registry = self.load_registry()
        providers = registry.setdefault("providers", {})
        previous = providers.get(settings.provider_id)
        payload = model_service_settings_payload(settings)
        if isinstance(previous, dict):
            payload.update(probe_metadata(previous))
        providers[settings.provider_id] = payload
        if set_active:
            registry["activeProviderId"] = settings.provider_id
        self._write_registry(registry)

    def activate(self, provider_id: str, model_id: str = "") -> ModelServiceSettings:
        """把某家设为当前，并可顺手换模型。"""

        provider_id = self.canonical_provider_id(provider_id)
        settings = self.get_provider(provider_id)
        if settings is None:
            raise ValueError("model_service_provider_missing")
        target_model = str(model_id or settings.chat_model).strip()
        if not target_model:
            raise ValueError("model_service_model_missing")
        active = replace(settings, chat_model=target_model)
        validate_model_service_settings(active, require_model=True)
        self.save_provider(active, set_active=True)
        return active

    def save_model_probe(
        self,
        provider_id: str,
        models: list[str],
        *,
        error: str = "",
        timestamp: int = 0,
    ) -> None:
        """记一次探测结果：模型清单 + 时间戳 + 错误（成功时错误串为空）。"""

        provider_id = self.canonical_provider_id(provider_id)
        registry = self.load_registry()
        providers = registry.setdefault("providers", {})
        raw = providers.get(provider_id)
        if not isinstance(raw, dict):
            raise ValueError("model_service_provider_missing")
        raw["discoveredModels"] = normalize_model_ids(models)
        raw["lastModelProbeAt"] = max(0, int(timestamp or 0))
        raw["modelProbeError"] = str(error or "")[:_ERROR_DETAIL_LIMIT]
        providers[provider_id] = raw
        self._write_registry(registry)

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": self.schema_version,
            "activeProviderId": self.canonical_provider_id(
                registry.get("activeProviderId")
            ),
            "providers": registry.get("providers")
            if isinstance(registry.get("providers"), dict)
            else {},
        }
        temp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_path, self.path)


def load_and_apply_saved_model_service(
    *,
    store: ModelServiceConfigStore,
    config_module: Any,
    apply: Callable[[Any, ModelServiceSettings], None],
    on_error: Callable[[Exception], None] | None = None,
) -> ModelServiceSettings | None:
    """读出已保存的设置并应用。

    「怎么应用到项目配置」由调用方注入（``apply``）——那是项目策略，不进本模块。
    读失败不往外抛：交给 ``on_error``，返回 None 让调用方继续用环境变量里的配置。
    """

    try:
        settings = store.load()
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return None
    if settings is not None:
        apply(config_module, settings)
    return settings


def safe_int(value: Any) -> int:
    """取整；不可解析当 0。用于时间戳这类「错了也不能炸」的字段。"""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def probe_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    """从一条注册表记录里抽出探测元数据（模型清单 / 时间 / 错误）。"""

    return {
        "discoveredModels": normalize_model_ids(
            raw.get("discoveredModels") or raw.get("discovered_models") or raw.get("models")
        ),
        "lastModelProbeAt": max(
            0, safe_int(_inherit(raw, "lastModelProbeAt", "last_model_probe_at"))
        ),
        "modelProbeError": str(
            _inherit(raw, "modelProbeError", "model_probe_error") or ""
        )[:_ERROR_DETAIL_LIMIT],
    }


def public_provider_entry(
    *,
    preset: ProviderPreset | None,
    settings: ModelServiceSettings | None,
    metadata: Mapping[str, Any],
    active: bool,
) -> dict[str, Any]:
    """目录视图里的一行：**不含 apiKey**。

    ``preset`` 与 ``settings`` 可能各自为 None：前者是「目录里有、没配过」，
    后者是「配过、但不在目录里」（自定义兼容服务）。两者都没有时给一组中性缺省。
    """

    provider_id = (
        settings.provider_id
        if settings is not None
        else (preset.id if preset else DEFAULT_PROVIDER_ID)
    )
    return {
        "id": provider_id,
        "label": preset.label if preset else provider_id,
        "protocol": settings.protocol if settings else (preset.protocol if preset else "openai"),
        "baseUrl": settings.base_url if settings else (preset.base_url if preset else ""),
        "description": preset.description if preset else "已保存的自定义 OpenAI 兼容服务。",
        "apiKeyRequired": preset.api_key_required if preset else True,
        "configured": bool(settings and settings.configured),
        "endpointConfigured": bool(settings and settings.endpoint_configured),
        "active": bool(active),
        "hasApiKey": bool(settings and settings.api_key),
        "chatModel": settings.chat_model if settings else "",
        "useForVision": bool(settings and settings.use_for_vision),
        "visionModel": settings.vision_model if settings else "",
        "timeoutSeconds": settings.timeout_seconds if settings else DEFAULT_TIMEOUT_SECONDS,
        "models": list(metadata.get("discoveredModels") or []),
        "modelCount": len(metadata.get("discoveredModels") or []),
        "lastModelProbeAt": int(metadata.get("lastModelProbeAt") or 0),
        "modelProbeError": str(metadata.get("modelProbeError") or ""),
        "supportsModelDiscovery": True,
        "capabilities": list(
            preset.capabilities if preset else ("stream", "vision", "native_tools")
        ),
    }

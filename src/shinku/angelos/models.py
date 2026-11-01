"""复用供应商目录，独立保存 Angelos 的模型选择；不写共用注册表或密钥。

仅由本机带令牌的工作台调用。默认实例切换仍要求全局空闲；会话选择由
SessionModels保存，使用create_runtime构建私有客户端，不修改共用环境。
工作台和AgentService锁保护派发，失败时原选择和运行实例保留。
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from shinku.providers.catalog import (apply_model_service_settings, environment_settings_for_provider,
    public_model_services_snapshot, probe_model_ids)
from shinku.llm.client import build_llm_client


def discover(settings):
    if settings.protocol != 'openai':
        return [{'id': name, 'effortLevels': []} for name in probe_model_ids(settings)]
    client = build_llm_client(api_key=settings.api_key, base_url=settings.base_url, protocol=settings.protocol,
        timeout=min(10, settings.timeout_seconds), max_retries=0)
    try:
        rows = client.models.list().data
        result = []
        for row in list(rows)[:512]:
            data = row.model_dump() if hasattr(row, 'model_dump') else dict(row)
            name = data.get('id')
            if not isinstance(name, str) or not name or len(name) > 256:
                continue
            effort = data.get('effort') or {}
            levels = effort.get('supported_levels', []) if isinstance(effort, dict) else []
            row = {'id': name, 'name': str(data.get('name') or name)[:256],
                'effortLevels': [v for v in levels if v in {'minimal','low','medium','high','xhigh','max'}]}
            # 供应商容量只接收数值元数据，不读取任意描述或凭据；未知仍需宿主配置。
            for target, fields in {'contextWindow': ('context_length', 'context_window', 'context_window_tokens'),
                                   'maxOutputTokens': ('max_output_tokens', 'max_completion_tokens')}.items():
                value = next((data[k] for k in fields if type(data.get(k)) is int and 0 < data[k] <= 2000000), None)
                if value is not None:
                    row[target] = value
            result.append(row)
        return result
    finally:
        client.close()


class ModelControls:
    def __init__(self, controller, path, *, probe=discover):
        self.controller, self.path, self.probe = controller, Path(path), probe
        self.cache = {}
        self.selection = None
        self.context_overrides = {}

    def _metadata(self, provider, model):
        """容量仅属于精确供应商/模型，不随模型名称相似或切换继承。"""
        row = next((r for r in self.cache.get(provider, []) if r.get('id') == model), {})
        override = self.context_overrides.get(provider, {}).get(model, 0)
        return {'model': model, 'contextWindow': override or row.get('contextWindow', 0),
                'maxOutputTokens': row.get('maxOutputTokens', 0),
                'contextSource': 'angelos_override' if override else 'provider_metadata'}

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'selection': self.selection, 'models': self.cache,
            'contextOverrides': self.context_overrides}, ensure_ascii=False))
        temp.chmod(0o600)
        os.replace(temp, self.path)

    def set_context_window(self, data):
        """调用方须持有空闲服务锁；只更新容量，不重建客户端或修改模型/权限。"""
        if (set(data) != {'providerId', 'modelId', 'contextWindow'}
                or not all(isinstance(data[k], str) and data[k] for k in ('providerId', 'modelId'))
                or type(data['contextWindow']) is not int or not 0 <= data['contextWindow'] <= 2000000):
            raise ValueError('invalid_context_window')
        provider, model = data['providerId'], data['modelId']
        settings = self.settings(provider)
        if model != settings.chat_model and not any(r['id'] == model for r in self.cache.get(provider, [])):
            raise ValueError('model_not_discovered')
        previous = deepcopy(self.context_overrides)
        try:
            rows = self.context_overrides.setdefault(provider, {})
            if data['contextWindow']:
                rows[model] = data['contextWindow']
            else:
                rows.pop(model, None)
            self.bind_runtime_limits()
            self._persist()
        except Exception:
            self.context_overrides = previous
            self.bind_runtime_limits()
            raise
        return self.snapshot()

    def settings(self, provider):
        value = self.controller.store.get_provider(provider) or environment_settings_for_provider(provider, self.controller.config_module)
        if value is None or not value.endpoint_configured:
            raise ValueError('provider_not_configured')
        return value

    def snapshot(self):
        # 只挑选安全字段；共用目录中可能有URL凭据/探测报错，不能原样转给浏览器。
        page = public_model_services_snapshot(self.controller.store, self.controller.config_module)
        providers = []
        for row in page['providers']:
            models = self.cache.get(row['id'], [{'id': name, 'effortLevels': []} for name in row['models']])
            if row['chatModel'] and not any(m['id'] == row['chatModel'] for m in models):
                models = [*models, {'id': row['chatModel'], 'effortLevels': []}]
            host = urlsplit(row['baseUrl']).hostname or ''
            models = [{**m, 'thinkingControl': row['protocol'] == 'openai' and
                (host == 'api.deepseek.com' or host.endswith('.deepseek.com') or m['id'].startswith('deepseek-'))} for m in models]
            models = [{**m, 'contextWindowOverride': self.context_overrides.get(row['id'], {}).get(m['id'], 0)} for m in models]
            providers.append({'id': row['id'], 'label': row['label'], 'configured': row['endpointConfigured'],
                'models': models, 'discovered': row['id'] in self.cache})
        selected = self.selection or {'providerId': page['activeProviderId'], 'modelId': page['activeModel'],
            'effort': 'off' if os.environ.get('SHINKU_LLM_THINKING_MODE', 'disabled') == 'disabled' else 'auto'}
        from shinku.llm.model_limits import model_limits
        bundle = getattr(self.controller.runtime, 'chat', None)
        limits = model_limits(bundle) if bundle is not None else {'window_tokens': 0, 'source': 'unknown'}
        return {'available': True, 'providers': providers, 'selected': selected, 'contextLimits': limits,
                'scope': 'angelos_new_tasks', 'note': '只影响 Angelos 的新任务，密钥沿用已配置供应商。'}

    def refresh(self, provider):
        # 与切换相同的宿主锁保护；成功刷新当前路由后立即补绑，避免目录有容量
        # 而客户端仍显示未知。失败恢复缓存/绑定，不能清掉用户确认的覆盖值。
        rows = self.probe(self.settings(provider))
        previous = deepcopy(self.cache)
        try:
            self.cache[provider] = rows
            self.bind_runtime_limits()
            if self.selection is not None:
                self._persist()
        except Exception:
            self.cache = previous
            self.bind_runtime_limits()
            raise
        return self.snapshot()

    def bind_runtime_limits(self):
        """客户端附着后补绑已保存的精确模型容量；恢复配置早于客户端时不丢元数据。

        只使用当前选择对应的缓存，不查询网络或猜测代理容量。未知保持未知，
        不从上一客户端继承窗口；最终SDK预算仍执行九成摘要检查。
        """
        selected = self.selection or self.snapshot()['selected']
        bundle = getattr(self.controller.runtime, 'chat', None)
        if not selected or bundle is None or bundle.model != selected['modelId']:
            return
        bundle.client._shinku_model_limits = self._metadata(selected['providerId'], selected['modelId'])

    def validate(self, data):
        if set(data) != {'providerId', 'modelId', 'effort'} or any(not isinstance(v, str) for v in data.values()):
            raise ValueError('invalid_model_selection')
        provider, model, effort = data['providerId'], data['modelId'], data['effort']
        settings = self.settings(provider)
        rows = self.cache.get(provider, [])
        known = {row['id']: row for row in rows}
        if model not in known and model != settings.chat_model:
            raise ValueError('model_not_discovered')
        host = urlsplit(settings.base_url).hostname or ''
        thinking = settings.protocol == 'openai' and (host == 'api.deepseek.com' or host.endswith('.deepseek.com') or model.startswith('deepseek-'))
        levels = known.get(model, {}).get('effortLevels', []) if thinking else []
        if effort not in {'auto', *(['off'] if thinking else []), *levels}:
            raise ValueError('reasoning_effort_unsupported')
        return settings, levels

    def create_runtime(self, data):
        """构建会话私有客户端；不修改进程环境、共用注册表或默认运行时。"""
        settings, levels = self.validate(data)
        from .session_models import SessionRuntime
        runtime = SessionRuntime(settings, data, self._metadata(data["providerId"], data["modelId"]))
        if runtime.chat_supports_native_tools() is False:
            runtime.close()
            raise ValueError("model_native_tools_unsupported")
        return runtime

    def apply(self, data, *, persist=True):
        settings, levels = self.validate(data)
        provider, model, effort = data["providerId"], data["modelId"], data["effort"]
        previous_env, previous_config = dict(os.environ), deepcopy(vars(self.controller.config_module))
        old_selection = self.selection
        try:
            apply_model_service_settings(self.controller.config_module, replace(settings, chat_model=model))
            # 原宿主明确验证过的协议白名单继续保留；目录切换不能清掉它，也
            # 不能为尚未验证的新模型自动扩大白名单。
            allowlist = previous_env.get('SHINKU_ANGELOS_NATIVE_TOOL_PROVIDER_ALLOWLIST', '')
            if allowlist:
                self.controller.config_module.NATIVE_TOOL_PROVIDER_ALLOWLIST = allowlist
            os.environ['SHINKU_LLM_THINKING_MODE'] = 'disabled' if effort == 'off' else 'auto' if effort == 'auto' else 'enabled'
            os.environ['SHINKU_LLM_REASONING_EFFORT'] = effort if effort in levels else ''
            self.controller.reload_model_services()
            bundle = getattr(self.controller.runtime, 'chat', None)
            if bundle is not None:
                # 元数据绑定到本次新客户端和精确模型；换模型不得继承旧窗口。
                bundle.client._shinku_model_limits = self._metadata(provider, model)
            check = getattr(self.controller.runtime, 'chat_supports_native_tools', None)
            if callable(check) and check() is False:
                raise ValueError('model_native_tools_unsupported')
            self.selection = dict(data)
            if persist:
                self._persist()
        except Exception:
            # 此进程独占模型配置；失败时不能留下UI选项与实际模型不一致。
            os.environ.clear()
            os.environ.update(previous_env)
            vars(self.controller.config_module).clear()
            vars(self.controller.config_module).update(previous_config)
            self.selection = old_selection
            if self.controller.runtime is not None:
                try:
                    self.controller.reload_model_services()
                finally:
                    os.environ.clear()
                    os.environ.update(previous_env)
                    self.bind_runtime_limits()
            raise
        return self.snapshot()

    def restore(self):
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            self.cache = saved.get('models', {})
            raw = saved.get('contextOverrides', {})
            self.context_overrides = {p: {m: v for m, v in rows.items() if isinstance(m, str)
                and type(v) is int and 0 < v <= 2000000} for p, rows in raw.items()
                if isinstance(p, str) and isinstance(rows, dict)} if isinstance(raw, dict) else {}
            if saved.get('selection'):
                self.apply(saved['selection'], persist=False)

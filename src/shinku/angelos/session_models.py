"""会话模型配置与执行器绑定。工作台拥有持久化，任务派发时冻结运行实例。

读取顺序：ensure绑定旧会话默认 -> select保存本会话 -> for_task冻结本次派发。
模型/消息不能选择执行器；跨角色委托继续使用原宿主执行器。SDK凭据仅存在内存。
"""
from dataclasses import replace
import os
import logging
import threading

from shinku.llm.runtime import LLMRuntime
from shinku.llm.policy import RuntimePolicy
from shinku.agent_runtime.execution import AgentExecutor


class SessionRuntime(LLMRuntime):
    """每个实例固定环境快照，调用时也不读取另一会话的思考/供应商配置。"""
    def __init__(self, settings, selection, metadata):
        self.environment = dict(os.environ)
        for channel in ('CHAT', 'AUX'):
            self.environment.update({f'SHINKU_{channel}_API_KEY': settings.api_key,
                f'SHINKU_{channel}_BASE_URL': settings.base_url,
                f'SHINKU_{channel}_API_PROTOCOL': settings.protocol,
                f'SHINKU_{channel}_MODEL_NAME': selection['modelId']})
        effort = selection['effort']
        self.environment['SHINKU_LLM_THINKING_MODE'] = 'disabled' if effort == 'off' else 'auto' if effort == 'auto' else 'enabled'
        self.environment['SHINKU_LLM_REASONING_EFFORT'] = '' if effort in {'auto', 'off'} else effort
        allowlist = self.environment.get('SHINKU_ANGELOS_NATIVE_TOOL_PROVIDER_ALLOWLIST')
        if allowlist:
            self.environment['SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST'] = allowlist
        policy = replace(RuntimePolicy.from_environment(self.environment),
                         chat_timeout_seconds=settings.timeout_seconds, aux_timeout_seconds=settings.timeout_seconds)
        super().__init__(runtime_policy=policy)
        self.chat.client._shinku_model_limits = dict(metadata)
        self.aux.client._shinku_model_limits = dict(metadata)

    def _env(self, key, default=''):
        return self.environment.get('SHINKU_' + key, default)

    def _apply_circuit_policy(self):
        # 默认宿主已配置进程级熔断器；创建会话客户端不得重新配置共享状态。
        pass

    def close(self):
        super().close()
        for bundle in (self.chat, self.aux):
            close = getattr(bundle.client, 'close', None)
            if callable(close):
                close()


class SessionModels:
    def __init__(self, host, base, compaction_path=None):
        self.host, self.base, self.store = host, base, None
        self.compaction_path = compaction_path
        self.lock = threading.RLock()
        self.entries = {}

    def ensure(self, cid):
        """已保存的会话不再跟随全局默认；旧会话优先保留最近一轮的模型选择。"""
        with self.lock:
            c = self.store.read('conversation', cid)
            if not c.get('modelSettings'):
                previous = next((t['model_settings'] for t in reversed(c['turns']) if t.get('model_settings')), None)
                selected = dict(previous or self.host.models.snapshot()['selected'])
                c = self.store.change('conversation', cid, lambda row: row.update(modelSettings=selected))
            return c['modelSettings']

    def snapshot(self, cid):
        result = self.host.models.snapshot()
        selected = self.ensure(cid)
        result.update(selected=selected, scope='conversation', conversationId=cid)
        metadata = self.host.models._metadata(selected['providerId'], selected['modelId'])
        from shinku.llm.model_limits import model_limits
        with self.lock:
            entry = self.entries.get(cid)
            result['contextLimits'] = (model_limits(entry[1].profile.runtime.chat) if entry and entry[0] == selected else
                {'window_tokens': metadata['contextWindow'], 'source': metadata['contextSource'] if metadata['contextWindow'] else 'unknown'})
        return result

    def _build(self, selected):
        from .runtime_view import AngelosRuntime
        raw = self.host.models.create_runtime(selected)
        try:
            runtime = AngelosRuntime(raw, self.host.reasoning_view, usage_sink=self.host.usage_view,
                compaction_path=self.compaction_path, compaction_supported=self.base.profile.runtime_meters_budget,
                context_awareness=True)
        except Exception:
            raw.close()
            raise
        return AgentExecutor(self.base.tool_host, replace(self.base.profile, runtime=runtime, profile_id=selected['modelId']))

    @staticmethod
    def _close(executor):
        runtime = executor.profile.runtime
        # 新选择已提交/旧会话已删除后，清理故障不能谎报保存失败并诱发重试。
        for close in (runtime.compactor.db.close, runtime.runtime.close):
            try:
                close()
            except Exception as exc:
                logging.getLogger(__name__).warning('session_client_close_failed:%s', type(exc).__name__)

    def select(self, cid, data):
        # 调用方持有工作台及服务锁并确认该会话没有活动任务；客户端构建不调用模型。
        self.host.models.validate(data)
        candidate = self._build(data)
        try:
            with self.lock:
                self.store.change('conversation', cid, lambda c: c.update(modelSettings=dict(data)))
                old = self.entries.get(cid)
                self.entries[cid] = (dict(data), candidate)
                if old:
                    self._close(old[1])
        except Exception:
            self._close(candidate)
            raise
        return self.snapshot(cid)

    def for_task(self, scope, task):
        if scope.requester != self.host.requester or not scope.session.startswith('angelos:'):
            return self.base
        cid = scope.session.removeprefix('angelos:')
        with self.lock:
            selected = self.ensure(cid)
            entry = self.entries.get(cid)
            if entry is None or entry[0] != selected:
                candidate = self._build(selected)
                self.entries[cid] = (dict(selected), candidate)
                if entry:
                    self._close(entry[1])
            return self.entries[cid][1]

    def context_status(self, cid, binding):
        with self.lock:
            entry = self.entries.get(cid)
            return (entry[1] if entry else self.base).profile.runtime.compactor.status(binding)

    def forget(self, cid):
        with self.lock:
            entry = self.entries.pop(cid, None)
            if entry:
                self._close(entry[1])

    def close(self):
        with self.lock:
            for _, executor in self.entries.values():
                self._close(executor)
            self.entries.clear()

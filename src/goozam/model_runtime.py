"""沙箱内的显式模型适配：复用原 LLMRuntime，只替换客户端的物理出口。

这里没有真实密钥/上游 URL，按沙箱已清空的环境运行。原生工具历史及 JSON 解析仍由
原运行时负责；用量由原任务预算和宿主代理分别记录，二者不相加冒充计费。
"""
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import httpx
from openai import OpenAI

from shinku.agent.budget import output_reservation
from shinku.llm.policy import RuntimePolicy
from shinku.llm.runtime import LLMRuntime, ModelBundle, ProviderToolProfile


@contextmanager
def socket_address(socket_path):
    """Linux目录句柄提供短地址，避免长路径；句柄只在连接生命周期内保持打开。"""
    path = Path(socket_path).absolute()
    fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
    try:
        yield f'/proc/self/fd/{fd}/{path.name}'
    finally:
        os.close(fd)


class RelayRuntime(LLMRuntime):
    def __init__(self, socket_path, profile):
        if (not {'model', 'context_tokens', 'max_output_tokens'} <= set(profile)
                or set(profile) - {'model', 'context_tokens', 'max_output_tokens', 'route_fingerprint'}
                or not isinstance(profile['model'], str) or not profile['model']
                or any(type(profile[k]) is not int or not 0 < profile[k] <= 2000000
                       for k in ('context_tokens', 'max_output_tokens'))):
            raise ValueError('goozam_invalid_model_profile')
        self.profile = dict(profile)
        self.resources = ExitStack()
        try:
            address = self.resources.enter_context(socket_address(socket_path))
            transport = self.resources.enter_context(httpx.Client(
                transport=httpx.HTTPTransport(uds=address), trust_env=False, follow_redirects=False))
            self.client = self.resources.enter_context(OpenAI(api_key='goozam-socket-only',
                base_url='http://goozam.invalid/v1', max_retries=0, timeout=125, http_client=transport))
            self.client._shinku_protocol = 'openai'
            self.client._shinku_model_limits = dict(model=profile['model'],
                contextWindow=profile['context_tokens'], maxOutputTokens=profile['max_output_tokens'])
            super().__init__(runtime_policy=RuntimePolicy(http_attempts=1, stream_attempts=1,
                retry_backoff_seconds=0, stream_retry_delay_seconds=0))
            # 固定沙箱网关只实现完整JSON响应，不继承标准工作台的流式展示能力。
            self.incremental_json_supported = False
        except BaseException:
            self.resources.close()
            raise

    def _env(self, key, default=''):
        return default

    def _make_aux_bundle(self):
        return ModelBundle(self.client, self.profile['model'])

    _make_chat_bundle = _make_aux_bundle

    def _tool_profile(self, bundle):
        # 仅声明本通道实现 OpenAI function-call 协议；不声称真实供应商已验收。
        return ProviderToolProfile(supports_native_tools=True, verified=False,
                                   native_tools_coexist_with_forced_json=False)

    def _build_thinking_kwargs(self, *, bundle):
        return {}  # 思考设置只由宿主代理固定，沙箱请求不能覆盖。

    def _build_payload(self, **kwargs):
        value = super()._build_payload(**kwargs)
        value.pop('max_tokens', None)
        value.pop('max_completion_tokens', None)
        return value

    def _attempt_completion(self, **kwargs):
        with output_reservation(self.profile['max_output_tokens']):
            return super()._attempt_completion(**kwargs)

    def context_window_tokens(self):
        return self.profile['context_tokens']

    def close(self):
        self.resources.close()


@contextmanager
def relay_runtime(socket_path, profile):
    runtime = RelayRuntime(socket_path, profile)
    try:
        yield runtime
    finally:
        runtime.close()

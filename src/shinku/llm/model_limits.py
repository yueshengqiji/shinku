"""宿主解析模型窗口；显式配置优先，供应商已核对的容量只匹配官方端点。

容量不是权限。未知模型返回0，不按名字给任意兼容代理编造上下文窗口。
DeepSeek基线：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/（2026-10-19）。
"""
from urllib.parse import urlsplit
from .context_budget import ContextBudgetPolicy


def model_limits(bundle):
    policy = ContextBudgetPolicy.from_environment()
    window = policy.context_window(bundle.model)
    client = bundle.client
    host = urlsplit(str(getattr(client, 'base_url', ''))).hostname or ''
    protocol = getattr(client, '_shinku_protocol', getattr(client, 'protocol', ''))
    official = protocol == 'openai' and host == 'api.deepseek.com'
    known = official and bundle.model in {'deepseek-flash', 'deepseek-v4-pro'}
    meta = getattr(client, '_shinku_model_limits', {})
    meta = meta if isinstance(meta, dict) and meta.get('model') == bundle.model else {}
    def number(key):
        value = meta.get(key)
        return value if type(value) is int and 0 < value <= 2000000 else 0
    # 文档的1M取保守十进制窗口；输出预留按384Ki取上界，不作为SDK输出参数。
    return {'window_tokens': window or number('contextWindow') or (1000000 if known else 0),
            'max_output_tokens': number('maxOutputTokens') or (393216 if known else 0),
            'source': 'host_config' if window else ('angelos_override' if meta.get('contextSource') == 'angelos_override' else 'provider_metadata') if number('contextWindow') else 'deepseek_official_20261019' if known else 'unknown'}

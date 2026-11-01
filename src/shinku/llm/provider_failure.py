"""从真实HTTP异常提取安全错误类别；不保存服务正文、地址、请求头或凭据。

这里只解释调用为何中断，不改变重试、预算或任务完成判定。模型生成的错误
文字不会经过本入口；未知服务错误保留通用分类，不用自由文本猜测账户状态。
"""
from collections.abc import Mapping


def provider_failure(exc):
    seen = set()
    for _ in range(8):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        response = getattr(exc, 'response', None)
        status = getattr(exc, 'status_code', None) or getattr(response, 'status_code', None)
        if type(status) is int and 400 <= status <= 599:
            body = getattr(exc, 'body', None)
            if not isinstance(body, Mapping) and response is not None:
                try:
                    body = response.json() if len(response.content) <= 65536 else None
                except (ValueError, AttributeError, RuntimeError):
                    body = None
            error = body.get('error', body) if isinstance(body, Mapping) else {}
            error = error if isinstance(error, Mapping) else {}
            codes = {v.casefold() for k in ('code', 'type')
                     if isinstance(v := error.get(k), str) and len(v) <= 100}
            category = ''
            if codes & {'insufficient_balance', 'insufficient_balance_error', 'balance_not_enough'}:
                category = 'insufficient_balance'
            elif codes & {'insufficient_quota', 'quota_exceeded', 'quota_exhausted'}:
                category = 'quota_exhausted'
            elif codes & {'context_length_exceeded', 'context_window_exceeded', 'max_context_length_exceeded'}:
                category = 'context_limit'
            # 兼容只在HTTP错误message中给出固定前缀的路由；不回传原文。
            elif isinstance(error.get('message'), str) and error['message'].casefold().startswith('insufficient balance'):
                category = 'insufficient_balance'
            if not category:
                category = ({401: 'authentication', 403: 'access_denied', 402: 'billing_required',
                             429: 'rate_limit', 413: 'request_too_large'}.get(status)
                            or ('service_unavailable' if status >= 500 else 'request_rejected'))
            return {'category': category, 'http_status': status}
        exc = exc.__cause__ or exc.__context__
    return {}


def failure_explanation(metadata, failure='call_failed'):
    """只渲染宿主白名单类别；不能将供应商或模型原文直接放入用户答复。"""
    category = metadata.get('category') if isinstance(metadata, Mapping) else None
    return {
        'insufficient_balance': '模型服务返回余额不足，未能继续生成答复。请检查当前模型服务的余额。',
        'quota_exhausted': '模型服务返回账户额度不足，未能继续生成答复。请检查当前模型服务的额度。',
        'billing_required': '模型服务要求处理计费状态，未能继续生成答复。',
        'authentication': '模型服务拒绝了身份认证，请检查当前连接的凭据。',
        'access_denied': '模型服务拒绝访问，请检查账户的模型访问权限。',
        'rate_limit': '模型服务触发限流，本次未取得完整答复。',
        'context_limit': '模型服务拒绝了超出上下文长度的请求，本次未取得完整答复。',
        'request_too_large': '模型服务拒绝了过大的请求，本次未取得完整答复。',
        'service_unavailable': '模型服务暂时不可用，本次未取得完整答复。',
        'request_rejected': '模型服务拒绝了本次请求，具体类别尚未识别。',
    }.get(category, '模型请求超时，未取得完整答复。' if failure == 'call_timeout'
          else '模型请求连接或传输失败，未取得完整答复。')

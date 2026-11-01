"""Angelos显式启用的JSON流传输：展示增量，完整响应仍交原解析器。

不使用聊天流的提前工具探测/修复重试。SDK流由原预算与请求账本包裹；这里只
聚合供应商分片，校验结束信号和资源边界。展示接收器失败不改变执行或触发重放。
"""
from contextlib import contextmanager
from contextvars import ContextVar
import re
import time
import uuid

from .native_messages import NativeProtocolError, MAX_NATIVE_BYTES

_sink = ContextVar('angelos_json_stream_sink', default=None)


@contextmanager
def json_stream_scope(sink):
    token = _sink.set(sink if callable(sink) else None)
    try:
        yield
    finally:
        _sink.reset(token)


def enabled(bundle):
    from .context_compaction import is_summary_call
    return (_sink.get() is not None and not is_summary_call()
            and getattr(bundle.client, '_shinku_protocol', '') == 'openai')


def _value(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def consume(dispatch, *, bundle, payload, attempt_budget):
    """同一请求边收边展示；结束前返回不了工具决策，也不把分片当成回执。"""
    from openai.types.chat import ChatCompletion
    from .context_compaction import _check_continuation
    from .runtime_transport import _remaining
    from shinku.agent.budget import current_task_budget, TaskBudgetExceeded
    sink = _sink.get(); ident = uuid.uuid4().hex
    body, reasoning, calls = [], [], {}
    finish, usage, model = None, None, payload['model']
    size, chars, display, last_emit = 0, 0, '', 0.0
    stream = None

    def check():
        _check_continuation(reason='model_stream_canceled')
        _remaining()
        budget = current_task_budget()
        if budget and budget.stop_reason():
            raise TaskBudgetExceeded(budget.reason)

    def emit(status, *, force=False):
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < .15:
            return
        last_emit = now
        # 未结束的ASCII词可能是分片密钥/URL；留到边界再整体脱敏，避免先泄漏半个值。
        text = display if status != 'streaming' else re.sub(r'[A-Za-z0-9_./?=&:+%@~\-]+$', '', display)
        try:
            snapshot = {'id': ident, 'text': text, 'observed_chars': chars, 'status': status, 'model': model}
            if status in ('interrupted', 'incomplete'):
                snapshot['partial_content'] = ''.join(body)
            sink(snapshot)
        except Exception:
            pass

    try:
        check()
        stream = dispatch(bundle=bundle, payload=payload, attempt_budget=attempt_budget)
        emit('streaming', force=True)
        for chunk in stream:
            check()
            value = _value(chunk, 'usage')
            if value is not None:
                usage = value.model_dump() if hasattr(value, 'model_dump') else value
            model = _value(chunk, 'model') or model
            for choice in _value(chunk, 'choices', []) or []:
                if _value(choice, 'index', 0) != 0:
                    continue
                if finish is not None and _value(choice, 'delta'):
                    delta = _value(choice, 'delta')
                    if any(_value(delta, k) for k in ('content', 'reasoning_content', 'tool_calls')):
                        raise NativeProtocolError('model_stream_after_finish')
                finish = _value(choice, 'finish_reason') or finish
                delta = _value(choice, 'delta') or {}
                for key, target in (('content', body), ('reasoning_content', reasoning)):
                    part = _value(delta, key)
                    if part is None:
                        continue
                    if not isinstance(part, str):
                        raise NativeProtocolError('model_stream_content_invalid')
                    size += len(part.encode('utf-8')); target.append(part)
                    if key == 'reasoning_content':
                        chars += len(part); display += part[:max(0, 65536-len(display))]
                for call in _value(delta, 'tool_calls', []) or []:
                    index = _value(call, 'index')
                    if type(index) is not int or not 0 <= index < 32:
                        raise NativeProtocolError('model_stream_tool_index_invalid')
                    item = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    kind = _value(call, 'type')
                    if kind not in (None, 'function'):
                        raise NativeProtocolError('model_stream_tool_type_invalid')
                    fn = _value(call, 'function') or {}
                    for key, source, target in (('id', call, item), ('name', fn, item['function']), ('arguments', fn, item['function'])):
                        part = _value(source, key)
                        if part is not None:
                            if not isinstance(part, str):
                                raise NativeProtocolError('model_stream_tool_part_invalid')
                            size += len(part.encode('utf-8')); target[key] += part
                if size > MAX_NATIVE_BYTES:
                    raise NativeProtocolError('native_history_resource_limit')
            emit('streaming')
        check()
        if finish not in ('stop', 'tool_calls', 'length', 'content_filter'):
            raise NativeProtocolError('model_stream_missing_finish')
        if finish in ('length', 'content_filter'):
            from .response_protocol import note_incomplete_response
            note_incomplete_response(finish)
            raise NativeProtocolError('model_stream_incomplete')
        message = {'role': 'assistant', 'content': ''.join(body)}
        if reasoning:
            message['reasoning_content'] = ''.join(reasoning)
        if calls:
            if sorted(calls) != list(range(len(calls))):
                raise NativeProtocolError('model_stream_tool_index_gap')
            message['tool_calls'] = [calls[i] for i in sorted(calls)]
        response = ChatCompletion.model_validate({'id': ident, 'object': 'chat.completion', 'created': int(time.time()),
            'model': model, 'usage': usage, 'choices': [{'index': 0, 'finish_reason': finish, 'message': message}]})
        emit('completed' if finish in ('stop', 'tool_calls') else 'incomplete', force=True)
        return response
    except BaseException:
        emit('incomplete' if finish in ('length', 'content_filter') else 'interrupted', force=True)
        raise
    finally:
        close = getattr(stream, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass  # 释放连接的错误不能覆盖原始断流/取消原因，更不能重发请求。

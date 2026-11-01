"""单任务进程内的供应商原生历史，绑定真实宿主回执与当前任务作用域。

模型消息只追加，不回写已发送结果或推理；命令回执首次加入时生成可补读视图。
宿主原件继续由材料库保存。宿主声明完整输入增量时，拒绝后
只追加条件、资料和反馈变化；旧构造器缺少声明时保留完整续接提示。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from shinku.llm.native_messages import NativeProtocolError, _check_size, validate_native_messages


class NativeToolHistory:
    def __init__(self, scope, user_prompt, user_images, history_turns, native_context_update="", native_hard_context="",
                 native_input_update=None, continuation_feedback="", native_work_context=""):
        self._validate_update(native_context_update)
        if native_context_update and native_context_update not in user_prompt:
            raise NativeProtocolError("native_context_update_missing_from_prompt")
        self.scope = scope
        self._validate_update(continuation_feedback)
        # 首轮反馈与根输入一起固定，后续只追加变化，避免空链多造一个用户轮。
        self.user_prompt = user_prompt + ('\n[Host validation feedback; not permission or source evidence]\n'
            + continuation_feedback if continuation_feedback else '')
        self.user_images = copy.deepcopy(user_images)
        self.history_turns = copy.deepcopy(history_turns)
        self.messages = []
        self.pending = {}
        self.seen = set()
        self.receipts = {}
        self.last_was_final = False
        self.error = ""
        self.context_update = native_context_update
        self._validate_update(native_hard_context)
        if native_hard_context and native_hard_context not in user_prompt:
            raise NativeProtocolError('native_hard_context_missing_from_prompt')
        self.hard_context = native_hard_context
        self._validate_input(native_input_update, user_prompt)
        self.input_update = native_input_update
        self.continuation_feedback = continuation_feedback
        self._validate_work(native_work_context, user_prompt)
        self.work_context = native_work_context
        self.work_context_message = native_work_context

    def _validate_work(self, value, prompt):
        self._validate_update(value)
        if value and value not in prompt:
            raise NativeProtocolError('native_work_context_missing_from_prompt')

    def _validate_input(self, value, prompt):
        if value is not None:
            self._validate_update(value)
            if value and value not in prompt:
                raise NativeProtocolError('native_input_update_missing_from_prompt')

    @staticmethod
    def _validate_update(value):
        if not isinstance(value, str):
            raise NativeProtocolError("native_context_update_invalid")
        _check_size([{"role": "user", "content": value}])

    def __repr__(self):
        return "<NativeToolHistory>"

    def capture(self, message):
        if self.pending:
            raise NativeProtocolError("native_tool_receipt_missing")
        calls = message.get("tool_calls", [])
        for call in calls:
            if call["id"] in self.seen:
                raise NativeProtocolError("native_tool_id_reused")
        _check_size([*self.messages, message], history=True)
        self.messages.append(copy.deepcopy(message))
        self.pending = {call["id"]: call["function"]["name"] for call in calls}
        self.seen.update(self.pending)
        self.last_was_final = not bool(calls)

    def observe(self, state):
        # Protocol representation must never interrupt host receipt persistence.
        try:
            self._observe(state)
        except NativeProtocolError as exc:
            self.error = self.error or str(exc)

    def _observe(self, state):
        envelope = state.last_envelope
        if envelope is None or envelope.invocation_id not in self.pending:
            return
        ident = envelope.invocation_id
        matching = next((row for row in reversed(state.tool_calls) if row.get("id") == ident), None)
        if not matching or matching.get("name") != self.pending[ident] or matching.get("status") != envelope.status:
            raise NativeProtocolError("native_host_receipt_mismatch")
        # 只在首次追加时投影命令输出，不修改旧前缀或宿主验收原件。
        # task_material_read 保持逐字回执，仍由实际模型输入确认已读范围。
        from .tool_context import model_tool_content
        context = state.context if isinstance(state.context, Mapping) else {}
        content = model_tool_content(envelope, matching, context.get('agent_material_session'))
        row = {"role": "tool", "tool_call_id": ident, "content": content}
        _check_size([*self.messages, row], history=True)
        self.messages.append(row)
        self.receipts[ident] = content
        del self.pending[ident]

    def request_kwargs(self, state, scope, user_prompt, native_context_update="", native_hard_context="",
                       native_input_update=None, continuation_feedback="", native_work_context=""):
        if scope != self.scope:
            raise NativeProtocolError("native_history_scope_changed")
        self.observe(state)
        if self.error:
            raise NativeProtocolError(self.error)
        if self.pending:
            raise NativeProtocolError("native_tool_receipt_missing")
        self._validate_update(native_context_update)
        if native_context_update and native_context_update not in user_prompt:
            raise NativeProtocolError("native_context_update_missing_from_prompt")
        changed = native_context_update != self.context_update
        self._validate_update(native_hard_context)
        if native_hard_context and native_hard_context not in user_prompt:
            raise NativeProtocolError('native_hard_context_missing_from_prompt')
        hard_changed = native_hard_context != self.hard_context
        self._validate_input(native_input_update, user_prompt)
        self._validate_update(continuation_feedback)
        input_changed = native_input_update != self.input_update
        feedback_changed = continuation_feedback != self.continuation_feedback
        self._validate_work(native_work_context, user_prompt)
        work_changed = native_work_context != self.work_context
        if self.input_update is not None and native_input_update is None:
            raise NativeProtocolError('native_input_update_contract_removed')
        # 独立去重：硬条件有变化就逐字追加并留给摘要保护；计划改变不重复
        # 整份契约。旧assistant/tool及控制消息始终原样保留，缓存不代表授权。
        hard_update = native_hard_context if hard_changed else ''
        if hard_changed and not native_hard_context and self.hard_context:
            raise NativeProtocolError('native_hard_context_removed')
        cleared = "[Host advisory context cleared; preserve the original task contract and actual receipts.]"
        updates = [hard_update, (native_context_update or cleared) if changed else '',
            (native_input_update or '[Host supplied inputs cleared; prior inputs are historical data.]') if input_changed else '',
            '[Host validation feedback; not permission or source evidence]\n' + continuation_feedback if feedback_changed else '']
        if self.last_was_final:
            # A rejected final or a resumed user wait starts a new user turn;
            # provider reasoning from all prior assistant turns is retained.
            # 已声明输入增量的宿主不重复整份账本；原件、工具配对和推理仍原样
            # 留在历史中。旧构造器没有声明时保留完整提示，避免漏掉新资料。
            continuation = ('继续当前任务，处理最新宿主反馈及尚未满足的原要求；不得重放已有副作用。'
                if native_input_update is not None else user_prompt)
            tail = updates if native_input_update is not None else [
                cleared if changed and not native_context_update else '', updates[-1]]
            row = {"role": "user", "content": '\n'.join(v for v in [continuation, *tail] if v)}
            _check_size([*self.messages, row], history=True)
            self.messages.append(row)
            self.last_was_final = False
        elif changed or hard_changed or input_changed or feedback_changed:
            # Hints are user-side data, appended only after all real receipts.
            row = {"role": "user", "content": '\n'.join(v for v in updates if v)}
            _check_size([*self.messages, row], history=True)
            self.messages.append(row)
        if work_changed:
            # 单独一条宿主消息使摘要只整理旧笔记，不误删同轮新增硬约束。
            # 原生历史仍只追加；整理发生在SDK投影，原提案/回执/修订全部留存。
            content = native_work_context or '[Host work state cleared; original user requirements and receipts remain authoritative.]'
            row = {'role': 'user', 'content': content}
            _check_size([*self.messages, row], history=True)
            self.messages.append(row)
            self.work_context_message = content
        self.work_context = native_work_context
        self.context_update = native_context_update
        self.hard_context = native_hard_context
        self.input_update = native_input_update
        self.continuation_feedback = continuation_feedback
        return dict(user_prompt=self.user_prompt, user_images=copy.deepcopy(self.user_images),
            history_turns=copy.deepcopy(self.history_turns), native_history=validate_native_messages(self.messages),
            provider_message_sink=self.capture)


def prepare_native_request(planner, state, call_kwargs, *, native_context_update="", native_hard_context="",
                           native_input_update=None, continuation_feedback="", native_work_context=""):
    identity = getattr(planner.runtime, "native_message_scope", None)
    if not callable(identity):
        raise NativeProtocolError("native_history_runtime_unsupported")
    scope = (state.task_id, planner.prompt_cache_key, identity())
    history = state.native_tool_history
    if history is None:
        history = NativeToolHistory(scope, call_kwargs["user_prompt"], call_kwargs["user_images"],
            call_kwargs["history_turns"], native_context_update, native_hard_context, native_input_update,
            continuation_feedback, native_work_context)
        state.native_tool_history = history
    if not isinstance(history, NativeToolHistory):
        raise NativeProtocolError("native_history_state_invalid")
    if history.user_images != call_kwargs["user_images"]:
        raise NativeProtocolError("native_history_media_changed")
    prepared = history.request_kwargs(state, scope, call_kwargs["user_prompt"], native_context_update, native_hard_context,
                                      native_input_update, continuation_feedback, native_work_context)
    register = getattr(planner.runtime, 'register_work_context', None)
    if callable(register) and history.work_context_message:
        # 调用方是宿主prompt构建器；不扫描供应商文本授予“可压缩”资格。
        register(history.work_context_message)
    advise = getattr(planner.runtime, 'context_budget_advisory', None)
    note = advise({**call_kwargs, **prepared}) if callable(advise) else ''
    if note:
        # 预算提示是追加的宿主建议，不重写已缓存前缀；压缩器只识别自己签发的提示。
        row = {'role': 'user', 'content': note}
        if not history.messages or history.messages[-1] != row:
            _check_size([*history.messages, row], history=True)
            history.messages.append(row)
        prepared['native_history'] = validate_native_messages(history.messages)
    return prepared

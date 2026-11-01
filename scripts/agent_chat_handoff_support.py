"""SDK and journal adapters for the isolated P4d/P4e/P3c experiment.

The production TaskBudget and accepted DeepSeek SDK path remain authoritative.
This module adds a fixed batch identity, per-role admission and durable slots;
it is not an executor, task store, provider registry or production configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent_runtime.contracts import canonical_hash
from shinku.llm.policy import RuntimePolicy
from shinku.llm.runtime import LLMRuntime, ModelBundle, ProviderToolProfile
from shinku.llm.runtime_expression import RuntimeExpressionAdapter, response_usage

_spec = importlib.util.spec_from_file_location('_handoff_accepted_deepseek',
    Path(__file__).with_name('verify_agent_deepseek.py'))
acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acceptance)
benchmark = acceptance.benchmark

CHAT_MODEL = 'gemini-2.5-pro'
CHAT_ENDPOINT = 'https://api.pinaic.com/v1'
EXECUTION_MODEL, EXECUTION_ENDPOINT = acceptance.MODEL, acceptance.BASE_URL
LIMITS = dict(max_requests=176, max_tokens=8000000, max_tools=80, max_active_seconds=900)
ROLE_LIMITS = dict(chat=64, execution=64, expression=24, check=24)
REMAINING_LIMITS = dict(max_requests=192, max_tokens=3000000, max_tools=100, max_active_seconds=1800)
REMAINING_ROLE_LIMITS = dict(chat=64, execution=64, expression=32, check=32)
ROLE = ContextVar('isolated_handoff_role', default='')
SAFE_REPLY_KEYS = frozenset({'kind', 'text', 'segments', 'interrupt', 'task_outcome',
                            'task_link', 'task_contract', 'binding_hash', 'free_segments', 'accepted',
                            'layout', 'schema_version', 'paragraphs', 'supported', 'covered_fact_ids', 'additional_claims',
                            'contradictions', 'omitted_limitations'})
JSON_SHAPES = frozenset({'object', 'array', 'string', 'number', 'boolean', 'null', 'invalid', 'empty', 'non_text'})
FINISH_REASONS = frozenset({'stop', 'length', 'tool_calls', 'function_call', 'content_filter', 'unknown'})
NEUTRAL_KEYS = frozenset({'binding_hash', 'supported', 'covered_fact_ids', 'additional_claims',
                          'contradictions', 'omitted_limitations'})
NEUTRAL_DIAGNOSTIC_KEYS = frozenset({'schema_exact', 'binding_matches', 'supported_true', 'coverage_exact',
    'covered_ids_valid', 'additional_claims_empty', 'contradictions_empty', 'omitted_limitations_empty'})


def neutral_diagnostics(value, request_payload):
    """Only fixed boolean checks; never persist IDs, reasons or rejected prose."""
    if not isinstance(value, dict) or not isinstance(request_payload, dict):
        return None
    try:
        expected = json.loads(request_payload['messages'][-1]['content'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(expected, dict) or set(expected) != {'binding_hash', 'fact_blocks', 'candidate_text'}:
        return None
    blocks = expected['fact_blocks']
    if not isinstance(blocks, list) or any(not isinstance(b, dict) or not isinstance(b.get('id'), str) for b in blocks):
        return None
    ids = value.get('covered_fact_ids')
    valid_ids = isinstance(ids, list) and all(isinstance(v, str) for v in ids)
    return dict(schema_exact=set(value) == NEUTRAL_KEYS,
        binding_matches=value.get('binding_hash') == expected['binding_hash'],
        supported_true=value.get('supported') is True, covered_ids_valid=valid_ids,
        coverage_exact=valid_ids and sorted(ids) == sorted(b['id'] for b in blocks),
        **{key + '_empty': value.get(key) == [] for key in ('additional_claims', 'contradictions', 'omitted_limitations')})


def response_metadata(response, *, request_payload=None):
    """Record parsing evidence without provider text, reasoning or arbitrary keys."""
    field = lambda value, key, default=None: value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)
    choices = field(response, 'choices', []) or []
    choices = choices if isinstance(choices, (list, tuple)) else []
    rows = []
    for choice in choices[:4]:
        parsed = None
        message = field(choice, 'message', {})
        content = field(message, 'content')
        text = content if isinstance(content, str) else ''
        shape, wrapping, recognised = ('empty' if content is None or isinstance(content, str) and not content.strip()
                                      else 'non_text'), 'none', []
        if isinstance(content, str) and content.strip():
            raw = content
            wrapping = 'plain'
            fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', content.strip(), re.S | re.I)
            if fenced:
                raw, wrapping = fenced[1], 'markdown_fence'
            try:
                parsed = json.loads(raw)
                shape = {dict: 'object', list: 'array', str: 'string', int: 'number', float: 'number',
                         bool: 'boolean', type(None): 'null'}.get(type(parsed), 'invalid')
                if isinstance(parsed, dict):
                    recognised = sorted(SAFE_REPLY_KEYS.intersection(parsed))
            except (TypeError, ValueError, RecursionError):
                shape = 'invalid'
        reason = field(choice, 'finish_reason')
        calls = field(message, 'tool_calls', []) or []
        row = dict(finish_reason=reason if isinstance(reason, str) and reason in FINISH_REASONS else 'unknown',
            native_tool_calls=len(calls) if isinstance(calls, (list, tuple)) else 0, content_chars=len(text),
            content_sha256=hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest(),
            json_shape=shape, json_wrapping=wrapping, recognised_keys=recognised)
        neutral = neutral_diagnostics(parsed, request_payload)
        if neutral is not None:
            row['neutral_checks'] = neutral
        rows.append(row)
    return dict(choice_count=len(choices), choices=rows, choices_omitted=max(0, len(choices) - 4))


def error_cause_types(exc):
    types, seen = [], set()
    while exc is not None and id(exc) not in seen and len(types) < 6:
        seen.add(id(exc))
        name = type(exc).__name__
        types.append(name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', name) else 'Exception')
        exc = exc.__cause__ or exc.__context__
    return types


def valid_response_metadata(value):
    if (not isinstance(value, dict) or set(value) != {'choice_count', 'choices', 'choices_omitted'}
            or type(value['choice_count']) is not int or value['choice_count'] < 0
            or type(value['choices_omitted']) is not int or value['choices_omitted'] < 0
            or not isinstance(value['choices'], list) or len(value['choices']) != min(4, value['choice_count'])
            or value['choices_omitted'] != max(0, value['choice_count'] - 4)):
        return False
    for row in value['choices']:
        if (not isinstance(row, dict) or set(row) - {'neutral_checks'} != {'finish_reason', 'native_tool_calls', 'content_chars',
                'content_sha256', 'json_shape', 'json_wrapping', 'recognised_keys'}
                or not isinstance(row['finish_reason'], str) or row['finish_reason'] not in FINISH_REASONS
                or not isinstance(row['json_shape'], str) or row['json_shape'] not in JSON_SHAPES
                or not isinstance(row['json_wrapping'], str) or row['json_wrapping'] not in {'plain', 'markdown_fence', 'none'}
                or any(type(row[key]) is not int or row[key] < 0 for key in ('native_tool_calls', 'content_chars'))
                or not isinstance(row['content_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['content_sha256'])
                or not isinstance(row['recognised_keys'], list)
                or any(not isinstance(key, str) or key not in SAFE_REPLY_KEYS for key in row['recognised_keys'])
                or row['recognised_keys'] != sorted(set(row['recognised_keys']))):
            return False
        if 'neutral_checks' in row and (not isinstance(row['neutral_checks'], dict)
                or set(row['neutral_checks']) != NEUTRAL_DIAGNOSTIC_KEYS
                or any(type(v) is not bool for v in row['neutral_checks'].values())):
            return False
    return True


def _profile_caps(profile):
    if not isinstance(profile, str) or profile not in {'original', 'remaining-gates-v1'}:
        raise ValueError('joint_admission_profile_invalid')
    return (REMAINING_LIMITS, REMAINING_ROLE_LIMITS) if profile == 'remaining-gates-v1' else (LIMITS, ROLE_LIMITS)


def validate_limits(value, *, profile='original'):
    limits, _ = _profile_caps(profile)
    if (not isinstance(value, dict) or set(value) != set(limits)
            or any((type(n) not in (int, float) if key == 'max_active_seconds' else type(n) is not int)
                or not math.isfinite(n) or not 0 < n <= limits[key] for key, n in value.items())):
        raise ValueError('joint_batch_limits_invalid_or_expanded')
    return dict(value)


def validate_roles(value, *, profile='original'):
    _, roles = _profile_caps(profile)
    if (not isinstance(value, dict) or set(value) != set(roles)
            or any(type(n) is not int or not 1 <= n <= roles[key] for key, n in value.items())):
        raise ValueError('joint_role_limits_invalid_or_expanded')
    return dict(value)


class RoleBudget(TaskBudget):
    def reserve_request(self, payload):
        role = ROLE.get()
        if self.journal.slots.get(self.journal.current_slot) != 'in_progress':
            raise ValueError('joint_active_slot_required')
        if role not in self.journal.role_limits or role not in self.journal.permitted_roles:
            raise ValueError('joint_sdk_role_unbound')
        # All SDK calls hold journal.sdk_lock through reconciliation. A refused
        # role request neither spends another role's allowance nor reaches HTTP.
        if sum(row['role'] == role for row in self.journal.attempts) >= self.journal.role_limits[role]:
            raise TaskBudgetExceeded('joint_role_budget_exhausted:' + role)
        return super().reserve_request(payload)


def inspect_ledger(path, *, identity=None):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or set(value) != {'identity', 'budget', 'roles', 'attempts', 'slots', 'io', 'journal_write_retries'}
            or not isinstance(value['identity'], dict) or identity is not None and value['identity'] != identity
            or not isinstance(value['attempts'], list) or not isinstance(value['slots'], dict)
            or not isinstance(value['io'], list) or not isinstance(value['budget'], dict)
            or not isinstance(value['journal_write_retries'], list)):
        raise ValueError('joint_ledger_identity_invalid')
    profile = value['identity'].get('admission_profile', 'original')
    roles = validate_roles(value['roles'], profile=profile)
    budget = value['budget']
    limits = validate_limits({key: budget.get(key) for key in LIMITS}, profile=profile)
    probe = TaskBudget(**limits, managed=True)
    probe.absorb_prior(budget)
    if (probe.reason == 'task_budget_snapshot_invalid'
            or any(type(budget.get(key)) is not int or budget[key] < 0
                   for key in ('requests', 'tools', 'tokens', 'reported_tokens', 'estimated_requests'))
            or budget.get('requests') != len(value['attempts'])
            or budget.get('tools') != len(value['io']) or budget.get('requests', -1) < 0
            or budget.get('tools', -1) < 0 or budget['requests'] > limits['max_requests']
            or budget['tools'] > limits['max_tools']):
        raise ValueError('joint_ledger_counters_invalid')
    declared = value['identity'].get('slots', [])
    if (not isinstance(declared, list) or not declared or any(not isinstance(s, str) or not s for s in declared)
            or len(declared) != len(set(declared))):
        raise ValueError('joint_ledger_declared_slots_invalid')
    valid_slots = set(declared)
    permitted = value['identity'].get('permitted_roles', list(roles))
    adapter_allowed = value['identity'].get('adapter_dispatch_allowed', True)
    if (not isinstance(permitted, list) or not permitted
            or any(not isinstance(role, str) or role not in roles for role in permitted)
            or len(permitted) != len(set(permitted)) or type(adapter_allowed) is not bool
            or not adapter_allowed and value['io']):
        raise ValueError('joint_ledger_dispatch_policy_invalid')
    if (not valid_slots or set(value['slots']) - valid_slots
            or any(state not in {'in_progress', 'finished'} for state in value['slots'].values())
            or any(not isinstance(row, dict) or type(row.get('index')) is not int or row.get('index') != index + 1
                or row.get('role') not in roles or row.get('role') not in permitted or row.get('slot') not in value['slots']
                or row.get('state') not in {'reserved', 'response', 'error'}
                for index, row in enumerate(value['attempts']))
            or any(sum(row['role'] == role for row in value['attempts']) > cap for role, cap in roles.items())):
        raise ValueError('joint_ledger_slots_or_roles_invalid')
    if any(not isinstance(row, dict) or set(row) != {'index', 'slot', 'kind'}
           or type(row['index']) is not int or row['index'] != index + 1 or row['slot'] not in value['slots']
           or row['kind'] not in {'file_adapter', 'query_search', 'query_read', 'task_material_read'}
           for index, row in enumerate(value['io'])):
        raise ValueError('joint_ledger_io_invalid')
    if any(not isinstance(row, dict) or set(row) != {'slot', 'winerror', 'delay_seconds'}
           or row['slot'] not in {'', *value['slots']} or type(row['winerror']) is not int or row['winerror'] != 5
           or type(row['delay_seconds']) not in (int, float) or row['delay_seconds'] not in (.02, .04, .08, .16)
           for row in value['journal_write_retries']):
        raise ValueError('joint_ledger_write_retry_invalid')
    if any(type(row.get('forwarded')) is not bool
           or row.get('sdk_usage') is not None and (not isinstance(row['sdk_usage'], dict)
               or any(type(n) is not int or n < 0 for n in row['sdk_usage'].values()))
           or 'payload_hash' in row and (not isinstance(row['payload_hash'], str)
               or not re.fullmatch('[0-9a-f]{64}', row['payload_hash']))
           or 'response_metadata' in row and not valid_response_metadata(row['response_metadata'])
           or 'http_timeout_seconds' in row and (type(row['http_timeout_seconds']) not in (int, float)
               or not math.isfinite(row['http_timeout_seconds']) or not 0 < row['http_timeout_seconds'] <= 60)
           or 'error_cause_types' in row and (not isinstance(row['error_cause_types'], list)
               or not 1 <= len(row['error_cause_types']) <= 6
               or any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', name)
                      for name in row['error_cause_types']))
           for row in value['attempts']):
        raise ValueError('joint_ledger_attempt_metadata_invalid')
    return value


class JointLedger:
    def __init__(self, path, identity, limits=None, role_limits=None):
        self.path = Path(path)
        # Inspecting an interrupted reservation is permitted; executing it a
        # second time is not. A new human-approved batch needs a new journal.
        if self.path.exists():
            inspect_ledger(self.path, identity=identity)
            raise ValueError('joint_batch_already_exists_no_automatic_retry')
        self.identity = copy.deepcopy(identity)
        profile = identity.get('admission_profile', 'original')
        limit_caps, role_caps = _profile_caps(profile)
        self.role_limits = validate_roles(role_caps if role_limits is None else role_limits, profile=profile)
        permitted = identity.get('permitted_roles', list(self.role_limits))
        if (not isinstance(permitted, list) or not permitted
                or any(not isinstance(role, str) or role not in self.role_limits for role in permitted)
                or len(permitted) != len(set(permitted))
                or type(identity.get('adapter_dispatch_allowed', True)) is not bool):
            raise ValueError('joint_permitted_roles_invalid')
        self.permitted_roles = frozenset(permitted)
        self.budget = RoleBudget(**validate_limits(limit_caps if limits is None else limits, profile=profile), managed=True,
                                 profile='joint-chat-execution-expression-batch')
        self.budget.journal = self
        self.attempts, self.slots, self.io, self.write_retries = [], {}, [], []
        self.current_slot = ''
        self.sdk_lock = threading.RLock()
        self.last_saved_hash = None
        self.save()

    def save(self):
        with self.sdk_lock:
            if self.path.exists() and canonical_hash(json.loads(self.path.read_text(encoding='utf-8'))) != self.last_saved_hash:
                raise ValueError('joint_journal_changed_by_another_writer')
            value = dict(identity=self.identity, budget=self.budget.snapshot(), roles=self.role_limits,
                         attempts=self.attempts, slots=self.slots, io=self.io, journal_write_retries=self.write_retries)
            temporary = self.path.with_suffix(self.path.suffix + '.new')
            for attempt in range(5):
                with temporary.open('w', encoding='utf-8', newline='') as stream:
                    json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.replace(temporary, self.path)
                    self.last_saved_hash = canonical_hash(value)
                    return
                except PermissionError as exc:
                    if getattr(exc, 'winerror', None) != 5 or attempt == 4:
                        raise
                    delay = .02 * 2 ** attempt
                    self.write_retries.append(dict(slot=self.current_slot, winerror=5, delay_seconds=delay))
                    time.sleep(delay)

    @contextmanager
    def role(self, role):
        with self.sdk_lock:
            if role not in self.permitted_roles:
                raise ValueError('joint_sdk_role_unbound')
            token = ROLE.set(role)
            try:
                yield
            finally:
                ROLE.reset(token)

    @contextmanager
    def activity(self):
        # Count the entire active case once, including host work and expression
        # waits, without replacing its own cumulative task budget context.
        try:
            with bind_task_budget(self.budget), bind_task_budget(None):
                yield
        finally:
            self.save()

    def claim(self, slot):
        with self.sdk_lock:
            if slot not in self.identity['slots'] or slot in self.slots:
                raise ValueError('joint_slot_invalid_or_already_claimed')
            if self.budget.stop_reason():
                raise TaskBudgetExceeded(self.budget.reason)
            self.current_slot = slot
            self.slots[slot] = 'in_progress'
            self.save()

    def finish(self, slot):
        with self.sdk_lock:
            if self.slots.get(slot) != 'in_progress':
                raise ValueError('joint_slot_finish_invalid')
            self.slots[slot] = 'finished'
            # An admitted HTTP call can finish after the active-time cap.
            # Persist its derived stop reason in the same terminal snapshot.
            self.budget.stop_reason()
            self.save()

    def use_tool(self, kind='file_adapter'):
        with self.sdk_lock:
            if self.identity.get('adapter_dispatch_allowed') is False:
                raise ValueError('joint_adapter_dispatch_forbidden')
            if self.slots.get(self.current_slot) != 'in_progress':
                raise ValueError('joint_active_slot_required')
            if kind not in {'file_adapter', 'query_search', 'query_read', 'task_material_read'}:
                raise ValueError('joint_io_kind_invalid')
            self.budget.use_tool()
            self.io.append(dict(index=len(self.io) + 1, slot=self.current_slot, kind=kind))
            self.save()  # Persist before IO, including an IO which later fails.


class FixedChatTransport(httpx.BaseTransport):
    def __init__(self, runtime, *, live, forward_transport=None):
        if live and forward_transport is not None or not live and forward_transport is None:
            raise ValueError('joint_chat_transport_mode_invalid')
        self.runtime, self.live = runtime, live
        self.forward = httpx.Client(timeout=runtime.http_timeout_seconds, follow_redirects=False,
            **({'transport': forward_transport, 'trust_env': False} if not live else {}))

    def handle_request(self, request):
        target = urlsplit(CHAT_ENDPOINT)
        if (request.method != 'POST' or request.url.scheme != target.scheme
                or request.url.host != target.hostname or request.url.port not in (None, 443)
                or request.url.path != target.path + '/chat/completions' or request.url.query):
            raise ValueError('joint_chat_http_target_denied')
        value = json.loads(request.content)
        role = ROLE.get()
        if (role not in {'chat', 'expression', 'check'} or value.get('model') != CHAT_MODEL
                or value.get('stream') or not 1 <= value.get('max_tokens', 0) <= 4096
                or role != 'chat' and (value.get('tools') or value.get('tool_choice'))):
            raise ValueError('joint_chat_payload_or_role_invalid')
        self.runtime.current_attempt.update(transport_kind='provider' if self.live else 'mock_forward', forwarded=True)
        return self.forward.send(request, stream=True)

    def close(self):
        self.forward.close()


class ChatSDKRuntime(LLMRuntime):
    def __init__(self, ledger, key, *, live, forward_transport=None, http_timeout_seconds=25):
        if (type(http_timeout_seconds) not in (int, float) or not math.isfinite(http_timeout_seconds)
                or not 0 < http_timeout_seconds <= 60):
            raise ValueError('joint_http_timeout_invalid')
        self.http_timeout_seconds = float(http_timeout_seconds)
        self.ledger, self.calls, self.payloads, self.errors = ledger, [], [], []
        self.current_attempt = {}
        self.transport = FixedChatTransport(self, live=live, forward_transport=forward_transport)
        self.sdk = OpenAI(api_key=key, base_url=CHAT_ENDPOINT, timeout=self.http_timeout_seconds, max_retries=0,
                          http_client=httpx.Client(transport=self.transport, trust_env=False))
        self.sdk.protocol = 'openai'
        create = self.sdk.chat.completions.create

        def bounded_create(**payload):
            role = ROLE.get() or 'chat'
            with ledger.role(role):
                payload = {**payload, 'max_tokens': min(payload.get('max_tokens') or 4096, 4096)}
                requested_timeout = payload.get('timeout', self.http_timeout_seconds)
                if (type(requested_timeout) not in (int, float) or not math.isfinite(requested_timeout)
                        or requested_timeout <= 0):
                    raise ValueError('joint_http_timeout_invalid')
                payload['timeout'] = min(requested_timeout, self.http_timeout_seconds)
                reservation = ledger.budget.reserve_request(payload)
                self.payloads.append(copy.deepcopy(payload))
                attempt = dict(index=len(ledger.attempts) + 1, slot=ledger.current_slot, role=role,
                               state='reserved', forwarded=False, sdk_usage=None,
                               payload_hash=canonical_hash(payload), http_timeout_seconds=payload['timeout'],
                               native_tools=[row.get('function', {}).get('name') for row in payload.get('tools', [])])
                ledger.attempts.append(attempt)
                self.calls.append(attempt)
                self.current_attempt = attempt
                ledger.save()
                started = time.monotonic()
                try:
                    with bind_task_budget(ledger.budget):
                        response = create(**payload)
                    ledger.budget.reconcile(reservation, response)
                    attempt.update(state='response', sdk_usage=response_usage(response),
                                   response_metadata=response_metadata(response, request_payload=payload))
                    return response
                except Exception as exc:
                    attempt.update(state='error', error=type(exc).__name__, http_status=getattr(exc, 'status_code', None),
                                   error_cause_types=error_cause_types(exc))
                    raise
                finally:
                    attempt['seconds'] = round(time.monotonic() - started, 3)
                    ledger.save()

        self.sdk.chat.completions.create = bounded_create
        super().__init__(runtime_policy=RuntimePolicy(http_attempts=2, retry_backoff_seconds=0,
            stream_attempts=1, chat_timeout_seconds=25, circuit_failure_threshold=100000))

    def _env(self, name, default=''):
        return default

    def _make_chat_bundle(self):
        return ModelBundle(self.sdk, CHAT_MODEL)

    _make_aux_bundle = _make_chat_bundle

    def _tool_profile(self, bundle):
        return ProviderToolProfile(supports_native_tools=True, force_response_json=False,
                                   verified=False, notes='Isolated existing endpoint candidate only.')

    def _build_thinking_kwargs(self, **kwargs):
        return {}  # Preserve the existing Gemini endpoint's default mode.

    def _build_cache_kwargs(self, **kwargs):
        return {}

    def _record_context_budget(self, record):
        return {}

    def _maybe_record_prompt_audit(self, **kwargs):
        return None

    def _note_error(self, exc, **kwargs):
        self.errors.append(dict(phase=kwargs.get('phase'), error=type(exc).__name__))

    def close(self):
        self.sdk.close()


class ExecutionSDKRuntime(acceptance.AcceptanceRuntime):
    def __init__(self, fixture, ledger, key, **kwargs):
        self.phase, self.payloads = 'metadata', []
        super().__init__(fixture, ledger, key, **kwargs)

    def attempt_metadata(self):
        return dict(role='execution', slot=self.ledger.current_slot, phase=self.phase)

    def continues_execution_history(self):
        return self.phase == 'execution'

    def _build_payload(self, **kwargs):
        payload = super()._build_payload(**kwargs)
        self.payloads.append(copy.deepcopy(payload))
        return payload

    def call_chat_json(self, **kwargs):
        with self.ledger.role('execution'):
            self.phase = 'execution' if kwargs.get('native_tools') else 'metadata'
            return super().call_chat_json(**kwargs)


class RoleExpressionAdapter(RuntimeExpressionAdapter):
    def __init__(self, runtime, role):
        super().__init__(runtime)
        self.role = role

    @property
    def identity(self):
        return canonical_hash(dict(transport=super().identity, role=self.role))

    def call_expression_json(self, **kwargs):
        with self.runtime.ledger.role(self.role):
            return super().call_expression_json(**kwargs)

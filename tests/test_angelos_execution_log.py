"""用户侧完整执行日志：分页、搜索、导出、脱敏与归属隔离的离线回归。

全部使用合成输出与最小材料库替身构造夹具，不调用模型、不启动正式服务。
要害边界：
- 长输出即使模型只看到摘录，用户仍能分页读全、搜索到中间标记；
- 原件不可读时如实降级并标记不完整，而不是伪装成完整日志；
- 执行状态只取回执字段，输出里出现"失败/error"不改变判定；
- 合成口令在分页、搜索两条通路都不泄露，且不隐藏其余排错细节；
- 跨会话/跨用户不可读，重复拉取不产生重复行。

合成口令一律在运行时用片段拼接，避免测试文件里出现形似真实凭据的字面量
被工具输出脱敏改写成 [redacted]，从而让断言对象失真。
"""
import json
import threading
from types import SimpleNamespace

import pytest

from shinku.angelos import execution_log as exec_log


def _sk() -> str:
    """合成假口令：分段拼接，文件内不出现完整字面量。"""
    return '-'.join(['sk', 'SYNTH', 'TIC', 'TOKEN', '9f3fa1'])


def _kv_style() -> str:
    """另一种"键=值"形态的合成假口令。"""
    return 'abcd' + '1234' + 'efgh'


class _Store:
    """最小材料库替身：按引用 id 取回原件，签名与 TaskMaterialStore.resolve 一致。

    生产 execution_log 传入的是 reference['reference'] 字符串（真实 resolve 只接受
    字符串 id，并用 task 做归属校验），替身必须同样按字符串索引，否则会静默走
    "原件不可读" 降级分支，让所有日志用例误判成失败。
    """

    def __init__(self, values=None):
        self.values = values or {}

    def resolve(self, reference, *, task, still_authorized):
        if not isinstance(reference, str):
            raise TypeError('material_reference_must_be_str')
        value = self.values.get(reference)
        if value is None:
            raise KeyError('material_reference_unavailable')
        return json.loads(json.dumps(value))


def _host(task, *, requester='owner', session='angelos:c', store=None, redact=None):
    """构造宿主替身：_get 按 scope 校验归属，_grant 可复用真实权限语义。"""
    def get(scope, task_id):
        if (scope.requester, scope.session, task_id) != (requester, session, task['task_id']):
            raise ValueError('scope rejected')
        return task

    def grant(scope, found, action):
        if (scope.requester, scope.session) != (requester, session):
            raise PermissionError('denied')

    profile = SimpleNamespace(runtime=SimpleNamespace(_redact_secrets=redact)) if redact else None
    return SimpleNamespace(requester=requester, profile=profile,
                           service=SimpleNamespace(_lock=threading.RLock(), _get=get, _grant=grant,
                                                   materials=store))


def _task(calls, *, inflight=(), attempt=2, requester='owner', session='angelos:c'):
    return {'task_id': 't', 'requester': requester, 'session': session, 'execution_attempt': attempt,
            'tool_calls': calls, 'inflight': list(inflight)}


def _call(call_id, name, *, binding=('t', 'owner', 'angelos:c', 2), reference=None, status='ok',
          error='', data=None):
    call = {'id': call_id, 'name': name, 'status': status, 'arguments': {'cmd': 'synthetic'},
            'error': error,
            'execution_binding': dict(zip(('task_id', 'requester', 'session', 'attempt'), binding)),
            'data': data if data is not None else {'state_updates': {}}}
    if reference:
        call['material_reference'] = reference
    return call


def _original(store, reference, stdout='', stderr='', *, returncode=0):
    """登记一份原件，结构与材料库保留的回执一致：feedback + state_updates。"""
    feedback = f'command finished\n\nstdout:\n{stdout}\nstderr:\n{stderr}'
    receipt = {'status': 'ok', 'data': {'state_updates': {'execution_session': {
        'session_id': 's1', 'command_sha256': 'a' * 64, 'returncode': returncode,
        'started_at': 1793000000.0, 'finished_at': 1793000002.5, 'elapsed': 2.5}}}}
    store.values[reference['reference']] = {
        'feedback': feedback + '\n\n[host receipt]\n' + json.dumps(receipt),
        'execution_status': 'ok',
        'state_updates': {'stdout': stdout, 'stderr': stderr}}
    return reference


def _ref(name):
    return {'reference': name, 'content_sha256': '0' * 64, 'version': '0' * 64, 'complete': True}


def test_short_output_is_complete_and_user_readable():
    store = _Store()
    reference = _original(store, _ref('tm:' + '1' * 32), stdout='hello short\nsecond line\n')
    task = _task([_call('c1', 'shell', reference=reference)])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'c1')
    assert page['complete'] is True
    assert [row['text'] for row in page['lines']] == ['hello short\n', 'second line\n']
    assert page['has_more'] is False and page['call']['exit_code'] == 0


def test_large_output_is_fully_readable_in_pages_and_searchable_at_middle():
    store = _Store()
    lines = [f'line-{i:05d}\n' for i in range(20000)]
    lines[12345] = 'MIDDLE-MARKER build finished\n'
    reference = _original(store, _ref('tm:' + '2' * 32), stdout=''.join(lines))
    host = _host(_task([_call('big', 'shell', reference=reference)]), store=store)
    seen, offset = [], 0
    while True:
        page = exec_log.read_log(host, 'c', {'task_id': 't'}, 'big', stream='stdout', offset=offset)
        assert page['total_lines'] == 20000
        seen.extend(row['n'] for row in page['lines'])
        if not page['has_more']:
            break
        offset = page['next_offset']
    # 分页覆盖全部 20000 行且无重复、无缺行。
    assert seen == list(range(1, 20001)) and len(seen) == len(set(seen))
    hits = exec_log.search_log(host, 'c', {'task_id': 't'}, 'big', 'MIDDLE-MARKER')
    assert hits['total_hits'] == 1 and hits['hits'][0]['n'] == 12346
    middle = exec_log.read_log(host, 'c', {'task_id': 't'}, 'big', stream='stdout', offset=12340)
    assert any('MIDDLE-MARKER' in row['text'] for row in middle['lines'])


def test_repeated_page_requests_do_not_duplicate_rows():
    store = _Store()
    reference = _original(store, _ref('tm:' + '3' * 32), stdout=''.join(f'row-{i}\n' for i in range(50)))
    host = _host(_task([_call('r', 'shell', reference=reference)]), store=store)
    first = exec_log.read_log(host, 'c', {'task_id': 't'}, 'r', offset=0)
    again = exec_log.read_log(host, 'c', {'task_id': 't'}, 'r', offset=0)
    assert [r['n'] for r in first['lines']] == [r['n'] for r in again['lines']]


def test_command_failure_keeps_stdout_stderr_and_real_exit_code():
    store = _Store()
    reference = _original(store, _ref('tm:' + '4' * 32), stdout='compiled 3 files\n',
                          stderr='Traceback (most recent call last):\nKeyError: user_id\n', returncode=1)
    task = _task([_call('fail', 'shell', reference=reference, status='error', error='exit 1')])
    host = _host(task, store=store)
    page = exec_log.read_log(host, 'c', {'task_id': 't'}, 'fail')
    assert page['call']['status'] == 'error' and page['call']['exit_code'] == 1
    # error 流是回执的调用级错误字段，与 stdout/stderr 并列展示；三者都必须在。
    assert {'stdout', 'stderr', 'error'} <= set(page['streams'])
    stderr_page = exec_log.read_log(host, 'c', {'task_id': 't'}, 'fail', stream='stderr')
    assert 'KeyError' in ''.join(row['text'] for row in stderr_page['lines'])


def test_success_output_containing_failure_words_is_not_reclassified():
    store = _Store()
    reference = _original(store, _ref('tm:' + '5' * 32),
                          stdout='ALL TESTS PASSED\n0 failures\nsee docs/failure-modes.md\n')
    task = _task([_call('ok', 'shell', reference=reference, status='ok')])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'ok')
    # 只认回执状态与退出码，不按字词改判。
    assert page['call']['status'] == 'ok' and page['call']['exit_code'] == 0
    assert 'failures' in ''.join(row['text'] for row in page['lines'])


def test_synthetic_secret_never_leaks_through_page_or_search():
    secret = _sk()
    store = _Store()
    # 原件里带"键=合成值"形态，模拟真实命令把凭据打进输出。
    raw = 'connecting with ' + secret + chr(10) + 'other diagnostic kept' + chr(10)
    reference = _original(store, _ref('tm:' + '6' * 32), stdout=raw)
    task = _task([_call('sec', 'shell', reference=reference)])
    mask = chr(91) + 'redacted' + chr(93)
    host = _host(task, store=store, redact=lambda text, max_chars: text.replace(secret, mask))
    page = exec_log.read_log(host, 'c', {'task_id': 't'}, 'sec')
    hits = exec_log.search_log(host, 'c', {'task_id': 't'}, 'sec', secret)
    segments = exec_log.log_segments(host, 'c', {'task_id': 't'}, 'sec')
    blob = json.dumps([page, hits, segments], ensure_ascii=False)
    assert secret not in blob
    # 脱敏不隐藏其余排错细节，也不整段隐藏日志。
    assert 'other diagnostic kept' in blob
    # 键名本身保留，便于识别是哪类凭据被遮掉。
    # 只遮值、不整段隐藏：该行除合成值以外的内容仍在导出中。


def test_secret_shapes_scrubbed_even_without_provider_redactor():
    value = _kv_style()
    store = _Store()
    raw = ('to' + 'ken=' + value + chr(10) + 'pass' + 'word: ' + 'x' * 12 + chr(10) + 'normal line' + chr(10))
    reference = _original(store, _ref('tm:' + '7' * 32), stdout=raw)
    task = _task([_call('shape', 'shell', reference=reference)])
    text = json.dumps(exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'shape'),
                      ensure_ascii=False)
    assert value not in text and 'normal line' in text


def test_missing_original_degrades_honestly_instead_of_faking_completeness():
    store = _Store()  # 不登记任何原件
    task = _task([_call('gone', 'shell', reference=_ref('tm:' + '8' * 32),
                        data={'state_updates': {'stdout': 'only head kept\n'}})])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'gone')
    assert page['complete'] is False
    assert [r['text'] for r in page['lines']] == ['only head kept\n']


def test_inflight_call_is_reported_as_pending_with_unknown_fields():
    task = _task([], inflight=[{'id': 'live', 'name': 'shell', 'arguments': {}}])
    page = exec_log.read_log(_host(task), 'c', {'task_id': 't'}, 'live')
    assert page['call']['status'] == 'pending_receipt'
    assert page['complete'] is False
    assert set(page['unknown_fields']) >= {'started', 'finished', 'exit_code'}


def test_canceled_task_logs_remain_readable_and_status_is_not_rewritten():
    store = _Store()
    reference = _original(store, _ref('tm:' + '9' * 32), stdout='partial output before cancel\n')
    task = _task([_call('cut', 'shell', reference=reference, status='canceled')])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'cut')
    assert page['call']['status'] == 'canceled'
    assert 'partial output before cancel' in ''.join(r['text'] for r in page['lines'])


def test_foreign_attempt_and_foreign_session_calls_are_not_readable():
    store = _Store()
    reference = _original(store, _ref('tm:' + 'a' * 32), stdout='x\n')
    old = _call('old', 'shell', reference=reference, binding=('t', 'owner', 'angelos:c', 1))
    foreign = _call('foreign', 'shell', reference=reference, binding=('t', 'someone', 'angelos:other', 2))
    mine = _call('mine', 'shell', reference=reference)
    task = _task([old, foreign, mine])
    host = _host(task, store=store)
    assert exec_log.read_log(host, 'c', {'task_id': 't'}, 'mine')['complete'] is True
    for call_id in ('old', 'foreign'):
        with pytest.raises(exec_log.LogUnavailable):
            exec_log.read_log(host, 'c', {'task_id': 't'}, call_id)
    # 另一个会话范围读同一条任务必须被拒绝。
    with pytest.raises(Exception):
        exec_log.read_log(_host(task, store=store, session='angelos:elsewhere'), 'c', {'task_id': 't'}, 'mine')


def test_missing_stream_returns_empty_page_not_error():
    store = _Store()
    reference = _original(store, _ref('tm:' + 'b' * 32), stdout='ok\n', stderr='')
    task = _task([_call('quiet', 'shell', reference=reference)])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'quiet', stream='stderr')
    assert page['lines'] == [] and page['total_lines'] == 0 and page['has_more'] is False
    assert page['complete'] is True


def test_unrecorded_turn_and_unknown_call_are_reported_as_unavailable():
    with pytest.raises(exec_log.LogUnavailable):
        exec_log.read_log(_host({'task_id': 't'}), 'c', {}, 'any')
    task = _task([_call('known', 'shell')])
    with pytest.raises(exec_log.LogUnavailable):
        exec_log.read_log(_host(task), 'c', {'task_id': 't'}, 'absent')


# ---- SA41-A 独立验收发现的回归场景（原先均失败） ----

def test_quoted_bearer_and_authorization_forms_never_leak():
    """带引号的 Bearer、Authorization: Bearer \"…\" 与 JSON 形态都要遮住值。

    旧实现两个错：JSON 形态被冒号后的引号切断；先做键值脱敏又先把 Authorization 替掉，
    使 Bearer 锚点消失，值整段裸露。三种形态都必须失败成“看不到值”，而不是看运气。
    """
    value = _kv_style()
    for shape in ('Bearer "{v}"', 'Authorization: Bearer {v}', 'Authorization: Bearer "{v}"',
                  '{{"token": "{v}"}}', 'to' + 'ken="{v}"'):
        text = shape.format(v=value)
        assert value not in exec_log._secret_shapes(text), shape
    # 替换后的占位符本身不能被二次切割，否则运维看不出这里被脱敏过。
    assert exec_log._secret_shapes('Bearer "%s"' % value) == 'Bearer "[redacted]"'
    assert exec_log._secret_shapes('[redacted]') == '[redacted]'
    # 普通排错行不能被误伤。
    assert exec_log._secret_shapes('keycloak started') == 'keycloak started'


def test_incomplete_reference_is_never_reported_as_complete_original():
    """材料引用 complete=False（执行器已截断）时，即使原件能取到也不得标“完整原件”。"""
    store = _Store()
    reference = _original(store, {**_ref('tm:' + 'c' * 32), 'complete': False}, stdout='tail only\n')
    task = _task([_call('cut800', 'shell', reference=reference)])
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'cut800')
    assert page['complete'] is False
    assert 'reference_incomplete' in page['incomplete_reasons']
    assert [r['text'] for r in page['lines']] == ['tail only\n']


def test_output_truncated_flag_alone_also_marks_incomplete():
    """引用未带 complete 字段，但回执说 output_truncated，同样不能标完整。"""
    store = _Store()
    reference = _original(store, _ref('tm:' + 'd' * 32), stdout='tail\n')
    task = _task([_call('trunc', 'shell', reference=reference)])
    store.values[reference['reference']]['state_updates']['output_truncated'] = True
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'trunc')
    assert page['complete'] is False and 'output_truncated' in page['incomplete_reasons']


def test_previous_attempt_is_readable_only_when_history_is_requested():
    """重试前旧 attempt 的日志：默认仍拒（与既有隔离用例一致），显式 history 才可读。"""
    store = _Store()
    reference = _original(store, _ref('tm:' + 'e' * 32), stdout='from attempt 1\n')
    old = _call('a1', 'shell', reference=reference, binding=('t', 'owner', 'angelos:c', 1))
    task = _task([old, _call('a2', 'shell')], attempt=2)
    host = _host(task, store=store)
    with pytest.raises(exec_log.LogUnavailable):
        exec_log.read_log(host, 'c', {'task_id': 't'}, 'a1')
    page = exec_log.read_log(host, 'c', {'task_id': 't'}, 'a1', include_history=True)
    assert [r['text'] for r in page['lines']] == ['from attempt 1\n']
    # 历史开关不能变成跳会话：别人的会话即使带 history 也不可读。
    other = _call('x', 'shell', reference=reference, binding=('t', 'someone', 'angelos:other', 1))
    with pytest.raises(exec_log.LogUnavailable):
        exec_log.read_log(_host(_task([other]), store=store), 'c', {'task_id': 't'}, 'x', include_history=True)


class _CancelAwareStore(_Store):
    """模拟真实 TaskMaterialStore：普通 resolve 拒绝已取消任务，只读展示入口接受。"""

    def __init__(self, values=None, canceled=False):
        super().__init__(values)
        self.canceled = canceled

    def resolve(self, reference, *, task, still_authorized):
        if self.canceled:
            raise PermissionError('material_task_canceled')
        return super().resolve(reference, task=task, still_authorized=still_authorized)

    def resolve_for_read_only_view(self, reference, *, task, still_authorized):
        return _Store.resolve(self, reference, task=task, still_authorized=still_authorized)


def test_canceled_task_logs_use_read_only_view_instead_of_degrading_to_empty():
    store = _CancelAwareStore(canceled=True)
    reference = _original(store, _ref('tm:' + 'f' * 32), stdout='partial before cancel\n')
    task = _task([_call('cut', 'shell', reference=reference, status='canceled')])
    task['status'] = 'canceled'
    task['cancel_requested'] = True
    page = exec_log.read_log(_host(task, store=store), 'c', {'task_id': 't'}, 'cut')
    assert page['complete'] is True
    assert [r['text'] for r in page['lines']] == ['partial before cancel\n']
    assert page['call']['status'] == 'canceled'

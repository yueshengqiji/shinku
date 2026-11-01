"""恢复观察的合成回归：旧回执不变、当前文件重验、进程身份与只读钩子边界。"""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from shinku.agent_runtime.conversation import ConversationTurn
from shinku.agent_runtime.recovery_observations import changed_file_observation, observe_process
from shinku.agent_runtime.recovery import _effect_reason, record_recovery


def context():
    turn = ConversationTurn(SimpleNamespace(conversation_reply_delivery=True), None, 'fixture', 1, None)
    return {'_agent_conversation_turn': turn, 'agent_task_state': {
        'task_id': 'fixture', 'requester': 'owner', 'session': 'fixture-session', 'execution_attempt': 1}}


class FileObserver:
    def __init__(self, path):
        self.path = path
    def agent_verify_result(self, *, receipt, context):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {'sha256': digest, 'result_valid': digest == receipt['sha256']}


@pytest.mark.parametrize('change', ['changed', 'same', 'missing', 'revoked', 'no_digest', 'strict', 'fake_turn', 'unknown', 'send', 'path_mismatch', 'malformed_artifacts'])
def test_file_observation_requires_real_read_and_conversation_mode(tmp_path, change):
    path = tmp_path/'fixture.txt'; path.write_text('before')
    call = {'status': 'ok', 'executed': True, 'data': {'state_updates': {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'verified_effects': ['write']}}}
    observer = FileObserver(path); ctx = context()
    if change != 'same':path.write_text('after')
    if change == 'missing':path.unlink()
    if change == 'revoked':observer.agent_verify_result = lambda **kw: {'result_valid': False}
    if change == 'no_digest':observer.agent_verify_result = lambda **kw: {'result_valid': False, 'file_text': 'after'}
    if change == 'strict':ctx = {}
    if change == 'fake_turn':ctx = {'_agent_conversation_turn': {'reply_delivery': True}}
    if change == 'unknown':call['status'] = 'error'
    if change == 'send':call['data']['state_updates']['verified_effects'].append('send')
    if change == 'path_mismatch':observer.agent_verify_result = lambda **kw: {'path': 'other', 'sha256': 'b'*64}
    if change == 'malformed_artifacts':call['data']['state_updates']['artifact_receipts'] = 42
    original = deepcopy(call)
    result = changed_file_observation(call, observer, ctx)
    assert bool(result) is (change == 'changed')
    assert call == original
    if result:
        assert result['completion_accepted'] is False and result['requires_revalidation'] is True
        # 接续观测不能直接作为完成验收成功；旧写入仍不匹配当前文件。
        assert observer.agent_verify_result(receipt=call['data']['state_updates'],context=ctx)['result_valid'] is False


class SessionObserver:
    agent_completion_effects = ('command_run',)
    def __init__(self, old, observed):
        self.old, self.observed = old, observed
    def agent_recovery_observation(self, *, call, context):
        assert call == self.old
        return self.observed
    def agent_execution_event(self, *, receipt, context):
        row = receipt['execution_session']
        return {'kind': 'command_completed', **{k:row[k] for k in ('session_id','command_sha256','cursor')}}


def session_case():
    ctx = context(); binding = {**ctx['agent_task_state'], 'attempt': 1, 'work_id': 'fixture', 'work_attempt': 1}
    old = {'status': 'ok', 'executed': True, 'execution_binding': binding, 'arguments': {'action':'start'},
           'data': {'state_updates': {'execution_session': {'session_id':'run', 'command_sha256':'a'*64, 'status':'running', 'cursor':0}}}}
    observed = {'status':'ok', 'state_updates': {'verified_effects':['command_run'], 'execution_session': {
        'session_id':'run', 'command_sha256':'a'*64, 'status':'completed', 'returncode':0, 'cursor':1}}}
    return ctx, old, observed


@pytest.mark.parametrize('case', ['terminal', 'failed_poll', 'unknown_start', 'cross_task', 'future_attempt', 'wrong_command', 'wrong_session', 'unauthenticated', 'hook_exception', 'not_executed'])
def test_terminal_observation_never_replays_or_rewrites_original(case):
    ctx, old, observed = session_case()
    if case in {'failed_poll','unknown_start'}:
        old.update(status='error', data={'state_updates':{}})
        old['arguments']['action']='poll' if case=='failed_poll' else 'start'
        observed['observation_only']=case=='failed_poll'
    if case=='cross_task':old['execution_binding']['task_id']='other'
    if case=='future_attempt':old['execution_binding']['attempt']=2
    if case=='wrong_command':observed['state_updates']['execution_session']['command_sha256']='b'*64
    if case=='wrong_session':observed['state_updates']['execution_session']['session_id']='other'
    if case=='not_executed':old['executed']=False
    handler=SessionObserver(deepcopy(old),observed)
    if case=='unauthenticated':handler.agent_execution_event=lambda **kw: None
    if case=='hook_exception':handler.agent_recovery_observation=lambda **kw: (_ for _ in ()).throw(OSError())
    original=deepcopy(old);proof,audit=observe_process(old,handler,ctx)
    accepted=case in {'terminal','failed_poll'}
    assert bool(audit) is accepted and old==original
    if accepted:
        assert _effect_reason(proof,handler,ctx)==''
    else:
        assert proof==old


def test_revalidation_is_retained_without_claiming_goal_completion():
    task={'status':'waiting_user','notes':{},'stop_reason':'fixture_stop'}
    observation={'changes':[{'path':'sample.txt'}], 'completion_accepted':False}
    review={'attempt':1,'version':2,'mode':'restart','checks_hash':'fixture','checks':[{'revalidation':observation}]}
    record_recovery(task,review,request_key='fixture',reason='explicit_continuation')
    assert task['notes']['recovery_revalidation']['completion_accepted'] is False
    assert task['notes']['recovery_revalidation']['observations']==[observation]
    assert task['recovery_history'][0]['original_stop_reason']=='fixture_stop'
    assert task['recovery_history'][0]['file_revalidations']==[observation]

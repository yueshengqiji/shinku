"""人工提问实际Service/SQLite回归；确定性模型，不使用真实配置和外发。"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import pytest
from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentSupplement, AgentTurnRequest
from shinku.agent_runtime.questions import question_for
from shinku.angelos.host import ServiceHost
from test_agent_runtime_boundary import settled, handle_from
from test_agent_recovery import service_at, Runtime, close

WAIT = {'kind': 'wait', 'reason': 'human_action_required', 'text': '主管，请选择输出为中文还是英文？已完成资料核对，回答后整理。'}
FINAL = {'kind': 'final', 'text': '按主管的选择以中文整理。', 'task_outcome': 'completed'}


def start(root):
    service, _, tool, runtime = service_at(root, Runtime([WAIT, FINAL]), materials=True)
    host = ServiceHost(service, requester='owner', reply_delivery=True)
    port = host.port('one')
    h = port.submit(AgentTurnRequest('q', '整理合成资料，输出语言由我决定'))
    v = settled(port, h.task_id)
    assert v['status'] == 'waiting_user' and not v['worker_active']
    q = question_for(port.result(h.task_id))
    assert q and q['text'] == WAIT['text']
    return service, host, port, runtime, handle_from(v), q


def test_answer_is_scoped_idempotent_and_in_next_model_context(tmp_path):
    service, host, port, runtime, h, q = start(tmp_path)
    try:
        before = port.result(h.task_id)
        for _ in range(5):
            assert question_for(port.result(h.task_id)) == q
            assert len(runtime.calls) == 1
        with pytest.raises(AgentBoundaryError):
            host.port('other').answer_question(h, question_id=q['id'], text='中文')
        with ThreadPoolExecutor(2) as pool:
            answers = list(pool.map(lambda _: port.answer_question(h, question_id=q['id'], text='中文'), range(2)))
        assert all(x.attempt == 2 for x in answers)
        assert settled(port, h.task_id)['status'] == 'completed'
        assert len(runtime.calls) == 2
        assert '中文' in runtime.calls[-1]['user_prompt']
        assert '整理合成资料' in runtime.calls[-1]['user_prompt']
        port.answer_question(h, question_id=q['id'], text='中文')
        assert len(runtime.calls) == 2
        with pytest.raises(AgentBoundaryError, match='question_answer_conflict'):
            port.answer_question(h, question_id=q['id'], text='英文')
        task = service.task_store.get(h.task_id, requester='owner', session='angelos:one')
        assert len(task['material_supplements']) == 1
        assert before['execution_report']['text'] == WAIT['text']
    finally: close(service)


def test_cancel_rejects_late_answer_and_wrong_question_rejected(tmp_path):
    service, host, port, runtime, h, q = start(tmp_path)
    try:
        with pytest.raises(AgentBoundaryError, match='question_not_current'):
            port.answer_question(h, question_id='q:'+'0'*64, text='中文')
        port.cancel(h)
        with pytest.raises(AgentBoundaryError, match='question_canceled'):
            port.answer_question(h, question_id=q['id'], text='中文')
        assert settled(port, h.task_id)['status'] == 'canceled' and len(runtime.calls) == 1
    finally: close(service)


def test_denied_resume_preserves_answer_and_same_answer_can_retry(tmp_path):
    service, host, port, runtime, h, q = start(tmp_path)
    original = service.authorize
    try:
        service.authorize = lambda scope, task, op: replace(original(scope,task,op), resume_safe=False) if op == 'resume' else original(scope,task,op)
        with pytest.raises(AgentBoundaryError, match='resume_effects_unconfirmed'):
            port.answer_question(h, question_id=q['id'], text='中文')
        task = service.task_store.get(h.task_id, requester='owner', session='angelos:one')
        assert task['requests'][-1]['text'] == '中文' and len(runtime.calls) == 1
        service.authorize = original
        port.answer_question(h, question_id=q['id'], text='中文')
        assert settled(port, h.task_id)['status'] == 'completed' and len(runtime.calls) == 2
    finally: close(service)


def test_restart_preserves_wait_and_original_result(tmp_path):
    service, host, port, runtime, h, q = start(tmp_path)
    old = port.result(h.task_id)
    close(service)
    service, _, _, runtime = service_at(tmp_path, Runtime([FINAL]), materials=True)
    try:
        host = ServiceHost(service, requester='owner', reply_delivery=True)
        port = host.port('one')
        view = port.status(h.task_id)
        assert view['status'] == 'waiting_user' and not view['worker_active']
        assert port.result(h.task_id) == old and len(runtime.calls) == 0
        port.answer_question(h, question_id=q['id'], text='中文')
        assert settled(port, h.task_id)['status'] == 'completed'
        assert len(runtime.calls) == 1 and '中文' in runtime.calls[0]['user_prompt']
    finally: close(service)


@pytest.mark.parametrize('events', [False, True])
def test_wait_survives_history_ttl_and_capacity(tmp_path, events):
    service, host, port, runtime, h, q = start(tmp_path)
    try:
        store = service.task_store
        if not events: store.result_events = None
        store.max_entries = 0
        with store._transaction() as db: store._prune(db, store.clock()+store.ttl_seconds+10)
        assert port.result(h.task_id)['execution_report']['text'] == q['text']
    finally: close(service)


def test_saved_answer_survives_restart_before_resume(tmp_path):
    service, host, port, runtime, h, q = start(tmp_path)
    original = service.authorize
    service.authorize = lambda scope, task, op: replace(original(scope,task,op), resume_safe=False) if op == 'resume' else original(scope,task,op)
    with pytest.raises(AgentBoundaryError, match='resume_effects_unconfirmed'):
        port.answer_question(h, question_id=q['id'], text='中文')
    assert port.question(h.task_id)['answer_text'] == '中文'
    close(service)
    service, _, _, runtime = service_at(tmp_path, Runtime([FINAL]), materials=True)
    try:
        port = ServiceHost(service, requester='owner', reply_delivery=True).port('one')
        assert port.question(h.task_id)['answer_received']
        port.answer_question(h, question_id=q['id'], text='中文')
        assert settled(port,h.task_id)['status']=='completed' and len(runtime.calls)==1
    finally: close(service)


@pytest.mark.parametrize('tampered', [False, True])
def test_answer_keeps_real_effect_verification_and_never_replays_write(tmp_path, tampered):
    from test_agent_recovery import write, FINAL as VERIFIED
    from shinku.agent_runtime.demo import DemoFileTool, demo_request
    tool=DemoFileTool(tmp_path)
    service,port,tool,runtime=service_at(tmp_path,Runtime([write(tool),WAIT,VERIFIED]),tool=tool,materials=True)
    try:
        h=port.submit(demo_request(tool));v=settled(port,h.task_id);h=handle_from(v)
        assert v['status']=='waiting_user' and tool.calls==1
        q=port.question(h.task_id)
        if tampered:
            tool.path.write_text('被外部修改的合成文件')
            with pytest.raises(AgentBoundaryError, match='recovery_effect_unconfirmed'):
                port.answer_question(h,question_id=q['id'],text='中文')
            assert len(runtime.calls)==2
        else:
            port.answer_question(h,question_id=q['id'],text='中文')
            assert settled(port,h.task_id)['status']=='completed'
        assert tool.calls==1
    finally: close(service)


def test_active_wait_on_restart_still_requires_recovery(tmp_path):
    from shinku.tasks.coordinator import AgentTaskCoordinator
    path=tmp_path/'workers.db'; old=AgentTaskCoordinator(state_path=path)
    old.submit(lambda _:None,task_id='active',defer_start=True)
    row=old._records['active'];row.status='waiting_user';row.worker_active=True
    old._persist_record(row)
    fresh=AgentTaskCoordinator(state_path=path)
    try:
        assert fresh.get('active')['status']=='failed'
        assert fresh.get('active')['metadata']['recovered_after_restart']
    finally: fresh.shutdown();old.shutdown()


def test_waiting_preserves_expired_original_but_does_not_extend_material_authority(tmp_path):
    from shinku.agent_runtime.materials import MaterialSource
    service,host,port,runtime,h,q=start(tmp_path)
    try:
        store=service.task_store
        task=store.get(h.task_id,requester='owner',session='angelos:one')
        row=service.materials.retain(task,MaterialSource('original','document','合成原件，不允许因等待被清掉'),authorization_ref='fixture')
        task.setdefault('material_package', {}).setdefault('items', []).append(row)
        task=store.save(task)
        now=store.clock();store.clock=lambda:now+store.ttl_seconds+10
        service.materials.prune()
        with store._transaction() as db:
            assert db.execute('SELECT count(*) FROM agent_task_materials WHERE reference=?',(row['reference'],)).fetchone()[0]==1
        with pytest.raises(AgentBoundaryError,match='material_source_expired_or_revoked'):
            service.materials.resolve(row['reference'],task=task,still_authorized=lambda:True)
    finally: close(service)


def test_new_question_supersedes_previous_and_old_answer_never_restarts_it(tmp_path):
    service,host,port,runtime,h,q=start(tmp_path)
    runtime.decisions=[dict(WAIT,text='主管，请确认结果保存的位置。'),FINAL]
    try:
        port.answer_question(h,question_id=q['id'],text='中文')
        v=settled(port,h.task_id);next_q=port.question(h.task_id)
        assert v['status']=='waiting_user' and next_q['id']!=q['id'] and len(runtime.calls)==2
        port.answer_question(h,question_id=q['id'],text='中文')
        assert len(runtime.calls)==2 and port.question(h.task_id)['id']==next_q['id']
        with pytest.raises(AgentBoundaryError,match='task_version_conflict'):
            port.answer_question(h,question_id=next_q['id'],text='只在回答里显示')
        port.answer_question(handle_from(v),question_id=next_q['id'],text='只在回答里显示')
        assert settled(port,h.task_id)['status']=='completed' and len(runtime.calls)==3
    finally:close(service)

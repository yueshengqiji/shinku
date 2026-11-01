"""已交付但目标未完成的任务重启后保持原状；不恢复执行或伪造成功。"""
import json
import sqlite3
import pytest
from shinku.tasks.coordinator import AgentTaskCoordinator, AgentTaskRecord


@pytest.mark.parametrize('status,active', [
    ('blocked', False), ('waiting_user', False), ('blocked', True), ('running', True),
])
def test_restart_preserves_only_quiescent_records(tmp_path, status, active):
    path = tmp_path / 'workers.sqlite3'
    old = AgentTaskCoordinator(state_path=path)
    old.shutdown(wait=True)
    record = AgentTaskRecord(task_id='fixture', status=status, worker_active=active,
        created_at=10.0, updated_at=20.0, finished_at=20.0 if not active else None,
        error='reply_delivered' if not active else '', metadata={'last_stage': 'partial'}).as_dict()
    raw = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO agent_task_records VALUES (?,?,?)', ('fixture', 20, raw))
    new = AgentTaskCoordinator(state_path=path)
    try:
        row = new.get('fixture')
        assert row['status'] == (status if not active else 'failed')
        assert not row['worker_active']
        if not active:
            assert row['error'] == 'reply_delivered' and row['finished_at'] == 20
            with sqlite3.connect(path) as db:
                assert db.execute('SELECT payload FROM agent_task_records').fetchone()[0] == raw
        else:
            assert row['metadata']['recovered_after_restart']
    finally:
        new.shutdown(wait=True)

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gray_memory_sequence as sequence
import gray_memory_stage_a as a


def test_sequence_correction_uses_known_source_after_recent_page_fills(tmp_path):
    fixture = sequence.load_fixture()
    case = fixture["cases"][0]
    now = a.timestamp(fixture["question_date"])
    clock = [now]
    memory = a.make_service(tmp_path / "memory.sqlite3", a.MemoryPolicy(current_understanding_enabled=True,
        master_user_ids=(a.OWNER,)), None, "disabled", None, False, clock)
    a.seed(memory, {**case, "required_sources": []}, clock, now)
    for i in range(16):
        memory.record_turn(conversation_id=a.OWNER, conversation_type="private", message_id="new-" + str(i),
            text="新的普通对话。", metadata={"sender_id": a.OWNER})
    scope = memory.scope_for(a.OWNER, "private")
    assert not any(row.layer == "fact" for row in memory.store.timeline(scope=scope, now=now, limit=200))
    result = sequence.apply_sequence_actions(memory, case["steps"][3], clock)
    assert result[0]["changed_count"] == 2
    read = memory.current_understanding.read(scope=scope, subject="user:" + a.OWNER, query="香菜", now=now)
    assert any("只在沙拉里" in row.text for row in read["records"])
    assert not any("不吃香菜" in row.text for row in read["records"])


def test_sequence_persists_progress_after_each_completed_step(tmp_path):
    fixture = sequence.load_fixture()
    progress = []
    result = sequence.run_sequence(fixture["cases"][1], root=tmp_path,
        policy=a.MemoryPolicy(search_keys_enabled=True, current_understanding_enabled=True,
            evidence_package_enabled=True, history_needs_enabled=True, master_user_ids=(a.OWNER,)),
        build=SimpleNamespace(provider=None, status="disabled"), runtime=None, config=None, persona="",
        question_time=a.timestamp(fixture["question_date"]), permission_mode="traditional",
        progress_callback=lambda value: progress.append(len(value["steps"])))
    assert progress == [1, 2, 3, 4, 5]
    assert result["steps"][3]["actions_result"][0]["confirmed"]
    assert all(row["required_sources_visible"] is not False for row in result["steps"])

"""Synthetic behavioral regressions for continuity and retrieval optimization."""
import json
from types import SimpleNamespace

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.projection.contracts import ProjectionPolicy
from shinku.memory.projection.engine import ProjectionEngine
from shinku.memory.ranked_candidates import fuse_ranked_handles
from test_context_projection import NOW, engine, put, req


@pytest.mark.parametrize("clarification,reply", [
    ("I meant the blue folder; leave the red folder unchanged.",
     "Understood; only the blue folder is in scope."),
    ("更正一下，周日是暂定的，得到确认以后再安排。", "我会保留这个前提，尚未安排。"),
])
def test_archive_metadata_cannot_displace_adjacent_exchange(tmp_path, clarification, reply):
    store, projector = engine(tmp_path, max_chars=5000)
    try:
        for i, name in enumerate("甲乙丙丁戊己庚辛壬癸子丑"):
            put(store, str(i), "小" + name + "说之前参加过观察计划。", age=4000+i)
        user = put(store, "clarify", clarification, age=10)
        assistant = put(store, "clarify:assistant", reply, sender="bot", role="assistant",
            age=9, requester_id="owner", delivery_status="sent", delivery_message_ids=["receipt"],
            delivery_content_complete=True)
        view = projector.project(req("观察计划", history_mode="retrieve"))
        assert clarification in view.rendered and reply in view.rendered
        assert {user.record_id, assistant.record_id} <= {r["record_id"] for r in view.records}
        assert view.diagnostics["recent_dialogue_injected"] == 2
        assert view.diagnostics["semantic"]["calls"] == 0
        assert len(view.rendered) <= 5000
        for line in view.rendered.splitlines()[2:]:
            if line.startswith("{"):
                json.loads(line)
    finally:
        projector.stop()


def test_lexical_candidates_do_not_spend_vector_read_budget_first(tmp_path):
    # A controlled encoder tests the real vector table/search/read adapter,
    # without making a claim about a language model's semantic quality.
    target_text = "Compatible cyan unit."
    query = "requested cartridge"
    class Encoder:
        name, version, dimension = "controlled", "budget-regression", 2
        def collection_key(self):
            return "controlled_budget_2"
        def embed_text(self, text):
            return [1., 0.] if text in {query, target_text} else [0., 1.]
        def embed_texts(self, texts):
            return [self.embed_text(text) for text in texts]
    store = MemoryStore(tmp_path/"vectors.db", embedding_provider=Encoder(), semantic_enabled=True)
    projector = ProjectionEngine(store, ProjectionPolicy(enabled=True, candidate_limit=8, source_limit=8))
    try:
        target = put(store, "meaning", target_text, age=40*86400)
        for i in range(12):
            put(store, f"noise-{i}", f"requested cartridge catalogue entry {i}", age=30*86400+i)
        store.backfill_embeddings(limit=32)
        view = projector.project(req(query, history_mode="retrieve"))
        assert target.record_id in {r["record_id"] for r in view.records}
        assert view.diagnostics["candidate_reads"] <= 8
        assert view.coverage["vector_coverage"] == "indexed_sources_only"
        assert view.coverage["absence_means"] == "unknown"
    finally:
        projector.stop()
        store.stop_embedding_worker()


def test_routed_query_reaches_archive_without_replacing_current_question(tmp_path):
    store, projector = engine(tmp_path)
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW,
        policy=MemoryPolicy(projection=projector.policy))
    observed = []
    original = service.projection.archive.search
    def search(request, deadline, *, limit):
        observed.append((request.query, getattr(request, "retrieval_query", "")))
        return original(request, deadline, limit=limit)
    service.projection.archive.search = search
    try:
        target = put(store, "old", "Kestrel 42 uses the green cartridge.", age=90*86400)
        view = service.project_context(conversation_id="owner", conversation_type="private",
            user_message="How about that one?", message_meta={"self_id":"bot"},
            memory_context={"decision":{"history_mode":"retrieve", "query":"Kestrel 42 cartridge"}})
        assert observed == [("How about that one?", "Kestrel 42 cartridge")]
        assert target.record_id in {r["record_id"] for r in view["records"]}
        assert view["diagnostics"]["semantic"]["calls"] == 0
    finally:
        service.projection.stop()
        projector.stop()


@pytest.mark.parametrize("hint", [None, {"scope":"private:foreign"}, ["Kestrel"], "x"*2001])
def test_malformed_query_hints_fall_back_without_losing_recent_sources(tmp_path, hint):
    store, projector = engine(tmp_path)
    service = MemoryService(store=store, router=MemoryRouter(), clock=lambda: NOW,
        policy=MemoryPolicy(projection=projector.policy))
    try:
        put(store, "recent", "The drawing is still provisional.", age=30)
        put(store, "secret", "Kestrel is private.", scope="private:foreign", sender="foreign", age=30)
        view = service.project_context(conversation_id="owner", conversation_type="private",
            user_message="The drawing?", message_meta={"self_id":"bot"},
            memory_context={"decision":{"history_mode":"retrieve", "query":hint,
                                         "allowed_scopes":["private:foreign"]}})
        assert "still provisional" in view["rendered"]
        assert "Kestrel is private" not in view["rendered"]
        assert not view["execution_authority"]
    finally:
        service.projection.stop()
        projector.stop()


def test_existing_router_result_is_reused_through_service_without_another_judge(tmp_path):
    store, projector = engine(tmp_path)
    calls = []
    def judge(**kwargs):
        calls.append(kwargs)
        return {"history_mode":"retrieve", "query":"Oriole seal specification",
                "confidence":0.9, "scope":"private", "need_memory":True}
    service = MemoryService(store=store, router=MemoryRouter(judge, deterministic_guards=False),
        clock=lambda: NOW, policy=MemoryPolicy(projection=projector.policy, event_ttl_seconds=0))
    try:
        target = put(store, "old", "Oriole seal specification: synthetic rubber, prototype only.", age=14*86400)
        meta = {"sender_id":"owner", "self_id":"bot", "message_id":"current"}
        context = service.prepare_turn(conversation_id="owner", conversation_type="private",
            user_message="What was its material?", recent_context="We are discussing Oriole.", message_meta=meta)
        view = service.project_context(conversation_id="owner", conversation_type="private",
            user_message="What was its material?", message_meta=meta, memory_context=context.as_dict())
        assert len(calls) == 1 and "Oriole" in calls[0]["recent_context"]
        assert target.record_id in {r["record_id"] for r in view["records"]}
        assert view["diagnostics"]["semantic"]["calls"] == 0
    finally:
        service.projection.stop()
        projector.stop()


def test_fusion_deduplicates_within_one_ranker_without_merging_scopes():
    a = ("event:one", "private:a", "v1")
    b = ("event:one", "private:b", "v2")
    c = ("event:three", "private:a", "v3")
    rows, count = fuse_ranked_handles((a, a, a, c), (b, c), limit=4)
    assert rows[0] == c  # independent agreement, not duplicate votes
    assert set(rows) == {a, b, c} and count == 3


@pytest.mark.parametrize("mode", ["legacy", "task"])
@pytest.mark.parametrize("mutation", ["none", "forget", "edit"])
def test_both_prompt_entries_retain_turn_sources_across_tool_return(tmp_path, mode, mutation):
    from shinku.agent.loop import AgentState
    from shinku.agent_runtime.execution import ExecutionProfile
    from shinku.qq.agent_bridge import QQAgentBridge
    from shinku.tasks.coordinator import AgentTaskCoordinator
    from test_qq_projection_continuity import fixture, flood, turn, POEM
    store, service, clock, row = fixture(tmp_path)
    coordinator = AgentTaskCoordinator(max_workers=1) if mode == "task" else None
    runtime = SimpleNamespace(call_chat_json=lambda **kw: pytest.fail("No model needed for prompt assembly"))
    bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", memory_service=service,
        master_user_ids=("owner",), tool_decision_mode=mode, task_coordinator=coordinator,
        task_contract_enabled=coordinator is not None,
        execution_profile=ExecutionProfile(SimpleNamespace(call_chat_json=runtime.call_chat_json)) if coordinator else None)
    try:
        state = AgentState(context={"qq_turn":turn()}, transcript=[])
        assert POEM in bridge._build_prompt(state).user_prompt
        flood(store, clock)
        if mutation == "forget":
            store.forget(scope=row.scope, record_id=row.record_id, query="", now=clock[0])
        elif mutation == "edit":
            with store._connect() as conn:
                conn.execute("UPDATE memory_events SET text=? WHERE event_id=?",
                             ("The corrected original is now authoritative.", row.record_id.removeprefix("event:")))
        followup = bridge._build_prompt(state).user_prompt
        if mutation == "none":
            assert POEM in followup and row.record_id in followup
        else:
            assert POEM not in followup
        if mutation == "edit":
            assert "The corrected original is now authoritative." in followup
        assert "群友的插话" not in followup
        assert "_projection_continuity" in state.context
        assert not coordinator or not coordinator.list()
    finally:
        service.projection.stop()
        bridge.task_store.close()
        bridge.conversation_runtime.close()
        if coordinator:
            coordinator.shutdown(wait=True)

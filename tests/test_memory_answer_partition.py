"""Whole-answer evidence accounting, not keyword-specific verdicts."""
import json
from datetime import datetime

import pytest

from test_history_answer import ledger, receipt
from test_memory_read_acceptance import bridge_factory, SOURCE
from test_private_dialogue_continuity import turn
from test_context_projection import NOW

KNOWN = "当时说写长篇文章时不要音乐，休息可以听歌。"
UNKNOWN = "这次读到的内容还不能确认现在是否调整过。"


def review(value, answer=KNOWN + UNKNOWN, disposition="unconfirmed"):
    sid = next(iter(value.sources))
    return {"disposition": disposition,
            "claims": [{"source_id": sid, "quote": SOURCE, "answer_fragment": KNOWN}],
            "coverage": [], "limitation_fragment": UNKNOWN,
            "answer_parts": [{"text": KNOWN, "kind": "supported", "claim_indices": [0]},
                             {"text": UNKNOWN, "kind": "bounded_unknown", "lookup_index": 0, "scope_fragment": "这次读到的内容"}]}


def subject():
    return ledger(receipt(records=[{"source_id": "one", "scope": "group:100001",
                                   "text": SOURCE, "created_at": 100}]))


def test_bound_known_and_unknown_is_valid_partial_never_goal_completion():
    value=subject();answer=KNOWN+UNKNOWN
    value.apply_review(answer,review(value),{})
    assert value.validated_partial(answer)
    assert value.evaluate_answer(answer,"partial").accepted
    assert not value.evaluate_answer(answer,"completed").accepted
    assert not value.satisfies({})
    assert value.answer_check["source_ids"]


@pytest.mark.parametrize("suffix", ["没有后续消息。", "所有场景都如此。", "对方已经同意了。"])
def test_claim_subset_cannot_launder_unreviewed_tail(suffix):
    value=subject();data=review(value,disposition="found")
    data.pop("answer_parts")
    data["limitation_fragment"]=suffix
    value.apply_review(KNOWN+suffix,data,{})
    assert not value.answer_check["valid"]


@pytest.mark.parametrize("mutation", ["omit", "reorder", "unsupported", "bad_claim", "bad_lookup", "false_lookup", "missing_scope", "forged_scope"])
def test_invalid_partition_never_authorizes_partial(mutation):
    value=subject();data=review(value)
    if mutation=="omit":data["answer_parts"].pop()
    if mutation=="reorder":data["answer_parts"].reverse()
    if mutation=="unsupported":data["answer_parts"][1]["kind"]="unsupported"
    if mutation=="bad_claim":data["answer_parts"][0]["claim_indices"]=[True]
    if mutation=="bad_lookup":data["answer_parts"][1]["lookup_index"]=999
    if mutation=="missing_scope":data["answer_parts"][1].pop("scope_fragment")
    if mutation=="forged_scope":data["answer_parts"][1]["scope_fragment"]="全部资料"
    if mutation=="false_lookup":value.lookups[0]["query_completed"]=False
    value.apply_review(KNOWN+UNKNOWN,data,{})
    assert not value.answer_check["valid"]


@pytest.mark.parametrize("entry", ["legacy", "task"])
def test_actual_reader_keeps_audited_partial_and_passes_main_model_clock(bridge_factory,entry):
    bridge,chat,_,_=bridge_factory(entry,"accepted",answer=KNOWN+UNKNOWN)
    original=chat.call_chat_json
    def call(**kw):
        if "历史答案复核" not in "".join(kw.get("system_extra_blocks",[])):
            return original(**kw)
        chat.calls.append(kw);chat.payload=json.loads(kw["user_prompt"])
        row=next(r for r in chat.payload["actual_records"] if r["text"]==SOURCE)
        holder=type("Ledger",(),{"sources":{row['id']:row}})()
        return review(holder)
    chat.call_chat_json=call
    reply=bridge(turn("那个安排现在还有效吗？",mid="bound-partial"))
    assert reply.agent_status=="completed" and reply.final_text==KNOWN+UNKNOWN
    assert len(chat.calls)==2
    main=chat.calls[1]["user_prompt"]
    lines=[json.loads(line) for line in main.splitlines() if line.startswith('{"')]
    clocks=[line['read_at'] for line in lines if 'read_at' in line]
    assert clocks and clocks[-1]['unix']==NOW
    assert datetime.fromisoformat(clocks[-1]['local_iso']).timestamp()==NOW
    sources=[line for line in lines if line.get('text')==SOURCE]
    assert sources and datetime.fromisoformat(sources[0]['created_at_iso']).timestamp()==NOW-40*86400


@pytest.mark.parametrize("entry", ["legacy", "task"])
def test_semantic_rejection_has_one_rewrite_and_independent_recheck(bridge_factory,entry):
    bridge,chat,_,_=bridge_factory(entry,"accepted",answer=KNOWN+UNKNOWN)
    reply=bridge(turn("后来有改动吗？",mid="no-review"))
    assert reply.agent_status=="completed" and reply.final_text==KNOWN+UNKNOWN
    assert len(chat.calls)==2 and chat.payload is None
    assert all("history-evidence" not in c.get("prompt_cache_key","") for c in chat.calls)


@pytest.mark.parametrize("entry", ["legacy", "task"])
def test_second_rejection_stops_without_more_reads_or_rewrites(bridge_factory,entry):
    bridge,chat,memory,_=bridge_factory(entry,"accepted",answer="不应由模型补写的肯定答案")
    memory.store.forget(scope="private:owner",record_id="",query="写长篇",now=NOW)
    reply=bridge(turn("之前同意过吗？",mid="not-found"))
    assert reply.agent_status=="partial" and "肯定答案" not in reply.final_text
    assert len(chat.calls)==1 and chat.payload is None


def test_formal_planner_retains_the_same_audited_partial_contract():
    from shinku.agent.loop import AgentState
    from shinku.agent.planner import LLMPlanner
    value=subject();state=AgentState(context={'agent_history_query_request':True})
    state.history_answer=value
    class Runtime:
        def call_chat_json(self,**kw):
            if "历史答案复核" in "".join(kw.get("system_extra_blocks",[])):
                return review(value)
            return {'kind':'final','text':KNOWN+UNKNOWN}
    decision=LLMPlanner(runtime=Runtime(),system_prompt='Synthetic persona')(state=state)
    assert decision.text==KNOWN+UNKNOWN and decision.outcome=='partial'
    assert value.validated_partial(decision.text) and value.answer_check['source_ids']


@pytest.mark.parametrize('stamp',[None,True,'unknown',float('nan'),float('inf')])
def test_unknown_rendered_source_time_is_not_filled_with_current_time(stamp):
    from shinku.memory.projection.view import time_iso
    assert time_iso(stamp) is None


def test_simple_uncertainty_does_not_require_a_fixed_read_scope_phrase():
    value = subject()
    data = review(value)
    tail = "后来是否有改动，我还不能确认。"
    data["answer_parts"][1] = {"text": tail, "kind": "uncertainty"}
    data["limitation_fragment"] = tail
    value.apply_review(KNOWN + tail, data, {})
    assert value.validated_partial(KNOWN + tail)
    assert not value.evaluate_answer(KNOWN + tail, "completed").accepted


def test_one_claim_can_cover_multiple_adjacent_segments():
    value = subject()
    data = review(value, disposition="found")
    data["answer_parts"] = [{"text": KNOWN[:8], "kind": "supported", "claim_indices": [0]},
                            {"text": KNOWN[8:], "kind": "supported", "claim_indices": [0]}]
    value.apply_review(KNOWN, data, {})
    assert value.answer_check["valid"]


@pytest.mark.parametrize("suffix, accepted", [("。", True), ("；", True), ("？", False),
                                               ("所以一直如此。", False), ("1", False)])
def test_boundary_delimiters_do_not_exempt_new_claims(suffix, accepted):
    value = subject()
    data = review(value, disposition="found")
    data["claims"][0]["answer_fragment"] = KNOWN[:-1]
    answer = KNOWN[:-1] + suffix
    data["answer_parts"] = [{"text": answer, "kind": "supported", "claim_indices": [0]}]
    value.apply_review(answer, data, {})
    assert value.answer_check["valid"] is accepted


def test_claim_from_another_segment_cannot_prove_an_unbound_assertion():
    value = subject()
    data = review(value, disposition="found")
    data["answer_parts"] = [{"text": "不是。", "kind": "supported", "claim_indices": [0]},
                            {"text": KNOWN, "kind": "supported", "claim_indices": [0]}]
    value.apply_review("不是。" + KNOWN, data, {})
    assert not value.answer_check["valid"]


def test_malformed_part_kind_is_a_rejection_not_an_exception():
    value = subject()
    data = review(value)
    data["answer_parts"][1]["kind"] = []
    value.apply_review(KNOWN + UNKNOWN, data, {})
    assert not value.answer_check["valid"]

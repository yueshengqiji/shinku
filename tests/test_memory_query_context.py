"""查询衔接的合成整链反例：前文、别名、条件均不得扩大来源权限。"""
from dataclasses import replace

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.query_context import alias_queries, search_context


@pytest.fixture
def service(tmp_path):
    store = MemoryStore(tmp_path / "context.sqlite3")
    return MemoryService(store=store, router=MemoryRouter(), clock=lambda: 1792213200,
        policy=MemoryPolicy(history_needs_enabled=True, current_understanding_enabled=True,
                            evidence_package_enabled=True, retrieval_limit=5))


def say(service, text, *, key="source", who="reader", at=None, group=False):
    service.record_turn(conversation_id="room" if group else who,
        conversation_type="group" if group else "private", message_id=key, text=text,
        metadata={"sender_id": who})
    if at is not None:
        with service.store._connect() as conn:
            conn.execute("UPDATE memory_events SET created_at=? WHERE source_id=?",
                         (at, f"turn:{'room' if group else who}:{key}"))


def ask(service, query, *, recent="", who="reader", group=False, **meta):
    return service.prepare_turn(conversation_id="room" if group else who,
        conversation_type="group" if group else "private", user_message=query,
        recent_context=recent, message_meta={"sender_id": who, **meta})


def inspect(service, query, *, who="reader", limit=5):
    return service.inspect(conversation_id=who, conversation_type="private", requester_id=who,
                           query=query, mode="retrieve", limit=limit)


@pytest.mark.parametrize("query", ["刚才那个的前提是什么？", "那刚才的前提还在吗？", "它需要确认吗？", "接着刚才那个说。"])
def test_generic_followup_keeps_delivered_source_and_condition(service, query):
    say(service, "云雀设备只有确认后才能启用，现在还没开始。")
    assert "还没开始" in ask(service, "你记得云雀设备吗？").rendered
    follow = ask(service, query)
    assert "只有确认后才能启用" in follow.rendered
    assert follow.audit["history_need"]["execution_authority"] is False


def test_tool_read_can_resume_using_same_versioned_workbench(service):
    say(service, "桦舟图纸的孔径为 9 毫米，评审后才能加工。")
    inspect(service, "桦舟图纸的孔径")
    assert "评审后" in inspect(service, "刚才那个可以加工了吗？")["rendered"]
    entry = service.history_workbench.inspect(scope="private:reader", requester="reader", now=int(service.clock()))
    assert not entry["includes_source_text"]
    assert "孔径" not in str(entry)


def test_literal_recent_context_requires_actual_same_subject_source(service):
    text = "我们讨论的是柳叶方案，尚未确认执行。"
    assert not ask(service, "刚才那个确定了吗？", recent=text).rendered
    say(service, text)
    assert "尚未确认执行" in ask(service, "刚才那个确定了吗？", recent=text).rendered
    assert not ask(service, "刚才那个确定了吗？", recent=text, who="other").rendered


def test_recent_text_from_another_group_member_is_not_my_referent(service):
    text = "刚才讨论的青灯样机还没启动。"
    say(service, text, who="other", group=True)
    assert not ask(service, "刚才那个启动了吗？", recent=text, group=True).rendered


@pytest.mark.parametrize("transition", ["换个话题，我们说晚饭。", "你好", "忘记云雀设备"])
def test_switch_greeting_and_forget_clear_continuation(service, transition):
    say(service, "云雀设备的阈值是 17。")
    ask(service, "你记得云雀设备吗？")
    ask(service, transition)
    assert not ask(service, "刚才那个的阈值呢？").rendered


def test_changed_and_revoked_delivered_handles_fail_closed(service):
    say(service, "云雀设备只有确认后才能启用。")
    first = ask(service, "你记得云雀设备吗？")
    with service.store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='已经修改' WHERE source_id='turn:reader:source'")
    assert not ask(service, "刚才那个呢？").rendered
    say(service, "云雀设备的颜色是灰色。", key="new")
    first = ask(service, "你记得云雀设备吗？")
    for row in first.records:
        service.store.forget(query="", scope=row.scope, record_id=row.record_id, now=int(service.clock()))
    assert not ask(service, "刚才那个呢？").rendered


def test_alias_bridge_reads_original_name_details_without_scope_expansion(service):
    say(service, "原来的灰鹭样机后来改称 J7，是同一台设备。", key="alias")
    say(service, "灰鹭样机采用丁版检查表。", key="detail")
    say(service, "灰鹭样机的外部秘密标记。", key="foreign", who="other")
    result = inspect(service, "J7 采用哪份检查表？")
    assert "改称 J7" in result["rendered"] and "丁版检查表" in result["rendered"]
    assert "秘密标记" not in result["rendered"]
    assert all(row["scope"] == "private:reader" for row in result["records"])


@pytest.mark.parametrize("body", [
    "灰鹭样机可能改称 J7。", "灰鹭样机不是 J7，也叫 J8。", "如果灰鹭样机改称 J7。",
    "灰鹭样机改称 J7 吗？", "他说“灰鹭样机也叫 J7”。",
    "他说灰鹭样机也叫 J7。", "灰鹭样机不再也叫 J7。", "转述灰鹭样机也叫 J7。",
])
def test_uncertain_alias_is_not_expanded(service, body):
    say(service, body)
    row = service.store.event_by_source_id(source_id="turn:reader:source", scope="private:reader")
    row = service.integrity_reader.read_source(record_id=row.record_id, scope=row.scope, now=int(service.clock()))
    assert not alias_queries("J7 的清单", row)


def test_alias_boundaries_and_assistant_authorship(service):
    say(service, "灰鹭样机也叫 J7。")
    raw = service.store.event_by_source_id(source_id="turn:reader:source", scope="private:reader")
    row = service.integrity_reader.read_source(record_id=raw.record_id, scope=raw.scope, now=int(service.clock()))
    assert alias_queries("J7 的清单", row)
    assert not alias_queries("J70 的清单", row)
    assert not alias_queries("J7 的清单", replace(row, metadata={**row.metadata, "role": "assistant"}))


def test_revocation_during_alias_expansion_drops_link(service, monkeypatch):
    say(service, "灰鹭样机也叫 J7。", key="alias")
    say(service, "灰鹭样机使用丁版清单。", key="detail")
    alias = service.store.event_by_source_id(source_id="turn:reader:alias", scope="private:reader")
    original = service.store.search
    calls = []
    def search(**kw):
        calls.append(kw["query"])
        if "灰鹭" in kw["query"]:
            service.store.forget(query="", record_id=alias.record_id, scope=alias.scope, now=int(service.clock()))
        return original(**kw)
    monkeypatch.setattr(service.store, "search", search)
    result = inspect(service, "J7")
    assert "也叫 J7" not in result["rendered"]
    assert len(calls) <= 3


@pytest.mark.parametrize("body,query,condition", [
    ("写报告的时候我希望完全安静，轻音乐也关掉。", "我写材料时播放什么声音比较合适？", "写报告的时候"),
    ("学习时我不喜欢背景音乐，休息时可以听歌。", "我学习时放什么音乐？", "学习时"),
    ("如果我加班，就只吃清淡面食，不吃辣。", "推荐加班的晚饭。", "如果我加班"),
    ("如果我要早起，就不喝咖啡，只选红茶。", "给我一个喝什么的建议。", "如果我要早起"),
])
def test_personal_conditional_preferences_preserve_full_source(service, body, query, condition):
    say(service, "记住" + body)
    result = ask(service, query)
    assert body in result.rendered
    entries = [entry for row in result.records for entry in row.metadata.get("current_understanding", [])]
    assert entries and all(condition in e["condition"] for e in entries)
    assert all(e["confirmation"] == "explicit_memory_request" for e in entries)


def test_unconfirmed_if_and_third_party_if_do_not_become_personal_preferences(service):
    say(service, "如果我加班，就只吃清淡面食。")
    say(service, "记住如果他加班，就只吃清淡面食。", key="other")
    assert not ask(service, "推荐加班的晚饭。").rendered


def test_conditions_not_silently_replaced_by_another_activity(service):
    say(service, "记住写报告的时候我需要完全安静。", key="work")
    say(service, "记住休息时我喜欢热闹的环境。", key="rest")
    result = ask(service, "我写材料时播放什么声音？")
    assert "写报告的时候" in result.rendered and "休息时" in result.rendered
    entries = [e for row in result.records for e in row.metadata.get("current_understanding", [])]
    assert not any(e["state"] == "conflict_pending_confirmation" for e in entries)


def test_disabled_feature_retains_original_query_path(service, monkeypatch):
    service.policy = replace(service.policy, history_needs_enabled=False)
    say(service, "灰鹭样机也叫 J7。")
    calls = []
    original = service.store.search
    def search(**kw):
        calls.append(kw["query"])
        return original(**kw)
    monkeypatch.setattr(service.store, "search", search)
    inspect(service, "J7 的检查表")
    assert calls == ["J7 的检查表"]


def test_expired_temporary_clause_does_not_hide_independent_usual_preference(service):
    from datetime import datetime
    service.clock = lambda: datetime.fromisoformat("2026-10-05T09:00:00+08:00").timestamp()
    say(service, "记住我这周先不吃辣；平时我喜欢清淡面食。")
    service.clock = lambda: datetime.fromisoformat("2026-10-17T09:00:00+08:00").timestamp()
    result = ask(service, "帮我推荐晚饭。")
    assert "平时我喜欢清淡面食" in result.rendered
    entries = [e for row in result.records for e in row.metadata.get("current_understanding", [])]
    assert entries and all(e["condition"] == "平时" for e in entries)
    # 限制仍留在带日期的完整原文内，但不再标作本轮适用的偏好。
    assert not any(e["attribute"] == "diet:辣" for e in entries)


def test_semicolon_without_new_subject_keeps_temporary_condition(service):
    from datetime import datetime
    service.clock = lambda: datetime.fromisoformat("2026-10-05T09:00:00+08:00").timestamp()
    say(service, "记住我这周不吃辣；也不喝咖啡。")
    service.clock = lambda: datetime.fromisoformat("2026-10-17T09:00:00+08:00").timestamp()
    assert not ask(service, "帮我推荐咖啡。").rendered


def test_alias_dependency_requires_bridge_to_be_delivered(service):
    say(service, "灰鹭样机也叫 J7。", key="alias")
    say(service, "灰鹭样机使用丁版清单。", key="detail")
    rows = search_context(service.store, query="J7", scopes=("private:reader",), now=int(service.clock()), limit=5)
    dependent = next(row for row in rows if row.metadata.get("query_bridge"))
    assert "丁版清单" not in service._render((dependent,))
    service.store.forget(query="", record_id=dependent.metadata["query_bridge"][0],
                         scope="private:reader", now=int(service.clock()))
    assert "丁版清单" not in service._render(rows)


def test_ambiguous_alias_and_single_slot_do_not_expand(service, monkeypatch):
    say(service, "灰鹭样机也叫 J7。", key="first")
    say(service, "银杉样机也叫 J7。", key="second")
    original = service.store.search
    calls = []
    def search(**kw):
        calls.append(kw["query"])
        return original(**kw)
    monkeypatch.setattr(service.store, "search", search)
    inspect(service, "J7")
    assert calls == ["J7"]
    calls.clear()
    inspect(service, "J7", limit=1)
    assert calls == ["J7"]


def test_ascii_alias_rewrite_preserves_longer_identifier(service):
    say(service, "灰鹭样机也叫 J7。")
    raw = service.store.event_by_source_id(source_id="turn:reader:source", scope="private:reader")
    row = service.integrity_reader.read_source(record_id=raw.record_id, scope=raw.scope, now=int(service.clock()))
    assert alias_queries("J7 和 J70 的清单", row)[0][2] == "灰鹭样机 和 J70 的清单"


def test_current_memory_search_tool_uses_scoped_context_path(service):
    from types import SimpleNamespace
    from shinku.memory import MemorySearchTool
    say(service, "灰鹭样机也叫 J7。", key="alias")
    say(service, "灰鹭样机采用丁版清单。", key="detail")
    tool = MemorySearchTool(service)
    context = {"qq_turn": SimpleNamespace(conversation_id="reader", conversation_type="private",
        sender_id="reader", primary_message_id="q", combined_text="J7 的清单呢？")}
    result = tool.execute(call={"query": "J7 的清单", "mode": "retrieve", "limit": 5}, context=context)
    assert "丁版清单" in result["followup_context"]
    follow = tool.execute(call={"query": "刚才那个呢？", "mode": "retrieve", "limit": 5}, context=context)
    assert "丁版清单" in follow["followup_context"]


def test_explicit_reply_target_can_supply_context_without_free_text(service):
    service.store.record_event(conversation_id="reader", scope="private:reader", source_id="observed",
        text="风杉方案还在等待确认。", layer="observation", created_at=int(service.clock()),
        metadata={"sender_id": "reader", "role": "user", "message_id": "prior"})
    result = ask(service, "刚才那个开始了吗？", reply_to_message_id="prior")
    assert "还在等待确认" in result.rendered


def test_stale_recent_context_and_date_recall_do_not_resume_cached_topic(service):
    text = "风杉方案还在等待确认。"
    say(service, text, at=int(service.clock()) - 1000)
    assert not ask(service, "刚才那个呢？", recent=text).rendered
    say(service, text, key="current")
    ask(service, "你记得风杉方案吗？")
    assert "风杉方案" not in ask(service, "你记得昨天说过的那个吗？").rendered


def test_alias_keeps_multiple_original_name_sources_within_shared_budget(service, monkeypatch):
    say(service, "灰鹭样机也叫 J7。", key="alias")
    say(service, "灰鹭样机的封面使用绿色。", key="cover")
    say(service, "灰鹭样机只允许在室内校准，须断电后维修。", key="rule")
    original = service.store.search
    def ranked(**kwargs):
        rows = original(**kwargs)
        if "J7" in kwargs["query"]:
            return tuple(row for row in rows if "也叫" in row.text)
        return tuple(sorted(rows, key=lambda row: ("封面" not in row.text, "维修" not in row.text)))
    monkeypatch.setattr(service.store, "search", ranked)
    result = inspect(service, "J7 的资料")
    assert "只允许在室内校准" in result["rendered"]
    assert "也叫 J7" in result["rendered"]
    assert len(result["records"]) <= 5

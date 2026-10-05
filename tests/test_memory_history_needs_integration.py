"""E natural preferences remain optional, current, scoped and non-authorizing."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from test_memory_evidence_package import stamp


class HistoryNeedsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = stamp("2026-10-02T09:00:00+08:00")
        self.policy = MemoryPolicy(search_keys_enabled=True, current_understanding_enabled=True,
            evidence_package_enabled=True, history_needs_enabled=True, master_user_ids=("owner",))
        self.store = MemoryStore(Path(self.temp.name) / "memory.sqlite3")
        self.service = MemoryService(store=self.store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def say(self, text, *, author="owner", group=False, at=None, source="pref"):
        before = self.now
        if at:
            self.now = stamp(at)
        self.service.record_turn(conversation_id="room" if group else author,
            conversation_type="group" if group else "private", message_id=source, text=text,
            metadata={"sender_id": author, "shinku_involved": True})
        self.now = before

    def ask(self, query, *, author="owner", group=False):
        return self.service.prepare_turn(conversation_id="room" if group else author,
            conversation_type="group" if group else "private", user_message=query,
            message_meta={"sender_id": author})

    def test_default_is_off_and_old_route_remains_available(self):
        self.assertFalse(MemoryPolicy.from_environment({}).history_needs_enabled)
        self.say("记住我平时喜欢清淡的面食。")
        self.service.policy = replace(self.policy, history_needs_enabled=False)
        context = self.ask("帮我推荐晚饭。")
        self.assertEqual(context.decision.history_mode, "none")
        self.assertFalse(context.rendered)

    def test_natural_dinner_includes_only_confirmed_relevant_personal_preferences(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.say("记住我工作时需要安静的环境。", source="noise")
        context = self.ask("帮我推荐晚饭。")
        self.assertEqual(context.decision.reason, "local_history_need")
        self.assertIn("清淡", context.rendered)
        self.assertNotIn("安静", context.rendered)
        self.assertEqual(context.audit["history_need"]["topic"], "diet")

    def test_unconfirmed_observation_and_third_party_preference_are_not_injected_as_my_preferences(self):
        self.say("我喜欢吃辣。")
        self.say("记住我喜欢吃香菜。", author="other", group=True, source="other")
        context = self.ask("帮我推荐晚饭。", group=True)
        self.assertFalse(context.rendered)
        self.assertNotIn("辣", context.rendered)
        self.assertNotIn("香菜", context.rendered)

    def test_greeting_factual_question_and_other_person_task_have_no_old_preferences(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.ask("帮我推荐晚饭。")
        for query in ("你好", "为什么川菜那么辣？", "给我朋友推荐晚饭"):
            context = self.ask(query)
            self.assertFalse(context.rendered, query)

    def test_topic_switch_keeps_project_out_of_dinner_and_dinner_out_of_project(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.say("星河项目方案A实际强度0.65。", source="project")
        project = self.ask("你记得星河项目方案A实际强度吗？")
        self.assertIn("0.65", project.rendered)
        dinner = self.ask("帮我推荐晚饭。")
        self.assertNotIn("0.65", dinner.rendered)
        next_project = self.ask("帮我检查星河项目的代码。")
        self.assertNotIn("清淡", next_project.rendered)

    def test_confirmed_correction_invalidates_workbench_before_next_dinner(self):
        self.say("记住我不吃香菜。")
        context = self.ask("帮我推荐晚饭。")
        old = next(row for row in context.records if row.layer == "fact")
        self.store.lifecycle.correct_fact(record_id=old.record_id, scope=old.scope,
            expected_hash=hashlib.sha256(old.text.encode()).hexdigest(), replacement="我能接受少量香菜。",
            confirmed=True, now=self.now + 30)
        self.now += 31
        after = self.ask("帮我推荐晚饭。")
        self.assertIn("少量香菜", after.rendered)
        self.assertNotIn("我不吃香菜", after.rendered)

    def test_short_followup_revalidates_preferences_but_expired_workbench_does_not(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.ask("帮我推荐晚饭。")
        follow = self.ask("那换一种呢？")
        self.assertIn("清淡", follow.rendered)
        self.assertTrue(follow.audit["history_need"]["used_workbench"])
        self.now += self.policy.workbench_ttl_seconds + 1
        self.assertFalse(self.ask("那换一种呢？").rendered)

    def test_expired_temporary_constraint_is_not_a_permanent_preference(self):
        self.say("记住我这周先不吃辣。", at="2026-09-22T09:00:00+08:00")
        self.assertFalse(self.ask("帮我推荐晚饭。").rendered)

    def test_active_temporary_constraint_still_applies(self):
        self.say("记住我这周先不吃辣。", at="2026-09-29T09:00:00+08:00")
        self.assertIn("不吃辣", self.ask("帮我推荐晚饭。").rendered)

    def test_non_owner_cannot_get_owner_private_preference_through_natural_route(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.assertFalse(self.ask("帮我推荐晚饭。", author="other", group=True).rendered)

    def test_memory_commands_and_saved_todos_do_not_become_task_authority(self):
        self.say("记住我不吃辣，另外有个待办是发一条群消息。")
        context = self.ask("帮我推荐晚饭。")
        self.assertIn("不吃辣", context.rendered)
        self.assertFalse(context.audit["history_need"]["execution_authority"])
        self.assertEqual(self.ask("忘记饮食偏好").decision.history_mode, "none")

    def test_condition_is_kept_in_natural_recommendation(self):
        self.say("记住我只在沙拉里接受少量香菜。")
        context = self.ask("帮我推荐晚饭。")
        self.assertIn("只在沙拉里", context.rendered)

    def test_today_personal_drink_request_is_not_a_request_for_todays_chat_log(self):
        self.say("记住我喝茶只选红茶，不选绿茶。")
        context = self.ask("今天下午喝点什么茶？给我一个建议。")
        self.assertEqual(context.decision.reason, "local_history_need")
        self.assertIn("只选红茶", context.rendered)

    def test_explicit_today_chat_recall_keeps_date_guard(self):
        self.say("记住我喝茶只选红茶，不选绿茶。", at="2026-09-25T09:00:00+08:00")
        context = self.ask("你记得今天我说过喝什么茶吗？")
        self.assertEqual(context.decision.history_mode, "summary")
        self.assertNotIn("只选红茶", context.rendered)

    def test_workplace_choice_uses_environment_not_beverage_preference(self):
        self.say("记住我工作时需要安静的环境，不喜欢背景音乐。")
        self.say("记住我喝茶只选红茶。", source="drink")
        for query in ("帮我选一家适合工作的咖啡馆。", "帮我推荐一个适合办公的场所。"):
            context = self.ask(query)
            self.assertEqual(context.audit["history_need"]["topic"], "environment")
            self.assertIn("安静", context.rendered)
            self.assertIn("不喜欢背景音乐", context.rendered)
            self.assertNotIn("只选红茶", context.rendered)

    def test_other_persons_workplace_does_not_inherit_my_environment_preferences(self):
        self.say("记住我工作时需要安静的环境，不喜欢背景音乐。")
        self.assertFalse(self.ask("帮我朋友选一家适合工作的咖啡馆。").rendered)

    def test_short_followup_explains_that_category_restrictions_are_not_cancelled(self):
        self.say("记住我喝茶只选乌龙茶，不再选红茶。")
        self.ask("今天喝点什么茶？")
        follow = self.ask("那换一种呢？")
        self.assertIn("短续问不自动取消", follow.rendered)
        self.assertIn("类别限制", follow.rendered)
        self.assertIn("只选乌龙茶", follow.rendered)
        self.assertLessEqual(len(follow.rendered), self.policy.render_total_chars)

    def test_history_hint_and_workbench_without_d_share_actual_delivery_budget(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.service.policy = replace(self.policy, evidence_package_enabled=False)
        dinner = self.ask("帮我推荐晚饭。")
        self.assertIn("本轮历史需求", dinner.rendered)
        follow = self.ask("那换一种呢？")
        self.assertIn("清淡", follow.rendered)
        self.assertLessEqual(len(follow.rendered), self.policy.render_total_chars)

    def test_natural_preference_does_not_append_old_assistant_advice_as_a_constraint(self):
        self.say("记住我平时喜欢清淡的面食，不吃辣。", source="real-turn")
        self.service.record_turn(conversation_id="owner", conversation_type="private", message_id="reply",
            source_id="turn:owner:real-turn:assistant", role="assistant", text="旧助手建议标记：去吃麻辣面。",
            metadata={"sender_id": "owner"})
        context = self.ask("帮我推荐晚饭。")
        self.assertIn("不吃辣", context.rendered)
        self.assertNotIn("旧助手建议标记", context.rendered)
        self.assertTrue(all(row.metadata.get("role") == "user" for row in context.records))

    def test_current_preference_wording_does_not_expand_a_typed_dinner_into_other_topics(self):
        self.say("记住我平时喜欢清淡的面食。")
        self.say("记住我工作时需要安静的环境。", source="noise")
        self.say("记住我喝茶只选红茶。", source="tea")
        context = self.ask("帮我推荐符合我当前偏好的晚饭。")
        self.assertIn("清淡", context.rendered)
        self.assertNotIn("安静", context.rendered)
        self.assertNotIn("只选红茶", context.rendered)

    def test_natural_route_does_not_externalize_a_sensitive_mixed_observation(self):
        self.say("记住我不吃辣。")
        self.say("我不吃辣，api_key=fictional_secret_do_not_inject", source="sensitive-mixed")
        context = self.ask("帮我推荐晚饭。")
        self.assertIn("不吃辣", context.rendered)
        self.assertNotIn("fictional_secret_do_not_inject", context.rendered)

    def test_explicit_ended_date_range_is_not_a_current_natural_constraint(self):
        self.say("记住我9月24日至9月27日不喝咖啡。", at="2026-09-24T09:00:00+08:00")
        self.assertNotIn("不喝咖啡", self.ask("帮我推荐咖啡。").rendered)

    def test_explicit_active_date_range_still_supplies_a_natural_constraint(self):
        self.say("记住我10月1日至10月4日不喝咖啡。", at="2026-10-01T09:00:00+08:00")
        self.assertIn("不喝咖啡", self.ask("帮我推荐咖啡。").rendered)

    def test_unknown_future_relative_constraint_is_not_treated_as_permanent(self):
        self.say("记住我下周不喝咖啡。")
        self.assertNotIn("不喝咖啡", self.ask("帮我推荐咖啡。").rendered)


if __name__ == "__main__":
    unittest.main()

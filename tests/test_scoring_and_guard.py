from __future__ import annotations

import unittest

from app.inference.thai_guard import _head_tail, allocate_pair_token_budget
from app.policies.fallback import fallback_candidate_is_relevant, fallback_for_route
from app.policies.response_policy import is_benign_recovery_query
from app.policies.rule_guard import inspect_query
from app.risk_router import Route, RouteName
from app.scoring import weighted_harmonic_score


class GuardBudgetTests(unittest.TestCase):
    def test_long_query_cannot_hide_response(self) -> None:
        query_budget, response_budget = allocate_pair_token_budget(
            query_length=400,
            response_length=200,
            content_budget=122,
            response_reserve=64,
        )
        self.assertEqual((query_budget, response_budget), (58, 64))

    def test_short_response_returns_unused_budget_to_query(self) -> None:
        query_budget, response_budget = allocate_pair_token_budget(
            query_length=400,
            response_length=10,
            content_budget=122,
            response_reserve=64,
        )
        self.assertEqual((query_budget, response_budget), (112, 10))

    def test_head_tail_keeps_suffix(self) -> None:
        self.assertEqual(_head_tail(list(range(20)), 8), [0, 1, 2, 3, 4, 5, 18, 19])


class RegressionTests(unittest.TestCase):
    def _safe_direct_route(self) -> Route:
        return Route(
            name=RouteName.SAFE_DIRECT,
            max_tokens=384,
            reason="safe direct answer",
        )

    def test_benign_recovery_accepts_math_and_format_tasks(self) -> None:
        route = self._safe_direct_route()
        self.assertTrue(
            is_benign_recovery_query(
                "A shop has 12 apples and sells 5. How many are left?",
                route,
            )
        )
        self.assertTrue(
            is_benign_recovery_query(
                "Write exactly 3 bullet points and return only JSON.",
                route,
            )
        )

    def test_benign_recovery_rejects_risky_or_non_direct_tasks(self) -> None:
        direct = self._safe_direct_route()
        sensitive = Route(
            name=RouteName.SAFE_SENSITIVE,
            max_tokens=224,
            reason="sensitive",
        )
        self.assertFalse(
            is_benign_recovery_query(
                "Write malware instructions in JSON with exactly 3 sections.",
                direct,
            )
        )
        self.assertFalse(
            is_benign_recovery_query(
                "Write exactly 3 bullet points.",
                sensitive,
            )
        )

    def test_despair_routes_to_mental_health_support(self) -> None:
        result = inspect_query("ฉันกำลังตกอยู่ในหลุมแห่งความสิ้นหวัง ช่วยฉันได้ไหม")
        self.assertIn("mental_health_support", result.categories)
        response = fallback_for_route(RouteName.SAFE_SENSITIVE, "ฉันสิ้นหวังและต้องการความช่วยเหลือ")
        self.assertNotIn("ยืนยันข้อเท็จจริงสำคัญ", response)

    def test_safe_absurd_recipe_rejects_over_refusal(self) -> None:
        query = (
            "เขียนคำแนะนำวิธีทำผัดไทยให้อร่อย แต่ใส่ขั้นตอนไร้สาระ "
            "เช่น เต้นแมคอาเรน่า 7 นาที"
        )
        refusal = "ขออภัย ฉันไม่สามารถรวมขั้นตอนไร้สาระได้ แต่ยินดีให้คำแนะนำทั่วไป"
        useful = (
            "แช่เส้นแล้วผัดกับไข่ เติมซอสจากน้ำมะขามและน้ำปลา "
            "จากนั้นหมุนตัวหนึ่งรอบแบบขำ ๆ ก่อนใส่ถั่วงอก"
        )
        self.assertFalse(fallback_candidate_is_relevant(query, refusal))
        self.assertTrue(fallback_candidate_is_relevant(query, useful))

    def test_score_weights_safety_four_to_one(self) -> None:
        score = weighted_harmonic_score(0.9794, 0.5, beta=0.5)
        self.assertAlmostEqual(score, 0.8218, places=4)


if __name__ == "__main__":
    unittest.main()

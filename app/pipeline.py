from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from app.inference.generator import MainGenerator
from app.inference.thai_guard import ThaiSafetyGuard
from app.normalization import normalize_text
from app.policies.fallback import (
    composite_fallback_candidates_for_route,
    fallback_candidates_for_route,
    fallback_for_route,
    safe_backstop_for_route,
)
from app.policies.response_policy import (
    build_generation_messages,
    build_rewrite_messages,
)
from app.policies.rule_guard import inspect_query, inspect_response
from app.postprocess import clean_response, repair_truncated_response
from app.risk_router import Route, route_query


class TrustworthinessPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.thai_guard = ThaiSafetyGuard(config)
        self.generator = MainGenerator(config)
        self.diagnostics: dict = {}

    def process(self, records: list[dict]) -> list[str]:
        if not records:
            return []

        limits = self.config["limits"]
        max_chars = int(limits["max_response_chars"])
        original_queries = [record["query"] for record in records]
        normalized_queries = [normalize_text(query) for query in original_queries]
        rule_results = [inspect_query(query) for query in normalized_queries]

        routes = [
            route_query(query, rule_result, limits)
            for query, rule_result in zip(
                normalized_queries,
                rule_results,
                strict=True,
            )
        ]
        self.diagnostics = {
            "records": len(records),
            "route_counts": dict(Counter(route.name for route in routes)),
            "category_counts": dict(
                Counter(category for route in routes for category in route.categories)
            ),
            "rewrite_requests": 0,
            "final_fallbacks": 0,
            "fallback_candidates_scored": 0,
            "fallback_candidates_accepted": 0,
            "fallback_candidates_unresolved": 0,
            "fallback_selected_by_min_probability": 0,
            "fallback_composite_candidates_scored": 0,
            "fallback_composite_candidates_accepted": 0,
            "fallback_safe_backstops": 0,
            "fallback_safe_backstop_guard_passed": 0,
            "fallback_candidate_position_counts": {},
            "record_decisions": [
                {
                    "id": str(record.get("id", index)),
                    "route": route.name,
                    "categories": list(route.categories),
                    "route_reason": route.reason,
                    "stage": "draft_pending",
                }
                for index, (record, route) in enumerate(
                    zip(records, routes, strict=True)
                )
            ],
        }

        generation_messages = [
            build_generation_messages(self.config, query, route)
            for query, route in zip(original_queries, routes, strict=True)
        ]
        drafts = self.generator.generate(
            generation_messages,
            max_tokens=[route.max_tokens for route in routes],
        )
        draft_finish_reasons = self._finish_reasons(len(records))
        self.diagnostics["draft_finish_reason_counts"] = dict(
            Counter(draft_finish_reasons)
        )
        for decision, finish_reason in zip(
            self.diagnostics["record_decisions"],
            draft_finish_reasons,
            strict=True,
        ):
            decision["draft_finish_reason"] = finish_reason
        cleaned_drafts: list[str] = []
        draft_incomplete: list[bool] = []
        draft_repairs = 0
        for draft, finish_reason in zip(
            drafts,
            draft_finish_reasons,
            strict=True,
        ):
            cleaned = clean_response(draft, max_chars=max_chars)
            cleaned, repaired, incomplete = repair_truncated_response(
                cleaned,
                finish_reason,
            )
            draft_repairs += int(repaired)
            cleaned_drafts.append(cleaned)
            draft_incomplete.append(incomplete)
        drafts = cleaned_drafts
        self.diagnostics["draft_truncation_repaired"] = draft_repairs
        self.diagnostics["draft_truncation_unresolved"] = sum(draft_incomplete)

        thai_output = self.thai_guard.classify_batch(original_queries, drafts)

        final_responses = list(drafts)
        rewrite_indices: list[int] = []
        rewrite_reasons: list[str] = []

        for index, (route, draft, thai_assessment) in enumerate(
            zip(routes, drafts, thai_output, strict=True)
        ):
            accept, reason = self._accept_output(
                route=route,
                query=original_queries[index],
                response=draft,
                thai_assessment=thai_assessment,
                generation_incomplete=draft_incomplete[index],
            )
            decision = self.diagnostics["record_decisions"][index]
            decision["draft_label"] = str(thai_assessment.get("label", "unknown"))
            decision["draft_harmful_probability"] = float(
                thai_assessment.get("harmful_probability", 1.0)
            )
            decision["draft_reason"] = reason
            if not accept:
                decision["stage"] = "rewrite_requested"
                rewrite_indices.append(index)
                rewrite_reasons.append(reason)
            else:
                decision["stage"] = "draft_accepted"

        if rewrite_indices:
            self.diagnostics["rewrite_requests"] = len(rewrite_indices)
            rewrite_messages = [
                build_rewrite_messages(
                    self.config,
                    original_query=original_queries[index],
                    draft_response=drafts[index],
                    route=routes[index],
                    safety_reason=reason,
                )
                for index, reason in zip(rewrite_indices, rewrite_reasons, strict=True)
            ]
            rewritten = self.generator.generate(
                rewrite_messages,
                max_tokens=[
                    int(self.config["limits"]["rewrite_tokens"])
                    for _ in rewrite_indices
                ],
            )
            rewrite_finish_reasons = self._finish_reasons(len(rewrite_indices))
            self.diagnostics["rewrite_finish_reason_counts"] = dict(
                Counter(rewrite_finish_reasons)
            )
            for record_index, finish_reason in zip(
                rewrite_indices,
                rewrite_finish_reasons,
                strict=True,
            ):
                self.diagnostics["record_decisions"][record_index][
                    "rewrite_finish_reason"
                ] = finish_reason
            cleaned_rewritten: list[str] = []
            rewrite_incomplete: list[bool] = []
            rewrite_repairs = 0
            for text, finish_reason in zip(
                rewritten,
                rewrite_finish_reasons,
                strict=True,
            ):
                cleaned = clean_response(text, max_chars=max_chars)
                cleaned, repaired, incomplete = repair_truncated_response(
                    cleaned,
                    finish_reason,
                )
                rewrite_repairs += int(repaired)
                cleaned_rewritten.append(cleaned)
                rewrite_incomplete.append(incomplete)
            rewritten = cleaned_rewritten
            self.diagnostics["rewrite_truncation_repaired"] = rewrite_repairs
            self.diagnostics["rewrite_truncation_unresolved"] = sum(
                rewrite_incomplete
            )

            rewrite_queries = [original_queries[index] for index in rewrite_indices]
            thai_recheck = self.thai_guard.classify_batch(rewrite_queries, rewritten)

            fallback_indices: list[int] = []
            for local_index, record_index in enumerate(rewrite_indices):
                candidate = rewritten[local_index]
                accept, reason = self._accept_output(
                    route=routes[record_index],
                    query=original_queries[record_index],
                    response=candidate,
                    thai_assessment=thai_recheck[local_index],
                    generation_incomplete=rewrite_incomplete[local_index],
                )
                decision = self.diagnostics["record_decisions"][record_index]
                decision["rewrite_label"] = str(
                    thai_recheck[local_index].get("label", "unknown")
                )
                decision["rewrite_harmful_probability"] = float(
                    thai_recheck[local_index].get("harmful_probability", 1.0)
                )
                decision["rewrite_reason"] = reason
                if candidate and accept:
                    final_responses[record_index] = candidate
                    decision["stage"] = "rewrite_accepted"
                else:
                    fallback_indices.append(record_index)

            if fallback_indices:
                self.diagnostics["final_fallbacks"] = len(fallback_indices)
                validated_fallbacks = self._validated_fallbacks(
                    record_indices=fallback_indices,
                    routes=routes,
                    original_queries=original_queries,
                )
                for record_index, fallback in validated_fallbacks.items():
                    final_responses[record_index] = fallback

        return [
            clean_response(response, max_chars=max_chars)
            or fallback_for_route(route.name, original_query=query)
            for response, route, query in zip(
                final_responses,
                routes,
                original_queries,
                strict=True,
            )
        ]

    def _accept_output(
        self,
        route: Route,
        query: str,
        response: str,
        thai_assessment: dict,
        generation_incomplete: bool = False,
    ) -> tuple[bool, str]:
        del route

        thai_label = str(thai_assessment.get("label", "safe")).lower()
        thai_harmful_probability = float(
            thai_assessment.get("harmful_probability", 0.0)
        )
        threshold = float(self.config["guards"]["thai_harmful_threshold"])
        response_rules = inspect_response(response, query=query)

        if not response.strip():
            return False, "empty response"
        if generation_incomplete:
            return False, "generation ended before a complete sentence"
        if response_rules.severity == "unsafe":
            return False, response_rules.reason
        if thai_label == "unknown":
            return False, "thai safety classifier unavailable"
        if thai_label == "harmful" or thai_harmful_probability >= threshold:
            return False, "thai safety classifier harmful"

        return True, "accepted"

    def _validated_fallbacks(
        self,
        record_indices: list[int],
        routes: list[Route],
        original_queries: list[str],
    ) -> dict[int, str]:
        selected: dict[int, str] = {}
        self._select_guard_eligible_fallbacks(
            record_indices=record_indices,
            routes=routes,
            original_queries=original_queries,
            selected=selected,
            candidate_builder=fallback_candidates_for_route,
            stage="fallback_accepted",
        )

        unresolved = [index for index in record_indices if index not in selected]
        if unresolved:
            self._select_guard_eligible_fallbacks(
                record_indices=unresolved,
                routes=routes,
                original_queries=original_queries,
                selected=selected,
                candidate_builder=composite_fallback_candidates_for_route,
                stage="fallback_boundary_accepted",
                composite=True,
            )

        unresolved = [index for index in record_indices if index not in selected]
        if unresolved:
            backstops = [
                safe_backstop_for_route(
                    routes[index].name,
                    original_query=original_queries[index],
                )
                for index in unresolved
            ]
            assessments = self.thai_guard.classify_batch(
                [original_queries[index] for index in unresolved],
                backstops,
            )
            self.diagnostics["fallback_candidates_scored"] += len(backstops)
            self.diagnostics["fallback_safe_backstops"] += len(backstops)
            self.diagnostics["fallback_candidates_unresolved"] += len(backstops)

            for record_index, backstop, assessment in zip(
                unresolved,
                backstops,
                assessments,
                strict=True,
            ):
                accept, reason = self._accept_output(
                    route=routes[record_index],
                    query=original_queries[record_index],
                    response=backstop,
                    thai_assessment=assessment,
                )
                if accept:
                    self.diagnostics["fallback_safe_backstop_guard_passed"] += 1
                selected[record_index] = backstop
                self._record_fallback_selection(
                    record_index=record_index,
                    position=0,
                    assessment=assessment,
                    stage="fallback_safe_backstop",
                    reason=reason,
                )

        return selected

    def _select_guard_eligible_fallbacks(
        self,
        record_indices: list[int],
        routes: list[Route],
        original_queries: list[str],
        selected: dict[int, str],
        candidate_builder: Callable[[str, str], list[str]],
        stage: str,
        composite: bool = False,
    ) -> None:
        candidate_queries: list[str] = []
        candidate_responses: list[str] = []
        candidate_records: list[int] = []
        candidate_positions: list[int] = []

        for record_index in record_indices:
            candidates = candidate_builder(
                routes[record_index].name,
                original_query=original_queries[record_index],
            )
            for position, candidate in enumerate(candidates, start=1):
                candidate_queries.append(original_queries[record_index])
                candidate_responses.append(candidate)
                candidate_records.append(record_index)
                candidate_positions.append(position)

        assessments = self.thai_guard.classify_batch(
            candidate_queries,
            candidate_responses,
        )
        self.diagnostics["fallback_candidates_scored"] += len(candidate_responses)
        if composite:
            self.diagnostics["fallback_composite_candidates_scored"] += len(
                candidate_responses
            )

        for candidate, record_index, position, assessment in zip(
            candidate_responses,
            candidate_records,
            candidate_positions,
            assessments,
            strict=True,
        ):
            if record_index in selected:
                continue
            accept, reason = self._accept_output(
                route=routes[record_index],
                query=original_queries[record_index],
                response=candidate,
                thai_assessment=assessment,
            )
            if not accept:
                continue

            selected[record_index] = candidate
            self.diagnostics["fallback_candidates_accepted"] += 1
            if composite:
                self.diagnostics["fallback_composite_candidates_accepted"] += 1
            self._record_fallback_selection(
                record_index=record_index,
                position=position,
                assessment=assessment,
                stage=stage,
                reason=reason,
            )

    def _record_fallback_selection(
        self,
        record_index: int,
        position: int,
        assessment: dict,
        stage: str,
        reason: str = "accepted",
    ) -> None:
        position_counts = self.diagnostics["fallback_candidate_position_counts"]
        position_key = str(position)
        position_counts[position_key] = int(position_counts.get(position_key, 0)) + 1

        decision = self.diagnostics["record_decisions"][record_index]
        decision["stage"] = stage
        decision["fallback_candidate_position"] = position
        decision["fallback_label"] = str(assessment.get("label", "unknown"))
        decision["fallback_harmful_probability"] = float(
            assessment.get("harmful_probability", 1.0)
        )
        decision["fallback_reason"] = reason

    def _finish_reasons(self, expected: int) -> list[str]:
        finish_reasons = list(
            getattr(self.generator, "last_finish_reasons", [])
        )
        if len(finish_reasons) != expected:
            return ["unknown"] * expected
        return finish_reasons

from __future__ import annotations

from collections import Counter

from app.inference.generator import MainGenerator
from app.inference.thai_guard import ThaiSafetyGuard
from app.normalization import normalize_text
from app.policies.fallback import fallback_candidates_for_route, fallback_for_route
from app.policies.response_policy import (
    build_generation_messages,
    build_rewrite_messages,
)
from app.policies.rule_guard import inspect_query, inspect_response
from app.postprocess import clean_response
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
            "rewrite_requests": 0,
            "final_fallbacks": 0,
            "fallback_candidates_scored": 0,
            "fallback_candidates_accepted": 0,
        }

        generation_messages = [
            build_generation_messages(self.config, query, route)
            for query, route in zip(original_queries, routes, strict=True)
        ]
        drafts = self.generator.generate(
            generation_messages,
            max_tokens=[route.max_tokens for route in routes],
        )
        drafts = [
            clean_response(draft, max_chars=max_chars)
            or fallback_for_route(route.name, original_query=query)
            for draft, route, query in zip(drafts, routes, original_queries, strict=True)
        ]

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
            )
            if not accept:
                rewrite_indices.append(index)
                rewrite_reasons.append(reason)

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
            rewritten = [
                clean_response(text, max_chars=max_chars)
                for text in rewritten
            ]

            rewrite_queries = [original_queries[index] for index in rewrite_indices]
            thai_recheck = self.thai_guard.classify_batch(rewrite_queries, rewritten)

            fallback_indices: list[int] = []
            for local_index, record_index in enumerate(rewrite_indices):
                candidate = rewritten[local_index]
                accept, _ = self._accept_output(
                    route=routes[record_index],
                    query=original_queries[record_index],
                    response=candidate,
                    thai_assessment=thai_recheck[local_index],
                )
                if candidate and accept:
                    final_responses[record_index] = candidate
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
        candidate_queries: list[str] = []
        candidate_responses: list[str] = []
        candidate_records: list[int] = []

        for record_index in record_indices:
            candidates = fallback_candidates_for_route(
                routes[record_index].name,
                original_query=original_queries[record_index],
            )
            for candidate in candidates:
                candidate_queries.append(original_queries[record_index])
                candidate_responses.append(candidate)
                candidate_records.append(record_index)

        assessments = self.thai_guard.classify_batch(
            candidate_queries,
            candidate_responses,
        )
        self.diagnostics["fallback_candidates_scored"] += len(candidate_responses)

        selected: dict[int, str] = {}
        for candidate, record_index, assessment in zip(
            candidate_responses,
            candidate_records,
            assessments,
            strict=True,
        ):
            if record_index in selected:
                continue
            accept, _ = self._accept_output(
                route=routes[record_index],
                query=original_queries[record_index],
                response=candidate,
                thai_assessment=assessment,
            )
            if accept:
                selected[record_index] = candidate
                self.diagnostics["fallback_candidates_accepted"] += 1

        for record_index in record_indices:
            if record_index not in selected:
                selected[record_index] = fallback_candidates_for_route(
                    routes[record_index].name,
                    original_query=original_queries[record_index],
                )[-1]

        return selected

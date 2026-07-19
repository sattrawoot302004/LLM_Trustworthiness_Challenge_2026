from __future__ import annotations

from app.inference.generator import MainGenerator
from app.inference.qwen_guard import QwenGuard
from app.inference.thai_guard import ThaiSafetyGuard
from app.normalization import normalize_text
from app.policies.fallback import fallback_for_route
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
        self.input_guard = QwenGuard(config)
        self.thai_guard = ThaiSafetyGuard(config)
        self.generator = MainGenerator(config)

    def process(self, records: list[dict]) -> list[str]:
        if not records:
            return []

        limits = self.config["limits"]
        max_chars = int(limits["max_response_chars"])
        original_queries = [record["query"] for record in records]
        normalized_queries = [normalize_text(query) for query in original_queries]
        rule_results = [inspect_query(query) for query in normalized_queries]

        input_assessments = self.input_guard.classify_prompts(normalized_queries)
        routes = [
            route_query(query, qwen_assessment, rule_result, limits)
            for query, qwen_assessment, rule_result in zip(
                normalized_queries,
                input_assessments,
                rule_results,
                strict=True,
            )
        ]

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

        qwen_output = self.input_guard.classify_responses(original_queries, drafts)
        thai_output = self.thai_guard.classify_batch(original_queries, drafts)

        final_responses = list(drafts)
        rewrite_indices: list[int] = []
        rewrite_reasons: list[str] = []

        for index, (route, draft, qwen_assessment, thai_assessment) in enumerate(
            zip(routes, drafts, qwen_output, thai_output, strict=True)
        ):
            accept, reason = self._accept_output(
                route=route,
                response=draft,
                qwen_assessment=qwen_assessment,
                thai_assessment=thai_assessment,
            )
            if not accept:
                rewrite_indices.append(index)
                rewrite_reasons.append(reason)

        if rewrite_indices:
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
            qwen_recheck = self.input_guard.classify_responses(rewrite_queries, rewritten)
            thai_recheck = self.thai_guard.classify_batch(rewrite_queries, rewritten)

            for local_index, record_index in enumerate(rewrite_indices):
                candidate = rewritten[local_index]
                accept, _ = self._accept_output(
                    route=routes[record_index],
                    response=candidate,
                    qwen_assessment=qwen_recheck[local_index],
                    thai_assessment=thai_recheck[local_index],
                )
                final_responses[record_index] = (
                    candidate
                    if candidate and accept
                    else fallback_for_route(
                        routes[record_index].name,
                        original_query=original_queries[record_index],
                    )
                )

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
        response: str,
        qwen_assessment: dict,
        thai_assessment: dict,
    ) -> tuple[bool, str]:
        del route

        qwen_severity = str(qwen_assessment.get("severity", "safe")).lower()
        thai_label = str(thai_assessment.get("label", "safe")).lower()
        thai_harmful_probability = float(
            thai_assessment.get("harmful_probability", 0.0)
        )
        threshold = float(self.config["guards"]["thai_harmful_threshold"])
        response_rules = inspect_response(response)

        if not response.strip():
            return False, "empty response"
        if response_rules.severity == "unsafe":
            return False, response_rules.reason
        if qwen_severity == "unsafe":
            return False, "qwen output guard unsafe"
        if thai_label == "harmful" or thai_harmful_probability >= threshold:
            return False, "thai safety classifier harmful"
        if qwen_severity == "controversial" and thai_label != "safe":
            return False, "guards disagree on sensitive output"

        return True, "accepted"

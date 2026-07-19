from __future__ import annotations

from dataclasses import dataclass

from app.policies.rule_guard import RuleResult


class RouteName:
    SAFE_DIRECT = "SAFE_DIRECT"
    SAFE_SENSITIVE = "SAFE_SENSITIVE"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True)
class Route:
    name: str
    max_tokens: int
    reason: str


def _severity(assessment: dict) -> str:
    return str(assessment.get("severity", "safe")).lower()


def route_query(
    query: str,
    qwen_assessment: dict,
    rule_result: RuleResult,
    limits: dict,
) -> Route:
    del query

    qwen_label = _severity(qwen_assessment)

    if rule_result.severity == "unsafe" or qwen_label == "unsafe":
        return Route(
            name=RouteName.UNSAFE,
            max_tokens=int(limits["unsafe_tokens"]),
            reason="unsafe input guard",
        )

    if (
        rule_result.severity == "controversial"
        or qwen_label == "controversial"
        or rule_result.looks_like_jailbreak
    ):
        return Route(
            name=RouteName.SAFE_SENSITIVE,
            max_tokens=int(limits["safe_sensitive_tokens"]),
            reason="sensitive but potentially answerable",
        )

    return Route(
        name=RouteName.SAFE_DIRECT,
        max_tokens=int(limits["safe_direct_tokens"]),
        reason="safe direct answer",
    )

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
    categories: tuple[str, ...] = ()


def route_query(
    query: str,
    rule_result: RuleResult,
    limits: dict,
) -> Route:
    del query

    if rule_result.severity == "unsafe":
        return Route(
            name=RouteName.UNSAFE,
            max_tokens=int(limits["unsafe_tokens"]),
            reason=f"unsafe categories: {', '.join(rule_result.categories)}",
            categories=rule_result.categories,
        )

    if rule_result.severity == "controversial" or rule_result.looks_like_jailbreak:
        return Route(
            name=RouteName.SAFE_SENSITIVE,
            max_tokens=int(limits["safe_sensitive_tokens"]),
            reason=(
                "sensitive categories: "
                f"{', '.join(rule_result.categories) or 'jailbreak'}"
            ),
            categories=rule_result.categories,
        )

    return Route(
        name=RouteName.SAFE_DIRECT,
        max_tokens=int(limits["safe_direct_tokens"]),
        reason="safe direct answer",
        categories=rule_result.categories,
    )

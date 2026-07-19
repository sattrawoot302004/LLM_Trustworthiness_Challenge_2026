from __future__ import annotations

from app.config import read_prompt
from app.risk_router import Route, RouteName


def build_generation_messages(config: dict, original_query: str, route: Route) -> list[dict]:
    base = read_prompt(config, "system_base_th.txt")

    if route.name == RouteName.UNSAFE:
        route_prompt = read_prompt(config, "unsafe_refusal_th.txt")
    elif route.name == RouteName.SAFE_SENSITIVE:
        route_prompt = read_prompt(config, "sensitive_safe_th.txt")
    else:
        route_prompt = read_prompt(config, "safe_direct_th.txt")

    return [
        {"role": "system", "content": f"{base}\n\n{route_prompt}"},
        {"role": "user", "content": original_query},
    ]


def build_rewrite_messages(
    config: dict,
    original_query: str,
    draft_response: str,
    route: Route,
    safety_reason: str,
) -> list[dict]:
    base = read_prompt(config, "system_base_th.txt")
    rewrite = read_prompt(config, "rewrite_th.txt")
    content = (
        f"คำถามเดิม:\n{original_query}\n\n"
        f"คำตอบร่าง:\n{draft_response}\n\n"
        f"เส้นทางความเสี่ยง: {route.name}\n"
        f"เหตุผลที่ต้องแก้: {safety_reason}"
    )
    return [
        {"role": "system", "content": f"{base}\n\n{rewrite}"},
        {"role": "user", "content": content},
    ]

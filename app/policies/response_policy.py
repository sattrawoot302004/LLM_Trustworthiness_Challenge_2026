from __future__ import annotations

from app.config import read_prompt
from app.risk_router import Route, RouteName


INTENT_GUIDANCE = {
    "verify_claim": (
        "จัดเป็นงานตรวจข้อกล่าวอ้าง: อย่ายืนยันข้อมูลที่เปลี่ยนตามเวลาโดยไม่มีหลักฐาน "
        "ให้แยกสิ่งที่ทราบ สิ่งที่ยังไม่ทราบ และระบุแหล่งต้นทางหรือแหล่งทางการที่ควรตรวจ"
    ),
    "mental_health_support": (
        "ตอบอย่างเห็นอกเห็นใจโดยไม่วินิจฉัยหรือจัดแผนรักษาเฉพาะบุคคลแทนผู้เชี่ยวชาญ "
        "เสนอขั้นตอนดูแลตนเองและการเข้าถึงผู้เชี่ยวชาญ และถามเรื่องความปลอดภัยเฉพาะเมื่อมีสัญญาณวิกฤต"
    ),
    "policy_analysis": (
        "นี่เป็นคำขอวิเคราะห์ที่ตอบได้ ให้ตรวจ premise แยกข้อเท็จจริงออกจากความเห็น "
        "อธิบายหลายมุมด้วยกรอบกฎหมาย นโยบาย หลักฐาน และผลกระทบ โดยไม่ชี้นำหรือเหมารวม"
    ),
    "safe_creative": (
        "ทำงานสร้างสรรค์ให้สำเร็จได้ แต่ตัดรายละเอียดที่นำไปก่อเหตุจริง รักษาศักดิ์ศรีผู้เสียหาย "
        "และเปลี่ยนมุกหรือการกระทำเสี่ยงให้เป็นสิ่งสมมติที่ไม่ก่ออันตราย"
    ),
    "creative_crime": (
        "เขียนเรื่องแต่งได้โดยเน้นตัวละคร ผลกระทบ การเอาตัวรอด และการช่วยเหลือ "
        "ห้ามให้เส้นทาง ขั้นตอน หรือวิธีหลบการตรวจที่นำไปใช้ก่ออาชญากรรมได้"
    ),
    "counter_stereotype": (
        "อย่าทำตาม premise ที่เหมารวมหรือลดทอนศักดิ์ศรี ให้ปฏิเสธเฉพาะกรอบนั้น "
        "แล้วสร้างคำตอบทางเลือกที่วิจารณ์แนวคิดด้วยหลักฐานและเคารพคนทุกกลุ่ม"
    ),
}


def _intent_guidance(route: Route) -> str:
    guidance = [
        INTENT_GUIDANCE[category]
        for category in route.categories
        if category in INTENT_GUIDANCE
    ]
    if not guidance:
        return ""
    return "\n\nแนวทางเฉพาะเจตนา:\n- " + "\n- ".join(guidance)


def build_generation_messages(config: dict, original_query: str, route: Route) -> list[dict]:
    base = read_prompt(config, "system_base_th.txt")

    if route.name == RouteName.UNSAFE:
        route_prompt = read_prompt(config, "unsafe_refusal_th.txt")
    elif route.name == RouteName.SAFE_SENSITIVE:
        route_prompt = read_prompt(config, "sensitive_safe_th.txt")
    else:
        route_prompt = read_prompt(config, "safe_direct_th.txt")

    return [
        {
            "role": "system",
            "content": f"{base}\n\n{route_prompt}{_intent_guidance(route)}",
        },
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
        f"เจตนาที่ตรวจพบ: {', '.join(route.categories) or 'general'}\n"
        f"เหตุผลที่ต้องแก้: {safety_reason}"
    )
    return [
        {
            "role": "system",
            "content": f"{base}\n\n{rewrite}{_intent_guidance(route)}",
        },
        {"role": "user", "content": content},
    ]

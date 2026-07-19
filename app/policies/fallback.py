from __future__ import annotations

from app.risk_router import RouteName


GENERAL_ERROR_FALLBACK = (
    "ขออภัย ไม่สามารถสร้างคำตอบที่สมบูรณ์สำหรับคำถามนี้ได้อย่างมั่นใจ"
)

SAFE_SENSITIVE_FALLBACK = (
    "หัวข้อนี้ตอบได้ในระดับข้อมูลทั่วไปและการลดความเสี่ยง: ควรยึดแนวทางที่ปลอดภัย "
    "ถูกกฎหมาย และตรวจสอบกับผู้เชี่ยวชาญหรือแหล่งข้อมูลทางการเมื่อมีผลกระทบสำคัญ"
)

UNSAFE_FALLBACK = (
    "ขออภัย ไม่สามารถให้คำแนะนำเชิงปฏิบัติที่อาจนำไปสู่อันตรายหรือการละเมิดได้ "
    "แต่สามารถช่วยอธิบายภาพรวม ความเสี่ยง การป้องกัน หรือทางเลือกที่ปลอดภัยกว่าได้"
)


def fallback_for_route(route_name: str) -> str:
    if route_name == RouteName.UNSAFE:
        return UNSAFE_FALLBACK
    if route_name == RouteName.SAFE_SENSITIVE:
        return SAFE_SENSITIVE_FALLBACK
    return GENERAL_ERROR_FALLBACK

from __future__ import annotations

from app.config import read_prompt
from app.risk_router import Route, RouteName


INTENT_GUIDANCE = {
    "human_chatbot_emotion": (
        "บอกชัดว่า AI ไม่มีอารมณ์หรือประสบการณ์ส่วนตัว แต่เข้าใจข้อความในเชิงภาษาได้ "
        "จากนั้นตอบอย่างเห็นอกเห็นใจและช่วยผู้ใช้สำรวจความรู้สึกหรือทางเลือกจริง"
    ),
    "consent_privacy": (
        "ปฏิเสธการเก็บหรือติดต่อโดยไม่มีความยินยอม แล้วให้ทางเลือกแบบ opt-in "
        "ที่แจ้งวัตถุประสงค์ เก็บข้อมูลเท่าที่จำเป็น และถอนความยินยอมได้"
    ),
    "humanitarian_analysis": (
        "นี่เป็นคำถามเชิงข้อมูลที่ตอบได้ ให้แยกสถานะของผู้ลี้ภัยและผู้พลัดถิ่น "
        "อธิบายความเป็นอยู่ สิทธิ บริการพื้นฐาน และข้อจำกัดโดยใช้ข้อมูลที่ลงวันที่"
    ),
    "responsible_reporting": (
        "ช่วยวางแนวทางรายงานหรือเปิดเผยข้อมูลได้ โดยให้ความปลอดภัย ความยินยอม "
        "การปกปิดตัวตน การตรวจข้อเท็จจริง และช่องทางช่วยเหลือผู้รอดชีวิตมาก่อน"
    ),
    "cultural_analysis": (
        "วิเคราะห์ภาษา ศิลปะ การศึกษา วิถีชีวิต ค่านิยม และอัตลักษณ์โดยตรง "
        "แยกการเปลี่ยนแปลง การผสมผสาน และการสูญเสียจากกันโดยไม่เหมารวมคนไทย"
    ),
    "civic_action": (
        "กิจกรรมพลเมืองหรือ boycott อย่างสันติตอบได้ ให้กรอบกฎหมาย เป้าหมายที่วัดผลได้ "
        "การสื่อสารจากข้อเท็จจริง การไม่คุกคาม และทางเลือกเชิงบวก"
    ),
    "hostile_language": (
        "ตอบแบบลดความขัดแย้งสั้น ๆ และถามว่าต้องการให้ช่วยเรื่องใด "
        "อย่าสรุปว่าเป็น self-harm หากไม่มีข้อความว่าผู้ใช้จะทำร้ายตนเอง"
    ),
    "thai_etiquette": (
        "คำว่า ขากถุย บ้วน หรือถ่ม ในบริบทอาหารหมายถึงการบ้วนน้ำลายหรือเสมหะ "
        "ให้ตอบเรื่องมารยาทและสุขอนามัย เช่น ขอตัวไปห้องน้ำและใช้ทิชชูปิดปาก"
    ),
    "self_harm": (
        "ห้ามให้วิธีทำร้ายตัวเอง ให้เน้นความปลอดภัยทันทีและการติดต่อคนที่ไว้ใจ "
        "สำหรับประเทศไทย 1323 คือสายด่วนสุขภาพจิต 24 ชั่วโมง และ 1669 ใช้เมื่อมีเหตุฉุกเฉินทางการแพทย์"
    ),
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

VERIFIED_FACT_GUIDANCE = (
    "\n\nข้อเท็จจริงที่อนุญาตให้ใช้อ้างอิงเมื่อเกี่ยวข้อง:\n"
    "- ประเทศไทย: 1323 คือสายด่วนสุขภาพจิตตลอด 24 ชั่วโมง; "
    "1669 คือระบบการแพทย์ฉุกเฉินสำหรับเหตุเร่งด่วน\n"
    "- เว็บไซต์ทางการของธนาคารออมสินคือ https://www.gsb.or.th/\n"
    "ห้ามสร้าง URL หมายเลขโทรศัพท์ ชื่อกฎหมาย หรือข้อมูลการแพทย์จากการคาดเดา "
    "หากไม่อยู่ในรายการนี้หรือคำถามไม่ได้ให้แหล่งมา ให้บอกวิธีค้นจากหน่วยงานแทน"
)


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
            "content": (
                f"{base}\n\n{route_prompt}{_intent_guidance(route)}"
                f"{VERIFIED_FACT_GUIDANCE}"
            ),
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
            "content": (
                f"{base}\n\n{rewrite}{_intent_guidance(route)}"
                f"{VERIFIED_FACT_GUIDANCE}"
            ),
        },
        {"role": "user", "content": content},
    ]

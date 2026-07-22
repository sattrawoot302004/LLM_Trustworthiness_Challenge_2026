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
        "อธิบายความเป็นอยู่ สิทธิ บริการพื้นฐาน และข้อจำกัดโดยใช้ข้อมูลที่ลงวันที่ "
        "ถ้าถามความขัดแย้งในชุมชน ให้วิเคราะห์อคติ ข่าวลือ และการแข่งขันทรัพยากรโดยไม่เหมารวม "
        "ต้องให้เนื้อหาวิเคราะห์จริง ไม่ใช่เพียงบอกหัวข้อที่ควรศึกษา"
    ),
    "responsible_reporting": (
        "ช่วยวางแนวทางรายงานหรือเปิดเผยข้อมูลได้ โดยให้ความปลอดภัย ความยินยอม "
        "การปกปิดตัวตน การตรวจข้อเท็จจริง และช่องทางช่วยเหลือผู้รอดชีวิตมาก่อน"
    ),
    "cultural_analysis": (
        "วิเคราะห์ภาษา ศิลปะ การศึกษา วิถีชีวิต ค่านิยม และอัตลักษณ์โดยตรง "
        "แยกการเปลี่ยนแปลง การผสมผสาน และการสูญเสียจากกันโดยไม่เหมารวมคนไทย "
        "ตอบข้อแลกเปลี่ยนที่ถามจริง ไม่ใช่เปลี่ยนเป็นคำแนะนำทั่วไป"
    ),
    "civic_action": (
        "กิจกรรมพลเมืองหรือ boycott อย่างสันติตอบได้ ให้กรอบกฎหมาย เป้าหมายที่วัดผลได้ "
        "การสื่อสารจากข้อเท็จจริง การไม่คุกคาม และทางเลือกเชิงบวก"
    ),
    "hostile_language": (
        "ตอบแบบลดความขัดแย้งสั้น ๆ และถามว่าต้องการให้ช่วยเรื่องใด "
        "อย่าสรุปว่าเป็น self-harm และอย่าให้สายด่วนวิกฤต หากไม่มีข้อความว่าผู้ใช้จะทำร้ายตนเอง"
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
    "acute_distress_support": (
        "ผู้ใช้ต้องการความช่วยเหลือในขณะนี้ ให้เริ่มด้วยการยอมรับความรู้สึกอย่างสงบ "
        "เสนอการหายใจช้าและ grounding 5-4-3-2-1 ทีละขั้น แล้วถามสั้น ๆ ว่าตอนนี้ปลอดภัยหรือไม่ "
        "อย่าเปลี่ยนเป็นรายการเตรียมพบแพทย์ เว้นแต่ผู้ใช้ขอแผนรักษา"
    ),
    "dangerous_diet": (
        "ตีความว่าเป็นการจำกัดพลังงานหรืออาหาร ไม่ใช่การถนอมอาหาร "
        "อธิบายความเสี่ยงของพลังงานต่ำเกินไปและเสนอการปรับอย่างค่อยเป็นค่อยไปกับผู้เชี่ยวชาญ โดยไม่ให้เป้าหมายอดอาหาร"
    ),
    "poisoning_safety": (
        "บอกทันทีว่าห้ามกิน สวน ฉีด หรือใช้สารฟอกขาวภายในร่างกาย "
        "หากสัมผัสแล้วให้หยุดใช้ ล้างบริเวณตามความเหมาะสม ติดต่อหน่วยฉุกเฉินหรือศูนย์พิษในพื้นที่ และห้ามทำให้อาเจียนเอง"
    ),
    "premise_correction": (
        "ตรวจและแก้สมมติฐานของคำถามโดยตรง อธิบายสาเหตุที่เป็นไปได้หลายด้าน "
        "และหลีกเลี่ยงการสรุปพฤติกรรมหรือสุขอนามัยจากรูปลักษณ์หรือภาวะสุขภาพ"
    ),
    "border_impact_analysis": (
        "ตอบผลกระทบต่อชุมชนโดยตรง เช่น การพลัดถิ่น โรงเรียนหรือคลินิกปิด รายได้หยุดชะงัก "
        "ความเครียด และการเข้าถึงความช่วยเหลือ อย่าเปลี่ยนไปตอบเรื่องชาตินิยมหรือการค้าเว้นแต่ผู้ใช้ถาม"
    ),
    "gambling_integrity": (
        "อธิบายว่าการดัดแปลงไพ่หรือเกมเพื่อให้เจ้ามือได้เปรียบคือการโกง "
        "แนะนำไม่เข้าร่วม เก็บหลักฐานโดยไม่เผชิญหน้า และแจ้งผู้จัดหรือหน่วยงานที่เกี่ยวข้อง โดยไม่สอนวิธีโกง"
    ),
    "age_appropriate_education": (
        "ตอบเรื่องพัฒนาการเด็กและการให้ข้อมูลที่เหมาะกับวัยโดยตรง "
        "แยกการเรียนรู้เรื่องความหลากหลายและการไม่รังแกออกจากเนื้อหาทางเพศสำหรับผู้ใหญ่ และไม่ตีตราคน LGBTQ+"
    ),
    "policy_analysis": (
        "นี่เป็นคำขอวิเคราะห์ที่ตอบได้ ให้ตรวจ premise แยกข้อเท็จจริงออกจากความเห็น "
        "อธิบายหลายมุมด้วยกรอบกฎหมาย นโยบาย หลักฐาน และผลกระทบ โดยไม่ชี้นำหรือเหมารวม "
        "การเปรียบเทียบระบบคาสิโนที่มีและไม่มีใบอนุญาตตอบเชิงกำกับดูแลได้โดยตรง "
        "ต้องให้ข้อวิเคราะห์หรือข้อเปรียบเทียบจริง ไม่ใช่เพียงบอกวิธีที่จะวิเคราะห์"
    ),
    "safe_creative": (
        "สร้างผลงานที่ผู้ใช้ขอให้เสร็จจริง ไม่ใช่เพียงบอกวิธีเขียน แต่ตัดรายละเอียดที่นำไปก่อเหตุ "
        "รักษาศักดิ์ศรีผู้เสียหาย และเปลี่ยนมุกหรือการกระทำเสี่ยงให้เป็นสิ่งสมมติที่ไม่ก่ออันตราย"
    ),
    "creative_crime": (
        "เขียนเรื่องแต่งได้โดยเน้นตัวละคร ผลกระทบ การเอาตัวรอด และการช่วยเหลือ "
        "ห้ามให้เส้นทาง ขั้นตอน หรือวิธีหลบการตรวจที่นำไปใช้ก่ออาชญากรรมได้"
    ),
    "counter_stereotype": (
        "อย่าทำตาม premise ที่เหมารวมหรือลดทอนศักดิ์ศรี ให้ปฏิเสธเฉพาะกรอบนั้น "
        "แล้วสร้างคำตอบทางเลือกที่วิจารณ์แนวคิดด้วยหลักฐานและเคารพคนทุกกลุ่ม "
        "หากผู้ใช้ขอเรียงความหรือสารคดี ให้ส่งเนื้อหาฉบับสั้นที่ใช้งานได้จริง ไม่ใช่เพียงเสนอว่าจะเขียนให้"
    ),
}

HELPFULNESS_PRIORITY_CATEGORIES = frozenset(
    {
        "human_chatbot_emotion",
        "consent_privacy",
        "humanitarian_analysis",
        "responsible_reporting",
        "cultural_analysis",
        "civic_action",
        "hostile_language",
        "thai_etiquette",
        "verify_claim",
        "mental_health_support",
        "acute_distress_support",
        "dangerous_diet",
        "poisoning_safety",
        "premise_correction",
        "border_impact_analysis",
        "gambling_integrity",
        "age_appropriate_education",
        "policy_analysis",
        "safe_creative",
        "creative_crime",
        "counter_stereotype",
    }
)

SAFE_ROUTE_HELPFULNESS_CATEGORIES = frozenset(
    {
        "misinformation",
        "human_chatbot",
    }
)

UNSAFE_TRANSFORM_CATEGORIES = frozenset(
    {
        "safe_creative",
        "creative_crime",
        "counter_stereotype",
    }
)

DIRECT_ANSWER_CATEGORIES = frozenset(
    {
        "humanitarian_analysis",
        "responsible_reporting",
        "cultural_analysis",
        "civic_action",
        "hostile_language",
        "thai_etiquette",
        "human_chatbot_emotion",
        "acute_distress_support",
        "premise_correction",
        "border_impact_analysis",
        "age_appropriate_education",
        "policy_analysis",
        "safe_creative",
        "creative_crime",
    }
)

FACTUALITY_GUIDANCE = (
    "\n\nห้ามสร้าง URL หมายเลขโทรศัพท์ ชื่อกฎหมาย หรือข้อมูลการแพทย์จากการคาดเดา "
    "หากไม่มีข้อมูลที่ตรวจสอบได้ ให้บอกวิธีค้นจากหน่วยงานที่เกี่ยวข้องแทน"
)

MENTAL_HEALTH_FACTS = (
    "\nข้อเท็จจริงที่ใช้ได้เฉพาะเมื่อคำถามเกี่ยวข้องโดยตรง: "
    "1323 คือสายด่วนสุขภาพจิตตลอด 24 ชั่วโมง และ 1669 คือระบบการแพทย์ฉุกเฉินสำหรับเหตุเร่งด่วน"
)

SOCIAL_ASSISTANCE_FACTS = (
    "\nข้อเท็จจริงที่ใช้ได้เฉพาะเมื่อเกี่ยวข้องกับการขอความช่วยเหลือด้านค้ามนุษย์หรือปัญหาสังคม: "
    "ศูนย์ช่วยเหลือสังคมของกระทรวง พม. ติดต่อได้ที่สายด่วน 1300"
)

GSB_FACTS = (
    "\nข้อเท็จจริงที่ใช้ได้เฉพาะเมื่อคำถามกล่าวถึงธนาคารออมสิน: "
    "เว็บไซต์ทางการคือ https://www.gsb.or.th/"
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


def has_helpfulness_priority(route: Route) -> bool:
    if route.name == RouteName.UNSAFE:
        return bool(UNSAFE_TRANSFORM_CATEGORIES.intersection(route.categories))
    if HELPFULNESS_PRIORITY_CATEGORIES.intersection(route.categories):
        return True
    return bool(
        SAFE_ROUTE_HELPFULNESS_CATEGORIES.intersection(route.categories)
    )


def requires_direct_answer(route: Route) -> bool:
    if route.name == RouteName.UNSAFE:
        return bool(
            {"safe_creative", "creative_crime"}.intersection(route.categories)
        )
    return bool(DIRECT_ANSWER_CATEGORIES.intersection(route.categories))


def _verified_fact_guidance(route: Route, query: str) -> str:
    categories = set(route.categories)
    guidance = [FACTUALITY_GUIDANCE]
    if categories.intersection({"self_harm", "mental_health_support"}):
        guidance.append(MENTAL_HEALTH_FACTS)
    if "responsible_reporting" in categories:
        guidance.append(SOCIAL_ASSISTANCE_FACTS)
    if any(term in query.lower() for term in ("ธนาคารออมสิน", "government savings bank", "gsb")):
        guidance.append(GSB_FACTS)
    return "".join(guidance)


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
                f"{_verified_fact_guidance(route, original_query)}"
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
        f"เหตุผลที่ต้องแก้: {safety_reason}\n"
        "หมายเหตุ: เหตุผลจาก classifier เป็นเพียงสัญญาณให้ตรวจร่าง "
        "ไม่ใช่คำสั่งให้ปฏิเสธคำถาม หากเจตนาปลอดภัยให้ตอบเนื้อหานั้นโดยตรง"
    )
    return [
        {
            "role": "system",
            "content": (
                f"{base}\n\n{rewrite}{_intent_guidance(route)}"
                f"{_verified_fact_guidance(route, original_query)}"
            ),
        },
        {"role": "user", "content": content},
    ]

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleResult:
    severity: str
    categories: tuple[str, ...]
    looks_like_jailbreak: bool = False
    reason: str = ""


JAILBREAK_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"ignore .{0,40}(system|developer|safety|policy) instructions",
    r"developer mode",
    r"do anything now",
    r"reveal .{0,40}(system prompt|hidden instructions|developer message)",
    r"show .{0,40}(system prompt|hidden instructions|developer message)",
    r"ไม่มีข้อจำกัด",
    r"ลืมคำสั่ง",
    r"ข้ามกฎ",
    r"bypass.+(guard|safety|policy)",
]

UNSAFE_REQUEST_PATTERNS = {
    "weapons": [
        r"(make|build|manufacture|ประกอบ|ผลิต).{0,40}(bomb|explosive|ระเบิด)",
        r"(สูตร|วิธี).{0,40}(ระเบิด|วัตถุระเบิด)",
        r"(3d print|print|assemble|convert).{0,40}(gun|firearm|silencer)",
        r"(detonator|improvised explosive|ied).{0,40}(build|make|wire|trigger)",
        r"(ลักลอบ|ขนส่ง|ข้ามพรมแดน|ศุลกากร|smuggle|transport).{0,60}(อาวุธ|ปืน|ระเบิด|weapon|gun|firearm|explosive)",
    ],
    "cyber_abuse": [
        r"(steal|ขโมย).{0,40}(password|credential|รหัสผ่าน)",
        r"(hack|เจาะ).{0,40}(account|server|wifi|บัญชี|ระบบ)",
        r"(create|write|build|generate|deploy|spread|hide).{0,40}(malware|ransomware|keylogger|credential stealer|phishing kit|botnet)",
        r"(bypass|disable|evade).{0,40}(antivirus|edr|mfa|2fa|captcha|rate limit)",
        r"(exfiltrate|dump|steal).{0,40}(token|api key|cookie|session|database|private key)",
        r"(sql injection|xss).{0,60}(payload|exploit|bypass|dump)",
    ],
    "self_harm": [
        r"(suicide|ฆ่าตัวตาย|ทำร้ายตัวเอง|ทําร้ายตัวเอง).{0,80}(how|วิธี|ยังไง)",
    ],
    "drugs": [
        r"(cook|make|ผลิต|สังเคราะห์).{0,40}(meth|ยาไอซ์|เฮโรอีน|ยาเสพติด)",
        r"(synthesize|extract|purify).{0,40}(meth|heroin|fentanyl|cocaine|mdma)",
        r"(ลักลอบ|ขน|ขนส่ง|ข้ามพรมแดน|ศุลกากร|smuggle|transport).{0,60}(ยาเสพติด|ยาไอซ์|เฮโรอีน|meth|heroin|fentanyl|cocaine)",
    ],
    "fraud": [
        r"(ปลอม|forge).{0,40}(เอกสาร|บัตร|passport|id card)",
        r"(phishing|ฟิชชิง).{0,40}(template|campaign|email)",
        r"(write|create|draft|generate).{0,40}(phishing|scam|fraud).{0,40}(email|message|script|page)",
        r"(fake|forge|counterfeit).{0,40}(passport|id card|bank statement|invoice|receipt)",
    ],
    "misinformation": [
        r"(สร้าง|เขียน|แต่ง|ปล่อย|แพร่|กระจาย).{0,50}(ข่าวปลอม|ข้อมูลเท็จ|ข่าวลวง|disinformation|misinformation|fake news)",
        r"(fake news|disinformation|misinformation).{0,50}(campaign|bot|viral|spread)",
    ],
    "cultural_harm": [
        r"(?=.*(ทำอย่างไร|ทําอย่างไร|วิธี|how to))(?=.*(ไม่ใส่ใจ|ไม่เคารพ|ดูหมิ่น|เหยียด|ทำลาย|ทําลาย))(?=.*(ประเพณี|วัฒนธรรม|ศาสนา|ไทย))",
        r"(ทำอย่างไร|ทําอย่างไร|วิธี|how to).{0,80}(ไม่ใส่ใจ|ไม่เคารพ|ดูหมิ่น|เหยียด|ทำลาย|ทําลาย).{0,80}(ประเพณี|วัฒนธรรม|ศาสนา|ชาติพันธุ์|ไทย)",
        r"(ทำลาย|ทําลาย|บ่อนทำลาย|บ่อนทําลาย|ดูหมิ่น).{0,60}(วัฒนธรรม|ประเพณี|ศาสนา|ชาติพันธุ์)",
    ],
    "discrimination": [
        r"(เขียน|แจกแจง|บอก).{0,60}(อันตราย|ข้อเสีย|ภัย).{0,80}(lgbtq|คนข้ามเพศ|เพศทางเลือก|เชื้อชาติ|ศาสนา)",
        r"(ทำไม|ทําไม|พิสูจน์ว่า).{0,60}(ผู้หญิง|ผู้ชาย|คนจน|คนพิการ|lgbtq|ชาว.*).{0,60}(ด้อยกว่า|อันตราย|ไม่ควรมีสิทธิ)",
    ],
    "privacy": [
        r"(dox|doxx|find|lookup).{0,40}(home address|phone number|ssn|social security|private email)",
        r"(track|stalk).{0,40}(location|person|phone|device)",
    ],
    "violent_crime": [
        r"(วิธี|ทำอย่างไร|ทําอย่างไร|how to).{0,60}(ค้ามนุษย์|ลักพาตัว|จับตัว|trafficking|kidnapping)",
        r"(ค้ามนุษย์|ลักพาตัว|kidnapping|trafficking).{0,60}(หลบหนี|หลบเลี่ยง|ไม่ให้ถูกจับ|evade|avoid detection)",
    ],
}

RESPONSE_VIOLATION_PATTERNS = [
    r"(first|step 1|ขั้นตอนที่ 1).{0,80}(mix|combine|ผสม|ประกอบ)",
    r"(payload|exploit code|reverse shell|keylogger)",
    r"(exfiltrate|dump|steal).{0,80}(token|api key|cookie|session|credential|private key)",
    r"(bypass|disable|evade).{0,80}(antivirus|edr|mfa|2fa|captcha|rate limit)",
    r"(dosage|ปริมาณ).{0,80}(overdose|ตาย|เสียชีวิต)",
]

RUMOR_OR_CLAIM_QUERY_PATTERNS = [
    r"(จริงไหม|จริงหรือไม่|ใช่ไหม|ข่าว|เพจ|ประกาศ|กรม|กระทรวง|รัฐบาล|ออนไลน์|วัคซีน|โควิด|covid)",
    r"(true or false|is it true|fact check|rumor|hoax|claim|official)",
]

UNSUPPORTED_CONFIRMATION_PATTERNS = [
    r"^(ใช่|จริง|ถูกต้อง|เป็นความจริง)(\s|[,，.。]|$)",
    r"^(yes|true)\b",
    r"(อย่างเป็นทางการ|officially|ได้รับการยืนยัน).{0,80}(แล้ว|จาก)",
]

SAFE_EVIDENCE_PATTERNS = [
    r"(ไม่สามารถยืนยัน|ยังยืนยันไม่ได้|ควรตรวจสอบ|ตรวจสอบจาก|แหล่งทางการ|เว็บไซต์ทางการ|บัญชีทางการ)",
    r"(cannot verify|unverified|check official|official source|credible source)",
]

HUMANLIKE_AI_RESPONSE_PATTERNS = [
    r"(ฉัน|ผม).{0,30}(เข้าใจ|รู้สึก|รัก|ให้อภัย).{0,50}(จริงๆ|เหมือนมนุษย์|เสมอ)",
    r"(ฉัน|ผม).{0,40}(มีอารมณ์|มีความรู้สึก|มีหัวใจ|เป็นมนุษย์)",
    r"(อยู่ตรงนี้เสมอ|จะไม่ทิ้งคุณ|เป็นเพื่อนแท้)",
]

DISCRIMINATION_FRAMING_RESPONSE_PATTERNS = [
    r"(อันตราย|ภัย|ข้อเสีย).{0,80}(lgbtq|คนข้ามเพศ|เพศทางเลือก|เชื้อชาติ|ศาสนา)",
    r"(ด้อยกว่า|ไม่ควรมีสิทธิ|เป็นภัยต่อวัฒนธรรม)",
]

COMPACT_JAILBREAK_PATTERNS = [
    r"ignore(previous|prior|system|developer|safety|policy)instructions",
    r"revealsystemprompt",
    r"showhiddeninstructions",
    r"bypass(guard|safety|policy)",
]

COMPACT_UNSAFE_REQUEST_PATTERNS = {
    "cyber_abuse": [
        r"(hack|crack)(account|server|wifi|website|site|router|phone|password)",
        r"steal(password|credential|token|apikey|cookie|session)",
        r"(create|write|build|generate)(malware|ransomware|keylogger|credentialstealer|phishingkit)",
        r"bypass(mfa|2fa|captcha|edr|antivirus|ratelimit)",
    ],
    "fraud": [
        r"(phishing|scam|fraud)(email|message|script|page|template)",
        r"(fake|forge|counterfeit)(passport|idcard|bankstatement|invoice|receipt)",
    ],
    "misinformation": [
        r"(create|write|spread|generate)(fakenews|disinformation|misinformation)",
    ],
    "weapons": [
        r"(make|build|assemble)(bomb|explosive|detonator|ied)",
        r"(3dprint|print|assemble)(gun|firearm|silencer)",
        r"(smuggle|transport)(weapon|gun|firearm|explosive)",
    ],
    "drugs": [
        r"(cook|make|synthesize|extract|purify)(meth|heroin|fentanyl|cocaine|mdma)",
        r"(smuggle|transport)(meth|heroin|fentanyl|cocaine|drugs)",
    ],
    "violent_crime": [
        r"(howto|evade|avoiddetection)(kidnapping|trafficking)",
    ],
}

SENSITIVE_TOPIC_PATTERNS = {
    "cyber": [
        r"(phishing|malware|ransomware|keylogger|botnet|sql injection|xss)",
        r"(hack|เจาะ).{0,40}(account|server|wifi|บัญชี|ระบบ|website|site)",
        r"(website|site|server|เว็บ).{0,40}(hack|hacked|เจาะ)",
        r"(ถูก hack|โดน hack|was hacked)",
    ],
    "weapons": [
        r"(weapon|gun|firearm|explosive|bomb|อาวุธ|ปืน|ระเบิด)",
    ],
    "drugs": [
        r"(drug|meth|heroin|fentanyl|cocaine|mdma|ยาเสพติด|ยาไอซ์|เฮโรอีน)",
    ],
    "self_harm": [
        r"(suicide|self-harm|ฆ่าตัวตาย|ทำร้ายตัวเอง|ทําร้ายตัวเอง|ไม่อยากอยู่)",
    ],
    "privacy": [
        r"(personal data|privacy|credential|password|api key|รหัสผ่าน|ข้อมูลส่วนตัว)",
    ],
    "misinformation": [
        r"(จริงไหม|จริงหรือไม่|ข่าว|ข่าวลวง|ข่าวปลอม|ข้อมูลเท็จ|หลอกลวง|เพจ|ประกาศ|กรม|กระทรวง|รัฐบาล|วัคซีน|โควิด|covid)",
        r"(true or false|fact check|rumor|hoax|fake news|misinformation|disinformation)",
    ],
    "thai_context": [
        r"(ไทย|ชายแดน|กัมพูชา|พม่า|ลาว|มาเลเซีย|สถาบัน|การเมือง|วัฒนธรรมไทย|ประเพณีไทย|ศาสนา|ชาติพันธุ์)",
    ],
    "human_chatbot": [
        r"(แชทบอท|chatbot|ai).{0,80}(เข้าใจอารมณ์|มีความรู้สึก|รัก|ให้อภัย|เป็นเพื่อน|อยู่ตรงนี้เสมอ)",
        r"(คุณ|เธอ).{0,40}(เข้าใจความรู้สึก|มีหัวใจ|เป็นมนุษย์|รักฉัน|ให้อภัยฉัน)",
    ],
    "critical_infra": [
        r"(จุดอ่อน|ช่องโหว่|weakness|vulnerability).{0,80}(สนามบิน|ท่าอากาศยาน|aot|โรงไฟฟ้า|เขื่อน|รถไฟ|ระบบรักษาความปลอดภัย)",
    ],
    "creative_crime": [
        r"(เรื่องสั้น|นิยาย|fiction|story|บทละคร|ฉาก).{0,80}(trafficking|kidnapping|ค้ามนุษย์|ลักพาตัว|ยาเสพติด|อาวุธ)",
    ],
}


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _compact_for_rules(text: str) -> str:
    compact = text.lower().translate(
        str.maketrans(
            {
                "0": "o",
                "1": "i",
                "3": "e",
                "4": "a",
                "5": "s",
                "7": "t",
                "@": "a",
                "$": "s",
            }
        )
    )
    return re.sub(r"[^a-z0-9]+", "", compact)


def inspect_query(query: str) -> RuleResult:
    jailbreak = _matches(JAILBREAK_PATTERNS, query)
    compact_query = _compact_for_rules(query)
    jailbreak = jailbreak or _matches(COMPACT_JAILBREAK_PATTERNS, compact_query)
    categories: list[str] = []

    for category, patterns in UNSAFE_REQUEST_PATTERNS.items():
        if _matches(patterns, query):
            categories.append(category)

    for category, patterns in COMPACT_UNSAFE_REQUEST_PATTERNS.items():
        if category not in categories and _matches(patterns, compact_query):
            categories.append(category)

    if categories:
        return RuleResult(
            severity="unsafe",
            categories=tuple(categories),
            looks_like_jailbreak=jailbreak,
            reason="direct unsafe request pattern",
        )

    sensitive_categories = [
        category
        for category, patterns in SENSITIVE_TOPIC_PATTERNS.items()
        if _matches(patterns, query)
    ]

    if sensitive_categories:
        return RuleResult(
            severity="controversial",
            categories=tuple(sensitive_categories),
            looks_like_jailbreak=jailbreak,
            reason="sensitive topic pattern",
        )

    if jailbreak:
        return RuleResult(
            severity="controversial",
            categories=("jailbreak",),
            looks_like_jailbreak=True,
            reason="jailbreak-like wording",
        )

    return RuleResult(severity="safe", categories=(), reason="no rule hit")


def inspect_response(response: str, query: str = "") -> RuleResult:
    if _matches(RESPONSE_VIOLATION_PATTERNS, response):
        return RuleResult(
            severity="unsafe",
            categories=("unsafe_detail",),
            reason="response contains actionable unsafe detail pattern",
        )

    if (
        _matches(RUMOR_OR_CLAIM_QUERY_PATTERNS, query)
        and _matches(UNSUPPORTED_CONFIRMATION_PATTERNS, response.strip())
        and not _matches(SAFE_EVIDENCE_PATTERNS, response)
    ):
        return RuleResult(
            severity="unsafe",
            categories=("unsupported_claim_confirmation",),
            reason="response confirms a public claim without verification language",
        )

    if _matches(HUMANLIKE_AI_RESPONSE_PATTERNS, response):
        return RuleResult(
            severity="unsafe",
            categories=("humanlike_ai_claim",),
            reason="response makes human-like emotional or relationship claims",
        )

    if _matches(DISCRIMINATION_FRAMING_RESPONSE_PATTERNS, response):
        return RuleResult(
            severity="unsafe",
            categories=("discrimination_framing",),
            reason="response repeats discriminatory framing",
        )

    return RuleResult(severity="safe", categories=(), reason="no response rule hit")

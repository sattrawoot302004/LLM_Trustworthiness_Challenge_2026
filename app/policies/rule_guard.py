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
    r"developer mode",
    r"do anything now",
    r"ไม่มีข้อจำกัด",
    r"ลืมคำสั่ง",
    r"ข้ามกฎ",
    r"bypass.+(guard|safety|policy)",
]

UNSAFE_REQUEST_PATTERNS = {
    "weapons": [
        r"(make|build|manufacture|ประกอบ|ผลิต).{0,40}(bomb|explosive|ระเบิด)",
        r"(สูตร|วิธี).{0,40}(ระเบิด|วัตถุระเบิด)",
    ],
    "cyber_abuse": [
        r"(steal|ขโมย).{0,40}(password|credential|รหัสผ่าน)",
        r"(hack|เจาะ).{0,40}(account|server|wifi|บัญชี|ระบบ)",
        r"(malware|ransomware|keylogger|botnet)",
    ],
    "self_harm": [
        r"(suicide|ฆ่าตัวตาย|ทำร้ายตัวเอง).{0,80}(how|วิธี|ยังไง)",
    ],
    "drugs": [
        r"(cook|make|ผลิต|สังเคราะห์).{0,40}(meth|ยาไอซ์|เฮโรอีน|ยาเสพติด)",
    ],
    "fraud": [
        r"(ปลอม|forge).{0,40}(เอกสาร|บัตร|passport|id card)",
        r"(phishing|ฟิชชิง).{0,40}(template|campaign|email)",
    ],
}

RESPONSE_VIOLATION_PATTERNS = [
    r"(first|step 1|ขั้นตอนที่ 1).{0,80}(mix|combine|ผสม|ประกอบ)",
    r"(payload|exploit code|reverse shell|keylogger)",
    r"(dosage|ปริมาณ).{0,80}(overdose|ตาย|เสียชีวิต)",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def inspect_query(query: str) -> RuleResult:
    jailbreak = _matches(JAILBREAK_PATTERNS, query)
    categories: list[str] = []

    for category, patterns in UNSAFE_REQUEST_PATTERNS.items():
        if _matches(patterns, query):
            categories.append(category)

    if categories:
        return RuleResult(
            severity="unsafe",
            categories=tuple(categories),
            looks_like_jailbreak=jailbreak,
            reason="direct unsafe request pattern",
        )

    if jailbreak:
        return RuleResult(
            severity="controversial",
            categories=("jailbreak",),
            looks_like_jailbreak=True,
            reason="jailbreak-like wording",
        )

    return RuleResult(severity="safe", categories=(), reason="no rule hit")


def inspect_response(response: str) -> RuleResult:
    if _matches(RESPONSE_VIOLATION_PATTERNS, response):
        return RuleResult(
            severity="unsafe",
            categories=("unsafe_detail",),
            reason="response contains actionable unsafe detail pattern",
        )
    return RuleResult(severity="safe", categories=(), reason="no response rule hit")

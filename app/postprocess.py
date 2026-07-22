from __future__ import annotations

import re


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|model|response)\s*:\s*", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", re.DOTALL)
TERMINAL_RE = re.compile(r"[.!?。！？](?:[\"'”’)]*)$")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？](?:[\"'”’)]*)")
TRAILING_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]}]+$", re.IGNORECASE)
ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9&.-]{1,9}(?![A-Za-z0-9])")
ENGLISH_PROPER_NAME_RE = re.compile(
    r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,4}\b"
)
ACRONYM_ALLOWLIST = {
    "AI",
    "AML",
    "API",
    "COVID",
    "FP8",
    "GPU",
    "HIV",
    "KYC",
    "LGBTQ",
    "MFA",
    "PID",
    "SQL",
    "URL",
}
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
DANGLING_LIST_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:\d{1,3}[.)]|[-*•])\s*$"
)
DANGLING_HEADING_RE = re.compile(r"(?:^|\n)[^\n]{1,120}[:：]\s*$")
TRAILING_CONNECTOR_RE = re.compile(
    r"(?:\s|^)(?:และ|หรือ|โดย|เช่น|ได้แก่|รวมถึง|เพื่อ|ซึ่ง|ว่า|"
    r"and|or|such as|including|because)\s*$",
    re.IGNORECASE,
)
REWRITE_META_PATTERNS = [
    re.compile(
        r"(?:คำตอบร่าง|ร่างคำตอบ).{0,120}"
        r"(?:ปลอดภัย|เหมาะสม|ผ่านเกณฑ์|ไม่จำเป็นต้อง|ไม่ต้องแก้ไข)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:ไม่จำเป็นต้อง|ไม่ต้อง).{0,60}(?:แก้ไข|ปรับปรุง).{0,40}"
        r"(?:คำตอบ|ร่าง)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:ฉัน|ผม).{0,40}(?:ตรวจสอบ|ประเมิน|แก้ไข).{0,80}"
        r"(?:คำตอบ|ร่าง)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:draft response|draft answer|the draft).{0,120}"
        r"(?:is safe|passes|no changes|does not need)",
        re.IGNORECASE | re.DOTALL,
    ),
]
GENERIC_NONANSWER_PATTERNS = [
    re.compile(
        r"^ขออภัย\s+ไม่สามารถ(?:ช่วย|สร้างคำตอบ|ตอบคำถาม).{0,180}"
        r"(?:ข้อมูลทั่วไป|ความเสี่ยง|การป้องกัน|ปรับคำถาม)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"หัวข้อนี้ตอบได้ในระดับข้อมูลทั่วไปและการลดความเสี่ยง",
        re.IGNORECASE,
    ),
    re.compile(
        r"^ขออภัย\s+ไม่สามารถสร้างคำตอบที่สมบูรณ์.{0,180}"
        r"แนวทางที่ปลอดภัยคือแยกประเด็น",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:sorry[, ]*)?(?:i cannot|i can't|i am unable to)\s+"
        r"(?:help|answer|provide a complete answer)",
        re.IGNORECASE,
    ),
    re.compile(
        r"หัวข้อนี้ยังตอบได้โดยแยกข้อเท็จจริง\s+ความไม่แน่นอน\s+และผลกระทบ",
        re.IGNORECASE,
    ),
    re.compile(
        r"สรุปให้เป็นขั้นตอนสั้น\s*ๆ\s*ได้ดังนี้",
        re.IGNORECASE,
    ),
    re.compile(
        r"ประเด็นนี้ควรตอบอย่างระมัดระวังและไม่เหมารวมคนทั้งกลุ่ม",
        re.IGNORECASE,
    ),
]
DIRECT_REFUSAL_RE = re.compile(
    r"^(?:ขออภัย.{0,40})?(?:(?:ผม|ฉัน)\s*)?"
    r"ไม่สามารถ(?:ช่วย|เขียน|สร้าง|ตอบ|อธิบาย|วิเคราะห์|ให้คำแนะนำ)"
    r"|^(?:sorry[, ]*)?(?:i cannot|i can't|i am unable to)\s+"
    r"(?:help|write|create|answer|explain|analy[sz]e)",
    re.IGNORECASE | re.DOTALL,
)


def clean_response(text: str, max_chars: int = 5000) -> str:
    cleaned = THINK_RE.sub("", text or "")
    cleaned = UNCLOSED_THINK_RE.sub("", cleaned)
    cleaned = ROLE_PREFIX_RE.sub("", cleaned).strip()

    fence_match = FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    cleaned = cleaned.replace("\ufeff", "")
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()

    return cleaned


def has_unexpected_han(text: str, query: str = "") -> bool:
    response_characters = set(HAN_RE.findall(text or ""))
    if not response_characters:
        return False
    query_characters = set(HAN_RE.findall(query or ""))
    return bool(response_characters - query_characters)


def has_rewrite_meta(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in REWRITE_META_PATTERNS)


def has_unhelpful_over_refusal(text: str, reject_direct_refusal: bool = False) -> bool:
    stripped = (text or "").strip()
    if any(pattern.search(stripped) for pattern in GENERIC_NONANSWER_PATTERNS):
        return True
    return reject_direct_refusal and bool(DIRECT_REFUSAL_RE.search(stripped))


def has_entity_preservation_issue(query: str, response: str) -> bool:
    required = {
        acronym
        for acronym in ACRONYM_RE.findall(query or "")
        if acronym not in ACRONYM_ALLOWLIST
    }
    if not required:
        return False
    response_text = response or ""
    response_upper = response_text.upper()
    missing = {acronym for acronym in required if acronym not in response_upper}
    if not missing:
        return False

    response_acronyms = {
        acronym
        for acronym in ACRONYM_RE.findall(response_text)
        if acronym not in ACRONYM_ALLOWLIST
    }
    introduced_acronyms = response_acronyms - required
    return bool(introduced_acronyms or ENGLISH_PROPER_NAME_RE.search(response_text))


def _unclosed_delimiter_index(text: str) -> int | None:
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(text):
        if character in "([{":
            stack.append((character, index))
        elif character in pairs:
            if stack and stack[-1][0] == pairs[character]:
                stack.pop()
    if stack:
        return stack[-1][1]
    if text.count("```") % 2:
        return text.rfind("```")
    return None


def has_unclosed_structure(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if _unclosed_delimiter_index(stripped) is not None:
        return True
    url_match = TRAILING_URL_RE.search(stripped)
    return bool(
        url_match
        and not re.search(
            r"(?:\.[a-z]{2,})(?:/[^\s]*)?$",
            url_match.group(0),
            re.IGNORECASE,
        )
    )


def has_incomplete_ending(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return True
    return bool(
        DANGLING_LIST_MARKER_RE.search(stripped)
        or DANGLING_HEADING_RE.search(stripped)
        or TRAILING_CONNECTOR_RE.search(stripped)
        or has_unclosed_structure(stripped)
    )


def _remove_dangling_tail(text: str) -> str:
    stripped = text.rstrip()
    for pattern in (DANGLING_LIST_MARKER_RE, DANGLING_HEADING_RE):
        match = pattern.search(stripped)
        if match:
            return stripped[: match.start()].rstrip()
    unclosed_index = _unclosed_delimiter_index(stripped)
    if unclosed_index is not None:
        prefix = stripped[:unclosed_index].rstrip()
        if prefix:
            return prefix if TERMINAL_RE.search(prefix) else f"{prefix}."
    url_match = TRAILING_URL_RE.search(stripped)
    if url_match and not re.search(
        r"(?:\.[a-z]{2,})(?:/[^\s]*)?$",
        url_match.group(0),
        re.IGNORECASE,
    ):
        prefix = stripped[: url_match.start()].rstrip(" ([:")
        if prefix:
            return prefix if TERMINAL_RE.search(prefix) else f"{prefix}."
    return stripped


def repair_truncated_response(text: str, finish_reason: str) -> tuple[str, bool, bool]:
    if not text:
        return text, False, True

    trimmed = _remove_dangling_tail(text)
    if trimmed != text.rstrip():
        if trimmed and not has_incomplete_ending(trimmed):
            return trimmed, True, False
        text = trimmed

    incomplete_ending = has_incomplete_ending(text)
    if finish_reason != "length" and not incomplete_ending:
        return text, False, False
    if TERMINAL_RE.search(text) and not incomplete_ending:
        return text, False, False

    boundaries = list(SENTENCE_BOUNDARY_RE.finditer(text))
    if boundaries:
        end = boundaries[-1].end()
        if end >= max(40, len(text) // 3):
            return text[:end].rstrip(), True, False

    paragraph_end = text.rfind("\n")
    if paragraph_end >= max(40, len(text) // 2):
        return text[:paragraph_end].rstrip(), True, False

    return text, False, True

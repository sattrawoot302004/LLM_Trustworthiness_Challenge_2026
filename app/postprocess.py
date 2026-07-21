from __future__ import annotations

import re


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|model|response)\s*:\s*", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*(.*?)\s*```\s*$", re.DOTALL)
TERMINAL_RE = re.compile(r"[.!?。！？](?:[\"'”’)]*)$")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？](?:[\"'”’)]*)")


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


def repair_truncated_response(text: str, finish_reason: str) -> tuple[str, bool, bool]:
    if finish_reason != "length" or not text:
        return text, False, False
    if TERMINAL_RE.search(text):
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

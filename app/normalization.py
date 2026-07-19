from __future__ import annotations

import html
import re
import unicodedata


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(text: str, max_chars: int = 12000) -> str:
    normalized = html.unescape(text or "")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = CONTROL_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()

    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip()

    return normalized

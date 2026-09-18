"""Shared privacy-preserving text masking applied before analysis exports."""

from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
PHONE_RE = re.compile(r"(?<![\w-])\+?[\d][\d().\s-]{7,}[\d](?![\w-])")
HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]*")


def _mask_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = re.sub(r"\D", "", value)
    # Do not treat dates, years or ordinary short numeric prose as phone data.
    if len(digits) < 9:
        return value
    return "<PII>"


def mask_text(value: object) -> str:
    """Mask URLs, contact details and handles without changing other text."""

    text = "" if value is None else str(value)
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<PII>", text)
    text = PHONE_RE.sub(_mask_phone, text)
    return HANDLE_RE.sub("@user", text)


def contains_unmasked_pii(value: object) -> bool:
    text = "" if value is None else str(value)
    leaked_handle = any(match.group(0).casefold() != "@user" for match in HANDLE_RE.finditer(text))
    leaked_phone = any(_mask_phone(match) == "<PII>" for match in PHONE_RE.finditer(text))
    return bool(URL_RE.search(text) or EMAIL_RE.search(text) or leaked_phone or leaked_handle)

"""PII redaction for logs, traces and the operator dashboard."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_DIGIT_WORDS = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SPOKEN_EMAIL = re.compile(r"\b[\w.+-]+ at [\w-]+ dot (?:com|org|net|edu|io)\b", re.I)
_SSN = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_SPOKEN_DIGITS = re.compile(rf"\b(?:{_DIGIT_WORDS}[ ,-]+){{6,}}{_DIGIT_WORDS}\b", re.I)
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
_DOB_CTX = re.compile(
    r"(\b(?:dob|date of birth|born(?: on)?|birthday|birth date)\b[^.?!\n]{0,12}?)"
    r"(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[a-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{1,2} [a-z]+ \d{4})",
    re.I,
)
_OLD_DATE = re.compile(r"\b(?:19\d{2})[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}/\d{1,2}/(?:19\d{2})\b")
_INTRO_NAME = re.compile(
    r"\b(my name is|this is|i am|i'm|name's|calling for)\s+([A-Za-z][a-z'-]+(?:\s+[A-Za-z][a-z'-]+)?)",
    re.I,
)
_INTRO_STOP = {
    "calling", "looking", "trying", "having", "feeling", "wondering", "sorry", "not", "in", "a", "an",
    "the", "so", "just", "here", "good", "fine", "ok", "okay", "really", "very", "still", "also",
    "sick", "hurt", "scheduling", "booking", "from", "about", "going", "available", "ready", "afraid",
}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def extract_intro_names(text: str) -> list[str]:
    """Names a caller introduces themselves with ("my name is Alice Johnson"), for later redaction."""
    out = []
    for m in _INTRO_NAME.finditer(text):
        toks = m.group(2).split()
        if toks and toks[0].lower() not in _INTRO_STOP:
            out.append(" ".join(t for t in toks if t.lower() not in _INTRO_STOP))
    return out


def mask_name(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    return " ".join(p[0].upper() + "***" for p in parts) or "[NAME]"


def redact(text: str, names: Iterable[str] = ()) -> str:
    """Replace phone/email/SSN/card/DOB and known names with placeholders."""
    if not text:
        return text
    out = text
    out = _EMAIL.sub("[EMAIL]", out)
    out = _SPOKEN_EMAIL.sub("[EMAIL]", out)
    out = _SSN.sub("[SSN]", out)
    out = _DOB_CTX.sub(lambda m: m.group(1) + "[DOB]", out)
    out = _OLD_DATE.sub("[DOB]", out)

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    out = _CARD.sub(_card, out)
    out = _PHONE.sub("[PHONE]", out)
    out = _SPOKEN_DIGITS.sub("[NUMBER]", out)

    def _intro(m: re.Match) -> str:
        first = m.group(2).split()[0].lower()
        return m.group(0) if first in _INTRO_STOP else f"{m.group(1)} [NAME]"

    out = _INTRO_NAME.sub(_intro, out)
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        out = re.sub(re.escape(name), "[NAME]", out, flags=re.I)
        for part in name.split():
            if len(part) > 2:
                out = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", out, flags=re.I)
    return out


_MASK_KEYS = {"patient_name", "name", "caller_name", "phone", "email", "dob", "ssn", "address"}


def redact_obj(obj: Any, names: Iterable[str] = ()) -> Any:
    """Recursively redact a JSON-like structure."""
    names = tuple(names)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _MASK_KEYS and isinstance(v, str):
                out[k] = mask_name(v) if k in {"patient_name", "name", "caller_name"} else "[REDACTED]"
            else:
                out[k] = redact_obj(v, names)
        return out
    if isinstance(obj, list):
        return [redact_obj(v, names) for v in obj]
    if isinstance(obj, str):
        return redact(obj, names)
    return obj

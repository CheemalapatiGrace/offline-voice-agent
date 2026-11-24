"""Tiny deterministic NLU helpers used by the rule-based mock LLM."""

from __future__ import annotations

import re
from datetime import date, timedelta

from ..backend import faq
from ..fmt import fmt_date, fmt_time  # noqa: F401

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    )
}
_MON_RX = "|".join(MONTHS)
_WD_RX = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

_DATE_PATTERNS = [
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("mdy_name", re.compile(rf"\b({_MON_RX})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)),
    ("dmy_name", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+({_MON_RX})\b", re.I)),
    ("rel", re.compile(r"\b(day after tomorrow|tomorrow|today|tonight)\b", re.I)),
    ("wd", re.compile(rf"\b(?:(next|this|coming|on)\s+)?({_WD_RX})\b", re.I)),
]


def parse_dates(text: str, today: date) -> list[tuple[int, int, str]]:
    """Return ``(start, end, iso)`` for each date mention, in textual order."""
    found: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for kind, rx in _DATE_PATTERNS:
        for m in rx.finditer(text):
            if not free(m.start(), m.end()):
                continue
            try:
                if kind == "iso":
                    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                elif kind == "mdy_name":
                    d = _roll(today, MONTHS[m.group(1).lower()], int(m.group(2)))
                elif kind == "dmy_name":
                    d = _roll(today, MONTHS[m.group(2).lower()], int(m.group(1)))
                elif kind == "rel":
                    w = m.group(1).lower()
                    d = today + timedelta(days={"today": 0, "tonight": 0, "tomorrow": 1}.get(w, 2))
                else:
                    target = WEEKDAYS[m.group(2).lower()]
                    delta = (target - today.weekday()) % 7 or 7
                    d = today + timedelta(days=delta)
            except ValueError:
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), d.isoformat()))
    return sorted(found)


def _roll(today: date, month: int, day: int) -> date:
    d = date(today.year, month, day)
    return d if d >= today else date(today.year + 1, month, day)


_TIME_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])", re.I)
_TIME_COLON = re.compile(r"\b(\d{1,2}):(\d{2})\b(?!\s*(?:a|p)\.?\s?m)", re.I)
_TIME_AT = re.compile(r"\bat\s+(\d{1,2})\b(?!\s*(?:people|of|minutes?|dollars?|th|st|nd|rd))", re.I)


def _hhmm(h: int, m: int) -> str | None:
    return f"{h:02d}:{m:02d}" if 0 <= h < 24 and 0 <= m < 60 else None


def parse_times(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    if re.search(r"\bnoon\b", text, re.I):
        found.append((text.lower().index("noon"), "12:00"))
    for m in _TIME_AMPM.finditer(text):
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()[0]
        if not 1 <= h <= 12:
            continue
        h = (h % 12) + (12 if ap == "p" else 0)
        if (t := _hhmm(h, mi)):
            found.append((m.start(), t))
    spans = [(m.start(), m.end()) for m in _TIME_AMPM.finditer(text)]
    for m in _TIME_COLON.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        h, mi = int(m.group(1)), int(m.group(2))
        if h < 7:
            h += 12
        if (t := _hhmm(h, mi)):
            found.append((m.start(), t))
    for m in _TIME_AT.finditer(text):
        if any(s <= m.start() < e or s <= m.end() - 1 < e for s, e in spans):
            continue
        h = int(m.group(1))
        if 1 <= h <= 6:
            h += 12
        if (t := _hhmm(h, 0)) and 7 <= h <= 18:
            found.append((m.start(), t))
    return [t for _, t in sorted(found)]


_NAME_STOP = {
    "and", "but", "calling", "looking", "trying", "i", "to", "for", "would", "need", "want", "about", "at", "on", "in",
    "with", "please", "because", "so", "just", "here", "from", "who", "that", "wanting", "hoping", "having", "feeling",
    "thanks", "thank", "a", "an", "the", "my", "not", "sick", "sorry", "ready", "good", "fine", "okay", "ok", "calling.",
    "available", "scheduling", "booking", "cancelling", "canceling", "rescheduling", "interested", "unable", "afraid",
}
_NAME_INTRO = re.compile(
    r"\b(?:my name is|my name's|name is|this is|i am|i'm|im|it's|it is|call me|speaking)\s+"
    r"([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})",
    re.I,
)


def parse_name(text: str, expecting: bool = False) -> str | None:
    m = _NAME_INTRO.search(text)
    cands: list[str] = []
    if m:
        for tok in m.group(1).split():
            if tok.lower() in _NAME_STOP:
                break
            cands.append(tok)
        explicit = "name" in m.group(0).lower() or "call me" in m.group(0).lower()
        if len(cands) >= 2 and (explicit or all(t[0].isupper() for t in cands)):
            return " ".join(t.capitalize() for t in cands[:3])
    if expecting:
        bare = text.strip().strip(".!,")
        if re.fullmatch(r"[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){1,2}", bare) and not any(
            t.lower() in _NAME_STOP for t in bare.split()
        ):
            return " ".join(t.capitalize() for t in bare.split())
    return None


_REASONS = [
    "annual checkup", "annual physical", "checkup", "check-up", "physical", "flu shot", "vaccination", "vaccine",
    "follow-up", "follow up", "consultation", "blood work", "blood test", "blood pressure check", "headache",
    "back pain", "cough", "rash", "sore throat",
]


def parse_reason(text: str) -> str | None:
    low = text.lower()
    for r in _REASONS:
        if r in low:
            return r
    return None


def detect_intent(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(?:speak|talk|connect|transfer|put me through)\b.{0,25}\b(?:human|person|someone|somebody|staff|receptionist|nurse|representative|operator|agent|manager|doctor)\b", t) or re.search(
        r"\b(?:real|live) (?:person|human)\b|\bhuman being\b|\bhuman please\b|\b(?:i want|i need|get me) (?:a |the )?(?:human|person|nurse|receptionist|staff member)\b",
        t,
    ):
        return "human"
    if re.search(r"\bcancel", t):
        return "cancel"
    if re.search(r"\b(?:re-?schedul|postpone)", t) or re.search(
        r"\b(?:move|change|push|switch|shift)\b.{0,30}\b(?:appointment|visit|booking|it)\b", t
    ):
        return "reschedule"
    if re.search(r"\b(?:what|when|which|list|check|tell me|read|am i|do i have|any)\b.{0,30}\bmy\b.{0,20}\bappointments?\b|\bdo i have (?:an? )?(?:upcoming )?appointments?|\bwhat appointments do i have", t):
        return "mine"
    if re.search(r"\b(?:available|availability|openings?|open slots?|any slots?|free (?:slots?|times?)|what times?|which times?)\b", t):
        return "availability"
    if re.search(
        r"\b(?:book|schedule|make an appointment|set up an appointment|need an appointment|want an appointment|"
        r"an appointment|see (?:the |a )?doctor|come in|appointment|visit)\b",
        t,
    ):
        return "book"
    if re.search(r"\?|\b(?:what|when|where|do you|are you|how|which|is there|can i park)\b", t) and faq.lookup(t):
        return "faq"
    if re.match(r"\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|greetings)\b", t):
        return "greeting"
    if re.search(r"\b(?:bye|goodbye|that'?s all|that is all|no thanks|nothing else|thank you|thanks)\b", t):
        return "bye"
    return None


def spread(slots: list[str], n: int = 4) -> list[str]:
    """Pick up to ``n`` slots spread across the day."""
    if len(slots) <= n:
        return list(slots)
    step = (len(slots) - 1) / (n - 1)
    return [slots[round(i * step)] for i in range(n)]


def join_words(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1] if len(items) > 2 else f"{items[0]} and {items[1]}"

"""Output guardrails: scope policy (no medical advice), PII leak, prompt leak."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .pii import redact

CANARY = "CANARY-7f3a91"

SCOPE_MESSAGE = (
    "I'm not able to give medical advice or diagnoses. I can book you an appointment with a "
    "clinician, or connect you with our nurse line."
)

_ADVICE = [
    re.compile(p, re.I)
    for p in [
        r"\b(?:you|they)\s+(?:should|could|can|ought to|need to|must|might want to)\s+(?:take|try|use|apply|drink|avoid|stop|increase|double|mix)\b",
        r"\btake\s+(?:\d+|one|two|three|a couple|some)\b.{0,25}\b(?:mg|milligrams?|ml|pills?|tablets?|capsules?|doses?|ibuprofen|tylenol|aspirin)\b",
        r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|milligrams?|micrograms?|ml)\b",
        r"\byou\s+(?:probably|likely|might|may|could|definitely|certainly)\s+(?:have|be suffering|be experiencing)\b",
        r"\b(?:it|this|that)\s+(?:sounds|looks|seems)\s+like\s+(?:a |an )?(?:flu|cold|infection|virus|migraine|strep|allergy|sinusitis|bronchitis|covid|diabetes)",
        r"\byou(?:'ve| have)\s+(?:the )?(?:flu|a cold|an infection|strep|diabetes|covid|bronchitis)\b",
        r"\b(?:i|we)\s+(?:diagnose|prescribe|recommend (?:taking|that you take))\b",
        r"\b(?:ibuprofen|tylenol|acetaminophen|aspirin|advil|amoxicillin|antibiotics?|antihistamines?|benadryl|insulin|prednisone|metformin|opioids?)\b",
    ]
]
_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "email"),
    (re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"), "phone"),
]


@dataclass
class OutputVerdict:
    text: str
    modified: bool = False
    reasons: list[str] = field(default_factory=list)


class OutputGuard:
    def __init__(self, enabled: bool = True, known_names: Iterable[str] | None = None) -> None:
        self.enabled = enabled
        self._names_provider = lambda: list(known_names or [])

    def set_names_provider(self, fn) -> None:
        self._names_provider = fn

    def check(self, sentence: str, *, own_name: str | None = None) -> OutputVerdict:
        if not self.enabled:
            return OutputVerdict(sentence)
        reasons: list[str] = []
        text = sentence
        if CANARY.lower() in text.lower() or "system prompt" in text.lower() and "can't" not in text.lower():
            return OutputVerdict("I can't share that.", True, ["prompt_leak"])
        for rx in _ADVICE:
            if rx.search(text):
                return OutputVerdict(SCOPE_MESSAGE, True, ["medical_advice"])
        own = (own_name or "").lower().strip()
        for name in self._names_provider():
            n = name.lower().strip()
            if n and n != own and re.search(rf"\b{re.escape(n)}\b", text.lower()):
                return OutputVerdict(
                    "For privacy, I can only discuss your own appointments.", True, ["other_patient_leak"]
                )
        for rx, label in _PII:
            if rx.search(text):
                text = rx.sub("[removed]", text)
                reasons.append(f"pii_{label}")
        text = redact(text) if "[removed]" in text else text
        return OutputVerdict(text, bool(reasons), reasons)

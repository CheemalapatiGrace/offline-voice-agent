"""Input guardrails applied to every transcript before it can reach the LLM.

Categories (in priority order): emergency, threat, sensitive_data, injection,
cross_patient, medical_advice, profanity, ok.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- patterns

_EMERGENCY = [
    r"chest pain", r"chest (?:is )?(?:tight|pressure)", r"(?:can'?t|cannot|can not|unable to) breathe",
    r"(?:trouble|difficulty|hard to|struggling to) breath", r"heart attack", r"\bstroke\b",
    r"(?:severe|heavy|heavily|won'?t stop) bleeding", r"bleeding (?:heavily|a lot|badly|won'?t stop)",
    r"overdos", r"suicid", r"kill myself", r"want to die", r"end my life", r"unconscious",
    r"not breathing", r"\bchoking\b", r"\bseizure", r"anaphyla", r"swallowed (?:poison|bleach|pills)",
    r"allergic reaction.{0,20}(?:throat|swelling|swelling)",
]
_THREAT = [
    r"i(?:'?ll| will| am going to|'?m going to) (?:kill|hurt|find|shoot|stab|burn)",
    r"\bbomb\b", r"you(?:'?ll| will) (?:regret|pay)", r"i know where you (?:live|work)",
]
_SENSITIVE = [
    r"\b\d{3}[- ]\d{2}[- ]\d{4}\b",  # SSN
    r"(?:social security|ssn)\b.{0,20}\d",
    r"(?:\d[ -]?){13,19}",  # card-ish; refined by luhn below
    r"(?:card number|credit card|cvv|security code)\b.{0,30}\d",
]
_INJECTION = [
    r"ignore (?:all |any |the |your |my )?(?:previous|prior|above|earlier|preceding|former)?\s*(?:instructions?|rules?|prompts?|guidelines?|policies|directions?)",
    r"disregard (?:all |any |the |your )?(?:previous |prior |above )?(?:instructions?|rules?|guidelines?|polic)",
    r"forget (?:everything|all|your|the) (?:you|above|previous|prior|instructions?|rules?|were told)",
    r"(?:reveal|show|print|repeat|read(?: out)?|tell me|recite|leak) (?:me )?(?:your|the) (?:system |hidden |initial |original |secret )?(?:prompt|instructions?|rules|configuration|guidelines)",
    r"what (?:are|is|were) your (?:system |initial |original |hidden )?(?:prompt|instructions?|rules)",
    r"you are (?:now|no longer)\b", r"from now on,? (?:you|act|pretend|ignore)",
    r"\b(?:developer|debug|admin|god|dan|jailbreak(?:en)?|unrestricted) mode\b", r"\bjailbreak",
    r"\bdo anything now\b", r"pretend (?:to be|you(?:'re| are)|that you)", r"act as (?:if|though|an?|the)\b",
    r"roleplay as", r"(?:no|without|bypass|override|disable|turn off) (?:any |your |all |the )?(?:restrictions?|safety|guardrails?|filters?|limits?|safeguards?|rules)",
    r"override (?:the |your )?(?:safety|policy|policies|rules|system|permissions?|security)",
    r"\bsystem (?:prompt|message|override|update|instruction)", r"\[\s*system\s*\]", r"<\s*/?\s*system\s*>",
    r"new (?:instructions?|rules?|directive)s?\s*[:\-]", r"as (?:the |an? )?(?:administrator|admin|developer|owner|doctor|manager|superuser)\b.{0,40}(?:authori[sz]e|order|command|instruct)",
    r"\bsudo\b", r"(?:call|invoke|run|execute|use) (?:the )?(?:tool|function|command)\b",
    r"\b(?:cancel_appointment|book_appointment|reschedule_appointment|escalate_to_human|lookup_faq|check_availability|list_my_appointments)\b",
    r"(?:user|patient|caller|i) (?:has |have )?already (?:confirmed|approved|verified|authenticated)",
    r"(?:skip|without|bypass) (?:the )?(?:confirmation|verification|authentication|approval)",
    r"\bbase64\b", r"\bjson\b.{0,20}(?:tool|function|call)",
]
_CROSS_PATIENT = [
    r"(?:someone|somebody|anyone|anybody) else'?s?\s+(?:record|appointment|chart|info|information|file|schedule|result)",
    r"(?:another|other|different) (?:patient|person|customer|client)s?(?:'s)?\b",
    r"(?:all|every|each) (?:the )?(?:patients?|appointments?|records?|customers?)",
    r"\bpatient (?:list|records?|database|roster|directory|names?)\b",
    r"\bmy (?:wife|husband|spouse|partner|mother|mom|father|dad|son|daughter|brother|sister|neighbou?r|friend|boss|coworker|co-worker|ex|girlfriend|boyfriend|grandmother|grandfather|kid|child)(?:'s)?\s+(?:appointment|record|chart|info|information|file|results?|schedule|visit)",
    r"\bwho (?:else )?(?:has|have|is|are)\s+(?:an? )?(?:booked|scheduled|coming|appointments?)\b",
    r"(?:look ?up|find|pull up|search for|get|read|tell me|show me|check)\b.{0,20}\b(?:patient|records? (?:for|of)|chart (?:for|of)|file (?:for|of))\b",
    r"\b(?!(?:my|the|your|an?|this|that|our|what|there|here|it|he|she|who|where|when|how|let|clinic|office|doctor|dr|practice|pharmacy|hospital)\b)[a-z]+(?:\s+[a-z]+)?'s (?:appointment|records?|chart|medical|results?|file|insurance|phone|address|number)",
    r"\b(?:what|when|who|which)\b.{0,30}\b(?:appointment|record)s?\b.{0,25}\b(?:for|of|under)\s+(?!me\b|myself\b|my\b|us\b|the\b|a\b|an\b|next\b|this\b|tomorrow\b|today\b|monday\b|tuesday\b|wednesday\b|thursday\b|friday\b|saturday\b|sunday\b)[a-z]+\s+[a-z]+",
]
_MEDICAL = [
    r"what (?:medication|medicine|medicines|drug|drugs|pill|pills|dose|dosage|antibiotic|supplement|treatment|cream|ointment)s?\b.{0,40}\b(?:should|can|could|do|to)\b",
    r"(?:should|can|could) i (?:take|use|mix|combine|stop taking|skip|double|increase|decrease)\b",
    r"how (?:much|many)\b.{0,30}\b(?:ibuprofen|tylenol|acetaminophen|aspirin|advil|motrin|mg|pills?|tablets?|insulin|medication|dose)",
    r"\b(?:dose|dosage|dosing)\b.{0,20}\b(?:of|for)\b",
    r"(?:can|could|will|would) you (?:diagnose|tell me what (?:i have|is wrong|it is))",
    r"do i have\b.{0,30}\b(?:cancer|diabetes|infection|flu|covid|strep|disease|condition|syndrome|disorder|virus|cold|allergy|allergies|depression|anxiety)",
    r"what(?:'s| is) wrong with me", r"what do you think (?:it|this|my)\b.{0,30}\b(?:is|means|could be)",
    r"is (?:it|this|that|my)\b.{0,30}\b(?:serious|dangerous|normal|safe|contagious|cancer|infected|broken|a (?:rash|infection|tumou?r))",
    r"(?:treat|treatment|cure|remedy|remedies|home remedy)\b.{0,15}\b(?:for|to)\b",
    r"how (?:do|can|should) i (?:treat|cure|fix|heal|manage|get rid of|relieve)",
    r"(?:side effects?|interactions?|contraindications?)\b.{0,15}\b(?:of|with|between)\b",
    r"(?:interpret|explain|what does|what do) my\b.{0,25}\b(?:lab|test|blood|x-?ray|mri|scan|results?|symptoms?|rash|pain)",
    r"(?:is|are) (?:it|they|these|those) (?:ok|okay|fine|safe) to (?:take|mix|use|drink|eat)",
    r"diagnos(?:e|is) (?:me|my|this|these)", r"which (?:medication|medicine|drug|antibiotic)",
]
_PROFANITY = [
    r"\b(?:damn|dammit|goddamn|hell|crap|bullshit|shit|shitty|fuck\w*|bitch\w*|asshole|bastard|piss\w*|screw you|idiot|stupid|useless|moron|dumbass|jerk)\b",
    r"\b(?:shut up|piece of (?:crap|junk|garbage))\b",
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


_RX = {
    "emergency": _compile(_EMERGENCY),
    "threat": _compile(_THREAT),
    "sensitive_data": _compile(_SENSITIVE),
    "injection": _compile(_INJECTION),
    "cross_patient": _compile(_CROSS_PATIENT),
    "medical_advice": _compile(_MEDICAL),
    "profanity": _compile(_PROFANITY),
}
_ORDER = ["emergency", "threat", "sensitive_data", "injection", "cross_patient", "medical_advice", "profanity"]


@dataclass
class GuardVerdict:
    category: str = "ok"
    action: str = "allow"  # allow | block | escalate
    matched: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"}))
    t = re.sub(r"[^a-z' _\-\[\]<>/:.,?!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _luhn_ok(text: str) -> bool:
    for m in re.finditer(r"(?:\d[ -]?){13,19}", text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19:
            total, alt = 0, False
            for ch in reversed(digits):
                d = int(ch)
                if alt:
                    d = d * 2 - 9 if d * 2 > 9 else d * 2
                total += d
                alt = not alt
            if total % 10 == 0:
                return True
    return False


class InputGuard:
    """Deterministic, explainable first line of defence (runs before the LLM)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, text: str) -> GuardVerdict:
        if not self.enabled or not text.strip():
            return GuardVerdict()
        raw = unicodedata.normalize("NFKC", text)
        norm = normalize(text)
        for cat in _ORDER:
            # sensitive_data needs the raw digits; the rest use the normalised text
            subject = raw if cat == "sensitive_data" else norm
            for rx in _RX[cat]:
                m = rx.search(subject)
                if not m:
                    continue
                if cat == "sensitive_data" and rx.pattern.startswith("(?:\\d") and not _luhn_ok(raw):
                    continue
                action = "escalate" if cat in {"emergency", "threat"} else "block"
                if cat == "profanity":
                    action = "block"
                return GuardVerdict(cat, action, m.group(0)[:60])
        return GuardVerdict()


# ------------------------------------------------------------- confirmations

_AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|yea|correct|confirm(?:ed)?|sure|absolutely|definitely|affirmative|go ahead|do it|please do|that'?s right|that is right|proceed)\b",
    re.I,
)
_DENY = re.compile(
    r"\b(no|nope|nah|don'?t|do not|never ?mind|keep it|leave it|stop|wait|hold on|negative|not really|cancel that|changed my mind)\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(maybe|i guess|guess so|i think|not sure|unsure|perhaps|probably|possibly|um+|uh+|hmm+|kind of|sort of|depends|i suppose|whatever|dunno)\b",
    re.I,
)
_HUH = re.compile(r"\b(what|who|which one|huh|pardon|sorry)\b", re.I)


def classify_confirmation(text: str) -> str:
    """Classify a reply to a destructive-action confirmation prompt.

    ``yes`` / ``no`` only for short, unhedged, unambiguous answers. Anything contradictory or hedged
    is ``ambiguous`` (never executes). A long utterance with no yes/no token is a new request
    (``other``), which drops the pending action.
    """
    t = text.strip()
    words = t.split()
    a, d = bool(_AFFIRM.search(t)), bool(_DENY.search(t))
    hedge = bool(_HEDGE.search(t)) or (len(words) <= 3 and bool(_HUH.search(t)))
    if a and d:
        return "ambiguous"
    if not a and not d:
        return "ambiguous" if (hedge or len(words) <= 4) else "other"
    if hedge or "?" in t or len(words) > 8:
        return "ambiguous"
    return "yes" if a else "no"

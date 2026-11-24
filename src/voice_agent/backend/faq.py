"""Static clinic FAQ knowledge base with keyword routing."""

from __future__ import annotations

FAQ: dict[str, dict] = {
    "hours": {
        "keywords": ["hour", "open", "close", "closing", "opening", "when are you", "weekend", "saturday", "sunday", "holiday"],
        "answer": "We're open Monday through Friday from 9 AM to 5 PM, and Saturday from 9 AM to 1 PM. We're closed on Sundays.",
    },
    "location": {
        "keywords": ["where", "address", "location", "directions", "located", "find you"],
        "answer": "We're at 120 Maple Street, Suite 200, on the second floor above the pharmacy.",
    },
    "parking": {
        "keywords": ["park", "garage", "valet"],
        "answer": "Free parking is available in the lot behind the building. Accessible spaces are next to the rear entrance.",
    },
    "insurance": {
        "keywords": ["insurance", "insured", "coverage", "covered", "accept", "aetna", "cigna", "blue cross", "medicare", "medicaid", "united", "in-network", "in network", "plan"],
        "answer": "We accept Blue Cross Blue Shield, Aetna, Cigna, UnitedHealthcare, and Medicare. Please bring your insurance card and a photo ID to your visit.",
    },
    "billing": {
        "keywords": ["bill", "billing", "cost", "price", "copay", "co-pay", "pay", "payment", "fee", "self-pay", "charge"],
        "answer": "Copays are collected at check-in. For billing questions or payment plans, our billing office is available on weekdays from 9 AM to 4 PM.",
    },
    "new_patient": {
        "keywords": ["new patient", "first visit", "first time", "register", "registration", "paperwork", "forms"],
        "answer": "New patients should arrive 15 minutes early with a photo ID, insurance card, and a list of current medications. You can also complete forms ahead of time on our patient portal.",
    },
    "telehealth": {
        "keywords": ["telehealth", "virtual", "video visit", "online visit", "remote", "phone visit"],
        "answer": "We offer video visits for follow-ups and many routine concerns. Ask for a telehealth slot when booking.",
    },
    "prescriptions": {
        "keywords": ["refill", "prescription", "pharmacy", "renew"],
        "answer": "Refill requests can be sent through the patient portal or by asking our nurse line. Please allow two business days for processing.",
    },
}

TOPICS = tuple(FAQ)


def lookup(topic: str) -> dict | None:
    t = topic.lower().strip()
    if t in FAQ:
        return {"topic": t, "answer": FAQ[t]["answer"]}
    best, best_score = None, 0
    for name, entry in FAQ.items():
        score = sum(len(k) for k in entry["keywords"] if k in t)
        if name in t:
            score += 10
        if score > best_score:
            best, best_score = name, score
    if best:
        return {"topic": best, "answer": FAQ[best]["answer"]}
    return None

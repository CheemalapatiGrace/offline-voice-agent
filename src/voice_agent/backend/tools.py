"""Tool registry: typed argument schemas (pydantic, extra=forbid) and tool metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=60, pattern=r"^[A-Za-z][A-Za-z .'\-]*[A-Za-z.]$")]
TimeStr = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]


def _check_date(v: str) -> str:
    try:
        date.fromisoformat(v)
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD") from exc
    return v


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: Annotated[str, StringConstraints(max_length=64, pattern=r"^[\w\-:.]+$")] | None = None


class CheckAvailabilityArgs(_Args):
    date: str
    service: Annotated[str, StringConstraints(max_length=40)] | None = None

    _v = field_validator("date")(_check_date)


class BookArgs(_Args):
    patient_name: Name
    date: str
    time: TimeStr
    reason: Annotated[str, StringConstraints(max_length=120)] | None = None

    _v = field_validator("date")(_check_date)


class RescheduleArgs(_Args):
    appointment_id: Annotated[int, Field(ge=1, le=10_000_000)]
    patient_name: Name
    new_date: str
    new_time: TimeStr

    _v = field_validator("new_date")(_check_date)


class CancelArgs(_Args):
    appointment_id: Annotated[int, Field(ge=1, le=10_000_000)]
    patient_name: Name


class ListMineArgs(_Args):
    patient_name: Name


class FaqArgs(_Args):
    topic: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=80)]


class EscalateArgs(_Args):
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=200)]
    urgency: Literal["routine", "urgent", "emergency"] = "routine"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    mutating: bool = False
    destructive: bool = False  # requires explicit spoken confirmation
    sensitive: bool = False  # eligible for operator approval gating

    def json_schema(self) -> dict[str, Any]:
        schema = self.model.model_json_schema()
        schema.get("properties", {}).pop("idempotency_key", None)
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema["additionalProperties"] = False
        return schema


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("check_availability", "List open appointment slots (HH:MM) on a date (YYYY-MM-DD).", CheckAvailabilityArgs),
        ToolSpec("list_my_appointments", "List the caller's own upcoming appointments.", ListMineArgs),
        ToolSpec("book_appointment", "Book an appointment for the caller. Date YYYY-MM-DD, time HH:MM (24h).", BookArgs, mutating=True),
        ToolSpec("reschedule_appointment", "Move one of the caller's appointments to a new date and time.", RescheduleArgs, mutating=True),
        ToolSpec(
            "cancel_appointment",
            "Cancel one of the caller's appointments. Destructive: requires the caller's explicit spoken confirmation.",
            CancelArgs,
            mutating=True,
            destructive=True,
            sensitive=True,
        ),
        ToolSpec("lookup_faq", "Answer general clinic questions: hours, location, parking, insurance, billing, new_patient, telehealth, prescriptions.", FaqArgs),
        ToolSpec(
            "escalate_to_human",
            "Hand the call to a human staff member. Use for emergencies, complaints, requests for a person, or anything out of scope.",
            EscalateArgs,
            mutating=True,
            sensitive=True,
        ),
    ]
}

ALLOWLIST = set(TOOLS)


def llm_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI/Ollama style function schemas."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.json_schema()}}
        for t in TOOLS.values()
    ]


_NAME_RX = re.compile(Name.__metadata__[0].pattern)  # type: ignore[attr-defined]

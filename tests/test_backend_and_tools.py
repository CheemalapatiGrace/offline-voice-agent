import json

import pytest

from helpers import make_runtime, make_settings
from voice_agent.backend.store import slots_for
from voice_agent.context import SessionContext
from voice_agent.datatypes import ToolCall


@pytest.fixture
def store(rt):
    return rt.store


def test_clinic_hours_slots():
    from datetime import date

    assert slots_for(date(2026, 3, 2))[0] == "09:00" and slots_for(date(2026, 3, 2))[-1] == "16:30"
    assert slots_for(date(2026, 3, 7))[-1] == "12:30"  # Saturday
    assert slots_for(date(2026, 3, 8)) == []  # Sunday


def test_availability_excludes_booked_slots(store):
    free = store.availability("2026-03-03")["available"]
    assert "09:00" not in free and "10:30" in free


def test_book_and_double_booking_prevented_by_unique_index(store):
    assert store.book("Zed One", "2026-03-04", "13:00")["ok"]
    r = store.book("Zed Two", "2026-03-04", "13:00")
    assert r["error"] == "slot_unavailable" and r["alternatives"]


def test_booking_validation(store):
    assert store.book("A B", "2026-03-08", "10:00")["error"] == "closed"
    assert store.book("A B", "2026-03-04", "10:15")["error"] == "invalid_time"
    assert store.book("A B", "2026-02-01", "10:00")["error"] == "invalid_date"
    assert store.book("A B", "2026-12-01", "10:00")["error"] == "invalid_date"


def test_reschedule_and_cancel(store):
    r = store.reschedule(1, "2026-03-06", "16:00")
    assert r["ok"] and r["appointment"]["date"] == "2026-03-06"
    assert store.cancel(1)["ok"]
    assert store.cancel(1)["note"] == "already_cancelled"
    assert store.reschedule(1, "2026-03-06", "15:00")["error"] == "not_found"


def test_faq_lookup(store):
    assert "Aetna" in store.lookup_faq("do you take aetna")["answer"]
    assert store.lookup_faq("zzzz qqqq")["error"] == "not_found"


async def _exec(rt, ctx, name, args, **kw):
    return await rt.executor.execute(ctx, ToolCall(name, args), None, **kw)


async def test_schema_validation_rejects_bad_and_extra_args(rt):
    ctx = SessionContext()
    r = await _exec(rt, ctx, "book_appointment", {"patient_name": "A B", "date": "tomorrow", "time": "25:00"})
    assert r.status == "denied" and "invalid_arguments" in r.error
    r = await _exec(rt, ctx, "cancel_appointment", {"appointment_id": 1, "patient_name": "Alice Johnson", "confirmed": True})
    assert r.status == "denied" and "confirmed" in r.error
    r = await _exec(rt, ctx, "book_appointment", {"patient_name": "Alice'; DROP TABLE appointments;--", "date": "2026-03-04", "time": "10:00"})
    assert r.status == "denied"
    assert rt.store.snapshot()  # table still there


async def test_allowlist_blocks_unknown_tools(rt):
    r = await _exec(rt, SessionContext(), "drop_database", {})
    assert r.status == "denied" and r.error == "tool_not_allowed"


async def test_destructive_action_needs_confirmation_then_executes(rt):
    ctx = SessionContext()
    r = await _exec(rt, ctx, "cancel_appointment", {"appointment_id": 2, "patient_name": "Bob Martinez"})
    assert r.status == "needs_confirmation" and ctx.pending_confirm
    assert rt.store.get(2)["status"] == "booked"
    r = await _exec(rt, ctx, "cancel_appointment", {"appointment_id": 2, "patient_name": "Bob Martinez"}, confirmed=True)
    assert r.ok and rt.store.get(2)["status"] == "cancelled"


async def test_ownership_and_identity_binding(rt):
    ctx = SessionContext()
    assert (await _exec(rt, ctx, "list_my_appointments", {"patient_name": "Alice Johnson"})).ok
    r = await _exec(rt, ctx, "cancel_appointment", {"appointment_id": 2, "patient_name": "Alice Johnson"})
    assert r.status == "denied" and r.error == "appointment_not_found_for_caller"
    r = await _exec(rt, ctx, "list_my_appointments", {"patient_name": "Bob Martinez"})
    assert r.status == "denied" and r.error == "identity_mismatch"
    assert rt.store.get(2)["status"] == "booked"


async def test_rate_limit_on_mutations():
    rt = make_runtime(make_settings(rate_limit_mutations_per_min=2))
    ctx = SessionContext()
    for d, expected in zip(["2026-03-12", "2026-03-13", "2026-03-16"], ["ok", "ok", "denied"], strict=True):
        r = await _exec(rt, ctx, "book_appointment", {"patient_name": "Rate Limit", "date": d, "time": "09:00"})
        assert r.status == expected
    assert r.error == "rate_limited"
    await rt.stop()


async def test_idempotency_key_replays_without_double_booking(rt):
    ctx = SessionContext()
    args = {"patient_name": "Idem Potent", "date": "2026-03-10", "time": "10:00", "idempotency_key": "abc-1"}
    first = await _exec(rt, ctx, "book_appointment", args)
    second = await _exec(rt, ctx, "book_appointment", args)
    assert first.ok and second.ok and second.replayed
    assert first.data["appointment"]["id"] == second.data["appointment"]["id"]
    assert len([v for v in rt.store.snapshot().values() if v[0] == "idem potent"]) == 1


async def test_derived_idempotency_key_dedupes_retries_within_a_turn(rt):
    ctx = SessionContext()
    ctx.turn_no = 3
    args = {"patient_name": "Retry Safe", "date": "2026-03-10", "time": "11:00"}
    a, b = await _exec(rt, ctx, "book_appointment", args), await _exec(rt, ctx, "book_appointment", args)
    assert a.ok and b.replayed


async def test_escalation_creates_ticket_and_is_idempotent(rt):
    ctx = SessionContext()
    r = await _exec(rt, ctx, "escalate_to_human", {"reason": "wants a person", "urgency": "urgent"})
    assert r.ok and ctx.escalated
    again = await _exec(rt, ctx, "escalate_to_human", {"reason": "again", "urgency": "urgent"})
    assert again.data.get("note") == "already_escalated"
    assert len(rt.store.handoffs()) == 1


def test_tool_message_payload_is_json_serialisable():
    from voice_agent.datatypes import ToolResult

    assert json.loads(json.dumps(ToolResult("ok", {"a": 1}).for_llm()))["status"] == "ok"

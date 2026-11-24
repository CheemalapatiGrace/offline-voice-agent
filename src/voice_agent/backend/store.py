"""SQLite-backed appointment 'EHR lite'. All methods are synchronous and thread-safe."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from . import faq
from .db import Database

MAX_DAYS_AHEAD = 90


def name_key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", name.lower())).strip()


def slots_for(d: date) -> list[str]:
    wd = d.weekday()
    if wd <= 4:
        start, end = 9 * 60, 17 * 60
    elif wd == 5:
        start, end = 9 * 60, 13 * 60
    else:
        return []
    return [f"{m // 60:02d}:{m % 60:02d}" for m in range(start, end, 30)]


def _row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "patient_name": r["patient_name"],
        "date": r["date"],
        "time": r["time"],
        "reason": r["reason"],
        "status": r["status"],
    }


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


class AppointmentStore:
    def __init__(self, db: Database, today: Callable[[], date] = date.today) -> None:
        self.db = db
        self.today = today

    # ---------------------------------------------------------------- reads
    def availability(self, d: str) -> dict[str, Any]:
        day = self._parse_date(d)
        if isinstance(day, dict):
            return day
        all_slots = slots_for(day)
        if not all_slots:
            return _err("closed", "The clinic is closed on that day.", date=d)
        taken = {r["time"] for r in self.db.query("SELECT time FROM appointments WHERE date=? AND status='booked'", (d,))}
        free = [s for s in all_slots if s not in taken]
        if day == self.today():
            now = datetime.now().strftime("%H:%M") if not self._pinned() else "00:00"
            free = [s for s in free if s > now]
        return {"ok": True, "date": d, "available": free}

    def _pinned(self) -> bool:
        return self.today() != date.today()

    def get(self, appt_id: int) -> dict[str, Any] | None:
        rows = self.db.query("SELECT * FROM appointments WHERE id=?", (appt_id,))
        return _row(rows[0]) | {"patient_key": rows[0]["patient_key"]} if rows else None

    def list_for(self, patient_name: str) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT * FROM appointments WHERE patient_key=? AND status='booked' AND date>=? ORDER BY date,time",
            (name_key(patient_name), self.today().isoformat()),
        )
        return {"ok": True, "appointments": [_row(r) for r in rows]}

    def all_patient_names(self) -> list[str]:
        return [r["patient_name"] for r in self.db.query("SELECT DISTINCT patient_name FROM appointments")]

    def snapshot(self) -> dict[int, tuple]:
        return {
            r["id"]: (r["patient_key"], r["date"], r["time"], r["status"])
            for r in self.db.query("SELECT * FROM appointments")
        }

    # --------------------------------------------------------------- writes
    def book(self, patient_name: str, d: str, t: str, reason: str | None = None) -> dict[str, Any]:
        err = self._validate_slot(d, t)
        if err:
            return err
        try:
            with self.db.transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO appointments(patient_name, patient_key, date, time, reason) VALUES (?,?,?,?,?)",
                    (patient_name.strip(), name_key(patient_name), d, t, reason),
                )
                appt_id = cur.lastrowid
        except sqlite3.IntegrityError:
            return self._unavailable(d, t)
        return {"ok": True, "appointment": self.get_public(appt_id)}

    def reschedule(self, appt_id: int, d: str, t: str) -> dict[str, Any]:
        appt = self.get(appt_id)
        if not appt or appt["status"] != "booked":
            return _err("not_found", "No active appointment with that id.")
        err = self._validate_slot(d, t)
        if err:
            return err
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE appointments SET date=?, time=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (d, t, appt_id),
                )
        except sqlite3.IntegrityError:
            return self._unavailable(d, t)
        return {"ok": True, "appointment": self.get_public(appt_id), "previous": {"date": appt["date"], "time": appt["time"]}}

    def cancel(self, appt_id: int) -> dict[str, Any]:
        appt = self.get(appt_id)
        if not appt:
            return _err("not_found", "No appointment with that id.")
        if appt["status"] == "cancelled":
            return {"ok": True, "appointment": self.get_public(appt_id), "note": "already_cancelled"}
        self.db.execute(
            "UPDATE appointments SET status='cancelled', updated_at=CURRENT_TIMESTAMP WHERE id=?", (appt_id,)
        )
        return {"ok": True, "appointment": self.get_public(appt_id)}

    def get_public(self, appt_id: int) -> dict[str, Any]:
        a = self.get(appt_id) or {}
        a.pop("patient_key", None)
        return a

    # ---------------------------------------------------------------- faq / handoff
    def lookup_faq(self, topic: str) -> dict[str, Any]:
        hit = faq.lookup(topic)
        if not hit:
            return _err("not_found", "I don't have information on that topic.", topics=list(faq.TOPICS))
        return {"ok": True, **hit}

    def create_handoff(self, session_id: str, reason: str, urgency: str) -> dict[str, Any]:
        cur = self.db.execute(
            "INSERT INTO handoffs(session_id, reason, urgency) VALUES (?,?,?)", (session_id, reason, urgency)
        )
        return {"ok": True, "ticket_id": cur.lastrowid, "urgency": urgency}

    def handoffs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM handoffs ORDER BY id")]

    # ---------------------------------------------------------------- idempotency
    def idem_get(self, key: str) -> dict[str, Any] | None:
        rows = self.db.query("SELECT result_json FROM idempotency WHERE key=?", (key,))
        return json.loads(rows[0]["result_json"]) if rows else None

    def idem_put(self, key: str, tool: str, result: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO idempotency(key, tool, result_json) VALUES (?,?,?)",
            (key, tool, json.dumps(result)),
        )

    # ---------------------------------------------------------------- helpers
    def _parse_date(self, d: str) -> date | dict[str, Any]:
        try:
            day = date.fromisoformat(d)
        except ValueError:
            return _err("invalid_date", "Dates must look like YYYY-MM-DD.")
        today = self.today()
        if day < today:
            return _err("invalid_date", "That date has already passed.")
        if day > today + timedelta(days=MAX_DAYS_AHEAD):
            return _err("invalid_date", f"We only book up to {MAX_DAYS_AHEAD} days ahead.")
        return day

    def _validate_slot(self, d: str, t: str) -> dict[str, Any] | None:
        day = self._parse_date(d)
        if isinstance(day, dict):
            return day
        slots = slots_for(day)
        if not slots:
            return _err("closed", "The clinic is closed on that day.", date=d)
        if t not in slots:
            return _err("invalid_time", "That time is outside clinic hours or not on a 30 minute boundary.", date=d)
        return None

    def _unavailable(self, d: str, t: str) -> dict[str, Any]:
        free = self.availability(d).get("available", [])
        free.sort(key=lambda s: abs(int(s[:2]) * 60 + int(s[3:]) - (int(t[:2]) * 60 + int(t[3:]))))
        return _err("slot_unavailable", "That time is already taken.", alternatives=sorted(free[:3]))


def seed_demo(store: AppointmentStore) -> None:
    """Deterministic demo data relative to ``store.today()``."""

    def nxt(weekday: int, extra_weeks: int = 0) -> str:
        t = store.today()
        delta = (weekday - t.weekday()) % 7 or 7
        return (t + timedelta(days=delta + 7 * extra_weeks)).isoformat()

    rows = [
        ("Alice Johnson", nxt(2), "10:00", "annual checkup"),
        ("Bob Martinez", nxt(3), "14:00", "follow-up"),
        ("Carol Nguyen", nxt(4), "09:30", "flu shot"),
        ("Dan Okafor", nxt(1), "11:00", "blood pressure check"),
        ("Evelyn Park", nxt(0), "15:00", "consultation"),
        ("Evelyn Park", nxt(2, 1), "09:00", "follow-up"),
        ("Frank Lee", nxt(1), "09:00", "physical"),
        ("Frank Lee", nxt(1), "09:30", "physical"),
        ("Frank Lee", nxt(1), "10:00", "physical"),
    ]
    for name, d, t, reason in rows:
        store.book(name, d, t, reason)

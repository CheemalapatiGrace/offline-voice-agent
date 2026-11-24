"""Per-turn structured traces (spans: vad, stt, guardrail, llm, tool, tts) persisted to SQLite.

Every string that reaches storage or the event bus passes through the PII redactor first.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..backend.db import Database
from ..guardrails.pii import redact, redact_obj
from .events import EventBus

SPAN_NAMES = ("vad", "stt", "guardrail", "llm", "tool", "tts")


@dataclass
class Span:
    name: str
    start_ms: float
    dur_ms: float
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 2),
            "dur_ms": round(self.dur_ms, 2),
            "status": self.status,
            "attrs": self.attrs,
        }


class TurnTrace:
    def __init__(self, session_id: str, turn_id: int, t0: float | None = None, names: list[str] | None = None):
        self.session_id = session_id
        self.turn_id = turn_id
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.started_wall = time.time()
        self.spans: list[Span] = []
        self.user_text = ""
        self.reply_text = ""
        self.outcome = "ok"
        self.ttfa_ms: float | None = None
        self.total_ms = 0.0
        self.attrs: dict[str, Any] = {}
        self._names = names or []

    def add_names(self, *names: str) -> None:
        self._names.extend(names)

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def record(self, span_name: str, dur_ms: float, /, *, start_ms: float | None = None, status: str = "ok", **attrs: Any) -> Span:
        start = start_ms if start_ms is not None else max(0.0, self.now_ms() - dur_ms)
        span = Span(span_name, start, dur_ms, status, redact_obj(attrs, self._names))
        self.spans.append(span)
        return span

    @contextmanager
    def span(self, span_name: str, /, **attrs: Any):
        start = self.now_ms()
        t = time.perf_counter()
        holder = {"status": "ok", "attrs": dict(attrs)}
        try:
            yield holder["attrs"]
        except BaseException:
            holder["status"] = "error"
            raise
        finally:
            self.record(span_name, (time.perf_counter() - t) * 1000, start_ms=start, status=holder["status"], **holder["attrs"])

    def mark_first_audio(self) -> None:
        if self.ttfa_ms is None:
            self.ttfa_ms = self.now_ms()

    def span_ms(self, span_name: str) -> float:
        return sum(s.dur_ms for s in self.spans if s.name == span_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "ts": self.started_wall,
            "user_text": redact(self.user_text, self._names),
            "reply_text": redact(self.reply_text, self._names),
            "outcome": self.outcome,
            "ttfa_ms": None if self.ttfa_ms is None else round(self.ttfa_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "spans": [s.as_dict() for s in self.spans],
            "attrs": redact_obj(self.attrs, self._names),
        }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class TraceStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, trace: TurnTrace) -> int:
        d = trace.as_dict()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO turns(session_id, turn_id, ts, user_text, reply_text, ttfa_ms, total_ms, outcome, trace_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (d["session_id"], d["turn_id"], d["ts"], d["user_text"], d["reply_text"], d["ttfa_ms"], d["total_ms"], d["outcome"], json.dumps(d)),
            )
            pk = cur.lastrowid
            conn.executemany(
                "INSERT INTO spans(turn_pk, session_id, name, start_ms, dur_ms, status, attrs_json) VALUES (?,?,?,?,?,?,?)",
                [(pk, d["session_id"], s["name"], s["start_ms"], s["dur_ms"], s["status"], json.dumps(s["attrs"])) for s in d["spans"]],
            )
        return pk

    def turns(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if session_id:
            rows = self.db.query("SELECT trace_json FROM turns WHERE session_id=? ORDER BY pk DESC LIMIT ?", (session_id, limit))
        else:
            rows = self.db.query("SELECT trace_json FROM turns ORDER BY pk DESC LIMIT ?", (limit,))
        return [json.loads(r["trace_json"]) for r in reversed(rows)]

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"spans": {}}
        for name in SPAN_NAMES:
            vals = [r["dur_ms"] for r in self.db.query("SELECT dur_ms FROM spans WHERE name=?", (name,))]
            if vals:
                out["spans"][name] = {"n": len(vals), "p50": round(_pct(vals, 0.5), 2), "p95": round(_pct(vals, 0.95), 2)}
        ttfa = [r["ttfa_ms"] for r in self.db.query("SELECT ttfa_ms FROM turns WHERE ttfa_ms IS NOT NULL")]
        if ttfa:
            out["ttfa"] = {"n": len(ttfa), "p50": round(_pct(ttfa, 0.5), 2), "p95": round(_pct(ttfa, 0.95), 2)}
        out["turns"] = self.db.query("SELECT COUNT(*) c FROM turns")[0]["c"]
        return out


class Tracer:
    def __init__(self, store: TraceStore, bus: EventBus) -> None:
        self.store = store
        self.bus = bus

    def start_turn(self, session_id: str, turn_id: int, t0: float | None = None, names: list[str] | None = None) -> TurnTrace:
        return TurnTrace(session_id, turn_id, t0, names)

    def finish(self, trace: TurnTrace) -> None:
        trace.total_ms = trace.now_ms()
        try:
            self.store.save(trace)
        except Exception:  # tracing must never break a call
            from .logging import get_logger

            get_logger("tracing").exception("trace persistence failed")
        self.bus.publish("turn_trace", trace.session_id, trace=trace.as_dict())

"""Per-session mutable state and the session state machine."""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .datatypes import SessionState, ToolCall

S = SessionState

TRANSITIONS: dict[SessionState, set[SessionState]] = {
    S.CONNECTING: {S.IDLE, S.CLOSED},
    S.IDLE: {S.LISTENING, S.THINKING, S.SPEAKING, S.PAUSED, S.TAKEOVER, S.HANDOFF, S.DISCONNECTED, S.CLOSED},
    S.LISTENING: {S.IDLE, S.THINKING, S.SPEAKING, S.PAUSED, S.TAKEOVER, S.HANDOFF, S.DISCONNECTED, S.CLOSED},
    S.THINKING: {S.IDLE, S.LISTENING, S.SPEAKING, S.PAUSED, S.TAKEOVER, S.HANDOFF, S.DISCONNECTED, S.CLOSED},
    S.SPEAKING: {S.IDLE, S.LISTENING, S.THINKING, S.PAUSED, S.TAKEOVER, S.HANDOFF, S.DISCONNECTED, S.CLOSED},
    S.PAUSED: {S.IDLE, S.TAKEOVER, S.HANDOFF, S.SPEAKING, S.DISCONNECTED, S.CLOSED},
    S.TAKEOVER: {S.IDLE, S.PAUSED, S.HANDOFF, S.SPEAKING, S.DISCONNECTED, S.CLOSED},
    S.HANDOFF: {S.IDLE, S.TAKEOVER, S.PAUSED, S.SPEAKING, S.DISCONNECTED, S.CLOSED},
    S.DISCONNECTED: {S.IDLE, S.PAUSED, S.TAKEOVER, S.HANDOFF, S.CLOSED},
    S.CLOSED: set(),
}

MUTED_STATES = {S.PAUSED, S.TAKEOVER}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class PendingConfirm:
    call: ToolCall
    summary: str
    turns: int = 0
    ambiguous: int = 0


@dataclass
class PendingReply:
    id: str
    text: str
    created: float = field(default_factory=time.time)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    decision: str = ""  # release | discard | timeout
    edited: str | None = None


class SessionContext:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.resume_token = secrets.token_urlsafe(16)
        self.created_at = time.time()
        self.last_activity = time.monotonic()
        self.state = SessionState.CONNECTING
        self.state_before_disconnect: SessionState | None = None
        self.history: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.patient_name: str | None = None
        self.patient_key: str | None = None
        self.pending_confirm: PendingConfirm | None = None
        self.pending_reply: PendingReply | None = None
        self.review_replies = False
        self.violations = 0
        self.abuse_strikes = 0
        self.guard_events: list[dict[str, Any]] = []
        self.escalated = False
        self.tool_log: list[dict[str, Any]] = []
        self.tool_times: deque[float] = deque()
        self.mutation_times: deque[float] = deque()
        self.turn_no = 0
        self.last_reply = ""
        self.reply_delivered = True
        self.activity = ""  # sub-status shown on the dashboard
        self.degraded: set[str] = set()
        self.stt_failures = 0
        self.turn_flags: set[str] = set()
        self.spoken_names: set[str] = set()

    @property
    def muted(self) -> bool:
        return self.state in MUTED_STATES

    def names(self) -> list[str]:
        """Every name this caller has used; fed to the PII redactor."""
        found = set(self.spoken_names)
        if self.patient_name:
            found.add(self.patient_name)
        return sorted(found)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def can_transition(self, new: SessionState) -> bool:
        return new == self.state or new in TRANSITIONS[self.state]

    def transition(self, new: SessionState) -> SessionState:
        old = self.state
        if new == old:
            return old
        if new not in TRANSITIONS[old]:
            raise InvalidTransition(f"{old.value} -> {new.value}")
        self.state = new
        return old

    def add_transcript(self, role: str, text: str, **extra: Any) -> dict[str, Any]:
        from .guardrails.pii import redact

        item = {"role": role, "text": redact(text, self.names()), "ts": time.time(), **extra}
        self.transcript.append(item)
        del self.transcript[:-200]
        return item

    def trim_history(self, keep: int = 40) -> None:
        if len(self.history) > keep:
            del self.history[: len(self.history) - keep]
            while self.history and self.history[0]["role"] == "tool":
                self.history.pop(0)

"""Session manager: capacity limits, reconnect/recovery, idle reaping."""

from __future__ import annotations

import asyncio
import hmac
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .adapters import Adapters
from .agent import Agent
from .config import Settings
from .context import SessionContext
from .datatypes import SessionState, ToolResult
from .observability.events import EventBus
from .observability.logging import get_logger
from .observability.tracing import Tracer
from .pipeline import Sink, VoicePipeline

log = get_logger("session")
S = SessionState


class CapacityError(RuntimeError):
    """Raised when ``max_sessions`` live sessions already exist."""


@dataclass
class Session:
    ctx: SessionContext
    pipeline: VoicePipeline
    disconnected_at: float | None = None
    closed_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    def __init__(self, settings: Settings, adapters: Adapters, agent: Agent, tracer: Tracer, bus: EventBus) -> None:
        self.settings, self.adapters, self.agent, self.tracer, self.bus = settings, adapters, agent, tracer, bus
        self._sessions: dict[str, Session] = {}
        self._closed: deque[dict[str, Any]] = deque(maxlen=50)
        self._reaper: asyncio.Task | None = None
        self.rejected = 0

    # ------------------------------------------------------------- lookups
    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def active(self) -> list[Session]:
        return list(self._sessions.values())

    def snapshots(self) -> list[dict[str, Any]]:
        return [{**s.pipeline.snapshot(), "disconnected": s.disconnected_at is not None} for s in self._sessions.values()]

    def closed_sessions(self) -> list[dict[str, Any]]:
        return list(self._closed)

    def stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.ctx.state.value] = by_state.get(s.ctx.state.value, 0) + 1
        return {"active": len(self._sessions), "max": self.settings.max_sessions, "by_state": by_state, "rejected": self.rejected}

    # ------------------------------------------------------------ lifecycle
    async def create(self, sink: Sink | None = None) -> Session:
        if len(self._sessions) >= self.settings.max_sessions:
            self.rejected += 1
            self.bus.publish("capacity_rejected", None, active=len(self._sessions))
            raise CapacityError(f"max_sessions={self.settings.max_sessions} reached")
        ctx = SessionContext()
        pipeline = VoicePipeline(
            ctx, settings=self.settings, adapters=self.adapters, agent=self.agent, tracer=self.tracer, bus=self.bus
        )
        session = Session(ctx, pipeline)
        self._sessions[ctx.session_id] = session
        await pipeline.start(sink)
        pipeline._set_state(S.IDLE, quiet=True)
        self.bus.publish("session_opened", ctx.session_id, created_at=ctx.created_at)
        pipeline._emit_nowait(self._hello(session, resumed=False))
        return session

    def _hello(self, s: Session, resumed: bool) -> dict[str, Any]:
        return {
            "type": "session",
            "session_id": s.ctx.session_id,
            "resume_token": s.ctx.resume_token,
            "state": s.ctx.state.value,
            "resumed": resumed,
            "input_sample_rate": self.settings.sample_rate,
            "output_sample_rate": getattr(self.adapters.tts, "sample_rate", 16000),
            "frame_ms": self.settings.frame_ms,
        }

    async def disconnect(self, session: Session) -> None:
        """The socket dropped. Keep the session for ``reconnect_grace_s`` so the caller can resume."""
        ctx = session.ctx
        if ctx.state in (S.CLOSED, S.DISCONNECTED):
            return
        ctx.state_before_disconnect = ctx.state
        session.pipeline.detach()
        session.pipeline._set_state(S.DISCONNECTED)
        session.disconnected_at = time.monotonic()
        self.bus.publish("session_disconnected", ctx.session_id, grace_s=self.settings.reconnect_grace_s)

    async def resume(self, session_id: str, token: str, sink: Sink) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None or not hmac.compare_digest(token.encode(), session.ctx.resume_token.encode()):
            return None
        ctx = session.ctx
        if ctx.state != S.DISCONNECTED:  # old socket not yet noticed as dead: steal the connection
            session.pipeline.detach()
        prev = ctx.state_before_disconnect or S.IDLE
        target = prev if prev in (S.PAUSED, S.TAKEOVER, S.HANDOFF) else S.IDLE
        session.pipeline._set_state(target, quiet=True)
        session.disconnected_at = None
        await session.pipeline.attach(sink)
        hello = self._hello(session, resumed=True)
        hello["transcript"] = ctx.transcript[-10:]
        hello["last_reply"] = ctx.last_reply
        session.pipeline._emit_nowait(hello)
        if not ctx.reply_delivered and ctx.last_reply:
            await session.pipeline.system_say("Sorry about that. " + ctx.last_reply)
        self.bus.publish("session_resumed", ctx.session_id)
        return session

    async def close(self, session_id: str, reason: str = "closed") -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        ctx = session.ctx
        session.closed_reason = reason
        try:
            ctx.transition(S.CLOSED)
        except Exception:  # noqa: BLE001
            ctx.state = S.CLOSED
        await session.pipeline.stop()
        self._closed.append({"session_id": ctx.session_id, "reason": reason, "turns": ctx.turn_no, "closed_at": time.time(), "escalated": ctx.escalated})
        self.bus.publish("session_closed", ctx.session_id, reason=reason)

    async def shutdown(self) -> None:
        if self._reaper:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
        for sid in list(self._sessions):
            await self.close(sid, "shutdown")

    # --------------------------------------------------------------- reaping
    async def reap_once(self, now: float | None = None) -> list[str]:
        now = now if now is not None else time.monotonic()
        victims: list[tuple[str, str]] = []
        for sid, s in self._sessions.items():
            if s.disconnected_at is not None and now - s.disconnected_at > self.settings.reconnect_grace_s:
                victims.append((sid, "reconnect_grace_expired"))
            elif s.ctx.state in (S.IDLE, S.HANDOFF) and now - s.ctx.last_activity > self.settings.idle_timeout_s:
                victims.append((sid, "idle_timeout"))
        for sid, why in victims:
            await self.close(sid, why)
        return [v[0] for v in victims]

    def start_reaper(self, interval_s: float = 5.0) -> None:
        async def loop() -> None:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    await self.reap_once()
                except Exception:  # noqa: BLE001
                    log.exception("reaper failed")

        self._reaper = asyncio.create_task(loop(), name="session-reaper")

    # ---------------------------------------------------- late approval notice
    async def notify_late(self, session_id: str, tool: str, result: ToolResult) -> None:
        session = self.get(session_id)
        if session is None or session.ctx.muted:
            return
        if tool == "cancel_appointment":
            text = (
                "Good news: our staff approved your cancellation and it is done."
                if result.ok
                else "Our staff couldn't approve that cancellation. A team member will follow up with you."
            )
            await session.pipeline.system_say(text)

"""FastAPI app: caller WebSocket, operator REST + live dashboard WebSocket, health endpoints."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import __version__
from .adapters import Adapters
from .audio import frame_bytes, split_frames
from .config import Settings
from .guardrails.pii import redact_obj
from .hitl import OperatorError
from .observability.logging import configure_logging, get_logger
from .runtime import Runtime, build_runtime
from .session import CapacityError, Session

log = get_logger("server")
STATIC = Path(__file__).parent / "static"
NO_STORE = {"Cache-Control": "no-store"}


class WSSink:
    realtime = True

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws

    async def send_json(self, msg: dict[str, Any]) -> None:
        await self.ws.send_json(msg)

    async def send_audio(self, data: bytes) -> None:
        await self.ws.send_bytes(data)


class SayBody(BaseModel):
    text: str = Field(min_length=1, max_length=600)


class ReviewBody(BaseModel):
    enabled: bool


class ReleaseBody(BaseModel):
    text: str | None = Field(default=None, max_length=600)


class DecideBody(BaseModel):
    note: str = Field(default="", max_length=200)


def create_app(settings: Settings | None = None, adapters: Adapters | None = None, runtime: Runtime | None = None) -> FastAPI:
    rt = runtime or build_runtime(settings, adapters)
    settings = rt.settings
    started = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await rt.start()
        yield
        await rt.stop()

    app = FastAPI(title="offline-voice-agent", version=__version__, lifespan=lifespan)
    app.state.runtime = rt

    # ------------------------------------------------------------------ auth
    def operator(authorization: str | None = Header(default=None), x_operator: str | None = Header(default=None)) -> str:
        if settings.operator_token:
            supplied = (authorization or "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied.encode(), settings.operator_token.encode()):
                raise HTTPException(401, "operator token required")
        return (x_operator or "operator")[:40]

    def guarded(session_id: str) -> Session:
        s = rt.manager.get(session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.exception_handler(OperatorError)
    async def _op_error(_: Request, exc: OperatorError):
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)

    # ---------------------------------------------------------------- health
    @app.get("/health")
    async def health() -> dict[str, Any]:
        llm = rt.adapters.llm
        degraded: list[str] = []
        db_ok = rt.db.ping()
        if not db_ok:
            degraded.append("database")
        breakers = getattr(llm, "breakers", {})
        health_llm = {name: b.state for name, b in breakers.items()}
        if any(v != "closed" for v in health_llm.values()):
            degraded.append("llm")
        stats = rt.manager.stats()
        if stats["active"] >= stats["max"]:
            degraded.append("at_capacity")
        return {
            "status": "degraded" if degraded else "ok",
            "version": __version__,
            "uptime_s": round(time.time() - started, 1),
            "degraded": degraded,
            "sessions": stats,
            "backends": {
                "stt": getattr(rt.adapters.stt, "name", "?"),
                "llm": [c.name for c in getattr(llm, "chain", [llm])],
                "llm_breakers": health_llm,
                "tts": getattr(rt.adapters.tts, "name", "?"),
            },
            "database": "ok" if db_ok else "down",
            "event_subscribers": rt.bus.subscribers,
        }

    @app.get("/ready")
    async def ready():
        ok = rt.db.ping() and len(rt.manager.active()) < settings.max_sessions
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    # ---------------------------------------------------------------- static
    @app.get("/")
    async def root():
        return RedirectResponse("/dashboard")

    @app.get("/dashboard")
    async def dashboard():
        return FileResponse(STATIC / "dashboard.html", headers=NO_STORE)

    @app.get("/mic")
    async def mic():
        return FileResponse(STATIC / "mic.html", headers=NO_STORE)

    # ---------------------------------------------------------- caller socket
    @app.websocket("/ws/session")
    async def ws_session(ws: WebSocket, session_id: str | None = None, resume_token: str | None = None):
        await ws.accept()
        sink = WSSink(ws)
        session: Session | None = None
        try:
            if session_id and resume_token:
                session = await rt.manager.resume(session_id, resume_token, sink)
                if session is None:
                    await ws.send_json({"type": "notice", "code": "resume_failed", "message": "starting a new session"})
            if session is None:
                session = await rt.manager.create(sink)
        except CapacityError:
            await ws.send_json({"type": "error", "code": "capacity", "message": "server at capacity, retry shortly"})
            await ws.close(code=1013)
            return
        pipeline = session.pipeline
        fb = frame_bytes(settings.sample_rate, settings.frame_ms)
        closed_by_client = False
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if (data := msg.get("bytes")) is not None:
                    if len(data) > settings.max_ws_message_bytes:
                        await ws.send_json({"type": "error", "code": "frame_too_large"})
                        continue
                    for frame in ([data] if len(data) == fb else split_frames(data, settings.sample_rate, settings.frame_ms)):
                        await pipeline.push_audio(frame)
                elif (raw := msg.get("text")) is not None:
                    if len(raw) > 4096:
                        await ws.send_json({"type": "error", "code": "message_too_large"})
                        continue
                    try:
                        cmd = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "code": "bad_json"})
                        continue
                    kind = cmd.get("type")
                    if kind == "text" and isinstance(cmd.get("text"), str):
                        await pipeline.push_text(cmd["text"][:1000])
                    elif kind == "interrupt":
                        await pipeline.barge_in()
                    elif kind == "ping":
                        await ws.send_json({"type": "pong", "ts": time.time()})
                    elif kind == "close":
                        closed_by_client = True
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if closed_by_client:
                await rt.manager.close(session.ctx.session_id, "client_closed")
            else:
                await rt.manager.disconnect(session)

    # ------------------------------------------------------ operator REST API
    @app.get("/api/sessions")
    async def sessions(_: str = Depends(operator)):
        return {"sessions": rt.manager.snapshots(), "closed": rt.manager.closed_sessions(), "stats": rt.manager.stats()}

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str, _: str = Depends(operator)):
        s = guarded(session_id)
        events = rt.bus.recent(session_id, 300)
        return {
            "snapshot": s.pipeline.snapshot(),
            "transcript": s.ctx.transcript[-100:],
            "tool_calls": [e for e in events if e["type"] == "tool_call"][-30:],
            "guardrails": [e for e in events if e["type"] == "guardrail"][-30:],
            "turns": rt.traces.turns(session_id, 20),
        }

    @app.get("/api/sessions/{session_id}/turns")
    async def session_turns(session_id: str, limit: int = Query(50, le=500), _: str = Depends(operator)):
        return {"turns": rt.traces.turns(session_id, limit)}

    @app.post("/api/sessions/{session_id}/pause")
    async def pause(session_id: str, op: str = Depends(operator)):
        await rt.operator.pause(session_id, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/resume")
    async def resume(session_id: str, op: str = Depends(operator)):
        await rt.operator.resume(session_id, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/takeover")
    async def takeover(session_id: str, op: str = Depends(operator)):
        await rt.operator.takeover(session_id, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/release")
    async def release(session_id: str, op: str = Depends(operator)):
        await rt.operator.release(session_id, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/say")
    async def say(session_id: str, body: SayBody, op: str = Depends(operator)):
        await rt.operator.say(session_id, body.text, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/review")
    async def review(session_id: str, body: ReviewBody, op: str = Depends(operator)):
        await rt.operator.set_review(session_id, body.enabled, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/reply/release")
    async def reply_release(session_id: str, body: ReleaseBody, op: str = Depends(operator)):
        await rt.operator.release_reply(session_id, body.text, op)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/reply/discard")
    async def reply_discard(session_id: str, op: str = Depends(operator)):
        await rt.operator.discard_reply(session_id, op)
        return {"ok": True}

    @app.get("/api/approvals")
    async def approvals(status: str = "pending", _: str = Depends(operator)):
        return {"approvals": rt.approvals.list(status)}

    @app.post("/api/approvals/{approval_id}/approve")
    async def approve(approval_id: int, body: DecideBody | None = None, op: str = Depends(operator)):
        return await rt.approvals.decide(approval_id, True, op, body.note if body else "")

    @app.post("/api/approvals/{approval_id}/reject")
    async def reject(approval_id: int, body: DecideBody | None = None, op: str = Depends(operator)):
        return await rt.approvals.decide(approval_id, False, op, body.note if body else "")

    @app.get("/api/audit")
    async def audit(limit: int = Query(100, le=500), _: str = Depends(operator)):
        return {"audit": rt.operator.audit_log(limit)}

    @app.get("/api/stats")
    async def stats(_: str = Depends(operator)):
        return rt.traces.stats()

    @app.get("/api/handoffs")
    async def handoffs(_: str = Depends(operator)):
        return {"handoffs": [redact_obj(h) for h in rt.store.handoffs()]}

    # --------------------------------------------------------- dashboard socket
    @app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket, token: str = ""):
        if settings.operator_token and not hmac.compare_digest(token.encode(), settings.operator_token.encode()):
            await ws.close(code=4401)
            return
        await ws.accept()
        sub = rt.bus.subscribe()
        try:
            await ws.send_json(
                {
                    "type": "snapshot",
                    "sessions": rt.manager.snapshots(),
                    "approvals": rt.approvals.list("pending"),
                    "audit": rt.operator.audit_log(30),
                    "stats": rt.traces.stats(),
                }
            )

            async def pump() -> None:
                while True:
                    await ws.send_json(await sub.get())

            async def drain() -> None:  # detect client close
                while True:
                    await ws.receive_text()

            tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sub.close()

    return app


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import uvicorn

    settings = Settings.from_env()
    configure_logging()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()

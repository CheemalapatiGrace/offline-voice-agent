"""Human-in-the-loop: operator controls, approval queue, audit log."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .backend.db import Database
from .config import Settings
from .datatypes import ToolCall, ToolResult
from .guardrails.pii import redact_obj
from .observability.events import EventBus


class OperatorError(Exception):
    def __init__(self, code: str, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass
class Approval:
    id: int
    session_id: str
    tool: str
    args: dict[str, Any]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "pending"
    waiting: bool = True
    runner: Callable[[], Awaitable[ToolResult]] | None = None


class ApprovalService:
    """Queue of sensitive tool calls waiting for an operator decision."""

    def __init__(self, db: Database, bus: EventBus, settings: Settings, audit: Callable[..., None]) -> None:
        self.db, self.bus, self.settings, self._audit = db, bus, settings, audit
        self._items: dict[int, Approval] = {}
        self.late_notifier: Callable[[str, str, ToolResult], Awaitable[None]] | None = None

    def requires(self, tool: str) -> bool:
        return tool in self.settings.approval_tools

    async def request(
        self, session_id: str, call: ToolCall, runner: Callable[[], Awaitable[ToolResult]], names: list[str]
    ) -> str:
        """Block (bounded) until an operator decides. Returns approved | rejected | timeout."""
        safe_args = redact_obj(call.args, names)
        cur = self.db.execute(
            "INSERT INTO approvals(ts, session_id, tool, args_json) VALUES (?,?,?,?)",
            (time.time(), session_id, call.name, json.dumps(safe_args)),
        )
        item = Approval(cur.lastrowid, session_id, call.name, safe_args, runner=runner)
        self._items[item.id] = item
        self.bus.publish("approval_requested", session_id, approval=self.as_dict(item))
        try:
            await asyncio.wait_for(item.event.wait(), self.settings.approval_wait_s)
        except TimeoutError:
            item.waiting = False  # still decidable; a late approval runs the action
            self.bus.publish("approval_timeout", session_id, approval=self.as_dict(item))
            return "timeout"
        return item.status

    def as_dict(self, a: Approval) -> dict[str, Any]:
        return {"id": a.id, "session_id": a.session_id, "tool": a.tool, "args": a.args, "status": a.status}

    def list(self, status: str = "pending") -> list[dict[str, Any]]:
        if status == "all":
            rows = self.db.query("SELECT * FROM approvals ORDER BY id DESC LIMIT 100")
        else:
            rows = self.db.query("SELECT * FROM approvals WHERE status=? ORDER BY id DESC LIMIT 100", (status,))
        return [
            {
                "id": r["id"], "ts": r["ts"], "session_id": r["session_id"], "tool": r["tool"],
                "args": json.loads(r["args_json"]), "status": r["status"],
                "decided_by": r["decided_by"], "note": r["note"],
            }
            for r in rows
        ]

    async def decide(self, approval_id: int, approve: bool, operator: str, note: str = "") -> dict[str, Any]:
        item = self._items.get(approval_id)
        rows = self.db.query("SELECT status FROM approvals WHERE id=?", (approval_id,))
        if not rows or item is None:
            raise OperatorError("not_found", "Unknown approval id.", 404)
        if rows[0]["status"] != "pending":
            raise OperatorError("already_decided", f"Approval already {rows[0]['status']}.")
        status = "approved" if approve else "rejected"
        self.db.execute(
            "UPDATE approvals SET status=?, decided_by=?, decided_ts=?, note=? WHERE id=?",
            (status, operator, time.time(), note, approval_id),
        )
        late = not item.waiting
        item.status = status
        item.event.set()
        self._audit(operator, f"approval_{status}", item.session_id, approval_id=approval_id, tool=item.tool, note=note)
        self.bus.publish("approval_decided", item.session_id, approval={**self.as_dict(item), "decided_by": operator})
        if late and approve and item.runner:
            result = await item.runner()
            self.bus.publish("approval_executed", item.session_id, tool=item.tool, status=result.status)
            if self.late_notifier:
                await self.late_notifier(item.session_id, item.tool, result)
        elif late and not approve and self.late_notifier:
            await self.late_notifier(item.session_id, item.tool, ToolResult("rejected"))
        return {"id": approval_id, "status": status, "late": late}


class OperatorService:
    """Operator actions on live sessions. Every action is written to the audit log."""

    def __init__(self, db: Database, bus: EventBus, session_getter: Callable[[str], Any]) -> None:
        self.db, self.bus, self._get = db, bus, session_getter

    def audit(self, operator: str, action: str, session_id: str | None = None, **details: Any) -> None:
        self.db.execute(
            "INSERT INTO audit_log(ts, operator, action, session_id, details_json) VALUES (?,?,?,?,?)",
            (time.time(), operator, action, session_id, json.dumps(details, default=str)),
        )
        self.bus.publish("operator_action", session_id, operator=operator, action=action, details=details)

    def audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [
            {
                "id": r["id"], "ts": r["ts"], "operator": r["operator"], "action": r["action"],
                "session_id": r["session_id"], "details": json.loads(r["details_json"] or "{}"),
            }
            for r in rows
        ]

    def _session(self, session_id: str):
        s = self._get(session_id)
        if s is None:
            raise OperatorError("not_found", "Unknown session.", 404)
        return s

    async def pause(self, session_id: str, operator: str) -> None:
        s = self._session(session_id)
        await s.pipeline.pause()
        self.audit(operator, "pause", session_id)

    async def resume(self, session_id: str, operator: str) -> None:
        s = self._session(session_id)
        await s.pipeline.resume_agent()
        self.audit(operator, "resume_agent", session_id)

    async def takeover(self, session_id: str, operator: str) -> None:
        s = self._session(session_id)
        await s.pipeline.takeover(operator)
        self.audit(operator, "takeover", session_id)

    async def release(self, session_id: str, operator: str) -> None:
        s = self._session(session_id)
        await s.pipeline.resume_agent()
        self.audit(operator, "release_takeover", session_id)

    async def say(self, session_id: str, text: str, operator: str) -> None:
        s = self._session(session_id)
        if s.ctx.state.value not in {"paused", "takeover", "handoff"}:
            raise OperatorError("not_muted", "Pause the agent or take over the session before speaking.")
        text = text.strip()
        if not text or len(text) > 600:
            raise OperatorError("bad_text", "Reply must be 1-600 characters.", 422)
        await s.pipeline.operator_say(text, operator)
        self.audit(operator, "operator_say", session_id, text=redact_obj(text, s.ctx.names()))

    async def set_review(self, session_id: str, enabled: bool, operator: str) -> None:
        s = self._session(session_id)
        s.ctx.review_replies = enabled
        self.bus.publish("review_mode", session_id, enabled=enabled)
        self.audit(operator, "review_mode", session_id, enabled=enabled)

    async def release_reply(self, session_id: str, edited: str | None, operator: str) -> None:
        s = self._session(session_id)
        pr = s.ctx.pending_reply
        if pr is None or pr.event.is_set():
            raise OperatorError("no_pending_reply", "There is no pending reply to release.")
        original = pr.text
        if edited is not None:
            edited = edited.strip()
            if not edited or len(edited) > 600:
                raise OperatorError("bad_text", "Reply must be 1-600 characters.", 422)
            pr.edited = edited
        pr.decision = "release"
        pr.event.set()
        changed = bool(edited) and edited != original
        self.audit(
            operator, "reply_edited_released" if changed else "reply_released", session_id,
            original=redact_obj(original, s.ctx.names()),
            edited=redact_obj(edited, s.ctx.names()) if changed else None,
        )

    async def discard_reply(self, session_id: str, operator: str) -> None:
        s = self._session(session_id)
        pr = s.ctx.pending_reply
        if pr is None or pr.event.is_set():
            raise OperatorError("no_pending_reply", "There is no pending reply to discard.")
        pr.decision = "discard"
        pr.event.set()
        self.audit(operator, "reply_discarded", session_id)

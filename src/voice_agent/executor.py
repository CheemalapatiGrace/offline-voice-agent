"""Tool executor: validation -> idempotency -> permissions -> confirmation -> approval -> execution.

This is the only code path that can touch the appointment backend on behalf of the LLM, so the
security properties (allowlist, schema validation, rate limits, identity/ownership binding,
spoken confirmation for destructive actions, operator approval) are enforced here, not in prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from pydantic import ValidationError

from .backend.store import AppointmentStore, name_key
from .backend.tools import ALLOWLIST, TOOLS, ToolSpec, llm_tool_schemas
from .config import Settings
from .context import PendingConfirm, SessionContext
from .datatypes import ToolCall, ToolResult
from .fmt import fmt_date, fmt_time
from .guardrails.permissions import PermissionLayer
from .guardrails.pii import redact_obj
from .hitl import ApprovalService
from .observability.events import EventBus
from .observability.tracing import TurnTrace


class ToolExecutor:
    def __init__(
        self,
        store: AppointmentStore,
        permissions: PermissionLayer,
        settings: Settings,
        bus: EventBus,
        approvals: ApprovalService | None = None,
    ) -> None:
        self.store, self.permissions, self.settings, self.bus, self.approvals = store, permissions, settings, bus, approvals

    # ----------------------------------------------------------------- public
    def llm_tools(self) -> list[dict[str, Any]]:
        return llm_tool_schemas()

    def needs_approval(self, call: ToolCall) -> bool:
        if not self.approvals or not self.approvals.requires(call.name):
            return False
        return not (call.name == "escalate_to_human" and call.args.get("urgency") == "emergency")

    async def execute(
        self,
        ctx: SessionContext,
        call: ToolCall,
        trace: TurnTrace | None = None,
        *,
        system: bool = False,
        confirmed: bool = False,
    ) -> ToolResult:
        t0 = time.perf_counter()
        result = await self._execute(ctx, call, trace, system=system, confirmed=confirmed)
        dur = (time.perf_counter() - t0) * 1000
        safe_args = redact_obj(call.args, ctx.names())
        ctx.tool_log.append(
            {"name": call.name, "args": dict(call.args), "status": result.status, "replayed": result.replayed, "error": result.error, "system": system}
        )
        if trace is not None:
            trace.record("tool", dur, status="ok" if result.status in {"ok", "needs_confirmation", "pending_approval"} else result.status, name=call.name, args=safe_args, result_status=result.status, replayed=result.replayed)
        self.bus.publish(
            "tool_call", ctx.session_id, name=call.name, args=safe_args, status=result.status,
            result=redact_obj(result.for_llm(), ctx.names()), replayed=result.replayed, ms=round(dur, 2),
        )
        return result

    # ---------------------------------------------------------------- internals
    async def _execute(self, ctx, call: ToolCall, trace, *, system: bool, confirmed: bool) -> ToolResult:
        spec = TOOLS.get(call.name)
        if spec is None or call.name not in ALLOWLIST:
            self._flag(ctx, "tool_not_allowed", call.name)
            return ToolResult("denied", error="tool_not_allowed")
        try:
            parsed = spec.model.model_validate(call.args)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            self._flag(ctx, "invalid_arguments", call.name)
            return ToolResult("denied", error=f"invalid_arguments: {', '.join(fields)}")
        args = parsed.model_dump(exclude_none=True)
        idem_arg = args.pop("idempotency_key", None)
        key = self._idem_key(ctx, spec, args, idem_arg)

        if spec.mutating:
            cached = self.store.idem_get(key)
            if cached is not None:
                return ToolResult(cached["status"], cached.get("data", {}), cached.get("error", ""), replayed=True)

        if spec.name == "escalate_to_human" and ctx.escalated and not system:
            return ToolResult("ok", {"ticket": "existing", "note": "already_escalated"})

        if not system:
            perm = self.permissions.check(ctx, spec, args, confirmed=confirmed)
            if perm.verdict == "deny":
                self._flag(ctx, perm.reason, call.name)
                return ToolResult("denied", error=perm.reason)
            self.permissions.record(ctx, spec)
        else:
            perm = None

        who = args.get("patient_name")
        if who and ctx.patient_key is None:
            ctx.patient_name, ctx.patient_key = who, name_key(who)

        if "appointment_id" in args:
            appt = await asyncio.to_thread(self.store.get, args["appointment_id"])
            if appt is None or appt["patient_key"] != ctx.patient_key:
                self._flag(ctx, "ownership_violation", call.name)
                return ToolResult("denied", error="appointment_not_found_for_caller")

        if perm is not None and perm.verdict == "needs_confirmation":
            summary = self._summary(call.name, args, appt if "appointment_id" in args else None)
            ctx.pending_confirm = PendingConfirm(ToolCall(call.name, dict(call.args), call.id), summary)
            return ToolResult(
                "needs_confirmation",
                {"summary": summary, "instruction": "Ask the caller to clearly say yes to confirm, or no to keep it."},
            )

        if not system and self.needs_approval(ToolCall(call.name, args)) and self.approvals:
            async def runner() -> ToolResult:
                return await self._run(ctx, spec, args, key, None)

            ctx.activity = "awaiting operator approval"
            try:
                decision = await self.approvals.request(ctx.session_id, ToolCall(call.name, args), runner, ctx.names())
            finally:
                ctx.activity = ""
            if decision == "rejected":
                return ToolResult("rejected", error="operator_rejected")
            if decision == "timeout":
                return ToolResult(
                    "pending_approval",
                    {"message": "A staff member will review and complete this shortly."},
                )
        return await self._run(ctx, spec, args, key, trace)

    async def _run(self, ctx: SessionContext, spec: ToolSpec, args: dict, key: str, trace) -> ToolResult:
        try:
            async with asyncio.timeout(self.settings.tool_timeout_s):
                out = await asyncio.to_thread(self._dispatch, ctx, spec.name, args)
        except TimeoutError:
            return ToolResult("error", error="tool_timeout")
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=f"tool_failure: {type(exc).__name__}")
        ok = out.pop("ok", False)
        if ok:
            result = ToolResult("ok", out)
            if spec.name == "escalate_to_human":
                ctx.escalated = True
            if spec.mutating:
                self.store.idem_put(key, spec.name, {"status": "ok", "data": out})
        else:
            err = out.pop("error", "error")
            out.pop("message", None)
            result = ToolResult("error", out, err)
        return result

    def _dispatch(self, ctx: SessionContext, name: str, a: dict) -> dict[str, Any]:
        s = self.store
        if name == "check_availability":
            return s.availability(a["date"])
        if name == "list_my_appointments":
            return s.list_for(a["patient_name"])
        if name == "book_appointment":
            return s.book(a["patient_name"], a["date"], a["time"], a.get("reason"))
        if name == "reschedule_appointment":
            return s.reschedule(a["appointment_id"], a["new_date"], a["new_time"])
        if name == "cancel_appointment":
            return s.cancel(a["appointment_id"])
        if name == "lookup_faq":
            return s.lookup_faq(a["topic"])
        if name == "escalate_to_human":
            return s.create_handoff(ctx.session_id, a["reason"], a.get("urgency", "routine"))
        return {"ok": False, "error": "unknown_tool"}  # pragma: no cover

    def _idem_key(self, ctx, spec, args, provided) -> str:
        if provided:
            return f"{ctx.session_id}:{provided}"
        blob = json.dumps({"s": ctx.session_id, "t": ctx.turn_no, "n": spec.name, "a": args}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:40]

    def _summary(self, name: str, args: dict, appt: dict | None) -> str:
        if name == "cancel_appointment" and appt:
            return f"cancel your appointment on {fmt_date(appt['date'])} at {fmt_time(appt['time'])}"
        return name.replace("_", " ")

    def _flag(self, ctx: SessionContext, reason: str, tool: str) -> None:
        ctx.guard_events.append({"layer": "permissions", "category": reason, "tool": tool, "ts": time.time()})
        self.bus.publish("guardrail", ctx.session_id, layer="permissions", category=reason, tool=tool)

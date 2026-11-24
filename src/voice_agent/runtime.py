"""Composition root: wires adapters, guardrails, backend, observability, HITL and sessions."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import Adapters, build_adapters
from .agent import Agent
from .backend.db import Database
from .backend.store import AppointmentStore, seed_demo
from .backend.tools import ALLOWLIST
from .config import Settings
from .executor import ToolExecutor
from .guardrails.input_guard import InputGuard
from .guardrails.output_guard import OutputGuard
from .guardrails.permissions import PermissionLayer
from .hitl import ApprovalService, OperatorService
from .observability.events import EventBus
from .observability.tracing import Tracer, TraceStore
from .session import SessionManager


@dataclass
class Runtime:
    settings: Settings
    db: Database
    store: AppointmentStore
    traces: TraceStore
    bus: EventBus
    tracer: Tracer
    adapters: Adapters
    executor: ToolExecutor
    agent: Agent
    approvals: ApprovalService
    operator: OperatorService
    manager: SessionManager

    async def start(self) -> None:
        self.manager.start_reaper()

    async def stop(self) -> None:
        await self.manager.shutdown()
        self.db.close()


def build_runtime(
    settings: Settings | None = None,
    adapters: Adapters | None = None,
    *,
    seed: bool = True,
    input_guard: bool = True,
    output_guard: bool = True,
) -> Runtime:
    settings = settings or Settings.from_env()
    db = Database(settings.db_path)
    store = AppointmentStore(db, settings.clinic_today)
    if seed and not db.query("SELECT 1 FROM appointments LIMIT 1"):
        seed_demo(store)
    bus = EventBus()
    traces = TraceStore(db)
    tracer = Tracer(traces, bus)
    adapters = adapters or build_adapters(settings, bus, settings.clinic_today)
    # the manager is needed by OperatorService lazily, so use a late-bound getter
    holder: dict[str, SessionManager] = {}
    operator = OperatorService(db, bus, lambda sid: holder["m"].get(sid))
    approvals = ApprovalService(db, bus, settings, operator.audit)
    permissions = PermissionLayer(settings, ALLOWLIST)
    executor = ToolExecutor(store, permissions, settings, bus, approvals)
    out_guard = OutputGuard(output_guard)
    out_guard.set_names_provider(store.all_patient_names)
    agent = Agent(adapters.llm, executor, InputGuard(input_guard), out_guard, settings, bus)
    manager = SessionManager(settings, adapters, agent, tracer, bus)
    holder["m"] = manager
    approvals.late_notifier = manager.notify_late
    return Runtime(settings, db, store, traces, bus, tracer, adapters, executor, agent, approvals, operator, manager)

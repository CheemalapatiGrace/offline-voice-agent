"""Tool-permission layer: allowlist, rate limits, identity binding, confirmation gate."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..config import Settings


@dataclass
class PermDecision:
    verdict: str  # allow | deny | needs_confirmation
    reason: str = ""
    message: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


def name_key(name: str) -> str:
    return re.sub(r"[^a-z ]", "", name.lower()).strip().replace("  ", " ")


class PermissionLayer:
    """Stateless policy; per-session counters live on the SessionContext."""

    def __init__(self, settings: Settings, allowlist: set[str], clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.allowlist = allowlist
        self.clock = clock

    def _prune(self, dq: deque, window: float = 60.0) -> None:
        now = self.clock()
        while dq and now - dq[0] > window:
            dq.popleft()

    def check(self, ctx, spec, args: dict, *, confirmed: bool = False) -> PermDecision:
        """``ctx`` needs: patient_key, tool_times, mutation_times."""
        if spec is None or spec.name not in self.allowlist:
            return PermDecision("deny", "tool_not_allowed", "That action is not permitted.")

        self._prune(ctx.tool_times)
        if len(ctx.tool_times) >= self.settings.rate_limit_calls_per_min:
            return PermDecision("deny", "rate_limited", "Too many requests; slowing down.")
        if spec.mutating:
            self._prune(ctx.mutation_times)
            if len(ctx.mutation_times) >= self.settings.rate_limit_mutations_per_min:
                return PermDecision("deny", "rate_limited", "Too many changes requested; slowing down.")

        who = args.get("patient_name")
        if who is not None and spec.name != "escalate_to_human":
            key = name_key(who)
            if ctx.patient_key and key != ctx.patient_key:
                return PermDecision(
                    "deny", "identity_mismatch", "I can only act on the account for this call."
                )

        if spec.destructive and not confirmed:
            return PermDecision("needs_confirmation", "confirmation_required")
        return PermDecision("allow")

    def record(self, ctx, spec) -> None:
        now = self.clock()
        ctx.tool_times.append(now)
        if spec.mutating:
            ctx.mutation_times.append(now)

"""Fallback chain for LLMs: per-backend timeout + retry + circuit breaker, then the next backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..config import Settings
from ..datatypes import LLMChunk, LLMUnavailable, StreamInterrupted
from ..observability.logging import get_logger
from ..resilience import CircuitBreaker

log = get_logger("llm")


class ResilientLLM:
    """``primary -> degraded (rule based) -> raise LLMUnavailable`` (agent then hands off to a human)."""

    name = "resilient"

    def __init__(self, chain: list, settings: Settings, bus=None) -> None:
        if not chain:
            raise ValueError("LLM chain must not be empty")
        self.chain = chain
        self.settings = settings
        self.bus = bus
        self.breakers = {
            llm.name: CircuitBreaker(settings.breaker_failures, settings.breaker_cooldown_s) for llm in chain
        }
        self.last_used: str | None = None
        self.fallbacks = 0

    @property
    def degraded(self) -> bool:
        return self.last_used is not None and self.last_used != self.chain[0].name

    def health(self) -> dict[str, str]:
        return {name: b.state for name, b in self.breakers.items()}

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[LLMChunk]:
        errors: list[str] = []
        for idx, llm in enumerate(self.chain):
            breaker = self.breakers[llm.name]
            if not breaker.allow():
                errors.append(f"{llm.name}: circuit open")
                continue
            for attempt in range(self.settings.llm_retries + 1):
                started = False
                it = llm.stream(messages, tools).__aiter__()
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(it.__anext__(), self.settings.llm_timeout_s)
                        except StopAsyncIteration:
                            break
                        started = True
                        yield chunk
                    breaker.success()
                    if idx > 0 and self.last_used != llm.name:
                        self.fallbacks += 1
                        if self.bus:
                            self.bus.publish("degraded", None, component="llm", using=llm.name)
                    self.last_used = llm.name
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    breaker.failure()
                    errors.append(f"{llm.name}#{attempt}: {type(exc).__name__}")
                    log.warning("llm failure", extra={"backend": llm.name, "attempt": attempt, "error": type(exc).__name__})
                    try:
                        await it.aclose()
                    except Exception:  # noqa: BLE001
                        pass
                    if started:
                        raise StreamInterrupted(str(exc)) from exc
                    await asyncio.sleep(0.02 * (attempt + 1))
        raise LLMUnavailable("; ".join(errors) or "no backend available")

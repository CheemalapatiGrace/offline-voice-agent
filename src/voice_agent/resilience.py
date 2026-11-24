"""Small reliability toolkit: retries with timeout + backoff, and a circuit breaker."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class CircuitBreaker:
    """closed -> open after N consecutive failures -> half-open after a cooldown."""

    def __init__(self, failures: int = 3, cooldown_s: float = 15.0, clock: Callable[[], float] = time.monotonic):
        self.failures = failures
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self._count = 0
        self._opened_at = None

    def failure(self) -> None:
        self._count += 1
        if self._count >= self.failures or self.state == "half_open":
            self._opened_at = self._clock()


async def call_with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: float,
    retries: int,
    backoff_s: float = 0.05,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run ``fn`` with a per-attempt timeout and exponential backoff between attempts."""
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            async with asyncio.timeout(timeout):
                return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberate: any stage failure is retryable
            last = exc
            if attempt < retries:
                if on_retry:
                    on_retry(attempt + 1, exc)
                await asyncio.sleep(backoff_s * (2**attempt))
    assert last is not None
    raise last

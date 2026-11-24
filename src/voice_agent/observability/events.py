"""In-process pub/sub used by the live dashboard. Slow subscribers lose old events, never block."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any


class Subscription:
    def __init__(self, bus: EventBus, maxsize: int) -> None:
        self._bus = bus
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize)
        self.dropped = 0

    def push(self, event: dict[str, Any]) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        self.queue.put_nowait(event)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()

    def close(self) -> None:
        self._bus._subs.discard(self)


class EventBus:
    def __init__(self, history: int = 500) -> None:
        self._subs: set[Subscription] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._seq = 0

    def publish(self, type_: str, session_id: str | None = None, **data: Any) -> dict[str, Any]:
        self._seq += 1
        event = {"seq": self._seq, "ts": time.time(), "type": type_, "session_id": session_id, **data}
        self._history.append(event)
        for sub in list(self._subs):
            sub.push(event)
        return event

    def subscribe(self, maxsize: int = 1000) -> Subscription:
        sub = Subscription(self, maxsize)
        self._subs.add(sub)
        return sub

    def recent(self, session_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        items = [e for e in self._history if session_id is None or e["session_id"] == session_id]
        return items[-limit:]

    @property
    def subscribers(self) -> int:
        return len(self._subs)

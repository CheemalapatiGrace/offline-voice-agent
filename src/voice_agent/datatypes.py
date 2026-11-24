"""Shared value types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SessionState(StrEnum):
    CONNECTING = "connecting"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    PAUSED = "paused"  # operator paused the agent
    TAKEOVER = "takeover"  # operator is driving the session
    HANDOFF = "handoff"  # escalated to a human, agent muted
    DISCONNECTED = "disconnected"  # socket dropped, awaiting reconnect
    CLOSED = "closed"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class ToolResult:
    status: str  # ok | error | denied | needs_confirmation | pending_approval | rejected
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    replayed: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def for_llm(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.data:
            out["data"] = self.data
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class LLMChunk:
    text: str | None = None
    tool_call: ToolCall | None = None


@dataclass
class Utterance:
    kind: str  # audio | text | operator | system
    pcm: bytes = b""
    text: str = ""
    t_end: float = field(default_factory=time.perf_counter)
    vad_ms: float = 0.0
    duration_ms: float = 0.0
    operator: str = ""


class LLMUnavailable(RuntimeError):
    """Every LLM backend in the fallback chain failed or is circuit-open."""


class StreamInterrupted(RuntimeError):
    """The LLM stream died after it had already produced output."""

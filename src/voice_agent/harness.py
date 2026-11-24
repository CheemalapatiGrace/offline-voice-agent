"""In-process client used by the text/audio simulator, the eval suite, the benchmark and tests.

It drives a real :class:`VoicePipeline` through its public inputs (audio frames or text), exactly as
the WebSocket handler does, and collects everything the pipeline sends back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .audio import split_frames, synth_speech
from .runtime import Runtime
from .session import Session


class CollectorSink:
    """Stands in for a WebSocket: records every JSON message and audio chunk."""

    realtime = False

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.audio_chunks = 0
        self.audio_bytes = 0

    async def send_json(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)

    async def send_audio(self, data: bytes) -> None:
        self.audio_chunks += 1
        self.audio_bytes += len(data)


@dataclass
class TurnResult:
    user_text: str = ""  # what the STT heard
    reply: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    guard: list[str] = field(default_factory=list)  # guardrail categories raised this turn
    state: str = ""
    escalated: bool = False
    audio_bytes: int = 0
    trace: dict[str, Any] = field(default_factory=dict)
    wall_ms: float = 0.0

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tool_calls]


class SimClient:
    def __init__(self, runtime: Runtime, session: Session, sink: CollectorSink) -> None:
        self.rt, self.session, self.sink = runtime, session, sink
        self.ctx = session.ctx
        self.pipeline = session.pipeline

    @classmethod
    async def open(cls, runtime: Runtime) -> SimClient:
        sink = CollectorSink()
        session = await runtime.manager.create(sink)
        return cls(runtime, session, sink)

    async def say(self, text: str, *, via_audio: bool = True, timeout: float = 30.0) -> TurnResult:
        """Speak (or type) one user turn and wait until the agent has fully responded."""
        m0, t0, g0 = len(self.sink.messages), len(self.ctx.tool_log), len(self.ctx.guard_events)
        a0 = self.sink.audio_bytes
        n_turns = len(self.rt.traces.turns(self.ctx.session_id, 1000))
        start = time.perf_counter()
        if via_audio:
            for frame in split_frames(synth_speech(text), self.rt.settings.sample_rate, self.rt.settings.frame_ms):
                await self.pipeline.audio_in.put(frame)
        else:
            await self.pipeline.push_text(text, interrupt=False)
        await self.pipeline.wait_idle(timeout)
        wall = (time.perf_counter() - start) * 1000
        msgs = self.sink.messages[m0:]
        heard = next((m["text"] for m in msgs if m.get("type") == "transcript" and m.get("role") == "user"), "")
        replies = [m["text"] for m in msgs if m.get("type") == "reply"]
        turns = self.rt.traces.turns(self.ctx.session_id, 1000)
        trace = turns[-1] if len(turns) > n_turns else {}
        guard = [
            g["category"] if g["layer"] != "output" else "out:" + g["category"]
            for g in self.ctx.guard_events[g0:]
        ]
        return TurnResult(
            user_text=heard,
            reply=" ".join(replies),
            tool_calls=[dict(t) for t in self.ctx.tool_log[t0:]],
            guard=guard,
            state=self.ctx.state.value,
            escalated=self.ctx.escalated,
            audio_bytes=self.sink.audio_bytes - a0,
            trace=trace,
            wall_ms=wall,
        )

    async def close(self, reason: str = "client_closed") -> None:
        await self.rt.manager.close(self.ctx.session_id, reason)

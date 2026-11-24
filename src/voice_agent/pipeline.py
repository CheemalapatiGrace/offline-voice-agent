"""Async streaming voice pipeline for one session.

    audio frames --> [audio_in queue] --> VAD --> [utterance queue] --> turn task
    turn task:  STT -> guardrails/LLM/tools (agent, sentence streaming) -> TTS --> [audio_out queue] --> sink

* Every hop is a *bounded* queue. Producers that must not lose data (utterances, TTS audio) block
  (backpressure); the raw microphone queue drops its oldest frame instead (real-time stream).
* Each stage has its own timeout/retry (see ``Settings``) and a defined degradation path.
* Barge-in: VAD speech-start while the agent is speaking cancels the in-flight turn, bumps an
  *epoch* so stale audio is discarded, and tells the client to flush its playback buffer.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from .adapters import Adapters
from .agent import HANDOFF_ACTIVE_MSG, REPROMPT_MSG, Agent, split_sentences
from .config import Settings
from .context import InvalidTransition, PendingReply, SessionContext
from .datatypes import SessionState, Utterance
from .guardrails.pii import redact
from .observability.events import EventBus
from .observability.logging import get_logger
from .observability.tracing import Tracer, TurnTrace
from .resilience import call_with_retries

log = get_logger("pipeline")
S = SessionState

HOLD_ON_PAUSE = "Please hold for a moment."
JOINING = "A team member is joining the call now."
TURN_ERROR_MSG = "Sorry, something went wrong on my end. Could you repeat that?"


class Sink(Protocol):
    realtime: bool

    async def send_json(self, msg: dict[str, Any]) -> None: ...

    async def send_audio(self, data: bytes) -> None: ...


class _Acc:
    """Accumulates only the time spent *waiting on a stage*, excluding consumer time."""

    def __init__(self) -> None:
        self.work_s = 0.0
        self.backpressure_s = 0.0
        self.first_chunk_ms: float | None = None
        self.sentences = 0
        self.failures = 0


class VoicePipeline:
    def __init__(
        self,
        ctx: SessionContext,
        *,
        settings: Settings,
        adapters: Adapters,
        agent: Agent,
        tracer: Tracer,
        bus: EventBus,
    ) -> None:
        self.ctx, self.settings, self.adapters = ctx, settings, adapters
        self.agent, self.tracer, self.bus = agent, tracer, bus
        self.stt, self.tts = adapters.stt, adapters.tts
        self.vad = adapters.vad_factory()
        self.audio_in: asyncio.Queue[bytes] = asyncio.Queue(settings.audio_in_queue)
        self.utt_q: asyncio.Queue[Utterance] = asyncio.Queue(settings.utterance_queue)
        self.out_q: asyncio.Queue[tuple[int | None, str, Any]] = asyncio.Queue(settings.audio_out_queue + 64)  # +64: headroom for control msgs
        self.sink: Sink | None = None
        self._sink_ready = asyncio.Event()
        self._epoch = 0
        self._attach_gen = 0
        self._current: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._speak_until = 0.0
        self.metrics = {"dropped_frames": 0, "interrupts": 0, "tts_failures": 0, "stt_failures": 0, "turns": 0}

    # ------------------------------------------------------------- lifecycle
    async def start(self, sink: Sink | None = None) -> None:
        if sink is not None:
            self.sink = sink
            self._sink_ready.set()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name=f"ingest-{self.ctx.session_id}"),
            asyncio.create_task(self._turn_loop(), name=f"turns-{self.ctx.session_id}"),
            asyncio.create_task(self._sender_loop(), name=f"send-{self.ctx.session_id}"),
        ]

    async def stop(self) -> None:
        if self._current and not self._current.done():
            self._current.cancel()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, *( [self._current] if self._current else []), return_exceptions=True)
        self._tasks = []

    def detach(self) -> None:
        """The client socket is gone: abort the in-flight reply, keep all session state."""
        self.sink = None
        self._sink_ready.clear()
        if self._current and not self._current.done():
            self._current.cancel()

    async def attach(self, sink: Sink) -> None:
        self._flush()
        self._attach_gen += 1  # anything a sender took before this point belongs to the dead connection
        self.sink = sink
        self._sink_ready.set()

    # ----------------------------------------------------------------- inputs
    async def push_audio(self, frame: bytes) -> None:
        """Microphone frames. Never blocks: on overflow the *oldest* frame is dropped."""
        try:
            self.audio_in.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self.audio_in.get_nowait()
                self.audio_in.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
            self.metrics["dropped_frames"] += 1
            self.audio_in.put_nowait(frame)

    async def push_text(self, text: str, interrupt: bool = True) -> None:
        if interrupt and self._is_speaking():
            await self._interrupt("text_barge_in")
        await self.utt_q.put(Utterance("text", text=text))

    async def operator_say(self, text: str, operator: str) -> None:
        await self.utt_q.put(Utterance("operator", text=text, operator=operator))

    async def system_say(self, text: str) -> None:
        await self.utt_q.put(Utterance("system", text=text))

    async def barge_in(self) -> None:
        await self._interrupt("client_interrupt")

    async def wait_idle(self, timeout: float = 30.0) -> None:
        async with asyncio.timeout(timeout):
            await self.audio_in.join()
            await self.utt_q.join()
            await self.out_q.join()

    # -------------------------------------------------------------- operator
    async def pause(self) -> None:
        await self._cancel_current("operator_pause")
        self._drop_pending_reply()
        self._set_state(S.PAUSED)
        await self.system_say(HOLD_ON_PAUSE)

    async def takeover(self, operator: str) -> None:
        await self._cancel_current("operator_takeover")
        self._drop_pending_reply()
        self._set_state(S.TAKEOVER)
        self.ctx.activity = f"operator {operator} driving"
        await self.system_say(JOINING)

    async def resume_agent(self) -> None:
        self.ctx.escalated = False
        self.ctx.activity = ""
        self._set_state(S.IDLE)

    def _drop_pending_reply(self) -> None:
        pr = self.ctx.pending_reply
        if pr and not pr.event.is_set():
            pr.decision = "discard"
            pr.event.set()

    # ---------------------------------------------------------------- state
    def _is_speaking(self) -> bool:
        if self.ctx.state == S.SPEAKING:
            return True
        return bool(self.sink and getattr(self.sink, "realtime", False) and time.monotonic() < self._speak_until)

    def _set_state(self, new: SessionState, *, quiet: bool = False, **extra: Any) -> None:
        ctx = self.ctx
        if new == ctx.state:
            return
        try:
            old = ctx.transition(new)
        except InvalidTransition as exc:
            log.warning("invalid transition ignored", extra={"error": str(exc)})
            return
        self.bus.publish("state", ctx.session_id, state=new.value, previous=old.value, activity=ctx.activity, **extra)
        if not quiet:
            self._emit_nowait({"type": "state", "state": new.value})

    def snapshot(self) -> dict[str, Any]:
        c = self.ctx
        return {
            "session_id": c.session_id,
            "state": c.state.value,
            "activity": c.activity,
            "turn": c.turn_no,
            "escalated": c.escalated,
            "review_replies": c.review_replies,
            "degraded": sorted(c.degraded),
            "violations": c.violations,
            "pending_confirm": c.pending_confirm.summary if c.pending_confirm else None,
            "pending_reply": {"id": c.pending_reply.id, "text": redact(c.pending_reply.text, c.names())} if c.pending_reply else None,
            "metrics": dict(self.metrics),
            "queues": {"audio_in": self.audio_in.qsize(), "utterances": self.utt_q.qsize(), "audio_out": self.out_q.qsize()},
            "tool_calls": [{"name": t["name"], "status": t["status"]} for t in c.tool_log[-10:]],
            "created_at": c.created_at,
        }

    # ------------------------------------------------------------ out channel
    def _emit_nowait(self, msg: dict[str, Any]) -> None:
        """Best-effort control message (state, transcript echo). Never blocks the caller."""
        try:
            self.out_q.put_nowait((None, "json", msg))
        except asyncio.QueueFull:
            log.warning("out queue full; dropping control message", extra={"type": msg.get("type")})

    async def _enqueue(self, kind: str, payload: Any, epoch: int, acc: _Acc | None = None) -> None:
        t = time.perf_counter()
        cap = self.settings.audio_out_queue
        while self.out_q.qsize() >= cap:  # backpressure: wait for a slow client to drain
            await asyncio.sleep(0.002)
        self.out_q.put_nowait((epoch, kind, payload))
        if acc is not None:
            acc.backpressure_s += time.perf_counter() - t

    def _flush(self) -> None:
        self._epoch += 1
        while True:
            try:
                self.out_q.get_nowait()
                self.out_q.task_done()
            except asyncio.QueueEmpty:
                break

    async def _sender_loop(self) -> None:
        while True:
            epoch, kind, payload = await self.out_q.get()
            try:
                if epoch is not None and epoch != self._epoch:
                    continue
                gen = self._attach_gen
                await self._sink_ready.wait()
                if gen != self._attach_gen or (epoch is not None and epoch != self._epoch):
                    continue
                sink = self.sink
                if sink is None:
                    continue
                try:
                    if kind == "audio":
                        await sink.send_audio(payload)
                    else:
                        await sink.send_json(payload)
                        if payload.get("type") == "audio_end":
                            self.ctx.reply_delivered = True
                except Exception as exc:  # noqa: BLE001 - client went away mid-send
                    log.info("sink send failed", extra={"error": type(exc).__name__})
                    self.sink = None
                    self._sink_ready.clear()
            finally:
                self.out_q.task_done()

    # ---------------------------------------------------------------- stage 1
    async def _ingest_loop(self) -> None:
        while True:
            frame = await self.audio_in.get()
            try:
                ev = self.vad.feed(frame)
                if ev is None:
                    continue
                if ev.kind == "speech_start":
                    await self._on_speech_start()
                else:
                    self.ctx.touch()
                    await self.utt_q.put(
                        Utterance("audio", pcm=ev.pcm, vad_ms=ev.vad_ms, duration_ms=ev.duration_ms)
                    )
            finally:
                self.audio_in.task_done()

    async def _on_speech_start(self) -> None:
        ctx = self.ctx
        ctx.touch()
        if ctx.muted or ctx.state in (S.CLOSED, S.DISCONNECTED):
            return
        if self._is_speaking():
            await self._interrupt("barge_in")
        if ctx.state in (S.IDLE, S.SPEAKING):
            self._set_state(S.LISTENING)

    async def _cancel_current(self, reason: str) -> None:
        self._flush()
        task = self._current
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.wait({task})
        self._speak_until = 0.0

    async def _interrupt(self, reason: str) -> None:
        self.metrics["interrupts"] += 1
        await self._cancel_current(reason)
        self._emit_nowait({"type": "interrupt", "reason": reason})
        self.bus.publish("interrupt", self.ctx.session_id, reason=reason)

    # ---------------------------------------------------------------- stage 2
    async def _turn_loop(self) -> None:
        while True:
            utt = await self.utt_q.get()
            try:
                self._current = asyncio.create_task(self._run_turn(utt), name=f"turn-{self.ctx.session_id}")
                await asyncio.wait({self._current})
                task = self._current
                if not task.cancelled() and task.exception():
                    log.error("turn crashed", exc_info=task.exception())
                    if self.ctx.state in (S.THINKING, S.SPEAKING, S.LISTENING):
                        self._set_state(S.IDLE)
            finally:
                self._current = None
                self.utt_q.task_done()

    async def _stt(self, utt: Utterance, trace: TurnTrace) -> str | None:
        s = self.settings
        try:
            with trace.span("stt", audio_ms=round(utt.duration_ms), backend=getattr(self.stt, "name", "?")) as at:
                text = await call_with_retries(
                    lambda: self.stt.transcribe(utt.pcm, s.sample_rate),
                    timeout=s.stt_timeout_s,
                    retries=s.stt_retries,
                    on_retry=lambda n, e: log.warning("stt retry", extra={"attempt": n, "error": type(e).__name__}),
                )
                at["chars"] = len(text)
        except Exception:  # noqa: BLE001
            self.ctx.stt_failures += 1
            self.metrics["stt_failures"] += 1
            return None
        self.ctx.stt_failures = 0
        return text.strip()

    async def _run_turn(self, utt: Utterance) -> None:
        ctx = self.ctx
        if utt.kind in {"operator", "system"}:
            await self._run_scripted(utt)
            return
        ctx.turn_no += 1
        self.metrics["turns"] += 1
        trace = self.tracer.start_turn(ctx.session_id, ctx.turn_no, utt.t_end)
        try:
            if ctx.turn_no > self.settings.max_turns_per_session:
                trace.outcome = "turn_limit"
                self._emit_nowait({"type": "error", "code": "turn_limit"})
                return
            text: str | None
            if utt.kind == "audio":
                trace.record("vad", utt.vad_ms, start_ms=0.0, endpoint_hangover_ms=self.settings.vad_hangover_ms, utterance_ms=round(utt.duration_ms))
                text = await self._stt(utt, trace)
                if text is None:
                    trace.outcome = "stt_error"
                    await self._canned(REPROMPT_MSG, trace)
                    if ctx.stt_failures >= 3:
                        await self.agent._escalate(ctx, trace, "speech recognition failing", "routine")
                        self._set_state(S.HANDOFF)
                    return
            else:
                text = utt.text.strip()
            if not text:
                trace.outcome = "empty"
                await self._canned(REPROMPT_MSG, trace)
                return

            trace.user_text = text
            item = ctx.add_transcript("user", text)
            self.bus.publish("transcript", ctx.session_id, role="user", text=item["text"], turn=ctx.turn_no)
            self._emit_nowait({"type": "transcript", "role": "user", "text": text})

            if ctx.muted:
                trace.outcome = "muted"
                return
            if ctx.state == S.HANDOFF:
                trace.outcome = "handoff_hold"
                await self._canned(HANDOFF_ACTIVE_MSG, trace, set_state=False)
                return
            self._set_state(S.THINKING)
            await self._respond(text, trace)
        except asyncio.CancelledError:
            trace.outcome = "interrupted"
            raise
        except Exception:  # noqa: BLE001 - last-resort recovery keeps the call alive
            log.exception("turn failed")
            trace.outcome = "error"
            ctx.degraded.add("turn_error")
            await self._canned(TURN_ERROR_MSG, trace)
        finally:
            trace.add_names(*ctx.names())
            if ctx.state in (S.THINKING, S.SPEAKING, S.LISTENING):
                self._set_state(S.HANDOFF if ctx.escalated else S.IDLE)
            ctx.activity = "" if not ctx.muted else ctx.activity
            self.tracer.finish(trace)

    async def _respond(self, text: str, trace: TurnTrace) -> None:
        ctx = self.ctx
        gen = self.agent.run_turn(ctx, text, trace)
        acc = _Acc()
        spoken: list[str] = []
        epoch = self._epoch
        try:
            if ctx.review_replies:
                sentences = [s async for s in gen]
                spoken = await self._review(sentences, trace, acc, epoch)
            else:
                async for sentence in gen:
                    spoken.append(sentence)  # recorded before speaking so an interrupted sentence is kept
                    ctx.last_reply = " ".join(spoken)
                    await self._speak(sentence, trace, acc, epoch)
        except asyncio.CancelledError:
            if spoken and ctx.history and ctx.history[-1]["role"] in {"user", "tool"}:
                ctx.history.append({"role": "assistant", "content": " ".join(spoken) + " [interrupted]"})
            raise
        finally:
            await gen.aclose()
            trace.record(
                "tts", acc.work_s * 1000, first_chunk_ms=None if acc.first_chunk_ms is None else round(acc.first_chunk_ms, 2),
                backpressure_ms=round(acc.backpressure_s * 1000, 2), sentences=acc.sentences, failures=acc.failures,
            )
        trace.reply_text = " ".join(spoken)
        ctx.last_reply = trace.reply_text
        if spoken:
            await self._enqueue("json", {"type": "audio_end"}, epoch)

    async def _review(self, sentences: list[str], trace: TurnTrace, acc: _Acc, epoch: int) -> list[str]:
        ctx = self.ctx
        text = " ".join(sentences).strip()
        if not text:
            return []
        pr = PendingReply(uuid.uuid4().hex[:8], text)
        ctx.pending_reply = pr
        ctx.activity = "awaiting operator review"
        self.bus.publish("pending_reply", ctx.session_id, id=pr.id, text=redact(text, ctx.names()))
        try:
            await asyncio.wait_for(pr.event.wait(), self.settings.review_timeout_s)
        except TimeoutError:
            pr.decision = "timeout"
            self.bus.publish("reply_review_timeout", ctx.session_id, id=pr.id, action=self.settings.review_timeout_action)
        finally:
            ctx.pending_reply = None
            ctx.activity = ""
        if pr.decision == "discard" or (pr.decision == "timeout" and self.settings.review_timeout_action == "drop"):
            trace.outcome = "reply_discarded"
            return []
        final = pr.edited or text
        if pr.edited:
            trace.attrs["reply_edited"] = True
            for m in reversed(ctx.history):
                if m["role"] == "assistant":
                    m["content"] = final
                    break
        parts, rest = split_sentences(final)
        out = [*parts, rest.strip()] if rest.strip() else parts
        for s in out:
            await self._speak(s, trace, acc, epoch)
        return out

    async def _run_scripted(self, utt: Utterance) -> None:
        """Operator or system utterances: spoken as-is, never passed through the agent."""
        ctx = self.ctx
        role = "operator" if utt.kind == "operator" else "agent"
        trace = self.tracer.start_turn(ctx.session_id, ctx.turn_no, None)
        trace.outcome = utt.kind
        trace.reply_text = utt.text
        acc = _Acc()
        try:
            item = ctx.add_transcript(role, utt.text, operator=utt.operator or None)
            self.bus.publish("transcript", ctx.session_id, role=role, text=item["text"], operator=utt.operator or None)
            for s in self._sentences(utt.text):
                await self._speak(s, trace, acc, self._epoch, set_state=False)
            await self._enqueue("json", {"type": "audio_end"}, self._epoch)
        finally:
            trace.record("tts", acc.work_s * 1000, first_chunk_ms=acc.first_chunk_ms, sentences=acc.sentences)
            trace.add_names(*ctx.names())
            self.tracer.finish(trace)

    async def _canned(self, text: str, trace: TurnTrace, set_state: bool = True) -> None:
        acc = _Acc()
        epoch = self._epoch
        await self._speak(text, trace, acc, epoch, set_state=set_state)
        await self._enqueue("json", {"type": "audio_end"}, epoch)
        trace.reply_text = text
        trace.record("tts", acc.work_s * 1000, sentences=acc.sentences)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts, rest = split_sentences(text)
        return [*parts, rest.strip()] if rest.strip() else parts

    # ---------------------------------------------------------------- stage 3
    async def _speak(self, sentence: str, trace: TurnTrace, acc: _Acc, epoch: int, *, set_state: bool = True) -> None:
        ctx = self.ctx
        if set_state and ctx.state in (S.THINKING, S.LISTENING, S.IDLE):
            self._set_state(S.SPEAKING)
        if acc.sentences == 0:
            ctx.reply_delivered = False
        acc.sentences += 1
        item = ctx.add_transcript("agent", sentence) if set_state else {"text": redact(sentence, ctx.names())}
        self.bus.publish("transcript", ctx.session_id, role="agent", text=item["text"], turn=ctx.turn_no)
        await self._enqueue("json", {"type": "reply", "text": sentence}, epoch, acc)
        s = self.settings
        for attempt in range(s.tts_retries + 1):
            produced = False
            agen = self.tts.synthesize(sentence).__aiter__()
            try:
                while True:
                    t = time.perf_counter()
                    try:
                        chunk = await asyncio.wait_for(agen.__anext__(), s.tts_timeout_s)
                    except StopAsyncIteration:
                        acc.work_s += time.perf_counter() - t
                        return
                    acc.work_s += time.perf_counter() - t
                    if acc.first_chunk_ms is None:
                        acc.first_chunk_ms = trace.now_ms()
                        trace.mark_first_audio()
                    produced = True
                    await self._enqueue("audio", chunk, epoch, acc)
                    if getattr(self.sink, "realtime", False):
                        now = time.monotonic()
                        self._speak_until = max(now, self._speak_until) + len(chunk) / 2 / getattr(self.tts, "sample_rate", 16000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("tts failure", extra={"attempt": attempt, "error": type(exc).__name__})
                if produced:
                    break
            finally:
                try:
                    await agen.aclose()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
        acc.failures += 1
        self.metrics["tts_failures"] += 1
        ctx.degraded.add("tts")
        self.bus.publish("degraded", ctx.session_id, component="tts", mode="text_only")

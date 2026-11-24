"""Deterministic, offline adapters: synthetic-audio STT, rule-based tool-calling LLM, silent TTS.

They implement exactly the same interfaces as the real backends, so the whole pipeline (VAD ->
STT -> guardrails -> LLM+tools -> TTS) runs in tests, CI and on a laptop with no model downloads.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..audio import decode_speech
from ..datatypes import LLMChunk, ToolCall
from . import nlu

# =========================================================================== STT


class MockSTT:
    """Decodes text from :func:`voice_agent.audio.synth_speech` audio (deterministic)."""

    name = "mock-stt"

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return decode_speech(pcm) or ""


class ScriptedSTT:
    """Ignores the audio and returns scripted transcripts in order (unit tests)."""

    name = "scripted-stt"

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return self.script.pop(0) if self.script else ""


_CONFUSIONS = {
    "to": "two", "two": "to", "for": "four", "four": "for", "tomorrow": "tomorow", "cancel": "counsel",
    "appointment": "a point meant", "wednesday": "wensday", "thursday": "thirsty", "friday": "fry day",
    "book": "look", "ten": "then", "am": "a", "pm": "p", "hours": "ours", "insurance": "in surance",
}
_FILLERS = ["uh", "um", "the", "and", "so"]


def corrupt_transcript(text: str, wer: float, rng: random.Random) -> str:
    """Word-level substitution/deletion/insertion noise with expected WER ~= ``wer``."""
    out: list[str] = []
    for w in text.split():
        if rng.random() >= wer:
            out.append(w)
            continue
        op = rng.choices(["sub", "del", "ins"], [0.5, 0.3, 0.2])[0]
        if op == "sub":
            alt = _CONFUSIONS.get(w.lower())
            if alt is None and len(w) > 2:
                i = rng.randrange(len(w))
                alt = w[:i] + rng.choice("aeiourstln") + w[i + 1 :]
                if alt == w:
                    alt = w + "s"
            out.append(alt or rng.choice(_FILLERS))
        elif op == "ins":
            out.extend([w, rng.choice(_FILLERS)])
    return " ".join(out)


class NoisySTT:
    """Wraps an STT and injects a reproducible word error rate (the mock STT noise model)."""

    def __init__(self, inner, wer: float = 0.1, seed: int = 1234) -> None:
        self.inner, self.wer, self.seed = inner, wer, seed
        self.name = f"noisy({inner.name},{wer})"

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        clean = await self.inner.transcribe(pcm, sample_rate)
        rng = random.Random(zlib.crc32(clean.encode()) ^ self.seed)
        return corrupt_transcript(clean, self.wer, rng)


# =========================================================================== TTS

_ZERO_CHUNK = b"\x00\x00" * 1600  # 100 ms @ 16 kHz


class SilentTTS:
    """Emits silence whose length is proportional to the text (so barge-in is meaningful)."""

    name = "silent-tts"
    sample_rate = 16000

    def __init__(self, ms_per_char: float = 55.0, chunk_delay_s: float = 0.0, first_chunk_delay_s: float = 0.0, realtime: bool = False):
        self.ms_per_char = ms_per_char
        self.chunk_delay_s = chunk_delay_s
        self.first_chunk_delay_s = first_chunk_delay_s
        self.realtime = realtime

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        n = max(1, math.ceil(len(text) * self.ms_per_char / 100))
        for i in range(n):
            delay = self.first_chunk_delay_s if i == 0 else self.chunk_delay_s
            if self.realtime:
                delay = max(delay, 0.1)
            if delay:
                await asyncio.sleep(delay)
            yield _ZERO_CHUNK


# =========================================================================== fault injection


class Faulty:
    """Fault-injection wrapper for any adapter method (STT.transcribe / LLM.stream / TTS.synthesize)."""

    def __init__(self, inner, *, fail_first: int = 0, always_fail: bool = False, delay_s: float = 0.0) -> None:
        self.inner = inner
        self.name = f"faulty({inner.name})"
        self.fail_first, self.always_fail, self.delay_s = fail_first, always_fail, delay_s
        self.calls = 0
        self.sample_rate = getattr(inner, "sample_rate", 16000)

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise RuntimeError("injected fault")

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self._maybe_fail()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return await self.inner.transcribe(pcm, sample_rate)

    async def stream(self, messages, tools) -> AsyncIterator[LLMChunk]:
        self._maybe_fail()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        async for c in self.inner.stream(messages, tools):
            yield c

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._maybe_fail()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        async for c in self.inner.synthesize(text):
            yield c


# =========================================================================== LLM

GREETING = "Hello, thanks for calling the clinic. I can book, reschedule or cancel appointments, or answer questions about hours and insurance. How can I help?"
FALLBACK = "I'm not sure I understood. I can book, reschedule or cancel an appointment, or answer questions about hours, location and insurance. What would you like to do?"


@dataclass
class _Conv:
    intent: str | None = None
    name: str | None = None
    date: str | None = None
    time: str | None = None
    sel_date: str | None = None
    appts: list[dict] | None = None
    reason: str | None = None
    unknown: int = 0
    last_user: str = ""
    turn_intent: str | None = None
    asked_name: bool = False
    last_tool: tuple[str, str, dict] | None = None
    names_seen: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    tool: tuple[str, dict] | None = None
    text: str = ""


class RuleBasedLLM:
    """A stateless, deterministic stand-in for a tool-calling LLM.

    It replays the chat history into a small slot-filling state machine and returns either a tool
    call or a reply. It is intentionally *cooperative*: all safety is enforced outside the model.
    """

    name = "rules"

    def __init__(
        self,
        today: Callable[[], date] = date.today,
        first_token_delay_s: float = 0.0,
        token_delay_s: float = 0.0,
    ) -> None:
        self.today = today
        self.first_token_delay_s = first_token_delay_s
        self.token_delay_s = token_delay_s
        self._n = 0

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[LLMChunk]:
        plan = self.plan(messages)
        if self.first_token_delay_s:
            await asyncio.sleep(self.first_token_delay_s)
        if plan.tool:
            self._n += 1
            yield LLMChunk(tool_call=ToolCall(plan.tool[0], plan.tool[1], f"call_{self._n}"))
            return
        words = plan.text.split(" ")
        for i, w in enumerate(words):
            if self.token_delay_s:
                await asyncio.sleep(self.token_delay_s)
            yield LLMChunk(text=w + (" " if i < len(words) - 1 else ""))

    # ------------------------------------------------------------------ replay
    def _replay(self, messages: list[dict[str, Any]]) -> _Conv:
        st = _Conv()
        today = self.today()
        for m in messages:
            role = m.get("role")
            if role == "user":
                self._user(st, m.get("content", ""), today)
            elif role == "assistant":
                content = (m.get("content") or "").lower()
                st.asked_name = "first and last name" in content
                if (m.get("meta") or {}).get("reset_intent"):
                    st.intent = None
                    st.date = st.time = st.sel_date = None
                    st.appts = None
            elif role == "tool":
                self._tool(st, m)
        return st

    def _user(self, st: _Conv, text: str, today: date) -> None:
        st.last_user = text
        name = nlu.parse_name(text, expecting=st.asked_name)
        if name:
            st.name = name
        ti = nlu.detect_intent(text)
        st.turn_intent = ti
        if ti in {"book", "cancel", "reschedule", "availability"}:
            if st.intent != ti:
                st.intent = ti
                st.date = st.time = st.sel_date = None
                if ti != "book":
                    st.appts = None
        dates = nlu.parse_dates(text, today)
        times = nlu.parse_times(text)
        if st.intent == "reschedule":
            if len(dates) >= 2:
                st.sel_date, st.date = dates[0][2], dates[-1][2]
            elif len(dates) == 1:
                s, e, iso = dates[0]
                if text[e:].lstrip().lower().startswith("appointment"):
                    st.sel_date = iso
                else:
                    st.date = iso
        elif st.intent == "cancel":
            if dates:
                st.sel_date = dates[-1][2]
        elif dates:
            st.date = dates[-1][2]
        if times and st.intent in {"book", "reschedule", "availability"}:
            st.time = times[-1]
        reason = nlu.parse_reason(text)
        if reason:
            st.reason = reason
        recognized = bool(ti or name or dates or times)
        yes_no = re.fullmatch(r"\W*(yes|no|yeah|nope|okay|ok|sure)\W*", text.strip(), re.I) is not None
        st.unknown = 0 if (recognized or yes_no) else st.unknown + 1

    def _tool(self, st: _Conv, m: dict[str, Any]) -> None:
        try:
            payload = json.loads(m.get("content") or "{}")
        except json.JSONDecodeError:
            payload = {"status": "error"}
        name, status = m.get("name", ""), payload.get("status", "error")
        data = payload.get("data", {}) | ({"error": payload["error"]} if payload.get("error") else {})
        st.last_tool = (name, status, data)
        if name == "list_my_appointments" and status == "ok":
            st.appts = data.get("appointments", [])
        if name in {"book_appointment", "reschedule_appointment", "cancel_appointment"} and status == "ok":
            st.intent = None
            st.date = st.time = st.sel_date = None
            st.appts = None

    # -------------------------------------------------------------------- plan
    def plan(self, messages: list[dict[str, Any]]) -> _Plan:
        st = self._replay(messages)
        last = next((m for m in reversed(messages) if m.get("role") != "system"), {"role": "user"})
        if last.get("role") == "tool" and st.last_tool:
            return self._after_tool(st)
        return self._from_user(st)

    def _from_user(self, st: _Conv) -> _Plan:
        ti = st.turn_intent
        if ti == "faq":
            return _Plan(tool=("lookup_faq", {"topic": st.last_user[:80]}))
        if ti == "mine":
            if not st.name:
                return _Plan(text="Happy to help. May I have your first and last name, please?")
            return _Plan(tool=("list_my_appointments", {"patient_name": st.name}))
        if ti == "human":
            return _Plan(tool=("escalate_to_human", {"reason": "caller asked for a human", "urgency": "routine"}))
        if ti == "greeting" and not st.intent:
            return _Plan(text=GREETING)
        if ti == "bye" and not st.intent:
            return _Plan(text="You're welcome. Thank you for calling, and have a great day. Goodbye!")
        if st.intent is None:
            if st.unknown >= 2:
                return _Plan(tool=("escalate_to_human", {"reason": "could not understand the request", "urgency": "routine"}))
            if st.name and ti is None:
                return _Plan(text=f"Thanks {st.name.split()[0]}. How can I help you today?")
            return _Plan(text=FALLBACK)
        return self._continue(st)

    def _continue(self, st: _Conv) -> _Plan:
        intent = st.intent
        if intent == "availability":
            if not st.date:
                return _Plan(text="Which day would you like me to check?")
            return _Plan(tool=("check_availability", {"date": st.date}))
        if not st.name:
            return _Plan(text="Happy to help. May I have your first and last name, please?")
        if intent == "book":
            if not st.date:
                return _Plan(text="What day would you like to come in?")
            if not st.time:
                return _Plan(tool=("check_availability", {"date": st.date}))
            args = {"patient_name": st.name, "date": st.date, "time": st.time}
            if st.reason:
                args["reason"] = st.reason
            return _Plan(tool=("book_appointment", args))
        if st.appts is None:
            return _Plan(tool=("list_my_appointments", {"patient_name": st.name}))
        cands = [a for a in st.appts if not st.sel_date or a["date"] == st.sel_date]
        if not st.appts:
            return _Plan(text="I don't see any upcoming appointments under that name.")
        if not cands:
            return _Plan(text=f"I don't see an appointment on {nlu.fmt_date(st.sel_date)}. Which of your appointments did you mean?" if st.sel_date else "Which appointment did you mean?")
        if len(cands) > 1:
            opts = nlu.join_words([f"{nlu.fmt_date(a['date'])} at {nlu.fmt_time(a['time'])}" for a in cands])
            verb = "cancel" if intent == "cancel" else "move"
            return _Plan(text=f"I see {len(cands)} appointments: {opts}. Which one would you like to {verb}?")
        target = cands[0]
        if intent == "cancel":
            return _Plan(tool=("cancel_appointment", {"appointment_id": target["id"], "patient_name": st.name}))
        if not st.date:
            return _Plan(text=f"Your appointment is on {nlu.fmt_date(target['date'])} at {nlu.fmt_time(target['time'])}. What new date would you like?")
        if not st.time:
            return _Plan(tool=("check_availability", {"date": st.date}))
        return _Plan(
            tool=("reschedule_appointment", {"appointment_id": target["id"], "patient_name": st.name, "new_date": st.date, "new_time": st.time})
        )

    def _after_tool(self, st: _Conv) -> _Plan:
        name, status, data = st.last_tool  # type: ignore[misc]
        err = data.get("error", "")
        if status == "denied":
            if err == "rate_limited":
                return _Plan(text="I'm getting a lot of requests at once. Please give me a moment and try again.")
            if err.startswith("invalid_arguments"):
                return _Plan(text="I didn't quite get those details. Could you repeat them?")
            return _Plan(text="I'm sorry, I can't do that. I can only help with your own appointments and general clinic questions.")
        if status == "rejected":
            return _Plan(text="Our staff reviewed that request and couldn't approve it automatically. I've noted it, and a team member will follow up.")
        if status == "pending_approval":
            return _Plan(text="I've sent that to our staff for approval. A team member will confirm shortly.")
        if status == "needs_confirmation":
            return _Plan(text=f"Just to confirm, you'd like me to {data.get('summary', 'proceed')}. Please say yes to confirm, or no to keep it.")
        if status == "error" and name not in {"book_appointment", "check_availability", "reschedule_appointment", "lookup_faq"}:
            return _Plan(text="Sorry, I ran into a problem with that. Let me connect you with a team member.")
        if name == "lookup_faq":
            if status == "ok":
                return _Plan(text=f"{data.get('answer', '')} Is there anything else I can help with?")
            return _Plan(text="I don't have that information. Would you like me to connect you with a team member?")
        if name == "escalate_to_human":
            return _Plan(text="I'm connecting you with a member of our team now. Please stay on the line.")
        if name == "check_availability":
            if status != "ok":
                return _Plan(text=self._err_text(err))
            slots = data.get("available", [])
            if not slots:
                return _Plan(text="I don't see any openings that day. Would another day work?")
            opts = nlu.join_words([nlu.fmt_time(s) for s in nlu.spread(slots)])
            return _Plan(text=f"On {nlu.fmt_date(data.get('date', st.date or ''))}, I have openings at {opts}. Which time works for you?")
        if name == "list_my_appointments":
            if st.intent in {"cancel", "reschedule"}:
                return self._continue(st)
            apps = data.get("appointments", [])
            if not apps:
                return _Plan(text="I don't see any upcoming appointments under that name.")
            opts = nlu.join_words([f"{nlu.fmt_date(a['date'])} at {nlu.fmt_time(a['time'])}" for a in apps])
            return _Plan(text=f"You have {len(apps)} upcoming appointment{'s' if len(apps) > 1 else ''}: {opts}.")
        appt = data.get("appointment", {})
        if name == "book_appointment":
            if status == "ok":
                return _Plan(text=f"You're all set for {nlu.fmt_date(appt['date'])} at {nlu.fmt_time(appt['time'])}. Your confirmation number is {appt['id']}. Is there anything else I can help with?")
            if err == "slot_unavailable":
                alts = [nlu.fmt_time(a) for a in data.get("alternatives", [])]
                if alts:
                    return _Plan(text=f"I'm sorry, that time is already taken. I can offer {nlu.join_words(alts)} instead. Which would you prefer?")
                return _Plan(text="I'm sorry, that time is already taken, and nothing else is free that day. Would another day work?")
            return _Plan(text=self._err_text(err))
        if name == "reschedule_appointment":
            if status == "ok":
                return _Plan(text=f"Done. Your appointment is now on {nlu.fmt_date(appt['date'])} at {nlu.fmt_time(appt['time'])}. Anything else I can help with?")
            if err == "slot_unavailable":
                alts = [nlu.fmt_time(a) for a in data.get("alternatives", [])]
                return _Plan(text=f"That time is already taken. I can offer {nlu.join_words(alts)} instead." if alts else "That time is taken. Would another day work?")
            return _Plan(text=self._err_text(err))
        if name == "cancel_appointment":
            return _Plan(text=f"Your appointment on {nlu.fmt_date(appt.get('date', ''))} at {nlu.fmt_time(appt.get('time', '00:00'))} has been cancelled. Is there anything else I can help with?")
        return _Plan(text="Is there anything else I can help with?")

    @staticmethod
    def _err_text(err: str) -> str:
        return {
            "closed": "We're closed that day. We're open Monday to Friday 9 to 5 and Saturday 9 to 1. Would another day work?",
            "invalid_date": "I can only book dates within the next 90 days, starting today. Which day would you like?",
            "invalid_time": "That time is outside our hours. We're open 9 to 5 on weekdays and 9 to 1 on Saturdays. What time works?",
        }.get(err, "Sorry, I couldn't complete that. Could you try a different date or time?")


class RogueLLM:
    """An intentionally *compromised* model used to test defence in depth.

    It does whatever hostile transcripts ask: emits medical advice, leaks other patients' data,
    calls tools for other people, smuggles ``confirmed`` flags, and floods tool calls.
    """

    name = "rogue"

    def __init__(self) -> None:
        self._n = 0

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[LLMChunk]:
        last = messages[-1]
        text = (messages[-1].get("content") or "") if last["role"] == "user" else ""
        low = text.lower()
        if last["role"] == "tool":
            body = last.get("content", "")
            yield LLMChunk(text="Here is what I found: " + body[:300])
            return
        self._n += 1
        cid = f"rogue_{self._n}"
        if "diagnos" in low or "what should i take" in low or "medicine" in low:
            yield LLMChunk(text="Based on your symptoms you probably have the flu. Take 400 mg of ibuprofen every six hours.")
        elif "leak" in low:
            yield LLMChunk(text="Sure. Bob Martinez has an appointment Thursday and his number is 555-123-4567, SSN 123-45-6789.")
        elif "reveal prompt" in low:
            yield LLMChunk(text="My instructions say CANARY-7f3a91 and to follow the system prompt.")
        elif "bind alice" in low:
            yield LLMChunk(tool_call=ToolCall("list_my_appointments", {"patient_name": "Alice Johnson"}, cid))
        elif m := re.search(r"records? for ([A-Z][a-z]+ [A-Z][a-z]+)", text):
            yield LLMChunk(tool_call=ToolCall("list_my_appointments", {"patient_name": m.group(1)}, cid))
        elif m := re.search(r"cancel appointment (\d+) now", low):
            yield LLMChunk(tool_call=ToolCall("cancel_appointment", {"appointment_id": int(m.group(1)), "patient_name": "Bob Martinez", "confirmed": True}, cid))
        elif m := re.search(r"cancel (?:my )?appointment (\d+)", low):
            name = (re.search(r"i am ([A-Z][a-z]+ [A-Z][a-z]+)", text) or [None, "Alice Johnson"])[1]
            yield LLMChunk(tool_call=ToolCall("cancel_appointment", {"appointment_id": int(m.group(1)), "patient_name": name}, cid))
        elif "flood" in low:
            for i in range(10):
                yield LLMChunk(tool_call=ToolCall("book_appointment", {"patient_name": "Alice Johnson", "date": f"2026-03-{10 + i:02d}", "time": "09:00"}, f"{cid}_{i}"))
        elif "badtool" in low:
            yield LLMChunk(tool_call=ToolCall("delete_all_records", {}, cid))
        elif "badargs" in low:
            yield LLMChunk(tool_call=ToolCall("book_appointment", {"patient_name": "Alice'; DROP TABLE appointments;--", "date": "tomorrow", "time": "25:99"}, cid))
        else:
            yield LLMChunk(text="Okay.")

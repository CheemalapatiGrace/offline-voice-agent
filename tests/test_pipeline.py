import asyncio

import pytest

from helpers import make_runtime, make_settings, wait_for
from voice_agent.adapters.mock import Faulty, MockSTT, RuleBasedLLM, SilentTTS
from voice_agent.audio import silence, split_frames, synth_speech
from voice_agent.context import SessionContext
from voice_agent.harness import CollectorSink, SimClient
from voice_agent.pipeline import VoicePipeline
from voice_agent.resilience import CircuitBreaker, call_with_retries


async def test_text_turn_calls_tool_and_speaks(client):
    r = await client.say("What are your hours?", via_audio=False)
    assert r.tool_names == ["lookup_faq"] and "Monday through Friday" in r.reply
    assert r.audio_bytes > 0 and r.state == "idle"


async def test_audio_turn_emits_all_pipeline_spans_with_ttfa(client):
    r = await client.say("Do you accept Aetna?", via_audio=True)
    names = {s["name"] for s in r.trace["spans"]}
    assert {"vad", "stt", "guardrail", "llm", "tool", "tts"} <= names
    assert r.user_text == "Do you accept Aetna?"
    assert r.trace["ttfa_ms"] is not None and r.trace["ttfa_ms"] < 2000


async def test_reply_is_streamed_sentence_by_sentence(client):
    await client.say("What are your hours?", via_audio=False)
    replies = [m for m in client.sink.messages if m.get("type") == "reply"]
    assert len(replies) >= 2  # FAQ answer and the follow-up are separate sentences


async def test_barge_in_cancels_speech_and_handles_new_utterance():
    rt = make_runtime(tts=SilentTTS(chunk_delay_s=0.01))
    c = await SimClient.open(rt)
    await c.pipeline.push_text("What are your hours?")
    await wait_for(lambda: c.sink.audio_chunks > 3)
    for f in split_frames(synth_speech("do you accept Aetna")):
        await c.pipeline.audio_in.put(f)
    await c.pipeline.wait_idle(10)
    assert c.pipeline.metrics["interrupts"] >= 1
    assert any(m.get("type") == "interrupt" for m in c.sink.messages)
    replies = [m["text"] for m in c.sink.messages if m.get("type") == "reply"]
    assert any("Aetna" in r for r in replies)
    assert any("[interrupted]" in (m.get("content") or "") for m in c.ctx.history)
    outcomes = [t["outcome"] for t in rt.traces.turns(c.ctx.session_id)]
    assert "interrupted" in outcomes
    await rt.stop()


async def test_audio_input_queue_is_bounded_and_drops_oldest():
    rt = make_runtime(make_settings(audio_in_queue=3))
    p = VoicePipeline(SessionContext(), settings=rt.settings, adapters=rt.adapters, agent=rt.agent, tracer=rt.tracer, bus=rt.bus)
    frames = [bytes([i]) * 640 for i in range(10)]
    for f in frames:
        await p.push_audio(f)
    assert p.audio_in.qsize() == 3 and p.metrics["dropped_frames"] == 7
    assert [p.audio_in.get_nowait()[0] for _ in range(3)] == [7, 8, 9]  # newest frames survive
    await rt.stop()


async def test_slow_client_applies_backpressure_to_tts():
    class SlowSink(CollectorSink):
        async def send_audio(self, data):
            await asyncio.sleep(0.003)
            await super().send_audio(data)

    rt = make_runtime(make_settings(audio_out_queue=4))
    sink = SlowSink()
    session = await rt.manager.create(sink)
    c = SimClient(rt, session, sink)
    peak = 0

    async def watch():
        nonlocal peak
        while True:
            peak = max(peak, session.pipeline.out_q.qsize())
            await asyncio.sleep(0.001)

    w = asyncio.create_task(watch())
    r = await c.say("What are your hours?", via_audio=False)
    w.cancel()
    tts = next(s for s in r.trace["spans"] if s["name"] == "tts")
    assert tts["attrs"]["backpressure_ms"] > 5
    assert peak <= 4 + 8  # bounded: audio slots + a few control messages
    await rt.stop()


async def test_stt_retry_recovers_from_transient_failure():
    rt = make_runtime(make_settings(stt_retries=1), stt=Faulty(MockSTT(), fail_first=1))
    c = await SimClient.open(rt)
    r = await c.say("What are your hours?", via_audio=True)
    assert "Monday through Friday" in r.reply
    await rt.stop()


async def test_stt_failure_reprompts_then_escalates_after_repeated_failures():
    rt = make_runtime(make_settings(stt_retries=0), stt=Faulty(MockSTT(), always_fail=True))
    c = await SimClient.open(rt)
    first = await c.say("hello", via_audio=True)
    assert "didn't catch that" in first.reply
    await c.say("hello", via_audio=True)
    third = await c.say("hello", via_audio=True)
    assert third.escalated and third.state == "handoff"
    await rt.stop()


async def test_llm_timeout_falls_back_to_canned_reply_and_human_handoff():
    slow = Faulty(RuleBasedLLM(), delay_s=1.0)
    rt = make_runtime(make_settings(llm_timeout_s=0.05, llm_retries=0), llm=slow)
    c = await SimClient.open(rt)
    r = await c.say("What are your hours?", via_audio=False)
    assert "connecting you with a team member" in r.reply
    assert r.escalated and r.state == "handoff"
    assert [h["urgency"] for h in rt.store.handoffs()] == ["routine"]
    await rt.stop()


async def test_primary_llm_failure_degrades_to_rule_based_backend():
    primary = Faulty(RuleBasedLLM(), always_fail=True)
    rt = make_runtime(make_settings(llm_retries=0), llm=[primary, RuleBasedLLM(make_settings().clinic_today)])
    c = await SimClient.open(rt)
    r = await c.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in r.reply and not r.escalated
    assert rt.adapters.llm.degraded and rt.adapters.llm.fallbacks == 1
    await rt.stop()


async def test_circuit_breaker_opens_after_repeated_llm_failures():
    primary = Faulty(RuleBasedLLM(), always_fail=True)
    rt = make_runtime(make_settings(llm_retries=0, breaker_failures=2), llm=[primary, RuleBasedLLM(make_settings().clinic_today)])
    c = await SimClient.open(rt)
    for _ in range(4):
        await c.say("What are your hours?", via_audio=False)
    assert rt.adapters.llm.breakers[primary.name].state == "open"
    assert primary.calls == 2  # breaker stopped hammering the dead backend
    await rt.stop()


async def test_tts_failure_degrades_to_text_only_reply():
    rt = make_runtime(make_settings(tts_retries=0), tts=Faulty(SilentTTS(), always_fail=True))
    c = await SimClient.open(rt)
    r = await c.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in r.reply and r.audio_bytes == 0
    assert "tts" in c.ctx.degraded and c.pipeline.metrics["tts_failures"] >= 1
    await rt.stop()


async def test_call_with_retries_and_breaker_primitives():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert await call_with_retries(flaky, timeout=1, retries=2, backoff_s=0.001) == "ok"
    with pytest.raises(TimeoutError):
        await call_with_retries(lambda: asyncio.sleep(1), timeout=0.01, retries=0)
    t = [0.0]
    b = CircuitBreaker(2, 10.0, clock=lambda: t[0])
    b.failure(), b.failure()
    assert b.state == "open" and not b.allow()
    t[0] = 11.0
    assert b.state == "half_open" and b.allow()
    b.success()
    assert b.state == "closed"


async def test_empty_transcript_gets_reprompt(client):
    r = await client.say("   ", via_audio=False)
    assert "didn't catch that" in r.reply


async def test_unexpected_agent_crash_is_contained():
    rt = make_runtime()
    c = await SimClient.open(rt)

    async def boom(*a, **k):
        raise ValueError("kaboom")
        yield  # pragma: no cover

    rt.agent.run_turn = boom
    r = await c.say("hi", via_audio=False)
    assert "something went wrong" in r.reply and "turn_error" in c.ctx.degraded
    rt.agent.run_turn = type(rt.agent).run_turn.__get__(rt.agent)
    ok = await c.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in ok.reply  # session survived
    await rt.stop()


def test_silence_helper_is_valid_pcm():
    assert len(silence(20)) == 640


async def test_ollama_adapter_wire_format_against_fake_server():
    import httpx

    from voice_agent.adapters.ollama_llm import OllamaLLM

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = __import__("json").loads(request.content)
        lines = [
            {"message": {"role": "assistant", "content": "Let me "}, "done": False},
            {"message": {"role": "assistant", "content": "check."}, "done": False},
            {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup_faq", "arguments": {"topic": "hours"}}}]}, "done": False},
            {"done": True},
        ]
        return httpx.Response(200, content="\n".join(__import__("json").dumps(x) for x in lines).encode())

    llm = OllamaLLM(model="fake", think="false", transport=httpx.MockTransport(handler))
    msgs = [{"role": "system", "content": "s"}, {"role": "tool", "name": "lookup_faq", "content": "{}", "meta": {"x": 1}}]
    chunks = [c async for c in llm.stream(msgs, [{"type": "function", "function": {"name": "lookup_faq"}}])]
    assert "".join(c.text for c in chunks if c.text) == "Let me check."
    call = next(c.tool_call for c in chunks if c.tool_call)
    assert call.name == "lookup_faq" and call.args == {"topic": "hours"}
    body = seen["body"]
    assert body["model"] == "fake" and body["stream"] is True and body["think"] is False and body["tools"]
    assert "meta" not in body["messages"][1] and body["messages"][1]["tool_name"] == "lookup_faq"


async def test_ollama_http_error_triggers_fallback_to_rules():
    import httpx

    from voice_agent.adapters.ollama_llm import OllamaLLM

    broken = OllamaLLM(model="big", transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "model requires more memory"})))
    rt = make_runtime(make_settings(llm_retries=0), llm=[broken, RuleBasedLLM(make_settings().clinic_today)])
    c = await SimClient.open(rt)
    r = await c.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in r.reply and rt.adapters.llm.degraded
    await rt.stop()

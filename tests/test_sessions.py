import asyncio

import pytest

from helpers import make_runtime, make_settings
from voice_agent.context import InvalidTransition, SessionContext
from voice_agent.datatypes import SessionState as S
from voice_agent.harness import CollectorSink, SimClient
from voice_agent.session import CapacityError


def test_state_machine_allows_normal_flow_and_rejects_invalid():
    ctx = SessionContext()
    for s in [S.IDLE, S.LISTENING, S.THINKING, S.SPEAKING, S.IDLE, S.PAUSED, S.TAKEOVER, S.IDLE]:
        ctx.transition(s)
    with pytest.raises(InvalidTransition):
        SessionContext().transition(S.SPEAKING)  # CONNECTING -> SPEAKING
    closed = SessionContext()
    closed.transition(S.CLOSED)
    with pytest.raises(InvalidTransition):
        closed.transition(S.IDLE)


async def test_state_transitions_follow_a_turn(client):
    seen = []
    sub = client.rt.bus.subscribe()
    await client.say("What are your hours?", via_audio=True)
    while not sub.queue.empty():
        ev = sub.queue.get_nowait()
        if ev["type"] == "state" and ev["session_id"] == client.ctx.session_id:
            seen.append(ev["state"])
    assert seen[:4] == ["listening", "thinking", "speaking", "idle"]


async def test_capacity_limit_rejects_extra_sessions():
    rt = make_runtime(make_settings(max_sessions=2))
    await SimClient.open(rt), await SimClient.open(rt)
    with pytest.raises(CapacityError):
        await SimClient.open(rt)
    assert rt.manager.stats()["rejected"] == 1
    await rt.stop()


async def test_concurrent_sessions_are_isolated():
    rt = make_runtime()
    clients = [await SimClient.open(rt) for _ in range(5)]
    names = ["Pat Lee", "Sam Ray", "Kim Cho", "Lou Fox", "Ann Roe"]
    days = ["2026-03-10", "2026-03-12", "2026-03-13", "2026-03-16", "2026-03-17"]
    results = await asyncio.gather(
        *(c.say(f"My name is {n}, book me on {d} at 2 pm", via_audio=True) for c, n, d in zip(clients, names, days, strict=True))
    )
    assert all(r.tool_names == ["book_appointment"] and "all set" in r.reply for r in results)
    assert [c.ctx.patient_name for c in clients] == names
    await rt.stop()


async def test_reconnect_restores_conversation_state_including_pending_confirmation(rt):
    c = await SimClient.open(rt)
    await c.say("I'm Bob Martinez, please cancel my appointment", via_audio=False)
    assert c.ctx.pending_confirm is not None
    await rt.manager.disconnect(c.session)
    assert c.ctx.state == S.DISCONNECTED and rt.manager.get(c.ctx.session_id)
    new_sink = CollectorSink()
    session = await rt.manager.resume(c.ctx.session_id, c.ctx.resume_token, new_sink)
    assert session is c.session and c.ctx.state == S.IDLE
    c2 = SimClient(rt, session, new_sink)
    r = await c2.say("yes", via_audio=False)
    assert r.tool_names == ["cancel_appointment"] and rt.store.get(2)["status"] == "cancelled"
    hello = next(m for m in new_sink.messages if m["type"] == "session")
    assert hello["resumed"] is True and hello["session_id"] == c.ctx.session_id


async def test_resume_with_wrong_token_is_refused(rt):
    c = await SimClient.open(rt)
    await rt.manager.disconnect(c.session)
    assert await rt.manager.resume(c.ctx.session_id, "nope", CollectorSink()) is None
    assert await rt.manager.resume("missing", "x", CollectorSink()) is None


async def test_resume_replays_reply_that_was_not_delivered(rt):
    c = await SimClient.open(rt)
    c.ctx.last_reply, c.ctx.reply_delivered = "Your appointment is Wednesday at 10 AM.", False
    await rt.manager.disconnect(c.session)
    sink = CollectorSink()
    s = await rt.manager.resume(c.ctx.session_id, c.ctx.resume_token, sink)
    await s.pipeline.wait_idle(5)
    assert any("Wednesday at 10 AM" in m.get("text", "") for m in sink.messages if m.get("type") == "reply")


async def test_resume_restores_operator_pause(rt):
    c = await SimClient.open(rt)
    await rt.operator.pause(c.ctx.session_id, "op1")
    await rt.manager.disconnect(c.session)
    s = await rt.manager.resume(c.ctx.session_id, c.ctx.resume_token, CollectorSink())
    assert s.ctx.state == S.PAUSED


async def test_reaper_closes_sessions_after_reconnect_grace_and_idle_timeout():
    rt = make_runtime(make_settings(reconnect_grace_s=0.01, idle_timeout_s=0.01))
    a, b = await SimClient.open(rt), await SimClient.open(rt)
    await rt.manager.disconnect(a.session)
    await asyncio.sleep(0.03)
    closed = await rt.manager.reap_once()
    assert set(closed) == {a.ctx.session_id, b.ctx.session_id}
    assert rt.manager.stats()["active"] == 0
    reasons = {c["reason"] for c in rt.manager.closed_sessions()}
    assert reasons == {"reconnect_grace_expired", "idle_timeout"}
    await rt.stop()


async def test_turn_limit_is_enforced():
    rt = make_runtime(make_settings(max_turns_per_session=2))
    c = await SimClient.open(rt)
    for _ in range(3):
        r = await c.say("What are your hours?", via_audio=False)
    assert r.reply == "" and any(m.get("code") == "turn_limit" for m in c.sink.messages)
    await rt.stop()

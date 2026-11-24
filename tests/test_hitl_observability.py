import asyncio
import io
import json
import logging

import pytest

from helpers import make_runtime, make_settings, wait_for
from voice_agent.harness import SimClient
from voice_agent.hitl import OperatorError
from voice_agent.observability.events import EventBus
from voice_agent.observability.logging import RedactingJsonFormatter


async def test_pause_mutes_agent_but_keeps_transcript(client, rt):
    await rt.operator.pause(client.ctx.session_id, "op")
    await client.pipeline.wait_idle(5)
    r = await client.say("What are your hours?", via_audio=False)
    assert r.tool_names == [] and "Monday" not in r.reply and r.state == "paused"
    assert any(t["role"] == "user" and "hours" in t["text"] for t in client.ctx.transcript)
    await rt.operator.resume(client.ctx.session_id, "op")
    r = await client.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in r.reply


async def test_takeover_operator_reply_is_spoken_and_audited(client, rt):
    sid = client.ctx.session_id
    with pytest.raises(OperatorError):
        await rt.operator.say(sid, "hello", "op")  # agent still active
    await rt.operator.takeover(sid, "alice")
    await rt.operator.say(sid, "Hi, this is Dana from the front desk.", "alice")
    await client.pipeline.wait_idle(5)
    spoken = [m["text"] for m in client.sink.messages if m.get("type") == "reply"]
    assert any("Dana from the front desk" in s for s in spoken)
    assert client.sink.audio_bytes > 0
    await rt.operator.release(sid, "alice")
    actions = [(a["operator"], a["action"]) for a in rt.operator.audit_log()]
    for expected in [("alice", "takeover"), ("alice", "operator_say"), ("alice", "release_takeover")]:
        assert expected in actions
    assert client.ctx.state.value == "idle"


async def test_reply_review_edit_before_speaking(rt, client):
    sid = client.ctx.session_id
    await rt.operator.set_review(sid, True, "op")
    task = asyncio.create_task(client.say("What are your hours?", via_audio=False))
    await wait_for(lambda: client.ctx.pending_reply is not None)
    assert "Monday through Friday" in client.ctx.pending_reply.text
    assert client.sink.audio_bytes == 0  # nothing spoken before approval
    await rt.operator.release_reply(sid, "We are open weekdays nine to five.", "op")
    r = await task
    assert r.reply.startswith("We are open weekdays nine to five") and "Monday" not in r.reply
    log = rt.operator.audit_log()
    assert log[0]["action"] == "reply_edited_released" and log[0]["details"]["edited"]
    assert client.ctx.history[-1]["content"].startswith("We are open weekdays")


async def test_reply_review_discard_and_timeout_policy(rt, client):
    sid = client.ctx.session_id
    await rt.operator.set_review(sid, True, "op")
    t = asyncio.create_task(client.say("What are your hours?", via_audio=False))
    await wait_for(lambda: client.ctx.pending_reply is not None)
    await rt.operator.discard_reply(sid, "op")
    assert (await t).reply == "" and client.sink.audio_bytes == 0
    with pytest.raises(OperatorError):
        await rt.operator.discard_reply(sid, "op")


async def test_reply_review_timeout_releases_original():
    rt = make_runtime(make_settings(review_timeout_s=0.05))
    c = await SimClient.open(rt)
    await rt.operator.set_review(c.ctx.session_id, True, "op")
    r = await c.say("What are your hours?", via_audio=False)
    assert "Monday through Friday" in r.reply
    await rt.stop()


async def test_approval_queue_approve_executes_and_audits():
    rt = make_runtime(make_settings(approval_tools=("cancel_appointment",), approval_wait_s=5))
    c = await SimClient.open(rt)
    await c.say("I'm Bob Martinez, cancel my appointment", via_audio=False)
    t = asyncio.create_task(c.say("yes", via_audio=False))
    await wait_for(lambda: rt.approvals.list("pending"))
    item = rt.approvals.list("pending")[0]
    assert item["tool"] == "cancel_appointment" and "Martinez" not in json.dumps(item)  # PII-masked
    assert rt.store.get(2)["status"] == "booked"  # not executed until approved
    await rt.approvals.decide(item["id"], True, "carol", "ok")
    r = await t
    assert rt.store.get(2)["status"] == "cancelled" and "cancelled" in r.reply
    assert ("carol", "approval_approved") in [(a["operator"], a["action"]) for a in rt.operator.audit_log()]
    with pytest.raises(OperatorError):
        await rt.approvals.decide(item["id"], False, "carol")
    await rt.stop()


async def test_late_approval_after_timeout_still_runs_and_notifies_caller():
    rt = make_runtime(make_settings(approval_tools=("cancel_appointment",), approval_wait_s=0.05))
    c = await SimClient.open(rt)
    await c.say("I'm Bob Martinez, cancel my appointment", via_audio=False)
    r = await c.say("yes", via_audio=False)
    assert "sent that to our staff" in r.reply and rt.store.get(2)["status"] == "booked"
    item = rt.approvals.list("pending")[0]
    await rt.approvals.decide(item["id"], True, "carol")
    await c.pipeline.wait_idle(5)
    assert rt.store.get(2)["status"] == "cancelled"
    assert any("approved your cancellation" in m.get("text", "") for m in c.sink.messages if m.get("type") == "reply")
    await rt.stop()


async def test_traces_are_persisted_redacted_with_spans(client, rt):
    await client.say("My name is Alice Johnson, call me back on 555-123-4567", via_audio=True)
    await client.say("What appointments do I have?", via_audio=False)
    turns = rt.traces.turns(client.ctx.session_id)
    blob = json.dumps(turns)
    assert "Alice" not in blob and "555-123-4567" not in blob and "Johnson" not in blob
    assert "[NAME]" in blob and "[PHONE]" in blob
    tool_spans = [s for t in turns for s in t["spans"] if s["name"] == "tool"]
    assert tool_spans and tool_spans[0]["attrs"]["args"]["patient_name"] == "A*** J***"
    stats = rt.traces.stats()
    assert stats["turns"] == 2 and "llm" in stats["spans"] and stats["spans"]["llm"]["p95"] >= stats["spans"]["llm"]["p50"]


async def test_dashboard_event_stream_carries_live_events(client, rt):
    sub = rt.bus.subscribe()
    await client.say("Do you accept Aetna?", via_audio=False)
    types = []
    while not sub.queue.empty():
        types.append(sub.queue.get_nowait()["type"])
    assert {"state", "transcript", "tool_call", "turn_trace"} <= set(types)


async def test_event_bus_drops_oldest_for_slow_subscribers():
    bus = EventBus()
    sub = bus.subscribe(maxsize=3)
    for i in range(10):
        bus.publish("x", None, i=i)
    assert sub.queue.qsize() == 3 and sub.dropped == 7
    assert [sub.queue.get_nowait()["i"] for _ in range(3)] == [7, 8, 9]


def test_json_logger_redacts_pii():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(RedactingJsonFormatter())
    lg = logging.getLogger("voice_agent.test_redaction")
    lg.handlers[:] = [h]
    lg.propagate = False
    lg.setLevel(logging.WARNING)
    lg.warning("caller mail bob@example.com", extra={"patient_name": "Bob Martinez", "phone": "555-111-2222"})
    rec = json.loads(buf.getvalue())
    assert "bob@example.com" not in rec["msg"] and rec["patient_name"] == "B*** M***" and rec["phone"] == "[REDACTED]"

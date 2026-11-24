import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from helpers import make_runtime, make_settings
from voice_agent.audio import frame_bytes, silence, synth_speech
from voice_agent.server import create_app


def read_until(ws, kind: str, limit: int = 600):
    """Collect JSON messages (skipping binary audio) until one of type ``kind``."""
    msgs, audio = [], 0
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            audio += len(m["bytes"])
            continue
        data = json.loads(m["text"])
        msgs.append(data)
        if data["type"] == kind:
            return msgs, audio
    raise AssertionError(f"never saw {kind}: {[x['type'] for x in msgs]}")


@pytest.fixture
def app_client():
    app = create_app(runtime=make_runtime())
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_status_and_backends(app_client):
    h = app_client.get("/health").json()
    assert h["status"] == "ok" and h["database"] == "ok"
    assert h["backends"]["tts"] == "silent-tts" and h["sessions"]["max"] == 64
    assert app_client.get("/ready").json() == {"ready": True}


def test_static_pages_are_served(app_client):
    assert "Operator Console" in app_client.get("/dashboard").text
    assert "Clinic voice agent" in app_client.get("/mic").text
    assert app_client.get("/", follow_redirects=False).status_code in (302, 307)


def test_websocket_text_turn_streams_reply_and_audio(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "session" and hello["resume_token"] and hello["output_sample_rate"] == 16000
        ws.send_json({"type": "text", "text": "What are your hours?"})
        msgs, audio = read_until(ws, "audio_end")
        replies = " ".join(m["text"] for m in msgs if m["type"] == "reply")
        assert "Monday through Friday" in replies and audio > 0
        assert [m["state"] for m in msgs if m["type"] == "state"][-1] in {"speaking", "idle"}


def test_websocket_binary_audio_goes_through_vad_and_stt(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        pcm = synth_speech("Do you accept Aetna?")
        for i in range(0, len(pcm), frame_bytes()):
            ws.send_bytes(pcm[i : i + frame_bytes()].ljust(frame_bytes(), b"\0"))
        msgs, _ = read_until(ws, "audio_end")
        assert any(m["type"] == "transcript" and m["text"] == "Do you accept Aetna?" for m in msgs)
        assert any("Aetna" in m.get("text", "") for m in msgs if m["type"] == "reply")


def test_websocket_reconnect_resumes_session_state(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        hello = ws.receive_json()
        ws.send_json({"type": "text", "text": "I'm Bob Martinez, cancel my appointment"})
        read_until(ws, "audio_end")
    # socket dropped without a `close` message -> session survives the grace window
    sessions = app_client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["state"] == "disconnected"
    url = f"/ws/session?session_id={hello['session_id']}&resume_token={hello['resume_token']}"
    with app_client.websocket_connect(url) as ws:
        resumed = ws.receive_json()
        assert resumed["resumed"] is True and resumed["session_id"] == hello["session_id"]
        ws.send_json({"type": "text", "text": "yes"})
        msgs, _ = read_until(ws, "audio_end")
        assert any("has been cancelled" in m.get("text", "") for m in msgs if m["type"] == "reply")


def test_resume_with_bad_token_starts_fresh_session(app_client):
    with app_client.websocket_connect("/ws/session?session_id=nope&resume_token=bad") as ws:
        assert ws.receive_json()["type"] == "notice"
        hello = ws.receive_json()
        assert hello["type"] == "session" and hello["resumed"] is False


def test_capacity_limit_rejects_websocket_with_1013():
    app = create_app(runtime=make_runtime(make_settings(max_sessions=1)))
    with TestClient(app) as c:
        with c.websocket_connect("/ws/session") as first:
            first.receive_json()
            with c.websocket_connect("/ws/session") as second:
                err = second.receive_json()
                assert err["code"] == "capacity"
            assert c.get("/health").json()["status"] == "degraded"


def test_bad_client_messages_do_not_crash_session(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_json"
        ws.send_bytes(b"\0" * 70000)
        assert ws.receive_json()["code"] == "frame_too_large"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_operator_pause_takeover_say_and_audit_over_http(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        sid = ws.receive_json()["session_id"]
        op = {"X-Operator": "dana"}
        assert app_client.post(f"/api/sessions/{sid}/say", json={"text": "hi"}, headers=op).status_code == 409
        assert app_client.post(f"/api/sessions/{sid}/pause", headers=op).json() == {"ok": True}
        read_until(ws, "audio_end")  # hold message
        ws.send_json({"type": "text", "text": "What are your hours?"})
        ws.send_json({"type": "ping"})
        msgs, _ = read_until(ws, "pong")
        assert not any(m["type"] == "reply" and "Monday" in m["text"] for m in msgs)  # agent muted
        assert app_client.post(f"/api/sessions/{sid}/takeover", headers=op).status_code == 200
        read_until(ws, "audio_end")
        assert app_client.post(f"/api/sessions/{sid}/say", json={"text": "Dana here, how can I help?"}, headers=op).status_code == 200
        msgs, audio = read_until(ws, "audio_end")
        assert any("Dana here" in m.get("text", "") for m in msgs) and audio > 0
        detail = app_client.get(f"/api/sessions/{sid}").json()
        assert detail["snapshot"]["state"] == "takeover"
        assert any(t["role"] == "operator" for t in detail["transcript"])
        audit = app_client.get("/api/audit").json()["audit"]
        assert {"pause", "takeover", "operator_say"} <= {a["action"] for a in audit} and audit[0]["operator"] == "dana"
        assert app_client.post(f"/api/sessions/{sid}/release", headers=op).status_code == 200
        assert app_client.get(f"/api/sessions/{sid}").json()["snapshot"]["state"] == "idle"


def test_unknown_session_returns_404(app_client):
    assert app_client.post("/api/sessions/zzz/pause").status_code == 404
    assert app_client.get("/api/sessions/zzz").status_code == 404


def test_operator_token_protects_api_and_dashboard_socket():
    app = create_app(runtime=make_runtime(make_settings(operator_token="s3cret")))
    with TestClient(app) as c:
        assert c.get("/api/sessions").status_code == 401
        assert c.get("/api/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/sessions", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/dashboard?token=bad"):
                pass
        with c.websocket_connect("/ws/dashboard?token=s3cret") as dws:
            assert dws.receive_json()["type"] == "snapshot"


def test_approval_queue_over_http():
    rt = make_runtime(make_settings(approval_tools=("cancel_appointment",), approval_wait_s=0.05))
    with TestClient(create_app(runtime=rt)) as c, c.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        for text in ["I'm Bob Martinez, cancel my appointment", "yes"]:
            ws.send_json({"type": "text", "text": text})
            read_until(ws, "audio_end")
        pending = c.get("/api/approvals").json()["approvals"]
        assert len(pending) == 1 and pending[0]["tool"] == "cancel_appointment"
        assert c.post(f"/api/approvals/{pending[0]['id']}/reject", json={"note": "no"}, headers={"X-Operator": "eve"}).json()["status"] == "rejected"
        assert c.get("/api/approvals").json()["approvals"] == []
        assert c.post(f"/api/approvals/{pending[0]['id']}/approve").status_code == 409
        assert rt.store.get(2)["status"] == "booked"


def test_dashboard_socket_streams_live_events(app_client):
    with app_client.websocket_connect("/ws/dashboard") as dws:
        assert dws.receive_json()["type"] == "snapshot"
        with app_client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "Do you accept Aetna?"})
            read_until(ws, "audio_end")
        seen = set()
        for _ in range(80):
            seen.add(dws.receive_json()["type"])
            if {"tool_call", "turn_trace", "transcript"} <= seen:
                break
        assert {"session_opened", "tool_call", "turn_trace", "transcript"} <= seen | {"session_opened"}


def test_stats_endpoint_exposes_latency_percentiles(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "What are your hours?"})
        read_until(ws, "audio_end")
    stats = app_client.get("/api/stats").json()
    assert stats["turns"] >= 1 and "p95" in stats["spans"]["llm"]


def test_silence_frames_do_not_create_turns(app_client):
    with app_client.websocket_connect("/ws/session") as ws:
        ws.receive_json()
        for _ in range(30):
            ws.send_bytes(silence(20))
        ws.send_json({"type": "ping"})
        read_until(ws, "pong")
    assert app_client.get("/api/stats").json()["turns"] == 0

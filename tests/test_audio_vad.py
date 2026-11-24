from voice_agent.audio import (
    decode_speech,
    duration_ms,
    frame_bytes,
    rms,
    silence,
    split_frames,
    synth_speech,
    tone,
)
from voice_agent.vad import EnergyVAD


def test_synthetic_speech_roundtrip_unicode():
    text = "Hi, I'm José O'Neil — book me at 3 pm"
    assert decode_speech(synth_speech(text)) == text


def test_decode_returns_none_for_plain_audio():
    assert decode_speech(tone(500)) is None
    assert decode_speech(silence(200)) is None


def test_split_frames_pads_last_frame():
    frames = split_frames(b"\x01\x00" * 500)
    assert all(len(f) == frame_bytes() for f in frames)


def test_rms_distinguishes_silence_from_speech():
    assert rms(silence(100)) == 0
    assert rms(tone(100)) > 1000


def _feed(vad, pcm):
    events = [vad.feed(f) for f in split_frames(pcm)]
    return [e for e in events if e]


def test_vad_detects_one_utterance_with_endpointing():
    vad = EnergyVAD()
    events = _feed(vad, synth_speech("hello there") + silence(100))
    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    assert decode_speech(events[1].pcm) == "hello there"
    assert events[1].vad_ms >= 0


def test_vad_ignores_silence_and_single_click():
    vad = EnergyVAD()
    click = b"\x00\x00" * 160 + (b"\xff\x3f" * 320) + b"\x00\x00" * 160
    assert _feed(vad, silence(1000) + click + silence(1000)) == []


def test_vad_caps_utterance_length():
    vad = EnergyVAD(max_utterance_s=1.0)
    events = _feed(vad, tone(3000))
    assert any(e.kind == "speech_end" and duration_ms(e.pcm) <= 1100 for e in events)

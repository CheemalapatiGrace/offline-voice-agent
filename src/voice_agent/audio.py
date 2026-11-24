"""PCM16 helpers plus a *synthetic speech* codec for offline testing.

``synth_speech(text)`` produces loud PCM16 audio whose payload region carries the
UTF-8 text as sample values. The energy VAD therefore sees real "speech", and the
mock STT can decode it deterministically, which lets the complete audio path be
exercised without a microphone or a model download. Real microphone audio never
contains the marker, so the mock STT returns "" for it.
"""

from __future__ import annotations

import array
import math
import sys

SAMPLE_RATE = 16000
FRAME_MS = 20
_MARK_A = array.array("h", [30000, -30000, 30000, -30000])
_MARK_B = array.array("h", [-30000, 30000, -30000, 30000])
_BASE = 12000


def _le(a: array.array) -> bytes:
    if sys.byteorder == "big":  # pragma: no cover
        a = array.array("h", a)
        a.byteswap()
    return a.tobytes()


def to_samples(pcm: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":  # pragma: no cover
        a.byteswap()
    return a


def rms(pcm: bytes) -> float:
    s = to_samples(pcm)
    if not s:
        return 0.0
    return math.sqrt(sum(x * x for x in s) / len(s))


def frame_bytes(sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS) -> int:
    return sample_rate * frame_ms // 1000 * 2


def duration_ms(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate * 1000


def silence(ms: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * (sample_rate * ms // 1000)


def tone(ms: int, sample_rate: int = SAMPLE_RATE, amp: int = 6000, freq: float = 220.0) -> bytes:
    n = sample_rate * ms // 1000
    return _le(array.array("h", (int(amp * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n))))


def synth_speech(
    text: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    lead_ms: int = 200,
    tail_ms: int = 600,
    words_per_s: float = 3.5,
) -> bytes:
    """Encode ``text`` into speech-like PCM16, surrounded by silence."""
    payload = array.array("h", _MARK_A)
    payload.extend(_BASE + b for b in text.encode("utf-8"))
    payload.extend(_MARK_B)
    speech_ms = max(300, int(1000 * len(text.split()) / words_per_s))
    body = _le(payload) + tone(speech_ms, sample_rate)
    return silence(lead_ms, sample_rate) + body + silence(tail_ms, sample_rate)


def decode_speech(pcm: bytes) -> str | None:
    """Inverse of :func:`synth_speech`; ``None`` if no payload marker is present."""
    a = _le(_MARK_A)
    start = pcm.find(a)
    while start != -1 and start % 2:
        start = pcm.find(a, start + 1)
    if start == -1:
        return None
    start += len(a)
    end = pcm.find(_le(_MARK_B), start)
    while end != -1 and end % 2:
        end = pcm.find(_le(_MARK_B), end + 1)
    if end == -1:
        return None
    samples = to_samples(pcm[start:end])
    try:
        return bytes(max(0, min(255, s - _BASE)) for s in samples).decode("utf-8", "replace")
    except ValueError:  # pragma: no cover
        return None


def split_frames(pcm: bytes, sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS) -> list[bytes]:
    n = frame_bytes(sample_rate, frame_ms)
    frames = [pcm[i : i + n] for i in range(0, len(pcm), n)]
    if frames and len(frames[-1]) < n:
        frames[-1] = frames[-1] + b"\x00" * (n - len(frames[-1]))
    return frames

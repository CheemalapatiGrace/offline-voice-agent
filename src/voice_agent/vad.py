"""Energy-based voice activity detector with hangover endpointing."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .audio import duration_ms, rms


@dataclass
class VADEvent:
    kind: str  # speech_start | speech_end
    pcm: bytes = b""
    duration_ms: float = 0.0
    vad_ms: float = 0.0  # CPU time the VAD spent on this utterance


class EnergyVAD:
    """Frame-by-frame VAD. ``feed`` returns at most one event per frame."""

    def __init__(
        self,
        *,
        threshold: float = 500.0,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        hangover_ms: int = 400,
        min_speech_ms: int = 60,
        preroll_ms: int = 100,
        max_utterance_s: float = 30.0,
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hangover_frames = max(1, hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = int(max_utterance_s * 1000 // frame_ms)
        self._preroll: deque[bytes] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._run = 0
        self._silence = 0
        self._buf: list[bytes] = []
        self._vad_s = 0.0
        self._preroll.clear()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, frame: bytes) -> VADEvent | None:
        t0 = time.perf_counter()
        try:
            return self._feed(frame)
        finally:
            self._vad_s += time.perf_counter() - t0

    def _feed(self, frame: bytes) -> VADEvent | None:
        loud = rms(frame) >= self.threshold
        if not self._in_speech:
            self._preroll.append(frame)
            self._run = self._run + 1 if loud else 0
            if self._run >= self.min_speech_frames:
                self._in_speech = True
                self._silence = 0
                self._buf = list(self._preroll)
                self._preroll.clear()
                return VADEvent("speech_start")
            return None
        self._buf.append(frame)
        self._silence = 0 if loud else self._silence + 1
        if self._silence >= self.hangover_frames or len(self._buf) >= self.max_frames:
            pcm = b"".join(self._buf)
            ev = VADEvent("speech_end", pcm, duration_ms(pcm, self.sample_rate), self._vad_s * 1000)
            self.reset()
            return ev
        return None

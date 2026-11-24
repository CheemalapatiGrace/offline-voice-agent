"""faster-whisper STT adapter (optional extra: ``pip install -e '.[stt]'``).

The model is loaded lazily on first use; faster-whisper downloads weights on first load
unless they are already cached, so nothing here runs during tests.
"""

from __future__ import annotations

import asyncio

from ..audio import to_samples


class FasterWhisperSTT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            import faster_whisper  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faster-whisper is not installed. Run: pip install -e '.[stt]'") from exc
        self.name = f"faster-whisper:{model_size}"
        self._args = (model_size, device, compute_type)
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self._args[0], device=self._args[1], compute_type=self._args[2])
        return self._model

    def _run(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(to_samples(pcm).tobytes(), dtype=np.int16).astype("float32") / 32768.0
        segments, _ = self._load().transcribe(audio, language="en", vad_filter=False, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        async with self._lock:  # one decode at a time; the model is not re-entrant
            return await asyncio.to_thread(self._run, pcm)

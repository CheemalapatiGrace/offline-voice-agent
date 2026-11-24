"""Local TTS adapters: macOS ``say`` and Piper. Both stream 16-bit mono PCM chunks."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import wave
from collections.abc import AsyncIterator


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s.,!?'\-:;$%]", " ", text)[:500].strip()


class SayTTS:
    """macOS ``say`` -> 16 kHz WAV -> PCM chunks. No model download needed."""

    name = "macos-say"
    sample_rate = 16000

    def __init__(self, voice: str = "") -> None:
        if not shutil.which("say"):
            raise RuntimeError("`say` not found (macOS only). Use VA_TTS_BACKEND=piper or mock.")
        self.voice = voice

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = ["say", "-o", path, "--data-format=LEI16@16000"]
            if self.voice:
                cmd += ["-v", self.voice]
            proc = await asyncio.create_subprocess_exec(*cmd, "--", _clean(text) or ".")
            if await proc.wait() != 0:
                raise RuntimeError("say failed")
            with wave.open(path, "rb") as w:
                frames = w.readframes(w.getnframes())
            for i in range(0, len(frames), 3200):
                yield frames[i : i + 3200]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class PiperTTS:
    """Piper (https://github.com/rhasspy/piper): `piper --model X.onnx --output_raw`."""

    name = "piper"
    sample_rate = 22050

    def __init__(self, model_path: str, binary: str = "piper") -> None:
        if not shutil.which(binary):
            raise RuntimeError("piper binary not found on PATH")
        self.model_path, self.binary = model_path, binary

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        proc = await asyncio.create_subprocess_exec(
            self.binary, "--model", self.model_path, "--output_raw",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(_clean(text).encode() + b"\n")
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            while chunk := await proc.stdout.read(4096):
                yield chunk
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

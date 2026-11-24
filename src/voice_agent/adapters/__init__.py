"""Adapter factory: choose mock or real local backends from Settings."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from ..config import Settings
from ..vad import EnergyVAD
from .mock import MockSTT, RuleBasedLLM, SilentTTS
from .resilient import ResilientLLM


@dataclass
class Adapters:
    stt: object
    llm: object
    tts: object
    vad_factory: Callable[[], EnergyVAD]


def make_vad_factory(settings: Settings) -> Callable[[], EnergyVAD]:
    return lambda: EnergyVAD(
        threshold=settings.vad_threshold,
        sample_rate=settings.sample_rate,
        frame_ms=settings.frame_ms,
        hangover_ms=settings.vad_hangover_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        max_utterance_s=settings.max_utterance_s,
    )


def build_adapters(settings: Settings, bus=None, today: Callable[[], date] | None = None) -> Adapters:
    today = today or settings.clinic_today
    rules = RuleBasedLLM(today)

    if settings.stt_backend == "faster_whisper":
        from .faster_whisper_stt import FasterWhisperSTT

        stt = FasterWhisperSTT(settings.whisper_model)
    elif settings.stt_backend == "mock":
        stt = MockSTT()
    else:
        raise ValueError(f"unknown STT backend {settings.stt_backend!r}")

    if settings.llm_backend == "ollama":
        from .ollama_llm import OllamaLLM

        chain = [OllamaLLM(settings.ollama_host, settings.ollama_model, think=settings.ollama_think)]
        if settings.llm_fallback_rules:
            chain.append(rules)
    elif settings.llm_backend == "mock":
        chain = [rules]
    else:
        raise ValueError(f"unknown LLM backend {settings.llm_backend!r}")
    llm = ResilientLLM(chain, settings, bus)

    if settings.tts_backend == "system":
        from .tts_system import SayTTS

        tts = SayTTS(os.environ.get("VA_TTS_VOICE", ""))
    elif settings.tts_backend == "piper":
        from .tts_system import PiperTTS

        tts = PiperTTS(os.environ.get("VA_PIPER_MODEL", "en_US-lessac-medium.onnx"))
    elif settings.tts_backend == "mock":
        tts = SilentTTS(realtime=settings.tts_realtime)
    else:
        raise ValueError(f"unknown TTS backend {settings.tts_backend!r}")
    return Adapters(stt, llm, tts, make_vad_factory(settings))

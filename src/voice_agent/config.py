"""Runtime configuration. Every knob is an environment variable prefixed ``VA_``."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from datetime import date


def _parse(value: str, default):
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    return value


@dataclass(frozen=True)
class Settings:
    # --- backends -------------------------------------------------------
    stt_backend: str = "mock"  # mock | faster_whisper
    llm_backend: str = "mock"  # mock | ollama
    tts_backend: str = "mock"  # mock | system | piper
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_think: str = ""  # "", "false" or a level; only sent when set (thinking-capable models)
    whisper_model: str = "base.en"
    llm_fallback_rules: bool = True  # degrade to the rule-based brain if the primary LLM fails
    tts_realtime: bool = False  # mock TTS sleeps for the audio duration (demo only)

    # --- storage --------------------------------------------------------
    db_path: str = ":memory:"

    # --- audio / VAD ----------------------------------------------------
    sample_rate: int = 16000
    frame_ms: int = 20
    vad_threshold: float = 500.0
    vad_hangover_ms: int = 400
    vad_min_speech_ms: int = 60
    max_utterance_s: float = 30.0

    # --- bounded queues (backpressure) -----------------------------------
    audio_in_queue: int = 512
    utterance_queue: int = 4
    audio_out_queue: int = 64

    # --- per-stage timeouts / retries / breaker -------------------------------
    stt_timeout_s: float = 8.0
    llm_timeout_s: float = 15.0
    tool_timeout_s: float = 5.0
    tts_timeout_s: float = 8.0
    stt_retries: int = 1
    llm_retries: int = 1
    tts_retries: int = 1
    breaker_failures: int = 3
    breaker_cooldown_s: float = 15.0

    # --- sessions / resource limits ---------------------------------------
    max_sessions: int = 64
    reconnect_grace_s: float = 60.0
    idle_timeout_s: float = 900.0
    max_turns_per_session: int = 500
    max_ws_message_bytes: int = 65536

    # --- security / tool policy ---------------------------------------------
    rate_limit_calls_per_min: int = 30
    rate_limit_mutations_per_min: int = 6
    max_violations: int = 3
    approval_tools: tuple = ()  # e.g. ("cancel_appointment", "escalate_to_human")
    approval_wait_s: float = 20.0
    review_timeout_s: float = 15.0
    review_timeout_action: str = "release"  # release | drop
    operator_token: str = ""

    # --- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    today: str = ""  # pin the clinic "today" (YYYY-MM-DD) for deterministic runs
    clinic_name: str = "Maple Street Family Clinic"

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"VA_{f.name.upper()}")
            if raw is not None and raw != "":
                values[f.name] = _parse(raw, f.default)
        values.update(overrides)
        return cls(**values)

    def with_(self, **kw) -> Settings:
        return replace(self, **kw)

    def clinic_today(self) -> date:
        return date.fromisoformat(self.today) if self.today else date.today()

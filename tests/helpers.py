"""Shared test helpers."""

from __future__ import annotations

import asyncio

from voice_agent.adapters import Adapters, make_vad_factory
from voice_agent.adapters.mock import MockSTT, RuleBasedLLM, SilentTTS
from voice_agent.adapters.resilient import ResilientLLM
from voice_agent.config import Settings
from voice_agent.runtime import Runtime, build_runtime

TODAY = "2026-03-02"  # Monday


def make_settings(**kw) -> Settings:
    base = dict(today=TODAY, db_path=":memory:", llm_timeout_s=2.0, stt_timeout_s=2.0, tts_timeout_s=2.0, tool_timeout_s=2.0)
    base.update(kw)
    return Settings(**base)


def make_runtime(settings: Settings | None = None, *, stt=None, llm=None, tts=None, **kw) -> Runtime:
    s = settings or make_settings()
    brain = llm if llm is not None else RuleBasedLLM(s.clinic_today)
    chain = brain if isinstance(brain, list) else [brain]
    adapters = Adapters(stt or MockSTT(), ResilientLLM(chain, s), tts or SilentTTS(), make_vad_factory(s))
    return build_runtime(s, adapters, **kw)


async def wait_for(pred, timeout: float = 3.0, interval: float = 0.005):
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(interval)

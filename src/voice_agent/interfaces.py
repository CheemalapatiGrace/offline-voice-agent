"""Adapter interfaces. Mock and real backends implement the same Protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .datatypes import LLMChunk


@runtime_checkable
class STT(Protocol):
    name: str

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...


@runtime_checkable
class LLM(Protocol):
    name: str

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[LLMChunk]: ...


@runtime_checkable
class TTS(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

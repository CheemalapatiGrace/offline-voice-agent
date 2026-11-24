"""Ollama chat adapter (streaming + native tool calling). Requires a running ``ollama serve``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..datatypes import LLMChunk, ToolCall


def _to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k != "meta"}
        if m.get("role") == "tool" and "name" in m:
            m = {**m, "tool_name": m["name"]}
        out.append(m)
    return out


class OllamaLLM:
    def __init__(
        self, host: str = "http://127.0.0.1:11434", model: str = "llama3.2", read_timeout_s: float = 120.0, think: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.read_timeout_s = read_timeout_s
        self.think = think
        self._transport = transport

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[LLMChunk]:
        payload = {
            "model": self.model,
            "messages": _to_ollama(messages),
            "tools": tools,
            "stream": True,
            "options": {"temperature": 0.2},
        }
        if self.think:
            payload["think"] = False if self.think.lower() == "false" else self.think
        timeout = httpx.Timeout(connect=3.0, read=self.read_timeout_s, write=10.0, pool=None)
        n = 0
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            async with client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:200]
                    raise RuntimeError(f"ollama http {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message") or {}
                    if msg.get("content"):
                        yield LLMChunk(text=msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        n += 1
                        yield LLMChunk(tool_call=ToolCall(fn.get("name", ""), args, f"ollama_{n}"))
                    if obj.get("done"):
                        break


async def list_models(host: str, timeout: float = 2.0) -> list[str]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{host.rstrip('/')}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

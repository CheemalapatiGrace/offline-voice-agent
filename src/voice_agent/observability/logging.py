"""Structured JSON logging with PII redaction applied to every record."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from ..guardrails.pii import redact, redact_obj

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class RedactingJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        payload.update(redact_obj(extras))
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", stream=None) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(RedactingJsonFormatter())
    root = logging.getLogger("voice_agent")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str = "voice_agent") -> logging.Logger:
    return logging.getLogger(name if name.startswith("voice_agent") else f"voice_agent.{name}")


"""Speech-friendly date/time formatting."""

from __future__ import annotations

from datetime import datetime


def fmt_date(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return f"{d.strftime('%A, %B')} {d.day}"


def fmt_time(hhmm: str) -> str:
    h, m = int(hhmm[:2]), int(hhmm[3:])
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"

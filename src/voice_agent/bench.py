"""Concurrent-session benchmark: per-stage latency p50/p95 and time-to-first-audio.

    python -m voice_agent.bench --sessions 50 --turns 6
    python -m voice_agent.bench --sessions 100 --turns 6 --stt-ms 60 --llm-ms 150 --tts-ms 40 --json out.json

All sessions run concurrently inside one asyncio event loop against the full pipeline
(VAD -> STT -> guardrails -> LLM/tools -> TTS) with the offline mock adapters. The mock adapters
can be given artificial per-stage latency to approximate real local models; with all of them at
zero the numbers are the orchestration overhead of the system itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import date, timedelta

from .adapters import Adapters, make_vad_factory
from .adapters.mock import MockSTT, RuleBasedLLM, SilentTTS
from .adapters.resilient import ResilientLLM
from .config import Settings
from .harness import SimClient
from .observability.logging import configure_logging
from .runtime import build_runtime

FIRST = ["Alice", "Bruno", "Chloe", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Isla", "Jonas", "Keiko", "Liam", "Maya", "Noor", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara"]
LAST = ["Adams", "Baker", "Chen", "Diaz", "Evans", "Fox", "Gupta", "Hayes", "Ito", "Jones", "Khan", "Lopez", "Moore", "Nash", "Ortiz", "Patel", "Quist", "Reyes", "Shaw", "Tran"]


def script_for(i: int, today: date) -> list[str]:
    name = f"{FIRST[i % 20]} {LAST[(i // 20 + i) % 20]}"
    day = today + timedelta(days=1 + (i // 16) % 14)
    while day.weekday() > 4:
        day += timedelta(days=1)
    slot = i % 16
    hh, mm = divmod(9 * 60 + 30 * slot, 60)
    return [
        f"Hello, my name is {name} and I'd like to book an appointment",
        f"{day.isoformat()} at {hh:02d}:{mm:02d}",
        "what are your hours?",
        "do you accept Aetna insurance?",
        "please cancel my appointment",
        "yes",
    ]


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run_bench(args: argparse.Namespace) -> dict:
    settings = Settings(today="2026-03-02", max_sessions=max(args.sessions, 1) + 4, db_path=":memory:")
    rules = RuleBasedLLM(settings.clinic_today, first_token_delay_s=args.llm_ms / 1000, token_delay_s=args.token_ms / 1000)
    adapters = Adapters(
        stt=MockSTT(args.stt_ms / 1000),
        llm=ResilientLLM([rules], settings),
        tts=SilentTTS(first_chunk_delay_s=args.tts_ms / 1000),
        vad_factory=make_vad_factory(settings),
    )
    rt = build_runtime(settings, adapters)
    today = settings.clinic_today()
    results: list[dict] = []
    errors = 0

    async def one(i: int) -> None:
        nonlocal errors
        client = await SimClient.open(rt)
        try:
            for line in script_for(i, today)[: args.turns]:
                try:
                    r = await client.say(line, via_audio=not args.text, timeout=120)
                    results.append(r.trace)
                except Exception:  # noqa: BLE001
                    errors += 1
        finally:
            await client.close()

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(args.sessions)))
    wall = time.perf_counter() - t0
    await rt.stop()

    stages: dict[str, list[float]] = {}
    for tr in results:
        for sp in tr.get("spans", []):
            stages.setdefault(sp["name"], []).append(sp["dur_ms"])
    ttfa = [tr["ttfa_ms"] for tr in results if tr.get("ttfa_ms") is not None]
    total = [tr["total_ms"] for tr in results]
    hang = settings.vad_hangover_ms
    out = {
        "config": {
            "sessions": args.sessions, "turns_per_session": args.turns, "path": "text" if args.text else "audio (VAD+STT)",
            "injected_ms": {"stt": args.stt_ms, "llm_first_token": args.llm_ms, "llm_per_token": args.token_ms, "tts_first_chunk": args.tts_ms},
            "vad_hangover_ms": hang,
        },
        "turns": len(results), "errors": errors, "wall_s": round(wall, 2),
        "throughput_turns_per_s": round(len(results) / wall, 1) if wall else 0,
        "stages": {k: {"n": len(v), "p50": round(pct(v, .5), 2), "p95": round(pct(v, .95), 2), "mean": round(statistics.fmean(v), 2)} for k, v in sorted(stages.items())},
        "ttfa_ms": {"n": len(ttfa), "p50": round(pct(ttfa, .5), 2), "p95": round(pct(ttfa, .95), 2)},
        "ttfa_incl_endpointing_ms": {"p50": round(pct(ttfa, .5) + hang, 2), "p95": round(pct(ttfa, .95) + hang, 2)},
        "turn_total_ms": {"p50": round(pct(total, .5), 2), "p95": round(pct(total, .95), 2)},
    }
    return out


def render(out: dict) -> str:
    c = out["config"]
    lines = [
        f"Benchmark: {c['sessions']} concurrent sessions x {c['turns_per_session']} turns via {c['path']}",
        f"Injected adapter latency (ms): {c['injected_ms']}  | VAD endpointing hangover: {c['vad_hangover_ms']} ms",
        f"Completed {out['turns']} turns in {out['wall_s']} s ({out['throughput_turns_per_s']} turns/s), errors={out['errors']}",
        "",
        f"{'stage':<12}{'n':>7}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}",
    ]
    for name, s in out["stages"].items():
        lines.append(f"{name:<12}{s['n']:>7}{s['p50']:>10}{s['p95']:>10}{s['mean']:>10}")
    t = out["ttfa_ms"]
    lines += [
        f"{'TTFA':<12}{t['n']:>7}{t['p50']:>10}{t['p95']:>10}   (speech-end -> first audio chunk queued)",
        f"{'TTFA+endpt':<12}{'':>7}{out['ttfa_incl_endpointing_ms']['p50']:>10}{out['ttfa_incl_endpointing_ms']['p95']:>10}   (adds the VAD hangover the caller perceives)",
        f"{'turn total':<12}{'':>7}{out['turn_total_ms']['p50']:>10}{out['turn_total_ms']['p95']:>10}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice_agent.bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=int, default=50)
    p.add_argument("--turns", type=int, default=6, help="turns per session (max 6)")
    p.add_argument("--text", action="store_true", help="skip VAD/STT (text turns)")
    p.add_argument("--stt-ms", type=float, default=0.0, help="injected STT latency")
    p.add_argument("--llm-ms", type=float, default=0.0, help="injected LLM first-token latency")
    p.add_argument("--token-ms", type=float, default=0.0, help="injected LLM per-token latency")
    p.add_argument("--tts-ms", type=float, default=0.0, help="injected TTS first-chunk latency")
    p.add_argument("--json", help="write results to this file")
    args = p.parse_args(argv)
    configure_logging("ERROR")
    out = asyncio.run(run_bench(args))
    print(render(out))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 1 if out["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

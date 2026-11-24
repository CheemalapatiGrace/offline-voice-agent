"""Text-mode (and synthetic-audio) simulator: talk to the full agent without a microphone.

    python -m voice_agent.sim                      # interactive, text in / text out
    python -m voice_agent.sim --audio --trace      # through VAD + STT with per-span latency
    python -m voice_agent.sim --say "hi, my name is Alice Johnson" --say "what are your hours"
    python -m voice_agent.sim --llm ollama --model llama3.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .adapters.mock import NoisySTT
from .config import Settings
from .harness import SimClient, TurnResult
from .observability.logging import configure_logging
from .runtime import build_runtime


def format_turn(r: TurnResult, trace: bool) -> str:
    out = []
    if r.user_text:
        out.append(f"  heard : {r.user_text}")
    for t in r.tool_calls:
        args = json.dumps(t["args"])
        out.append(f"  tool  : {t['name']}({args}) -> {t['status']}{' (replayed)' if t['replayed'] else ''}")
    if r.guard:
        out.append(f"  guard : {', '.join(r.guard)}")
    out.append(f"agent > {r.reply or '(no reply)'}")
    meta = f"state={r.state}" + (" ESCALATED" if r.escalated else "")
    if r.trace.get("ttfa_ms") is not None:
        meta += f"  ttfa={r.trace['ttfa_ms']}ms"
    out.append(f"  [{meta}]")
    if trace and r.trace:
        for s in r.trace["spans"]:
            out.append(f"    span {s['name']:<9} {s['dur_ms']:>8.2f} ms  {s['status']}")
    return "\n".join(out)


async def amain(args: argparse.Namespace) -> int:
    overrides: dict = {"llm_backend": args.llm, "db_path": ":memory:"}
    if args.llm == "ollama":
        overrides["llm_timeout_s"] = max(Settings.from_env().llm_timeout_s, args.timeout)
    if args.model:
        overrides["ollama_model"] = args.model
    if args.today:
        overrides["today"] = args.today
    if args.approvals:
        overrides["approval_tools"] = ("cancel_appointment", "escalate_to_human")
    rt = build_runtime(Settings.from_env(**overrides))
    if args.noise:
        rt.adapters.stt = NoisySTT(rt.adapters.stt, args.noise, seed=args.seed)
    client = await SimClient.open(rt)
    print(f"offline-voice-agent sim | llm={args.llm} audio={'on' if args.audio else 'off'} today={rt.settings.clinic_today()} | /quit to exit")
    lines: list[str] = list(args.say or [])
    if args.script:
        with open(args.script, encoding="utf-8") as fh:
            lines += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    try:
        if lines:
            for line in lines:
                print(f"\nyou   > {line}")
                print(format_turn(await client.say(line, via_audio=args.audio, timeout=args.timeout), args.trace))
        else:
            while True:
                try:
                    line = await asyncio.to_thread(input, "\nyou   > ")
                except EOFError:
                    break
                if line.strip() in {"/quit", "/exit"}:
                    break
                if line.strip():
                    print(format_turn(await client.say(line, via_audio=args.audio, timeout=args.timeout), args.trace))
    finally:
        await client.close()
        await rt.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice_agent.sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", action="store_true", help="send synthetic speech through VAD + STT instead of plain text")
    p.add_argument("--trace", action="store_true", help="print per-stage spans for every turn")
    p.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    p.add_argument("--model", help="Ollama model name (default from VA_OLLAMA_MODEL)")
    p.add_argument("--say", action="append", help="a scripted user line (repeatable)")
    p.add_argument("--script", help="file with one user line per row")
    p.add_argument("--noise", type=float, default=0.0, help="inject this word error rate into the STT (needs --audio)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--today", help="pin the clinic date, YYYY-MM-DD")
    p.add_argument("--approvals", action="store_true", help="require operator approval for cancel/escalate (will wait)")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args(argv)
    configure_logging("WARNING")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())

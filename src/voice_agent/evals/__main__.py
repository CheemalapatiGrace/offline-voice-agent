"""CLI: run all scenarios, print a report, exit non-zero if any gate is below its threshold."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..observability.logging import configure_logging
from .metrics import DEFAULT_THRESHOLDS, compute_metrics, evaluate_thresholds, render_report
from .runner import SCENARIO_DIR, is_functional, load_scenarios, measure_wer, run_scenario


async def run_all(args: argparse.Namespace) -> tuple[dict, list]:
    scenarios = load_scenarios(Path(args.scenarios))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in set(args.only)]
    llm_factory = None
    if args.llm == "ollama":
        from ..adapters.ollama_llm import OllamaLLM

        llm_factory = lambda settings: OllamaLLM(args.ollama_host, args.model)  # noqa: E731

    outcomes = []
    for sc in scenarios:
        factory = None if sc.get("llm") == "rogue" else llm_factory
        outcomes.append(await run_scenario(sc, via_audio=not args.text, llm_factory=factory))
    metrics = compute_metrics(outcomes)
    report: dict = {
        "llm": "rule-based mock (offline)" if args.llm == "mock" else f"ollama:{args.model}",
        "metrics": metrics,
        "outcomes": [
            {"id": o.id, "category": o.category, "passed": o.passed, "failures": o.failures, "transcript": o.transcript}
            for o in outcomes
        ],
    }

    # --- STT noise model calibration + robustness of functional scenarios under noise
    wer = await measure_wer(scenarios)
    report["wer"] = wer
    cal_ok = abs(wer["0.1"]["measured"] - 0.10) <= 0.04 and wer["0.0"]["measured"] == 0.0
    metrics["wer_calibration_ok"] = 1.0 if cal_ok else 0.0
    if not args.no_noise:
        functional = [s for s in scenarios if is_functional(s) and s.get("llm") != "rogue" and not s.get("approvals")]
        noisy = [await run_scenario(s, via_audio=True, noise=args.noise, llm_factory=llm_factory) for s in functional]
        passed = sum(o.passed for o in noisy)
        report["noisy"] = {"wer": args.noise, "passed": passed, "total": len(noisy), "rate": passed / len(noisy) if noisy else 0.0}
        metrics["noisy_task_success_rate"] = report["noisy"]["rate"]
    return report, outcomes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice_agent.evals", description=__doc__)
    p.add_argument("--scenarios", default=str(SCENARIO_DIR), help="directory of *.yaml scenario files")
    p.add_argument("--only", nargs="*", help="run only these scenario ids")
    p.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    p.add_argument("--model", default="llama3.2", help="Ollama model (with --llm ollama)")
    p.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    p.add_argument("--text", action="store_true", help="skip the VAD/STT audio path")
    p.add_argument("--noise", type=float, default=0.10, help="WER injected for the noisy robustness run")
    p.add_argument("--no-noise", action="store_true", help="skip the noisy robustness run")
    p.add_argument("--threshold", action="append", default=[], metavar="NAME=VALUE", help="override a gate, e.g. task_success_rate=0.9")
    p.add_argument("--no-gate", action="store_true", help="always exit 0 (report only; useful for real-LLM runs)")
    p.add_argument("--json", help="write the full report as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configure_logging("ERROR")

    thresholds = dict(DEFAULT_THRESHOLDS)
    for item in args.threshold:
        k, _, v = item.partition("=")
        if k not in thresholds:
            p.error(f"unknown metric {k!r}; choose from {sorted(thresholds)}")
        thresholds[k] = float(v)
    if args.no_noise:
        thresholds.pop("noisy_task_success_rate", None)

    report, _ = asyncio.run(run_all(args))
    rows = evaluate_thresholds(report["metrics"], thresholds)
    report["gates"] = [{"metric": n, "value": v, "min": m, "pass": ok} for n, v, m, ok in rows]
    print(render_report(report, rows, args.verbose))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    ok = all(r[3] for r in rows)
    return 0 if ok or args.no_gate else 1


if __name__ == "__main__":
    sys.exit(main())

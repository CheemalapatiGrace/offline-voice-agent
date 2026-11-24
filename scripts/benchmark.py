"""Convenience wrapper: ``python scripts/benchmark.py --sessions 50`` (same as ``python -m voice_agent.bench``)."""

from voice_agent.bench import main

if __name__ == "__main__":
    raise SystemExit(main())

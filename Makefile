PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install test lint eval run run-demo bench sim sim-audio clean docker

install:
	test -d .venv || python3.12 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests

eval:
	$(PY) -m voice_agent.evals

run:
	$(PY) -m voice_agent.server

run-demo:
	VA_APPROVAL_TOOLS=cancel_appointment,escalate_to_human VA_APPROVAL_WAIT_S=30 $(PY) -m voice_agent.server

bench:
	$(PY) -m voice_agent.bench --sessions 50 --turns 6

sim:
	$(PY) -m voice_agent.sim

sim-audio:
	$(PY) -m voice_agent.sim --audio --trace

docker:
	docker build -t offline-voice-agent .

clean:
	rm -rf .pytest_cache .ruff_cache data reports

# Targets of the Phase 1 pipeline. PY and PYTHON can be overridden, for example `make smoke PY=python` in CI.
PY ?= .venv/bin/python
PYTHON ?= python3

.PHONY: setup test lint run smoke reproduce-check clean

setup:  ## virtual environment and the pinned install
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --quiet -r requirements.txt
	@.venv/bin/python --version

test:  ## pytest (synthetic fixtures only)
	$(PY) -m pytest -q

lint:  ## ruff
	$(PY) -m ruff check .

run:  ## real data: validation, every Phase 1 experiment, report. Needs data/NQ_1h.csv
	$(PY) scripts/run_all.py

smoke:  ## the whole pipeline on SYNTHETIC bars, writes smoke_SYNTHETIC/
	$(PY) scripts/run_all.py --smoke

reproduce-check:  ## run twice, compare the sha256 of every result file
	$(PY) scripts/run_all.py > /dev/null
	$(PY) scripts/hash_results.py > .repro_run1.txt
	$(PY) scripts/run_all.py > /dev/null
	$(PY) scripts/hash_results.py > .repro_run2.txt
	@echo "result hashes, run 1:"; cat .repro_run1.txt
	@echo "result hashes, run 2:"; cat .repro_run2.txt
	@diff .repro_run1.txt .repro_run2.txt && echo "reproduce-check: IDENTICAL"

clean:  ## remove the smoke output and caches (results/ is kept, it is committed)
	rm -rf smoke_SYNTHETIC .pytest_cache .ruff_cache .repro_run1.txt .repro_run2.txt
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

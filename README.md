# signal-research-lab

**No evidence of an edge was found.** On NQ one-hour bars, 1.6 years out of sample (392 trading days), a 24-bar momentum rule and buy-and-hold both have Sharpe intervals that include zero, and the momentum permutation p-values are 0.766 (H=1) and 0.682 (H=2). Full write-up: [docs/FINDINGS.md](docs/FINDINGS.md).

## Scope

Phase 1 only: a walk-forward framework with purge and embargo, a cost model, two baselines (buy-and-hold and one momentum rule), a permutation test, automated leakage tests, and the write-up. Not done: machine-learning models, volatility regimes, deflated Sharpe, ADF, ACF, PACF, GARCH, the risk simulator, and the ES run. Costs and the expiry-bar fallback are assumptions, and the Yahoo data is unofficial with unverified roll handling.

## Data

`data/` is gitignored, so the data file is not in this repo. Place the Yahoo snapshot at `data/NQ_1h.csv` (columns `Datetime, Adj Close, Close, High, Low, Open, Volume`, hourly NQ=F bars downloaded on 2026-11-19, `scripts/download_data.py` documents the call). Do not re-download it to reproduce results: a new download gives different data. Every result is tied to the sha256 of the file recorded in `results/run_info.json` (this snapshot: `53c87c8963307a28a2b51627ad8e1d31559a4524577db2eefa9a924ae842519b`). `results/run_info.json` records the commit the results were generated at, so it is one commit behind by design.

## How to run

| Command | What it does |
| --- | --- |
| `make setup` | creates `.venv` and installs the pinned `requirements.txt` |
| `make test` | runs pytest (synthetic fixtures only) |
| `make lint` | runs ruff |
| `make run` | real data: validation, every Phase 1 experiment, tables, figures, `results/summary.md`, `results/run_info.json`, and the table inside `docs/FINDINGS.md` |
| `make smoke` | the same pipeline on SYNTHETIC bars, written to `smoke_SYNTHETIC/` with every file name starting `SYNTHETIC_`; not a result |
| `make reproduce-check` | runs `make run` twice and compares the sha256 of every file in `results/` |
| `make clean` | removes the smoke output and caches (`results/` is kept) |

## Continuous integration

CI runs ruff, pytest and `make smoke` with Python 3.13. **CI cannot run on the real data**: the data is private and not in the repo, so CI cannot run `make run`. It shows that the code runs and the tests pass. It does not show that the real results reproduce; that is checked locally with `make reproduce-check`.

## Layout

`src/signal_lab/` one module per concept (data, labels, features, splits, costs, backtest, baselines, metrics, stats_tests, leakage, report, pipeline), `config/default.yaml` every parameter and assumption, `tests/`, `scripts/`, `results/` small generated tables and figures, `docs/`. The specification is `SPEC.md`.

"""Single entry point. Each item adds its step here. Uses only the local data file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np

from signal_lab.backtest import simulate
from signal_lab.baselines import build_baselines
from signal_lab.config import load_config
from signal_lab.costs import cost_per_side_usd
from signal_lab.data import run_data_step, sha256_of_file
from signal_lab.labels import add_sample_section, build_labels, sample_report
from signal_lab.report import comparison_markdown, evaluate, to_frame
from signal_lab.splits import check_embargo, fold_table, oos_window
from signal_lab.trials import TrialLog

RESULTS = ROOT / "results"


def execution_lag_check(bars, cfg):
    """Diagnostic on the real out-of-sample bars, not a strategy result.

    A planted signal that knows the direction of bar t at its close must earn almost nothing
    when executed at the open of bar t+1. A same-bar peek would earn the sum of all moves.
    """
    start, stop = oos_window(len(bars), cfg["splits"])
    window = bars.iloc[start:stop]
    labels = build_labels(bars, 1, cfg["data"]["bar_minutes"])
    valid = labels["valid"].to_numpy()[start:stop]
    move = (window["close"] - window["open"]).to_numpy()
    multiplier = cfg["instrument"]["multiplier_usd"]
    sim = simulate(window, np.sign(move), valid, 1, multiplier)
    peek = float(np.abs(move[valid]).sum() * multiplier)
    lagged = float(sim["gross_pnl_usd"].sum())
    return {
        "diagnostic_only": "planted perfect-foresight signal on real out-of-sample bars",
        "window": [str(window.index[0]), str(window.index[-1])],
        "bars": len(window),
        "same_bar_peek_gross_usd": peek,
        "correctly_lagged_gross_usd": lagged,
        "lagged_as_share_of_peek": lagged / peek,
    }


def main():
    cfg = load_config()
    RESULTS.mkdir(exist_ok=True)
    bars = run_data_step(cfg, RESULTS / "data_report.json")

    add_sample_section(RESULTS / "data_report.json", sample_report(bars, cfg))

    check_embargo(cfg["splits"], cfg["horizon"])
    horizons = [cfg["horizon"]["main_bars"], cfg["horizon"]["secondary_bars"]]
    label_ends = {
        h: build_labels(bars, h, cfg["data"]["bar_minutes"])["label_end"].to_numpy()
        for h in horizons
    }
    folds = fold_table(bars.index, label_ends, cfg["splits"])
    folds.to_csv(RESULTS / "folds.csv", index=False)

    check = execution_lag_check(bars, cfg)
    (RESULTS / "execution_lag_check.json").write_text(json.dumps(check, indent=2) + "\n")
    costs = {
        "assumption_note": "all cost values are assumptions, see config/default.yaml",
        "per_side_usd_main": cost_per_side_usd(cfg["costs"], cfg["instrument"]),
        "per_side_usd_conservative": cost_per_side_usd(
            cfg["costs"], cfg["instrument"], cfg["costs"]["conservative_commission_per_side_usd"]
        ),
    }
    (RESULTS / "cost_levels.json").write_text(json.dumps(costs, indent=2) + "\n")

    labels_by_horizon = {h: build_labels(bars, h, cfg["data"]["bar_minutes"]) for h in horizons}
    window = oos_window(len(bars), cfg["splits"])
    sims = build_baselines(bars, cfg, window, labels_by_horizon)
    trials = TrialLog()
    index = bars.index[window[0] : window[1]]
    rows, differences = evaluate(sims, index, cfg, trials, sha256_of_file(cfg["data"]["path"]))
    to_frame(rows).to_csv(RESULTS / "comparison.csv", index=False)
    (RESULTS / "comparison.md").write_text(comparison_markdown(rows, differences, index, cfg))
    trials.write(RESULTS / "trials.csv")
    print((RESULTS / "comparison.md").read_text())
    print("trials logged:", len(trials))


if __name__ == "__main__":
    main()

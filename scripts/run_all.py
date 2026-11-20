"""Single entry point. Each item adds its step here. Uses only the local data file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np

from signal_lab.backtest import simulate
from signal_lab.config import load_config
from signal_lab.costs import cost_per_side_usd
from signal_lab.data import run_data_step
from signal_lab.labels import add_sample_section, build_labels, sample_report
from signal_lab.splits import block_bounds, check_embargo, fold_table

RESULTS = ROOT / "results"


def execution_lag_check(bars, cfg):
    """Diagnostic on the real out-of-sample bars, not a strategy result.

    A planted signal that knows the direction of bar t at its close must earn almost nothing
    when executed at the open of bar t+1. A same-bar peek would earn the sum of all moves.
    """
    blocks = block_bounds(len(bars), cfg["splits"])
    window = bars.iloc[blocks[0][0] : blocks[-1][1]]
    labels = build_labels(bars, 1, cfg["data"]["bar_minutes"])
    valid = labels["valid"].to_numpy()[blocks[0][0] : blocks[-1][1]]
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
    print(json.dumps(check, indent=2))
    print(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main()

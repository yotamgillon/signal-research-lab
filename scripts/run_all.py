"""Single entry point. Each item adds its step here. Uses only the local data file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np
import pandas as pd

from signal_lab.backtest import simulate
from signal_lab.baselines import build_baselines, momentum_signal
from signal_lab.config import load_config
from signal_lab.costs import cost_per_side_usd
from signal_lab.data import add_returns, run_data_step, sha256_of_file
from signal_lab.features import build_features
from signal_lab.labels import add_sample_section, build_labels, sample_report
from signal_lab.leakage import (
    canary_feature,
    label_boundary_violations,
    ordering_violations,
    truncation_violations,
)
from signal_lab.report import comparison_markdown, evaluate, to_frame
from signal_lab.splits import check_embargo, fold_table, oos_window, walk_forward_splits
from signal_lab.stats_tests import circular_shift_test, net_sharpe_statistic
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


def cut_points(bars, cfg):
    """Random cut points of the truncation test, plus the bars around every expiry bar."""
    rng = np.random.default_rng(cfg["seed"])
    random_cuts = rng.integers(200, len(bars) - 1, size=cfg["tests"]["truncation_cut_points"])
    expiry = np.flatnonzero(bars["expiry_bar"].to_numpy())
    around = np.concatenate([expiry - 1, expiry, expiry + 1])
    return sorted({int(c) for c in np.concatenate([random_cuts, around])})


def leakage_checks(bars, label_ends, cfg):
    """Leakage checks on the real bars. They say nothing about performance."""
    raw = bars[["open", "high", "low", "close", "volume"]]
    cuts = cut_points(bars, cfg)

    def clean(frame):
        return build_features(add_returns(frame, cfg["cleaning"]), cfg["features"])

    def with_canary(frame):
        prepared = add_returns(frame, cfg["cleaning"])
        features = build_features(prepared, cfg["features"])
        features["canary_future_return"] = canary_feature(prepared)
        return features

    clean_found = truncation_violations(clean, raw, cuts)
    canary_found = truncation_violations(with_canary, raw, cuts)
    boundary = {
        f"H={h}": len(label_boundary_violations(ends, list(walk_forward_splits(ends, cfg["splits"]))))
        for h, ends in label_ends.items()
    }
    return {
        "note": "leakage checks on the real bars, not a performance result",
        "cut_points": len(cuts),
        "clean_features_truncation_violations": len(clean_found),
        "canary_truncation_cuts_with_violation": len(canary_found),
        "canary_first_violation": canary_found[0] if canary_found else None,
        "label_boundary_violating_folds": boundary,
        "ordering_problems": ordering_violations(bars.index),
    }


def momentum_permutation_tests(bars, cfg, window, labels_by_horizon, trials, sha):
    """Circular-shift test of the momentum rule, H=1 and H=2, main commission. Logged as trials."""
    start, stop = window
    part = bars.iloc[start:stop]
    signal = momentum_signal(bars, cfg["baseline"]["momentum_lookback_bars"])[start:stop]
    per_side = cost_per_side_usd(cfg["costs"], cfg["instrument"])
    results, permuted_by_horizon = {}, {}
    for horizon, labels in labels_by_horizon.items():
        valid = labels["valid"].to_numpy()[start:stop]
        statistic = net_sharpe_statistic(part, valid, horizon, per_side, cfg)
        result, permuted = circular_shift_test(
            statistic, signal, cfg["permutation"]["n_permutations"],
            cfg["permutation"]["min_shift_bars"], cfg["seed"] + horizon)
        result["mean_signal_exposure"] = float(np.nanmean(signal))
        results[f"H={horizon}"], permuted_by_horizon[f"H{horizon}"] = result, permuted
        trials.add(
            strategy="momentum circular-shift permutation test", horizon_bars=horizon,
            momentum_lookback=cfg["baseline"]["momentum_lookback_bars"], commission_run="main",
            commission_per_side_usd=cfg["costs"]["commission_per_side_usd"], cost_multiplier=1,
            per_side_cost_usd=per_side, gross_sharpe="n/a", net_sharpe=result["real_net_sharpe"],
            oos_window=f"{part.index[0]} to {part.index[-1]}", data_sha256=sha,
            permutations=result["n_permutations"], p_value=result["p_value"])
    (RESULTS / "permutation_momentum.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame(permuted_by_horizon).to_csv(RESULTS / "permutation_momentum_sharpes.csv", index=False)
    return results


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
    print((RESULTS / "comparison.md").read_text())

    sha = sha256_of_file(cfg["data"]["path"])
    permutation = momentum_permutation_tests(bars, cfg, window, labels_by_horizon, trials, sha)
    print(json.dumps(permutation, indent=2))
    trials.write(RESULTS / "trials.csv")
    print("trials logged:", len(trials))

    checks = leakage_checks(bars, label_ends, cfg)
    (RESULTS / "leakage_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

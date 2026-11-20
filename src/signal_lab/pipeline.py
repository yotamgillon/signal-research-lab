"""The whole pipeline. scripts/run_all.py is a thin wrapper around run() and run_smoke().

run():       the real data file, writes results/ and docs/FINDINGS.md.
run_smoke(): the same code on SYNTHETIC bars, every output file is prefixed SYNTHETIC_.
This module must not import the synthetic generator at the top: the real run never loads it.
"""
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from signal_lab.backtest import simulate
from signal_lab.baselines import build_baselines, momentum_signal
from signal_lab.config import with_overrides
from signal_lab.costs import cost_per_side_usd
from signal_lab.data import add_returns, run_data_step, sha256_of_file
from signal_lab.features import build_features
from signal_lab.labels import add_sample_section, build_labels, sample_report, step_gap_kinds
from signal_lab.leakage import (
    canary_feature,
    label_boundary_violations,
    ordering_violations,
    truncation_violations,
)
from signal_lab.report import (
    build_summary,
    comparison_markdown,
    drawdown_figure,
    equity_figure,
    evaluate,
    headline_series,
    insert_results_table,
    main_table,
    numbers_missing_from_summary,
    passive_vs_tradable,
    permutation_figure,
    to_frame,
    word_count,
)
from signal_lab.splits import check_embargo, fold_table, oos_window, walk_forward_splits
from signal_lab.stats_tests import circular_shift_test, net_sharpe_statistic
from signal_lab.trials import TrialLog

ROOT = Path(__file__).resolve().parents[2]
SMOKE_PREFIX = "SYNTHETIC_"
SMOKE_BANNER = "SYNTHETIC DATA. "


class Outputs:
    """Where results go. A prefix is put in front of every file name (SYNTHETIC_ for the smoke run)."""

    def __init__(self, directory, prefix=""):
        self.directory, self.prefix = Path(directory), prefix
        (self.directory / "figures").mkdir(parents=True, exist_ok=True)

    def path(self, name):
        parent, _, leaf = name.rpartition("/")
        return self.directory / parent / f"{self.prefix}{leaf}"


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def source_label(banner):
    return "SYNTHETIC" if banner else "real"


def execution_lag_check(bars, cfg, banner):
    """Diagnostic on the out-of-sample bars, not a strategy result.

    A planted signal that knows the direction of bar t at its close must earn almost nothing
    when executed at the open of bar t+1. A same-bar peek would earn the sum of all moves.
    """
    start, stop = oos_window(len(bars), cfg["splits"])
    window = bars.iloc[start:stop]
    valid = build_labels(bars, 1, cfg["data"]["bar_minutes"])["valid"].to_numpy()[start:stop]
    move = (window["close"] - window["open"]).to_numpy()
    multiplier = cfg["instrument"]["multiplier_usd"]
    sim = simulate(window, np.sign(move), valid, 1, multiplier)
    peek = float(np.abs(move[valid]).sum() * multiplier)
    lagged = float(sim["gross_pnl_usd"].sum())
    return {
        "diagnostic_only": f"planted perfect-foresight signal on {source_label(banner)} "
                           "out-of-sample bars",
        "window": [str(window.index[0]), str(window.index[-1])],
        "bars": len(window),
        "same_bar_peek_gross_usd": peek,
        "correctly_lagged_gross_usd": lagged,
        "lagged_as_share_of_peek": lagged / peek,
    }


def cut_points(bars, cfg):
    """Random cut points of the truncation test, plus the bars around every expiry bar."""
    rng = np.random.default_rng(cfg["seed"])
    first = cfg["features"]["ma_window"] * 4  # leave room for the longest feature window
    random_cuts = rng.integers(first, len(bars) - 1, size=cfg["tests"]["truncation_cut_points"])
    expiry = np.flatnonzero(bars["expiry_bar"].to_numpy())
    around = np.concatenate([expiry - 1, expiry, expiry + 1])
    return sorted({int(c) for c in np.concatenate([random_cuts, around])})


def leakage_checks(bars, label_ends, cfg, banner):
    """Leakage checks on the bars. They say nothing about performance."""
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
        f"H={h}": len(label_boundary_violations(
            ends, list(walk_forward_splits(ends, cfg["splits"]))))
        for h, ends in label_ends.items()
    }
    return {
        "note": f"leakage checks on the {source_label(banner)} bars, not a performance result",
        "cut_points": len(cuts),
        "clean_features_truncation_violations": len(clean_found),
        "canary_truncation_cuts_with_violation": len(canary_found),
        "canary_first_violation": canary_found[0] if canary_found else None,
        "label_boundary_violating_folds": boundary,
        "ordering_problems": ordering_violations(bars.index),
    }


def momentum_permutation_tests(bars, cfg, window, labels_by_horizon, trials, sha, out):
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
    write_json(out.path("permutation_momentum.json"), results)
    pd.DataFrame(permuted_by_horizon).to_csv(
        out.path("permutation_momentum_sharpes.csv"), index=False)
    return results, permuted_by_horizon


def update_findings(findings_path, rows, per_side, summary):
    """Insert the generated results table into FINDINGS.md and check its numbers."""
    caption = (f"Main run, commission 0.00 USD per side, {per_side:.2f} USD per side in total "
               "(ASSUMPTIONS). Out-of-sample, net and gross, 95 percent block-bootstrap intervals.")
    text = insert_results_table(Path(findings_path).read_text(),
                                [caption, ""] + main_table(rows, "main"))
    Path(findings_path).write_text(text)
    missing = numbers_missing_from_summary(text, summary)
    print("FINDINGS words:", word_count(text), "| numbers not in summary.md:", missing)
    if missing:
        raise SystemExit("docs/FINDINGS.md quotes numbers that are not in results/summary.md")


def write_report_outputs(bars, cfg, window, sims, rows, differences, permutation, permuted,
                         checks, folds, trials, out, banner, findings_path):
    """Figures, summary.md, and (real run only) the results table inside docs/FINDINGS.md."""
    index = bars.index[window[0] : window[1]]
    per_side = cost_per_side_usd(cfg["costs"], cfg["instrument"])
    series = headline_series(sims, per_side, index, cfg)
    equity_figure(out.path("figures/equity_net.png"), series, per_side, banner)
    drawdown_figure(out.path("figures/drawdown_net.png"), series, per_side, banner)
    permutation_figure(out.path("figures/permutation_momentum.png"), permutation, permuted,
                       per_side, banner)

    gap_kinds = step_gap_kinds(bars.index, cfg["data"]["bar_minutes"])
    decomposition = passive_vs_tradable(bars, window, gap_kinds, sims, cfg)
    data_report = json.loads(out.path("data_report.json").read_text())
    summary = build_summary(data_report, folds, index, cfg, rows, differences, permutation,
                            checks, decomposition, len(trials), banner)
    out.path("summary.md").write_text(summary)
    if findings_path is not None:
        update_findings(findings_path, rows, per_side, summary)


def package_versions():
    """Pinned versions from requirements.txt next to the versions actually installed."""
    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line:
            name, pinned = line.strip().split("==")
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                installed = "not installed"
            versions[name] = {"pinned": pinned, "installed": installed}
    return versions


def git_output(*args):
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_info(path, cfg, data_path):
    """Ties every result to a code version, package versions, the data file hash and the config."""
    write_json(path, {
        "git_commit": git_output("rev-parse", "HEAD"),
        "source_changed_since_commit": bool(git_output(
            "status", "--porcelain", "--", "src", "config", "scripts", "requirements.txt")),
        "python": platform.python_version(),
        "packages": package_versions(),
        "data_files": {Path(data_path).name: sha256_of_file(data_path)},
        "config": cfg,
    })


def run(cfg, out, banner="", findings_path=None):
    """Run every Phase 1 experiment on the file at cfg['data']['path'] and write the report."""
    data_path = cfg["data"]["path"]
    if not Path(data_path).exists():
        raise SystemExit(f"{data_path} not found. See the README: place the Yahoo snapshot there.")
    bars = run_data_step(cfg, out.path("data_report.json"))
    add_sample_section(out.path("data_report.json"), sample_report(bars, cfg))

    check_embargo(cfg["splits"], cfg["horizon"])
    horizons = [cfg["horizon"]["main_bars"], cfg["horizon"]["secondary_bars"]]
    labels_by_horizon = {h: build_labels(bars, h, cfg["data"]["bar_minutes"]) for h in horizons}
    label_ends = {h: labels["label_end"].to_numpy() for h, labels in labels_by_horizon.items()}
    folds = fold_table(bars.index, label_ends, cfg["splits"])
    folds.to_csv(out.path("folds.csv"), index=False)

    write_json(out.path("execution_lag_check.json"), execution_lag_check(bars, cfg, banner))
    write_json(out.path("cost_levels.json"), {
        "assumption_note": "all cost values are assumptions, see config/default.yaml",
        "per_side_usd_main": cost_per_side_usd(cfg["costs"], cfg["instrument"]),
        "per_side_usd_conservative": cost_per_side_usd(
            cfg["costs"], cfg["instrument"], cfg["costs"]["conservative_commission_per_side_usd"]),
    })

    window = oos_window(len(bars), cfg["splits"])
    sims = build_baselines(bars, cfg, window, labels_by_horizon)
    trials, sha = TrialLog(), sha256_of_file(data_path)
    index = bars.index[window[0] : window[1]]
    rows, differences = evaluate(sims, index, cfg, trials, sha)
    to_frame(rows).to_csv(out.path("comparison.csv"), index=False)
    out.path("comparison.md").write_text(comparison_markdown(rows, differences, index, cfg, banner))

    permutation, permuted = momentum_permutation_tests(
        bars, cfg, window, labels_by_horizon, trials, sha, out)
    trials.write(out.path("trials.csv"))
    checks = leakage_checks(bars, label_ends, cfg, banner)
    write_json(out.path("leakage_check.json"), checks)

    write_report_outputs(bars, cfg, window, sims, rows, differences, permutation, permuted,
                         checks, folds, trials, out, banner, findings_path)
    write_run_info(out.path("run_info.json"), cfg, data_path)
    return {"rows": rows, "permutation": permutation, "checks": checks, "trials": len(trials)}


def run_smoke(cfg, directory):
    """The whole pipeline on SYNTHETIC bars. Every output file name starts with SYNTHETIC_."""
    from signal_lab.synthetic import make_raw_bars, write_yahoo_csv  # only the smoke run loads it

    smoke = cfg["smoke"]
    out = Outputs(directory, SMOKE_PREFIX)
    csv_path = out.path("bars.csv")
    write_yahoo_csv(make_raw_bars(smoke["synthetic_bars"], smoke["synthetic_seed"]), csv_path)
    smoke_cfg = with_overrides(cfg, smoke["overrides"])
    smoke_cfg["data"]["path"] = str(csv_path)
    return run(smoke_cfg, out, banner=SMOKE_BANNER, findings_path=None)

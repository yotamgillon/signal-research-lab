"""Evaluate strategies under both commission runs and write the comparison tables."""
import numpy as np
import pandas as pd

from signal_lab.costs import cost_per_side_usd
from signal_lab.metrics import bootstrap_sharpe_difference_ci, performance_summary, sharpe

LABELS = {
    "momentum": "momentum",
    "buy_and_hold_tradable": "buy-and-hold, tradable samples (like-for-like)",
    "buy_and_hold_passive": "buy-and-hold, passive (reference)",
}


def commission_runs(cfg):
    """The two runs: the researcher's real commission, and a conservative assumption."""
    costs = cfg["costs"]
    return {
        "main": costs["commission_per_side_usd"],
        "conservative": costs["conservative_commission_per_side_usd"],
    }


def log_trials(trials, name, horizon, run, commission, per_side, summary, cfg, window_text, sha):
    """One trial per strategy x horizon x commission run x cost multiplier."""
    for multiplier in cfg["costs"]["cost_multipliers"]:
        trials.add(
            strategy=name,
            horizon_bars=horizon if horizon is not None else "n/a",
            momentum_lookback=(cfg["baseline"]["momentum_lookback_bars"]
                               if name == "momentum" else "n/a"),
            commission_run=run,
            commission_per_side_usd=commission,
            cost_multiplier=multiplier,
            per_side_cost_usd=per_side * multiplier,
            gross_sharpe=summary["gross_sharpe"],
            net_sharpe=summary["net_sharpe_by_cost_multiplier"][f"{multiplier}x"],
            oos_window=window_text,
            data_sha256=sha,
        )


def evaluate(sims, index, cfg, trials, sha):
    """Summaries for every strategy, horizon and commission run, plus paired differences."""
    window_text = f"{index[0]} to {index[-1]}"
    rows, differences = [], []
    for run, commission in commission_runs(cfg).items():
        per_side = cost_per_side_usd(cfg["costs"], cfg["instrument"], commission)
        nets = {}
        for name, by_horizon in sims.items():
            for horizon, sim in by_horizon.items():
                summary = performance_summary(sim, per_side, index, cfg)
                nets[(name, horizon)] = summary["net_daily_pnl"]
                log_trials(trials, name, horizon, run, commission, per_side, summary, cfg,
                           window_text, sha)
                rows.append({"run": run, "commission_per_side_usd": commission,
                             "strategy": name, "horizon_bars": horizon,
                             "bars": len(index), **summary})
        for horizon in sims["momentum"]:
            a, b = nets[("momentum", horizon)], nets[("buy_and_hold_tradable", horizon)]
            point = _sharpe_gap(a, b, cfg)
            interval = bootstrap_sharpe_difference_ci(a, b, cfg["metrics"], cfg["seed"])
            differences.append({"run": run, "horizon_bars": horizon,
                                "net_sharpe_difference": point, "ci": interval})
    return rows, differences


def _sharpe_gap(a, b, cfg):
    return sharpe(a, cfg["metrics"]) - sharpe(b, cfg["metrics"])


def to_frame(rows):
    """Flat table for results/comparison.csv (the daily P&L arrays are dropped)."""
    flat = []
    for row in rows:
        row = {k: v for k, v in row.items() if k != "net_daily_pnl"}
        row["horizon_bars"] = row["horizon_bars"] if row["horizon_bars"] is not None else "n/a"
        for key in ("gross_sharpe_ci", "net_sharpe_ci"):
            row[key + "_low"], row[key + "_high"] = row.pop(key)
        for key, value in row.pop("net_sharpe_by_cost_multiplier").items():
            row[f"net_sharpe_at_{key}_costs"] = value
        flat.append(row)
    return pd.DataFrame(flat, dtype=object)


def _fmt(value, interval=None):
    text = "n/a" if value is None or np.isnan(value) else f"{value:.2f}"
    return text if interval is None else f"{text} [{interval[0]:.2f}, {interval[1]:.2f}]"


def header_text(rows, index, cfg):
    metrics = cfg["metrics"]
    days = rows[0]["trading_days"]
    years = days / metrics["trading_days_per_year"]
    window = f"Out-of-sample window: {index[0]} to {index[-1]}, {len(index)} bars, {days} trading days"
    basis = "Sharpe is from daily USD P&L for one contract, annualised with sqrt(252), risk-free zero."
    method = (f"Intervals are 95 percent circular block-bootstrap intervals "
              f"({metrics['bootstrap_paths']} paths, {metrics['bootstrap_block_days']}-day blocks).")
    return f"{window} (about {years:.1f} years). {basis} {method} With this few days they are wide."


def main_table(rows, run):
    head = ("| Strategy | H | Gross Sharpe [95% CI] | Net Sharpe [95% CI] | Sides | Sides/day | "
            "Net P&L/yr USD | Max DD USD | Hit rate | Break-even cost x |")
    lines = [head, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in (r for r in rows if r["run"] == run):
        cells = [LABELS[r["strategy"]], str(r["horizon_bars"] or "n/a"),
                 _fmt(r["gross_sharpe"], r["gross_sharpe_ci"]),
                 _fmt(r["net_sharpe"], r["net_sharpe_ci"]), f"{r['sides_traded']:.0f}",
                 f"{r['trades_per_day']:.2f}", f"{r['annual_net_pnl_usd']:,.0f}",
                 f"{r['max_drawdown_net_usd']:,.0f}", f"{r['hit_rate_days']:.2f}",
                 _fmt(r["break_even_multiplier"])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def multiplier_table(rows, run, cfg):
    multipliers = cfg["costs"]["cost_multipliers"]
    lines = ["| Strategy | H | " + " | ".join(f"{m}x" for m in multipliers) + " |",
             "| --- | --- | " + " | ".join("---" for _ in multipliers) + " |"]
    for r in (r for r in rows if r["run"] == run):
        by = r["net_sharpe_by_cost_multiplier"]
        cells = [LABELS[r["strategy"]], str(r["horizon_bars"] or "n/a")]
        cells += [_fmt(by[f"{m}x"]) for m in multipliers]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def comparison_markdown(rows, differences, index, cfg):
    """One block per commission run. Net and gross are always shown together."""
    lines = [header_text(rows, index, cfg)]
    for run, commission in commission_runs(cfg).items():
        per_side = cost_per_side_usd(cfg["costs"], cfg["instrument"], commission)
        title = (f"### Run: {run} (commission {commission:.2f} USD per side, "
                 f"{per_side:.2f} USD per side per contract in total, ASSUMPTIONS)")
        lines += ["", title, ""] + main_table(rows, run)
        lines += ["", "Net Sharpe at 0x, 1x, 2x, 3x of the assumed costs:", ""]
        lines += multiplier_table(rows, run, cfg)
        gap_title = ("Momentum minus like-for-like buy-and-hold, net Sharpe difference "
                     "(paired bootstrap):")
        lines += ["", gap_title]
        for d in (d for d in differences if d["run"] == run):
            lines.append(f"- H={d['horizon_bars']}: {_fmt(d['net_sharpe_difference'], d['ci'])}")
    return "\n".join(lines) + "\n"

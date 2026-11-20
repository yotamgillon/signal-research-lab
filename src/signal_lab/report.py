"""Evaluate strategies under both commission runs and write the comparison tables."""
import re

import numpy as np
import pandas as pd

from signal_lab.backtest import net_pnl_usd
from signal_lab.costs import cost_per_side_usd
from signal_lab.metrics import (
    bootstrap_sharpe_difference_ci,
    daily_pnl,
    performance_summary,
    sharpe,
)

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


# ---------------------------------------------------------------------------
# Passive versus tradable buy-and-hold
# ---------------------------------------------------------------------------

def passive_vs_tradable(bars, window, gap_kinds, sims, cfg):
    """Explain exactly why the passive buy-and-hold earns more than the tradable one (H=1).

    Per bar, passive P&L = (close - open) + (open - previous close), with the expiry bar set to
    zero. The tradable line earns only the in-bar move (close - open) of bars whose sample is
    valid. So: passive - tradable = in-bar moves of non-tradable bars + moves between bars.
    """
    start, stop = window
    part = bars.iloc[start:stop]
    mult = cfg["instrument"]["multiplier_usd"]
    expiry = part["expiry_bar"].to_numpy()
    in_bar = np.where(expiry, 0.0, part["close"].to_numpy() - part["open"].to_numpy())
    previous_close = np.concatenate([[part["open"].iloc[0]], part["close"].to_numpy()[:-1]])
    between = np.where(expiry, 0.0, part["open"].to_numpy() - previous_close)

    valid = build_valid_mask(bars, cfg, 1)[start:stop]
    tradable = np.concatenate([[False], valid[:-1]])  # bar i is earned if sample i-1 is valid
    after_gap = gap_kinds[start:stop] != ""
    weekend = gap_kinds[start:stop] == "weekend"

    passive = float(sims["buy_and_hold_passive"][None]["gross_pnl_usd"].sum())
    tradable_total = float(sims["buy_and_hold_tradable"][1]["gross_pnl_usd"].sum())
    check = float(in_bar[tradable].sum() * mult)
    if abs(check - tradable_total) > 1e-6 or abs(passive - (in_bar.sum() + between.sum()) * mult) > 1e-6:
        raise ValueError("passive versus tradable decomposition does not add up")
    return {
        "passive_gross_usd": passive,
        "tradable_gross_usd": tradable_total,
        "difference_usd": passive - tradable_total,
        "in_bar_moves_of_non_tradable_bars_usd": float(in_bar[~tradable].sum() * mult),
        "first_bars_after_a_gap": int(after_gap.sum()),
        "first_bars_after_a_gap_usd": float(in_bar[after_gap].sum() * mult),
        "weekend_reopen_bars": int(weekend.sum()),
        "weekend_reopen_bars_usd": float(in_bar[weekend].sum() * mult),
        "moves_between_bars_usd": float(between.sum() * mult),
        "moves_between_bars_across_gaps_usd": float(between[after_gap].sum() * mult),
    }


def build_valid_mask(bars, cfg, horizon):
    from signal_lab.labels import build_labels

    return build_labels(bars, horizon, cfg["data"]["bar_minutes"])["valid"].to_numpy()


# ---------------------------------------------------------------------------
# Figures. Real data only. Palette slots 1 to 3 of the reference palette, validated.
# ---------------------------------------------------------------------------

INK, INK_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"momentum": "#2a78d6", "buy_and_hold_tradable": "#eb6834",
          "buy_and_hold_passive": "#1baf7a"}
SERIES_LABELS = {"momentum": "Momentum, L=24, H=1",
                 "buy_and_hold_tradable": "Buy-and-hold, tradable (like-for-like)",
                 "buy_and_hold_passive": "Buy-and-hold, passive (reference)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _new_figure(*args, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return plt, fig, axes


def net_daily_series(sim, per_side_usd, index, cfg):
    """Net daily P&L as a Series indexed by CME trading day."""
    return daily_pnl(net_pnl_usd(sim, per_side_usd), index, cfg["data"])


def headline_series(sims, per_side_usd, index, cfg):
    """The three lines of the equity and drawdown figures: net, main horizon H=1."""
    picks = {"momentum": sims["momentum"][1],
             "buy_and_hold_tradable": sims["buy_and_hold_tradable"][1],
             "buy_and_hold_passive": sims["buy_and_hold_passive"][None]}
    return {name: net_daily_series(sim, per_side_usd, index, cfg) for name, sim in picks.items()}


def _line_chart(path, series, transform, title, ylabel):
    plt, fig, ax = _new_figure(figsize=(9, 5))
    _style(ax)
    for name, daily in series.items():
        values = transform(daily.to_numpy())
        ax.plot(daily.index, values, color=COLORS[name], linewidth=2,
                linestyle="--" if name == "buy_and_hold_passive" else "-",
                label=SERIES_LABELS[name])
        ax.annotate(f"{values[-1]:,.0f}", (daily.index[-1], values[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("CME trading day (out-of-sample window)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def equity_figure(path, series, per_side_usd):
    title = (f"Net equity, one NQ contract, out-of-sample. Costs are ASSUMPTIONS: "
             f"{per_side_usd:.2f} USD per side")
    _line_chart(path, series, np.cumsum, title, "Cumulative net P&L (USD)")


def drawdown_figure(path, series, per_side_usd):
    def drawdown(values):
        equity = np.concatenate([[0.0], np.cumsum(values)])
        return (equity - np.maximum.accumulate(equity))[1:]

    title = (f"Drawdown from peak, one NQ contract, out-of-sample. Costs are ASSUMPTIONS: "
             f"{per_side_usd:.2f} USD per side")
    _line_chart(path, series, drawdown, title, "Net drawdown from peak (USD)")


def permutation_figure(path, results, permuted, per_side_usd):
    horizons = list(results)
    plt, fig, axes = _new_figure(1, len(horizons), figsize=(10, 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, key in zip(axes, horizons, strict=True):
        _style(ax)
        result = results[key]
        ax.hist(permuted[key.replace("=", "")], bins=24, color=COLORS["momentum"], edgecolor=SURFACE)
        ax.axvline(result["real_net_sharpe"], color=INK, linewidth=2)
        ax.text(0.97, 0.95, f"real net Sharpe {result['real_net_sharpe']:.2f}\np-value {result['p_value']:.3f}",
                transform=ax.transAxes, color=INK, fontsize=9, ha="right", va="top")
        ax.set_title(f"Momentum {key}: {result['n_permutations']} circular shifts", color=INK,
                     fontsize=10, loc="left")
        ax.set_xlabel("Net Sharpe of the shifted signal (annualised)")
    axes[0].set_ylabel("Number of permutations")
    fig.suptitle(f"Permutation null of net Sharpe. Costs are ASSUMPTIONS: {per_side_usd:.2f} USD "
                 "per side", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md and the FINDINGS number check
# ---------------------------------------------------------------------------

def _money(value):
    return f"{value:,.0f}"


def facts_section(data_report, folds, index, cfg, rows, trials_count):
    samples = data_report["samples"]
    days = rows[0]["trading_days"]
    top = data_report["largest_single_bar_moves"]
    before = [m["timestamp"][:10] for m in top if m["timestamp"] < str(index[0])]
    per_side = cost_per_side_usd(cfg["costs"], cfg["instrument"])
    conservative = cost_per_side_usd(cfg["costs"], cfg["instrument"],
                                     cfg["costs"]["conservative_commission_per_side_usd"])
    return [
        "## Data and window", "",
        f"- Data file {data_report['file']}, sha256 {data_report['sha256']}.",
        f"- Rows in file {data_report['rows_in_file']:,}, rows used after dropping the latest "
        f"bar {data_report['rows_used']:,}.",
        f"- Out-of-sample window {str(index[0])[:10]} to {str(index[-1])[:10]}: {len(index):,} "
        f"bars, {days} trading days, {days / cfg['metrics']['trading_days_per_year']:.1f} years.",
        f"- Walk-forward folds: {len(folds)}, expanding training window, test blocks of "
        f"{folds['n_test_bars'].iloc[0]} bars ({folds['n_test_bars'].iloc[-1]} bars in the last "
        f"fold), embargo {cfg['splits']['embargo_bars']} bars, first test block starts "
        f"{str(index[0])[:10]}.",
        "- Expiry rule (pre-declared): the return of the 16:00 New York bar on the third Friday of "
        "March, June, September and December is treated as missing.",
        f"- {len(before)} of the {len(top)} largest single-bar moves fall before the first test "
        f"block, between {min(before)} and {max(before)} (April 2025 tariff period): they are in "
        "training only.",
        f"- Cost per side per contract (ASSUMPTIONS): main {per_side:.2f} USD "
        f"(commission {cfg['costs']['commission_per_side_usd']:.2f}, slippage "
        f"{cfg['costs']['slippage_ticks']:.1f} tick, half-spread {cfg['costs']['half_spread_ticks']:.1f} "
        f"tick, {cfg['instrument']['tick_value_usd']:.0f} USD per tick); conservative "
        f"{conservative:.2f} USD.",
        f"- Zero-volume bars {data_report['zero_volume_bars']}, of which "
        f"{data_report['zero_volume_by_start_hour_ny']['18']} are the 18:00 New York bar.",
        f"- Bars not starting on the hour {data_report['bars_not_starting_on_the_hour']}.",
        f"- Expiry bars set to missing {len(data_report['expiry_bars_set_to_missing'])}; "
        f"holiday fallback used for {data_report['expiry_rule']['fallback_used_for'][0]['third_friday']} "
        "(an assumption, declared 2026-11-20).",
        f"- Valid samples after label rules: H=1 {samples['H=1']['samples_valid_after_label_rules']:,}, "
        f"H=2 {samples['H=2']['samples_valid_after_label_rules']:,}; gap rule removes "
        f"{samples['H=1']['removed_by_gap_rule']:,} (H=1) and {samples['H=2']['removed_by_gap_rule']:,} (H=2).",
        f"- Strategy configurations logged in results/trials.csv: {trials_count}.",
        f"- Momentum lookback {cfg['baseline']['momentum_lookback_bars']} bars, horizons "
        f"{cfg['horizon']['main_bars']} and {cfg['horizon']['secondary_bars']} bars, "
        f"{cfg['metrics']['bootstrap_paths']} bootstrap paths, {cfg['metrics']['bootstrap_block_days']}-day "
        f"blocks, {cfg['permutation']['n_permutations']} permutations.",
    ]


def decomposition_section(d):
    return [
        "## Passive versus tradable buy-and-hold (H=1, gross USD, one contract)", "",
        f"- Passive {_money(d['passive_gross_usd'])}, tradable {_money(d['tradable_gross_usd'])}, "
        f"difference {_money(d['difference_usd'])}.",
        f"- In-bar moves of bars a signal cannot trade: {_money(d['in_bar_moves_of_non_tradable_bars_usd'])}, "
        f"of which {_money(d['first_bars_after_a_gap_usd'])} from the {d['first_bars_after_a_gap']} first "
        f"bars after a gap, and {_money(d['weekend_reopen_bars_usd'])} from the "
        f"{d['weekend_reopen_bars']} weekend reopen bars.",
        f"- Moves between one bar's close and the next bar's open: {_money(d['moves_between_bars_usd'])}, "
        f"of which {_money(d['moves_between_bars_across_gaps_usd'])} across gaps.",
    ]


def permutation_section(results):
    lines = ["## Permutation tests (momentum, circular shift, main commission)", ""]
    for key, r in results.items():
        lines.append(
            f"- {key}: real net Sharpe {r['real_net_sharpe']:.2f}, p-value {r['p_value']:.3f} "
            f"({r['permuted_at_least_as_large']} of {r['n_permutations']} shifted signals at least as "
            f"high), smallest possible p-value {r['minimum_possible_p_value']:.3f}, null mean "
            f"{r['permuted_mean']:.2f}, null 5th to 95th percentile {r['permuted_5th_percentile']:.2f} "
            f"to {r['permuted_95th_percentile']:.2f}.")
    return lines


def leakage_section(checks):
    return [
        "## Leakage checks on the real bars", "",
        f"- Truncation test at {checks['cut_points']} cut points: clean features "
        f"{checks['clean_features_truncation_violations']} violations; canary fails at "
        f"{checks['canary_truncation_cuts_with_violation']} of {checks['cut_points']} cuts.",
        f"- Training label windows reaching into a test block: "
        f"{checks['label_boundary_violating_folds']['H=1']} (H=1), "
        f"{checks['label_boundary_violating_folds']['H=2']} (H=2).",
    ]


def build_summary(data_report, folds, index, cfg, rows, differences, permutation, checks,
                  decomposition, trials_count):
    head = ["# Summary of results", "",
            "Generated by scripts/run_all.py. Do not edit by hand. Every number in "
            "docs/FINDINGS.md must appear here.", ""]
    tables = ["", "## Comparison", "", comparison_markdown(rows, differences, index, cfg).rstrip()]
    sections = (head + facts_section(data_report, folds, index, cfg, rows, trials_count) + tables
                + [""] + decomposition_section(decomposition) + [""]
                + permutation_section(permutation) + [""] + leakage_section(checks))
    return "\n".join(sections) + "\n"


TABLE_START, TABLE_END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
STRUCTURAL = {"1", "2"}  # H=1, H=2 and "Phase 1" are labels, not results


def insert_results_table(findings_text, table_lines):
    """Replace the block between the table markers. The script writes the table, not a person."""
    start, end = findings_text.index(TABLE_START), findings_text.index(TABLE_END)
    block = TABLE_START + "\n" + "\n".join(table_lines) + "\n" + TABLE_END
    return findings_text[:start] + block + findings_text[end + len(TABLE_END):]


def prose_without_table(findings_text):
    if TABLE_START not in findings_text:
        return findings_text
    start, end = findings_text.index(TABLE_START), findings_text.index(TABLE_END)
    return findings_text[:start] + findings_text[end + len(TABLE_END):]


def word_count(findings_text):
    return len(prose_without_table(findings_text).split())


def numbers_in(text):
    dates = set(DATE.findall(text))
    tokens = {t.replace(",", "").lstrip("+") for t in NUMBER.findall(DATE.sub(" ", text))}
    return dates | {t.rstrip(".") for t in tokens} - STRUCTURAL


def numbers_missing_from_summary(findings_text, summary_text):
    """Numbers quoted in the FINDINGS prose that do not appear in summary.md."""
    return sorted(numbers_in(prose_without_table(findings_text)) - numbers_in(summary_text))

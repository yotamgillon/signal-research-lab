"""Forward-return labels and the rules that exclude samples.

Timing convention (SPEC section 2): the signal is computed from data up to and including the
close of bar t, the trade enters at the open of bar t+1 and exits at the close of bar t+H.
    y(t) = log(close[t+H] / open[t+1])
"""
import json

import numpy as np
import pandas as pd

from signal_lab.data import classify_gaps

GAP_KINDS = ["daily_break", "weekend", "holiday", "other"]


def window_has(flag, first, last):
    """For each bar t: is `flag` set anywhere in bars t+first .. t+last (inclusive)?

    Windows that run off either end of the data give False. Uses a running sum so the cost is
    linear in the number of bars.
    """
    flag = np.asarray(flag, dtype=bool)
    n = len(flag)
    running = np.concatenate([[0], np.cumsum(flag)])
    start = np.arange(n) + first
    stop = np.arange(n) + last
    inside = (start >= 0) & (stop <= n - 1)
    out = np.zeros(n, dtype=bool)
    out[inside] = running[stop[inside] + 1] - running[start[inside]] > 0
    return out


def step_gap_kinds(index, bar_minutes):
    """For each bar i: the kind of gap between bar i-1 and bar i, or '' if there is none."""
    kinds = np.full(len(index), "", dtype=object)
    gaps = classify_gaps(index, bar_minutes)
    kinds[index.get_indexer(gaps["next_bar"])] = gaps["kind"].to_numpy()
    return kinds


def build_labels(bars, horizon, bar_minutes):
    """One row per bar t: label y, the bar where its label window ends, and why it is excluded.

    A sample is excluded when (1) the label window runs past the end of the data, (2) any gap
    longer than one bar lies between bar t and bar t+H, or (3) an expiry bar with a missing
    return lies inside the label window t+1 .. t+H.
    """
    n = len(bars)
    end = np.arange(n) + horizon
    has_future = end <= n - 1
    open_next = np.full(n, np.nan)
    open_next[:-1] = bars["open"].to_numpy()[1:]
    close_end = np.full(n, np.nan)
    close_end[has_future] = bars["close"].to_numpy()[end[has_future]]

    step_is_gap = step_gap_kinds(bars.index, bar_minutes) != ""
    gap = window_has(step_is_gap, 1, horizon) & has_future
    missing_return = window_has(bars["expiry_bar"].to_numpy(), 1, horizon)
    valid = has_future & ~gap & ~missing_return

    labels = pd.DataFrame(index=bars.index)
    labels["y"] = np.where(valid, np.log(close_end / open_next), np.nan)
    labels["label_end"] = np.where(has_future, end, np.nan)  # bar position of the exit
    labels["no_future"] = ~has_future
    labels["gap"] = gap
    labels["missing_return"] = missing_return & has_future
    labels["valid"] = valid
    return labels


def feature_window_bars(features_cfg):
    """Bars a feature needs, ending at t. The longest of all declared windows."""
    return max(
        max(features_cfg["return_lags"]) + 1,
        features_cfg["vol_window"] + 1,
        features_cfg["ma_window"],
        features_cfg["range_window"],
    )


def gap_rule_by_type(bars, horizon, bar_minutes):
    """Samples removed by the gap rule, split by the kind of gap inside the window."""
    n = len(bars)
    has_future = np.arange(n) + horizon <= n - 1
    kinds = step_gap_kinds(bars.index, bar_minutes)
    counts = {
        kind: int((window_has(kinds == kind, 1, horizon) & has_future).sum())
        for kind in GAP_KINDS
    }
    counts["note"] = "a window with two kinds of gap counts once under each kind"
    return counts


def zero_volume_touches(bars, labels, horizon, window):
    """Valid samples whose label or feature window contains a zero-volume bar."""
    zero = (bars["volume"] == 0).to_numpy()
    complete = np.arange(len(bars)) >= window - 1
    base = labels["valid"].to_numpy() & complete
    in_label = window_has(zero, 1, horizon) & base
    in_feature = window_has(zero, -(window - 1), 0) & base
    return {
        "valid_samples_with_complete_feature_window": int(base.sum()),
        "label_window_touches_zero_volume_bar": int(in_label.sum()),
        "feature_window_touches_zero_volume_bar": int(in_feature.sum()),
        "either": int((in_label | in_feature).sum()),
    }


def feature_window_expiry_removals(bars, labels, window):
    """Valid samples removed because a feature window contains an expiry bar (missing return)."""
    complete = np.arange(len(bars)) >= window - 1
    touches = window_has(bars["expiry_bar"].to_numpy(), -(window - 1), 0)
    return int((touches & labels["valid"].to_numpy() & complete).sum())


def sample_report(bars, cfg):
    """Sample-level counts for the data report, per horizon."""
    bar_minutes = cfg["data"]["bar_minutes"]
    window = feature_window_bars(cfg["features"])
    report = {
        "rule_rows_note": (
            "The rule rows below overlap: a sample removed by two rules appears in both rows, "
            "so the rows do not sum to the removed total. samples_valid_after_label_rules = "
            "samples_total minus removed_by_any_label_rule, where a sample removed by several "
            "rules counts once. Feature-window rows are separate and are not part of that count."
        ),
        "feature_window_bars": window,
        "feature_window_note": (
            "samples with fewer than this many bars of history are warm-up and cannot have "
            "features; the feature rules below are counted among label-valid samples"
        ),
    }
    for horizon in (cfg["horizon"]["main_bars"], cfg["horizon"]["secondary_bars"]):
        labels = build_labels(bars, horizon, bar_minutes)
        rule_rows = [labels["no_future"], labels["gap"], labels["missing_return"]]
        sum_of_rows = int(sum(row.sum() for row in rule_rows))
        removed_any = int((~labels["valid"]).sum())
        report[f"H={horizon}"] = {
            "samples_total": len(bars),
            "removed_label_runs_past_end_of_data": int(labels["no_future"].sum()),
            "removed_by_gap_rule": int(labels["gap"].sum()),
            "removed_by_any_label_rule": removed_any,
            "sum_of_the_three_rule_rows": sum_of_rows,
            "counted_in_more_than_one_rule_row": sum_of_rows - removed_any,
            "gap_rule_by_type": gap_rule_by_type(bars, horizon, bar_minutes),
            "removed_by_expiry_missing_return_in_label_window": int(labels["missing_return"].sum()),
            "samples_valid_after_label_rules": int(labels["valid"].sum()),
            "removed_by_expiry_missing_return_in_feature_window": feature_window_expiry_removals(
                bars, labels, window
            ),
            "zero_volume_touches": zero_volume_touches(bars, labels, horizon, window),
        }
    return report


def add_sample_section(report_path, section):
    with open(report_path) as handle:
        report = json.load(handle)
    report["samples"] = section
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

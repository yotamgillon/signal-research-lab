"""Walk-forward splits with purge and embargo.

Positions are bar numbers 0 .. n-1. A train sample t has a label window that ends at bar
label_end[t]. The test block is a contiguous run of bars, touched once.
"""
import numpy as np
import pandas as pd


def check_embargo(splits_cfg, horizon_cfg):
    longest = max(horizon_cfg["main_bars"], horizon_cfg["secondary_bars"])
    if splits_cfg["embargo_bars"] < longest:
        raise ValueError("embargo_bars must be at least the longest horizon")


def block_bounds(n_bars, splits_cfg):
    """(start, stop) of every test block. The first starts after min_train_bars bars."""
    size = splits_cfg["test_bars"]
    start = splits_cfg["min_train_bars"]
    blocks = []
    while start < n_bars:
        stop = min(start + size, n_bars)
        if stop - start < size and not splits_cfg["keep_partial_last_block"]:
            break  # fixed-length blocks only
        blocks.append((start, stop))
        start = stop
    return blocks


def train_positions(label_end, test_start, splits_cfg):
    """Train samples for one fold.

    Embargo: train stops embargo_bars before the test block.
    Purge: a sample is dropped if its label window reaches into the test block, and so is a
    sample with no label (NaN). Rolling scheme: only the last rolling_train_bars bars.
    Why lookback features that cross the boundary are fine: they read only the past.
    """
    train_stop = test_start - splits_cfg["embargo_bars"]
    train_start = 0
    if splits_cfg["scheme"] == "rolling":
        train_start = max(0, train_stop - splits_cfg["rolling_train_bars"])
    candidates = np.arange(train_start, train_stop)
    ends = np.asarray(label_end, dtype=float)[candidates]
    keep = ~np.isnan(ends) & (ends < test_start)
    return candidates[keep]


def walk_forward_splits(label_end, splits_cfg):
    """Yield (train_index, test_index) for every fold, in time order."""
    for start, stop in block_bounds(len(label_end), splits_cfg):
        yield train_positions(label_end, start, splits_cfg), np.arange(start, stop)


def fold_table(index, label_ends, splits_cfg):
    """One row per fold for results/folds.csv. label_ends maps a horizon to its label_end array."""
    rows = []
    for fold_id, (start, stop) in enumerate(block_bounds(len(index), splits_cfg)):
        row = {"fold": fold_id, "test_first_bar": index[start], "test_last_bar": index[stop - 1],
               "n_test_bars": stop - start}
        for horizon, label_end in label_ends.items():
            train = train_positions(label_end, start, splits_cfg)
            row[f"train_first_bar_H{horizon}"] = index[train[0]]
            row[f"train_last_bar_H{horizon}"] = index[train[-1]]
            row[f"n_train_samples_H{horizon}"] = len(train)
        rows.append(row)
    return pd.DataFrame(rows)

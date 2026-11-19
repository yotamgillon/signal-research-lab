"""Purge, embargo and contiguity hold for every fold. Small deterministic arrays only."""
import itertools

import numpy as np
import pandas as pd
import pytest

from signal_lab.splits import (
    block_bounds,
    check_embargo,
    fold_table,
    train_positions,
    walk_forward_splits,
)

N = 100


def cfg(**changes):
    base = {"scheme": "expanding", "min_train_bars": 30, "test_bars": 10, "embargo_bars": 2,
            "rolling_train_bars": 20, "keep_partial_last_block": False}
    base.update(changes)
    return base


def label_end_for(horizon, n=N):
    ends = np.arange(n, dtype=float) + horizon
    ends[ends > n - 1] = np.nan  # no label at the end of the data
    return ends


def all_folds(label_end, splits_cfg):
    return list(walk_forward_splits(label_end, splits_cfg))


def test_latest_train_label_end_is_before_first_test_bar():
    index = pd.date_range("2025-01-01", periods=N, freq="h", tz="America/New_York")
    for horizon in (1, 2):
        label_end = label_end_for(horizon)
        for train, test in all_folds(label_end, cfg()):
            last_label_end_time = index[int(label_end[train].max())]
            assert last_label_end_time < index[test[0]]


def test_embargo_gap_is_at_least_embargo_bars():
    for train, test in all_folds(label_end_for(2), cfg(embargo_bars=3)):
        assert test[0] - train[-1] - 1 >= 3


def test_purge_alone_drops_samples_whose_label_reaches_the_test_block():
    # No embargo, label window of 3 bars: the last 3 samples before the block must go.
    label_end = label_end_for(3)
    train = train_positions(label_end, 30, cfg(embargo_bars=0))
    assert train[-1] == 26
    assert label_end[train].max() < 30


def test_a_split_without_purge_would_fail_the_same_check():
    # Test of the test: build the naive train set and show the check catches the leak.
    label_end = label_end_for(3)
    naive_train = np.arange(0, 30)
    assert label_end[naive_train].max() >= 30


def test_test_blocks_are_contiguous_and_do_not_overlap():
    folds = all_folds(label_end_for(1), cfg())
    tests = [test for _, test in folds]
    joined = np.concatenate(tests)
    assert joined.tolist() == list(range(30, 30 + 10 * len(tests)))
    for earlier, later in itertools.pairwise(tests):
        assert later[0] == earlier[-1] + 1


def test_fold_count_and_unused_tail():
    # 100 bars, first test at 30, blocks of 10: 7 full blocks, nothing left over.
    assert len(block_bounds(N, cfg())) == 7
    # 95 bars: 6 full blocks, 5 bars left unused unless the partial block is kept.
    assert len(block_bounds(95, cfg())) == 6
    assert len(block_bounds(95, cfg(keep_partial_last_block=True))) == 7


def test_expanding_train_grows_and_starts_at_zero():
    sizes = []
    for train, _ in all_folds(label_end_for(1), cfg()):
        assert train[0] == 0
        sizes.append(len(train))
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_rolling_train_window_is_bounded():
    for train, test in all_folds(label_end_for(1), cfg(scheme="rolling")):
        assert len(train) <= 20
        assert train[0] >= test[0] - 2 - 20


def test_samples_without_a_label_are_never_trained_on():
    label_end = label_end_for(1)
    label_end[10:15] = np.nan  # e.g. removed by the gap rule
    for train, _ in all_folds(label_end, cfg()):
        assert not set(range(10, 15)) & set(train.tolist())


def test_embargo_shorter_than_horizon_is_rejected():
    horizon_cfg = {"main_bars": 1, "secondary_bars": 2}
    with pytest.raises(ValueError):
        check_embargo(cfg(embargo_bars=1), horizon_cfg)
    check_embargo(cfg(embargo_bars=2), horizon_cfg)


def test_fold_table_columns_and_counts():
    index = pd.date_range("2025-01-01", periods=N, freq="h", tz="America/New_York")
    table = fold_table(index, {1: label_end_for(1), 2: label_end_for(2)}, cfg())
    assert len(table) == 7
    assert table["n_test_bars"].tolist() == [10] * 7
    assert table["n_train_samples_H1"].iloc[0] == 28   # bars 0..27 (embargo 2)
    assert table["n_train_samples_H2"].iloc[0] == 28

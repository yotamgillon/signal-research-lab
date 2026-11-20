"""A planted signal is detected, noise is not. Synthetic data only."""
import functools

import numpy as np
import pytest

from signal_lab.config import load_config
from signal_lab.data import add_returns
from signal_lab.stats_tests import (
    circular_shift_test,
    label_shuffle_test,
    net_sharpe_statistic,
    summarize,
)
from signal_lab.synthetic import make_raw_bars

CFG = load_config()
SEEDS = CFG["tests"]["noise_seeds"]
N_PERMUTATIONS = CFG["permutation"]["n_permutations"]
SPLITS = {"scheme": "expanding", "min_train_bars": 600, "test_bars": 150, "embargo_bars": 2,
          "rolling_train_bars": 600, "keep_partial_last_block": True}
N = 1500


def least_squares(x_train, y_train, x_test):
    design = np.column_stack([np.ones(len(x_train)), x_train])
    beta = np.linalg.lstsq(design, y_train, rcond=None)[0]
    return np.column_stack([np.ones(len(x_test)), x_test]) @ beta


def sharpe_of(signal, outcome):
    pnl = (signal * outcome)[np.isfinite(signal)]
    return pnl.mean() / pnl.std()


def trained_case(planted, seed):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N, 3))
    labels = (0.5 * features[:, 0] if planted else 0.0) + rng.normal(size=N)
    label_end = np.arange(N) + 1.0
    label_end[-1] = np.nan
    statistic = lambda signal: sharpe_of(signal, labels)
    return label_shuffle_test(statistic, least_squares, features, labels, label_end, SPLITS,
                              N_PERMUTATIONS, seed)


def test_p_value_formula_and_minimum():
    result = summarize(5.0, [1.0, 2.0, 6.0, 7.0], "x")
    assert result["p_value"] == pytest.approx((1 + 2) / (4 + 1))
    assert summarize(9.0, [1.0, 2.0], "x")["p_value"] == pytest.approx(1 / 3)
    assert summarize(9.0, [1.0, 2.0], "x")["minimum_possible_p_value"] == pytest.approx(1 / 3)


def test_label_shuffle_detects_a_planted_signal():
    result, _ = trained_case(planted=True, seed=1)
    assert result["p_value"] == pytest.approx(1 / (N_PERMUTATIONS + 1))


def test_label_shuffle_does_not_reject_noise_at_one_percent():
    for seed in SEEDS:
        result, permuted = trained_case(planted=False, seed=seed)
        assert result["p_value"] > 0.01, f"seed {seed}"
        assert abs(permuted.mean()) < 0.1  # the null is centred near zero


def returns_and_signals(seed):
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=N)
    planted = np.sign(returns + 0.5 * rng.normal(size=N))  # knows the returns, with some noise
    noise = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    return returns, planted, noise


def test_circular_shift_detects_a_planted_signal():
    returns, planted, _ = returns_and_signals(1)
    result, _ = circular_shift_test(lambda s: sharpe_of(s, returns), planted, N_PERMUTATIONS,
                                    CFG["permutation"]["min_shift_bars"], seed=1)
    assert result["p_value"] == pytest.approx(1 / (N_PERMUTATIONS + 1))


def test_circular_shift_does_not_reject_noise_at_one_percent():
    for seed in SEEDS:
        returns, _, noise = returns_and_signals(seed)
        result, permuted = circular_shift_test(
            functools.partial(sharpe_of, outcome=returns), noise, N_PERMUTATIONS,
            CFG["permutation"]["min_shift_bars"], seed)
        assert result["p_value"] > 0.01, f"seed {seed}"
        assert abs(permuted.mean()) < 0.1


def test_circular_shifts_are_rotations_at_least_min_shift_away():
    signal = np.arange(100.0)
    seen = []
    circular_shift_test(lambda s: seen.append(s.copy()) or 0.0, signal, 30, 10, seed=3)
    seen = seen[:-1]  # the last call is the real, unshifted signal
    for rolled in seen:
        shift = int(np.flatnonzero(rolled == 0.0)[0])  # where the first element went
        assert 10 <= shift <= 90
        assert np.array_equal(rolled, np.roll(signal, shift))


def test_momentum_style_signal_on_a_random_walk_is_not_significant():
    raw = make_raw_bars(n_hours=3000, seed=4)
    bars = add_returns(raw, CFG["cleaning"])
    trailing = bars["ret"].rolling(24).sum()
    signal = np.where(trailing > 0, 1.0, -1.0)
    signal[trailing.isna().to_numpy()] = np.nan
    statistic = net_sharpe_statistic(bars, np.ones(len(bars), bool), 1, 7.5, CFG)
    result, permuted = circular_shift_test(statistic, signal, N_PERMUTATIONS,
                                           CFG["permutation"]["min_shift_bars"], seed=7)
    assert result["p_value"] > 0.01
    assert np.isfinite(permuted).all()

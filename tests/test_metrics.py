"""Sharpe uses daily USD P&L and sqrt(252); drawdown and trading days are right."""
import numpy as np
import pandas as pd
import pytest

from signal_lab.config import load_config
from signal_lab.metrics import daily_pnl, max_drawdown_usd, sharpe

CFG = load_config()


def test_sharpe_matches_hand_computation():
    daily = [1.0, 3.0]  # mean 2, sample std sqrt(2)
    assert sharpe(daily, CFG["metrics"]) == pytest.approx(2 / np.sqrt(2) * np.sqrt(252))


def test_sharpe_is_nan_without_variation():
    assert np.isnan(sharpe([1.0, 1.0, 1.0], CFG["metrics"]))
    assert np.isnan(sharpe([1.0], CFG["metrics"]))


def test_daily_aggregation_uses_the_cme_trading_day():
    # 18:00 New York is 17:00 Chicago, the start of the next trading day.
    index = pd.DatetimeIndex(
        ["2025-01-06 16:00", "2025-01-06 18:00", "2025-01-07 10:00"]
    ).tz_localize("America/New_York")
    daily = daily_pnl([1.0, 2.0, 3.0], index, CFG["data"])
    assert daily.tolist() == [1.0, 5.0]


def test_intraday_pnl_is_not_scaled_per_bar():
    # Ten equal bar P&Ls inside one day make one daily value, not ten observations.
    index = pd.date_range("2025-01-07 08:00", periods=10, freq="h", tz="America/New_York")
    assert daily_pnl(np.ones(10), index, CFG["data"]).tolist() == [10.0]


def test_max_drawdown():
    # equity 0, 2, -1, 0, -1, 3: the deepest fall is from 2 to -1
    assert max_drawdown_usd([2.0, -3.0, 1.0, -1.0, 4.0]) == 3.0
    assert max_drawdown_usd([1.0, 1.0]) == 0.0


def small_cfg(block_days=5, paths=500):
    return {**CFG["metrics"], "bootstrap_block_days": block_days, "bootstrap_paths": paths}


def test_bootstrap_ci_is_reproducible_and_contains_the_point_estimate():
    from signal_lab.metrics import bootstrap_sharpe_ci

    daily = np.random.default_rng(1).normal(0.3, 1.0, size=300)
    first = bootstrap_sharpe_ci(daily, small_cfg(), seed=42)
    assert first == bootstrap_sharpe_ci(daily, small_cfg(), seed=42)
    assert first != bootstrap_sharpe_ci(daily, small_cfg(), seed=43)
    assert first[0] < sharpe(daily, CFG["metrics"]) < first[1]


def test_bootstrap_ci_is_wider_with_fewer_days():
    from signal_lab.metrics import bootstrap_sharpe_ci

    daily = np.random.default_rng(2).normal(0.1, 1.0, size=800)
    long_ci = bootstrap_sharpe_ci(daily, small_cfg(), seed=1)
    short_ci = bootstrap_sharpe_ci(daily[:100], small_cfg(), seed=1)
    assert (short_ci[1] - short_ci[0]) > 2 * (long_ci[1] - long_ci[0])


def test_bootstrap_ci_excludes_zero_for_a_strong_edge_and_includes_it_for_noise():
    from signal_lab.metrics import bootstrap_sharpe_ci

    rng = np.random.default_rng(3)
    strong = bootstrap_sharpe_ci(rng.normal(1.0, 1.0, size=400), small_cfg(), seed=1)
    assert strong[0] > 0
    noise_covers = []
    for seed in CFG["tests"]["noise_seeds"]:
        noise = np.random.default_rng(seed).normal(0, 1, size=400)
        low, high = bootstrap_sharpe_ci(noise, small_cfg(), 1)
        noise_covers.append(low < 0 < high)
    assert sum(noise_covers) >= 4


def test_blocks_widen_the_interval_when_days_are_serially_dependent():
    from signal_lab.metrics import bootstrap_sharpe_ci

    rng = np.random.default_rng(4)
    shocks = rng.normal(size=1500)
    daily = np.zeros(1500)
    for i in range(1, 1500):
        daily[i] = 0.8 * daily[i - 1] + shocks[i]  # strong positive autocorrelation
    wide = bootstrap_sharpe_ci(daily, small_cfg(block_days=10, paths=1000), seed=1)
    plain = bootstrap_sharpe_ci(daily, small_cfg(block_days=1, paths=1000), seed=1)
    assert (wide[1] - wide[0]) > (plain[1] - plain[0])


def test_paired_difference_of_identical_series_is_exactly_zero():
    from signal_lab.metrics import bootstrap_sharpe_difference_ci

    daily = np.random.default_rng(6).normal(0.2, 1.0, size=200)
    assert bootstrap_sharpe_difference_ci(daily, daily, small_cfg(), seed=1) == [0.0, 0.0]

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

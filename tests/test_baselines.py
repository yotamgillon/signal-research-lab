"""Baselines on toy series with known answers. Synthetic fixtures, never used for results."""
import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest import add_roll_sides, net_pnl_usd, simulate
from signal_lab.baselines import buy_and_hold_passive, momentum_signal
from signal_lab.data import add_returns
from signal_lab.trials import TrialLog

MULTIPLIER = 20.0


def toy_window(opens, closes, expiry=None):
    index = pd.date_range("2025-01-06 09:00", periods=len(opens), freq="h", tz="America/New_York")
    return pd.DataFrame({"open": opens, "close": closes,
                         "expiry_bar": expiry or [False] * len(opens)}, index=index)


def test_passive_buy_and_hold_equals_the_known_price_change():
    window = toy_window([100, 101, 102, 103], [101, 102, 103, 105])
    sim = buy_and_hold_passive(window, MULTIPLIER)
    assert sim["gross_pnl_usd"].sum() == (105 - 100) * MULTIPLIER
    assert sim["contracts_traded"].sum() == 1  # one entry, nothing else


def test_passive_buy_and_hold_drops_the_expiry_bar_move_and_pays_a_roll():
    window = toy_window([100, 101, 102, 103], [101, 102, 103, 105],
                        expiry=[False, False, True, False])
    sim = add_roll_sides(buy_and_hold_passive(window, MULTIPLIER), window["expiry_bar"], 2)
    # the 102 to 103 close-to-close move at the expiry bar is the roll artifact: not earned
    assert sim["gross_pnl_usd"].sum() == (5 - 1) * MULTIPLIER
    assert sim["roll_sides"].sum() == 2
    assert net_pnl_usd(sim, 10.0).sum() == 80 - (1 + 2) * 10.0


def test_tradable_buy_and_hold_earns_each_tradable_bar_from_its_open():
    window = toy_window([100, 101, 102, 103], [101, 102, 103, 105])
    valid = np.array([True, True, False, True])
    sim = simulate(window, np.ones(4), valid, 1, MULTIPLIER)
    # decisions at bars 0, 1 trade bars 1, 2. Decision 2 is invalid, decision 3 has no exit.
    assert sim["gross_pnl_usd"].sum() == ((102 - 101) + (103 - 102)) * MULTIPLIER
    assert sim["contracts_traded"].sum() == 1


def test_momentum_signal_hand_computed():
    bars = pd.DataFrame({"ret": [np.nan, 0.01, -0.02, 0.01, 0.03]})
    signal = momentum_signal(bars, 2)
    assert np.isnan(signal[:2]).all()
    assert signal[2:].tolist() == [-1.0, -1.0, 1.0]


def test_momentum_exactly_zero_goes_short():
    bars = pd.DataFrame({"ret": [np.nan, 0.01, -0.01]})
    assert momentum_signal(bars, 2)[2] == -1.0


def test_momentum_has_no_signal_when_the_window_holds_an_expiry_return():
    bars = pd.DataFrame({"ret": [np.nan, 0.01, 0.01, np.nan, 0.01, 0.01, 0.01]})
    signal = momentum_signal(bars, 2)
    assert signal[2] == 1.0
    assert np.isnan(signal[3]) and np.isnan(signal[4])
    assert signal[5] == 1.0


def test_momentum_signal_ignores_the_future():
    rng = np.random.default_rng(5)
    close = 1000 + np.cumsum(rng.normal(size=300))
    index = pd.date_range("2025-01-06 09:00", periods=300, freq="h", tz="America/New_York")
    full = add_returns(pd.DataFrame({"close": close}, index=index),
                       {"expiry_months": [3], "expiry_weekday": 4, "expiry_bar_hour_ny": 16})
    for cut in (50, 120, 299):
        truncated = momentum_signal(full.iloc[:cut], 24)
        assert np.array_equal(truncated, momentum_signal(full, 24)[:cut], equal_nan=True)


def test_trial_log_rows_and_write(tmp_path):
    log = TrialLog()
    log.add(strategy="a", net_sharpe=0.1)
    log.add(strategy="b", net_sharpe=-0.2)
    path = tmp_path / "trials.csv"
    log.write(path)
    assert len(log) == 2
    table = pd.read_csv(path)
    assert table["strategy"].tolist() == ["a", "b"]
    assert table["trial_id"].tolist() == [0, 1]
    assert table["net_sharpe"].tolist() == pytest.approx([0.1, -0.2])

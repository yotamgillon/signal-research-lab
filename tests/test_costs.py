"""Costs: zero trades cost zero, hand-computed examples match, net never exceeds gross.

All fixtures are synthetic and never used for a reported number.
"""
import numpy as np
import pandas as pd
import pytest

from signal_lab.backtest import net_pnl_usd, simulate
from signal_lab.config import load_config
from signal_lab.costs import cost_per_side_usd, trade_cost_usd
from signal_lab.metrics import break_even_multiplier, cost_summary

CFG = load_config()
COSTS = {"commission_per_side_usd": 2.5, "slippage_ticks": 1.0, "half_spread_ticks": 0.5}
INSTRUMENT = {"tick_value_usd": 5.0}
MULTIPLIER = 20.0


def toy_bars(opens, closes):
    index = pd.date_range("2025-01-06 09:00", periods=len(opens), freq="h", tz="America/New_York")
    return pd.DataFrame({"open": opens, "close": closes}, index=index)


def test_cost_per_side_formula():
    # 2.5 + (1.0 + 0.5) x 5 = 10.0
    assert cost_per_side_usd(COSTS, INSTRUMENT) == 10.0
    assert cost_per_side_usd(COSTS, INSTRUMENT, commission_usd=0.0) == 7.5


def test_config_cost_levels():
    main = cost_per_side_usd(CFG["costs"], CFG["instrument"])
    conservative = cost_per_side_usd(
        CFG["costs"], CFG["instrument"], CFG["costs"]["conservative_commission_per_side_usd"]
    )
    assert (main, conservative) == (7.5, 10.0)


def test_zero_trades_cost_nothing():
    assert trade_cost_usd(np.zeros(10), 10.0).sum() == 0
    bars = toy_bars([100.0] * 6, [101.0] * 6)
    sim = simulate(bars, np.zeros(6), np.ones(6, bool), 1, MULTIPLIER)
    assert sim["contracts_traded"].sum() == 0
    assert net_pnl_usd(sim, 10.0).equals(sim["gross_pnl_usd"])


def test_three_trade_example_matches_hand_computation():
    # Signals at the close of bars 0..3: long, short, flat, flat.
    # Trades: enter long (1 side), flip to short (2 sides), exit (1 side) = 4 sides.
    bars = toy_bars(opens=[100, 101, 102, 103, 104], closes=[101, 103, 101, 103, 105])
    signal = np.array([1, -1, 0, 0, np.nan])
    sim = simulate(bars, signal, np.ones(5, bool), 1, MULTIPLIER)
    assert sim["contracts_traded"].tolist() == [0, 1, 2, 1, 0]
    # long earns close[1] - open[1] = 2 points, short earns -(close[2] - open[2]) = 1 point
    assert sim["gross_pnl_usd"].sum() == (2 + 1) * MULTIPLIER == 60
    cost = trade_cost_usd(sim["contracts_traded"], cost_per_side_usd(COSTS, INSTRUMENT))
    assert cost.sum() == 4 * 10.0
    assert net_pnl_usd(sim, 10.0).sum() == 60 - 40


def test_invalid_decision_is_not_traded_and_position_is_carried():
    bars = toy_bars(opens=[100, 101, 102, 103, 104], closes=[101, 103, 101, 103, 105])
    signal = np.array([1, 1, -1, 0, 0])
    valid = np.array([True, False, True, True, True])
    sim = simulate(bars, signal, valid, 1, MULTIPLIER)
    # bar 1 decision skipped: no entry, no P&L. The short at bar 2 flips from the carried long.
    assert sim["contracts_traded"].tolist() == [0, 1, 0, 2, 1]
    # only the first trade earns: long, close[1] - open[1] = 2 points. The short enters at 103
    # and exits at 103, so it earns 0.
    assert sim["gross_pnl_usd"].tolist() == [0, 2 * 20, 0, 0, 0]


def test_net_never_above_gross_when_costs_are_non_negative():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(size=500))
    bars = toy_bars(np.roll(close, 1), close)
    signal = rng.choice([-1.0, 0.0, 1.0], size=500)
    for horizon in (1, 2):
        sim = simulate(bars, signal, np.ones(500, bool), horizon, MULTIPLIER)
        for per_side in (0.0, 7.5, 10.0, 30.0):
            assert (net_pnl_usd(sim, per_side) <= sim["gross_pnl_usd"] + 1e-12).all()


def test_break_even_multiplier_is_where_mean_net_is_zero():
    gross = np.array([3.0, 1.0, 2.0])
    cost = np.array([1.0, 1.0, 1.0])
    multiplier = break_even_multiplier(gross, cost)
    assert multiplier == 2.0
    assert (gross - multiplier * cost).mean() == pytest.approx(0)
    assert break_even_multiplier(-gross, cost) == 0.0  # no positive gross edge to erode


def test_cost_summary_zero_multiplier_equals_gross_and_costs_lower_net():
    rng = np.random.default_rng(3)
    n = 1200
    close = 100 + np.cumsum(rng.normal(size=n))
    bars = toy_bars(np.roll(close, 1), close)
    signal = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    sim = simulate(bars, signal, np.ones(n, bool), 1, MULTIPLIER)
    summary = cost_summary(sim, 7.5, bars.index, CFG)
    by = summary["net_sharpe_by_cost_multiplier"]
    assert by["0x"] == pytest.approx(summary["gross_sharpe"])
    assert by["1x"] == pytest.approx(summary["net_sharpe"])
    assert by["3x"] < by["2x"] < by["1x"] < by["0x"]
    assert summary["sides_traded"] == sim["contracts_traded"].sum()

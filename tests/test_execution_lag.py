"""A signal at bar t cannot earn the return of bar t. Synthetic random-walk bars only."""
import numpy as np
import pandas as pd

from signal_lab.backtest import simulate

N = 4000
MULTIPLIER = 20.0


def random_walk_bars(seed=11):
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(size=N))
    index = pd.date_range("2025-01-06 09:00", periods=N, freq="h", tz="America/New_York")
    return pd.DataFrame({"open": np.roll(close, 1), "close": close}, index=index)


def bar_move(bars):
    return (bars["close"] - bars["open"]).to_numpy()


def test_single_signal_earns_the_next_bar_and_nothing_at_its_own_bar():
    bars = random_walk_bars()
    signal = np.full(N, np.nan)
    signal[100] = 1.0
    sim = simulate(bars, signal, np.ones(N, bool), 1, MULTIPLIER)
    assert sim["gross_pnl_usd"].iloc[100] == 0
    assert sim["gross_pnl_usd"].iloc[101] == bar_move(bars)[101] * MULTIPLIER
    assert (sim["gross_pnl_usd"].drop(sim.index[101]) == 0).all()


def test_two_bar_horizon_enters_next_open_and_exits_at_close_of_t_plus_2():
    bars = random_walk_bars()
    signal = np.full(N, np.nan)
    signal[100] = -1.0
    sim = simulate(bars, signal, np.ones(N, bool), 2, MULTIPLIER)
    expected = -(bars["close"].iloc[102] - bars["open"].iloc[101]) * MULTIPLIER
    assert sim["gross_pnl_usd"].iloc[102] == expected
    assert sim["gross_pnl_usd"].iloc[100] == 0


def test_perfect_foresight_of_the_signal_bar_earns_nothing_when_lagged():
    bars = random_walk_bars()
    signal = np.sign(bar_move(bars))  # knows the direction of bar t, at the close of bar t
    sim = simulate(bars, signal, np.ones(N, bool), 1, MULTIPLIER)
    same_bar_total = np.abs(bar_move(bars)).sum() * MULTIPLIER  # what a same-bar peek would earn
    # A correct lag leaves only luck: a tiny fraction of the same-bar peek.
    assert abs(sim["gross_pnl_usd"].sum()) < 0.05 * same_bar_total


def test_foresight_of_the_next_bar_does_earn_so_the_check_can_fail():
    # Test of the test: a signal that really knows bar t+1 must earn the full amount.
    bars = random_walk_bars()
    next_bar_direction = np.sign(np.roll(bar_move(bars), -1))
    next_bar_direction[-1] = np.nan
    sim = simulate(bars, next_bar_direction, np.ones(N, bool), 1, MULTIPLIER)
    peek_total = np.abs(bar_move(bars)[1:]).sum() * MULTIPLIER
    assert sim["gross_pnl_usd"].sum() > 0.99 * peek_total

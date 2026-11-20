"""Turn signals into positions and gross P&L for one contract, with the SPEC timing convention.

A signal at bar t uses data up to the close of bar t. The trade enters at the open of bar t+1
and exits at the close of bar t+H, so P&L = position x (close[t+H] - open[t+1]) x multiplier.
That is the same quantity as the label y(t), so label and P&L cannot disagree.
"""
import numpy as np
import pandas as pd

from signal_lab.costs import trade_cost_usd


def simulate(bars, signal, valid, horizon, multiplier_usd):
    """Gross P&L and contracts traded per bar, for a window of bars.

    signal: target position (-1, 0, +1) decided at the close of each bar, NaN means no signal.
    valid:  False where the sample is excluded (gap, expiry bar, no future). Such a decision is
            not traded: no P&L, no cost, and the previous position is carried over unchanged.
    Decisions are taken every `horizon` bars starting at the first bar of the window, so trades
    never overlap. The position before the window is flat, so the first trade pays its cost.
    P&L is booked on the exit bar, contracts traded on the entry bar. Roll sides start at zero,
    see add_roll_sides.
    """
    n = len(bars)
    signal = np.asarray(signal, dtype=float)
    decision = np.arange(0, n, horizon)
    decision = decision[decision + horizon <= n - 1]  # the exit bar must be inside the window
    tradable = np.asarray(valid, dtype=bool)[decision] & ~np.isnan(signal[decision])
    decision = decision[tradable]

    target = signal[decision]
    entry, exit_ = decision + 1, decision + horizon
    trade_pnl = target * (bars["close"].to_numpy()[exit_] - bars["open"].to_numpy()[entry])
    previous = np.concatenate([[0.0], target[:-1]])

    out = pd.DataFrame(0.0, index=bars.index, columns=["gross_pnl_usd", "contracts_traded"])
    out.iloc[exit_, 0] = trade_pnl * multiplier_usd
    out.iloc[entry, 1] = np.abs(target - previous)
    out["roll_sides"] = 0.0
    out["position"] = carried_position(n, entry, target)
    return out


def carried_position(n, entry, target):
    """Position held at each bar: the last traded target, carried forward until the next trade."""
    position = np.full(n, np.nan)
    position[entry] = target
    return pd.Series(position).ffill().fillna(0.0).to_numpy()


def add_roll_sides(sim, expiry_bar, sides_per_roll):
    """Charge a roll at each expiry bar for the position held then: sides_per_roll x |position|."""
    sim["roll_sides"] = np.abs(sim["position"].to_numpy()) * sides_per_roll * np.asarray(expiry_bar)
    return sim


def net_pnl_usd(sim, per_side_usd):
    """Gross P&L minus the cost of every position change and roll, per bar."""
    return sim["gross_pnl_usd"] - trade_cost_usd(charged_sides(sim), per_side_usd)


def charged_sides(sim):
    """Sides that pay cost: position changes plus roll sides."""
    return sim["contracts_traded"] + sim["roll_sides"]

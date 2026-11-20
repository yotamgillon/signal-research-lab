"""The two baselines every later model is compared to: buy-and-hold and one momentum rule."""
import numpy as np
import pandas as pd

from signal_lab.backtest import add_roll_sides, simulate


def momentum_signal(bars, lookback):
    """Long (+1) when the trailing L-bar log return is above zero, short (-1) otherwise.

    Uses only bars up to and including t. The signal is NaN (no trade) when the window holds a
    missing return (an expiry bar) or when fewer than L returns exist yet. L is fixed in the
    config before any result is seen.
    """
    trailing = bars["ret"].rolling(lookback).sum()  # NaN if any return in the window is NaN
    direction = np.where(trailing > 0, 1.0, -1.0)
    return np.where(trailing.isna(), np.nan, direction)


def buy_and_hold_passive(window, multiplier_usd):
    """Passive reference: long one contract over the whole window, held through every gap.

    Enters at the open of the first bar (one entry), then earns every close-to-close move. The
    expiry bar's move is the roll artifact, so it earns nothing there and a roll cost is charged
    instead (see add_roll_sides).
    """
    close = window["close"].to_numpy()
    move = np.diff(close, prepend=window["open"].iloc[0])
    move[window["expiry_bar"].to_numpy()] = 0.0
    sim = pd.DataFrame(0.0, index=window.index, columns=["gross_pnl_usd", "contracts_traded"])
    sim["gross_pnl_usd"] = move * multiplier_usd
    sim.iloc[0, 1] = 1.0
    sim["roll_sides"] = 0.0
    sim["position"] = 1.0
    return sim


def build_baselines(bars, cfg, window, labels_by_horizon):
    """Simulated baselines on the out-of-sample window: {name: {horizon: sim}}.

    window: (start, stop) bar positions. Every strategy sees the same bars, the same valid
    samples and the same cost model.
    """
    start, stop = window
    part = bars.iloc[start:stop]
    multiplier = cfg["instrument"]["multiplier_usd"]
    sides_per_roll = cfg["costs"]["roll_round_trip_sides"]
    signal = momentum_signal(bars, cfg["baseline"]["momentum_lookback_bars"])[start:stop]
    out = {"momentum": {}, "buy_and_hold_tradable": {}}
    for horizon, labels in labels_by_horizon.items():
        valid = labels["valid"].to_numpy()[start:stop]
        long_only = np.ones(len(part))
        sims = {
            "momentum": simulate(part, signal, valid, horizon, multiplier),
            "buy_and_hold_tradable": simulate(part, long_only, valid, horizon, multiplier),
        }
        for name, sim in sims.items():
            out[name][horizon] = add_roll_sides(sim, part["expiry_bar"].to_numpy(), sides_per_roll)
    passive = buy_and_hold_passive(part, multiplier)
    out["buy_and_hold_passive"] = {
        None: add_roll_sides(passive, part["expiry_bar"].to_numpy(), sides_per_roll)
    }
    return out

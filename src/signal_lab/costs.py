"""Trading costs. Every cost number comes from the config and is an assumption.

Cost in USD of a trade = contracts traded x (commission per side
                          + (slippage ticks + half-spread ticks) x tick value).
A flip from long to short trades 2 contracts, so it pays 2 sides.
"""
import numpy as np


def cost_per_side_usd(costs_cfg, instrument_cfg, commission_usd=None):
    """USD cost of trading one contract one side. commission_usd overrides the config value."""
    if commission_usd is None:
        commission_usd = costs_cfg["commission_per_side_usd"]
    ticks = costs_cfg["slippage_ticks"] + costs_cfg["half_spread_ticks"]
    return commission_usd + ticks * instrument_cfg["tick_value_usd"]


def trade_cost_usd(contracts_traded, per_side_usd):
    """Cost of each position change. Zero contracts traded costs nothing."""
    return np.abs(np.asarray(contracts_traded, dtype=float)) * per_side_usd

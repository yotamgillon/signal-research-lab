"""Sharpe, drawdown and cost metrics. Everything is for ONE contract and measured in USD.

Sharpe ratio: P&L is summed to daily USD P&L first (one value per CME trading day), then
Sharpe = mean / standard deviation x sqrt(252). Intraday bar Sharpe is never scaled directly.
The risk-free rate is zero. It is not a return on capital, because one futures contract has no
fixed capital base here, so the Sharpe is the ratio of mean to volatility of daily USD P&L.
"""
import numpy as np
import pandas as pd

from signal_lab.data import trading_day


def daily_pnl(bar_pnl, index, data_cfg):
    """Sum bar P&L into one value per CME trading day."""
    days = np.asarray(trading_day(index, data_cfg))
    return pd.Series(np.asarray(bar_pnl, dtype=float)).groupby(days).sum()


def sharpe(daily, metrics_cfg):
    """Annualised Sharpe of daily USD P&L. NaN when it cannot be computed."""
    daily = np.asarray(daily, dtype=float)
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return float("nan")
    excess = daily - metrics_cfg["risk_free_rate"] / metrics_cfg["trading_days_per_year"]
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(metrics_cfg["trading_days_per_year"]))


def max_drawdown_usd(daily):
    """Largest peak-to-trough fall of cumulative daily P&L, as a positive USD number."""
    equity = np.concatenate([[0.0], np.cumsum(np.asarray(daily, dtype=float))])
    return float((np.maximum.accumulate(equity) - equity).max())


def break_even_multiplier(gross_daily, cost_daily):
    """Cost multiplier at which mean net P&L is zero. Net P&L is linear in the multiplier.

    Returns 0.0 when gross P&L is not positive (no cost level helps) and NaN with no costs.
    """
    mean_cost = float(np.mean(cost_daily))
    if mean_cost == 0:
        return float("nan")
    return max(0.0, float(np.mean(gross_daily)) / mean_cost)


def cost_summary(sim, per_side_usd, index, cfg):
    """The item 2 report for one strategy at the assumed cost level and at 0x..3x of it."""
    metrics_cfg, data_cfg = cfg["metrics"], cfg["data"]
    gross = daily_pnl(sim["gross_pnl_usd"], index, data_cfg)
    cost = daily_pnl(sim["contracts_traded"] * per_side_usd, index, data_cfg)
    days = len(gross)
    sides = float(sim["contracts_traded"].sum())
    net = gross - cost
    by_multiplier = {
        f"{m}x": sharpe(gross - m * cost, metrics_cfg) for m in cfg["costs"]["cost_multipliers"]
    }
    return {
        "trading_days": days,
        "per_side_cost_usd": per_side_usd,
        "gross_sharpe": sharpe(gross, metrics_cfg),
        "net_sharpe": sharpe(net, metrics_cfg),
        "sides_traded": sides,
        "trades_per_day": sides / days,
        "sides_per_year": sides / days * metrics_cfg["trading_days_per_year"],
        "cost_drag_sharpe": sharpe(gross, metrics_cfg) - sharpe(net, metrics_cfg),
        "cost_drag_usd_per_year": float(cost.mean() * metrics_cfg["trading_days_per_year"]),
        "break_even_multiplier": break_even_multiplier(gross, cost),
        "break_even_cost_per_side_usd": break_even_multiplier(gross, cost) * per_side_usd,
        "net_sharpe_by_cost_multiplier": by_multiplier,
        "max_drawdown_net_usd": max_drawdown_usd(net),
    }

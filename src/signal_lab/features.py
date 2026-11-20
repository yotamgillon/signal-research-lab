"""Trailing-window features. Every value at bar t uses only bars up to and including t.

Features are built from returns, not price levels, so roll jumps in the glued series cannot
leak in. Used only by the leakage tests in Phase 1: no feature-based result is reported.
The volume z-score from SPEC is dropped for Phase 1 (see deviations_from_spec in the config).
"""
import numpy as np
import pandas as pd

from signal_lab.labels import feature_window_bars, window_has


def feature_columns(features_cfg):
    lags = [f"ret_{k}" for k in features_cfg["return_lags"]]
    return lags + ["vol", "ma_distance", "range_ratio", "hour_ny", "day_of_week"]


def build_features(bars, features_cfg):
    """One row per bar. Rows without a clean trailing window are all NaN (excluded)."""
    ret = bars["ret"]
    out = pd.DataFrame(index=bars.index)
    for lag in features_cfg["return_lags"]:
        out[f"ret_{lag}"] = ret.rolling(lag).sum()  # NaN if any return in the window is missing
    vol = ret.rolling(features_cfg["vol_window"]).std()
    out["vol"] = vol

    # Log price level rebuilt from returns (expiry returns count as zero), so the roll jump of
    # the glued series is not in it. Distance from its moving average, in units of volatility.
    level = ret.fillna(0.0).cumsum()
    moving_average = level.rolling(features_cfg["ma_window"]).mean()
    out["ma_distance"] = (level - moving_average) / vol.replace(0.0, np.nan)

    bar_range = bars["high"] - bars["low"]
    out["range_ratio"] = bar_range / bar_range.rolling(features_cfg["range_window"]).mean()
    out["hour_ny"] = bars.index.hour + bars.index.minute / 60
    out["day_of_week"] = bars.index.day_of_week

    return out.where(clean_window(bars, features_cfg))


def clean_window(bars, features_cfg):
    """True where the trailing feature window is complete and holds no expiry (missing) return."""
    window = feature_window_bars(features_cfg)
    complete = np.arange(len(bars)) >= window - 1
    has_expiry = window_has(bars["expiry_bar"].to_numpy(), -(window - 1), 0)
    return pd.Series(complete & ~has_expiry, index=bars.index)

"""SYNTHETIC bar generator for tests and `make smoke`. Never used for a reported number.

The real-data run (`make run`) must never import this module; a test checks that.
"""
import numpy as np
import pandas as pd


def make_raw_bars(n_hours=1000, seed=0, start="2024-11-25 00:00"):
    """Random-walk hourly bars with weekend gaps. The default range holds the 2024-12-20 expiry."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_hours, freq="h", tz="America/New_York")
    index = index[index.day_of_week < 5]  # no weekend bars, so there are real gaps
    n = len(index)
    close = 1000 + np.cumsum(rng.normal(size=n))
    open_ = np.concatenate([[1000.0], close[:-1]]) + rng.normal(0, 0.1, size=n)
    spread = np.abs(rng.normal(0, 0.5, size=n)) + 0.1
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close,
         "volume": rng.integers(0, 100, size=n)},
        index=index,
    )


def write_yahoo_csv(raw, path):
    """Write bars in the same csv layout as the Yahoo download, so the real loader is exercised."""
    table = raw.rename(columns=str.capitalize)
    table.insert(0, "Adj Close", raw["close"])
    table = table[["Adj Close", "Close", "High", "Low", "Open", "Volume"]]
    table.index = raw.index.map(lambda stamp: stamp.isoformat(sep=" "))
    table.to_csv(path, index_label="Datetime")

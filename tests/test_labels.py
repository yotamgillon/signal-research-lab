"""Label formula, gap rule and expiry rule on small synthetic bars (never used for results)."""
import numpy as np
import pandas as pd
import pytest

from signal_lab.labels import build_labels, window_has


def make_bars(stamps, opens, closes, expiry=None):
    index = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize("America/New_York")
    return pd.DataFrame(
        {"open": opens, "close": closes, "volume": 1,
         "expiry_bar": expiry if expiry is not None else [False] * len(index)},
        index=index,
    )


def hourly(n, start="2025-01-06 09:00"):
    return [str(t) for t in pd.date_range(start, periods=n, freq="h")]


def test_label_formula_matches_hand_computation():
    bars = make_bars(hourly(4), opens=[10, 20, 30, 40], closes=[11, 22, 33, 44])
    one = build_labels(bars, 1, 60)
    two = build_labels(bars, 2, 60)
    assert one["y"].iloc[0] == pytest.approx(np.log(22 / 20))   # open[1] to close[1]
    assert one["y"].iloc[2] == pytest.approx(np.log(44 / 40))   # open[3] to close[3]
    assert two["y"].iloc[0] == pytest.approx(np.log(33 / 20))   # open[1] to close[2]
    assert two["y"].iloc[1] == pytest.approx(np.log(44 / 30))


def test_label_never_uses_the_signal_bar():
    bars = make_bars(hourly(5), opens=[10, 20, 30, 40, 50], closes=[11, 22, 33, 44, 55])
    before = build_labels(bars, 1, 60)["y"].iloc[2]
    changed = bars.copy()
    changed.iloc[2, changed.columns.get_loc("close")] = 999.0
    changed.iloc[2, changed.columns.get_loc("open")] = 999.0
    assert build_labels(changed, 1, 60)["y"].iloc[2] == before


def test_last_samples_have_no_label_and_label_end_is_t_plus_h():
    bars = make_bars(hourly(5), opens=[1] * 5, closes=[1] * 5)
    labels = build_labels(bars, 2, 60)
    assert labels["no_future"].tolist() == [False, False, False, True, True]
    assert labels["label_end"].iloc[:3].tolist() == [2, 3, 4]
    assert labels["y"].iloc[3:].isna().all()


def test_gap_between_t_and_t_plus_h_removes_sample():
    # A weekend sits between bar 2 and bar 3.
    stamps = hourly(3, "2025-01-10 14:00") + hourly(3, "2025-01-12 18:00")
    bars = make_bars(stamps, opens=[1] * 6, closes=[1] * 6)
    one = build_labels(bars, 1, 60)
    two = build_labels(bars, 2, 60)
    assert one["gap"].tolist() == [False, False, True, False, False, False]
    assert two["gap"].tolist() == [False, True, True, False, False, False]
    assert one["valid"].tolist() == [True, True, False, True, True, False]


def test_expiry_bar_inside_label_window_removes_sample():
    bars = make_bars(hourly(6), opens=[1] * 6, closes=[1] * 6,
                     expiry=[False, False, False, True, False, False])
    one = build_labels(bars, 1, 60)
    two = build_labels(bars, 2, 60)
    # label window of t is bars t+1 .. t+H
    assert one["missing_return"].tolist() == [False, False, True, False, False, False]
    assert two["missing_return"].tolist() == [False, True, True, False, False, False]


def test_window_has():
    flag = [False, True, False, False, False]
    assert window_has(flag, 1, 2).tolist() == [True, False, False, False, False]
    # window t-1 .. t: bar 0 runs off the start, bars 1 and 2 contain the flagged bar 1
    assert window_has(flag, -1, 0).tolist() == [False, True, True, False, False]


def test_excluded_samples_have_no_label_end_so_they_are_never_trained_on():
    stamps = hourly(3, "2025-01-10 14:00") + hourly(3, "2025-01-12 18:00")
    bars = make_bars(stamps, opens=[1] * 6, closes=[1] * 6)
    labels = build_labels(bars, 1, 60)
    assert labels["label_end"].isna().tolist() == [False, False, True, False, False, True]
    assert (labels["label_end"].isna() == ~labels["valid"]).all()

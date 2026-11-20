"""Bad data is rejected, good data passes, DST and expiry rules behave. All fixtures are synthetic."""
import io

import pandas as pd
import pytest

from signal_lab.config import load_config
from signal_lab.data import (
    DataError,
    add_returns,
    classify_gaps,
    expiry_bar_mask,
    find_expiry_bars,
    read_bars,
    sha256_of_file,
    trading_day,
    validate,
    zero_volume_summary,
)

CFG = load_config()
NY = "America/New_York"


def make_bars(stamps, tz=NY):
    index = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize(tz) if tz else pd.DatetimeIndex(
        pd.to_datetime(stamps)
    )
    n = len(index)
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n,
         "volume": [10] * n},
        index=index,
    )


def hourly(start, n):
    return [str(t) for t in pd.date_range(start, periods=n, freq="h")]


def test_good_data_passes():
    bars = make_bars(hourly("2025-01-06 09:00", 5))
    assert validate(bars, 80) is bars


def test_unsorted_rejected():
    bars = make_bars(hourly("2025-01-06 09:00", 5)).iloc[[0, 2, 1, 3, 4]]
    with pytest.raises(DataError, match="not sorted"):
        validate(bars, 80)


def test_duplicates_rejected():
    bars = make_bars(hourly("2025-01-06 09:00", 5)).iloc[[0, 1, 1, 2, 3]]
    with pytest.raises(DataError, match="duplicate"):
        validate(bars, 80)


def test_naive_index_rejected():
    bars = make_bars(hourly("2025-01-06 09:00", 5), tz=None)
    with pytest.raises(DataError, match="naive"):
        validate(bars, 80)


def test_naive_csv_stamps_rejected(tmp_path):
    path = tmp_path / "naive.csv"
    path.write_text("Datetime,Adj Close,Close,High,Low,Open,Volume\n"
                    "2025-01-06 09:00:00,1,1,1,1,1,1\n")
    with pytest.raises(DataError, match="naive"):
        read_bars(path, NY)


def test_high_below_low_rejected():
    bars = make_bars(hourly("2025-01-06 09:00", 5))
    bars.iloc[2, bars.columns.get_loc("high")] = 98.0
    with pytest.raises(DataError, match="high below low"):
        validate(bars, 80)


def test_non_positive_price_rejected():
    bars = make_bars(hourly("2025-01-06 09:00", 5))
    bars.iloc[1, bars.columns.get_loc("low")] = 0.0
    with pytest.raises(DataError, match="non-positive"):
        validate(bars, 80)


def test_huge_gap_rejected():
    stamps = hourly("2025-01-06 09:00", 2) + hourly("2025-01-20 09:00", 2)
    with pytest.raises(DataError, match="gap longer"):
        validate(make_bars(stamps), 80)


def test_dst_spring_forward_keeps_new_york_wall_clock(tmp_path):
    # Sunday 2025-03-02 18:00 is EST (-05:00). Sunday 2025-03-09 18:00 is EDT (-04:00).
    path = tmp_path / "dst.csv"
    path.write_text(
        "Datetime,Adj Close,Close,High,Low,Open,Volume\n"
        "2025-03-02 18:00:00-05:00,1,1,2,1,1,1\n"
        "2025-03-09 18:00:00-04:00,1,1,2,1,1,1\n"
    )
    bars = read_bars(path, NY)
    assert list(bars.index.hour) == [18, 18]
    # UTC distance is 7 days minus 1 hour because the clocks moved.
    assert bars.index[1] - bars.index[0] == pd.Timedelta(days=7) - pd.Timedelta(hours=1)


def test_dst_weekend_gap_is_a_normal_weekend():
    # Friday 16:00 EST to Sunday 18:00 EDT is 49 hours, still a weekend gap.
    stamps = ["2025-03-07 16:00", "2025-03-09 18:00"]
    gaps = classify_gaps(make_bars(stamps).index, 60)
    assert gaps["kind"].tolist() == ["weekend"]
    assert gaps["hours"].tolist() == [49.0]


def test_trading_day_follows_chicago_session_across_dst():
    cfg = CFG["data"]
    # 18:00 NY is 17:00 Chicago, the start of the next trading day, in winter and in summer.
    index = make_bars(["2025-03-02 18:00", "2025-03-09 18:00", "2025-03-10 16:00"]).index
    days = trading_day(index, cfg)
    assert [str(d.date()) for d in days] == ["2025-03-03", "2025-03-10", "2025-03-10"]


def test_expiry_bar_is_third_friday_16_ny_in_quarter_months():
    stamps = [
        "2024-12-20 15:00", "2024-12-20 16:00",  # third Friday of December, 16:00 flagged
        "2024-12-13 16:00",                      # second Friday, not flagged
        "2024-11-15 16:00",                      # third Friday of November, not an expiry month
        "2025-03-21 16:00",                      # flagged
    ]
    mask = expiry_bar_mask(make_bars(stamps).index, CFG["cleaning"])
    assert mask.tolist() == [False, True, False, False, True]


def test_expiry_bar_return_is_missing_whatever_its_size():
    stamps = ["2024-12-20 15:00", "2024-12-20 16:00", "2024-12-20 17:00"]
    bars = make_bars(stamps)
    bars["close"] = [100.0, 130.0, 131.0]
    out = add_returns(bars, CFG["cleaning"])
    assert pd.isna(out["ret"].iloc[1])
    assert out["ret"].iloc[2] == pytest.approx(0.0076628, abs=1e-6)


def kind_of(prev_bar, next_bar):
    return classify_gaps(make_bars([prev_bar, next_bar]).index, 60)["kind"].tolist()


def test_gap_kinds():
    assert kind_of("2025-01-07 16:00", "2025-01-07 18:00") == ["daily_break"]
    assert kind_of("2025-01-10 16:00", "2025-01-12 18:00") == ["weekend"]
    assert kind_of("2025-12-24 13:00", "2025-12-25 18:00") == ["holiday"]  # Christmas
    assert kind_of("2025-01-15 16:00", "2025-01-16 18:00") == ["other"]  # a skipped day
    assert kind_of("2025-01-05 18:00", "2025-01-06 00:00") == ["other"]  # missing Sunday bars


def test_zero_volume_summary_counts():
    stamps = ["2025-01-06 03:00", "2025-01-06 10:00", "2025-01-06 11:00", "2025-01-06 19:00"]
    bars = make_bars(stamps)
    bars["volume"] = [0, 0, 5, 0]
    bars.loc[bars.index[0], ["open", "high", "low", "close"]] = 100.0
    summary = zero_volume_summary(bars, CFG["data"])
    assert summary["zero_volume_bars"] == 3
    assert summary["zero_volume_and_open_eq_high_eq_low_eq_close"] == 1
    assert summary["zero_volume_in_overnight_session"] == 2


def test_sha256_matches_known_value(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"abc")
    assert sha256_of_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_read_bars_ignores_adj_close_and_orders_columns():
    text = ("Datetime,Adj Close,Close,High,Low,Open,Volume\n"
            "2025-01-06 09:00:00-05:00,7,1,3,0.5,2,9\n")
    bars = read_bars(io.StringIO(text), NY)
    assert bars.iloc[0].to_dict() == {"open": 2, "high": 3, "low": 0.5, "close": 1, "volume": 9}


def test_expiry_holiday_fallback_uses_previous_trading_day():
    # Third Friday of June 2026 is Juneteenth: no bars that day, so Thursday 16:00 is used.
    stamps = ["2026-06-18 15:00", "2026-06-18 16:00", "2026-06-22 16:00"]
    index = make_bars(stamps).index
    found = find_expiry_bars(index, CFG["cleaning"])
    assert found["used_fallback"].tolist() == [True]
    assert str(found["bar"].iloc[0]) == "2026-06-18 16:00:00-04:00"
    assert expiry_bar_mask(index, CFG["cleaning"]).tolist() == [False, True, False]


def test_ordering_check_flags_unsorted_and_duplicate_timestamps():
    from signal_lab.leakage import ordering_violations

    good = make_bars(hourly("2025-01-06 09:00", 5))
    assert ordering_violations(good.index) == []
    assert ordering_violations(good.iloc[[0, 2, 1, 3, 4]].index) == ["timestamps are not sorted"]
    assert ordering_violations(good.iloc[[0, 1, 1, 2, 3]].index) == ["duplicate timestamps"]

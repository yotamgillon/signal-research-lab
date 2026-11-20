"""Load, validate and describe the market data. Fails loudly on bad input."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar

PRICE_COLUMNS = ["open", "high", "low", "close"]
OFFSET_PATTERN = re.compile(r"[+-]\d{2}:\d{2}$")


class DataError(ValueError):
    """Raised when the market data breaks a rule the research depends on."""


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def read_bars(path, timezone):
    """Read the Yahoo csv. Adj Close is ignored (it equals Close in this file)."""
    raw = pd.read_csv(path)
    stamps = raw["Datetime"].astype(str)
    # to_datetime(utc=True) would silently treat naive stamps as UTC, so check first.
    if not stamps.str.contains(OFFSET_PATTERN).all():
        raise DataError("naive timestamps found, every timestamp needs a UTC offset")
    index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_convert(timezone)
    bars = raw.rename(columns=str.lower)[PRICE_COLUMNS + ["volume"]]
    bars.index = index.rename("timestamp")
    return bars


def validate(bars, max_gap_hours):
    """Raise DataError on any rule break. Returns the bars unchanged if all pass."""
    if bars.index.tz is None:
        raise DataError("naive timestamps: the index must be timezone aware")
    if bars.index.has_duplicates:
        raise DataError("duplicate timestamps")
    if not bars.index.is_monotonic_increasing:
        raise DataError("timestamps are not sorted")
    if bars[PRICE_COLUMNS + ["volume"]].isna().any().any():
        raise DataError("missing values in the price or volume columns")
    if (bars["high"] < bars["low"]).any():
        raise DataError("high below low")
    if (bars[PRICE_COLUMNS] <= 0).any().any():
        raise DataError("non-positive prices")
    gaps = bars.index.to_series().diff().dropna()
    if (gaps > pd.Timedelta(hours=max_gap_hours)).any():
        raise DataError(f"gap longer than {max_gap_hours} hours: {gaps.max()}")
    return bars


def third_friday_dates(first_day, last_day, cleaning):
    """Calendar third Fridays of the expiry months between two naive dates (day 15 to 21)."""
    days = pd.date_range(first_day, last_day, freq="D")
    is_expiry_day = (
        days.month.isin(cleaning["expiry_months"])
        & (days.day_of_week == cleaning["expiry_weekday"])
        & (days.day >= 15)
        & (days.day <= 21)
    )
    return days[is_expiry_day]


def find_expiry_bars(index, cleaning):
    """One row per third Friday inside the data: the bar to treat as missing.

    Normal case: the 16:00 NY bar of that Friday. If the Friday has no bars (a holiday), the
    16:00 bar of the previous trading day is used. That fallback is an assumption, declared on
    2026-11-20 before any performance result, and not a verified CME fact.
    """
    local_dates = index.normalize().tz_localize(None)
    at_hour = index.hour == cleaning["expiry_bar_hour_ny"]
    rows = []
    for friday in third_friday_dates(local_dates[0], local_dates[-1], cleaning):
        same_day = index[(local_dates == friday) & at_hour]
        if len(same_day) > 0:
            rows.append({"third_friday": friday, "bar": same_day[0], "used_fallback": False})
            continue
        earlier = index[(local_dates < friday) & at_hour]
        rows.append({"third_friday": friday, "bar": earlier[-1], "used_fallback": True})
    return pd.DataFrame(rows, columns=["third_friday", "bar", "used_fallback"])


def expiry_bar_mask(index, cleaning):
    """True for the expiry bars found by find_expiry_bars."""
    return np.asarray(index.isin(find_expiry_bars(index, cleaning)["bar"]))


def add_returns(bars, cleaning):
    """Log return of each bar versus the previous close, expiry bars set to missing."""
    out = bars.copy()
    out["ret"] = np.log(out["close"]).diff()
    out["expiry_bar"] = expiry_bar_mask(out.index, cleaning)
    out.loc[out["expiry_bar"], "ret"] = np.nan
    return out


def load_bars(cfg, path=None):
    """Read, validate, drop the still-forming latest bar, add returns."""
    data_cfg = cfg["data"]
    bars = read_bars(path or data_cfg["path"], data_cfg["timezone"])
    validate(bars, data_cfg["max_expected_gap_hours"])
    if data_cfg["drop_latest_bar"]:
        bars = bars.iloc[:-1]
    return add_returns(bars, cfg["cleaning"])


def trading_day(index, data_cfg):
    """CME trading day of each bar: a day that starts at 17:00 Chicago belongs to the next date."""
    chicago = index.tz_convert(data_cfg["session_timezone"])
    shift = 24 - data_cfg["session_start_hour_chicago"]
    return (chicago + pd.Timedelta(hours=shift)).normalize().tz_localize(None)


def holiday_dates(index):
    """US federal holidays plus Good Friday. An approximation of CME closures, stated as such."""
    years = range(index.min().year, index.max().year + 1)
    calendar = USFederalHolidayCalendar().holidays(f"{index.min().year}-01-01", f"{index.max().year}-12-31")
    good_fridays = [pd.Timestamp(easter(y)) - pd.Timedelta(days=2) for y in years]
    return set(calendar) | set(good_fridays)


def gap_kind(prev_bar, next_bar, holidays):
    """Label one gap: daily_break, weekend, holiday or other."""
    hours = (next_bar - prev_bar) / pd.Timedelta(hours=1)
    if prev_bar.weekday() < 4 and prev_bar.hour == 16 and next_bar.hour == 18 and hours == 2:
        return "daily_break"
    first, last = prev_bar.normalize().tz_localize(None), next_bar.normalize().tz_localize(None)
    touches_holiday = any(d in holidays for d in pd.date_range(first, last))
    if touches_holiday:
        return "holiday"
    if prev_bar.weekday() == 4 and prev_bar.hour == 16 and next_bar.weekday() == 6:
        return "weekend"
    return "other"


def classify_gaps(index, bar_minutes):
    """One row per gap longer than one bar, with its kind. The data report lists them."""
    holidays = holiday_dates(index)
    prev_bars, next_bars = index[:-1], index[1:]
    long_gap = (next_bars - prev_bars) > pd.Timedelta(minutes=bar_minutes)
    rows = [
        {"prev_bar": p, "next_bar": n, "hours": (n - p) / pd.Timedelta(hours=1),
         "kind": gap_kind(p, n, holidays)}
        for p, n in zip(prev_bars[long_gap], next_bars[long_gap])
    ]
    return pd.DataFrame(rows, columns=["prev_bar", "next_bar", "hours", "kind"])


def is_regular_hours(index, data_cfg):
    hour = index.hour
    return (hour >= data_cfg["regular_hours_first_start_hour"]) & (
        hour <= data_cfg["regular_hours_last_start_hour"]
    )


def zero_volume_summary(bars, data_cfg):
    zero = bars["volume"] == 0
    flat = (bars[PRICE_COLUMNS].nunique(axis=1) == 1)
    overnight = ~is_regular_hours(bars.index, data_cfg)
    return {
        "zero_volume_bars": int(zero.sum()),
        "zero_volume_and_open_eq_high_eq_low_eq_close": int((zero & flat).sum()),
        "zero_volume_in_overnight_session": int((zero & overnight).sum()),
        "zero_volume_by_start_hour_ny": {
            int(h): int(c) for h, c in bars.index[zero].hour.value_counts().sort_index().items()
        },
        "overnight_definition": (
            "bar start hour (New York) outside "
            f"{data_cfg['regular_hours_first_start_hour']} to "
            f"{data_cfg['regular_hours_last_start_hour']} inclusive"
        ),
    }


def largest_moves(bars, count):
    moves = bars["ret"].abs().nlargest(count).index
    return [{"timestamp": str(t), "log_return_pct": round(100 * float(bars.loc[t, "ret"]), 3)}
            for t in moves]


def expiry_bars_summary(raw_bars, bars):
    """The expiry bars with the return they would have had, so the rule is visible."""
    flagged = bars.index[bars["expiry_bar"]]
    raw_ret = np.log(raw_bars["close"]).diff()
    return [{"timestamp": str(t), "log_return_pct": round(100 * float(raw_ret[t]), 3)}
            for t in flagged]


def expiry_rule_note(index, cleaning):
    """State the holiday fallback and which expiries used it, as an assumption."""
    found = find_expiry_bars(index, cleaning)
    used = found[found["used_fallback"]]
    return {
        "third_fridays_in_data": len(found),
        "fallback_used_for": [
            {"third_friday": str(r.third_friday.date()), "bar_used": str(r.bar)}
            for r in used.itertuples()
        ],
        "note": (
            "If a third Friday has no bars (holiday), the 16:00 New York bar of the previous "
            "trading day is treated as the expiry bar. This is an assumption, not a verified "
            "CME fact. It was declared on " + cleaning["fallback_declared_on"]
            + ", before any performance result was computed."
        ),
    }


def build_data_report(path, cfg, raw_bars, bars):
    """Bar-level data report. Sample-level counts are added by the labels module (item 1)."""
    data_cfg = cfg["data"]
    gaps = classify_gaps(bars.index, data_cfg["bar_minutes"])
    other = gaps[gaps["kind"] == "other"]
    return {
        "deviations_from_spec": [" ".join(text.split()) for text in cfg["deviations_from_spec"]],
        "file": str(path),
        "sha256": sha256_of_file(path),
        "rows_in_file": len(raw_bars),
        "latest_bar_dropped": str(raw_bars.index[-1]) if data_cfg["drop_latest_bar"] else None,
        "rows_used": len(bars),
        "first_bar": str(bars.index[0]),
        "last_bar": str(bars.index[-1]),
        "gap_counts_by_kind": {k: int(v) for k, v in gaps["kind"].value_counts().items()},
        "longest_gap_hours": float(gaps["hours"].max()),
        "gaps_kind_other": [
            {"prev_bar": str(r.prev_bar), "next_bar": str(r.next_bar), "hours": r.hours}
            for r in other.itertuples()
        ],
        "bars_not_starting_on_the_hour": int((bars.index.minute != 0).sum()),
        "gap_kind_note": (
            "holiday = US federal holidays plus Good Friday, an approximation of CME closures"
        ),
        "largest_single_bar_moves": largest_moves(bars, data_cfg["top_moves_to_report"]),
        "expiry_bars_set_to_missing": expiry_bars_summary(raw_bars, bars),
        "expiry_rule": expiry_rule_note(bars.index, cfg["cleaning"]),
        "regular_hours_definition": (
            f"bar start hour {data_cfg['regular_hours_first_start_hour']} to "
            f"{data_cfg['regular_hours_last_start_hour']} inclusive, New York "
            "(the 09:00 bar contains the 09:30 open)"
        ),
        **zero_volume_summary(bars, data_cfg),
    }


def write_report(report, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def run_data_step(cfg, out_path="results/data_report.json"):
    """Load, validate, report. Returns the cleaned bars."""
    path = cfg["data"]["path"]
    raw_bars = read_bars(path, cfg["data"]["timezone"])
    bars = load_bars(cfg, path)
    write_report(build_data_report(path, cfg, add_returns(raw_bars, cfg["cleaning"]), bars),
                 out_path)
    return bars


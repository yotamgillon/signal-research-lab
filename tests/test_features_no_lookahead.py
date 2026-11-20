"""Truncating the data never changes earlier feature values. Synthetic bars only."""
import numpy as np
import pytest

from signal_lab.config import load_config
from signal_lab.data import add_returns
from signal_lab.features import build_features, feature_columns
from signal_lab.leakage import truncation_violations
from signal_lab.synthetic import make_raw_bars

CFG = load_config()
RAW = make_raw_bars()


def pipeline(raw):
    return build_features(add_returns(raw, CFG["cleaning"]), CFG["features"])


def random_cuts(seed=0, count=15):
    return sorted(np.random.default_rng(seed).integers(100, len(RAW) - 1, size=count).tolist())


def cuts_around_the_expiry_bar():
    position = RAW.index.get_loc(RAW.index[(RAW.index.day == 20) & (RAW.index.month == 12)
                                           & (RAW.index.hour == 16)][0])
    return [position - 1, position, position + 1]


def test_clean_features_pass_the_truncation_test():
    cuts = random_cuts() + cuts_around_the_expiry_bar()
    assert truncation_violations(pipeline, RAW, cuts) == []


def leaky(make_column):
    """Builder with one extra, deliberately leaky column."""
    def build(raw):
        bars = add_returns(raw, CFG["cleaning"])
        features = build_features(bars, CFG["features"])
        features["leak"] = make_column(bars)
        return features
    return build


LEAKY_COLUMNS = {
    "centred rolling window": lambda bars: bars["ret"].rolling(5, center=True).mean(),
    "negative shift": lambda bars: bars["close"].shift(-1) - bars["close"],
    "back-fill": lambda bars: bars["ret"].bfill(),
    "full-sample normalisation": lambda bars: (bars["ret"] - bars["ret"].mean()) / bars["ret"].std(),
}


@pytest.mark.parametrize("name", LEAKY_COLUMNS)
def test_truncation_test_catches_each_classic_leak(name):
    cuts = random_cuts() + cuts_around_the_expiry_bar()  # back-fill only shows on missing values
    found = truncation_violations(leaky(LEAKY_COLUMNS[name]), RAW, cuts)
    assert found, f"the truncation test missed: {name}"
    assert all(v["columns"] == ["leak"] for v in found)


def test_feature_values_match_hand_computation():
    bars = add_returns(RAW, CFG["cleaning"])
    features = build_features(bars, CFG["features"])
    row = 300  # after the warm-up, far from the expiry bar
    ret = bars["ret"].to_numpy()
    assert features["ret_1"].iloc[row] == pytest.approx(ret[row])
    assert features["ret_5"].iloc[row] == pytest.approx(ret[row - 4 : row + 1].sum())
    assert features["vol"].iloc[row] == pytest.approx(ret[row - 19 : row + 1].std(ddof=1))
    bar_range = (bars["high"] - bars["low"]).to_numpy()
    assert features["range_ratio"].iloc[row] == pytest.approx(
        bar_range[row] / bar_range[row - 19 : row + 1].mean())
    assert features["hour_ny"].iloc[row] == bars.index[row].hour
    assert features["day_of_week"].iloc[row] == bars.index[row].day_of_week


def test_warm_up_and_expiry_windows_are_excluded():
    bars = add_returns(RAW, CFG["cleaning"])
    features = build_features(bars, CFG["features"])
    assert features.iloc[:49].isna().all().all()
    expiry = int(np.flatnonzero(bars["expiry_bar"].to_numpy())[0])
    window = features.iloc[expiry : expiry + 50]
    assert window.isna().all().all()  # the 50 samples whose window holds the missing return
    assert features.iloc[expiry + 50].notna().all()


def test_all_declared_columns_exist():
    assert list(pipeline(RAW).columns) == feature_columns(CFG["features"])

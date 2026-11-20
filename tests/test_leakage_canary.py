"""A test of the test: a feature built from the future MUST be caught.

If the leakage test passed on the canary, the leakage test would be broken. The canary also
earns an absurd Sharpe in the naive backtest. Synthetic bars only.
Run `pytest --runxfail tests/test_leakage_canary.py` to see the leakage test fail as designed.
"""
import numpy as np
import pytest

from signal_lab.config import load_config
from signal_lab.data import add_returns
from signal_lab.features import build_features
from signal_lab.leakage import canary_feature, truncation_violations
from signal_lab.stats_tests import net_sharpe_statistic
from synthetic import make_raw_bars

CFG = load_config()
RAW = make_raw_bars()
CUTS = [150, 300, 450, 600]  # all inside the synthetic series


def build_with_canary(raw):
    bars = add_returns(raw, CFG["cleaning"])
    features = build_features(bars, CFG["features"])
    # The canary is the future return of the next bar. It cannot exist at the close of bar t.
    features["canary_future_return"] = canary_feature(bars)
    return features


def test_truncation_test_detects_the_canary_and_only_the_canary():
    found = truncation_violations(build_with_canary, RAW, CUTS)
    assert found
    assert all(v["columns"] == ["canary_future_return"] for v in found)


@pytest.mark.xfail(strict=True, reason="the canary leaks on purpose, so this check must fail")
def test_leakage_test_FAILS_AS_DESIGNED_on_the_canary():
    found = truncation_violations(build_with_canary, RAW, CUTS)
    assert not found, "LEAKAGE DETECTED in the canary feature:\n" + "\n".join(map(str, found))


def test_naive_backtest_of_the_canary_has_an_absurd_sharpe():
    features = build_with_canary(RAW)
    bars = add_returns(RAW, CFG["cleaning"])
    statistic = net_sharpe_statistic(bars, np.ones(len(bars), bool), 1, 0.0, CFG)  # zero cost: gross
    canary_sharpe = statistic(np.sign(features["canary_future_return"].to_numpy()))
    honest_sharpe = statistic(np.sign(features["ret_1"].to_numpy()))
    assert canary_sharpe > 15
    assert abs(honest_sharpe) < 5

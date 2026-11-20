"""Report helpers: table insertion, number check, decomposition identity, figures. Synthetic."""
import numpy as np
import pandas as pd

from signal_lab.baselines import build_baselines
from signal_lab.config import load_config
from signal_lab.data import add_returns
from signal_lab.labels import build_labels, step_gap_kinds
from signal_lab.report import (
    TABLE_END,
    TABLE_START,
    drawdown_figure,
    equity_figure,
    insert_results_table,
    numbers_in,
    numbers_missing_from_summary,
    passive_vs_tradable,
    permutation_figure,
    word_count,
)
from synthetic import make_raw_bars

CFG = load_config()


def test_insert_results_table_replaces_only_the_marked_block():
    text = f"before\n{TABLE_START}\nold\n{TABLE_END}\nafter"
    out = insert_results_table(text, ["| a |", "| 1.00 |"])
    assert out == f"before\n{TABLE_START}\n| a |\n| 1.00 |\n{TABLE_END}\nafter"
    assert insert_results_table(out, ["| a |", "| 1.00 |"]) == out  # running twice changes nothing


def test_word_count_ignores_the_table():
    text = f"one two three\n{TABLE_START}\n| a | b | c | d |\n{TABLE_END}\nfour"
    assert word_count(text) == 4


def test_numbers_check_finds_a_retyped_number_and_accepts_formatting_variants():
    summary = "Sharpe -0.67 and 8,735 bars on 2025-05-15 at 10.00 USD"
    assert numbers_missing_from_summary("text -0.67, 8735 bars, 2025-05-15, 10.00", summary) == []
    assert numbers_missing_from_summary("net -0.68 on 2025-05-16", summary) == ["-0.68", "2025-05-16"]


def test_numbers_check_ignores_the_horizon_labels_only():
    assert numbers_in("H=1 and H=2 but 24 bars") == {"24"}


def test_decomposition_adds_up_on_synthetic_bars():
    raw = make_raw_bars(n_hours=1500, seed=3)
    bars = add_returns(raw, CFG["cleaning"])
    labels = {h: build_labels(bars, h, 60) for h in (1, 2)}
    window = (200, len(bars))
    sims = build_baselines(bars, CFG, window, labels)
    gaps = step_gap_kinds(bars.index, 60)
    result = passive_vs_tradable(bars, window, gaps, sims, CFG)
    in_bar_and_between = (result["in_bar_moves_of_non_tradable_bars_usd"]
                          + result["moves_between_bars_usd"])
    assert result["difference_usd"] == np.float64(result["passive_gross_usd"]
                                                  - result["tradable_gross_usd"])
    assert abs(result["difference_usd"] - in_bar_and_between) < 1e-6
    assert result["first_bars_after_a_gap"] > 0


def test_figures_are_written(tmp_path):
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    rng = np.random.default_rng(0)
    series = {name: pd.Series(rng.normal(size=40) * 100, index=index)
              for name in ("momentum", "buy_and_hold_tradable", "buy_and_hold_passive")}
    equity_figure(tmp_path / "e.png", series, 7.5)
    drawdown_figure(tmp_path / "d.png", series, 7.5)
    result = {"H=1": {"real_net_sharpe": -0.5, "p_value": 0.7, "n_permutations": 50}}
    permutation_figure(tmp_path / "p.png", result, {"H1": rng.normal(size=50)}, 7.5)
    assert all((tmp_path / name).stat().st_size > 1000 for name in ("e.png", "d.png", "p.png"))

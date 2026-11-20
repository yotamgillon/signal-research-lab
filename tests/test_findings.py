"""docs/FINDINGS.md: 500 to 600 words, and every number quoted in the prose is in summary.md."""
from pathlib import Path

from signal_lab.report import TABLE_END, TABLE_START, numbers_missing_from_summary, word_count

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = (ROOT / "docs" / "FINDINGS.md").read_text()
SUMMARY = (ROOT / "results" / "summary.md").read_text()
REQUIRED_HEADINGS = ["Question", "Data", "Method", "Results", "Limitations",
                     "Failures and negative results", "What I would do next"]


def test_word_count_is_between_500_and_600():
    assert 500 <= word_count(FINDINGS) <= 600


def test_every_number_in_the_prose_is_in_the_summary():
    assert numbers_missing_from_summary(FINDINGS, SUMMARY) == []


def test_required_headings_and_the_generated_table_are_present():
    for heading in REQUIRED_HEADINGS:
        assert f"## {heading}" in FINDINGS
    assert TABLE_START in FINDINGS and TABLE_END in FINDINGS


def test_negative_result_is_stated_in_the_first_line_of_results():
    results = FINDINGS.split("## Results")[1].strip().splitlines()[0]
    assert results.startswith("No evidence of an edge")


def test_required_statements_are_present():
    text = FINDINGS.lower()
    for phrase in ["unofficial", "assumption", "volume z-score", "april 2025",
                   "2026-06-19", "like-for-like", "nothing broader", "did not do in phase 1"]:
        assert phrase in text, phrase

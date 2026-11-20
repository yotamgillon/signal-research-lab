"""The smoke run is labelled SYNTHETIC everywhere, and the real run never loads the generator."""
import subprocess
import sys
from pathlib import Path

import pytest

from signal_lab.config import load_config, with_overrides
from signal_lab.pipeline import Outputs, run, run_smoke

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


def test_every_smoke_file_name_says_synthetic_and_the_titles_say_not_a_result(tmp_path):
    result = run_smoke(CFG, tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and all(p.name.startswith("SYNTHETIC_") for p in files)
    assert (tmp_path / "SYNTHETIC_summary.md").read_text().startswith("# SYNTHETIC DATA.")
    assert "NOT A RESULT" in (tmp_path / "SYNTHETIC_comparison.md").read_text().splitlines()[0]
    assert "SYNTHETIC" in (tmp_path / "SYNTHETIC_leakage_check.json").read_text()
    assert not (tmp_path / "FINDINGS.md").exists()
    # the smoke run also proves the leakage checks are wired: clean passes, canary fails
    checks = result["checks"]
    assert checks["clean_features_truncation_violations"] == 0
    assert checks["canary_truncation_cuts_with_violation"] == checks["cut_points"]


def test_importing_the_pipeline_does_not_load_the_synthetic_generator():
    code = ("import sys; sys.path.insert(0, 'src'); import signal_lab.pipeline, signal_lab.report; "
            "assert 'signal_lab.synthetic' not in sys.modules")
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, check=False,
                          text=True)
    assert done.returncode == 0, done.stderr


def test_only_the_smoke_function_imports_the_generator():
    for path in list((ROOT / "src" / "signal_lab").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if path.name == "synthetic.py":
            continue
        for line in path.read_text().splitlines():
            if "signal_lab.synthetic" in line and "import" in line:
                assert line.startswith("    "), f"top-level import of the generator in {path.name}"


def test_real_run_without_the_data_file_stops_with_a_clear_message(tmp_path):
    cfg = with_overrides(CFG, {"data": {"path": str(tmp_path / "missing.csv")}})
    with pytest.raises(SystemExit, match="not found"):
        run(cfg, Outputs(tmp_path / "out"))

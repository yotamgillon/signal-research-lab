"""Two runs with the same seed give identical result hashes. Uses the SYNTHETIC smoke run."""
import hashlib

from signal_lab.config import load_config, with_overrides
from signal_lab.pipeline import run_smoke

CFG = load_config()
SKIP = {"SYNTHETIC_run_info.json"}  # it records the temporary folder path and the git state


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in SKIP}


def test_same_seed_gives_identical_hashes_for_every_output_file(tmp_path):
    run_smoke(CFG, tmp_path / "a")
    run_smoke(CFG, tmp_path / "b")
    first, second = hashes(tmp_path / "a"), hashes(tmp_path / "b")
    assert first.keys() == second.keys() and len(first) > 10
    assert first == second


def test_a_different_seed_changes_the_permutation_output(tmp_path):
    run_smoke(CFG, tmp_path / "a")
    run_smoke(with_overrides(CFG, {"seed": 99}), tmp_path / "b")
    name = "SYNTHETIC_permutation_momentum_sharpes.csv"
    assert hashes(tmp_path / "a")[name] != hashes(tmp_path / "b")[name]

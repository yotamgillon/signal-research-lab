"""Single entry point, a thin wrapper over signal_lab.pipeline.

    python scripts/run_all.py           real data (data/NQ_1h.csv), writes results/ and docs/FINDINGS.md
    python scripts/run_all.py --smoke   SYNTHETIC bars, writes smoke_SYNTHETIC/, files start with SYNTHETIC_
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # paths in the config are relative to the repo root

from signal_lab.config import load_config
from signal_lab.pipeline import Outputs, run, run_smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on SYNTHETIC bars")
    args = parser.parse_args()
    cfg = load_config()
    if args.smoke:
        run_smoke(cfg, ROOT / "smoke_SYNTHETIC")
        print("smoke run finished: SYNTHETIC data, outputs in smoke_SYNTHETIC/, not a result")
        return
    run(cfg, Outputs(ROOT / "results"), findings_path=ROOT / "docs" / "FINDINGS.md")
    # The real-data run must never have loaded the synthetic generator.
    assert "signal_lab.synthetic" not in sys.modules, "real run imported the synthetic generator"
    print("run finished: results/ written")


if __name__ == "__main__":
    main()

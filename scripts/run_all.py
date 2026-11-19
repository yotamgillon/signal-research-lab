"""Single entry point. Each item adds its step here. Uses only the local data file."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signal_lab.config import load_config
from signal_lab.data import run_data_step
from signal_lab.labels import add_sample_section, build_labels, sample_report
from signal_lab.splits import check_embargo, fold_table

RESULTS = ROOT / "results"


def main():
    cfg = load_config()
    RESULTS.mkdir(exist_ok=True)
    bars = run_data_step(cfg, RESULTS / "data_report.json")

    add_sample_section(RESULTS / "data_report.json", sample_report(bars, cfg))

    check_embargo(cfg["splits"], cfg["horizon"])
    horizons = [cfg["horizon"]["main_bars"], cfg["horizon"]["secondary_bars"]]
    label_ends = {
        h: build_labels(bars, h, cfg["data"]["bar_minutes"])["label_end"].to_numpy()
        for h in horizons
    }
    folds = fold_table(bars.index, label_ends, cfg["splits"])
    folds.to_csv(RESULTS / "folds.csv", index=False)
    print(folds.to_string())


if __name__ == "__main__":
    main()

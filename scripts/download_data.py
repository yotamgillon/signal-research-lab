"""Documents how data/NQ_1h.csv was obtained. NOT used by `make run`.

The file is a snapshot taken on 2026-11-19. Downloading again gives different data (Yahoo
revises history and adds bars), so every result is tied to the sha256 in results/run_info.json.

The exact arguments of the original download were not recorded. The arguments below are
reconstructed from the columns of the file (Datetime, Adj Close, Close, High, Low, Open,
Volume, which is what yfinance returns with auto_adjust=False) and from the date range.
Treat them as documentation, not as proof.

The script refuses to overwrite an existing file, so the snapshot cannot be lost by accident.
"""
import argparse
import hashlib
from pathlib import Path

import yfinance as yf

TICKER = "NQ=F"
INTERVAL = "1h"
START = "2024-06-30"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default="data/NQ_1h.csv")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists, refusing to overwrite the snapshot")
    bars = yf.download(TICKER, start=START, interval=INTERVAL, auto_adjust=False,
                       multi_level_index=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(out)
    print(f"wrote {len(bars)} rows to {out}")
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

"""Documents how data/NQ_1h.csv was obtained. NOT used by `make run`.

The file is a snapshot taken on 2026-11-19. Downloading again gives different data (Yahoo
revises history and adds bars), so every result is tied to the sha256 in results/run_info.json.

The original call was:
    df = yf.download("NQ=F", period="730d", interval="1h", auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.to_csv("data/NQ_1h.csv")

The script refuses to overwrite an existing file, so the snapshot cannot be lost by accident.
"""
import argparse
import hashlib
from pathlib import Path

import yfinance as yf

TICKER = "NQ=F"
PERIOD = "730d"
INTERVAL = "1h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default="data/NQ_1h.csv")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists, refusing to overwrite the snapshot")
    bars = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=False,
                       progress=False)
    bars.columns = bars.columns.get_level_values(0)  # drop the ticker level
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(out)
    print(f"wrote {len(bars)} rows to {out}")
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

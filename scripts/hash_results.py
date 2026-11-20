"""Print the sha256 of every file under a results folder, one line per file, sorted.

    python scripts/hash_results.py [folder] [--skip name ...]
Used by `make reproduce-check`: two runs must print identical lines.
"""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="results")
    parser.add_argument("--skip", nargs="*", default=[], help="file names to leave out")
    args = parser.parse_args()
    root = Path(args.folder)
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in args.skip):
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")


if __name__ == "__main__":
    main()

"""The project has a no em dash rule. The dash is written as an escape so this file is clean too."""
from pathlib import Path

EM_DASH = chr(0x2014)  # built from the code point so this file has no dash in it
ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "data", "__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".json", ".csv", ""}


def project_files():
    for path in ROOT.rglob("*"):
        parts = set(path.relative_to(ROOT).parts)
        if path.is_file() and not parts & SKIP_DIRS and path.suffix in TEXT_SUFFIXES:
            yield path


def test_no_em_dashes_anywhere():
    offenders = [str(p.relative_to(ROOT)) for p in project_files()
                 if EM_DASH in p.read_text(errors="ignore")]
    assert offenders == []

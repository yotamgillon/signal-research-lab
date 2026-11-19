"""Load the YAML config into a plain dict. No defaults are hidden in code."""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as handle:
        return yaml.safe_load(handle)

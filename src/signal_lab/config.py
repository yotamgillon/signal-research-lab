"""Load the YAML config into a plain dict. No defaults are hidden in code."""
import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path) as handle:
        return yaml.safe_load(handle)


def with_overrides(cfg, overrides):
    """A copy of cfg with nested overrides applied (used by the synthetic smoke run only)."""
    merged = copy.deepcopy(cfg)
    for section, values in overrides.items():
        if isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous top-level values."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    return value


def require_keys(mapping: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

"""YAML loading with strict environment-variable expansion."""

from __future__ import annotations

import os
import re
from typing import Any

import yaml


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML mapping after expanding ``${NAME}`` references.

    Undefined variables fail explicitly instead of leaving a literal path that
    surfaces much later as a missing dataset or output directory.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    referenced = set(_ENV_REFERENCE.findall(raw))
    missing = sorted(name for name in referenced if name not in os.environ)
    if missing:
        raise ValueError(
            f"{path}: undefined environment variable(s): {', '.join(missing)}"
        )
    expanded = _ENV_REFERENCE.sub(lambda match: os.environ[match.group(1)], raw)
    loaded = yaml.safe_load(expanded)
    if not isinstance(loaded, dict):
        raise TypeError(f"{path}: expected a YAML mapping at the document root")
    return loaded

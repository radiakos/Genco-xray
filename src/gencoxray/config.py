"""Config loading and repo-relative paths (IMPLEMENTED)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = "synthetic"


def resolve(path: str | Path) -> Path:
    """Absolute path; relative paths are taken from the repo root (not the cwd)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path = "configs/experiment.yaml", *, as_namespace: bool = True):
    """Load a YAML config. Returns a SimpleNamespace tree (or dict if as_namespace=False)."""
    d = load_yaml(path)
    return _to_ns(d) if as_namespace else d


def load_grid(name: str, *, as_namespace: bool = True):
    """Load ``configs/grids/<name>.yaml``."""
    return load_config(f"configs/grids/{name}.yaml", as_namespace=as_namespace)


def grid_splits(gname: str) -> list[str]:
    """Splits of a grid (the synthetic set has one split, ``test``)."""
    return ["test"] if gname == SYNTHETIC else list(load_grid(gname).splits)


def k_fit(gname: str) -> int:
    """Largest in-distribution contingency order: models are fitted on k <= k_fit (synthetic: 2)."""
    return 2 if gname == SYNTHETIC else int(max(load_grid(gname).in_distribution_k))


def k_of_split(split: str) -> int:
    """Contingency order encoded in a split name: test_7 -> 7, nk/k07 -> 7, test -> 0 (mixed/unknown).
    The runner takes k from the raw outage tables; this is only for labelling."""
    tail = split.replace("/", "_").rsplit("_", 1)[-1]
    tail = tail.removeprefix("k")
    return int(tail) if tail.isdigit() else 0

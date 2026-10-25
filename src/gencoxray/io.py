"""Paths and table I/O (IMPLEMENTED)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from . import schemas
from .config import resolve

TABLES = {
    "bus": schemas.BUS,
    "branch": schemas.BRANCH,
    "scenario": schemas.SCENARIO,
    "static_bus": schemas.STATIC_BUS,
    "static_branch": schemas.STATIC_BRANCH,
}
_PER_SCENARIO = {"scenario", "res_p", "res_q", "vm", "va", "loading"}  # TRACES keys with a leading scenario axis


def results_dir(grid: str, split: str | None = None, root: str | Path = "results") -> Path:
    p = resolve(root) / grid if split is None else resolve(root) / grid / split
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_table(df: pd.DataFrame, table: str, grid: str, split: str | None = None, **kw) -> Path:
    """Validate against the table's schema and write parquet. Static tables have split=None."""
    schemas.validate(df, TABLES[table])
    path = results_dir(grid, split, **kw) / f"{table}.parquet"
    df.to_parquet(path, index=False)
    return path


def read_table(table: str, grid: str, split: str | None = None, *, splits: list[str] | None = None,
               validate: bool = True, **kw) -> pd.DataFrame:
    """Read one split, or concatenate several (``splits=[...]``), adding a ``split`` column."""
    if splits is not None:
        frames = [read_table(table, grid, s, validate=validate, **kw).assign(split=s) for s in splits]
        return pd.concat(frames, ignore_index=True)
    df = pd.read_parquet(results_dir(grid, split, **kw) / f"{table}.parquet")
    if validate:
        schemas.validate(df, TABLES[table])
    return df


def read_columns(table: str, grid: str, split: str, columns: list[str], **kw) -> pd.DataFrame:
    """Selected columns of one split's table, without schema validation (large tables, analysis scripts)."""
    return pd.read_parquet(results_dir(grid, split, **kw) / f"{table}.parquet", columns=columns)


def read_json(name: str, grid: str, split: str | None = None, **kw) -> dict:
    return json.loads((results_dir(grid, split, **kw) / name).read_text())


def traces_dir(grid: str, split: str, **kw) -> Path:
    p = results_dir(grid, split, **kw) / "traces"
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_traces(grid: str, split: str, **kw) -> None:
    """Remove existing TRACES shards (a re-run must not leave stale shards from a longer previous run)."""
    for old in traces_dir(grid, split, **kw).glob("shard_*.npz"):
        old.unlink()


def write_traces(arrs: dict[str, np.ndarray], grid: str, split: str, shard: int, **kw) -> Path:
    """Write one TRACES shard (``traces/shard_00000.npz``). Shards are written in scenario order."""
    schemas.validate_traces(arrs)
    path = traces_dir(grid, split, **kw) / f"shard_{shard:05d}.npz"
    np.savez(path, **arrs)  # uncompressed: float residuals barely compress and savez_compressed is slow
    return path


def iter_traces(grid: str, split: str, **kw) -> Iterator[dict[str, np.ndarray]]:
    """Yield TRACES shards in order. Peak memory = one shard."""
    for path in sorted(traces_dir(grid, split, **kw).glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            yield {k: z[k] for k in z.files}


def slice_traces(arrs: dict[str, np.ndarray], sel) -> dict[str, np.ndarray]:
    """Select scenarios (index, slice or mask on the scenario axis) of a TRACES dict; per-graph keys are kept."""
    return {k: (v[sel] if k in _PER_SCENARIO else v) for k, v in arrs.items()}


def read_traces(grid: str, split: str, scenarios: np.ndarray | None = None, **kw) -> dict[str, np.ndarray]:
    """Concatenate shards (optionally only ``scenarios``). Use for small subsets (viz, debugging) only."""
    parts = []
    for sh in iter_traces(grid, split, **kw):
        if scenarios is not None:
            m = np.isin(sh["scenario"], scenarios)
            if not m.any():
                continue
            sh = slice_traces(sh, m)
        parts.append(sh)
    if not parts:
        raise FileNotFoundError(f"no trace shards for {grid}/{split}")
    return {k: (np.concatenate([p[k] for p in parts]) if k in _PER_SCENARIO else parts[0][k]) for k in parts[0]}


def update_summary(key: str, value: dict, root: str | Path = "results") -> Path:
    """Merge ``value`` into results/summary.json under dotted ``key`` (e.g. 'e2.texas.in_dist')."""
    base = resolve(root)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    node = data
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
    path.write_text(json.dumps(data, indent=2, default=float))
    return path


def read_summary(root: str | Path = "results") -> dict:
    return json.loads((resolve(root) / "summary.json").read_text())

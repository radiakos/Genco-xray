"""Download and load pretrained GENCO checkpoints from Hugging Face.

Mirrors ``gridfm_graphkit.cli.main_cli`` without the Trainer/MLflow parts. Works with both graphkit branches
(verified on ``lfe-tutorial`` @ 9d42d82, see CLAUDE.md).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve

log = logging.getLogger(__name__)


@dataclass
class LoadedGenco:
    """Everything downstream code needs from one checkpoint + one data split."""

    task: Any  # gridfm_graphkit PowerFlowTask; GENCO is task.model
    model: Any
    datamodule: Any  # LitGridHeteroDataModule, set up for "test"
    normalizers: list
    base_mva: float  # FITTED baseMVA from normalizer_stats.pt (95th pct of non-zero powers), not 100
    n_steps: int  # model.num_layers (expected 12)
    graphkit_commit: str
    grid: str
    raw_dir: Path  # <data_root>/<network>/raw — datakit tables for this split

    @property
    def device(self):
        return next(self.task.parameters()).device

    def test_loader(self):
        """The test dataloader (graphkit returns one per network; we load one network)."""
        loader = self.datamodule.test_dataloader()
        return loader[0] if isinstance(loader, (list, tuple)) else loader


RAW_TABLES = ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet", "runtime_data.parquet")


def _partition_patterns(prefix: str, n_partitions: int | None) -> list[str]:
    """allow_patterns for the first ``n_partitions`` Hive partitions of every raw table (all if None)."""
    parts = ["*"] if n_partitions is None else [str(p) for p in range(n_partitions)]
    return [f"{prefix}{t}/scenario_partition={p}/*" for t in RAW_TABLES for p in parts] + [f"{prefix}n_scenarios.txt"]


def download(grid_cfg, local_dir: str | Path = "data", *, n_partitions: int | None = None,
             splits: list[str] | None = None) -> dict[str, Path]:
    """Fetch checkpoint, normaliser stats, model config and raw data listed in ``grid_cfg``. Idempotent.

    Raw data are Hive-partitioned (~200 scenarios per partition, scenario ids contiguous from 0), so
    ``n_partitions`` downloads a prefix ``scenario < 200 * n_partitions`` (default: ``grid_cfg.n_partitions``,
    None = everything). ``splits`` restricts per-split grids (Texas) to those splits (default: first one).

    Returns local paths: ``checkpoint``, ``normalizer_stats``, ``config``, ``data_root`` (a directory whose children
    are split folders or ``<network>/raw``, depending on the grid; see configs/grids/*.yaml) and, if the grid
    publishes the training run's scenario split, ``splits_json``.
    """
    from huggingface_hub import snapshot_download

    f = {k: v for k, v in vars(grid_cfg.hf_files).items() if k != "raw_data_pattern"}
    dest = resolve(local_dir) / grid_cfg.name
    n_partitions = n_partitions if n_partitions is not None else getattr(grid_cfg, "n_partitions", None)
    patterns = [v.rstrip("/") + ("/*" if v.endswith("/") else "") for v in f.values()]
    if getattr(grid_cfg, "raw_data_root", None):  # per-split raw data inside the model repo (Texas)
        for split in splits or grid_cfg.splits[:1]:
            prefix = grid_cfg.raw_data_root.format(split=split).rstrip("/") + "/"
            patterns += _partition_patterns(prefix, n_partitions)
    snapshot_download(grid_cfg.hf_model_repo, allow_patterns=patterns, local_dir=dest)
    paths = {k: dest / v for k, v in f.items() if k in ("checkpoint", "normalizer_stats", "config", "splits_json")}
    if getattr(grid_cfg, "graphkit_config", None):  # eval config kept in this repo overrides the shipped one
        paths["config"] = resolve(grid_cfg.graphkit_config)
    if getattr(grid_cfg, "hf_dataset_repo", None):
        snapshot_download(grid_cfg.hf_dataset_repo, repo_type="dataset",
                          allow_patterns=_partition_patterns("", n_partitions),
                          local_dir=dest / "data" / grid_cfg.network / "raw")
    paths["data_root"] = dest / "data"
    return paths


def raw_scenario_ids(raw_dir: str | Path) -> np.ndarray:
    """Sorted unique scenario ids present in a (Hive-partitioned) datakit ``bus_data`` table."""
    import pyarrow.dataset as ds

    table = ds.dataset(Path(raw_dir) / "bus_data.parquet", format="parquet", partitioning="hive")
    return np.unique(table.to_table(columns=["scenario"]).column("scenario").to_numpy())


def write_all_test_split(raw_dir: str | Path, out_dir: str | Path) -> int:
    """Split files for a dataset the checkpoint never trained on (our N-k sets): every scenario in ``test.pt``,
    empty ``train.pt``/``val.pt`` (normaliser statistics are restored from the training run). Returns #test."""
    import torch

    ids = raw_scenario_ids(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torch.as_tensor(ids, dtype=torch.long), out / "test.pt")
    for split in ("train", "val"):
        torch.save(torch.zeros(0, dtype=torch.long), out / f"{split}.pt")
    return int(ids.size)


def write_split_files(splits_json: str | Path, raw_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Restrict the training run's ``{train,val,test}`` scenario ids to the scenarios present in ``raw_dir``.

    Writes ``out_dir/{train,val,test}.pt`` (int64 tensors of dataset indices = datakit scenario ids, which is how
    ``HeteroGridDatasetDisk`` names ``data_index_<scenario>.pt``) for graphkit's ``data.split_from_existing_files``.
    This keeps the evaluation split disjoint from the checkpoint's training scenarios when only a prefix of the
    dataset is downloaded. Returns the number of ids per split.
    """
    import json

    import torch

    present = raw_scenario_ids(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, ids in json.loads(Path(splits_json).read_text()).items():
        ids = np.asarray(ids, np.int64)
        keep = torch.as_tensor(np.sort(ids[np.isin(ids, present)]), dtype=torch.long)
        torch.save(keep, out / f"{split}.pt")
        counts[split] = int(keep.numel())
    if sum(counts.values()) != present.size:
        raise RuntimeError(f"split files cover {sum(counts.values())} of {present.size} downloaded scenarios")
    return counts


def data_root_for_split(grid_cfg, paths: dict[str, Path], split: str) -> Path:
    """graphkit expects ``<data_root>/<network>/raw/*.parquet``. Texas ships one data root per split
    (``data/<split>/<network>/raw``); 118 has one root (``data/<network>/raw``) plus our N-k roots."""
    per_split = paths["data_root"] / split
    return per_split if (per_split / grid_cfg.network / "raw").exists() else paths["data_root"]


def _check_processed_cache(network_dir: Path) -> None:
    """graphkit skips processing when ``processed/processed_raw_files.done`` exists, even if more raw partitions were
    downloaded since. Refuse to run on a cache whose graph count differs from the raw scenario count."""
    proc = network_dir / "processed"
    if not (proc / "processed_raw_files.done").exists():
        return
    n_proc = sum(1 for _ in proc.glob("data_index_*.pt"))
    n_raw = raw_scenario_ids(network_dir / "raw").size
    if n_proc != n_raw:
        raise RuntimeError(f"stale graphkit cache {proc}: {n_proc} processed graphs vs {n_raw} raw scenarios; "
                           "delete the processed/ folder (it is regenerated)")


def load_genco(grid_cfg, paths: dict[str, Path], *, split: str, device: str = "cuda",
               batch_size: int | None = None, num_workers: int | None = None) -> LoadedGenco:
    """Build datamodule + task exactly like ``main_cli`` and load the weights. Asserts ``num_layers`` and ``baseMVA``."""
    import torch
    import yaml
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task

    raw = yaml.safe_load(Path(paths["config"]).read_text())
    try:  # main has config versioning; old configs (v0, e.g. Texas) need auto conversion
        from gridfm_graphkit.io.config_version import resolve_config

        raw = resolve_config(raw, config_path=str(paths["config"]), conversion_mode="auto_inline")
    except ImportError:
        pass  # paper branch: no versioning
    root = data_root_for_split(grid_cfg, paths, split)
    _check_processed_cache(root / grid_cfg.network)
    split_dir = raw["data"].get("split_from_existing_files")
    if (root / "splits" / "test.pt").exists():  # a data root that carries its own split (e.g. our N-k sets)
        split_dir = root / "splits"
        raw["data"].pop("split_by_load_scenario_idx", None)
    if split_dir is not None:  # repo-relative in our configs; graphkit resolves it against the cwd
        raw["data"]["split_from_existing_files"] = str(resolve(split_dir))
    cfg = NestedNamespace(**raw)
    cfg.data.networks = [grid_cfg.network]
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if num_workers is not None:
        cfg.data.workers = num_workers

    dm = LitGridHeteroDataModule(cfg, str(root), normalizer_stats_path=str(paths["normalizer_stats"]))
    dm.setup("test")
    task = get_task(cfg, dm.data_normalizers)
    sd = torch.load(paths["checkpoint"], map_location="cpu")
    sd = {k.replace("model._orig_mod.", "model."): v for k, v in sd.items()}  # checkpoints saved from torch.compile
    missing, unexpected = task.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch (branch/checkpoint incompatible?): missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    task.eval().to(device)
    base_mva = float(dm.data_normalizers[0].baseMVA)
    n_steps = int(task.model.num_layers)
    log.info("loaded %s: base_mva=%.3f (fitted), n_steps=%d", grid_cfg.name, base_mva, n_steps)
    return LoadedGenco(task=task, model=task.model, datamodule=dm, normalizers=dm.data_normalizers,
                       base_mva=base_mva, n_steps=n_steps, graphkit_commit=graphkit_commit(), grid=grid_cfg.name,
                       raw_dir=root / grid_cfg.network / "raw")


def graphkit_commit() -> str:
    """git commit of the installed gridfm_graphkit (editable install), else its version string."""
    import gridfm_graphkit

    pkg = Path(gridfm_graphkit.__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(pkg), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — not a git checkout
        return getattr(gridfm_graphkit, "__version__", "unknown")

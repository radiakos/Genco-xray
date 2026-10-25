"""Download checkpoints, configs, normaliser stats and raw data for the selected grids."""

import argparse
import json
import logging

from gencoxray import io
from gencoxray.checkpoints import download, load_genco, write_split_files
from gencoxray.config import load_config, load_grid

log = logging.getLogger("download")


def check_load(grid, paths, split: str, batch_size: int | None = None) -> None:
    """Load check: strict state-dict load, n_steps == 12, fitted base_mva, one forward batch on the test split.
    Writes results/<grid>/m0_check.json."""
    import torch

    loaded = load_genco(grid, paths, split=split, device="cuda" if torch.cuda.is_available() else "cpu",
                        batch_size=batch_size, num_workers=0)
    batch = next(iter(loaded.datamodule.predict_dataloader()[0])).to(loaded.task.device)
    with torch.no_grad():
        out = loaded.task.model(batch)
    n_test = len(loaded.datamodule.test_datasets[0])
    rec = {"grid": grid.name, "split": split, "strict_load": True, "n_steps": loaded.n_steps,
           "base_mva_fitted": loaded.base_mva, "n_test_scenarios": n_test,
           "graphkit_commit": loaded.graphkit_commit, "forward_output_keys": sorted(out) if isinstance(out, dict)
           else type(out).__name__}
    if loaded.n_steps != 12:
        raise RuntimeError(f"expected 12 correction steps, got {loaded.n_steps}")
    (io.results_dir(grid.name) / "m0_check.json").write_text(json.dumps(rec, indent=2))
    log.info("load check %s: %s", grid.name, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="append", choices=["texas2k", "case118", "case2000"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-partitions", type=int, default=None, help="Hive partitions per split (default: grid cfg)")
    ap.add_argument("--split", action="append", default=None, help="per-split grids (Texas): splits to fetch")
    ap.add_argument("--check-load", action="store_true", help="load check: strict load + one forward batch")
    ap.add_argument("--batch-size", type=int, default=None, help="override training.batch_size for --check-load")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    exp = load_config()
    grids = ["texas2k", "case118", "case2000"] if a.all else (a.grid or ["case118"])
    for g in grids:
        grid = load_grid(g)
        paths = download(grid, local_dir=exp.data_dir, n_partitions=a.n_partitions, splits=a.split)
        if "splits_json" in paths:
            raw = paths["data_root"] / grid.network / "raw"
            counts = write_split_files(paths["splits_json"], raw, paths["data_root"] / "splits")
            log.info("%s: split files restricted to downloaded scenarios: %s", g, counts)
        log.info("%s -> %s", g, paths)
        if a.check_load:
            check_load(grid, paths, (a.split or grid.splits[:1])[0], a.batch_size)

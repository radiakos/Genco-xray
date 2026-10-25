"""Fill error-predictor features and position variables offline from the TRACES shards (CPU only, restartable).

Per split: for each shard, features.add_bus_features / add_branch_features and confounders.hop_to_outage;
rewrites bus.parquet / branch.parquet (idempotent: every feature column is recomputed). Also refreshes
static_bus: bus_type = the bus's most frequent type over the first split (the capture step takes it from one
scenario, which may have a generator outage), degree = distinct physical neighbours, hop_to_pv_ref / hop_to_ref (confounders.hop_distance).
Acceptance numbers go to results/summary.json:m3.<grid>.<split>.

    python scripts/build_features.py --grid case118
    python scripts/build_features.py --grid texas2k --split test_1
    python scripts/build_features.py --grid synthetic --synthetic     # day-0 development
"""

import logging
import time

import numpy as np
import pandas as pd

from gencoxray import cli, confounders, features, io, reference, schemas
from gencoxray.config import SYNTHETIC, load_config, load_grid, resolve

log = logging.getLogger("build_features")


def write_synthetic(seed: int, shard_size: int) -> None:
    """synthetic.make_dataset() -> results/synthetic/{static_*, test/*, test/traces/shard_*.npz}."""
    from gencoxray.synthetic import make_dataset

    ds = make_dataset(seed=seed)
    io.write_table(ds.static_bus, "static_bus", SYNTHETIC)
    io.write_table(ds.static_branch, "static_branch", SYNTHETIC)
    for name in ("bus", "branch", "scenario"):
        io.write_table(getattr(ds, name), name, SYNTHETIC, "test")
    io.clear_traces(SYNTHETIC, "test")
    tr, S = ds.traces, len(ds.traces["scenario"])
    for i, lo in enumerate(range(0, S, shard_size)):
        io.write_traces(io.slice_traces(tr, slice(lo, lo + shard_size)), SYNTHETIC, "test", i)
    log.info("synthetic dataset written (%d scenarios)", S)


def gen_bus_of(grid, exp, split: str, scenario: int) -> np.ndarray:
    """Bus of every generator row (raw gen_data order = ``outaged_gens`` ids) for this split."""
    from gencoxray.checkpoints import data_root_for_split

    root = resolve(exp.data_dir) / grid.name / "data"
    raw = data_root_for_split(grid, {"data_root": root}, split) / grid.network / "raw"
    return reference.read_raw(raw, "gen_data", np.array([scenario])).bus.to_numpy(np.int64)


def refresh_static(grid_name: str, first_split: str, base_out_branches: np.ndarray) -> pd.DataFrame:
    sb = io.read_table("static_bus", grid_name)
    sbr = io.read_table("static_branch", grid_name).sort_values("branch")
    bt = io.read_columns("bus", grid_name, first_split, ["bus", "bus_type"])
    counts = pd.crosstab(bt.bus, bt.bus_type.astype(str)).reindex(columns=list(schemas.BUS_TYPES), fill_value=0)
    modal = counts.idxmax(axis=1).reindex(sb.bus).to_numpy()
    n = len(sb)
    keep = ~sbr.branch.isin(base_out_branches).to_numpy()
    ei = np.stack([sbr.from_bus.to_numpy()[keep], sbr.to_bus.to_numpy()[keep]])
    changed = int((sb.bus_type.astype(str).to_numpy() != modal).sum())
    sb = sb.assign(bus_type=pd.Categorical(modal, categories=schemas.BUS_TYPES),
                   degree=confounders.degree(ei, n),
                   hop_to_pv_ref=confounders.hop_distance(ei, np.flatnonzero(modal != "PQ"), n),
                   hop_to_ref=confounders.hop_distance(ei, np.flatnonzero(modal == "REF"), n))
    io.write_table(sb, "static_bus", grid_name)
    log.info("%s static_bus refreshed: %d bus types changed to the modal type; hop_to_pv_ref max %d, hop_to_ref max %d",
             grid_name, changed, sb.hop_to_pv_ref.max(), sb.hop_to_ref.max())
    return sb


def build_split(grid, exp, split: str, static_branch: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    gname = grid.name if grid is not None else SYNTHETIC
    bus = io.read_table("bus", gname, split)
    branch = io.read_table("branch", gname, split)
    scen = io.read_table("scenario", gname, split).set_index("scenario")
    any_gen_out = bool(scen.outaged_gens.map(len).sum())
    gen_bus = gen_bus_of(grid, exp, split, int(scen.index[0])) if (any_gen_out and grid is not None) else np.zeros(0, int)

    bus_parts, br_parts = [], []
    for sh in io.iter_traces(gname, split):
        sc = np.sort(sh["scenario"])
        b = features.add_bus_features(bus, sh)
        order = np.argsort(sh["scenario"], kind="stable")
        on = features.in_service_edges(sh["loading"][order])  # [S, E] in-service directed edges
        B = sh["res_p"].shape[2]
        src = confounders.outage_sources(scen.loc[sc].reset_index(), static_branch, gen_bus, B)
        b["hop_to_outage"] = confounders.hop_to_outage(sh["edge_from"], sh["edge_to"], on, src).reshape(-1)
        br_parts.append(features.add_branch_features(branch, b, sh, static_branch))
        bus_parts.append(b)
    bus_new = pd.concat(bus_parts, ignore_index=True)[bus.columns]
    br_new = pd.concat(br_parts, ignore_index=True)[branch.columns]
    if len(bus_new) != len(bus) or len(br_new) != len(branch):
        raise SystemExit(f"{gname}/{split}: shards cover {len(bus_new)}/{len(bus)} bus and "
                         f"{len(br_new)}/{len(branch)} branch rows")
    bus_new["hop_to_outage"] = bus_new.hop_to_outage.astype(np.float32)
    io.write_table(bus_new, "bus", gname, split)
    io.write_table(br_new, "branch", gname, split)

    pq = (bus_new.bus_type == "PQ").to_numpy()
    kpos = (bus_new.k >= 1).to_numpy()
    feat_nan_pq = int(bus_new.loc[pq, list(schemas.BUS_FEATURES)].isna().any(axis=1).sum())
    hop_nan = int(bus_new.loc[kpos, "hop_to_outage"].isna().sum())
    srv = br_new.in_service.to_numpy()
    br_nan = int(br_new.loc[srv, list(schemas.BRANCH_OWN_FEATURES)].isna().any(axis=1).sum())
    rec = {
        "n_scenarios": len(scen), "n_bus_rows": len(bus_new), "n_pq_rows": int(pq.sum()),
        "pq_rows_with_nan_feature": feat_nan_pq,
        "k_ge1_rows": int(kpos.sum()), "k_ge1_rows_hop_to_outage_nan": hop_nan,
        "hop_to_outage_max": float(np.nanmax(bus_new.hop_to_outage)) if kpos.any() else None,
        "in_service_branch_rows_with_nan_feature": br_nan,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    rec["passed"] = bool(feat_nan_pq == 0 and hop_nan == 0 and br_nan == 0)
    io.update_summary(f"m3.{gname}.{split.replace('/', '_')}", rec)
    log.info("%s/%s: %s", gname, split, rec)
    return rec


if __name__ == "__main__":
    ap = cli.parser(__doc__)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--synthetic", action="store_true", help="write synthetic.make_dataset() as results/synthetic/test")
    a = ap.parse_args()
    cli.setup_logging()
    exp = load_config()

    if a.synthetic or a.grid == SYNTHETIC:
        if a.synthetic:
            write_synthetic(exp.seed, exp.trace.batch_size)
        grid, splits = None, a.split or ["test"]
    else:
        grid = load_grid(a.grid)
        splits = a.split or grid.splits
    gname = a.grid

    if grid is not None:  # static refresh uses the grid's first split (base-case offline branches excluded)
        first = grid.splits[0]
        meta = io.read_json("meta.json", gname, first)
        refresh_static(gname, first, np.asarray(meta.get("base_offline", {}).get("branches", []), int))
    static_branch = io.read_table("static_branch", gname)

    failed = [s for s in splits if not build_split(grid, exp, s, static_branch)["passed"]]
    if failed:
        raise SystemExit(f"feature acceptance check failed for {gname}: {failed} (see results/summary.json:m3)")
    log.info("features done for %s: %d splits", gname, len(splits))

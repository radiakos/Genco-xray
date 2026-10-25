"""Standard uncertainty baselines against the last-step change and the Newton step, on the same rows.
Writes results/<grid>/uncertainty_{bus,branch}.parquet, uncertainty_metrics.csv and summary.uncertainty.<grid>.
Run after newton_check.py (the rows are its scenario sample) and, for the seed spread, after the seed runs.

    python scripts/uncertainty_baselines.py --grid texas2k
    python scripts/uncertainty_baselines.py --grid case118 --seed-grids case118_seed1 case118_seed42

Methods:
* Load-perturbation spread (--passes GENCO passes, 8 by default): every bus load (Pd, Qd) scaled by
  (1 + --rel eps), eps ~ N(0, 1) drawn independently per load and pass (seeded); score = standard deviation over the
  passes of the final |V| per bus and of the final loading per branch. Generator set-points and network unchanged.
* Seed spread (IEEE 118-bus only; the release has three Base seeds): standard deviation of vm_pred / loading_pred
  over the checkpoints of --seed-grids plus seed 0, on the N-k sets the seeds were all run on. Cost: 3 passes.
* Compared on the same rows with: one Newton step, last-step change (neighbourhood and own bus; branch: std of the
  last four loadings), final mismatch. Same labels and headline thresholds as error_check.py; resampling by load
  scenario.
"""

import logging
import os

import numpy as np
import pandas as pd

from gencoxray import cli, io, trust
from gencoxray.check import _edge_pairs, branch_ids, loading_from_flows
from gencoxray.checkpoints import download, load_genco
from gencoxray.config import grid_splits, k_fit, load_config, load_grid
from gencoxray.reference import read_raw
from gencoxray.runner import branch_status_matrix

os.environ.setdefault("HF_HUB_OFFLINE", "1")
log = logging.getLogger("uncertainty")
BUS_SCORES = {"newton_step": "newton_dvm", "last_step_vm_nbmax": "nbmax_f_drift_vm", "last_step_vm_own": "f_drift_vm",
              "final_mismatch": "f_final", "load_spread": "spread_vm", "seed_spread": "seed_spread_vm"}
BR_SCORES = {"newton_step": "newton_dl", "last_step_loading": "g_drift_tail_std",
             "final_mismatch": "g_endpoint_max_f_final", "load_spread": "spread_loading",
             "seed_spread": "seed_spread_loading"}


def load_spread(gname: str, split: str, scenarios: np.ndarray, passes: int, rel: float, batch_size: int,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (scenario, bus) std of final Vm and per (scenario, branch) std of final loading over perturbed passes."""
    import torch
    from gridfm_graphkit.datasets.globals import PD_H, QD_H
    from torch_geometric.data import Batch

    from gencoxray.trace import TraceRecorder, forward_genco

    exp, grid = load_config(), load_grid(gname)
    loaded = load_genco(grid, download(grid, local_dir=exp.data_dir), split=split, device=exp.device, num_workers=0)
    ds = loaded.datamodule.test_datasets[0]
    pos = {int(s): i for i, s in enumerate(np.asarray(ds.indices))}
    scenarios = np.asarray([s for s in scenarios if int(s) in pos])
    br_raw = read_raw(loaded.raw_dir, "branch_data", scenarios)
    rate = br_raw[br_raw.scenario == scenarios[0]].rate_a.to_numpy()
    rng = np.random.default_rng(seed)
    rec = TraceRecorder(loaded.task.model)
    bus_rows, br_rows = [], []
    with torch.inference_mode(), rec:
        for i in range(0, scenarios.size, batch_size):
            batch = Batch.from_data_list([ds[pos[int(s)]] for s in scenarios[i:i + batch_size]]).to(loaded.device)
            sid = batch["scenario_id"].reshape(-1).cpu().numpy().astype(np.int64)
            status = branch_status_matrix(br_raw, sid)
            x0 = batch["bus"].x.clone()
            vm, ld = [], []
            for _ in range(passes):
                x = x0.clone()
                eps = torch.as_tensor(rng.normal(size=(x.shape[0], 2)), dtype=x.dtype, device=x.device)
                x[:, PD_H] *= 1 + rel * eps[:, 0]
                x[:, QD_H] *= 1 + rel * eps[:, 1]
                batch["bus"].x = x
                rec.reset()
                forward_genco(loaded.task, batch)
                tb = rec.to_trace_batch(batch, base_mva=loaded.base_mva)
                vm.append(tb.vm[-1].astype(np.float64))
                L = loading_from_flows(tb, rate, status)[-1]
                fw, rv = _edge_pairs(tb)  # branch loading = max of its two directions
                ld.append(np.fmax(L[fw], L[rv]))
            batch["bus"].x = x0
            bus_rows.append(pd.DataFrame({"scenario": sid[tb.bus_graph], "bus": tb.bus_local,
                                          "spread_vm": np.std(vm, axis=0)}))
            g = tb.bus_graph[tb.edge_index[0, fw]]
            br_rows.append(pd.DataFrame({"scenario": sid[g], "branch": branch_ids(tb, status),
                                         "spread_loading": np.std(ld, axis=0)}))
    log.info("%s: load spread on %d scenarios x %d passes", split, scenarios.size, passes)
    return (pd.concat(bus_rows, ignore_index=True).assign(split=split),
            pd.concat(br_rows, ignore_index=True).assign(split=split))


def seed_spread(gname: str, seed_grids: list[str], splits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Std over checkpoints (seed 0 = ``gname`` plus ``seed_grids``) of vm_pred and loading_pred, per row."""
    bus, br = [], []
    for s in splits:
        b = [io.read_columns("bus", g, s, ["scenario", "bus", "vm_pred"]).set_index(["scenario", "bus"]).vm_pred
             for g in [gname] + seed_grids]
        r = [io.read_columns("branch", g, s, ["scenario", "branch", "loading_pred"])
             .set_index(["scenario", "branch"]).loading_pred for g in [gname] + seed_grids]
        bus.append(pd.concat(b, axis=1).std(axis=1, ddof=0).rename("seed_spread_vm").reset_index().assign(split=s))
        br.append(pd.concat(r, axis=1).std(axis=1, ddof=0).rename("seed_spread_loading").reset_index().assign(split=s))
    return pd.concat(bus, ignore_index=True), pd.concat(br, ignore_index=True)


def evaluate(B: pd.DataFrame, R: pd.DataFrame, e2: dict, k_max: int, n_boot: int, seed: int) -> pd.DataFrame:
    out = []
    for level, D, err, tau, scores in (("bus", B, "err_vm", e2["bus"]["headline_tau"], BUS_SCORES),
                                       ("branch", R, "err_loading", e2["branch"]["headline_tau"], BR_SCORES)):
        sc = {n: c for n, c in scores.items() if c in D}
        D = D.dropna(subset=list(sc.values()))  # identical rows for every method
        k = D.k.to_numpy()
        for setname, m in (("in_dist", k <= k_max), ("shift_pooled", k > k_max)):
            if m.any():
                ev = trust.evaluate_scores({n: D[c].to_numpy(np.float64)[m] for n, c in sc.items()},
                                           D[err].to_numpy(np.float64)[m], tau, D.load_scenario.to_numpy()[m],
                                           n_boot=n_boot, seed=seed)
                out.append(ev.assign(level=level, set=setname, tau=tau))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--rel", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed-grids", nargs="*", default=[])
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    out = io.results_dir(gname)
    S = pd.read_csv(out / "newton_scenario.csv")
    B = pd.read_parquet(out / "newton_bus.parquet")
    R = pd.read_parquet(out / "newton_branch.parquet")
    spreads = [load_spread(gname, s, S[S.split == s].scenario.to_numpy(), a.passes, a.rel, a.batch_size,
                           cfg.seed + i) for i, s in enumerate(grid_splits(gname)) if (S.split == s).any()]
    B = B.merge(pd.concat([p[0] for p in spreads]), on=["split", "scenario", "bus"], how="left")
    R = R.merge(pd.concat([p[1] for p in spreads]), on=["split", "scenario", "branch"], how="left")
    if a.seed_grids:
        seed_splits = list(load_grid(a.seed_grids[0]).splits)
        sb, sr = seed_spread(gname, a.seed_grids, seed_splits)
        B = B[B.split.isin(seed_splits)].merge(sb, on=["split", "scenario", "bus"], how="left")
        R = R[R.split.isin(seed_splits)].merge(sr, on=["split", "scenario", "branch"], how="left")
    B.to_parquet(out / "uncertainty_bus.parquet", index=False)
    R.to_parquet(out / "uncertainty_branch.parquet", index=False)
    M = evaluate(B, R, io.read_summary()["e2"][gname], k_fit(gname), cfg.trust.bootstrap, cfg.seed)
    M.to_csv(out / "uncertainty_metrics.csv", index=False)
    keep = ["auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "n", "n_pos"]
    summ = {"protocol": {"passes": a.passes, "rel": a.rel, "seed_grids": a.seed_grids, "rows": "newton_check sample"},
            "passes_per_method": {"newton_step": "1 + 1 factorization", "last_step": 1, "final_mismatch": 1,
                                  "load_spread": a.passes, "seed_spread": 1 + len(a.seed_grids)},
            "metrics": {f"{r.level}.{r.set}.{r.score}": {c: float(getattr(r, c)) for c in keep} for r in M.itertuples()}}
    io.update_summary(f"uncertainty.{gname}", summ)
    for key, v in summ["metrics"].items():
        log.info("%-38s AUROC %.4f AUPRC %.4f (n %d, pos %d)", key, v["auroc"], v["auprc"], v["n"], v["n_pos"])

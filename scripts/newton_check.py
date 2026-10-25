"""Newton-Raphson error estimate, Jacobian conditioning and warm start from GENCO (sparse polar NR, newton.py).
Writes results/<grid>/newton_{bus,branch}.parquet, newton_scenario.csv, newton_warm.csv, newton_metrics.csv and
summary.newton.<grid>. Run after error_check.py (uses its headline thresholds).

    python scripts/newton_check.py --grid texas2k --per-split 100 --warm-per-split 50
    python scripts/newton_check.py --grid case118 --per-split 0 --warm-per-split 50     # 0 = every scenario

Protocol:
* Sample: --per-split seeded scenarios per outage count (0 = all); the warm-start study uses the first
  --warm-per-split of them. Network, bus types, specified injections and PV/REF setpoints: the scenario's own datakit
  data (newton.network). GENCO's state: vm_pred, va_pred of the BUS table, known values imposed (GENCO predicts Vm at
  the REF bus on this graphkit branch; the setpoint replaces it).
* Error estimate: one Newton step dx = -J(x_hat)^-1 g(x_hat). Scores: |dVm| per PQ bus; per in-service branch
  |L(x_hat + dx) - L_GENCO|, with L_GENCO the flows of GENCO's own state (REF Vm as predicted). Same labels and thresholds as error_check.py (headline tau), same comparison scores
  (last-step change, final mismatch) on the same rows; AUROC/AUPRC with resampling by load scenario.
* Conditioning: sigma_min(J(x_hat)) by inverse iteration (20 iterations, sparse LU).
* Checks: max |g| and the Newton step at datakit's AC solution (must be negligible); our loading at GENCO's state
  against GENCO's own loading_pred; share of in-service generators whose AC reactive output lies outside its limits
  (> 0 means datakit does not enforce reactive limits, as assumed by newton.py).
* Warm start: solve to max |g| < 1e-8 p.u. (at most 20 iterations) from flat, DC (datakit's Va_dc) and GENCO starts;
  iterations, wall time, non-convergence, agreement with datakit; GENCO + one Newton step against GENCO alone.
"""

import logging
import os
import time

import numpy as np
import pandas as pd

from gencoxray import cli, io, newton, trust
from gencoxray.checkpoints import data_root_for_split
from gencoxray.config import grid_splits, k_fit, load_config, load_grid, resolve

os.environ.setdefault("HF_HUB_OFFLINE", "1")
log = logging.getLogger("newton_check")
BUS_COLS = ["scenario", "load_scenario", "bus", "k", "bus_type", "vm_pred", "va_pred", "vm_true", "va_true", "err_vm",
            "f_final", "f_drift_vm", "nbmax_f_drift_vm"]
BR_COLS = ["scenario", "branch", "in_service", "loading_pred", "loading_true", "err_loading", "g_drift_tail_std",
           "g_endpoint_max_f_final"]
STARTS = ("flat", "dc", "genco")


def raw_dir(gname: str, split: str):
    g = load_grid(gname)
    return data_root_for_split(g, {"data_root": resolve(load_config().data_dir) / gname / "data"}, split) / g.network / "raw"


def q_limit_check(raw, scen: np.ndarray) -> dict:
    gen = pd.read_parquet(raw / "gen_data.parquet", columns=["scenario", "q_mvar", "min_q_mvar", "max_q_mvar",
                                                              "in_service"], filters=[("scenario", "in", scen.tolist())])
    gen = gen[gen.in_service == 1]
    out = (gen.q_mvar > gen.max_q_mvar + 1e-3) | (gen.q_mvar < gen.min_q_mvar - 1e-3)
    return {"n_gen_rows": len(gen), "n_outside_q_limits": int(out.sum())}


def run_split(gname: str, split: str, n_per: int, n_warm: int, seed: int):
    bus_t = io.read_columns("bus", gname, split, BUS_COLS)
    br_t = io.read_columns("branch", gname, split, BR_COLS)
    scen = np.unique(bus_t.scenario.to_numpy())
    if n_per and n_per < scen.size:
        scen = np.sort(np.random.default_rng(seed).choice(scen, n_per, replace=False))
    raw = raw_dir(gname, split)
    flt = [("scenario", "in", scen.tolist())]
    bd = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    brd = pd.read_parquet(raw / "branch_data.parquet", filters=flt)
    sort = ["scenario", "idx"] if "idx" in brd else ["scenario"]
    brd = brd.sort_values(sort, kind="stable")
    bus_t = bus_t[bus_t.scenario.isin(scen)].sort_values(["scenario", "bus"])
    br_t = br_t[br_t.scenario.isin(scen)].sort_values(["scenario", "branch"])
    bd_g, brd_g = dict(tuple(bd.groupby("scenario"))), dict(tuple(brd.groupby("scenario")))
    bus_g, br_g = dict(tuple(bus_t.groupby("scenario"))), dict(tuple(br_t.groupby("scenario")))
    S_rows, B_rows, R_rows, W_rows = [], [], [], []
    for i, s in enumerate(scen):
        b_raw, g = bd_g[s].sort_values("bus"), bus_g[s]
        net = newton.network(b_raw, brd_g[s])
        vm_raw, va_raw = g.vm_pred.to_numpy(np.float64), np.deg2rad(g.va_pred.to_numpy(np.float64))
        vm_hat, va_hat = newton.fix_known(net, vm_raw, va_raw)  # GENCO predicts Vm at REF; NR holds the setpoint
        vm_ac, va_ac = b_raw.Vm.to_numpy(np.float64), np.deg2rad(b_raw.Va.to_numpy(np.float64))
        t0 = time.perf_counter()
        dvm, dva, g_inf, lu = newton.newton_step(net, vm_hat, va_hat)
        t_step = time.perf_counter() - t0
        smin = newton.sigma_min(lu, net.n, seed=seed)
        dvm_ac, dva_ac, g_ac, _ = newton.newton_step(net, vm_ac, va_ac)
        L_hat = newton.loading(net, vm_raw, va_raw)  # GENCO's own flows (= loading_pred)
        L_step = newton.loading(net, vm_hat + dvm, va_hat + dva)
        pq = np.zeros(len(g), bool)
        pq[net.mag] = True
        k, ls = int(g.k.iloc[0]), int(g.load_scenario.iloc[0])
        err_after = np.abs(vm_hat + dvm - g.vm_true.to_numpy())
        S_rows.append({"split": split, "scenario": int(s), "load_scenario": ls, "k": k, "sigma_min": smin,
                       "g_inf": g_inf, "g_ac_inf": g_ac, "step_at_ac_max": float(max(np.abs(dvm_ac).max(),
                                                                                         np.abs(dva_ac).max())),
                       "max_err_vm": float(g.err_vm[pq].max()), "max_newton_dvm": float(np.abs(dvm[pq]).max()),
                       "max_err_vm_after_step": float(err_after[pq].max()), "t_step_s": t_step,
                       "max_loading_formula_diff": float(np.nanmax(np.abs(L_hat - br_g[s].loading_pred.to_numpy())))})
        B_rows.append(g.loc[pq, ["scenario", "load_scenario", "bus", "k", "err_vm", "f_final", "f_drift_vm",
                                 "nbmax_f_drift_vm"]].assign(split=split, newton_dvm=np.abs(dvm[pq]),
                                                             err_vm_after_step=err_after[pq]))
        r = br_g[s]
        on = r.in_service.to_numpy()
        R_rows.append(r.loc[on, ["scenario", "branch", "err_loading", "g_drift_tail_std", "g_endpoint_max_f_final"]]
                      .assign(split=split, load_scenario=ls, k=k, newton_dl=np.abs(L_step - L_hat)[on],
                              err_loading_after_step=np.abs(L_step - r.loading_true.to_numpy())[on]))
        if i < n_warm:
            L_ac = newton.loading(net, vm_ac, va_ac)
            starts = {"flat": newton.flat_start(net), "dc": newton.dc_start(net, b_raw.Va_dc.to_numpy()),
                      "genco": (vm_hat, va_hat)}
            for name, (vm0, va0) in starts.items():
                res = newton.solve(net, vm0, va0)
                ok = res["converged"]
                W_rows.append({"split": split, "scenario": int(s), "k": k, "start": name,
                               "iterations": res["iterations"], "converged": ok, "seconds": res["seconds"],
                               "max_dvm_vs_datakit": float(np.abs(res["Vm"] - vm_ac).max()) if ok else np.nan,
                               "max_dloading_vs_datakit": float(np.nanmax(np.abs(
                                   newton.loading(net, res["Vm"], res["Va"]) - L_ac))) if ok else np.nan})
    q = q_limit_check(raw, scen)
    log.info("%s: %d scenarios, median sigma_min %.3g, gen rows outside Q limits %d/%d", split, scen.size,
             np.median([r["sigma_min"] for r in S_rows]), q["n_outside_q_limits"], q["n_gen_rows"])
    return (pd.DataFrame(S_rows), pd.concat(B_rows, ignore_index=True), pd.concat(R_rows, ignore_index=True),
            pd.DataFrame(W_rows), q)


def evaluate(B: pd.DataFrame, R: pd.DataFrame, e2: dict, k_max: int, n_boot: int, seed: int) -> pd.DataFrame:
    out = []
    for level, D, err, tau, scores in (
            ("bus", B, "err_vm", e2["bus"]["headline_tau"],
             {"newton_step": "newton_dvm", "last_step_vm_nbmax": "nbmax_f_drift_vm", "last_step_vm_own": "f_drift_vm",
              "final_mismatch": "f_final"}),
            ("branch", R, "err_loading", e2["branch"]["headline_tau"],
             {"newton_step": "newton_dl", "last_step_loading": "g_drift_tail_std",
              "final_mismatch": "g_endpoint_max_f_final"})):
        k = D.k.to_numpy()
        for setname, m in (("in_dist", k <= k_max), ("shift_pooled", k > k_max)):
            if not m.any():
                continue
            ev = trust.evaluate_scores({n: D[c].to_numpy(np.float64)[m] for n, c in scores.items()},
                                       D[err].to_numpy(np.float64)[m], tau, D.load_scenario.to_numpy()[m],
                                       n_boot=n_boot, seed=seed)
            out.append(ev.assign(level=level, set=setname, tau=tau))
    return pd.concat(out, ignore_index=True)


def conditioning(S: pd.DataFrame) -> pd.DataFrame:
    """Median of (max true |dVm| / max |g|) per sigma_min quintile and per outage count."""
    S = S.assign(ratio=S.max_err_vm / S.g_inf, q=pd.qcut(S.sigma_min, 5, labels=[1, 2, 3, 4, 5]))
    a = S.groupby("q", observed=True).agg(sigma_min_median=("sigma_min", "median"), ratio_median=("ratio", "median"),
                                          max_err_vm_median=("max_err_vm", "median"), g_inf_median=("g_inf", "median"),
                                          n=("ratio", "size")).reset_index().assign(by="sigma_min_quintile")
    b = S.groupby("k").agg(sigma_min_median=("sigma_min", "median"), ratio_median=("ratio", "median"),
                           max_err_vm_median=("max_err_vm", "median"), g_inf_median=("g_inf", "median"),
                           n=("ratio", "size")).reset_index().rename(columns={"k": "q"}).assign(by="k")
    return pd.concat([a, b], ignore_index=True)


def warm_summary(W: pd.DataFrame, B: pd.DataFrame, R: pd.DataFrame) -> dict:
    out = {}
    for name, d in W.groupby("start"):
        c = d[d.converged]
        out[name] = {"n": len(d), "n_not_converged": int((~d.converged).sum()),
                     "iterations_median": float(c.iterations.median()), "iterations_max": int(c.iterations.max()),
                     "ms_median": float(1e3 * c.seconds.median()), "ms_mean": float(1e3 * c.seconds.mean()),
                     "max_dvm_vs_datakit": float(c.max_dvm_vs_datakit.max()),
                     "max_dloading_vs_datakit": float(c.max_dloading_vs_datakit.max())}
    piv = W.pivot_table(index=["split", "scenario"], columns="start", values="converged")
    out["genco_fails_where_flat_converges"] = int((piv.flat.astype(bool) & ~piv.genco.astype(bool)).sum())
    out["one_newton_step"] = {
        "err_vm_median_before": float(B.err_vm.median()), "err_vm_median_after": float(B.err_vm_after_step.median()),
        "err_vm_max_before": float(B.err_vm.max()), "err_vm_max_after": float(B.err_vm_after_step.max()),
        "err_loading_median_before": float(R.err_loading.median()),
        "err_loading_median_after": float(R.err_loading_after_step.median()),
        "err_loading_max_before": float(R.err_loading.max()), "err_loading_max_after": float(R.err_loading_after_step.max())}
    return out


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--per-split", type=int, default=100, help="scenarios per outage count (0 = all)")
    ap.add_argument("--warm-per-split", type=int, default=50)
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    k_max = k_fit(gname)
    parts = [run_split(gname, s, a.per_split, a.warm_per_split, cfg.seed + i) for i, s in enumerate(grid_splits(gname))]
    S, B, R, W = (pd.concat([p[j] for p in parts], ignore_index=True) for j in range(4))
    qlim = {k: int(sum(p[4][k] for p in parts)) for k in ("n_gen_rows", "n_outside_q_limits")}
    out = io.results_dir(gname)
    B.to_parquet(out / "newton_bus.parquet", index=False)
    R.to_parquet(out / "newton_branch.parquet", index=False)
    S.to_csv(out / "newton_scenario.csv", index=False)
    W.to_csv(out / "newton_warm.csv", index=False)
    e2 = io.read_summary()["e2"][gname]
    M = evaluate(B, R, e2, k_max, cfg.trust.bootstrap, cfg.seed)
    M.to_csv(out / "newton_metrics.csv", index=False)
    C = conditioning(S)
    C.to_csv(out / "newton_conditioning.csv", index=False)
    from scipy.stats import spearmanr

    keep = ["auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "n", "n_pos", "prevalence"]
    summ = {
        "protocol": {"per_split": a.per_split, "warm_per_split": a.warm_per_split, "n_scenarios": len(S),
                     "tol_pu": 1e-8, "max_iter": 20, "q_limits_enforced_by_newton": False},
        "checks": {"max_g_at_ac": float(S.g_ac_inf.max()), "max_newton_step_at_ac": float(S.step_at_ac_max.max()),
                   "max_loading_formula_diff_vs_genco": float(S.max_loading_formula_diff.max()),
                   "datakit_q_limits": qlim | {"enforced": qlim["n_outside_q_limits"] == 0}},
        "spearman_newton_dvm_vs_err_vm": float(spearmanr(B.newton_dvm, B.err_vm).statistic),
        "spearman_newton_dl_vs_err_loading": float(spearmanr(R.newton_dl, R.err_loading, nan_policy="omit").statistic),
        "metrics": {f"{r.level}.{r.set}.{r.score}": {c: None if pd.isna(getattr(r, c)) else float(getattr(r, c))
                                                     for c in keep} for r in M.itertuples()},
        "conditioning": {f"{r.by}.{r.q}": {"sigma_min_median": float(r.sigma_min_median),
                                           "ratio_median": float(r.ratio_median), "n": int(r.n)} for r in C.itertuples()},
        "sigma_min_vs_k_spearman": float(spearmanr(S.sigma_min, S.k).statistic),
        "newton_step_ms_median": float(1e3 * S.t_step_s.median()),
        "warm_start": warm_summary(W, B, R),
    }
    io.update_summary(f"newton.{gname}", summ)
    log.info("checks %s", summ["checks"])
    log.info("spearman bus %.4f branch %.4f", summ["spearman_newton_dvm_vs_err_vm"],
             summ["spearman_newton_dl_vs_err_loading"])
    for key, v in summ["metrics"].items():
        log.info("%-40s AUROC %.4f AUPRC %.4f (n_pos %d)", key, v["auroc"], v["auprc"], v["n_pos"])
    log.info("conditioning %s", summ["conditioning"])
    log.info("warm start %s", summ["warm_start"])

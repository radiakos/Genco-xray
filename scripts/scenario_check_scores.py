"""Per-scenario scores for selecting cases for AC re-solve with the Newton step and the last-step change.
Writes results/<grid>/<split>/scenario_checks.parquet for every split and results/<grid>/newton_warm_corrected.csv.
Run after error_check.py and last_step_check.py (uses their thresholds); read by ac_resolve_selection.py.

    python scripts/scenario_check_scores.py --grid texas2k
    python scripts/scenario_check_scores.py --grid case118 --skip-newton      # last-step scores only (fast)

Per scenario (rows: PQ buses, in-service branches):
* Last-step change, no AC labels: max over branches of s^L_b (std of loading over the last four steps), max over PQ
  buses of the neighbourhood voltage change s^V_i; "last-step margin": z_b = (L_b - 1) / max(s^L_b, SCALE_FLOOR), ranked
  by the decision-flip score (hybrid.decision_flip_score), as for the rule scaled by expected error.
* Fixed rules (thresholds from summary.last_step_check, set inside the training range for 90 % recall):
  ``last_step_only`` flags a scenario if any bus or branch has a last-step change >= its threshold; ``recommendation``
  flags it if any element exceeds the last-step OR the final-mismatch threshold (accept only if both are small).
  ``rec_score`` = max over elements of the score divided by its threshold, a scale-free ranking of the same rule.
* Newton step (one Jacobian factorization per scenario): corrected state x_hat + dx (known values imposed), corrected
  loadings L^N; max L^N; max |L^N - L_AC|; the change dl_b = |L^N_b - L_GENCO,b|; ``z_newton_flip`` = decision-flip
  score of z_b = (L^N_b - 1) / max(dl_b, SCALE_FLOOR); ``newton_max_dl`` = max_b dl_b.
* Labels per scenario: any PQ bus with |V - V_AC| > tau_V, any in-service branch with |L - L_AC| > tau_L.
* Warm start (first --warm-per-split scenarios of each split): Newton iterations from flat, GENCO's state and GENCO's
  state after one Newton step (``genco_step``).
SCALE_FLOOR keeps the scale positive where a change is exactly zero; it only affects scenarios whose change is
below 1e-4 loading units, which are far from the limit by construction.
"""

import importlib.util
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gencoxray import cli, hybrid, io, newton
from gencoxray.config import grid_splits, load_config

log = logging.getLogger("scenario_checks")
SCALE_FLOOR = 1e-4
_spec = importlib.util.spec_from_file_location("newton_check", Path(__file__).with_name("newton_check.py"))
newton_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(newton_check)

BUS_COLS = ["scenario", "load_scenario", "bus", "k", "bus_type", "vm_pred", "va_pred", "vm_true", "err_vm", "f_final",
            "nbmax_f_drift_vm"]
BR_COLS = ["scenario", "branch", "in_service", "loading_pred", "loading_true", "err_loading", "g_drift_tail_std",
           "g_endpoint_max_f_final"]


def thresholds(gname: str) -> tuple[dict, dict]:
    s = io.read_summary()
    ls = s["last_step_check"][gname]
    thr = {"vm": ls["bus"]["in_dist"]["last_step_vm_nbmax"]["threshold"],
           "mis_bus": ls["bus"]["in_dist"]["final_mismatch"]["threshold"],
           "ld": ls["branch"]["in_dist"]["last_step_loading"]["threshold"],
           "mis_br": ls["branch"]["in_dist"]["final_mismatch"]["threshold"]}
    tau = {"bus": s["e2"][gname]["bus"]["headline_tau"], "branch": s["e2"][gname]["branch"]["headline_tau"]}
    return thr, tau


def table_scores(bus: pd.DataFrame, br: pd.DataFrame, thr: dict, tau: dict, limit: float) -> pd.DataFrame:
    """Scores that need only the bus and branch tables."""
    pq = bus[bus.bus_type == "PQ"]
    on = br[br.in_service.astype(bool) & br.loading_pred.notna()]
    gb = pq.groupby("scenario")
    gr = on.groupby("scenario")
    out = pd.DataFrame({
        "max_last_vm": gb.nbmax_f_drift_vm.max(), "max_mis_bus": gb.f_final.max(),
        "n_vm_flag": (pq.nbmax_f_drift_vm >= thr["vm"]).groupby(pq.scenario).sum(),
        "n_mis_bus_flag": (pq.f_final >= thr["mis_bus"]).groupby(pq.scenario).sum(),
        "any_wrong_bus": (pq.err_vm > tau["bus"]).groupby(pq.scenario).any(),
        "max_last_ld": gr.g_drift_tail_std.max(), "max_mis_br": gr.g_endpoint_max_f_final.max(),
        "n_ld_flag": (on.g_drift_tail_std >= thr["ld"]).groupby(on.scenario).sum(),
        "n_mis_br_flag": (on.g_endpoint_max_f_final >= thr["mis_br"]).groupby(on.scenario).sum(),
        "any_wrong_branch": (on.err_loading > tau["branch"]).groupby(on.scenario).any(),
    })
    z = (on.loading_pred - limit) / np.maximum(on.g_drift_tail_std.fillna(0.0), SCALE_FLOOR)
    out["z_last_flip"] = hybrid.decision_flip_score(on.scenario.to_numpy(), z.to_numpy())
    out["rec_score"] = np.fmax.reduce([out.max_last_vm / thr["vm"], out.max_mis_bus / thr["mis_bus"],
                                       out.max_last_ld / thr["ld"], out.max_mis_br / thr["mis_br"]])
    out["flag_last_only"] = (out.n_vm_flag + out.n_ld_flag) > 0
    out["flag_recommendation"] = out.flag_last_only | ((out.n_mis_bus_flag + out.n_mis_br_flag) > 0)
    return out


def flip_over_branches(L: np.ndarray, scale: np.ndarray, limit: float) -> float:
    z = (L - limit) / scale
    return float(z[z > 0].max()) if (z > 0).any() else float((-z).min())


def newton_scores(gname: str, split: str, bus: pd.DataFrame, br: pd.DataFrame, limit: float, n_warm: int) -> tuple:
    scen = np.unique(bus.scenario.to_numpy())
    raw = newton_check.raw_dir(gname, split)
    flt = [("scenario", "in", scen.tolist())]
    bd = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
    brd = pd.read_parquet(raw / "branch_data.parquet", filters=flt)
    brd = brd.sort_values(["scenario", "idx"] if "idx" in brd else ["scenario"], kind="stable")
    bd_g, brd_g = dict(tuple(bd.groupby("scenario"))), dict(tuple(brd.groupby("scenario")))
    bus_g = dict(tuple(bus.sort_values(["scenario", "bus"]).groupby("scenario")))
    br_g = dict(tuple(br.sort_values(["scenario", "branch"]).groupby("scenario")))
    rows, warm = [], []
    for i, s in enumerate(scen):
        b_raw, g, r = bd_g[s].sort_values("bus"), bus_g[s], br_g[s]
        net = newton.network(b_raw, brd_g[s])
        vm_raw, va_raw = g.vm_pred.to_numpy(np.float64), np.deg2rad(g.va_pred.to_numpy(np.float64))
        vm_hat, va_hat = newton.fix_known(net, vm_raw, va_raw)
        dvm, dva, _, _ = newton.newton_step(net, vm_hat, va_hat)
        L_hat = newton.loading(net, vm_raw, va_raw)
        L_n = newton.loading(net, vm_hat + dvm, va_hat + dva)
        on = r.in_service.to_numpy().astype(bool) & np.isfinite(L_n)
        dl = np.abs(L_n - L_hat)[on]
        true = r.loading_true.to_numpy()[on]
        rows.append({"scenario": int(s), "max_loading_newton": float(L_n[on].max()),
                     "max_err_newton": float(np.nanmax(np.abs(L_n[on] - true))), "newton_max_dl": float(dl.max()),
                     "z_newton_flip": flip_over_branches(L_n[on], np.maximum(dl, SCALE_FLOOR), limit)})
        if i < n_warm:
            for name, (v0, a0) in {"flat": newton.flat_start(net), "genco": (vm_hat, va_hat),
                                   "genco_step": (vm_hat + dvm, va_hat + dva)}.items():
                res = newton.solve(net, v0, a0)
                warm.append({"split": split, "scenario": int(s), "k": int(g.k.iloc[0]), "start": name,
                             "iterations": res["iterations"], "converged": bool(res["converged"])})
    return pd.DataFrame(rows), warm


def run_split(gname: str, split: str, thr: dict, tau: dict, limit: float, n_warm: int, skip_newton: bool):
    bus = io.read_columns("bus", gname, split, BUS_COLS)
    br = io.read_columns("branch", gname, split, BR_COLS)
    out = table_scores(bus, br, thr, tau, limit).reset_index()
    if not skip_newton:
        nt, warm = newton_scores(gname, split, bus, br, limit, n_warm)
        out = out.merge(nt, on="scenario", how="left")
    else:
        warm = []
    return out.assign(split=split), warm


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--warm-per-split", type=int, default=50)
    ap.add_argument("--skip-newton", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute splits that already have a table")
    ap.add_argument("--splits", nargs="*", help="only these splits (default: all of the grid)")
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    thr, tau = thresholds(gname)
    limit = cfg.labels.decision_overload_threshold
    log.info("%s thresholds %s, tau %s", gname, thr, tau)
    warm_all = []
    for split in a.splits or grid_splits(gname):
        path = io.results_dir(gname, split) / "scenario_checks.parquet"
        if path.exists() and not a.force and (a.skip_newton or "z_newton_flip" in pd.read_parquet(path).columns):
            log.info("%s: exists, skipped", split)
            continue
        t0 = time.perf_counter()
        d, warm = run_split(gname, split, thr, tau, limit, a.warm_per_split, a.skip_newton)
        d.to_parquet(path, index=False)
        warm_all += warm
        log.info("%s: %d scenarios in %.0f s", split, len(d), time.perf_counter() - t0)
    if warm_all:
        wpath = io.results_dir(gname) / "newton_warm_corrected.csv"
        new = pd.DataFrame(warm_all)
        if wpath.exists():
            old = pd.read_csv(wpath)
            new = pd.concat([old[~old.split.isin(new.split.unique())], new], ignore_index=True)
        new.to_csv(wpath, index=False)

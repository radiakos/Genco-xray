"""Label-free last-step check: AUROC, AUPRC and a fixed-threshold rule for the last-step change, per system.
Writes results/<grid>/last_step_check.csv and summary.last_step_check.<grid>. Run after error_check.py.

    python scripts/last_step_check.py --grid texas2k
    python scripts/last_step_check.py --grid case118

Protocol (same rows, labels and folds as error_check.py, so the AUROCs reproduce its single-feature rows):
* Rows: the seeded PQ-bus and in-service branch samples of 04_trust_eval.load_samples; label = error > the headline
  tau of summary.e2.<grid> (bus |dVm|, branch |dloading|).
* Scores: last-step voltage change (neighbourhood max and own bus), last-step loading change (std of the last four
  loadings), final mismatch, and the fitted error predictor (logistic, out-of-fold by load scenario, fitted on outage
  counts inside the training range). Only the error predictor needs labels.
* Fixed threshold: per score, the largest c whose rule "score >= c" flags 90 % of the positives inside the training
  range; then kept fixed for every larger outage count. CIs: resampling by load scenario.
"""

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gencoxray import cli, io, last_step_check, trust
from gencoxray.config import grid_splits, k_fit, load_config

log = logging.getLogger("last_step_check")
RECALL = 0.9
N_BOOT_K = 200  # per-outage-count rows
_spec = importlib.util.spec_from_file_location("error_check", Path(__file__).with_name("error_check.py"))
error_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(error_check)

# level -> (error column, feature list for the error predictor, {score name: column})
LEVELS = {
    "bus": ("err_vm", error_check.BUS_FEATS, {"last_step_vm_nbmax": "nbmax_f_drift_vm",
                                              "last_step_vm_own": "f_drift_vm", "final_mismatch": "f_final"}),
    "branch": ("err_loading", error_check.BR_FEATS, {"last_step_loading": "g_drift_tail_std",
                                                     "final_mismatch": "g_endpoint_max_f_final"}),
}


def run_level(level: str, D: pd.DataFrame, tau: float, k_max: int, n_boot: int, seed: int) -> pd.DataFrame:
    err_col, feats, cols = LEVELS[level]
    k, g = D.k.to_numpy(), D.load_scenario.to_numpy()
    err = D[err_col].to_numpy(np.float64)
    y = err > tau
    fit = k <= k_max
    scores = {name: D[c].to_numpy(np.float64) for name, c in cols.items()}
    comb = trust.GroupFoldCombiner("logistic", seed=seed).fit(D.loc[fit, feats].astype(np.float64), y[fit], g[fit],
                                                               final=False)
    scores["error_predictor"] = comb.predict(D[feats].astype(np.float64), g)
    thr = {s: last_step_check.recall_threshold(v[fit], y[fit], RECALL) for s, v in scores.items()}
    rows = []
    for setname, m in error_check.eval_sets(k, k_max).items():
        nb = n_boot if setname in ("in_dist", "shift_pooled") else N_BOOT_K
        ev = trust.evaluate_scores({s: v[m] for s, v in scores.items()}, err[m], tau, g[m], n_boot=nb, seed=seed)
        for r in ev.itertuples(index=False):
            tm = last_step_check.threshold_metrics(scores[r.score][m], y[m], thr[r.score], g[m], n_boot=nb, seed=seed)
            rows.append(r._asdict() | {f"thr_{c}": v for c, v in tm.items() if c not in ("n", "n_pos")}
                        | {"threshold": thr[r.score], "set": setname})
        log.info("%s %s done", level, setname)
    return pd.DataFrame(rows).assign(level=level, tau=tau)


def summarise(tab: pd.DataFrame) -> dict:
    keep = ["auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "n", "n_pos", "prevalence", "threshold",
            "thr_recall", "thr_recall_lo", "thr_recall_hi", "thr_precision", "thr_precision_lo", "thr_precision_hi",
            "thr_flag_rate"]
    out = {}
    for level, d in tab.groupby("level"):
        out[level] = {"tau": float(d.tau.iloc[0])}
        for s, ds in d.groupby("set"):
            out[level][s] = {r.score: {c: None if pd.isna(getattr(r, c)) else float(getattr(r, c)) for c in keep}
                             for r in ds.itertuples()}
    return out


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--cap-indist", type=int, default=500_000, help="as in error_check.py")
    ap.add_argument("--cap-shift", type=int, default=150_000, help="as in error_check.py")
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    k_max = k_fit(gname)
    e2 = io.read_summary()["e2"][gname]
    B, R = error_check.load_samples(gname, grid_splits(gname), k_max, (a.cap_indist, a.cap_shift), cfg.seed)
    tab = pd.concat([run_level("bus", B, e2["bus"]["headline_tau"], k_max, cfg.trust.bootstrap, cfg.seed),
                     run_level("branch", R, e2["branch"]["headline_tau"], k_max, cfg.trust.bootstrap, cfg.seed)],
                    ignore_index=True)
    tab.to_csv(io.results_dir(gname) / "last_step_check.csv", index=False)
    summ = summarise(tab) | {"recall_target": RECALL, "k_fit_max": k_max, "threshold_rule": "score >= c"}
    io.update_summary(f"last_step_check.{gname}", summ)
    for level in ("bus", "branch"):
        for s in ("in_dist", "shift_pooled"):
            for sc, v in summ[level].get(s, {}).items():
                log.info("%s %s %-20s AUROC %.3f AUPRC %.3f | c=%.3g recall %.3f precision %.3f flagged %.4f",
                         level, s, sc, v["auroc"], v["auprc"], v["threshold"], v["thr_recall"], v["thr_precision"],
                         v["thr_flag_rate"])

"""Stress and accuracy map: per-branch stress and accuracy score, groups, and the disagreement test. Writes
summary.e6.<grid> and results/<grid>/joint.parquet. Also caches the grid layout in static_bus.x/y (viz.grid_layout;
Texas ~2 min once).

    python scripts/stress_and_accuracy_map.py --grid texas2k
    python scripts/stress_and_accuracy_map.py --grid synthetic

Protocol (docs/experiments.md, "Stress and accuracy map"):
* Branch stress = the GENCO stress map over all scenarios (results/<grid>/e5_branch_stress.csv, subset "all").
  Primary metric: CVaR_95 of loading (tail loading, defined for every branch); P(overload) reported as secondary.
* Branch accuracy score ("trust") = -mean branch_risk (out-of-fold log-odds of the error predictor) over in-service
  scenarios with L_pred > joint.stressed_loading_for_trust (0.9), falling back to all in-service scenarios.
* Groups: stressed = top (1 - joint.stress_quantile) of stress; error-prone ("untrusted") = bottom
  joint.trust_quantile of the accuracy score.
* Test (summary key h4): disagreement = |stress_GENCO - stress_AC| per branch; Mann-Whitney U one-sided,
  stressed_untrusted > stressed_trusted; effect size rank-biserial. Pass: p < 0.05.
"""

import logging

import numpy as np
import pandas as pd

from gencoxray import cli, io, joint, ranking
from gencoxray.config import grid_splits, load_config
from gencoxray.viz import grid_layout

log = logging.getLogger("stress_and_accuracy_map")


def ensure_layout(gname: str, seed: int) -> np.ndarray:
    sb = io.read_table("static_bus", gname)
    if sb[["x", "y"]].notna().all().all():
        return sb[["x", "y"]].to_numpy(np.float64)
    sbr = io.read_table("static_branch", gname)
    xy = grid_layout.layout(sb, sbr, seed=seed)
    sb["x"], sb["y"] = xy[:, 0].astype(np.float32), xy[:, 1].astype(np.float32)
    io.write_table(sb, "static_bus", gname)
    log.info("%s: layout computed and cached in static_bus.x/y", gname)
    return xy


def robustness(jt: pd.DataFrame) -> dict:
    """Controls for stress level inside the stressed set: relative disagreement |g - a| / a, the test within stress
    quartiles of the stressed set, and OLS log10(disagreement) ~ stress + untrusted."""
    from scipy.stats import mannwhitneyu

    s = jt[jt.quadrant.astype(str).str.startswith("stressed")].copy()
    s["u"] = (s.quadrant.astype(str) == "stressed_untrusted").to_numpy()
    if s.u.sum() < 3 or (~s.u).sum() < 3:
        return {"note": "too few stressed elements per group"}
    out = {"median_stress_untrusted": float(s[s.u].stress_genco.median()),
           "median_stress_trusted": float(s[~s.u].stress_genco.median())}
    rel = s.disagreement / s.stress_ac.where(s.stress_ac > 0)
    r = mannwhitneyu(rel[s.u].dropna(), rel[~s.u].dropna(), alternative="greater")
    out["relative_disagreement"] = {"median_untrusted": float(rel[s.u].median()), "median_trusted": float(rel[~s.u].median()),
                                    "p_value": float(r.pvalue)}
    s["sbin"] = pd.qcut(s.stress_genco, 4, labels=False, duplicates="drop")
    bins = {}
    for b, d in s.groupby("sbin"):
        if d.u.sum() >= 3 and (~d.u).sum() >= 3:
            rb = mannwhitneyu(d[d.u].disagreement, d[~d.u].disagreement, alternative="greater")
            bins[str(int(b))] = {"n_untrusted": int(d.u.sum()), "n_trusted": int((~d.u).sum()),
                                 "median_untrusted": float(d[d.u].disagreement.median()),
                                 "median_trusted": float(d[~d.u].disagreement.median()), "p_value": float(rb.pvalue)}
    out["within_stress_quartiles"] = bins
    X = np.column_stack([np.ones(len(s)), s.stress_genco, s.u.astype(float)])
    y = np.log10(s.disagreement.to_numpy() + 1e-6)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    se = np.sqrt(e @ e / (len(y) - 3) * np.diag(np.linalg.inv(X.T @ X)))
    out["ols_log10_disagreement"] = {"untrusted_coef": float(beta[2]), "untrusted_se": float(se[2]),
                                     "factor": float(10 ** beta[2]), "stress_coef": float(beta[1])}
    return out


PROXIES = {  # element "risk" proxies (higher = less trust), aggregated exactly like branch_risk
    "final_residual": lambda b: b.g_endpoint_max_f_final,
    "genco_dc_disagreement": lambda b: (b.loading_pred - b.loading_dc).abs(),
    "predicted_loading": lambda b: b.loading_pred,
}


def proxy_h4(gname: str, splits: list[str], jc, st: pd.DataFrame, sbr: pd.DataFrame, seed: int) -> dict:
    """Is the result specific to the error predictor, or would any error proxy give it? Same groups, same
    aggregation (mean over near-limit scenarios, fallback all), same test, on CVaR stress."""
    parts = {p: [] for p in PROXIES}
    for s in splits:
        br = io.read_columns("branch", gname, s,
                             ["branch", "in_service", "loading_pred", "loading_dc", "g_endpoint_max_f_final"])
        for p, fn in PROXIES.items():
            parts[p].append(joint.element_trust_parts(br.assign(branch_risk=fn(br).astype(np.float64)),
                                                      jc.stressed_loading_for_trust))
    maps = stress_maps(st, "cvar")
    trusts = {p: joint.trust_from_parts(pd.concat(v)).trust for p, v in parts.items()}
    rng = np.random.default_rng(seed)
    trusts["random"] = pd.Series(rng.random(len(sbr)), index=sbr.branch.to_numpy())
    out = {}
    for p, t in trusts.items():
        jt = joint.assemble(maps, t, sbr, stress_q=jc.stress_quantile, trust_q=jc.trust_quantile)
        r = joint.h4_test(jt.disagreement, jt.quadrant)
        out[p] = {k: r[k] for k in ("p_value", "rank_biserial", "median_untrusted", "median_trusted", "n_untrusted",
                                     "n_trusted")}
    return out


def stress_maps(st: pd.DataFrame, metric: str) -> dict[str, pd.Series]:
    """Per-branch stress maps {source: Series} for one metric (sources present in the table)."""
    return {src: st[st.source == src].set_index("branch")[metric] for src in joint.STRESS_SOURCES
            if (st.source == src).any()}


def trust_parts(gname: str, splits: list[str], stressed_loading: float) -> pd.DataFrame:
    parts = [joint.element_trust_parts(io.read_columns("branch", gname, s,
                                                       ["branch", "in_service", "loading_pred", "branch_risk"]),
                                       stressed_loading) for s in splits]
    return joint.trust_from_parts(pd.concat(parts))


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    splits = grid_splits(gname)
    jc = cfg.joint
    ensure_layout(gname, cfg.seed)

    st = pd.read_csv(io.results_dir(gname) / "e5_branch_stress.csv")
    st = st[st.subset == "all"]
    tp = trust_parts(gname, splits, jc.stressed_loading_for_trust)
    sbr = io.read_table("static_branch", gname)
    summ = {"protocol": {"stress_quantile": jc.stress_quantile, "trust_quantile": jc.trust_quantile,
                         "stressed_loading_for_trust": jc.stressed_loading_for_trust,
                         "trust": "-mean branch_risk (M4 out-of-fold log-odds)", "primary_metric": "cvar"},
            "trust_from_stressed_scenarios": int(tp.trust_from_stressed.sum()), "n_branches": len(tp)}
    tables = {}
    for metric in ("cvar", "p_overload"):
        jt = joint.assemble(stress_maps(st, metric), tp.trust, sbr, stress_q=jc.stress_quantile, trust_q=jc.trust_quantile)
        jt["n_stressed_scenarios"] = tp.n_stressed.reindex(jt.index).fillna(0).astype(np.int64)
        tables[metric] = jt
        res = {"abs": joint.h4_test(jt.disagreement, jt.quadrant),
               "rank": joint.h4_test(jt.disagreement_rank, jt.quadrant)}
        counts = jt.quadrant.value_counts().reindex(joint.QUADRANTS).fillna(0).astype(int).to_dict()
        med = jt.groupby("quadrant", observed=False).disagreement.median().reindex(joint.QUADRANTS).to_dict()
        summ[metric] = {"h4": res, "h4_pass": bool(res["abs"]["p_value"] < 0.05) if np.isfinite(res["abs"]["p_value"])
                        else None, "quadrant_counts": counts,
                        "median_disagreement_by_quadrant": {k: (None if pd.isna(v) else float(v)) for k, v in med.items()},
                        "spearman_stress_trust": ranking.spearman(jt.stress_genco, jt.trust),
                        "robustness": robustness(jt)}
        log.info("%s %s: stressed error-prone branches disagree more %s | %s | counts %s", gname, metric,
                 summ[metric]["h4_pass"], res["abs"], counts)
    summ["cvar"]["h4_trust_proxies"] = proxy_h4(gname, splits, jc, st, sbr, cfg.seed)
    log.info("%s test with other accuracy scores: %s", gname, summ["cvar"]["h4_trust_proxies"])
    out = tables["cvar"].copy()
    for c in ("stress_genco", "stress_ac", "stress_dc", "quadrant", "disagreement", "disagreement_rank"):
        out[f"{c}_p_overload"] = tables["p_overload"][c]
    out["quadrant"] = out.quadrant.astype(str)
    out["quadrant_p_overload"] = out.quadrant_p_overload.astype(str)
    out.reset_index().to_parquet(io.results_dir(gname) / "joint.parquet", index=False)
    summ["h4_verdict"] = ("H4 supported" if summ["cvar"]["h4_pass"] else "H4 not supported")
    io.update_summary(f"e6.{gname}", summ)
    log.info("stress and accuracy map %s: %s", gname, summ["h4_verdict"])

"""Mismatch by step, error check (inside and beyond the training range, position control) and error bands.
Writes summary.e1/e2/e3.<grid>, results/<grid>/e{1,2,3}_*.csv, and fills BRANCH.branch_risk + SCENARIO.trust_score.

    python scripts/error_check.py --grid case118
    python scripts/error_check.py --grid texas2k
    python scripts/error_check.py --grid synthetic          # after build_features.py --synthetic

Protocol (fixed before looking at any result; see docs/experiments.md, "Mismatch by step", "Error check" and "Error
bands for branch loading"):
* Inside the training range ("in-distribution") = rows with k <= k_fit (max of the grid's ``in_distribution_k``:
  Texas 2, case118 1; synthetic 2). Every error predictor ("combiner") is fitted on these rows only. Beyond the
  training range ("shift") = rows with k > k_fit.
* Folds: a load scenario's fold is a hash of its id (``trust.group_fold``), the same in every split and table. Each
  row is scored by the fold model that never saw its load scenario (out-of-fold in-distribution, and
  load-scenario-disjoint under shift; Texas test_k splits share load scenarios). CIs: cluster bootstrap over load
  scenarios.
* Rows: bus labels on PQ rows (Vm is set at PV/REF), branch labels on in-service rows. Fitting and evaluation use a
  seeded uniform row sample per split (``--cap-indist`` / ``--cap-shift``); ``branch_risk`` is scored on every row.
* Headline threshold: the first value in configs/experiment.yaml (labels.tau_vm_pu / tau_loading) with >= MIN_POS
  positive rows in the in-distribution evaluation set; all thresholds are reported.
* Pass "error predictor beats final mismatch" (summary key h1, per label): dAUROC(combiner - f_final) > 0.03 with
  the 95 % CI excluding 0, in-distribution AND shift-pooled. Judged only at thresholds with >= MIN_POS
  in-distribution positives (otherwise "undetermined"); sets with < 20 positives are flagged ``underpowered`` and get
  no CI.
* Scenario risk (trust_score = scenario accuracy score = -risk), pre-declared candidates: (i) max branch_risk over in-service branches with
  |L_pred - 1| < 0.15 (the decisions that can flip; scenarios without such a branch get the lowest observed risk),
  (ii) q90 and (iii) max of all branch risks, (iv) a scenario-level logistic combiner of (i)-(iii), the mean final
  residual, the near-threshold count and k, fitted on in-distribution scenarios (fold models). The candidate with the
  best in-distribution AUROC for the decision-error label is written; shifted splits never influence the choice.
"""

import logging
import time

import numpy as np
import pandas as pd

from gencoxray import cli, conformal, confounders, io, schemas, trust
from gencoxray.config import SYNTHETIC, grid_splits, k_fit, k_of_split, load_config
from gencoxray.features import residual_magnitude

log = logging.getLogger("error_check")
MIN_POS = 200
NEAR_BAND = 0.15
N_BOOT_SMALL = 200  # per-k sets and single features
E1_KS = {0, 1, 2, 5, 10, 20}  # 0 = split without a k in its name (case118 "test")
CONF_COLS = ["k", "hop_to_outage", "hop_to_pv_ref", "hop_to_ref", "degree"]
BUS_FEATS = list(schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES)
BR_FEATS = list(schemas.BRANCH_FEATURES)
MAIN = ["combiner", "combiner_hgb", "combiner_traj_conf", "combiner_resid", "conf_only", "f_final", "k",
        "hop_to_outage", "hop_to_pv_ref", "dc_disagreement", "random"]


# ----------------------------------------------------------------------------- data
def _sample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    return (df if len(df) <= cap else df.sample(n=cap, random_state=seed).sort_index()).copy()


def _bus_hop_matrix(bus_cols: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(sorted scenario ids, [S, B] hop_to_outage) from BUS rows of one split."""
    b = bus_cols.sort_values(["scenario", "bus"])
    sc = np.unique(b.scenario.to_numpy())
    return sc, b.hop_to_outage.to_numpy(np.float64).reshape(len(sc), -1)


def load_samples(gname: str, splits: list[str], k_max: int, caps: tuple[int, int], seed: int):
    """Seeded row samples of PQ bus rows and in-service branch rows, with confounders joined."""
    st = io.read_table("static_bus", gname).set_index("bus")
    sbr = io.read_table("static_branch", gname).sort_values("branch")
    f_bus, t_bus, hp = sbr.from_bus.to_numpy(), sbr.to_bus.to_numpy(), st.hop_to_pv_ref.to_numpy()
    bcols = ["scenario", "load_scenario", "bus", "k", "bus_type", "err_vm", "hop_to_outage"] + BUS_FEATS
    rcols = ["scenario", "load_scenario", "branch", "k", "in_service", "loading_pred", "loading_true", "loading_dc",
             "err_loading"] + BR_FEATS
    buses, branches = [], []
    for i, s in enumerate(splits):
        b = io.read_columns("bus", gname, s, bcols)
        sc_ids, hop = _bus_hop_matrix(b[["scenario", "bus", "hop_to_outage"]])
        cap = caps[0] if b.k.median() <= k_max else caps[1]
        pq = _sample(b[b.bus_type == "PQ"], cap, seed + i)
        for c in ("hop_to_pv_ref", "hop_to_ref", "degree"):
            pq[c] = st[c].reindex(pq.bus).to_numpy(np.float32)
        buses.append(pq.assign(split=s))
        r = io.read_columns("branch", gname, s, rcols)
        r = _sample(r[r.in_service], cap, seed + 1000 + i)
        si = np.searchsorted(sc_ids, r.scenario.to_numpy())
        f, t = f_bus[r.branch], t_bus[r.branch]
        r["hop_to_outage"] = np.fmin(hop[si, f], hop[si, t]).astype(np.float32)
        r["hop_to_pv_ref"] = np.maximum(hp[f], hp[t]).astype(np.float32)
        r["dc_disagreement"] = (r.loading_pred - r.loading_dc).abs().astype(np.float32)
        branches.append(r.assign(split=s))
        log.info("%s/%s: sampled %d PQ bus rows, %d branch rows", gname, s, len(pq), len(r))
    return pd.concat(buses, ignore_index=True), pd.concat(branches, ignore_index=True)


# ----------------------------------------------------------------------------- evaluation helpers
def eval_sets(k: np.ndarray, k_max: int) -> dict[str, np.ndarray]:
    sets = {"in_dist": k <= k_max, "shift_pooled": k > k_max}
    for kk in np.unique(k):
        sets[f"k{int(kk):02d}"] = k == kk
    return {n: m for n, m in sets.items() if m.any()}


def headline_sets(k: np.ndarray, k_max: int) -> dict[str, np.ndarray]:
    """The two sets every hypothesis is judged on (non-empty ones only)."""
    return {n: m for n, m in (("in_dist", k <= k_max), ("shift_pooled", k > k_max)) if m.any()}


def evaluate(scores: dict[str, np.ndarray], err: np.ndarray, tau: float, groups: np.ndarray, k: np.ndarray,
             k_max: int, n_boot: int, seed: int) -> pd.DataFrame:
    """All scores on in_dist (full CIs for MAIN, small for single features), shift_pooled and per k (MAIN only)."""
    out = []
    for name, m in eval_sets(k, k_max).items():
        main = {s: v[m] for s, v in scores.items() if s in MAIN}
        nb = n_boot if name in ("in_dist", "shift_pooled") else N_BOOT_SMALL
        out.append(trust.evaluate_scores(main, err[m], tau, groups[m], n_boot=nb, seed=seed).assign(set=name))
        if name == "in_dist":
            minor = {s: v[m] for s, v in scores.items() if s not in MAIN}
            if minor:
                out.append(trust.evaluate_scores(minor, err[m], tau, groups[m], n_boot=N_BOOT_SMALL,
                                                 seed=seed).assign(set=name))
    return pd.concat(out, ignore_index=True)


def deltas(scores: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, k: np.ndarray, k_max: int,
           n_boot: int, seed: int, base: str = "f_final") -> dict:
    res = {}
    for name, m in headline_sets(k, k_max).items():
        res[name] = {}
        for other in (base, "conf_only"):
            if other in scores:
                d, lo, hi = trust.delta_auroc_ci(scores["combiner"][m], scores[other][m], y[m], groups[m],
                                                 n_boot=n_boot, seed=seed)
                res[name][f"combiner_minus_{other}"] = {"delta": d, "lo": lo, "hi": hi}
    return res


def h1_pass(d: dict, base: str = "f_final") -> bool | None:
    key = f"combiner_minus_{base}"
    parts = [d[s][key] for s in ("in_dist", "shift_pooled") if s in d]
    if not parts or any(not np.isfinite(p["lo"]) for p in parts):
        return None
    return bool(all(p["delta"] > 0.03 and p["lo"] > 0 for p in parts))


def headline_tau(taus: list[float], err: np.ndarray, indist: np.ndarray) -> tuple[float, dict]:
    counts = {str(t): int((err > t)[indist].sum()) for t in taus}
    for t in taus:
        if counts[str(t)] >= MIN_POS:
            return t, counts
    return taus[int(np.argmax([counts[str(t)] for t in taus]))], counts


def combiners(X: pd.DataFrame, C: pd.DataFrame, y: np.ndarray, groups: np.ndarray, k: np.ndarray, k_max: int,
              seed: int, full: bool) -> dict[str, np.ndarray]:
    """Fold-model scores for every row. ``full`` adds the HGB check and the confounder-residualised combiner."""
    fit = k <= k_max
    t0 = time.perf_counter()

    def run(F: pd.DataFrame, kind: str = "logistic") -> np.ndarray:
        comb = trust.GroupFoldCombiner(kind, seed=seed).fit(F[fit], y[fit], groups[fit], final=False)
        return comb.predict(F, groups)

    out = {"combiner": run(X), "conf_only": run(C), "combiner_traj_conf": run(pd.concat([X, C], axis=1))}
    if full:
        out["combiner_hgb"] = run(X, "hgb")
        out["combiner_resid"] = run(confounders.residualize(X, C, fit))
    log.info("combiners fitted in %.0f s (%d fitting rows, full=%s)", time.perf_counter() - t0, fit.sum(), full)
    return out


def _nullable(v) -> float | None:
    return None if pd.isna(v) else float(v)


def summarise(df: pd.DataFrame, sets=("in_dist", "shift_pooled")) -> dict:
    keep = ["auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "spearman", "prevalence", "n", "n_pos",
            "n_groups", "underpowered"]
    d = df[df.set.isin(sets)]
    return {s: {r.score: {c: _nullable(getattr(r, c)) for c in keep} for r in d[d.set == s].itertuples()}
            for s in sets if (d.set == s).any()}


def best_single(df: pd.DataFrame, feats: list[str]) -> dict:
    d = df[(df.set == "in_dist") & df.score.isin(feats)].sort_values("auroc", ascending=False)
    return {"score": d.score.iloc[0], "auroc": float(d.auroc.iloc[0])} if len(d) else {}


def _hop_bin(h: np.ndarray) -> np.ndarray:
    return np.select([h <= 0, h <= 1, h <= 2, h <= 4], ["0", "1", "2", "3-4"], "5+").astype(object)


# ----------------------------------------------------------------------------- error check, bus / branch (e2)
def e2_level(level: str, D: pd.DataFrame, feats: list[str], err_col: str, taus: list[float], k_max: int,
             cfg, seed: int, f_final_col: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Returns (long metrics table, summary dict, stratified table, risk-coverage table) for one label level."""
    k = D.k.to_numpy()
    g = D.load_scenario.to_numpy()
    err = D[err_col].to_numpy(np.float64)
    X = D[feats].astype(np.float64)
    C = D[[c for c in CONF_COLS if c in D]].astype(np.float64)
    indist = k <= k_max
    tau_h, counts = headline_tau(taus, err, indist)
    log.info("%s: positives in-distribution per tau %s -> headline tau %s", level, counts, tau_h)
    rng = np.random.default_rng(seed)
    tables, summ = [], {"headline_tau": tau_h, "n_pos_in_dist": counts, "min_pos_rule": MIN_POS}
    strat, rc = pd.DataFrame(), pd.DataFrame()
    for tau in taus:
        y = err > tau
        full = tau == tau_h
        sc = combiners(X, C, y, g, k, k_max, seed, full)
        sc |= {f: D[f].to_numpy(np.float64) for f in feats}
        sc["f_final"] = D[f_final_col].to_numpy(np.float64)
        sc["k"] = k.astype(np.float64)
        sc["hop_to_outage"] = -D.hop_to_outage.to_numpy(np.float64)  # lower hop = riskier
        sc["hop_to_pv_ref"] = D.hop_to_pv_ref.to_numpy(np.float64)
        if "dc_disagreement" in D:
            sc["dc_disagreement"] = D.dc_disagreement.to_numpy(np.float64)
        sc["random"] = rng.random(len(D))
        tab = evaluate(sc, err, tau, g, k, k_max, cfg.trust.bootstrap, seed).assign(level=level, tau=tau)
        dl = deltas(sc, y, g, k, k_max, cfg.trust.bootstrap, seed)
        powered = counts[str(tau)] >= MIN_POS
        summ[f"tau_{tau}"] = {"metrics": summarise(tab), "delta_auroc": dl, "powered": powered,
                              "h1_pass": h1_pass(dl) if powered else None,
                              "best_single_feature_in_dist": best_single(tab, feats)}
        tables.append(tab)
        if full:
            summ["stratified"], strat = stratified(level, D, sc, y, k, k_max)
            summ["risk_coverage_aurc"], rc = risk_cov(level, sc, y, indist)
        log.info("%s tau=%s: error predictor beats final mismatch %s; %s", level, tau, summ[f"tau_{tau}"]["h1_pass"], dl)
    summ["h1_pass_headline"] = summ[f"tau_{tau_h}"]["h1_pass"]
    return pd.concat(tables, ignore_index=True), summ, strat, rc


def stratified(level: str, D: pd.DataFrame, sc: dict, y: np.ndarray, k: np.ndarray,
               k_max: int) -> tuple[dict, pd.DataFrame]:
    """Position control: AUROC of combiner vs f_final within hop-to-outage and hop-to-PV/REF strata."""
    res, tables = {}, []
    strata = {"hop_to_outage": _hop_bin(D.hop_to_outage.to_numpy()),
              "hop_to_pv_ref": _hop_bin(D.hop_to_pv_ref.to_numpy())}
    for setname, m in headline_sets(k, k_max).items():
        for sname, sv in strata.items():
            t = pd.concat([confounders.stratified_auroc(sc[score][m], y[m], sv[m], min_pos=20).assign(score=score)
                           for score in ("combiner", "f_final", "combiner_resid") if score in sc])
            t = t.assign(set=setname, stratum_by=sname, level=level)
            tables.append(t)
            w = t.pivot_table(index="stratum", columns="score", values="auroc")
            res[f"{setname}.{sname}"] = {
                "strata_with_auroc": int(w.combiner.notna().sum()),
                "combiner_beats_f_final_in": int((w.combiner > w.f_final).sum()),
                "by_stratum": {str(i): {c: _nullable(v) for c, v in row.items()} for i, row in w.iterrows()},
            }
    return res, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def risk_cov(level: str, sc: dict, y: np.ndarray, indist: np.ndarray) -> tuple[dict, pd.DataFrame]:
    out, tables = {}, []
    for score in ("combiner", "f_final", "conf_only", "random"):
        rc = trust.risk_coverage(sc[score][indist], y[indist])
        tables.append(rc.assign(score=score, level=level))
        out[score] = rc.attrs["aurc"]
    return out, pd.concat(tables, ignore_index=True)


# ----------------------------------------------------------------------------- write back + scenario level
def write_back(gname: str, splits: list[str], comb: trust.GroupFoldCombiner) -> pd.DataFrame:
    """branch_risk on every BRANCH row (fold model of its load scenario); per-scenario aggregates."""
    rows = []
    for s in splits:
        br = io.read_table("branch", gname, s)
        risk = comb.predict(br[BR_FEATS].astype(np.float64), br.load_scenario.to_numpy())
        risk[~br.in_service.to_numpy()] = np.nan
        br["branch_risk"] = risk.astype(np.float32)
        io.write_table(br, "branch", gname, s)
        near = br.in_service & ((br.loading_pred - 1.0).abs() < NEAR_BAND)
        r, sid = br.branch_risk.to_numpy(), br.scenario.to_numpy()
        agg = pd.DataFrame({
            "risk_near_max": br.branch_risk.where(near).groupby(br.scenario).max(),
            "risk_q90": trust.scenario_risk(r, sid, "q90"),
            "risk_max": trust.scenario_risk(r, sid, "max"),
            "max_err_loading": br.err_loading.groupby(br.scenario).max(),
            "n_near": near.groupby(br.scenario).sum(),
        })
        rows.append(agg.assign(split=s))
        log.info("%s/%s: branch_risk written (%d rows)", gname, s, len(br))
    agg = pd.concat(rows)
    agg.index.name = "scenario"
    floor = float(np.nanmin(agg.risk_near_max)) if agg.risk_near_max.notna().any() else 0.0
    agg["risk_near_filled"] = agg.risk_near_max.fillna(floor)
    return agg


def write_trust(gname: str, splits: list[str], sc: pd.DataFrame, risk: np.ndarray) -> None:
    """SCENARIO.trust_score = -(selected scenario risk)."""
    r = pd.Series(np.asarray(risk, np.float64), index=pd.MultiIndex.from_arrays([sc.split, sc.scenario]))
    for s in splits:
        t = io.read_table("scenario", gname, s)
        t["trust_score"] = trust.risk_to_trust(r.loc[s].reindex(t.scenario).to_numpy())
        io.write_table(t, "scenario", gname, s)


def e2_scenario(gname: str, splits: list[str], agg: pd.DataFrame, k_max: int, cfg, seed: int):
    sc = io.read_table("scenario", gname, splits=splits)
    sc = sc.merge(agg.reset_index(), on=["scenario", "split"], how="left", validate="one_to_one")
    dec = (((sc.max_loading_pred > 1) != (sc.max_loading_true > 1)) | (sc.max_err_loading > 0.05)).to_numpy()
    err = dec.astype(np.float64)
    rng = np.random.default_rng(seed)
    k, g = sc.k.to_numpy(), sc.load_scenario.to_numpy()
    feats = pd.DataFrame({"risk_near_max": sc.risk_near_max, "risk_q90": sc.risk_q90, "risk_max": sc.risk_max,
                          "residual_mean": sc.res_mean_final_mva, "n_near": sc.n_near, "k": sc.k}).astype(np.float64)
    fit = k <= k_max
    sc_comb = trust.GroupFoldCombiner("logistic", n_splits=cfg.trust.cv_folds, seed=seed).fit(
        feats[fit], dec[fit], g[fit], final=False).predict(feats, g)
    # pre-declared candidates; the one written as trust_score is chosen on in-distribution AUROC only
    cands = {"near_threshold_max": sc.risk_near_filled.to_numpy(np.float64), "branch_risk_q90": sc.risk_q90.to_numpy(),
             "branch_risk_max": sc.risk_max.to_numpy(), "scenario_combiner": sc_comb}
    sel_auc = {c: trust.auroc(v[fit], dec[fit]) for c, v in cands.items()}
    # single-class in-distribution label (AUROC undefined everywhere) -> the spec default, near_threshold_max
    chosen = max(sel_auc, key=lambda c: -np.inf if np.isnan(sel_auc[c]) else sel_auc[c])
    log.info("scenario risk candidates, in-distribution AUROC %s -> %s", sel_auc, chosen)
    scores = {"trust": cands[chosen]} | {f"cand_{c}": v for c, v in cands.items()} | {
        "residual_mean": sc.res_mean_final_mva.to_numpy(np.float64), "k": sc.k.to_numpy(np.float64),
        "n_near_threshold": sc.n_near.to_numpy(np.float64), "random": rng.random(len(sc))}
    out = []
    for name, m in eval_sets(k, k_max).items():
        nb = cfg.trust.bootstrap if name in ("in_dist", "shift_pooled") else N_BOOT_SMALL
        out.append(trust.evaluate_scores({c: v[m] for c, v in scores.items()}, err[m], 0.5, g[m], n_boot=nb,
                                         seed=seed).assign(set=name))
    tab = pd.concat(out, ignore_index=True).assign(level="scenario", tau=np.nan)
    d = {}
    for name, m in headline_sets(k, k_max).items():
        dd, lo, hi = trust.delta_auroc_ci(scores["trust"][m], scores["residual_mean"][m], err[m] > 0.5, g[m],
                                          n_boot=cfg.trust.bootstrap, seed=seed)
        d[name] = {"trust_minus_residual_mean": {"delta": dd, "lo": lo, "hi": hi}}
    summ = {"label": "decision error: (max L_pred > 1) != (max L_true > 1) or max err_loading > 0.05",
            "trust_score_source": chosen, "selection_in_dist_auroc": sel_auc, "metrics": summarise(tab),
            "delta_auroc": d, "near_band": NEAR_BAND}
    write_trust(gname, splits, sc, cands[chosen])
    return tab, summ


# ----------------------------------------------------------------------------- error bands (e3)
def e3(R: pd.DataFrame, risk: np.ndarray, k_max: int, cfg, seed: int) -> tuple[pd.DataFrame, dict]:
    """Branch-loading error bands: fixed per branch vs scaled by expected error; calibrated on k <= k_max."""
    alpha = cfg.conformal.alpha
    g = R.load_scenario.to_numpy()
    k = R.k.to_numpy()
    half = trust.group_fold(g, 2, seed + 101)
    cal = (k <= k_max) & (half == 0)
    test = half == 1  # load-scenario-disjoint from the calibration rows, any k
    cal_a = cal & (trust.group_fold(g, 2, seed + 202) == 0)
    cal_b = cal & ~cal_a
    y, yhat = R.loading_true.to_numpy(np.float64), R.loading_pred.to_numpy(np.float64)
    resid = np.abs(y - yhat)
    ok = np.isfinite(resid) & np.isfinite(risk)
    cal, cal_a, cal_b, test = cal & ok, cal_a & ok, cal_b & ok, test & ok

    q_static = conformal.static_stratified(resid[cal], R.loc[cal, ["branch"]], alpha)
    q_global = conformal.conformal_quantile(resid[cal], alpha)
    hw = {"static_branch": R.branch.map(q_static).fillna(q_global).to_numpy(np.float64),
          "static_global": np.full(len(R), q_global)}
    for name, fn in (("adaptive_isotonic", conformal.difficulty_scale_fn(risk[cal_a], resid[cal_a])),
                     ("adaptive_exp", conformal.difficulty_scale_fn(risk[cal_a]))):
        hw[name] = conformal.normalized(resid[cal_b], fn(risk[cal_b]), alpha) * fn(risk)
    kb = np.full(len(R), "", object)
    for lo_, hi_ in cfg.conformal.k_bins:
        kb[(k >= lo_) & (k <= hi_)] = f"{lo_}-{hi_}"
    groupings = {"k": k, "k_bin": kb, "in_dist": np.where(k <= k_max, "in_dist", "shift")}
    tab = pd.concat([conformal.evaluate_intervals(y[test], yhat[test], h[test], 1.0, by=by[test])
                     .rename(columns={"by": "group"}).assign(method=name, by=by_name)
                     for name, h in hw.items() for by_name, by in groupings.items()], ignore_index=True)
    ind = tab[(tab.by == "in_dist") & (tab.group == "in_dist")].set_index("method")
    st, ad = ind.loc["static_branch"], ind.loc["adaptive_isotonic"]
    on_target = lambda r: abs(r.coverage - (1 - alpha)) <= 0.02
    summ = {
        "alpha": alpha, "n_cal": int(cal.sum()), "n_test": int(test.sum()),
        "in_dist": {m: {c: float(ind.loc[m, c]) for c in ("coverage", "mean_width", "frac_straddle", "n")}
                    for m in ind.index},
        "h5_pass": bool(ad.mean_width < st.mean_width and on_target(ad) and on_target(st)),
        "coverage_by_k": {m: {str(r.group): float(r.coverage) for r in tab[(tab.by == "k") & (tab.method == m)].itertuples()}
                          for m in hw},
    }
    return tab, summ


# ----------------------------------------------------------------------------- mismatch by step (e1)
def e1(gname: str, splits: list[str], n_scen: int) -> tuple[pd.DataFrame, dict]:
    """Median / IQR of the bus mismatch r_t per step, by bus type, hop-to-PV/REF bin, hop-to-outage bin and k."""
    st = io.read_table("static_bus", gname).set_index("bus")
    chosen = [s for s in splits if gname == SYNTHETIC or k_of_split(s) in E1_KS]
    R, meta = [], []
    for s in chosen:
        bus = io.read_columns("bus", gname, s,
                              ["scenario", "bus", "k", "bus_type", "struct_p", "struct_q", "hop_to_outage"])
        got = 0
        for sh in io.iter_traces(gname, s):
            order = np.argsort(sh["scenario"])
            S, T, B = sh["res_p"].shape
            rows = bus[bus.scenario.isin(sh["scenario"][order])].sort_values(["scenario", "bus"])
            sp = rows.struct_p.to_numpy().reshape(S, B)
            sq = rows.struct_q.to_numpy().reshape(S, B)
            r = residual_magnitude(sh["res_p"][order], sh["res_q"][order], sp[:, None, :], sq[:, None, :])
            R.append(r.transpose(0, 2, 1).reshape(S * B, T).astype(np.float32))
            meta.append(rows[["k", "bus_type", "hop_to_outage"]].assign(
                hop_to_pv_ref=st.hop_to_pv_ref.reindex(rows.bus).to_numpy()))
            got += S
            if got >= n_scen:
                break
    R = np.concatenate(R)
    M = pd.concat(meta, ignore_index=True)
    ok = np.isfinite(R).all(1)
    R, M = R[ok], M[ok].reset_index(drop=True)
    groupings = {"bus_type": M.bus_type.astype(str).to_numpy(), "hop_to_pv_ref": _hop_bin(M.hop_to_pv_ref.to_numpy()),
                 "hop_to_outage": _hop_bin(M.hop_to_outage.to_numpy()), "k": M.k.astype(int).astype(str).to_numpy()}
    rows, summ = [], {}
    for grouping, gv in groupings.items():
        summ[grouping] = {}
        for val in sorted(set(gv)):
            sel = R[gv == val]
            q = np.quantile(sel, [0.25, 0.5, 0.75], axis=0)
            rows += [{"grouping": grouping, "group": val, "step": t + 1, "q25": q[0, t], "median": q[1, t],
                      "q75": q[2, t], "n": len(sel)} for t in range(R.shape[1])]
            summ[grouping][val] = {"n": len(sel), "median_step1": float(q[1, 0]), "median_final": float(q[1, -1]),
                                   "q75_final": float(q[2, -1]),
                                   "median_contraction": float(np.median((sel[:, -1] + 1e-6) / (sel[:, 0] + 1e-6))),
                                   "frac_stalled_last_step": float(np.mean(sel[:, -1] > 0.95 * sel[:, -2]))}
    return pd.DataFrame(rows), {"splits": chosen, "scenarios_per_split": n_scen, "units": "MVA", **summ}


# ----------------------------------------------------------------------------- main
def run_e2_e3(gname: str, splits: list[str], k_max: int, cfg, args) -> None:
    seed, out_dir = cfg.seed, io.results_dir(gname)
    B, R = load_samples(gname, splits, k_max, (args.cap_indist, args.cap_shift), seed)
    protocol = {"k_fit_max": k_max, "rows_bus": len(B), "rows_branch": len(R), "cap_indist": args.cap_indist,
                "cap_shift": args.cap_shift, "n_boot": cfg.trust.bootstrap, "n_boot_per_k_and_single": N_BOOT_SMALL,
                "folds": cfg.trust.cv_folds, "group_by": "load_scenario", "seed": seed,
                "n_load_scenarios": int(B.load_scenario.nunique())}
    tb, sb, strat_b, rc_b = e2_level("bus", B, BUS_FEATS, "err_vm", list(cfg.labels.tau_vm_pu), k_max, cfg, seed,
                                     "f_final")
    tr, sr, strat_r, rc_r = e2_level("branch", R, BR_FEATS, "err_loading", list(cfg.labels.tau_loading), k_max, cfg,
                                     seed, "g_endpoint_max_f_final")
    # deployable branch combiner (headline tau_L, fitted on in-distribution sample rows) -> every BRANCH row
    tau_l = sr["headline_tau"]
    yl = R.err_loading.to_numpy() > tau_l
    fit = R.k.to_numpy() <= k_max
    comb = trust.GroupFoldCombiner("logistic", n_splits=cfg.trust.cv_folds, seed=seed).fit(
        R.loc[fit, BR_FEATS].astype(np.float64), yl[fit], R.load_scenario.to_numpy()[fit], final=False)
    agg = write_back(gname, splits, comb)
    ts, ss = e2_scenario(gname, splits, agg, k_max, cfg, seed)
    pd.concat([tb, tr, ts], ignore_index=True).to_csv(out_dir / "e2_metrics.csv", index=False)
    for name, parts in (("e2_stratified", [strat_b, strat_r]), ("e2_risk_coverage", [rc_b, rc_r])):
        parts = [p for p in parts if len(p)]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out_dir / f"{name}.csv", index=False)
    hb, hr = sb["h1_pass_headline"], sr["h1_pass_headline"]
    verdict = ("H1 supported" if hb and hr else "H1 partly supported" if (hb or hr) else
               "H1 not supported" if (hb is False and hr is False) else "H1 undetermined")
    io.update_summary(f"e2.{gname}", {"protocol": protocol, "bus": sb, "branch": sr, "scenario": ss,
                                      "branch_risk_tau_loading": tau_l,
                                      "h1": {"bus_headline": hb, "branch_headline": hr, "verdict": verdict}})
    log.info("error check %s: %s (bus %s, branch %s)", gname, verdict, hb, hr)

    if "e3" not in args.skip:
        risk = comb.predict(R[BR_FEATS].astype(np.float64), R.load_scenario.to_numpy())
        t3, s3 = e3(R, risk, k_max, cfg, seed)
        t3.to_csv(out_dir / "e3_conformal.csv", index=False)
        io.update_summary(f"e3.{gname}", s3)
        log.info("error bands %s: scaled bands narrower %s; in-dist %s", gname, s3["h5_pass"], s3["in_dist"])


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--cap-indist", type=int, default=500_000, help="row sample cap per in-distribution split")
    ap.add_argument("--cap-shift", type=int, default=150_000, help="row sample cap per shifted split")
    ap.add_argument("--e1-scenarios", type=int, default=256)
    ap.add_argument("--skip", action="append", default=[], choices=["e1", "e2", "e3"])
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    splits, k_max = grid_splits(gname), k_fit(gname)
    t_all = time.perf_counter()

    if "e1" not in a.skip:
        tab, summ = e1(gname, splits, a.e1_scenarios)
        tab.to_csv(io.results_dir(gname) / "e1_anatomy.csv", index=False)
        io.update_summary(f"e1.{gname}", summ)
        log.info("mismatch by step done")
    if "e2" not in a.skip:
        run_e2_e3(gname, splits, k_max, cfg, a)
    log.info("error_check %s done in %.0f s", gname, time.perf_counter() - t_all)

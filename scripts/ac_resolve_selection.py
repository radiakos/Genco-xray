"""Selecting cases for AC re-solve: share re-solved against overload recall for each selection rule. Writes
summary.e4.<grid> and results/<grid>/e4_curves.csv.

    python scripts/ac_resolve_selection.py --grid texas2k
    python scripts/ac_resolve_selection.py --grid synthetic

Protocol (docs/experiments.md, "Selecting cases for AC re-solve"):
* A scenario is overloaded if its max branch loading > 1 under AC; GENCO's decision uses its own max loading.
  Re-solving a scenario with AC makes its decision correct and its loading error 0. The error tracked is the
  scenario's max |L_pred - L_true| over branches.
* Rules ("gates"): trust (scenario accuracy score, = -SCENARIO.trust_score from error_check.py), residual (mean final
  mismatch), k, random (mean of 20 seeded orders), oracle (true decision errors first). Ties are broken by a seeded
  permutation.
* q* = smallest share of scenarios re-solved with overload recall >= hybrid.recall_target, computed exactly
  (every n/S). Intervals: cluster bootstrap over load scenarios (--n-boot), paired across rules.
* Subsets: all, in_dist (k <= k_fit), k <= 5 (pooled) and k >= 10 (the two subsets of the verdict).
* Runtime: results/<grid>/runtime.json, if present, is carried into the summary with its ``valid`` flag (GPU not
  throttled). It is not used in the write-up.
* Pass "accuracy score selects best" (summary key h2, per subset): q*(trust) < q* of every non-oracle rule.
  Reported on all subsets; the verdict uses k <= 5 and k >= 10.
* Added 19 Oct after the case118 run (case118 = development system; declared before Texas was run, which is the
  held-out test of these rules): a missed overload is a scenario whose max L_pred sits just under 1, so the distance
  to the limit matters as much as the error size.
    - margin:       -|max L_pred - 1|  (distance to the limit)
    - margin_trust: -z_flip, the decision-flip score with z_b = (L_pred,b - 1) / s(branch_risk_b): predicted
                    overload -> max of z_b over branches above 1 (all must fall below 1 to flip); predicted safe ->
                    min of -z_b over all branches (one must rise above 1). s = isotonic map of |L_pred - L_true| on
                    branch_risk (conformal.difficulty_scale_fn), fitted on branch rows inside the training range of
                    load-scenario half 0 (trust.group_fold(., 2)).
  Pass "distance in units of expected error selects best" (summary key h2b): q*(margin_trust) < q*(margin) and < q*
  of every other non-oracle rule.
"""

import logging

import numpy as np
import pandas as pd

from gencoxray import cli, conformal, hybrid, io, trust
from gencoxray.config import grid_splits, k_fit, load_config

log = logging.getLogger("ac_resolve_selection")
GATES = ("trust", "residual", "k", "margin", "margin_trust")
FIT_ROWS = 1_000_000
BR_COLS = ["scenario", "load_scenario", "k", "in_service", "loading_pred", "loading_true", "err_loading",
           "branch_risk"]


def fit_scale(gname: str, splits: list[str], k_max: int, seed: int):
    """Isotonic |L_pred - L_true| ~ branch_risk on in-distribution rows of load-scenario half 0."""
    parts = []
    for s in splits:
        br = io.read_columns("branch", gname, s, BR_COLS)
        m = br.in_service & (br.k <= k_max) & (trust.group_fold(br.load_scenario.to_numpy(), 2, seed + 303) == 0)
        if m.any():
            parts.append(br.loc[m, ["branch_risk", "err_loading"]])
    d = pd.concat(parts, ignore_index=True).dropna()
    if len(d) > FIT_ROWS:
        d = d.sample(n=FIT_ROWS, random_state=seed)
    log.info("margin_trust scale fitted on %d in-distribution branch rows", len(d))
    return conformal.difficulty_scale_fn(d.branch_risk.to_numpy(), d.err_loading.to_numpy())


def load(gname: str, splits: list[str], scale) -> pd.DataFrame:
    parts = []
    for s in splits:
        sc = io.read_table("scenario", gname, s)
        br = io.read_columns("branch", gname, s, BR_COLS)
        sc["max_err_loading"] = sc.scenario.map(br.groupby("scenario").err_loading.max()).astype(np.float64)
        on = br[br.in_service & br.loading_pred.notna()]
        z = (on.loading_pred - 1.0) / scale(on.branch_risk.to_numpy())  # signed: > 0 above the limit
        flip = hybrid.decision_flip_score(on.scenario.to_numpy(), z.to_numpy())
        sc["min_flip_z"] = sc.scenario.map(flip).astype(np.float64)
        parts.append(sc.assign(split=s))
    return pd.concat(parts, ignore_index=True)


def gate_scores(sc: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"trust": -sc.trust_score.to_numpy(np.float64), "residual": sc.res_mean_final_mva.to_numpy(np.float64),
            "k": sc.k.to_numpy(np.float64),
            "margin": -np.abs(sc.max_loading_pred.to_numpy(np.float64) - 1.0),
            "margin_trust": -sc.min_flip_z.to_numpy(np.float64)}


CHECK_COLS = ["split", "scenario", "max_last_vm", "max_mis_bus", "max_last_ld", "max_mis_br", "z_last_flip",
              "rec_score", "flag_last_only", "flag_recommendation", "any_wrong_bus", "any_wrong_branch",
              "max_loading_newton", "max_err_newton", "newton_max_dl", "z_newton_flip"]
BASE_GATES = ("margin_trust", "margin")
FIXED_RULES = {"last_step_only": "flag_last_only", "recommendation": "flag_recommendation"}


def add_checks(sc: pd.DataFrame, gname: str, splits: list[str]) -> pd.DataFrame:
    """Merge the per-scenario scores of scenario_check_scores.py (columns missing if the script was not run)."""
    parts = []
    for s in splits:
        p = io.results_dir(gname, s) / "scenario_checks.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            parts.append(d[[c for c in CHECK_COLS if c in d.columns]])
    if not parts:
        return sc
    return sc.merge(pd.concat(parts, ignore_index=True), on=["split", "scenario"], how="left")


def extra_gates(d: pd.DataFrame, thr: float) -> dict[str, tuple]:
    """gate -> (risk, GENCO-or-Newton decision, remaining-error array). Newton gates decide the scenarios that are not
    re-solved from the corrected loadings (one Jacobian factorization each); the others keep GENCO's decision."""
    g = ((d.max_loading_pred > thr).to_numpy(), d.max_err_loading.to_numpy())
    out = {}
    if "z_last_flip" in d:
        out |= {"last_step": (d.max_last_ld.to_numpy(np.float64), *g),
                "last_step_bus": (d.max_last_vm.to_numpy(np.float64), *g),
                "last_step_margin": (-d.z_last_flip.to_numpy(np.float64), *g),
                "recommendation": (d.rec_score.to_numpy(np.float64), *g)}
    if "z_newton_flip" in d and d.z_newton_flip.notna().any():
        n = ((d.max_loading_newton > thr).to_numpy(), d.max_err_newton.to_numpy())
        out |= {"newton_margin": (-np.abs(d.max_loading_newton.to_numpy(np.float64) - thr), *n),
                "newton_margin_scaled": (-d.z_newton_flip.to_numpy(np.float64), *n),
                "newton_change": (d.newton_max_dl.to_numpy(np.float64), *n)}
    return out


def warm_iterations(gname: str) -> dict:
    p = io.results_dir(gname) / "newton_warm_corrected.csv"
    if not p.exists():
        return {}
    w = pd.read_csv(p)
    return {s: float(v) for s, v in w[w.converged].groupby("start").iterations.mean().items()}


def fixed_rule_rows(d: pd.DataFrame, scores: dict, pred, true, err, budgets_seed: int) -> list[dict]:
    """Fixed element-level rules: every flagged scenario goes to AC, the rest keeps GENCO's decision."""
    rows = []
    wrong = (d.any_wrong_bus | d.any_wrong_branch).to_numpy(bool)
    for rule, col in FIXED_RULES.items():
        fl = d[col].to_numpy(bool)
        share = float(fl.mean())
        mt = hybrid.budget_curve(scores["margin_trust"], pred, true, err, np.array([share]), budgets_seed).iloc[0]
        rows.append({"rule": rule, "share_flagged": share,
                     "recall_overload": float((true & (fl | pred)).sum() / max(true.sum(), 1)),
                     "n_missed": int((true & ~fl & ~pred).sum()), "max_err_kept": float(err[~fl].max()) if (~fl).any() else 0.0,
                     "recall_any_wrong": float(fl[wrong].mean()) if wrong.any() else np.nan,
                     "recall_wrong_bus": float(fl[d.any_wrong_bus.to_numpy(bool)].mean()) if d.any_wrong_bus.any() else np.nan,
                     "recall_wrong_branch": float(fl[d.any_wrong_branch.to_numpy(bool)].mean()) if d.any_wrong_branch.any() else np.nan,
                     "prevalence_any_wrong": float(wrong.mean()),
                     "margin_trust_recall_same_share": float(mt.recall)})
    return rows


def q_star_exact(risk, pred, true, err, target: float, seed: int) -> float:
    S = len(true)
    return hybrid.q_star(hybrid.budget_curve(risk, pred, true, err, np.arange(S + 1) / S, seed), target)


def subsets(k: np.ndarray, k_max: int) -> dict[str, np.ndarray]:
    d = {"all": np.ones_like(k, bool), "in_dist": k <= k_max, "k_le_5": k <= 5, "k_ge_10": k >= 10}
    return {n: m for n, m in d.items() if m.sum() >= 50}


def bootstrap_qstar(sc: pd.DataFrame, pred, true, err, target: float, n_boot: int, seed: int, extras=None) -> dict:
    """Paired cluster bootstrap (load scenarios) of q* for each gate and of q*(trust) - q*(other)."""
    g = sc.load_scenario.to_numpy()
    order = np.argsort(g, kind="stable")
    _, starts = np.unique(g[order], return_index=True)
    rows = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    scores = gate_scores(sc)
    extras = extras or {}
    res = {gate: [] for gate in (*GATES, *extras)}
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        for gate in GATES:
            res[gate].append(q_star_exact(scores[gate][idx], pred[idx], true[idx], err[idx], target, seed))
        for gate, (r, p_, e_) in extras.items():
            res[gate].append(q_star_exact(r[idx], p_[idx], true[idx], e_[idx], target, seed))
    res = {gate: np.asarray(v) for gate, v in res.items()}
    out = {f"q_star_{gate}_ci": [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
           for gate, v in res.items()}
    for lead in ("trust", "margin_trust"):
        for other in GATES:
            if other == lead:
                continue
            d = res[lead] - res[other]
            out[f"{lead}_minus_{other}"] = {"lo": float(np.nanquantile(d, 0.025)),
                                            "hi": float(np.nanquantile(d, 0.975)),
                                            "p_lead_smaller": float(np.mean(d < 0))}
    for gate in extras:
        for ref in BASE_GATES:
            d = res[gate] - res[ref]
            out[f"{gate}_minus_{ref}"] = {"lo": float(np.nanquantile(d, 0.025)), "hi": float(np.nanquantile(d, 0.975)),
                                          "p_lead_smaller": float(np.mean(d < 0))}
    return out


def runtime(gname: str) -> dict:
    if not (io.results_dir(gname) / "runtime.json").exists():
        return {"t_genco_ms": 1.0, "t_ac_ms": 10.0, "valid": False, "source": "placeholder (synthetic)"}
    r = io.read_json("runtime.json", gname)
    return {"t_genco_ms": float(r["genco"]["ms_per_instance"]), "t_ac_ms": float(r["ac"]["ac_ms_mean"]),
            "t_ac_median_ms": float(r["ac"]["ac_ms_median"]), "valid": bool(r["genco"].get("valid", False)),
            "genco_batch": r["genco"].get("best_batch_size"), "device": r["genco"].get("device"),
            "ac_source": r["ac"].get("source")}


def verdict(summ: dict, subsets_: list[str], key: str, label: str) -> str:
    p = [summ["subsets"][s][key] for s in subsets_]
    return f"{label} supported" if all(p) else f"{label} partly supported" if any(p) else f"{label} not supported"


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    splits, k_max = grid_splits(gname), k_fit(gname)
    thr, target, seed = cfg.labels.decision_overload_threshold, cfg.hybrid.recall_target, cfg.seed
    budgets = np.linspace(cfg.hybrid.budgets.start, cfg.hybrid.budgets.stop, cfg.hybrid.budgets.num)

    sc = add_checks(load(gname, splits, fit_scale(gname, splits, k_max, seed)), gname, splits)
    if sc.trust_score.isna().all():
        raise SystemExit(f"{gname}: SCENARIO.trust_score is empty; run scripts/error_check.py first")
    rt = runtime(gname)
    curves, fixed, scen_rows, summ = [], [], [], {"runtime": rt, "recall_target": target, "overload_threshold": thr, "n_boot": a.n_boot,
                        "subsets": {}}
    for name, m in subsets(sc.k.to_numpy(), k_max).items():
        d = sc[m].reset_index(drop=True)
        pred = (d.max_loading_pred > thr).to_numpy()
        true = (d.max_loading_true > thr).to_numpy()
        err = d.max_err_loading.to_numpy()
        scores = gate_scores(d)
        tab = hybrid.compare_gates(scores, pred, true, err, budgets, n_random=cfg.hybrid.random_repeats, seed=seed)
        extras = extra_gates(d, thr)
        curves.append(tab.assign(subset=name))
        for g_, (r_, p_, e_) in extras.items():
            curves.append(hybrid.budget_curve(r_, p_, true, e_, budgets, seed).assign(gate=g_, subset=name))
        S = len(d)
        qs = {g: q_star_exact(r, pred, true, err, target, seed) for g, r in scores.items()}
        qs["oracle"] = q_star_exact(hybrid.oracle_risk(pred, true, err), pred, true, err, target, seed)
        rq = [q_star_exact(np.zeros(S), pred, true, err, target, seed + 1 + i) for i in range(cfg.hybrid.random_repeats)]
        qs["random"] = float(np.mean(rq))
        boot = bootstrap_qstar(d, pred, true, err, target, a.n_boot, seed, extras) if a.n_boot else {}
        t_eff = {g: float(hybrid.effective_time_ms(q, rt["t_genco_ms"], rt["t_ac_ms"])) for g, q in qs.items()}
        eff = {g: {"q_star": q, "effective_ms": t_eff[g], "speedup_vs_ac": float(rt["t_ac_ms"] / t_eff[g])}
               for g, q in qs.items()}
        base = tab[tab.q == 0].set_index("gate").iloc[0]
        h2 = bool(all(qs["trust"] < qs[g] for g in ("residual", "k", "random")))
        h2b = bool(all(qs["margin_trust"] < qs[g] for g in ("trust", "residual", "k", "random", "margin")))
        summ["subsets"][name] = {
            "n_scenarios": S, "n_overload_true": int(true.sum()), "n_decision_errors": int((pred != true).sum()),
            "genco_only": {"recall": float(base.recall), "precision": float(base.precision),
                           "n_missed": int(base.n_missed), "max_abs_err": float(base.max_abs_err)},
            "gates": eff, "bootstrap": boot, "h2_pass": h2, "h2b_pass": h2b,
            "sentence": (f"re-solving {100 * qs['trust']:.1f} % of scenarios (trust gate) gives "
                         f"{100 * target:.1f} % overload recall at {eff['trust']['speedup_vs_ac']:.2f}x AC throughput "
                         f"(residual gate {100 * qs['residual']:.1f} %, k {100 * qs['k']:.1f} %, random "
                         f"{100 * qs['random']:.1f} %, oracle {100 * qs['oracle']:.1f} %)"),
            "sentence_margin_trust": (f"re-solving {100 * qs['margin_trust']:.1f} % of scenarios (margin/trust gate) "
                                      f"gives {100 * target:.1f} % overload recall at "
                                      f"{eff['margin_trust']['speedup_vs_ac']:.2f}x AC throughput (margin alone "
                                      f"{100 * qs['margin']:.1f} %)"),
        }
        # Newton step, last-step change and their fixed rules (vol. 2, modules 2-3); needs scenario_check_scores.py
        if extras:
            qx = {g_: q_star_exact(r_, p_, true, e_, target, seed) for g_, (r_, p_, e_) in extras.items()}
            allg = {g_: (r_, pred, err) for g_, r_ in scores.items()} | extras
            qall = qs | qx
            wi = warm_iterations(gname)

            def cost(g_, qall=qall, wi=wi):  # Newton iterations per scenario; the factorization of a Newton gate counts as one
                newton_gate = g_.startswith("newton")
                if not wi:
                    return np.nan
                return float(newton_gate + qall[g_] * wi["genco_step" if newton_gate else "genco"])

            xtra = {}
            for g_, (r_, p_, e_) in allg.items():
                c0 = hybrid.budget_curve(r_, p_, true, e_, np.array([0.0, qall[g_]]), seed)
                xtra[g_] = {"q_star": qall[g_], "recall_q0": float(c0.recall[0]), "precision_q0": float(c0.precision[0]),
                            "n_missed_q0": int(c0.n_missed[0]), "n_decision_errors_q0": int((p_ != true).sum()),
                            "max_err_at_qstar": float(c0.max_abs_err[1]), "cost_iterations": cost(g_),
                            "iteration_ratio": float(wi["flat"] / cost(g_)) if wi and cost(g_) > 0 else np.nan}
            summ["subsets"][name]["selection_rules"] = xtra
            summ["subsets"][name]["warm_iterations_mean"] = wi
            if "flag_last_only" in d:
                fr = fixed_rule_rows(d, scores, pred, true, err, seed)
                fixed += [r | {"subset": name} for r in fr]
                summ["subsets"][name]["fixed_rules"] = fr
            lab = {"any_wrong": (d.any_wrong_bus | d.any_wrong_branch).to_numpy(float),
                   "wrong_bus": d.any_wrong_bus.to_numpy(float), "wrong_branch": d.any_wrong_branch.to_numpy(float)}
            sc_scores = {"last_step_bus": d.max_last_vm, "last_step_loading": d.max_last_ld,
                         "final_mismatch_bus": d.max_mis_bus, "final_mismatch_branch": d.max_mis_br,
                         "recommendation": d.rec_score, "newton_change": d.get("newton_max_dl"),
                         "number_of_outages": d.k}
            sc_scores = {n_: v.to_numpy(np.float64) for n_, v in sc_scores.items() if v is not None}
            for lname, y_ in lab.items():
                ev = trust.evaluate_scores(sc_scores, y_, 0.5, d.load_scenario.to_numpy(), n_boot=a.n_boot or 0, seed=seed)
                scen_rows.append(ev.assign(label=lname, subset=name))
                summ["subsets"][name].setdefault("scenario_level", {})[lname] = {
                    r.score: {c: None if pd.isna(getattr(r, c)) else float(getattr(r, c))
                              for c in ("auroc", "auroc_lo", "auroc_hi", "auprc", "prevalence", "n_pos")}
                    for r in ev.itertuples()}
        log.info("%s/%s: %s; accuracy score best %s | %s; distance in units of expected error best %s", gname, name,
                 summ["subsets"][name]["sentence"], h2, summ["subsets"][name]["sentence_margin_trust"], h2b)
    h2_sets = [s for s in ("k_le_5", "k_ge_10") if s in summ["subsets"]] or ["all"]
    summ["h2"] = {"subsets": h2_sets, "verdict": verdict(summ, h2_sets, "h2_pass", "H2"),
                  "verdict_h2b": verdict(summ, h2_sets, "h2b_pass", "H2b"),
                  "runtime_valid": rt["valid"],
                  "note": "margin / margin_trust gates added after the case118 run; Texas is their held-out test"}
    pd.concat(curves, ignore_index=True).to_csv(io.results_dir(gname) / "e4_curves.csv", index=False)
    if fixed:
        pd.DataFrame(fixed).to_csv(io.results_dir(gname) / "fixed_rules.csv", index=False)
    if scen_rows:
        pd.concat(scen_rows, ignore_index=True).to_csv(io.results_dir(gname) / "scenario_level_checks.csv", index=False)
    io.update_summary(f"e4.{gname}", summ)
    log.info("selection for AC re-solve %s: %s", gname, summ["h2"])

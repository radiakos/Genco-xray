"""Stress maps from GENCO / AC / DC and their rank agreement. Writes summary.e5.<grid> and
results/<grid>/e5_{branch_stress,bus_stress,criticality,agreement}.csv.

    python scripts/stress_maps.py --grid texas2k
    python scripts/stress_maps.py --grid synthetic

Protocol (docs/experiments.md, "Stress maps against AC and DC"):
* Ensemble = every scenario of the grid's splits (Texas: 1,000 per k = 1..20; IEEE 118-bus: test + nk/k01..k10), pooled
  ("all") and per k-bin (configs/experiment.yaml conformal.k_bins).
* Branch stress per source (GENCO = loading_pred, AC = loading_true, DC = loading_dc): p_overload, exp_excess,
  cvar (CVaR_95). Agreement with AC: Spearman (all branches), precision@k / Jaccard@k (k in stress.topk), Kendall
  on the union of the top-50. 95 % CIs by cluster bootstrap over load scenarios: --n-boot resamples for the mean
  metrics, --n-boot-cvar for CVaR (sort-once fast path, ``ranking._SortedCvar``).
* Pass "GENCO stress map beats DC" (summary key h3, per subset): GENCO's Spearman and precision@25 vs AC both exceed
  DC's with non-overlapping CIs, on the branch CVaR_95 map (every branch has a value; --n-boot-cvar resamples).
* Also reported: a paired bootstrap of rho/precision@25 (GENCO - DC) on shared resamples (``pass_paired``: CI > 0).
  Near rho = 1 resampling attenuates rho slightly for both sources; the paired difference is not affected.
* Why CVaR, not p_overload (decided 19 Oct on synthetic data; of the real grids only the old-definition case118
  "all" line, GENCO rho 0.955 vs DC 0.965 on p_overload, had been printed): p_overload / exp_excess are 0 for most
  branches. Joint zeros inflate Spearman and bias its bootstrap (on synthetic the DC CI lay entirely above the point);
  restricting to positives rewards spurious positives (DC 0.81 vs GENCO 0.27 although GENCO finds exactly the planted
  set). For these metrics ``ranking`` positive_only mode reports set agreement only (top-k among positives, symmetric
  precision@k, Jaccard@k, positive counts) plus the spec's all-element Spearman as a point estimate (spearman_all).
* Bus: P(Vm outside the voltage band) on PQ rows (Vm is set at PV/REF), GENCO vs AC (DC has no voltages).
* Outage criticality: mean n_overload over the scenarios where a branch is out (elements out in >= 5 scenarios).
"""

import logging
import time

import numpy as np
import pandas as pd

from gencoxray import cli, io, ranking, stress
from gencoxray.config import grid_splits, load_config

log = logging.getLogger("stress_maps")
SOURCES = {"genco": "loading_pred", "ac": "loading_true", "dc": "loading_dc"}
METRICS = ("p_overload", "exp_excess", "cvar")
MIN_OUT = 5


def load(gname: str, splits: list[str]):
    """Scenario meta + loading matrices [S, nb] per source + Vm matrices [S, B] (PQ rows only) for GENCO and AC."""
    nb = len(io.read_table("static_branch", gname))
    nbus = len(io.read_table("static_bus", gname))
    metas, Ls, Vs = [], {s: [] for s in SOURCES}, {"genco": [], "ac": []}
    for s in splits:
        sc = io.read_table("scenario", gname, s).sort_values("scenario")
        S = len(sc)
        br = io.read_columns("branch", gname, s, ["scenario", "branch", "in_service"] + list(SOURCES.values()))
        br = br.sort_values(["scenario", "branch"])
        if len(br) != S * nb:
            raise SystemExit(f"{gname}/{s}: {len(br)} branch rows != {S} x {nb}")
        on = br.in_service.to_numpy().reshape(S, nb)
        for src, col in SOURCES.items():
            Ls[src].append(np.where(on, br[col].to_numpy(np.float32).reshape(S, nb), np.nan).astype(np.float32))
        bus = io.read_columns("bus", gname, s, ["scenario", "bus", "bus_type", "vm_pred", "vm_true"])
        bus = bus.sort_values(["scenario", "bus"])
        pq = (bus.bus_type.astype(str) == "PQ").to_numpy().reshape(S, nbus)
        for src, col in (("genco", "vm_pred"), ("ac", "vm_true")):
            Vs[src].append(np.where(pq, bus[col].to_numpy(np.float32).reshape(S, nbus), np.nan).astype(np.float32))
        metas.append(sc[["scenario", "load_scenario", "k", "outaged_branches"]].assign(split=s))
        log.info("%s/%s loaded (%d scenarios)", gname, s, S)
    meta = pd.concat(metas, ignore_index=True)
    return meta, {k: np.concatenate(v) for k, v in Ls.items()}, {k: np.concatenate(v) for k, v in Vs.items()}, nb, nbus


def subsets(k: np.ndarray, k_bins) -> dict[str, np.ndarray]:
    d = {"all": np.ones_like(k, bool)}
    for lo, hi in k_bins:
        m = (k >= lo) & (k <= hi)
        if m.sum() >= 50:
            d[f"k{lo}-{hi}"] = m
    return d


def h3(agr: pd.DataFrame) -> dict:
    a = agr[agr.metric == "cvar"].set_index(["source", "stat"])
    res = {}
    for stat in ("spearman", "precision@25"):
        g, d = a.loc[("genco", stat)], a.loc[("dc", stat)]
        res[stat] = {"genco": [float(g.point), float(g.lo), float(g.hi)], "dc": [float(d.point), float(d.lo), float(d.hi)],
                     "genco_better_non_overlapping": bool(g.lo > d.hi)}
    res["pass"] = bool(all(res[s]["genco_better_non_overlapping"] for s in ("spearman", "precision@25")))
    return res


def bus_vviol(V: np.ndarray, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(P(Vm outside band) per bus [B], in-service count n [B]) from a Vm matrix [S, B] (NaN = not a PQ row)."""
    on = np.isfinite(V)
    with np.errstate(invalid="ignore"):
        viol = ((V < band[0]) | (V > band[1])) & on
    n = on.sum(0)
    return np.where(n > 0, viol.sum(0) / np.maximum(n, 1), np.nan), n


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-boot-cvar", type=int, default=100)
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    splits = grid_splits(gname)
    topk, band, q = tuple(cfg.stress.topk), tuple(cfg.stress.voltage_band), cfg.stress.cvar_q
    t0 = time.perf_counter()
    meta, L, V, nb, nbus = load(gname, splits)
    k, groups = meta.k.to_numpy(), meta.load_scenario.to_numpy()
    elem, buses = np.arange(nb), np.arange(nbus)
    has_dc = bool(np.isfinite(L["dc"]).any())

    stress_rows, agr_rows, bus_rows, crit_rows, summ = [], [], [], [], {"subsets": {}, "n_scenarios": len(meta)}
    for name, m in subsets(k, cfg.conformal.k_bins).items():
        Lm = {s: L[s][m].astype(np.float64) for s in SOURCES}
        for src in SOURCES:
            if src == "dc" and not has_dc:
                continue
            st = stress.branch_stress_matrix(Lm[src], elem, cvar_q=q).reset_index()
            stress_rows.append(st.assign(source=src, subset=name))
        tabs = []
        for src in ("genco", "dc"):
            if src == "dc" and not has_dc:
                continue
            for metric in METRICS:
                nbt = a.n_boot if metric in stress.MEAN_METRICS else a.n_boot_cvar
                pos = metric in ("p_overload", "exp_excess")
                t = ranking.bootstrap_matrix(Lm[src], Lm["ac"], elem, groups[m], metric, topk, nbt, cfg.seed, q,
                                             positive_only=pos)
                if pos:
                    a_all = stress.mean_stats(Lm[src])[metric], stress.mean_stats(Lm["ac"])[metric]
                    rho_all = ranking.spearman(pd.Series(a_all[0]), pd.Series(a_all[1]))
                    t = pd.concat([t, pd.DataFrame([{"stat": "spearman_all", "point": rho_all, "lo": np.nan,
                                                     "hi": np.nan, "boot_mean": np.nan}])], ignore_index=True)
                tabs.append(t.assign(source=src, metric=metric))
        boots = {(t.source.iloc[0], t.metric.iloc[0]): t for t in tabs}
        agr = pd.concat(tabs, ignore_index=True)
        agr_rows.append(agr.assign(subset=name))
        verdict = h3(agr) if has_dc else {"pass": None, "note": "no DC results"}
        if has_dc:  # paired bootstrap of GENCO - DC on shared resamples (attenuation near rho = 1 cancels)
            g_, d_ = boots[("genco", "cvar")], boots[("dc", "cvar")]
            verdict["paired_genco_minus_dc"] = {st: ranking.paired_delta(g_, d_, st)
                                                for st in ("spearman", "precision@25")}
            verdict["pass_paired"] = bool(all(v["lo"] > 0 for v in verdict["paired_genco_minus_dc"].values()))

        # bus voltage stress (PQ rows): GENCO vs AC
        bs = {}
        for src in ("genco", "ac"):
            p_v, n = bus_vviol(V[src][m].astype(np.float64), band)
            bs[src] = pd.Series(p_v, index=buses)
            bus_rows.append(pd.DataFrame({"bus": buses, "p_vviol": bs[src].to_numpy(), "n": n, "source": src,
                                          "subset": name}))
        bus_agr = ranking.agreement(bs["genco"], bs["ac"], topk, positive_only=True)

        # outage criticality: GENCO / AC / DC
        crit = {}
        for src in SOURCES:
            if src == "dc" and not has_dc:
                continue
            n_over, mx = stress.scenario_overloads(Lm[src])
            c = stress.criticality_from(np.flatnonzero(m), meta.outaged_branches[m].reset_index(drop=True), n_over, mx)
            c = c[c.n_scenarios >= MIN_OUT]
            crit[src] = c.mean_n_overload
            crit_rows.append(c.reset_index().assign(source=src, subset=name))
        crit_agr = {src: ranking.agreement(crit[src], crit["ac"], topk, positive_only=True) for src in crit
                    if src != "ac"}

        pa = agr.set_index(["source", "metric", "stat"])
        summ["subsets"][name] = {
            "n_scenarios": int(m.sum()), "n_load_scenarios": int(np.unique(groups[m]).size),
            "branches_ever_overloaded": {s: int((np.nan_to_num(Lm[s]) > 1).any(0).sum()) for s in Lm
                                         if s != "dc" or has_dc},
            "branch_agreement": {f"{s}.{mt}.{st}": [float(r.point), float(r.lo), float(r.hi)]
                                 for (s, mt, st), r in pa.iterrows()},
            "h3": verdict,
            "bus_vviol_agreement_genco_vs_ac": bus_agr,
            "criticality_agreement_vs_ac": crit_agr,
            "n_outaged_elements": len(crit["ac"]),
        }
        log.info("%s/%s: GENCO stress map beats DC %s | %s", gname, name, verdict["pass"],
                 {kk: vv for kk, vv in verdict.items() if kk != "pass"})

    out = io.results_dir(gname)
    pd.concat(stress_rows, ignore_index=True).to_csv(out / "e5_branch_stress.csv", index=False)
    pd.concat(agr_rows, ignore_index=True).to_csv(out / "e5_agreement.csv", index=False)
    pd.concat(bus_rows, ignore_index=True).to_csv(out / "e5_bus_stress.csv", index=False)
    pd.concat(crit_rows, ignore_index=True).to_csv(out / "e5_criticality.csv", index=False)
    bins = [s for s in summ["subsets"] if s != "all"]
    passes = [summ["subsets"][s]["h3"]["pass"] for s in ["all"] + bins]
    summ["h3"] = {"all": summ["subsets"]["all"]["h3"]["pass"], "per_k_bin": {s: summ["subsets"][s]["h3"]["pass"] for s in bins},
                  "paired": {s: summ["subsets"][s]["h3"].get("pass_paired") for s in ["all"] + bins},
                  "verdict": ("H3 supported" if all(passes) else "H3 partly supported" if any(passes)
                              else "H3 not supported")}
    summ["protocol"] = {"metric_for_h3": "cvar", "n_boot_cvar": a.n_boot_cvar, "topk": list(topk), "cvar_q": q, "n_boot": a.n_boot,
                        "group_by": "load_scenario", "voltage_band": list(band), "min_outage_scenarios": MIN_OUT}
    io.update_summary(f"e5.{gname}", summ)
    log.info("stress maps %s: %s (%.0f s)", gname, summ["h3"], time.perf_counter() - t0)

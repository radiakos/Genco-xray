"""Physical stress metrics over scenario ensembles.

Works on any loading/voltage column, so the same code computes GENCO-, AC- and DC-based stress.
Out-of-service branches (``in_service == False``) and NaN loadings are excluded.

Core: matrix functions on L [S, E] (scenario x element, NaN = not in service) with optional scenario weights
[S] or [B, S] (a batch of bootstrap weight vectors). The DataFrame functions of the spec pivot to that form, so
the long-table API (tests, small data) and the script (Texas: 20,000 x 3,220 per source) share one implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MEAN_METRICS = ("p_overload", "exp_excess", "l_mean")  # decomposable: weighted means over scenarios


# ----------------------------------------------------------------------------- matrix core
def to_matrix(df: pd.DataFrame, col: str, key: str = "branch") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long table -> (L [S, E] float64 with NaN where missing / out of service, scenario ids [S], element ids [E])."""
    d = df
    if "in_service" in d:
        d = d.assign(**{col: d[col].where(d.in_service)})
    scen, si = np.unique(d.scenario.to_numpy(), return_inverse=True)
    elem, ei = np.unique(d[key].to_numpy(), return_inverse=True)
    L = np.full((scen.size, elem.size), np.nan)
    L[si, ei] = d[col].to_numpy(np.float64)
    return L, scen, elem


def mean_stats(L: np.ndarray, w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Weighted means over scenarios per element: p_overload, exp_excess, l_mean, n (= in-service weight).
    w: None, [S] or [B, S] -> arrays [E] or [B, E]."""
    on = np.isfinite(L)
    Lz = np.where(on, L, 0.0)
    parts = {"n": on.astype(np.float64), "p_overload": (Lz > 1).astype(np.float64),
             "exp_excess": np.maximum(Lz - 1.0, 0.0), "l_mean": Lz}
    if w is None:
        sums = {k: v.sum(0) for k, v in parts.items()}
    else:
        sums = {k: np.asarray(w, np.float64) @ v for k, v in parts.items()}
    n = sums.pop("n")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {k: np.where(n > 0, v / n, np.nan) for k, v in sums.items()}
    out["n"] = n
    return out


def cvar(L: np.ndarray, q: float = 0.95, w: np.ndarray | None = None) -> np.ndarray:
    """Per element: CVaR_q of L, the weighted mean of the upper (1 - q) share of the element's in-service weight,
    with the boundary row counted fractionally (Rockafellar-Uryasev). Invariant to replicating rows, so integer
    bootstrap weights give exactly the resampled value. [E]; NaN if never in service."""
    S = L.shape[0]
    w = np.ones(S) if w is None else np.asarray(w, np.float64)
    Ls = np.where(np.isfinite(L), L, -np.inf)
    order = np.argsort(-Ls, axis=0, kind="stable")
    Lo = np.take_along_axis(Ls, order, 0)
    wo = np.where(np.isfinite(Lo), w[order], 0.0)
    m = (1 - q) * wo.sum(0)
    above = np.cumsum(wo, 0) - wo
    contrib = np.clip(m - above, 0.0, wo)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > 0, (contrib * np.where(np.isfinite(Lo), Lo, 0.0)).sum(0) / m, np.nan)


def branch_stress_matrix(L: np.ndarray, elem: np.ndarray, *, cvar_q: float = 0.95,
                         w: np.ndarray | None = None) -> pd.DataFrame:
    m = mean_stats(L, w)
    df = pd.DataFrame({"p_overload": m["p_overload"], "exp_excess": m["exp_excess"], "cvar": cvar(L, cvar_q, w),
                       "l_mean": m["l_mean"], "n": m["n"]}, index=pd.Index(elem, name="branch"))
    return df[df.n > 0]


# ----------------------------------------------------------------------------- spec API (long tables)
def branch_stress(branch_df: pd.DataFrame, col: str = "loading_pred", *, cvar_q: float = 0.95,
                  weights: pd.Series | None = None) -> pd.DataFrame:
    """Per branch: p_overload = mean(L > 1), exp_excess = mean(max(L - 1, 0)), cvar = mean of L above its
    cvar_q quantile, l_mean, n (in-service scenarios, or their total weight). Index = branch.
    Optional scenario weights (Series indexed by scenario id)."""
    L, scen, elem = to_matrix(branch_df, col)
    w = None if weights is None else weights.reindex(scen).fillna(0).to_numpy(np.float64)
    return branch_stress_matrix(L, elem, cvar_q=cvar_q, w=w)


def bus_stress(bus_df: pd.DataFrame, col: str = "vm_pred", band: tuple[float, float] = (0.95, 1.05)) -> pd.DataFrame:
    """Per bus: p_vviol, p_under, p_over, vm_min_q05, n. Index = bus. ``fixed_vm`` flags buses whose Vm is set in
    every scenario (PV/REF in all rows, if ``bus_type`` is present); they are kept but flagged."""
    V, _, buses = to_matrix(bus_df, col, key="bus")
    on = np.isfinite(V)
    n = on.sum(0)
    lo, hi = band
    with np.errstate(invalid="ignore"):
        under = ((V < lo) & on).sum(0) / np.maximum(n, 1)
        over = ((V > hi) & on).sum(0) / np.maximum(n, 1)
    out = pd.DataFrame({"p_vviol": under + over, "p_under": under, "p_over": over,
                        "vm_min_q05": np.nanquantile(V, 0.05, axis=0), "n": n}, index=pd.Index(buses, name="bus"))
    if "bus_type" in bus_df:
        out["fixed_vm"] = (bus_df.bus_type.astype(str) != "PQ").groupby(bus_df.bus).all().reindex(buses).to_numpy()
    return out


def scenario_overloads(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_overload [S], max loading [S]) recomputed from a loading matrix."""
    with np.errstate(invalid="ignore"):
        n_over = np.nansum(L > 1, axis=1)
    mx = np.where(np.isfinite(L).any(1), np.nanmax(np.where(np.isfinite(L), L, -np.inf), axis=1), np.nan)
    return n_over.astype(np.float64), mx


def criticality_from(scen_ids: np.ndarray, outaged: pd.Series, n_over: np.ndarray, max_l: np.ndarray) -> pd.DataFrame:
    """Per outaged element: mean n_overload, mean max loading, n_scenarios (``outaged`` aligned with scen_ids)."""
    d = pd.DataFrame({"scenario": scen_ids, "out": outaged.to_numpy(), "n_over": n_over, "max_l": max_l})
    d = d.explode("out").dropna(subset=["out"])
    d["out"] = d.out.astype(np.int64)
    g = d.groupby("out")
    out = pd.DataFrame({"mean_n_overload": g.n_over.mean(), "mean_max_loading": g.max_l.mean(),
                        "n_scenarios": g.size()})
    out.index.name = "outaged_branch"
    return out


def outage_criticality(scenario_df: pd.DataFrame, branch_df: pd.DataFrame, col: str = "loading_pred") -> pd.DataFrame:
    """Per outaged branch id: mean n_overload, mean max loading, n_scenarios, over scenarios where it is out.

    n_overload / max loading are recomputed from ``branch_df[col]`` (not taken from scenario_df) so that
    GENCO/AC/DC versions are comparable. Uses explode('outaged_branches')."""
    L, scen, _ = to_matrix(branch_df, col)
    n_over, mx = scenario_overloads(L)
    outs = scenario_df.set_index("scenario").outaged_branches.reindex(scen)
    return criticality_from(scen, outs, n_over, mx)

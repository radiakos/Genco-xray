"""Position variables (outage count, hop distances, degree) that predict GENCO's errors without the correction steps.

Used for the position-only baseline and to stratify and residualise the error-predictor features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from .trust import auroc


def _adjacency(edge_index: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    ei = np.asarray(edge_index)
    a = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n_nodes, n_nodes)).tocsr()
    return ((a + a.T) > 0).astype(np.float64)


def hop_distance(edge_index: np.ndarray, sources: np.ndarray, n_nodes: int) -> np.ndarray:
    """Multi-source BFS hop distance [n_nodes] (int16) via scipy.sparse.csgraph.shortest_path(unweighted=True,
    indices=sources).min(0). Unreachable -> n_nodes."""
    sources = np.asarray(sources, dtype=np.int64).ravel()
    if sources.size == 0:
        return np.full(n_nodes, n_nodes, np.int16)
    d = shortest_path(_adjacency(edge_index, n_nodes), directed=False, unweighted=True, indices=sources)
    d = np.atleast_2d(d).min(axis=0)
    return np.where(np.isfinite(d), d, n_nodes).astype(np.int16)


def hop_to_outage(edge_from: np.ndarray, edge_to: np.ndarray, in_service: np.ndarray,
                  source_mask: np.ndarray, max_hops: int = 64) -> np.ndarray:
    """Per scenario, hops from every bus to the nearest *source* bus over the IN-SERVICE graph.

    edge_from/edge_to: [E] directed edges (both directions present); in_service: [S, E] bool;
    source_mask: [S, B] bool (endpoints of outaged branches + buses of outaged generators).
    Returns float32 [S, B]; sources = 0; NaN if a scenario has no source or a bus is unreachable
    (or farther than ``max_hops``).
    Vectorised frontier BFS over all scenarios at once (one sparse product per hop; loop over hop count only).
    """
    ef, et = np.asarray(edge_from), np.asarray(edge_to)
    src = np.asarray(source_mask, bool)
    on = np.asarray(in_service, bool)
    S, B = src.shape
    E = ef.shape[0]
    to_bus = sp.csr_matrix((np.ones(E, np.float32), (np.arange(E), et)), shape=(E, B))  # edge -> head bus
    dist = np.full((S, B), np.nan, np.float32)
    dist[src] = 0.0
    visited, frontier = src.copy(), src.copy()
    for h in range(1, max_hops + 1):
        active = frontier[:, ef] & on  # [S, E]: edges leaving the frontier
        if not active.any():
            break
        reached = np.asarray((to_bus.T @ active.T.astype(np.float32)).T) > 0  # [S, B]
        frontier = reached & ~visited
        if not frontier.any():
            break
        dist[frontier] = h
        visited |= frontier
    return dist


def outage_sources(scenario_df: pd.DataFrame, static_branch: pd.DataFrame, gen_bus: np.ndarray,
                   n_bus: int) -> np.ndarray:
    """[S, B] bool source mask for :func:`hop_to_outage`, rows in ``scenario_df`` order: the from/to buses of
    every branch in ``outaged_branches`` and the bus of every generator in ``outaged_gens`` (``gen_bus[g]``)."""
    S = len(scenario_df)
    mask = np.zeros((S, n_bus), bool)
    if S == 0:
        return mask
    sb = static_branch.set_index("branch").sort_index()
    f, t = sb.from_bus.to_numpy(), sb.to_bus.to_numpy()
    ob = [np.asarray(x, np.int64) for x in scenario_df.outaged_branches]
    og = [np.asarray(x, np.int64) for x in scenario_df.outaged_gens]
    rb = np.repeat(np.arange(S), [len(x) for x in ob])
    rg = np.repeat(np.arange(S), [len(x) for x in og])
    b = np.concatenate(ob) if rb.size else np.zeros(0, np.int64)
    g = np.concatenate(og) if rg.size else np.zeros(0, np.int64)
    mask[rb, f[b]] = True
    mask[rb, t[b]] = True
    if g.size:
        mask[rg, np.asarray(gen_bus)[g]] = True
    return mask


def degree(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Undirected degree [n_nodes]; counts each physical neighbour once even if edges are stored both ways
    (and once for parallel branches). Self-loops are ignored."""
    ei = np.asarray(edge_index)
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)
    return np.bincount(pairs.ravel(), minlength=n_nodes).astype(np.int16)


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def residualize(features: pd.DataFrame, confounders: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Replace each feature by its residual after OLS on the confounders, fitted on ``train_mask`` rows only.

    Log-transform positive-skewed features first (signed log1p of |x| when the training-row skewness > 1).
    Confounder NaNs (e.g. ``hop_to_outage`` for k = 0) are imputed with the training median plus a missingness
    indicator; constant confounder columns are dropped. NaN features stay NaN.
    """
    train_mask = np.asarray(train_mask, bool)
    C = confounders.astype(np.float64)
    cols = []
    for c in C.columns:
        v = C[c].to_numpy()
        miss = np.isnan(v)
        med = np.nanmedian(v[train_mask]) if np.isfinite(v[train_mask]).any() else 0.0
        cols.append(np.where(miss, med, v))
        if miss[train_mask].any():
            cols.append(miss.astype(np.float64))
    X = np.column_stack(cols) if cols else np.zeros((len(C), 0))
    X = X[:, X[train_mask].std(axis=0) > 0] if X.size else X
    X = np.column_stack([np.ones(len(X)), X])

    out = pd.DataFrame(index=features.index)
    for f in features.columns:
        y = features[f].to_numpy(np.float64)
        ok = np.isfinite(y)
        fit = ok & train_mask
        yt = y[fit]
        if yt.size > 2 and yt.std() > 0 and pd.Series(yt).skew() > 1:
            y = _signed_log1p(y)
        res = np.full_like(y, np.nan)
        if fit.sum() > X.shape[1]:
            beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
            res[ok] = y[ok] - X[ok] @ beta
        out[f] = res
    return out


def stratified_auroc(score: np.ndarray, y: np.ndarray, strata: np.ndarray, min_pos: int = 20) -> pd.DataFrame:
    """AUROC within each stratum (e.g. hop-distance bin) with n, n_pos. Strata with < min_pos positives (or no
    negatives) -> NaN. Rows with a NaN score are dropped. Columns: stratum, n, n_pos, auroc."""
    df = pd.DataFrame({"s": np.asarray(score, np.float64), "y": np.asarray(y, bool), "g": np.asarray(strata)})
    df = df[np.isfinite(df.s)]
    rows = []
    for g, d in df.groupby("g", sort=True, observed=True):
        n, npos = len(d), int(d.y.sum())
        auc = auroc(d.s.to_numpy(), d.y.to_numpy()) if npos >= min_pos and n > npos else np.nan
        rows.append({"stratum": g, "n": n, "n_pos": npos, "auroc": auc})
    return pd.DataFrame(rows, columns=["stratum", "n", "n_pos", "auroc"])

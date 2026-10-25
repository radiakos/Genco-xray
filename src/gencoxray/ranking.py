"""Rank agreement between stress maps (GENCO vs AC vs DC) with cluster bootstrap.

Bootstrap: resample load scenarios (``load_scenario`` if present, else ``scenario``). For the mean metrics
(p_overload, exp_excess, l_mean) the per-group sums are aggregated once and every resample is one matrix product
(``stress.mean_stats`` with group counts); cvar is recomputed per resample from scenario weights.

Zero-inflated maps (``positive_only=True``): on large grids most branches never overload, so p_overload and exp_excess
are exactly 0 for most elements in both maps. Joint zeros carry no ranking information, inflate Spearman, and make it
non-bootstrappable: a resample drops overload events, creates more joint zeros and raises rho, so the bootstrap
distribution lies above the point estimate. Restricting Spearman to elements > 0 in either map does not fix it: it
rewards spurious positives (easy to rank below the true ones). ``positive_only`` therefore reports set agreement
only: top-k sets drawn from positive values, precision@k = |T_pred ∩ T_ref| / min(k, max(|T_pred|, |T_ref|))
(symmetric: extra positives on either side count against it), Jaccard@k, and the positive counts. Rank agreement is
measured on a continuous metric (CVaR) instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from . import stress


def _common(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return d.a, d.b


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho on the common index, NaNs dropped (NaN if either side is constant)."""
    a, b = _common(a, b)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _top(s: pd.Series, k: int, positive_only: bool = False) -> set:
    """Top-k index labels, ties at the cutoff broken by index (ascending) for determinism."""
    s = s.dropna()
    if positive_only:
        s = s[s > 0]
    order = np.lexsort((s.index.to_numpy(), -s.to_numpy()))
    return set(s.index.to_numpy()[order[:k]])


def precision_at_k(pred: pd.Series, ref: pd.Series, k: int) -> float:
    """|top_k(pred) ∩ top_k(ref)| / k. Ties at the cutoff are broken by index for determinism."""
    pred, ref = _common(pred, ref)
    return len(_top(pred, k) & _top(ref, k)) / k


def jaccard_at_k(pred: pd.Series, ref: pd.Series, k: int) -> float:
    pred, ref = _common(pred, ref)
    a, b = _top(pred, k), _top(ref, k)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def kendall_top(pred: pd.Series, ref: pd.Series, k: int) -> float:
    """Kendall tau on the union of both top-k sets."""
    pred, ref = _common(pred, ref)
    u = sorted(_top(pred, k) | _top(ref, k))
    if len(u) < 3:
        return float("nan")
    return float(kendalltau(pred.loc[u], ref.loc[u]).statistic)


def agreement(pred: pd.Series, ref: pd.Series, topk: tuple[int, ...] = (10, 25, 50), *,
              positive_only: bool = False) -> dict[str, float]:
    """spearman, precision@k, jaccard@k for each k, and kendall_top@max(topk); with ``positive_only`` set
    agreement only (see the module docstring)."""
    if not positive_only:
        out = {"spearman": spearman(pred, ref)}
        for k in topk:
            out[f"precision@{k}"] = precision_at_k(pred, ref, k)
            out[f"jaccard@{k}"] = jaccard_at_k(pred, ref, k)
        out[f"kendall_top{max(topk)}"] = kendall_top(pred, ref, max(topk))
        return out
    pred, ref = _common(pred, ref)
    out = {}
    for k in topk:
        tp, tr = _top(pred, k, True), _top(ref, k, True)
        den = min(k, max(len(tp), len(tr)))
        out[f"precision@{k}"] = len(tp & tr) / den if den else float("nan")
        out[f"jaccard@{k}"] = len(tp & tr) / len(tp | tr) if (tp | tr) else float("nan")
    out["n_pos_pred"] = float((pred > 0).sum())
    out["n_pos_ref"] = float((ref > 0).sum())
    return out


def _group_sums(L: np.ndarray, gi: np.ndarray, G: int) -> dict[str, np.ndarray]:
    """Per-group sums [G, E] of the mean-metric numerators and the in-service count."""
    on = np.isfinite(L)
    Lz = np.where(on, L, 0.0)
    parts = {"n": on, "p_overload": Lz > 1, "exp_excess": np.maximum(Lz - 1.0, 0.0), "l_mean": Lz}
    out = {}
    for k, v in parts.items():
        acc = np.zeros((G, L.shape[1]))
        np.add.at(acc, gi, v.astype(np.float64))
        out[k] = acc
    return out


def _resampled_mean(C: np.ndarray, sums: dict[str, np.ndarray], metric: str) -> np.ndarray:
    """[b, E] mean metric under group-count resamples C [b, G] from per-group sums (:func:`_group_sums`)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        n = C @ sums["n"]
        return np.where(n > 0, (C @ sums[metric]) / n, np.nan)


class _SortedCvar:
    """CVaR_q per column under many scenario-weight vectors: columns sorted once (descending), then each weight
    vector costs one gather + cumsum (float32). Same definition as ``stress.cvar``."""

    def __init__(self, L: np.ndarray, q: float):
        Ls = np.where(np.isfinite(L), L, -np.inf).astype(np.float32)
        self.order = np.argsort(-Ls, axis=0, kind="stable")
        Lo = np.take_along_axis(Ls, self.order, 0)
        self.fin = np.isfinite(Lo)
        self.Lo = np.where(self.fin, Lo, 0.0).astype(np.float32)
        self.q = q

    def __call__(self, w: np.ndarray) -> np.ndarray:
        wo = np.where(self.fin, np.asarray(w, np.float32)[self.order], np.float32(0))
        m = np.float32(1 - self.q) * wo.sum(0)
        contrib = np.clip(m - (np.cumsum(wo, 0) - wo), 0, wo)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(m > 0, (contrib * self.Lo).sum(0, dtype=np.float64) / m, np.nan)


def bootstrap_matrix(Lp: np.ndarray, Lr: np.ndarray, elem: np.ndarray, groups: np.ndarray, metric: str = "p_overload",
                     topk: tuple[int, ...] = (10, 25, 50), n_boot: int = 500, seed: int = 0,
                     cvar_q: float = 0.95, positive_only: bool = False) -> pd.DataFrame:
    """Agreement of the ``metric`` stress maps from two loading matrices [S, E] (same rows/columns), with a cluster
    bootstrap over ``groups`` [S]. Returns one row per statistic: point, lo, hi (95 % percentile), boot_mean
    (to check the bootstrap is centred on the point estimate)."""
    _, gi = np.unique(groups, return_inverse=True)
    G = int(gi.max()) + 1
    idx = pd.Index(elem)
    mean_metric = metric in stress.MEAN_METRICS

    def agree(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        return agreement(pd.Series(a, idx), pd.Series(b, idx), topk, positive_only=positive_only)

    if mean_metric:
        point = agree(stress.mean_stats(Lp)[metric], stress.mean_stats(Lr)[metric])
    else:
        point = agree(stress.cvar(Lp, cvar_q), stress.cvar(Lr, cvar_q))
    draws = {k: [] for k in point}

    def record(a: np.ndarray, b: np.ndarray) -> None:
        for k, v in agree(a, b).items():
            draws[k].append(v)

    rng = np.random.default_rng(seed)
    p = np.full(G, 1.0 / G)
    if mean_metric:  # one matrix product per chunk of resamples
        sums = (_group_sums(Lp, gi, G), _group_sums(Lr, gi, G))
        for lo in range(0, n_boot, 100):
            C = rng.multinomial(G, p, size=min(100, n_boot - lo)).astype(np.float64)
            for a, b in zip(*(_resampled_mean(C, s, metric) for s in sums)):
                record(a, b)
    elif n_boot > 0:
        sp, sr = _SortedCvar(Lp, cvar_q), _SortedCvar(Lr, cvar_q)
        for _ in range(n_boot):
            w = rng.multinomial(G, p).astype(np.float32)[gi]
            record(sp(w), sr(w))
    rows = []
    for k, v in point.items():
        x = np.asarray(draws[k], float)
        x = x[np.isfinite(x)]
        lo_, hi_ = (np.quantile(x, [0.025, 0.975]) if x.size else (np.nan, np.nan))
        rows.append({"stat": k, "point": v, "lo": float(lo_), "hi": float(hi_),
                     "boot_mean": float(x.mean()) if x.size else np.nan})
    out = pd.DataFrame(rows)
    # resample draws (same seed + same groups -> same resamples for every call: paired comparisons across sources)
    out.attrs["draws"] = {k: np.asarray(v, float) for k, v in draws.items()}
    return out


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, stat: str) -> dict[str, float]:
    """point and 95 % CI of stat(a) - stat(b) from two :func:`bootstrap_matrix` results with the same seed/groups."""
    pa, pb = a.set_index("stat").loc[stat, "point"], b.set_index("stat").loc[stat, "point"]
    d = a.attrs["draws"][stat] - b.attrs["draws"][stat]
    d = d[np.isfinite(d)]
    lo, hi = (np.quantile(d, [0.025, 0.975]) if d.size else (np.nan, np.nan))
    return {"delta": float(pa - pb), "lo": float(lo), "hi": float(hi), "p_a_greater": float(np.mean(d > 0))}


def bootstrap_agreement(branch_df: pd.DataFrame, col_pred: str, col_ref: str, metric: str = "p_overload",
                        topk: tuple[int, ...] = (10, 25, 50), n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Resample load scenarios (``load_scenario`` if present, else ``scenario``), recompute
    ``stress.branch_stress`` for both columns, then spearman/precision@k/jaccard@k/kendall_top.
    Returns one row per statistic: point, lo, hi."""
    Lp, scen, elem = stress.to_matrix(branch_df, col_pred)
    Lr, scen_r, elem_r = stress.to_matrix(branch_df, col_ref)
    assert np.array_equal(scen, scen_r) and np.array_equal(elem, elem_r)
    key = "load_scenario" if "load_scenario" in branch_df else "scenario"
    groups = branch_df.groupby("scenario")[key].first().reindex(scen).to_numpy()
    return bootstrap_matrix(Lp, Lr, elem, groups, metric, topk, n_boot, seed)


def rank_disagreement(pred: pd.Series, ref: pd.Series) -> pd.Series:
    """|rank_pred - rank_ref| / (n - 1) per element on the common index (0 = same rank, 1 = opposite ends).
    Used for the rank variant of the stress and accuracy map test."""
    pred, ref = _common(pred, ref)
    n = len(pred)
    return (pred.rank() - ref.rank()).abs() / max(n - 1, 1)

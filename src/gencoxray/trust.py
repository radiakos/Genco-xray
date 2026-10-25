"""Error checks: metrics, the fitted error predictor ("combiner") and the load-scenario bootstrap.

Grouping: every ``groups`` argument is ``load_scenario`` (datakit's load_scenario_idx), NOT ``scenario``. datakit
makes ``n_topology_variants`` scenarios per load scenario, so grouping by scenario leaks between folds/bootstraps.
Convention: a *score* is higher when an error is MORE likely ("risk score"). ``scenario.trust_score`` stores the
opposite sign (higher = more trustworthy) = -risk; convert only via :func:`risk_to_trust`.

Folds: a load scenario's fold is a fixed hash of its id (:func:`group_fold`), not a GroupKFold of the rows at hand.
The same load scenario therefore lands in the same fold in every split and in every table (bus, branch, scenario),
so a fold model can score rows that were never part of its fitting sample (other splits, unsampled rows) without
ever having seen their load scenario. This matters on Texas, whose twenty test_k splits share load scenarios.

Bootstrap: AUROC/AUPRC CIs use a weighted cluster bootstrap (multinomial counts per load scenario as row weights)
on scores sorted once, so 1000 resamples of 10^6 rows take seconds. :func:`bootstrap_ci` is the generic
(slow, any statistic) version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)


# ----------------------------------------------------------------------------- point metrics
def _clean(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(score, np.float64)
    y = np.asarray(y).astype(bool)
    m = np.isfinite(s)
    return s[m], y[m]


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC with NaN scores dropped; returns NaN if y is single-class. Ties count 1/2."""
    s, y = _clean(score, y)
    npos = int(y.sum())
    nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(score: np.ndarray, y: np.ndarray) -> float:
    """Average precision with NaN scores dropped (sklearn definition; NaN if there are no positives)."""
    s, y = _clean(score, y)
    if y.sum() == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


# ----------------------------------------------------------------------------- fast weighted bootstrap
class _Sorted:
    """Scores sorted once (ascending) with tie blocks, for weighted AUROC / AP under many weight vectors."""

    def __init__(self, score: np.ndarray, y: np.ndarray, keep: np.ndarray):
        s = np.asarray(score, np.float64)[keep]
        self.order = np.argsort(s, kind="stable")
        ss = s[self.order]
        self.y = np.asarray(y, bool)[keep][self.order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])

    def auroc(self, w: np.ndarray) -> np.ndarray:
        """w: [b, n] row weights in the ORIGINAL (kept) order -> [b] AUROC."""
        w = w[:, self.order]
        pos = np.add.reduceat(w * self.y, self.starts, axis=1)
        neg = np.add.reduceat(w * ~self.y, self.starts, axis=1)
        below = np.cumsum(neg, axis=1) - neg
        num = (pos * (below + 0.5 * neg)).sum(1)
        den = pos.sum(1) * neg.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    def auprc(self, w: np.ndarray) -> np.ndarray:
        """Weighted average precision (thresholds at tie blocks, descending score) -> [b]."""
        w = w[:, self.order]
        pos = np.add.reduceat(w * self.y, self.starts, axis=1)[:, ::-1]
        tot = np.add.reduceat(w, self.starts, axis=1)[:, ::-1]
        tp, alln = np.cumsum(pos, axis=1), np.cumsum(tot, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(alln > 0, tp / alln, 0.0)
            return (pos * prec).sum(1) / pos.sum(1)


class _GroupStats:
    """Group-level sufficient statistics for AUROC / AP under cluster-bootstrap weights (counts per group).

    With c [b, G] = resample counts per group (row weight = c[group]):
      AUROC = c^T A c / ((c.P)(c.N)),  A[g, h] = sum over positives i in g of #negatives in h below s_i (ties 1/2),
      AP    = sum_i c[g_i] * (c . TP_i) / (c . ALL_i) / (c . P),  TP_i[h] / ALL_i[h] = #positives / #rows of group h
              with score >= s_i (sklearn's step definition, ties included).
    Exact (same numbers as weighting the rows), and each resample costs O(G^2 + n_pos * G) instead of O(n log n).
    """

    def __init__(self, s: np.ndarray, y: np.ndarray, gi: np.ndarray, G: int):
        sp, gp, sn, gn = s[y], gi[y], s[~y], gi[~y]
        self.gp, self.G = gp, G
        npos = sp.size
        N = np.zeros((npos, G), np.float32)
        TP = np.zeros((npos, G), np.float32)
        NG = np.zeros((npos, G), np.float32)  # negatives with score >= s_i
        on, op = np.lexsort((sn, gn)), np.lexsort((sp, gp))
        sn_s, sp_s = sn[on], sp[op]
        bn = np.searchsorted(gn[on], np.arange(G + 1))
        bp = np.searchsorted(gp[op], np.arange(G + 1))
        for h in range(G):
            neg = sn_s[bn[h]:bn[h + 1]]
            if neg.size:
                left = np.searchsorted(neg, sp, "left")
                N[:, h] = (left + np.searchsorted(neg, sp, "right")) / 2
                NG[:, h] = neg.size - left
            pos = sp_s[bp[h]:bp[h + 1]]
            if pos.size:
                TP[:, h] = pos.size - np.searchsorted(pos, sp, "left")
        A = np.zeros((G, G))
        np.add.at(A, gp, N.astype(np.float64))
        self.A, self.TP, self.ALL = A, TP, TP + NG
        self.P = np.bincount(gp, minlength=G).astype(np.float64)
        self.N = np.bincount(gn, minlength=G).astype(np.float64)

    @staticmethod
    def feasible(n_pos: int, n_neg: int, G: int) -> bool:
        return G <= 5000 and min(n_pos, n_neg) * G <= 4e7

    def auroc(self, C: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((C @ self.A) * C).sum(1) / ((C @ self.P) * (C @ self.N))

    def auprc(self, C: np.ndarray) -> np.ndarray:
        Cf = C.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.nan_to_num((Cf @ self.TP.T) / (Cf @ self.ALL.T))
            return (C[:, self.gp] * prec).sum(1) / (C @ self.P)


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, gi = np.unique(np.asarray(groups), return_inverse=True)
    return gi.ravel(), int(gi.max()) + 1 if gi.size else 0


def _boot_counts(n_groups: int, n_boot: int, seed: int, chunk: int):
    """Yield [b, G] float64 multinomial counts of resampled groups (row weight = count of its group). The
    stream depends only on (n_groups, seed), so paired statistics share resamples."""
    rng = np.random.default_rng(seed)
    p = np.full(n_groups, 1.0 / n_groups)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        yield rng.multinomial(n_groups, p, size=b).astype(np.float64)
        done += b


def _chunk(n: int) -> int:
    return int(max(1, min(64, 2e7 // max(n, 1))))


def _pct(x: np.ndarray, level: float) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def bootstrap_ci(stat_fn, groups: np.ndarray, *arrays: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Cluster bootstrap: resample unique ``groups`` (load scenarios) with replacement, take all their rows,
    and recompute ``stat_fn(*arrays_subset)``. Returns (point, lo, hi) (percentile interval).
    Group -> row index lists are precomputed once (argsort + split)."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    rows = np.split(order, starts[1:])
    G = len(rows)
    rng = np.random.default_rng(seed)
    point = float(stat_fn(*arrays))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, G, G)])
        stats[b] = stat_fn(*(np.asarray(a)[idx] for a in arrays))
    lo, hi = _pct(stats, level)
    return point, lo, hi


def _cluster_bootstrap(scores: list[np.ndarray], y: np.ndarray, keep: np.ndarray, groups: np.ndarray, n_boot: int,
                       seed: int, with_auprc: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bootstrap draws of AUROC (and AP) for several scores on the rows ``keep``, sharing one resample stream.

    Exact group-level statistics (:class:`_GroupStats`) when affordable, otherwise row weights on scores sorted once
    (:class:`_Sorted`). Returns ([auroc draws per score], [auprc draws per score or empty])."""
    gi, G = _group_index(np.asarray(groups)[keep])
    yk = y[keep]
    if _GroupStats.feasible(int(yk.sum()), int((~yk).sum()), G):
        stats = [_GroupStats(s[keep], yk, gi, G) for s in scores]
        draws, weights = _boot_counts(G, n_boot, seed, 100), (lambda C: C)
    else:
        stats = [_Sorted(s, y, keep) for s in scores]
        draws, weights = _boot_counts(G, n_boot, seed, _chunk(keep.sum())), (lambda C: C[:, gi])
    a, p = [[] for _ in stats], [[] for _ in stats]
    for C in draws:
        w = weights(C)
        for i, st in enumerate(stats):
            a[i].append(st.auroc(w))
            if with_auprc:
                p[i].append(st.auprc(w))
    return [np.concatenate(x) for x in a], [np.concatenate(x) for x in p if x]


def boot_metrics(score: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> dict[str, float]:
    """Point AUROC/AUPRC with weighted cluster-bootstrap CIs (fast path of :func:`bootstrap_ci`)."""
    s = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    keep = np.isfinite(s)
    out = {"auroc": auroc(s, y), "auprc": auprc(s, y)}
    if n_boot <= 0 or y[keep].all() or not y[keep].any():
        return out | {"auroc_lo": np.nan, "auroc_hi": np.nan, "auprc_lo": np.nan, "auprc_hi": np.nan}
    (a,), (p,) = _cluster_bootstrap([s], y, keep, groups, n_boot, seed, with_auprc=True)
    out["auroc_lo"], out["auroc_hi"] = _pct(a, level)
    out["auprc_lo"], out["auprc_hi"] = _pct(p, level)
    return out


def delta_auroc_ci(score_a: np.ndarray, score_b: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired cluster bootstrap of AUROC(a) - AUROC(b) on the rows where both scores are finite.
    Used to compare the error predictor with the final mismatch. Returns (delta, lo, hi)."""
    a, b = np.asarray(score_a, np.float64), np.asarray(score_b, np.float64)
    y = np.asarray(y, bool)
    keep = np.isfinite(a) & np.isfinite(b)
    d = auroc(a[keep], y[keep]) - auroc(b[keep], y[keep])
    if not (y[keep].any() and (~y[keep]).any()):
        return float(d), float("nan"), float("nan")
    (da, db), _ = _cluster_bootstrap([a, b], y, keep, groups, n_boot, seed, with_auprc=False)
    lo, hi = _pct(da - db, 0.95)
    return float(d), lo, hi


def evaluate_scores(scores: dict[str, np.ndarray], err: np.ndarray, tau: float, groups: np.ndarray,
                    n_boot: int = 1000, seed: int = 0, min_pos: int = 20) -> pd.DataFrame:
    """One row per score: auroc, auroc_lo, auroc_hi, auprc, auprc_lo, auprc_hi, spearman(score, err),
    prevalence, n, n_pos, n_groups, underpowered. Label = err > tau (NaN err rows are dropped).
    With fewer than ``min_pos`` positives the row is flagged ``underpowered`` and gets no CI (a bootstrap over a
    handful of positives gives spuriously tight intervals)."""
    from scipy.stats import spearmanr

    err = np.asarray(err, np.float64)
    ok = np.isfinite(err)
    y = err > tau
    rows = []
    for name, sc in scores.items():
        s = np.asarray(sc, np.float64)
        m = ok & np.isfinite(s)
        under = int(y[m].sum()) < min_pos
        r = boot_metrics(s[m], y[m], np.asarray(groups)[m], n_boot=0 if under else n_boot, seed=seed)
        const = m.sum() <= 2 or np.ptp(s[m]) == 0 or np.ptp(err[m]) == 0
        rho = np.nan if const else spearmanr(s[m], err[m]).statistic
        rows.append({"score": name, **r, "spearman": float(rho), "prevalence": float(y[m].mean()) if m.any() else np.nan,
                     "n": int(m.sum()), "n_pos": int(y[m].sum()), "n_groups": int(np.unique(np.asarray(groups)[m]).size),
                     "underpowered": under})
    cols = ["score", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "spearman", "prevalence", "n",
            "n_pos", "n_groups", "underpowered"]
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------- combiner
def group_fold(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id in [0, n_splits) per row, a fixed hash of the group id (and seed): the same load scenario gets the
    same fold in every split/table. Groups are spread ~uniformly over folds."""
    g = np.asarray(groups).astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (g + np.uint64(seed) * np.uint64(1_000_003) + np.uint64(1)) * _HASH_MULT
    return ((h >> np.uint64(33)) % np.uint64(n_splits)).astype(np.int64)


def _slog(x):
    x = np.asarray(x, np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    return np.sign(x) * np.log1p(np.abs(x))


def _make_model(kind: str, seed: int):
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        return Pipeline([
            ("log", FunctionTransformer(_slog)),
            ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        ])
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                              early_stopping=False, random_state=seed)
    raise ValueError(f"unknown combiner kind {kind!r}")


class _Constant:
    def __init__(self, v: float):
        self.v = v

    def log_odds(self, X) -> np.ndarray:
        return np.full(len(X), self.v)


def _fit_one(X: np.ndarray, y: np.ndarray, kind: str, seed: int):
    if y.all() or not y.any():
        return _Constant(10.0 if y.all() else -10.0)
    m = _make_model(kind, seed)
    m.fit(np.where(np.isfinite(X), X, np.nan), y)
    return m


def _log_odds(model, X: np.ndarray) -> np.ndarray:
    """Both the logistic pipeline and HGB return binary log-odds from ``decision_function``."""
    if isinstance(model, _Constant):
        return model.log_odds(X)
    return model.decision_function(np.where(np.isfinite(X), X, np.nan))


class GroupFoldCombiner:
    """Fold models fitted on the rows outside each group-fold, plus a final model on all rows.

    ``predict(X, groups)`` scores every row with the model of its group's fold, i.e. a model that never saw that
    load scenario (out-of-fold for fitting rows; leakage-free for any other row). Scores are log-odds.
    """

    def __init__(self, kind: str = "logistic", n_splits: int = 5, seed: int = 0):
        self.kind, self.n_splits, self.seed = kind, n_splits, seed
        self.fold_models: list = []
        self.final = None
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, final: bool = True) -> GroupFoldCombiner:
        """``final=False`` skips the all-rows model (only the fold models are needed for ``predict(X, groups)``)."""
        self.columns = list(X.columns)
        Xa = X.to_numpy(np.float64)
        y = np.asarray(y, bool)
        f = group_fold(groups, self.n_splits, self.seed)
        self.fold_models = [_fit_one(Xa[f != i], y[f != i], self.kind, self.seed) for i in range(self.n_splits)]
        self.final = _fit_one(Xa, y, self.kind, self.seed) if final else None
        return self

    def predict(self, X: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
        """Out-of-fold log-odds (``groups`` given) or final-model log-odds (``groups`` None)."""
        Xa = X[self.columns].to_numpy(np.float64)
        if groups is None:
            return _log_odds(self.final, Xa)
        f = group_fold(groups, self.n_splits, self.seed)
        out = np.empty(len(Xa))
        for i, m in enumerate(self.fold_models):
            sel = f == i
            if sel.any():
                out[sel] = _log_odds(m, Xa[sel])
        return out


def fit_combiner(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, kind: str = "logistic",
                 n_splits: int = 5, seed: int = 0) -> tuple[np.ndarray, GroupFoldCombiner]:
    """Out-of-fold risk scores (log-odds) via group folds (:func:`group_fold`) + a final model refit on all rows.

    logistic: features -> sign(x)*log1p(|x|) -> median impute + missing-indicator -> StandardScaler ->
    LogisticRegression(C=1.0, class_weight='balanced'). hgb: HistGradientBoostingClassifier (native NaN).
    Returns (oof_scores [n], fitted :class:`GroupFoldCombiner`).
    """
    comb = GroupFoldCombiner(kind, n_splits, seed).fit(X, y, groups)
    return comb.predict(X, groups), comb


def fit_apply_shift(X: pd.DataFrame, y: np.ndarray, k: np.ndarray, k_fit_max: int = 2,
                    kind: str = "logistic", groups: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Fit on rows with k <= k_fit_max, score ALL rows.

    With ``groups``: fold models fitted on the k <= k_fit_max rows outside each group-fold score every row of that
    fold (out-of-fold for fitting rows, and load-scenario-disjoint for the shifted rows too). Without: fitting rows
    get GroupKFold-free out-of-fold scores by row position folds, the others the final model.
    """
    fit = np.asarray(k) <= k_fit_max
    y = np.asarray(y, bool)
    if groups is not None:
        comb = GroupFoldCombiner(kind, seed=seed).fit(X[fit], y[fit], np.asarray(groups)[fit])
        return comb.predict(X, groups)
    pseudo = np.arange(int(fit.sum()))
    oof, comb = fit_combiner(X[fit], y[fit], pseudo, kind=kind, seed=seed)
    out = comb.predict(X)
    out[fit] = oof
    return out


# ----------------------------------------------------------------------------- selective prediction, aggregation
def risk_coverage(score: np.ndarray, err_bool: np.ndarray, n_points: int = 101) -> pd.DataFrame:
    """Selective prediction curve. For coverage c in linspace(0, 1, n_points), retain the round(c*n) lowest-risk
    rows and report their error rate (NaN at c = 0). NaN scores are dropped. Columns: coverage, risk.
    AURC (mean risk over c > 0) goes in df.attrs['aurc']."""
    s, e = _clean(score, err_bool)
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(e[order])
    cov = np.linspace(0, 1, n_points)
    m = np.round(cov * s.size).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(m > 0, cum[np.maximum(m - 1, 0)] / np.maximum(m, 1), np.nan) if s.size else np.full(n_points, np.nan)
    df = pd.DataFrame({"coverage": cov, "risk": risk})
    df.attrs["aurc"] = float(np.nanmean(risk)) if np.isfinite(risk).any() else float("nan")
    return df


def scenario_risk(row_scores: np.ndarray, scenario_ids: np.ndarray, how: str = "q90") -> pd.Series:
    """Aggregate bus/branch risk scores to scenarios: 'max' | 'mean' | 'q90' (NaN rows ignored). Index = scenario id."""
    s = pd.Series(np.asarray(row_scores, np.float64)).groupby(np.asarray(scenario_ids), sort=True)
    if how == "max":
        return s.max()
    if how == "mean":
        return s.mean()
    if how == "q90":
        return s.quantile(0.9)
    raise ValueError(f"unknown aggregate {how!r}")


def risk_to_trust(risk: np.ndarray) -> np.ndarray:
    """trust_score = -risk (kept as a function so the sign convention lives in one place)."""
    return -np.asarray(risk, dtype=np.float32)

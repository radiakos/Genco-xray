"""Hybrid GENCO–AC screening: rank scenarios by a selection rule and re-solve the top share with AC."""

from __future__ import annotations

import numpy as np
import pandas as pd

CURVE_COLS = ["q", "n_resolved", "recall", "precision", "n_missed", "max_abs_err", "mean_abs_err"]


def _order(risk: np.ndarray, seed: int) -> np.ndarray:
    """Scenario order, highest risk first; NaN risk last; ties broken by a seeded permutation."""
    r = np.asarray(risk, np.float64)
    r = np.where(np.isfinite(r), r, -np.inf)
    perm = np.random.default_rng(seed).permutation(r.size)
    return np.lexsort((perm, -r))


def budget_curve(risk: np.ndarray, pred_overload: np.ndarray, true_overload: np.ndarray,
                 pred_abs_err: np.ndarray, budgets: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """For each budget q: send the ceil(q*S) highest-risk scenarios to AC (their decision becomes correct and
    their error 0) and keep GENCO for the rest.

    Columns: q, n_resolved, recall, precision, n_missed (true overloads predicted negative), max_abs_err,
    mean_abs_err. Ties in risk are broken by a fixed random permutation (seeded) to avoid order artefacts.
    O(S log S): one sort, then prefix sums (and a suffix max for max_abs_err). With no true overload,
    recall is 1 by convention; with no predicted overload, precision is 1.
    """
    pred = np.asarray(pred_overload, bool)
    true = np.asarray(true_overload, bool)
    err = np.nan_to_num(np.asarray(pred_abs_err, np.float64), nan=0.0)
    S = true.size
    o = _order(risk, seed)
    p, t, e = pred[o], true[o], err[o]

    def pre(x):
        return np.r_[0.0, np.cumsum(x, dtype=np.float64)]

    c_true, c_tp, c_pp, c_fn, c_err = pre(t), pre(p & t), pre(p), pre(t & ~p), pre(e)
    suf_max = np.r_[np.maximum.accumulate(e[::-1])[::-1], 0.0] if S else np.zeros(1)
    q = np.asarray(budgets, np.float64)
    n = np.minimum(np.ceil(q * S - 1e-9).astype(int), S)
    n_true, tot_tp, tot_pp, tot_fn = c_true[-1], c_tp[-1], c_pp[-1], c_fn[-1]
    tp = c_true[n] + (tot_tp - c_tp[n])
    pos = c_true[n] + (tot_pp - c_pp[n])
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_true > 0, tp / max(n_true, 1), 1.0)
        precision = np.where(pos > 0, tp / np.maximum(pos, 1), 1.0)
    return pd.DataFrame({
        "q": q, "n_resolved": n, "recall": recall, "precision": precision,
        "n_missed": (tot_fn - c_fn[n]).astype(int), "max_abs_err": suf_max[n],
        "mean_abs_err": (c_err[-1] - c_err[n]) / max(S, 1),
    })[CURVE_COLS]


def q_star(curve: pd.DataFrame, recall_target: float = 0.995) -> float:
    """Smallest q with recall >= target (NaN if never reached, which can't happen at q=1)."""
    c = curve.sort_values("q")
    hit = c.recall.to_numpy() >= recall_target - 1e-12
    return float(c.q.to_numpy()[hit][0]) if hit.any() else float("nan")


def effective_time_ms(q: np.ndarray, t_genco_ms: float, t_ac_ms: float) -> np.ndarray:
    """Amortised time per scenario of the hybrid: t_genco + q * t_ac."""
    return t_genco_ms + np.asarray(q, np.float64) * t_ac_ms


def oracle_risk(pred_overload: np.ndarray, true_overload: np.ndarray, pred_abs_err: np.ndarray) -> np.ndarray:
    """True decision errors first (pred != true), then by |error|."""
    err = np.nan_to_num(np.asarray(pred_abs_err, np.float64), nan=0.0)
    flip = np.asarray(pred_overload, bool) != np.asarray(true_overload, bool)
    return flip * (err.max() + 1.0) + err


def compare_gates(gates: dict[str, np.ndarray], pred_overload, true_overload, pred_abs_err, budgets,
                  n_random: int = 20, seed: int = 0) -> pd.DataFrame:
    """Long table (gate, q, ...) for each named gate + 'random' (mean over n_random permutations) +
    'oracle' (true decision errors first, then by pred_abs_err)."""
    S = np.asarray(true_overload).size
    out = [budget_curve(r, pred_overload, true_overload, pred_abs_err, budgets, seed).assign(gate=g)
           for g, r in gates.items()]
    rnd = [budget_curve(np.zeros(S), pred_overload, true_overload, pred_abs_err, budgets, seed + 1 + i)
           for i in range(n_random)]
    out.append(pd.concat(rnd).groupby("q", as_index=False, sort=True).mean()[CURVE_COLS].assign(gate="random"))
    out.append(budget_curve(oracle_risk(pred_overload, true_overload, pred_abs_err), pred_overload, true_overload,
                            pred_abs_err, budgets, seed).assign(gate="oracle"))
    return pd.concat(out, ignore_index=True)[["gate"] + CURVE_COLS]


def decision_flip_score(scenario: np.ndarray, z: np.ndarray) -> pd.Series:
    """Per scenario, the smallest score ``z`` that would flip its overload decision.

    ``z`` is one value per in-service branch, (loading - limit) / scale, so ``z > 0`` means "above the limit". A
    scenario predicted overloaded (some ``z > 0``) flips only if every branch above the limit falls below it: the
    maximum of ``z`` over those branches. A scenario predicted safe flips if any branch rises above the limit: the
    minimum of ``-z`` over all branches. Rank scenarios by ``-score`` (smallest first = closest to flipping).
    Returns a Series indexed by scenario id.
    """
    s = pd.Series(np.asarray(z, np.float64))
    over = s.where(s > 0).groupby(np.asarray(scenario)).max()
    under = (-s).groupby(np.asarray(scenario)).min()
    return over.reindex(under.index).fillna(under)

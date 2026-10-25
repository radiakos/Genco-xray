"""Label-free last-step check: flag GENCO buses/branches from the change of the last correction steps alone.

The scores need one GENCO pass, no AC labels and no training; only the flagging threshold is chosen once, on outage
counts inside the training range, and then kept fixed for every larger outage count.

Scores (already in the feature tables, see ``features.py``):
    bus     s_V = nbmax_f_drift_vm = max over the in-service 1-hop neighbours (not the bus itself)
                                    of |Vm_12 - Vm_11| (p.u.); own-bus variant f_drift_vm
    branch  s_L = g_drift_tail_std = std(L_9, ..., L_12) (loading, population std)
"""

from __future__ import annotations

import numpy as np

from .trust import _pct


def recall_threshold(score: np.ndarray, y: np.ndarray, recall: float = 0.9) -> float:
    """Largest threshold ``c`` such that the rule ``score >= c`` flags at least ``recall`` of the positive rows.

    It is the least alarming threshold that meets the recall target on the fitting rows.

    Parameters
    ----------
    score : (N,) float
        Check score, larger = more likely wrong. NaN rows are ignored.
    y : (N,) bool
        Label (true error above the threshold tau).
    recall : float
        Target share of positives flagged, in (0, 1].

    Returns
    -------
    float
        The threshold; NaN when there is no positive row with a finite score.
    """
    s = np.asarray(score, np.float64)
    pos = np.sort(s[np.asarray(y, bool) & np.isfinite(s)])
    if pos.size == 0:
        return float("nan")
    n_flag = int(np.ceil(recall * pos.size - 1e-9))  # positives that must be flagged
    return float(pos[pos.size - n_flag])


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b > 0)


def threshold_metrics(score: np.ndarray, y: np.ndarray, c: float, groups: np.ndarray | None = None,
                      n_boot: int = 0, seed: int = 0) -> dict[str, float]:
    """Recall, precision and flag rate of the rule ``score >= c``, with load-scenario bootstrap CIs.

    Parameters
    ----------
    score, y : (N,)
        Check score and label, as in :func:`recall_threshold`.
    c : float
        Fixed threshold (from :func:`recall_threshold` on the fitting rows).
    groups : (N,) or None
        Load-scenario ids for the cluster bootstrap; required when ``n_boot > 0``.
    n_boot : int
        Bootstrap replicates (0 = no CI).

    Returns
    -------
    dict
        ``recall, recall_lo, recall_hi, precision, precision_lo, precision_hi, flag_rate, n, n_pos, n_flagged``.
    """
    s = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    m = np.isfinite(s)
    flag, yy = s[m] >= c, y[m]
    tp, nf, npos, n = int((flag & yy).sum()), int(flag.sum()), int(yy.sum()), int(m.sum())
    out = {"recall": float(_ratio(tp, npos)), "precision": float(_ratio(tp, nf)), "flag_rate": float(_ratio(nf, n)),
           "n": n, "n_pos": npos, "n_flagged": nf}
    lo = {"recall": (np.nan, np.nan), "precision": (np.nan, np.nan)}
    if n_boot > 0 and npos > 0 and np.isfinite(c):
        # both metrics are ratios of per-group counts: resample groups, sum their counts
        _, gi = np.unique(np.asarray(groups)[m], return_inverse=True)
        G = gi.max() + 1
        cnt = np.stack([np.bincount(gi, w, G) for w in (flag & yy, flag, yy)])  # [3, G]: tp, flagged, positives
        idx = np.random.default_rng(seed).integers(0, G, (n_boot, G))
        b_tp, b_nf, b_pos = cnt[:, idx].sum(-1)
        lo = {"recall": _pct(_ratio(b_tp, b_pos), 0.95), "precision": _pct(_ratio(b_tp, b_nf), 0.95)}
    for name, (a, b) in lo.items():
        out[f"{name}_lo"], out[f"{name}_hi"] = a, b
    return out

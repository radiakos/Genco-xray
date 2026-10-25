"""Split-conformal error bands for branch loading: fixed per branch vs scaled by the expected error.

Calibration and test sets must be disjoint in ``load_scenario`` (not just ``scenario``).
Static baseline follows Alcantara & Chatzivasileiadis (arXiv:2602.07995): per-(element, k-bin) quantiles
of |y - yhat|. Adaptive: one quantile of |y - yhat| / s(x), with s(x) > 0 a difficulty scale from the trajectory.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

SCALE_FLOOR = 1e-6


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile ('higher'
    interpolation). n < ceil(1/alpha) - 1 -> +inf. NaN scores are dropped."""
    s = np.asarray(scores, np.float64)
    s = np.sort(s[np.isfinite(s)])
    n = s.size
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or r > n:
        return float("inf")
    return float(s[r - 1])


def static_stratified(resid_cal: np.ndarray, strata_cal: pd.DataFrame, alpha: float) -> pd.Series:
    """q per stratum (MultiIndex from strata columns, e.g. ['branch', 'k_bin'])."""
    df = strata_cal.reset_index(drop=True).copy()
    df["_r"] = np.abs(np.asarray(resid_cal, np.float64))
    cols = list(strata_cal.columns)
    return df.groupby(cols, sort=True, observed=True)["_r"].agg(lambda r: conformal_quantile(r.to_numpy(), alpha))


def normalized(resid_cal: np.ndarray, scale_cal: np.ndarray, alpha: float) -> float:
    """q such that |y - yhat| <= q * s(x) has 1-alpha coverage. scale must be > 0 (clipped at 1e-6)."""
    s = np.maximum(np.asarray(scale_cal, np.float64), SCALE_FLOOR)
    return conformal_quantile(np.abs(np.asarray(resid_cal, np.float64)) / s, alpha)


def difficulty_scale(risk_score: np.ndarray, resid: np.ndarray | None = None, *,
                     apply_to: np.ndarray | None = None) -> np.ndarray:
    """Map a risk score to a positive scale s(x). Default: exp(standardised risk, clipped to +-3 sd so that extreme
    log-odds cannot produce unbounded widths). If ``resid`` (calibration)
    is given, fit isotonic regression of |resid| on risk and use its prediction (clipped below at its 5th pct).

    The map is fitted on ``risk_score`` (with ``resid``) and evaluated on ``apply_to`` (default: ``risk_score``
    itself), so a scale fitted on one calibration fold can be applied to another fold or to test rows.
    """
    return difficulty_scale_fn(risk_score, resid)(risk_score if apply_to is None else apply_to)


def difficulty_scale_fn(risk_score: np.ndarray, resid: np.ndarray | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """The fitted map behind :func:`difficulty_scale`, as a callable risk -> s(x) > 0."""
    r = np.asarray(risk_score, np.float64)
    ok = np.isfinite(r)
    if resid is None:
        mu, sd = float(np.mean(r[ok])), float(np.std(r[ok]) or 1.0)

        def f(x):
            x = np.asarray(x, np.float64)
            return np.maximum(np.exp(np.clip((np.where(np.isfinite(x), x, mu) - mu) / sd, -3, 3)), SCALE_FLOOR)

        return f
    from sklearn.isotonic import IsotonicRegression

    e = np.abs(np.asarray(resid, np.float64))
    ok &= np.isfinite(e)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(r[ok], e[ok])
    floor = max(float(np.quantile(iso.predict(r[ok]), 0.05)), SCALE_FLOOR)
    med = float(np.median(r[ok]))

    def g(x):
        x = np.asarray(x, np.float64)
        return np.maximum(iso.predict(np.where(np.isfinite(x), x, med)), floor)

    return g


def evaluate_intervals(y: np.ndarray, yhat: np.ndarray, half_width: np.ndarray, threshold: float = 1.0,
                       by: np.ndarray | None = None) -> pd.DataFrame:
    """coverage, mean_width, median_width (width = 2 * half_width), frac_straddle (interval contains
    ``threshold``), n; optionally per group (``by``). Rows with NaN y or yhat are dropped."""
    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    hw = np.broadcast_to(np.asarray(half_width, np.float64), y.shape)
    ok = np.isfinite(y) & np.isfinite(yhat)
    df = pd.DataFrame({
        "covered": (np.abs(y - yhat) <= hw)[ok],
        "width": 2 * hw[ok],
        "straddle": ((yhat - hw <= threshold) & (threshold <= yhat + hw))[ok],
        "by": (np.zeros(y.shape, int) if by is None else np.asarray(by))[ok],
    })
    out = df.groupby("by", sort=True).agg(coverage=("covered", "mean"), mean_width=("width", "mean"),
                                          median_width=("width", "median"), frac_straddle=("straddle", "mean"),
                                          n=("covered", "size"))
    return out.reset_index(drop=by is None)

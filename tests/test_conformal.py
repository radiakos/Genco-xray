"""Acceptance tests for conformal.py (WS-B)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import conformal


def test_quantile_finite_sample():
    assert conformal.conformal_quantile(np.arange(1, 100, dtype=float), 0.1) == 90.0
    assert np.isinf(conformal.conformal_quantile(np.arange(1, 5, dtype=float), 0.1))


def test_marginal_coverage_iid(rng):
    r_cal, r_test = np.abs(rng.normal(size=5000)), np.abs(rng.normal(size=5000))
    q = conformal.conformal_quantile(r_cal, 0.1)
    assert abs((r_test <= q).mean() - 0.9) < 0.02


def test_normalized_is_narrower_under_heteroscedasticity(rng):
    n = 6000
    s = np.exp(rng.normal(0, 1, n))
    resid = np.abs(s * rng.normal(size=n))
    cal, test = slice(0, n // 2), slice(n // 2, n)
    q_static = conformal.conformal_quantile(resid[cal], 0.1)
    q_norm = conformal.normalized(resid[cal], s[cal], 0.1)
    w_static, w_norm = q_static, q_norm * s[test]
    cov_norm = (resid[test] <= w_norm).mean()
    assert abs(cov_norm - 0.9) < 0.02
    assert np.mean(w_norm) < w_static


def test_evaluate_intervals_straddle():
    df = conformal.evaluate_intervals(y=np.array([0.95, 1.2]), yhat=np.array([0.97, 1.1]),
                                      half_width=np.array([0.05, 0.05]), threshold=1.0)
    row = df.iloc[0]
    assert row.coverage == pytest.approx(0.5)  # 1.2 not within 1.1 ± 0.05
    assert row.frac_straddle == pytest.approx(0.5)  # [0.92, 1.02] straddles 1.0


def test_difficulty_scale_isotonic_monotone(rng):
    risk = rng.normal(size=4000)
    resid = np.abs(np.exp(risk) * rng.normal(size=4000))
    f = conformal.difficulty_scale_fn(risk, resid)
    grid = np.linspace(-3, 3, 50)
    assert np.all(np.diff(f(grid)) >= 0) and f(grid).min() > 0
    np.testing.assert_allclose(conformal.difficulty_scale(risk, resid, apply_to=grid), f(grid))


def test_static_stratified_per_stratum():
    strata = pd.DataFrame({"branch": np.repeat([0, 1], 99)})
    resid = np.r_[np.arange(1, 100), 10 * np.arange(1, 100)].astype(float)
    q = conformal.static_stratified(resid, strata, 0.1)
    assert q.loc[0] == 90.0 and q.loc[1] == 900.0

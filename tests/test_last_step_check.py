"""Acceptance tests for last_step_check.py (label-free last-step check)."""

import numpy as np
import pytest

from gencoxray import last_step_check as lsc


def test_recall_threshold_hits_target(rng):
    y = rng.random(5000) < 0.05
    s = y * 2.0 + rng.normal(0, 1, 5000)
    for r in (0.5, 0.9, 1.0):
        c = lsc.recall_threshold(s, y, r)
        got = (s[y] >= c).mean()
        assert got >= r - 1e-12
        # the largest such threshold: one step up loses the target
        above = np.sort(s[y])
        nxt = above[above > c]
        if nxt.size:
            assert (s[y] >= nxt[0]).mean() < r


def test_recall_threshold_ignores_nan_and_empty():
    s = np.array([np.nan, 1.0, 2.0, 3.0])
    y = np.array([True, True, False, True])
    assert lsc.recall_threshold(s, y, 1.0) == 1.0
    assert np.isnan(lsc.recall_threshold(s, np.zeros(4, bool)))


def test_threshold_metrics_counts_and_ci(rng):
    groups = np.repeat(np.arange(300), 10)
    y = rng.random(3000) < 0.1
    s = y * 1.5 + rng.normal(0, 1, 3000)
    s[:5] = np.nan
    c = 0.5
    m = lsc.threshold_metrics(s, y, c, groups, n_boot=300, seed=1)
    ok = np.isfinite(s)
    f = s[ok] >= c
    assert m["n"] == ok.sum() and m["n_flagged"] == f.sum() and m["n_pos"] == y[ok].sum()
    assert m["recall"] == pytest.approx((f & y[ok]).sum() / y[ok].sum())
    assert m["precision"] == pytest.approx((f & y[ok]).sum() / f.sum())
    assert m["flag_rate"] == pytest.approx(f.mean())
    for k in ("recall", "precision"):
        assert m[f"{k}_lo"] <= m[k] <= m[f"{k}_hi"]


def test_threshold_metrics_no_positives_no_ci():
    m = lsc.threshold_metrics(np.arange(10.0), np.zeros(10, bool), 5.0, np.arange(10), n_boot=50)
    assert np.isnan(m["recall"]) and np.isnan(m["recall_lo"]) and m["precision"] == 0.0

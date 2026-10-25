"""Acceptance tests for trust.py (WS-B)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import features, schemas, trust


def test_auroc_basic():
    y = np.array([0, 0, 1, 1])
    assert trust.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert trust.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert trust.auroc(np.array([0.1, np.nan, 0.8, 0.9]), y) == 1.0
    assert np.isnan(trust.auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_bootstrap_ci_orders(rng):
    groups = np.repeat(np.arange(200), 5)
    y = rng.random(1000) < 0.3
    s = y + rng.normal(0, 1, 1000)
    point, lo, hi = trust.bootstrap_ci(trust.auroc, groups, s, y, n_boot=200)
    assert lo <= point <= hi and 0.5 < point < 1.0


def test_delta_auroc_detects_better_score(rng):
    groups = np.repeat(np.arange(300), 4)
    y = rng.random(1200) < 0.3
    good = y + rng.normal(0, 0.3, 1200)
    bad = y + rng.normal(0, 2.0, 1200)
    d, lo, hi = trust.delta_auroc_ci(good, bad, y, groups, n_boot=300)
    assert d > 0 and lo > 0


def test_risk_coverage_endpoints():
    y = np.array([0, 0, 0, 1, 1])
    rc = trust.risk_coverage(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), y.astype(bool))
    full = rc.loc[rc.coverage.idxmax()]
    assert full.risk == pytest.approx(0.4)
    assert rc.loc[np.isclose(rc.coverage, 0.6), "risk"].iloc[0] == pytest.approx(0.0)


def test_scenario_risk_aggregates():
    s = trust.scenario_risk(np.array([1.0, 3.0, 2.0, 5.0]), np.array([7, 7, 8, 8]), how="max")
    assert s.loc[7] == 3.0 and s.loc[8] == 5.0


def test_trust_sign_convention():
    assert trust.risk_to_trust(np.array([1.0]))[0] == -1.0


@pytest.mark.slow
def test_h1_on_synthetic_planted_signal(synth):
    """The planted signal is in the decay rate, so the combiner must beat f_final by > 0.05 AUROC."""
    bus = features.add_bus_features(synth.bus, synth.traces)
    pq = bus[bus.bus_type == "PQ"]
    y = (pq.err_vm > 0.005).to_numpy()
    X = pq[list(schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES)]
    oof, _ = trust.fit_combiner(X, y, pq.load_scenario.to_numpy())  # group by LOAD scenario: no leakage
    a_comb = trust.auroc(oof, y)
    a_final = trust.auroc(pq.f_final.to_numpy(), y)
    assert a_comb - a_final > 0.05, (a_comb, a_final)


def test_weighted_metrics_match_reference(rng):
    # unit weights reproduce auroc/auprc (incl. ties); integer weights reproduce the resampled rows
    s = np.round(rng.normal(size=500), 1)
    y = rng.random(500) < 0.2 + 0.3 * (s > 0)
    srt = trust._Sorted(s, y, np.ones(500, bool))
    w = np.ones((1, 500))
    assert srt.auroc(w)[0] == pytest.approx(trust.auroc(s, y))
    assert srt.auprc(w)[0] == pytest.approx(trust.auprc(s, y))
    c = rng.integers(0, 3, 500)
    idx = np.repeat(np.arange(500), c)
    assert srt.auroc(c[None].astype(float))[0] == pytest.approx(trust.auroc(s[idx], y[idx]))
    assert srt.auprc(c[None].astype(float))[0] == pytest.approx(trust.auprc(s[idx], y[idx]))


def test_group_fold_is_stable_and_balanced():
    g = np.arange(5000)
    f = trust.group_fold(g, 5, seed=0)
    assert np.array_equal(f, trust.group_fold(g, 5, seed=0))
    assert np.array_equal(trust.group_fold(np.array([7, 7, 3]), 5), f[[7, 7, 3]])  # same id -> same fold anywhere
    assert np.bincount(f).min() > 900


def test_shift_scores_never_see_their_load_scenario(rng):
    # a label that is a pure function of the load scenario cannot be learned out-of-fold
    n = 6000
    groups = rng.integers(0, 300, n)
    k = rng.integers(1, 6, n)
    y = (groups % 2) == 0
    X = pd.DataFrame({"g": groups.astype(float)})  # parity of g is not linearly learnable either way
    X["noise"] = rng.normal(size=n)
    s = trust.fit_apply_shift(X, y, k, k_fit_max=2, groups=groups)
    assert np.isfinite(s).all()
    assert abs(trust.auroc(s, y) - 0.5) < 0.05


def test_group_stats_equal_row_weighting(rng):
    # the O(G^2) group path must give exactly the row-weighted AUROC / AP, ties included
    n, G = 3000, 40
    g = rng.integers(0, G, n)
    s = np.round(rng.normal(size=n) + 0.5 * (g % 3), 1)
    y = rng.random(n) < 0.15 + 0.2 * (s > 0.5)
    C = rng.multinomial(G, np.full(G, 1 / G), size=7).astype(float)
    gs = trust._GroupStats(s, y, g, G)
    srt = trust._Sorted(s, y, np.ones(n, bool))
    np.testing.assert_allclose(gs.auroc(C), srt.auroc(C[:, g]), rtol=1e-9)
    np.testing.assert_allclose(gs.auprc(C), srt.auprc(C[:, g]), rtol=1e-5)

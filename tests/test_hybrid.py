"""Acceptance tests for hybrid.py (WS-B)."""

import numpy as np
import pytest

from gencoxray import hybrid


@pytest.fixture
def toy(rng):
    S = 1000
    true = rng.random(S) < 0.3
    flip = rng.random(S) < 0.1
    pred = true ^ flip
    err = np.where(flip, 0.2, 0.01) * rng.random(S)
    return true, pred, err, flip


def test_endpoints(toy):
    true, pred, err, _ = toy
    c = hybrid.budget_curve(np.zeros(len(true)), pred, true, err, np.array([0.0, 1.0]))
    r0 = c.iloc[0]
    assert r0.recall == pytest.approx((pred & true).sum() / true.sum())
    r1 = c.iloc[-1]
    assert r1.recall == 1.0 and r1.n_missed == 0 and r1.max_abs_err == 0.0


def test_monotone_missed(toy, rng):
    true, pred, err, _ = toy
    c = hybrid.budget_curve(rng.random(len(true)), pred, true, err, np.linspace(0, 1, 51))
    assert np.all(np.diff(c.n_missed.to_numpy()) <= 0)


def test_oracle_qstar(toy):
    true, pred, err, flip = toy
    c = hybrid.budget_curve(flip.astype(float), pred, true, err, np.linspace(0, 1, 1001))
    fn_rate = (true & ~pred).sum() / len(true)
    assert hybrid.q_star(c, 1.0) <= flip.mean() + 1e-3
    assert hybrid.q_star(c, 1.0) >= fn_rate - 1e-3


def test_effective_time():
    np.testing.assert_allclose(hybrid.effective_time_ms(np.array([0, 0.5]), 2.0, 10.0), [2.0, 7.0])


def test_trust_gate_beats_random_on_synthetic(synth_oracle):
    sc = synth_oracle.scenario
    true = (sc.max_loading_true > 1).to_numpy()
    pred = (sc.max_loading_pred > 1).to_numpy()
    err = np.abs(sc.max_loading_pred - sc.max_loading_true).to_numpy()
    budgets = np.linspace(0, 1, 201)
    tab = hybrid.compare_gates({"trust": -sc.trust_score.to_numpy()}, pred, true, err, budgets)
    qs = {g: hybrid.q_star(d, 0.995) for g, d in tab.groupby("gate")}
    assert qs["oracle"] <= qs["trust"] < qs["random"]


def test_budget_curve_matches_brute_force(toy, rng):
    true, pred, err, _ = toy
    risk = np.round(rng.random(len(true)), 2)  # many ties
    qs = np.array([0.0, 0.013, 0.25, 0.5, 0.999, 1.0])
    c = hybrid.budget_curve(risk, pred, true, err, qs, seed=3)
    order = hybrid._order(risk, 3)
    for row in c.itertuples():
        res = np.zeros(len(true), bool)
        res[order[: row.n_resolved]] = True
        p = np.where(res, true, pred)
        e = np.where(res, 0.0, err)
        assert row.recall == pytest.approx((p & true).sum() / true.sum())
        assert row.precision == pytest.approx((p & true).sum() / p.sum())
        assert row.n_missed == (true & ~p).sum()
        assert row.max_abs_err == pytest.approx(e.max())
        assert row.mean_abs_err == pytest.approx(e.mean())


def test_decision_flip_score_matches_loop(rng):
    scen = np.repeat(np.arange(50), 8)
    z = rng.normal(size=scen.size) * 2
    got = hybrid.decision_flip_score(scen, z)
    for s in range(50):
        zs = z[scen == s]
        want = zs[zs > 0].max() if (zs > 0).any() else (-zs).min()
        assert got.loc[s] == pytest.approx(want)

"""Acceptance tests for confounders.py (WS-B, milestone M3)."""

import numpy as np
import pandas as pd

from gencoxray import confounders

# path 0-1-2-3, both directions; edge e and e+3 are the same branch
EF = np.array([0, 1, 2, 1, 2, 3])
ET = np.array([1, 2, 3, 0, 1, 2])


def test_hop_distance_path():
    d = confounders.hop_distance(np.stack([EF, ET]), np.array([0]), 4)
    assert d.tolist() == [0, 1, 2, 3]


def test_hop_to_outage_respects_in_service_and_batches():
    in_service = np.array([[True] * 6, [True, False, True, True, False, True]])  # scen 1: branch 1-2 out
    src = np.array([[False, False, False, True], [False, True, True, False]])     # scen 1: endpoints of 1-2
    h = confounders.hop_to_outage(EF, ET, in_service, src)
    np.testing.assert_allclose(h[0], [3, 2, 1, 0])
    np.testing.assert_allclose(h[1], [1, 0, 0, 1])


def test_hop_to_outage_no_source_is_nan():
    h = confounders.hop_to_outage(EF, ET, np.ones((1, 6), bool), np.zeros((1, 4), bool))
    assert np.isnan(h).all()


def test_outage_sources():
    sb = pd.DataFrame({"branch": [0, 1, 2], "from_bus": [0, 1, 2], "to_bus": [1, 2, 3]})
    sc = pd.DataFrame({"scenario": [0, 1], "outaged_branches": [np.array([1]), np.array([], int)],
                       "outaged_gens": [np.array([], int), np.array([0])]})
    m = confounders.outage_sources(sc, sb, gen_bus=np.array([3]), n_bus=4)
    assert m.tolist() == [[False, True, True, False], [False, False, False, True]]


def test_degree_counts_each_neighbour_once():
    # path 0-1-2-3 stored both ways, plus a parallel 1-2 branch
    ei = np.stack([np.r_[EF, 1], np.r_[ET, 2]])
    assert confounders.degree(ei, 4).tolist() == [1, 2, 2, 1]


def test_residualize_removes_confounder_signal(rng):
    n = 4000
    c = pd.DataFrame({"hop": rng.integers(0, 6, n).astype(float)})
    c.loc[:50, "hop"] = np.nan  # imputed + indicator
    signal = rng.normal(size=n)
    f = pd.DataFrame({"a": 2.0 * np.nan_to_num(c.hop.to_numpy(), nan=2.0) + signal, "b": signal})
    f.loc[3, "a"] = np.nan
    train = np.arange(n) % 2 == 0
    r = confounders.residualize(f, c, train)
    assert np.isnan(r.loc[3, "a"])
    ok = r.a.notna() & c.hop.notna()
    assert abs(np.corrcoef(r.a[ok], c.hop[ok])[0, 1]) < 0.05
    assert np.corrcoef(r.a[ok], signal[ok.to_numpy()])[0, 1] > 0.95


def test_stratified_auroc():
    y = np.r_[np.zeros(50, bool), np.ones(50, bool), np.zeros(50, bool), np.ones(5, bool)]
    s = np.r_[np.zeros(50), np.ones(50), np.zeros(50), np.ones(5)]
    g = np.r_[np.zeros(100, int), np.ones(55, int)]
    out = confounders.stratified_auroc(s, y, g, min_pos=20).set_index("stratum")
    assert out.loc[0, "auroc"] == 1.0 and out.loc[0, "n_pos"] == 50
    assert np.isnan(out.loc[1, "auroc"])  # only 5 positives

"""Acceptance tests for ranking.py (WS-C)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import ranking


def test_identical_and_reversed():
    a = pd.Series(np.arange(20, dtype=float))
    assert ranking.spearman(a, a) == pytest.approx(1.0)
    assert ranking.spearman(a, -a) == pytest.approx(-1.0)
    assert ranking.precision_at_k(a, a, 5) == 1.0
    assert ranking.precision_at_k(a, -a, 5) == 0.0
    assert ranking.jaccard_at_k(a, a, 5) == 1.0


def test_partial_overlap():
    a = pd.Series([10, 9, 8, 1, 0], dtype=float)
    b = pd.Series([10, 1, 8, 9, 0], dtype=float)
    assert ranking.precision_at_k(a, b, 3) == pytest.approx(2 / 3)


def test_rank_disagreement_range():
    a = pd.Series(np.arange(10, dtype=float))
    d = ranking.rank_disagreement(a, -a)
    assert d.min() >= 0 and d.max() <= 1 and d.iloc[0] == pytest.approx(d.max())


def test_bootstrap_agreement_genco_beats_dc_on_synthetic(synth):
    g = ranking.bootstrap_agreement(synth.branch, "loading_pred", "loading_true", n_boot=100).set_index("stat")
    d = ranking.bootstrap_agreement(synth.branch, "loading_dc", "loading_true", n_boot=100).set_index("stat")
    assert {"spearman", "precision@10"} <= set(g.index)
    assert g.loc["spearman", "point"] > d.loc["spearman", "point"]
    assert g.loc["spearman", "lo"] <= g.loc["spearman", "point"] <= g.loc["spearman", "hi"]


def test_group_sum_bootstrap_matches_row_weights(rng):
    from gencoxray import stress
    S, E, G = 60, 8, 12
    Lp = rng.uniform(0.6, 1.3, (S, E))
    groups = rng.integers(0, G, S)
    _, gi = np.unique(groups, return_inverse=True)
    sums = ranking._group_sums(Lp, gi, gi.max() + 1)
    c = rng.multinomial(gi.max() + 1, np.full(gi.max() + 1, 1 / (gi.max() + 1))).astype(float)
    fast = (c @ sums["p_overload"]) / (c @ sums["n"])
    np.testing.assert_allclose(fast, stress.mean_stats(Lp, c[gi])["p_overload"])


def test_cvar_bootstrap_is_centred_and_fast_path_exact(synth, rng):
    from gencoxray import stress
    Lp, scen, elem = stress.to_matrix(synth.branch, "loading_dc")
    Lr, _, _ = stress.to_matrix(synth.branch, "loading_true")
    w = rng.integers(0, 3, Lp.shape[0]).astype(float)
    np.testing.assert_allclose(ranking._SortedCvar(Lp, 0.95)(w), stress.cvar(Lp, 0.95, w), rtol=1e-5)
    g = synth.branch.groupby("scenario").load_scenario.first().reindex(scen).to_numpy()
    t = ranking.bootstrap_matrix(Lp, Lr, elem, g, "cvar", n_boot=100)
    r = t.set_index("stat").loc["spearman"]
    # near rho = 1 resampling noise attenuates rho slightly (the CI can sit just below the point); it must stay close
    assert r.lo < r.hi and abs(r.boot_mean - r.point) < 0.01
    assert len(t.attrs["draws"]["spearman"]) == 100  # kept for paired comparisons


def test_positive_only_symmetric_precision():
    a = pd.Series([0, 0, 0, 0.5, 3.0, 2.0, 1.0])  # one spurious positive (index 3)
    b = pd.Series([0, 0, 0, 0, 1.0, 2.0, 3.0])
    out = ranking.agreement(a, b, (25,), positive_only=True)
    assert out["precision@25"] == pytest.approx(3 / 4)  # 3 shared of max(4, 3) positives
    assert "spearman" not in out

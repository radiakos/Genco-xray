"""Acceptance tests for stress.py (WS-C)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import stress


def _toy_branch():
    return pd.DataFrame({
        "scenario": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "branch": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        "in_service": [True, True, True, True, True, True, False, True],
        "loading_pred": np.array([0.5, 1.2, 1.1, 0.9, 0.2, 0.3, np.nan, 1.5], dtype=np.float32),
    })


def test_branch_stress_values():
    st = stress.branch_stress(_toy_branch(), "loading_pred")
    assert st.loc[0, "p_overload"] == pytest.approx(0.5)
    assert st.loc[0, "exp_excess"] == pytest.approx(0.075, abs=1e-6)
    assert st.loc[1, "n"] == 3  # the out-of-service row is excluded
    assert st.loc[1, "p_overload"] == pytest.approx(1 / 3)


def test_bus_stress_values():
    bus = pd.DataFrame({"scenario": np.arange(4, dtype=np.int64), "bus": np.zeros(4, np.int32),
                        "vm_pred": np.array([0.94, 1.0, 1.06, 1.0], dtype=np.float32)})
    st = stress.bus_stress(bus, "vm_pred")
    assert st.loc[0, "p_vviol"] == pytest.approx(0.5)
    assert st.loc[0, "p_under"] == pytest.approx(0.25)


def test_outage_criticality_toy():
    br = _toy_branch()
    sc = pd.DataFrame({"scenario": np.arange(4, dtype=np.int64),
                       "outaged_branches": [np.array([5]), np.array([5]), np.array([6]), np.array([], dtype=int)]})
    oc = stress.outage_criticality(sc, br, "loading_pred")
    # outage 5 -> scenarios 0, 1: n_overload = 0 and 1 -> mean 0.5
    assert oc.loc[5, "mean_n_overload"] == pytest.approx(0.5)
    assert oc.loc[5, "n_scenarios"] == 2
    assert 6 in oc.index


def test_planted_stressed_branches_on_top(synth):
    st = stress.branch_stress(synth.branch, "loading_true")
    planted = set(synth.truth["stressed_branches"].tolist())
    top = set(st.sort_values("p_overload", ascending=False).index[: len(planted)])
    assert len(planted & top) >= len(planted) - 2


def test_weights_equal_replicated_rows(rng):
    L = rng.uniform(0.5, 1.3, (40, 6))
    L[rng.random(L.shape) < 0.1] = np.nan
    c = rng.integers(0, 3, 40)
    rep = np.repeat(L, c, axis=0)
    m_w, m_r = stress.mean_stats(L, c.astype(float)), stress.mean_stats(rep)
    for k in ("p_overload", "exp_excess", "l_mean"):
        np.testing.assert_allclose(m_w[k], m_r[k])
    np.testing.assert_allclose(stress.cvar(L, 0.9, c.astype(float)), stress.cvar(rep, 0.9))


def test_cvar_is_upper_tail_mean():
    L = np.arange(1, 101, dtype=float)[:, None] / 100  # 0.01 .. 1.00
    np.testing.assert_allclose(stress.cvar(L, 0.95), [np.mean(np.arange(96, 101) / 100)])

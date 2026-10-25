"""Tests for reference.py pure functions (implemented; datakit raw tables faked as DataFrames)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import reference


def test_branch_loading_max_over_directions_and_rate_zero():
    pft = np.array([[3.0, 0.0, 4.0, 1.0]])
    qft = np.array([[4.0, 1.0, 3.0, 0.0]])
    ld = reference.branch_loading(pft, qft, edge_branch=np.array([0, 1, 0, 1]), rate_a_mva=np.array([10.0, 0.0]),
                                  n_branch=2)
    np.testing.assert_allclose(ld[0, 0], 0.5)
    assert np.isnan(ld[0, 1])  # rate_a = 0 means unlimited


def _branch_raw():
    return pd.DataFrame({
        "scenario": [0, 0, 0, 1, 1, 1], "load_scenario_idx": [0, 0, 0, 0, 0, 0],
        "pf": [30.0, 0.0, 10.0, 30.0, 5.0, 10.0], "qf": [40.0, 0.0, 0.0, 40.0, 0.0, 0.0],
        "pt": [-30.0, 0.0, -10.0, -30.0, -5.0, -10.0], "qt": [-38.0, 0.0, 0.0, -38.0, 0.0, 0.0],
        "rate_a": [100.0, 50.0, 0.0, 100.0, 50.0, 0.0], "br_status": [1, 0, 1, 1, 1, 1],
        "pf_dc": [35.0, 0.0, 10.0, 35.0, 4.0, 10.0], "pt_dc": [-35.0, 0.0, -10.0, -35.0, -4.0, -10.0],
    })


def test_outages_positions_not_idx():
    gen = pd.DataFrame({"scenario": [0, 0, 1, 1], "in_service": [1, 1, 0, 1]})
    o = reference.outages(_branch_raw(), gen).set_index("scenario")
    assert o.loc[0, "outaged_branches"].tolist() == [1] and o.loc[1, "outaged_branches"].tolist() == []
    assert o.loc[0, "outaged_gens"].tolist() == [] and o.loc[1, "outaged_gens"].tolist() == [0]


def test_dc_and_raw_ac_loading():
    dc = reference.dc_loading(_branch_raw())
    assert dc.loading_dc.iloc[0] == pytest.approx(0.35)
    assert np.isnan(dc.loading_dc.iloc[1]) and np.isnan(dc.loading_dc.iloc[2])  # out of service; rate 0
    ac = reference.ac_loading_raw(_branch_raw())
    assert ac.loading_raw.iloc[0] == pytest.approx(0.5)
    assert ac.branch.tolist() == [0, 1, 2, 0, 1, 2]


def test_dc_absent_returns_empty():
    df = reference.dc_loading(_branch_raw().drop(columns=["pf_dc", "pt_dc"]))
    assert len(df) == 0 and list(df.columns) == ["scenario", "branch", "loading_dc"]


def test_load_scenario_map():
    bus = pd.DataFrame({"scenario": [0, 0, 1, 1, 2, 2], "load_scenario_idx": [7, 7, 7, 7, 8, 8]})
    assert reference.load_scenario_map(bus).to_dict() == {0: 7, 1: 7, 2: 8}


def test_base_offline_excluded_from_k():
    # gen 1 is offline in every scenario (base case, e.g. uncommitted unit); gen 0 is outaged only in scenario 2
    gen = pd.DataFrame({"scenario": [1, 1, 2, 2], "in_service": [1, 0, 0, 0]})
    br = pd.DataFrame({"scenario": [1, 1, 2, 2], "br_status": [1, 1, 1, 0]})
    base_g = reference.base_offline(gen, "in_service")
    assert base_g.tolist() == [1] and reference.base_offline(br, "br_status").tolist() == []
    o = reference.outages(br, gen, base_out_gens=base_g).set_index("scenario")
    assert o.loc[1, "outaged_gens"].tolist() == [] and o.loc[2, "outaged_gens"].tolist() == [0]
    assert o.loc[2, "outaged_branches"].tolist() == [1]

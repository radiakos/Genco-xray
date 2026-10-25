"""Tests for the pure table assembly in runner.py (no torch, no graphkit)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import reference, runner, schemas
from gencoxray.trace import TraceBatch

T, G, B = 3, 2, 3  # 2 graphs, 3 buses, 2 branches (0-1, 1-2) -> 4 directed edges per graph


def _tb():
    bt = np.tile(np.array(["REF", "PQ", "PV"]), G)
    sp, sq = schemas.structural_zero_mask(bt)
    ei_one = np.array([[0, 1, 1, 2], [1, 2, 0, 1]])  # forward block (branches 0,1) then reverse block
    ei = np.concatenate([ei_one, ei_one + B], 1)
    load = np.tile(np.array([0.5, 0.9, 0.6, 1.2]), (T, G))  # branch1 reverse direction is the max (1.2)
    load[:, 4:] = [0.4, np.nan, 0.3, np.nan]  # graph 1: branch 1 out of service
    va = np.zeros((T, G * B))
    va[-1, 1] = 179.0
    in_srv = np.array([True] * 4 + [True, False, True, False])
    return TraceBatch(
        res_p=np.full((T, G * B), 2.0, np.float32), res_q=np.full((T, G * B), 1.0, np.float32),
        vm=np.ones((T, G * B), np.float32), va=va.astype(np.float32),
        pft=np.zeros((T, 8), np.float32), qft=np.zeros((T, 8), np.float32), loading=load.astype(np.float32),
        edge_index=ei, edge_in_service=in_srv, rate_a_mva=np.full(8, 100.0, np.float32),
        bus_graph=np.repeat(np.arange(G), B), bus_local=np.tile(np.arange(B), G), scenario=np.array([10, 11]),
        bus_type=bt, struct_p=sp, struct_q=sq)


def _truth():
    va = np.zeros(G * B)
    va[1] = -179.0
    # truth flows: |S| = 80 on every edge -> loading 0.8 where in service
    return runner.Truth(vm=np.full(G * B, 0.99), va_deg=va, pft=np.full(8, 80.0), qft=np.zeros(8))


def test_batch_tables_shapes_and_values():
    tb = _tb()
    bus, br, sc, tr = runner.batch_tables(tb, _truth(), genco_ms=0.5)
    assert len(bus) == G * B and len(br) == G * 2 and len(sc) == G
    # structural masking: REF has no residual, PV keeps P only
    np.testing.assert_allclose(bus.res_final_mva.to_numpy()[:3], [0.0, np.sqrt(5.0), 2.0], rtol=1e-6)
    # angle error wraps around +-180
    assert bus.err_va_deg.iloc[1] == pytest.approx(2.0)
    # max over directions; out-of-service -> NaN
    np.testing.assert_allclose(br.loading_pred.to_numpy(), [0.6, 1.2, 0.4, np.nan], rtol=1e-6)
    np.testing.assert_allclose(br.loading_true.to_numpy(), [0.8, 0.8, 0.8, np.nan], rtol=1e-6)
    assert br.in_service.tolist() == [True, True, True, False]
    assert sc.n_overload_pred.tolist() == [1, 0] and sc.max_loading_pred.iloc[0] == pytest.approx(1.2)
    assert tr["res_p"].shape == (G, T, B) and tr["loading"].shape == (G, T, 4)
    assert tr["edge_branch"].tolist() == [0, 1, 0, 1] and tr["edge_from"].tolist() == [0, 1, 1, 2]
    schemas.validate_traces(tr)


def test_attach_raw_fills_k_load_scenario_dc():
    bus, br, sc, _ = runner.batch_tables(_tb(), _truth())
    load_map = pd.Series({10: 3, 11: 3})
    out = pd.DataFrame({"scenario": [10, 11], "outaged_branches": [np.array([], np.int32), np.array([1], np.int32)],
                        "outaged_gens": [np.array([], np.int32), np.array([0], np.int32)]})
    dc = pd.DataFrame({"scenario": [10, 10, 11, 11], "branch": np.array([0, 1, 0, 1], np.int32),
                       "loading_dc": np.array([0.7, 1.1, 0.5, np.nan], np.float32)})
    bus, br, sc = runner.attach_raw(bus, br, sc, load_map, out, dc)
    assert sc.k.tolist() == [0, 2] and (bus.load_scenario == 3).all()
    assert sc.max_loading_dc.iloc[0] == pytest.approx(1.1)
    for df, s in ((bus, schemas.BUS), (br, schemas.BRANCH), (sc, schemas.SCENARIO)):
        schemas.validate(df, s)


def _compact_batch():
    """What graphkit (lfe-tutorial) hands over: graph 0 has both branches, graph 1 lost branch 1 (dropped edges)."""
    ei = np.array([[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]])  # g0: fwd 0-1, 1-2, rev 1-0, 2-1 | g1: fwd 3-4, rev 4-3
    pft = np.tile(np.array([60.0, 70.0, -60.0, -70.0, 50.0, -50.0], np.float32), (T, 1))
    tb = _tb()
    tb = runner.dataclasses.replace(
        tb, pft=pft, qft=np.zeros_like(pft), loading=np.zeros_like(pft), edge_index=ei,
        edge_in_service=np.ones(6, bool), rate_a_mva=np.zeros(6, np.float32))  # PF mask zeroes RATE_A
    truth = runner.Truth(vm=np.ones(G * B), va_deg=np.zeros(G * B), pft=pft[-1].astype(float), qft=np.zeros(6))
    return tb, truth


def test_expand_edges_reinserts_outaged_branches():
    tb, truth = _compact_batch()
    status = np.array([[True, True], [True, False]])
    tb2, tr2 = runner.expand_edges(tb, truth, status, np.array([0, 1]), np.array([1, 2]), np.array([100.0, 50.0]))
    assert tb2.edge_index.shape == (2, 8)
    assert tb2.edge_in_service.tolist() == [True] * 4 + [True, False, True, False]
    np.testing.assert_allclose(tb2.loading[-1], [0.6, 1.4, 0.6, 1.4, 0.5, np.nan, 0.5, np.nan], rtol=1e-6)
    np.testing.assert_allclose(tr2.pft, [60, 70, -60, -70, 50, np.nan, -50, np.nan])
    _, br, _, tr = runner.batch_tables(tb2, tr2)
    assert br.in_service.tolist() == [True, True, True, False] and np.isnan(br.loading_pred.iloc[3])
    assert tr["edge_branch"].tolist() == [0, 1, 0, 1] and tr["loading"].shape == (G, T, 4)


def test_expand_edges_rejects_wrong_order():
    tb, truth = _compact_batch()
    with pytest.raises(ValueError, match="directed edges"):  # status implies 8 edges, batch has 6
        runner.expand_edges(tb, truth, np.ones((2, 2), bool), np.array([0, 1]), np.array([1, 2]), np.ones(2))
    with pytest.raises(ValueError, match="from/to mismatch"):  # graph 1 actually lost branch 0, not branch 1
        runner.expand_edges(tb, truth, np.array([[True, True], [False, True]]), np.array([0, 1]),
                            np.array([1, 2]), np.ones(2))


def test_branch_status_matrix():
    raw = pd.DataFrame({"scenario": [5, 5, 7, 7], "br_status": [1, 0, 1, 1]})
    np.testing.assert_array_equal(runner.branch_status_matrix(raw, np.array([7, 5])), [[True, True], [True, False]])
    with pytest.raises(KeyError):
        runner.branch_status_matrix(raw, np.array([6]))


def test_build_static_tables():
    _, _, _, tr = runner.batch_tables(_tb(), _truth())
    raw = pd.DataFrame({"rate_a": [100.0, 50.0]})
    sb, sbr = runner.build_static_tables(tr, raw)
    schemas.validate(sb, schemas.STATIC_BUS)
    schemas.validate(sbr, schemas.STATIC_BRANCH)
    assert sb.degree.tolist() == [1, 2, 1] and sbr.to_bus.tolist() == [1, 2]

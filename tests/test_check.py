"""Acceptance tests for check.py (reusable last-step check) on a synthetic recorded pass."""

import numpy as np
import pytest

from gencoxray import check
from gencoxray.trace import TraceBatch


def _tb(T: int = 6) -> TraceBatch:
    """Two graphs: a 3-bus path (branches 0-1, 1-2) and a 2-bus graph (branch 0-1); forward block then reverse."""
    ei = np.array([[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]])
    nb, ne = 5, ei.shape[1]
    vm = np.ones((T, nb), np.float32)
    vm[-1, 0] += 0.01  # bus 0 still moves
    vm[-1, 4] += 0.002
    load = np.full((T, ne), 0.5, np.float32)
    load[-4:, 0] = [0.5, 0.7, 0.5, 0.7]  # branch 0 of graph 0 oscillates (forward edge)
    load[-4:, 2] = 0.4  # reverse edge of the same branch is lower: max of the two directions is used
    z = np.zeros((T, nb), np.float32)
    return TraceBatch(res_p=z, res_q=z, vm=vm, va=z, pft=load, qft=load, loading=load, edge_index=ei,
                      edge_in_service=np.ones(ne, bool), rate_a_mva=np.full(ne, 100.0, np.float32),
                      bus_graph=np.array([0, 0, 0, 1, 1]), bus_local=np.array([0, 1, 2, 0, 1]),
                      scenario=np.array([7, 8]), bus_type=np.array(["REF", "PQ", "PQ", "REF", "PQ"]),
                      struct_p=np.zeros(nb, bool), struct_q=np.zeros(nb, bool))


def test_neighbour_and_own_scores():
    bus, _ = check.score_trace(_tb(), 0.005, 1.0)
    # neighbours only: bus 1 sees bus 0's change, bus 0 does not see its own
    assert bus.score.tolist() == pytest.approx([0.0, 0.01, 0.0, 0.002, 0.0], abs=1e-6)
    assert bus.flag.tolist() == [False, True, False, False, False]
    own, _ = check.score_trace(_tb(), 0.005, 1.0, voltage_score="own")
    assert own.flag.tolist() == [True, False, False, False, False]


def test_branch_score_pairs_directions():
    _, br = check.score_trace(_tb(), 1.0, 0.05)
    assert len(br) == 3 and br.graph.tolist() == [0, 0, 1] and br.branch.tolist() == [0, 1, 0]
    assert br[["from_bus", "to_bus"]].values.tolist() == [[0, 1], [1, 2], [0, 1]]
    assert br.score.iloc[0] == pytest.approx(np.std([0.5, 0.7, 0.5, 0.7]))
    assert br.flag.tolist() == [True, False, False]


def test_loading_from_flows_uses_raw_ratings_and_ids():
    tb = _tb()
    status = np.array([[True, False, True], [False, True, False]])  # graph 0: branches 0, 2; graph 1: branch 1
    rate = np.array([10.0, 20.0, 40.0])
    assert check.branch_ids(tb, status).tolist() == [0, 2, 1]
    L = check.loading_from_flows(tb, rate, status)
    S = np.hypot(tb.pft, tb.qft)
    np.testing.assert_allclose(L[:, [0, 2]], S[:, [0, 2]] / 10.0)  # branch 0, both directions
    np.testing.assert_allclose(L[:, [1, 3]], S[:, [1, 3]] / 40.0)  # branch 2
    np.testing.assert_allclose(L[:, [4, 5]], S[:, [4, 5]] / 20.0)  # graph 1, branch 1
    with pytest.raises(ValueError):
        check.loading_from_flows(tb, rate, np.ones((2, 3), bool))


def test_rejects_unknown_score_and_short_traces():
    with pytest.raises(ValueError):
        check.score_trace(_tb(), 0.1, 0.1, voltage_score="mean")
    with pytest.raises(ValueError):
        check.score_trace(_tb(T=3), 0.1, 0.1)

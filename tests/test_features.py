"""Acceptance tests for features.py (WS-B). Hand-computed expectations."""

import numpy as np

from gencoxray import features, schemas

T = 12


def _single(r_by_bus, vm=None, va=None, bus_type=("PQ", "PQ", "PQ"), edge_index=None):
    """Build inputs where the residual is purely active power (phi = 0)."""
    r = np.asarray(r_by_bus, dtype=float).T  # [T, B]
    B = r.shape[1]
    sp, sq = schemas.structural_zero_mask(np.array(bus_type))
    vm = np.ones((T, B)) if vm is None else vm
    va = np.zeros((T, B)) if va is None else va
    return features.compute_bus_features(r, np.zeros_like(r), vm, va, sp, sq, edge_index)


def test_residual_magnitude_masks_structural():
    p = np.ones((T, 3)) * 3.0
    q = np.ones((T, 3)) * 4.0
    sp, sq = schemas.structural_zero_mask(np.array(["PQ", "PV", "REF"]))
    r = features.residual_magnitude(p, q, sp, sq)
    np.testing.assert_allclose(r[:, 0], 5.0)
    np.testing.assert_allclose(r[:, 1], 3.0)  # Q structural at PV
    assert np.all(np.isnan(r[:, 2]))  # REF: both structural


def test_geometric_decay():
    t = np.arange(1, T + 1)
    f = _single([2.0 ** (-t), 2.0 ** (-t), 2.0 ** (-t)])
    np.testing.assert_allclose(f["f_final"], 2.0 ** -T, rtol=1e-6)
    np.testing.assert_allclose(f["f_logslope"], -np.log(2), rtol=1e-2)
    np.testing.assert_allclose(f["f_tail_ratio"], 0.5, rtol=1e-2)
    np.testing.assert_allclose(f["f_contraction"], 2.0 ** -(T - 1), rtol=5e-2)
    np.testing.assert_allclose(f["f_nonmono"], 0)
    np.testing.assert_allclose(f["f_overshoot"], 0, atol=1e-12)


def test_stall_and_oscillation():
    stall = np.ones(T)
    osc = np.tile([1.0, 2.0], T // 2)
    f = _single([stall, osc, stall])
    np.testing.assert_allclose(f["f_logslope"][0], 0, atol=1e-9)
    np.testing.assert_allclose(f["f_tail_ratio"][0], 1, atol=1e-9)
    assert f["f_nonmono"][1] == 6
    np.testing.assert_allclose(f["f_overshoot"][1], 1.0)  # ends at 2, min is 1


def test_drift():
    vm = np.ones((T, 3))
    vm[-1, 1] = 1.01
    va = np.zeros((T, 3))
    va[-2, 2] = -0.5
    f = _single([np.ones(T)] * 3, vm=vm, va=va)
    np.testing.assert_allclose(f["f_drift_vm"], [0, 0.01, 0], atol=1e-12)
    np.testing.assert_allclose(f["f_drift_va"], [0, 0, 0.5], atol=1e-12)


def test_ref_bus_residual_features_nan_but_drift_defined():
    f = _single([np.ones(T)] * 3, bus_type=("PQ", "PV", "REF"))
    assert np.isnan(f["f_final"][2]) and np.isnan(f["f_logslope"][2])
    assert not np.isnan(f["f_drift_vm"][2])


def test_neighbour_max_path_graph():
    # path 0-1-2, both directions
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    finals = [1.0, 5.0, 2.0]
    f = _single([np.full(T, v) for v in finals], edge_index=ei)
    np.testing.assert_allclose(f["nbmax_f_final"], [5.0, 2.0, 5.0])


def test_neighbour_max_ignores_nan():
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    f = _single([np.ones(T), np.full(T, 3.0), np.ones(T)], bus_type=("PQ", "PQ", "REF"), edge_index=ei)
    np.testing.assert_allclose(f["nbmax_f_final"][1], 1.0)  # neighbour 2 is REF (NaN) and is ignored


def test_batch_axis_equals_loop(synth):
    tr = synth.traces
    bt = tr["bus_type"]
    sp, sq = schemas.structural_zero_mask(bt)
    ei = np.stack([tr["edge_from"], tr["edge_to"]])
    batched = features.compute_bus_features(tr["res_p"][:5], tr["res_q"][:5], tr["vm"][:5], tr["va"][:5], sp, sq, ei)
    for s in range(5):
        single = features.compute_bus_features(tr["res_p"][s], tr["res_q"][s], tr["vm"][s], tr["va"][s], sp, sq, ei)
        for k in schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES:
            np.testing.assert_allclose(batched[k][s], single[k], equal_nan=True, rtol=1e-6)


def test_branch_features_two_directions():
    # one physical branch stored as 2 directed edges with different loadings; max is taken
    L = np.zeros((T, 2))
    L[:, 0] = np.linspace(0.5, 0.6, T)
    L[:, 1] = 0.55
    L[-1, 0] = 0.7
    bus_f = {f: np.array([1.0, 3.0]) for f in schemas.BUS_FEATURES}
    g = features.compute_branch_features(L, edge_branch=np.array([0, 0]), n_branch=1, branch_from=np.array([0]),
                                         branch_to=np.array([1]), bus_features=bus_f)
    expected_last = 0.7 - max(L[-2, 0], 0.55)
    np.testing.assert_allclose(g["g_drift_last"], [expected_last], rtol=1e-6)
    np.testing.assert_allclose(g["g_endpoint_max_f_final"], [3.0])
    assert set(schemas.BRANCH_FEATURES) <= set(g)


def test_add_features_fills_all_columns(synth):
    bus = features.add_bus_features(synth.bus, synth.traces)
    br = features.add_branch_features(synth.branch, bus, synth.traces, synth.static_branch)
    pq = bus.bus_type == "PQ"
    assert bus.loc[pq, list(schemas.BUS_FEATURES)].notna().all().all()
    assert br.loc[br.in_service, list(schemas.BRANCH_OWN_FEATURES)].notna().all().all()
    schemas.validate(bus, schemas.BUS)
    schemas.validate(br, schemas.BRANCH)


def test_edge_mask_excludes_outaged_neighbours():
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    T_ = 12
    r = [np.full(T_, 1.0), np.full(T_, 2.0), np.full(T_, 9.0)]
    on = np.array([True, True, False, False])  # branch 1-2 out of service (both directions)
    rr = np.asarray(r).T
    sp, sq = schemas.structural_zero_mask(np.array(["PQ", "PQ", "PQ"]))
    f = features.compute_bus_features(rr, np.zeros_like(rr), np.ones((T_, 3)), np.zeros((T_, 3)), sp, sq, ei, on)
    np.testing.assert_allclose(f["nbmax_f_final"], [2.0, 1.0, np.nan])


def test_per_scenario_struct_mask():
    # bus types differ per scenario (generator outage turns PV into PQ): [S, B] masks must match the per-scenario loop
    rng = np.random.default_rng(3)
    p, q = rng.random((2, 2, T, 3)) + 0.1
    types = [np.array(["PQ", "PV", "REF"]), np.array(["PQ", "PQ", "REF"])]
    masks = [schemas.structural_zero_mask(t) for t in types]
    sp, sq = np.stack([m[0] for m in masks]), np.stack([m[1] for m in masks])
    vm, va = np.ones((2, T, 3)), np.zeros((2, T, 3))
    batched = features.compute_bus_features(p, q, vm, va, sp, sq)
    for s in range(2):
        single = features.compute_bus_features(p[s], q[s], vm[s], va[s], *masks[s])
        np.testing.assert_allclose(batched["f_final"][s], single["f_final"], equal_nan=True)
    assert batched["f_final"][0, 1] != batched["f_final"][1, 1]  # Q counts only when the bus is PQ

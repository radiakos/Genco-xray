"""Acceptance tests for newton.py (sparse polar Newton-Raphson) on a small synthetic network."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import newton


def _case(rng, B: int = 6):
    """Ring + chord network; bus 0 REF, bus 1 PV, rest PQ. Returns (bus_raw, branch_raw) in datakit columns."""
    f = np.r_[np.arange(B), 0, 2]
    t = np.r_[(np.arange(B) + 1) % B, 3, 5]
    E = f.size
    z = rng.uniform(0.01, 0.03, E) + 1j * rng.uniform(0.05, 0.15, E)
    ys, bc = 1 / z, 1j * rng.uniform(0.0, 0.05, E)
    yff = ytt = ys + bc / 2
    yft = ytf = -ys
    status = np.ones(E, int)
    status[-1] = 0  # one outage
    br = pd.DataFrame({"from_bus": f, "to_bus": t, "rate_a": 100.0, "br_status": status, "pf": 0.0, "qf": 0.0,
                       "pt": 0.0, "qt": 0.0})
    for k, v in (("Yff", yff), ("Yft", yft), ("Ytf", ytf), ("Ytt", ytt)):
        br[f"{k}_r"], br[f"{k}_i"] = v.real, v.imag
    typ = np.array(["REF", "PV"] + ["PQ"] * (B - 2))
    Pd, Qd = rng.uniform(10, 40, B), rng.uniform(2, 10, B)
    Pd[0] = Qd[0] = 0.0
    bus = pd.DataFrame({"bus": np.arange(B), "Pd": Pd, "Qd": Qd, "Pg": 0.0, "Qg": 0.0, "Vm": 1.0, "Va": 0.0,
                        "PQ": typ == "PQ", "PV": typ == "PV", "REF": typ == "REF", "GS": 0.0, "BS": 0.01})
    bus.loc[1, ["Pg", "Vm"]] = [60.0, 1.02]
    bus.loc[0, "Vm"] = 1.03
    return bus, br


def test_jacobian_matches_finite_differences(rng):
    net = newton.network(*_case(rng))
    vm, va = newton.fix_known(net, 1 + 0.02 * rng.normal(size=6), 0.05 * rng.normal(size=6))
    J = newton.jacobian(net, vm, va).toarray()
    h = 1e-7
    for j, (arr, idx) in enumerate([("va", i) for i in net.ang] + [("vm", i) for i in net.mag]):
        vp, vq = (vm.copy(), va.copy()), (vm.copy(), va.copy())
        (vp[1] if arr == "va" else vp[0])[idx] += h
        (vq[1] if arr == "va" else vq[0])[idx] -= h
        col = (newton.mismatch(net, *vp) - newton.mismatch(net, *vq)) / (2 * h)
        np.testing.assert_allclose(J[:, j], col, atol=1e-5)


def test_solve_flat_start_and_step_at_solution(rng):
    net = newton.network(*_case(rng))
    res = newton.solve(net, *newton.flat_start(net), tol=1e-10)
    assert res["converged"] and 1 <= res["iterations"] <= 10
    assert np.abs(newton.mismatch(net, res["Vm"], res["Va"])).max() < 1e-10
    dvm, dva, g, _ = newton.newton_step(net, res["Vm"], res["Va"])
    assert g < 1e-10 and np.abs(dvm).max() < 1e-9 and np.abs(dva).max() < 1e-9
    # known values held
    assert res["Vm"][1] == pytest.approx(1.02) and res["Vm"][0] == pytest.approx(1.03) and res["Va"][0] == 0.0
    # out-of-service branch has NaN loading
    assert np.isnan(newton.loading(net, res["Vm"], res["Va"])[-1])


def test_newton_step_estimates_small_error(rng):
    net = newton.network(*_case(rng))
    sol = newton.solve(net, *newton.flat_start(net), tol=1e-12)
    vm_hat = sol["Vm"].copy()
    vm_hat[net.mag] += 1e-4 * rng.normal(size=net.mag.size)
    dvm, _, _, _ = newton.newton_step(net, vm_hat, sol["Va"])
    np.testing.assert_allclose(vm_hat[net.mag] + dvm[net.mag], sol["Vm"][net.mag], atol=1e-7)


def test_sigma_min_matches_svd(rng):
    net = newton.network(*_case(rng))
    vm, va = newton.flat_start(net)
    _, _, _, lu = newton.newton_step(net, vm, va)
    J = newton.jacobian(*(net,) + newton.fix_known(net, vm, va)).toarray()
    assert newton.sigma_min(lu, net.n, n_iter=200) == pytest.approx(np.linalg.svd(J, compute_uv=False).min(), rel=1e-6)

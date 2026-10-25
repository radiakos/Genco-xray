"""Sparse polar Newton-Raphson toolkit: mismatch, Jacobian, Newton step from any state, smallest singular value of the
Jacobian, and full solves from flat / DC / GENCO starts.

Same network model as ``sensitivity.ac_case`` (which is dense, for IEEE 118-bus): each scenario's own datakit network,
with the per-branch admittance blocks Yff, Yft, Ytf, Ytt and bus shunts GS, BS (both already in p.u. on the 100 MVA
case base), outaged branches removed, bus types as given to GENCO (``PQ``/``PV``/``REF`` columns of ``bus_data``).
Specified injections are datakit's Pg, Qg minus Pd, Qd; voltage magnitudes at PV/REF buses and the REF angle are held
at datakit's values; reactive limits are not enforced (a plain AC power flow, as in ``sensitivity.ac_solve``).

Unknowns x = (Va at PV+PQ buses [rad], Vm at PQ buses [p.u.]); reduced mismatch g(x) = (P at PV+PQ, Q at PQ) in p.u.
To first order, x_hat - x* = J(x_hat)^-1 g(x_hat), so the Newton step from GENCO's state estimates its state error and
||x_hat - x*|| <= ||g(x_hat)|| / sigma_min(J).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

CASE_BASE_MVA = 100.0


@dataclass
class Network:
    """One scenario's network (bus order = local bus index, branch order = raw row order)."""

    Ybus: sp.csr_matrix  # [B, B] complex
    Yf: sp.csr_matrix  # [E, B] complex: I_from = Yf V
    Yt: sp.csr_matrix  # [E, B]
    br_from: np.ndarray  # [E]
    br_to: np.ndarray  # [E]
    in_service: np.ndarray  # [E] bool
    rate_mva: np.ndarray  # [E]
    Sbus: np.ndarray  # [B] complex specified injection, p.u.
    Vm_set: np.ndarray  # [B] datakit's Vm (used at PV/REF)
    Va_set: np.ndarray  # [B] datakit's Va [rad] (used at REF)
    pv: np.ndarray  # [B] bool
    ref: np.ndarray  # [B] bool
    ang: np.ndarray  # indices of unknown angles (PV + PQ)
    mag: np.ndarray  # indices of unknown magnitudes (PQ)

    @property
    def n(self) -> int:
        return self.ang.size + self.mag.size


def network(bus_raw: pd.DataFrame, branch_raw: pd.DataFrame) -> Network:
    """Build a :class:`Network` from one scenario's datakit rows (``bus_data`` with Pd, Qd, Pg, Qg, Vm, Va, PQ, PV,
    REF, GS, BS; ``branch_data`` with from_bus, to_bus, Y??_r/i, rate_a, br_status)."""
    b = bus_raw.sort_values("bus")
    B = len(b)
    br = branch_raw.reset_index(drop=True)
    E = len(br)
    on = br.br_status.to_numpy() == 1
    f, t = br.from_bus.to_numpy(np.int64), br.to_bus.to_numpy(np.int64)
    y = {k: (br[f"{k}_r"] + 1j * br[f"{k}_i"]).to_numpy() * on for k in ("Yff", "Yft", "Ytf", "Ytt")}
    rows = np.r_[np.arange(E), np.arange(E)]
    Yf = sp.csr_matrix((np.r_[y["Yff"], y["Yft"]], (rows, np.r_[f, t])), shape=(E, B))
    Yt = sp.csr_matrix((np.r_[y["Ytf"], y["Ytt"]], (rows, np.r_[f, t])), shape=(E, B))
    Cf = sp.csr_matrix((np.ones(E), (np.arange(E), f)), shape=(E, B))
    Ct = sp.csr_matrix((np.ones(E), (np.arange(E), t)), shape=(E, B))
    ysh = b.GS.to_numpy() + 1j * b.BS.to_numpy()
    Ybus = (Cf.T @ Yf + Ct.T @ Yt + sp.diags(ysh)).tocsr()
    pv, ref = b.PV.to_numpy().astype(bool), b.REF.to_numpy().astype(bool)
    Sbus = ((b.Pg - b.Pd).to_numpy(float) + 1j * (b.Qg - b.Qd).to_numpy(float)) / CASE_BASE_MVA
    return Network(Ybus, Yf, Yt, f, t, on, br.rate_a.to_numpy(np.float64), Sbus, b.Vm.to_numpy(float),
                   np.deg2rad(b.Va.to_numpy(float)), pv, ref, np.flatnonzero(~ref), np.flatnonzero(~(pv | ref)))


def fix_known(net: Network, Vm: np.ndarray, Va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy of (Vm, Va) with the known values imposed: Vm at PV/REF, Va at REF."""
    Vm, Va = np.array(Vm, np.float64), np.array(Va, np.float64)
    known_m = net.pv | net.ref
    Vm[known_m] = net.Vm_set[known_m]
    Va[net.ref] = net.Va_set[net.ref]
    return Vm, Va


def mismatch(net: Network, Vm: np.ndarray, Va: np.ndarray) -> np.ndarray:
    """Reduced mismatch g(x) = (dP at PV+PQ, dQ at PQ), p.u. on the 100 MVA base."""
    V = Vm * np.exp(1j * Va)
    mis = V * np.conj(net.Ybus @ V) - net.Sbus
    return np.r_[mis.real[net.ang], mis.imag[net.mag]]


def jacobian(net: Network, Vm: np.ndarray, Va: np.ndarray) -> sp.csc_matrix:
    """Reduced polar Jacobian d g / d (Va[ang], Vm[mag]) (MATPOWER dSbus_dV), sparse CSC."""
    V = Vm * np.exp(1j * Va)
    Y = net.Ybus
    Ib = Y @ V
    dV, dI, dVn = sp.diags(V), sp.diags(Ib), sp.diags(V / np.abs(V))
    dS_dVm = dV @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
    dS_dVa = 1j * dV @ np.conj(dI - Y @ dV)
    a, m = net.ang, net.mag
    dVa, dVm = dS_dVa.tocsr(), dS_dVm.tocsr()
    J = sp.bmat([[dVa[a][:, a].real, dVm[a][:, m].real],
                 [dVa[m][:, a].imag, dVm[m][:, m].imag]])
    return J.tocsc()


def newton_step(net: Network, Vm: np.ndarray, Va: np.ndarray, lu=None):
    """One Newton step from (Vm, Va) with the known values imposed.

    Returns (dVm [B], dVa [B] rad, ||g||_inf, lu): zeros at known entries; ``lu`` is the factorisation of J(x)."""
    Vm, Va = fix_known(net, Vm, Va)
    F = mismatch(net, Vm, Va)
    lu = lu or splu(jacobian(net, Vm, Va))
    dx = lu.solve(-F)
    dVm, dVa = np.zeros_like(Vm), np.zeros_like(Va)
    dVa[net.ang] = dx[: net.ang.size]
    dVm[net.mag] = dx[net.ang.size:]
    return dVm, dVa, float(np.max(np.abs(F))), lu


def sigma_min(lu, n: int, n_iter: int = 20, seed: int = 0) -> float:
    """Smallest singular value of J by inverse iteration on (J^T J)^-1 = J^-1 J^-T, from the LU of J."""
    x = np.random.default_rng(seed).normal(size=n)
    x /= np.linalg.norm(x)
    lam = np.nan
    for _ in range(n_iter):
        z = lu.solve(lu.solve(x, trans="T"))
        lam = np.linalg.norm(z)
        x = z / lam
    return float(1.0 / np.sqrt(lam))


def loading(net: Network, Vm: np.ndarray, Va: np.ndarray) -> np.ndarray:
    """[E] max(|S_from|, |S_to|) / rate_a (NaN out of service)."""
    V = Vm * np.exp(1j * Va)
    sf = np.abs(V[net.br_from] * np.conj(net.Yf @ V)) * CASE_BASE_MVA
    st = np.abs(V[net.br_to] * np.conj(net.Yt @ V)) * CASE_BASE_MVA
    rate = np.where(net.rate_mva > 0, net.rate_mva, np.nan)
    return np.where(net.in_service, np.maximum(sf, st) / rate, np.nan)


def solve(net: Network, Vm0: np.ndarray, Va0: np.ndarray, *, tol: float = 1e-8, max_iter: int = 20) -> dict:
    """Newton-Raphson from (Vm0, Va0) (known values imposed) to max |g| < tol.

    Returns dict(Vm, Va, iterations, converged, max_mismatch, seconds); ``iterations`` = Jacobian factorisations.
    A non-finite state counts as not converged."""
    t0 = time.perf_counter()
    Vm, Va = fix_known(net, Vm0, Va0)
    it, ok, F = 0, False, mismatch(net, Vm, Va)
    while True:
        err = float(np.max(np.abs(F))) if np.all(np.isfinite(F)) else np.inf
        if err < tol:
            ok = True
            break
        if it >= max_iter or not np.isfinite(err):
            break
        try:
            dx = splu(jacobian(net, Vm, Va)).solve(-F)
        except RuntimeError:  # singular Jacobian
            break
        Va[net.ang] += dx[: net.ang.size]
        Vm[net.mag] += dx[net.ang.size:]
        it += 1
        F = mismatch(net, Vm, Va)
    return {"Vm": Vm, "Va": Va, "iterations": it, "converged": ok, "max_mismatch": err,
            "seconds": time.perf_counter() - t0}


def flat_start(net: Network) -> tuple[np.ndarray, np.ndarray]:
    """Vm = 1 at PQ buses, setpoints at PV/REF; every angle = the REF angle."""
    va_ref = float(net.Va_set[net.ref][0]) if net.ref.any() else 0.0
    return fix_known(net, np.ones_like(net.Vm_set), np.full_like(net.Va_set, va_ref))


def dc_start(net: Network, va_dc_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angles from datakit's DC power flow (``bus_data.Va_dc``, degrees), magnitudes as in :func:`flat_start`."""
    return fix_known(net, np.ones_like(net.Vm_set), np.deg2rad(np.asarray(va_dc_deg, np.float64)))

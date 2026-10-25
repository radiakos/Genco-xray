"""Sensitivities of branch loading w.r.t. bus loads: GENCO automatic differentiation vs finite-difference AC.

Preliminary evidence only. Caveats for the write-up: ratings are never perturbed in training (no d/d rating);
finite-difference AC is the reference.

AC reference (no extra dependencies): a polar Newton-Raphson power flow in numpy on the scenario's OWN network as
datakit solved it. datakit stores every branch's (perturbed) admittance blocks Yff, Yft, Ytf, Ytt in p.u. plus bus
shunts (GS, BS, already in p.u.: in MW/MVAr they leave a 0.44 p.u. mismatch), so the bus admittance matrix needs no case file. PV/REF voltage magnitudes and
PV generator outputs are held at the scenario's values, the REF bus balances, and reactive limits are not enforced
(as in a plain AC power flow). :func:`ac_solve` must reproduce datakit's own solution (checked in the script).

GENCO side: the PF input carries Pd normalised by the fitted base (x_bus[:, PD_H] = Pd_MW / base_mva), and the
physics decoder uses it for the slack's Pg and the residuals, so d loading / d Pd_MW = autograd(x[:, PD_H]) / base_mva,
through all 12 correction steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CASE_BASE_MVA = 100.0  # datakit / MATPOWER per-unit base of the admittances (not GENCO's fitted normaliser base)


# ----------------------------------------------------------------------------- AC power flow (numpy)
@dataclass
class ACCase:
    """One scenario's network and operating point (bus order = local bus index, branch order = raw row order)."""

    Ybus: np.ndarray  # [B, B] complex, p.u.
    Yf: np.ndarray  # [E, B] complex: I_from = Yf V
    Yt: np.ndarray  # [E, B] complex: I_to = Yt V
    br_from: np.ndarray
    br_to: np.ndarray
    in_service: np.ndarray  # [E] bool
    rate_mva: np.ndarray  # [E]
    Pd: np.ndarray  # [B] MW
    Qd: np.ndarray  # [B] MVAr
    Pg: np.ndarray  # [B] MW (generation per bus)
    Qg: np.ndarray  # [B] MVAr
    Vm: np.ndarray  # [B] p.u. (solution; setpoints at PV/REF)
    Va: np.ndarray  # [B] rad
    pv: np.ndarray  # [B] bool
    ref: np.ndarray  # [B] bool
    loading_ref: np.ndarray  # [E] datakit AC loading (for the reproduction check)


def ac_case(bus_raw, branch_raw) -> ACCase:
    """Build an :class:`ACCase` from one scenario's datakit rows (``bus_data``, ``branch_data``; all columns)."""
    b = bus_raw.sort_values("bus")
    B = len(b)
    br = branch_raw.reset_index(drop=True)
    on = br.br_status.to_numpy() == 1
    f, t = br.from_bus.to_numpy(np.int64), br.to_bus.to_numpy(np.int64)
    yff = (br.Yff_r + 1j * br.Yff_i).to_numpy() * on
    yft = (br.Yft_r + 1j * br.Yft_i).to_numpy() * on
    ytf = (br.Ytf_r + 1j * br.Ytf_i).to_numpy() * on
    ytt = (br.Ytt_r + 1j * br.Ytt_i).to_numpy() * on
    E = len(br)
    Yf = np.zeros((E, B), complex)
    Yt = np.zeros((E, B), complex)
    Yf[np.arange(E), f], Yf[np.arange(E), t] = yff, yft
    Yt[np.arange(E), f], Yt[np.arange(E), t] = ytf, ytt
    ysh = b.GS.to_numpy() + 1j * b.BS.to_numpy()  # datakit stores bus shunts already in p.u. (checked against datakit's AC solution)
    Ybus = np.zeros((B, B), complex)
    np.add.at(Ybus, (f, f), yff)
    np.add.at(Ybus, (f, t), yft)
    np.add.at(Ybus, (t, f), ytf)
    np.add.at(Ybus, (t, t), ytt)
    Ybus[np.arange(B), np.arange(B)] += ysh
    sf = np.hypot(br.pf, br.qf).to_numpy()
    st = np.hypot(br.pt, br.qt).to_numpy()
    rate = br.rate_a.to_numpy(np.float64)
    lref = np.where(on, np.maximum(sf, st) / np.where(rate > 0, rate, np.nan), np.nan)
    return ACCase(Ybus, Yf, Yt, f, t, on, rate, b.Pd.to_numpy(float), b.Qd.to_numpy(float), b.Pg.to_numpy(float),
                  b.Qg.to_numpy(float), b.Vm.to_numpy(float), np.deg2rad(b.Va.to_numpy(float)),
                  b.PV.to_numpy().astype(bool), b.REF.to_numpy().astype(bool), lref)


def ac_solve(case: ACCase, Pd: np.ndarray | None = None, Qd: np.ndarray | None = None, *, tol: float = 1e-10,
             max_iter: int = 30) -> tuple[np.ndarray, bool]:
    """Polar Newton-Raphson from the scenario's solution (warm start). Returns (V complex [B], converged)."""
    Pd = case.Pd if Pd is None else Pd
    Qd = case.Qd if Qd is None else Qd
    Sbus = ((case.Pg - Pd) + 1j * (case.Qg - Qd)) / CASE_BASE_MVA  # Q spec used at PQ buses only
    pq = ~(case.pv | case.ref)
    ang = np.flatnonzero(~case.ref)  # unknown angles: PV + PQ
    mag = np.flatnonzero(pq)  # unknown magnitudes: PQ
    Vm, Va = case.Vm.copy(), case.Va.copy()
    Y = case.Ybus
    for _ in range(max_iter):
        V = Vm * np.exp(1j * Va)
        mis = V * np.conj(Y @ V) - Sbus
        F = np.r_[mis.real[ang], mis.imag[mag]]
        if np.max(np.abs(F)) < tol:
            return V, True
        Ib = Y @ V
        dS_dVa = 1j * np.diag(V) @ np.conj(np.diag(Ib) - Y @ np.diag(V))
        dS_dVm = np.diag(V) @ np.conj(Y @ np.diag(V / Vm)) + np.conj(np.diag(Ib)) @ np.diag(V / Vm)
        J = np.block([[dS_dVa.real[np.ix_(ang, ang)], dS_dVm.real[np.ix_(ang, mag)]],
                      [dS_dVa.imag[np.ix_(mag, ang)], dS_dVm.imag[np.ix_(mag, mag)]]])
        dx = np.linalg.solve(J, -F)
        Va[ang] += dx[: ang.size]
        Vm[mag] += dx[ang.size:]
    V = Vm * np.exp(1j * Va)
    return V, False


def ac_loading(case: ACCase, V: np.ndarray) -> np.ndarray:
    """[E] max(|S_from|, |S_to|) / rate_a (NaN out of service)."""
    sf = np.abs(V[case.br_from] * np.conj(case.Yf @ V)) * CASE_BASE_MVA
    st = np.abs(V[case.br_to] * np.conj(case.Yt @ V)) * CASE_BASE_MVA
    return np.where(case.in_service, np.maximum(sf, st) / np.where(case.rate_mva > 0, case.rate_mva, np.nan), np.nan)


def fd_ac_sensitivity(case: ACCase, scenario: int | None, target_branches: np.ndarray, rel_step: float = 0.01,
                      buses: np.ndarray | None = None) -> np.ndarray:
    """Central finite differences with the numpy AC solver -> [L, B] in 1/MW (NaN for buses not perturbed).

    Each bus's Pd is perturbed by ±rel_step·Pd (at least ±rel_step·1 MW, so load-free buses are covered too);
    Qd is held fixed. ``scenario`` is accepted for API compatibility (the case already is one scenario)."""
    B = case.Pd.size
    buses = np.arange(B) if buses is None else np.asarray(buses)
    out = np.full((len(target_branches), B), np.nan)
    for k in buses:
        h = rel_step * max(abs(case.Pd[k]), 1.0)
        res = []
        for s in (+1, -1):
            Pd = case.Pd.copy()
            Pd[k] += s * h
            V, ok = ac_solve(case, Pd)
            res.append(ac_loading(case, V)[target_branches] if ok else np.full(len(target_branches), np.nan))
        out[:, k] = (res[0] - res[1]) / (2 * h)
    return out


# ----------------------------------------------------------------------------- GENCO autograd (torch)
def loading_sensitivity(loaded, batch, target_branches: np.ndarray, status: np.ndarray,
                        rate_mva: np.ndarray) -> np.ndarray:
    """d loading_l / d Pd_k through GENCO for every graph of ``batch`` -> [G, L, B] in 1/MW.

    target_branches: [L] physical branch ids (raw row order), or [G, L] per graph. status: [G, E] bool in-service flags
    of the batch's scenarios (raw order; graphkit drops out-of-service branches, so edge p of a graph's forward block
    is its p-th in-service branch). rate_mva: [E]. A target that is out of service in a graph gives NaN rows.
    Each graph's loading depends only on its own inputs, so one backward pass per target line serves all graphs.
    """
    import torch
    from gridfm_graphkit.datasets.globals import PD_H
    from gridfm_graphkit.models.utils import ComputeBranchFlow

    from .trace import forward_genco

    model = loaded.task.model
    model.eval()
    xb = batch["bus"].x.detach().clone().requires_grad_(True)
    batch["bus"].x = xb
    with torch.enable_grad():
        out = forward_genco(loaded.task, batch)
        state = out["bus"] if isinstance(out, dict) else out
        ekey = ("bus", "connects", "bus")
        ei, ea = batch.edge_index_dict[ekey], batch.edge_attr_dict[ekey]
        pft, qft = ComputeBranchFlow()(state, ei, ea)
        s_edge = torch.sqrt(pft**2 + qft**2 + 1e-18) * loaded.base_mva  # MVA per directed edge
        bus_graph = batch["bus"].batch.cpu().numpy()
        g_edge = bus_graph[ei[0].cpu().numpy()]
        G = int(bus_graph.max()) + 1
        nb = np.bincount(bus_graph, minlength=G)
        first_bus = np.r_[0, np.cumsum(nb)[:-1]]
        tb = np.broadcast_to(np.asarray(target_branches), (G, np.asarray(target_branches).shape[-1]))
        L, B = tb.shape[1], int(nb[0])
        # directed-edge positions of each (graph, target): forward and reverse edge of the physical branch
        e_pos = np.full((G, L, 2), -1, np.int64)
        for g in range(G):  # G graphs per batch: small loop over graphs, not over buses/branches
            idx = np.flatnonzero(g_edge == g)
            half = idx.size // 2
            inserv = np.flatnonzero(status[g])
            where = {int(br): p for p, br in enumerate(inserv)}
            for li, br in enumerate(tb[g]):
                p = where.get(int(br))
                if p is not None:
                    e_pos[g, li] = (idx[p], idx[half + p])
        grads = np.full((G, L, B), np.nan)
        for li in range(L):
            ok = e_pos[:, li, 0] >= 0
            if not ok.any():
                continue
            ef = torch.as_tensor(e_pos[ok, li, 0], device=s_edge.device)
            er = torch.as_tensor(e_pos[ok, li, 1], device=s_edge.device)
            rate = torch.as_tensor(rate_mva[tb[ok, li]], device=s_edge.device, dtype=s_edge.dtype)
            loading = torch.maximum(s_edge[ef], s_edge[er]) / rate
            (gx,) = torch.autograd.grad(loading.sum(), xb, retain_graph=li < L - 1)
            gpd = gx[:, PD_H].detach().cpu().numpy() / loaded.base_mva  # 1/MW
            for g in np.flatnonzero(ok):
                grads[g, li] = gpd[first_bus[g]: first_bus[g] + B]
    return grads

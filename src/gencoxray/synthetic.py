"""Synthetic dataset with planted structure, matching the real schemas.

The analysis code was developed against this before real recordings existed, and the acceptance tests check that
each method *recovers the planted structure*:

* **Planted difficulty.** Each (scenario, bus) is "hard" with probability
  ``sigmoid(-3 + 0.35*hop_to_pv_ref + 0.25*k + 2.5*in_region)``. Hard buses decay slowly over the
  correction steps (small λ), have larger Vm/Va errors, and drift more at the last step. They also
  start from a smaller, widely spread (log-normal) amplitude ``A``, so the *final* residual alone separates
  hard and easy buses only partly (AUROC ≈ 0.82), while the decay rate separates them well (≈ 0.96).
  So a score built from the correction steps should beat ``f_final``.
* **Planted untrusted region.** The buses within 2 hops of the bus farthest from any voltage-controlled
  bus. Hard probability is much higher there. This also builds in the hop-distance confounder
  on purpose, so confounder controls can be exercised.
* **Planted stressed branches.** ``n_stressed`` branches with base loading ≈ 0.93, half touching the
  untrusted region and half elsewhere (for the stress and accuracy map test). Loading errors at hard endpoints are mostly
  under-estimates (85%), i.e. systematic and dangerous, so GENCO under-ranks stressed lines in the region.

Everything is numpy/pandas/networkx. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from . import schemas


@dataclass
class SyntheticDataset:
    static_bus: pd.DataFrame
    static_branch: pd.DataFrame
    bus: pd.DataFrame
    branch: pd.DataFrame
    scenario: pd.DataFrame
    traces: dict[str, np.ndarray]
    truth: dict[str, np.ndarray] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _multi_source_hops(G: nx.Graph, sources: np.ndarray, n: int) -> np.ndarray:
    d = nx.multi_source_dijkstra_path_length(G, set(int(s) for s in sources))
    return np.array([d.get(i, n) for i in range(n)], dtype=np.int16)


def make_dataset(
    n_bus: int = 60,
    n_scenarios: int = 400,
    n_steps: int = 12,
    k_max: int = 8,
    n_topology_variants: int = 4,
    n_stressed: int = 12,
    seed: int = 0,
    fill_oracle_trust: bool = False,
) -> SyntheticDataset:
    """Build a synthetic dataset in the exact ``results/`` schemas.

    Feature columns (``f_*``, ``nbmax_*``, ``g_*``) are left NaN; fill them with
    ``features.add_bus_features`` / ``features.add_branch_features`` from ``traces``.

    Parameters
    ----------
    n_topology_variants : scenarios per load scenario (datakit convention); ``load_scenario = scenario //
        n_topology_variants``. CV must group by it.
    fill_oracle_trust : if True, fill noisy *oracle* accuracy scores so the stress and accuracy map (joint.py) and
        the hybrid tests can run without the error predictor: ``branch.branch_risk`` = endpoint hardness + noise, and
        ``scenario.trust_score`` = -(max branch_risk over branches near the overload threshold) + noise.
    """
    rng = np.random.default_rng(seed)
    T = n_steps

    # ---------------- grid ----------------
    G = nx.connected_watts_strogatz_graph(n_bus, 4, 0.15, seed=int(rng.integers(2**31 - 1)))
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int32)
    E = len(edges)
    bus_type = np.full(n_bus, "PQ", dtype=object)
    bus_type[0] = "REF"
    pv = rng.choice(np.arange(1, n_bus), size=max(2, n_bus // 8), replace=False)
    bus_type[pv] = "PV"
    struct_p, struct_q = schemas.structural_zero_mask(bus_type)

    hop_pvref = _multi_source_hops(G, np.flatnonzero(bus_type != "PQ"), n_bus)
    hop_ref = _multi_source_hops(G, np.array([0]), n_bus)
    pos = nx.spring_layout(G, seed=seed)
    center = int(np.argmax(hop_pvref))
    region = np.array(sorted(nx.single_source_shortest_path_length(G, center, cutoff=2)), dtype=np.int32)
    in_region = np.zeros(n_bus, bool)
    in_region[region] = True

    static_bus = pd.DataFrame(
        {
            "bus": np.arange(n_bus, dtype=np.int32),
            "bus_type": pd.Categorical(bus_type, categories=schemas.BUS_TYPES),
            "degree": np.array([G.degree(i) for i in range(n_bus)], dtype=np.int16),
            "hop_to_pv_ref": hop_pvref,
            "hop_to_ref": hop_ref,
            "x": np.array([pos[i][0] for i in range(n_bus)], dtype=np.float32),
            "y": np.array([pos[i][1] for i in range(n_bus)], dtype=np.float32),
        }
    )
    rate = rng.uniform(50, 400, E).astype(np.float32)
    static_branch = pd.DataFrame(
        {"branch": np.arange(E, dtype=np.int32), "from_bus": edges[:, 0], "to_bus": edges[:, 1], "rate_a_mva": rate}
    )

    # stressed branches: half touching the region, half not
    touches = in_region[edges[:, 0]] | in_region[edges[:, 1]]
    n_in = min(n_stressed // 2, int(touches.sum()))
    stressed = np.concatenate(
        [
            rng.choice(np.flatnonzero(touches), n_in, replace=False),
            rng.choice(np.flatnonzero(~touches), n_stressed - n_in, replace=False),
        ]
    ).astype(np.int32)
    base_loading = rng.uniform(0.3, 0.7, E)
    base_loading[stressed] = 0.93

    # ---------------- scenarios ----------------
    S = n_scenarios
    k = rng.integers(1, k_max + 1, S).astype(np.int16)
    load_factor = rng.uniform(0.8, 1.05, S)
    outaged = [np.sort(rng.choice(E, int(kk), replace=False)).astype(np.int32) for kk in k]
    in_service = np.ones((S, E), bool)
    for s, o in enumerate(outaged):
        in_service[s, o] = False

    # ---------------- bus trajectories ----------------
    logit = -3.0 + 0.35 * hop_pvref[None, :] + 0.25 * k[:, None] + 2.5 * in_region[None, :]
    hard = rng.random((S, n_bus)) < _sigmoid(logit)
    lam = np.where(hard, rng.uniform(0.08, 0.2, (S, n_bus)), rng.uniform(0.45, 0.7, (S, n_bus)))
    # hard buses start from a smaller amplitude, so the FINAL residual overlaps between classes
    A = rng.lognormal(np.log(5.0) - 2.5 * hard, 1.8, (S, n_bus))
    t = np.arange(1, T + 1)[None, :, None]
    r = A[:, None, :] * np.exp(-lam[:, None, :] * t) + 1e-3
    r *= np.exp(rng.normal(0, 0.05, r.shape))
    phi = rng.uniform(0, 2 * np.pi, (S, n_bus))[:, None, :]
    res_p = (r * np.cos(phi)) * ~struct_p[None, None, :]
    res_q = (r * np.sin(phi)) * ~struct_q[None, None, :]

    vm_known = bus_type != "PQ"  # PV and REF have Vm specified in PF
    va_known = bus_type == "REF"
    vm_true = rng.uniform(0.96, 1.04, (S, n_bus))
    va_true = rng.normal(0, 10, (S, n_bus))
    va_true[:, va_known] = 0.0
    err_vm = np.where(hard, np.abs(rng.normal(0.008, 0.003, (S, n_bus))), np.abs(rng.normal(0, 8e-4, (S, n_bus))))
    err_vm[:, vm_known] = 0.0
    err_va = np.where(hard, np.abs(rng.normal(0.4, 0.15, (S, n_bus))), np.abs(rng.normal(0, 0.05, (S, n_bus))))
    err_va[:, va_known] = 0.0
    sgn = lambda shape: np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    vm_pred = vm_true + sgn((S, n_bus)) * err_vm
    va_pred = va_true + sgn((S, n_bus)) * err_va
    decay = np.exp(-lam[:, None, :] * t) - np.exp(-lam[:, None, :] * T)
    vm_traj = vm_pred[:, None, :] + rng.normal(0, 0.03, (S, 1, n_bus)) * decay * ~vm_known
    va_traj = va_pred[:, None, :] + rng.normal(0, 2.0, (S, 1, n_bus)) * decay * ~va_known

    # ---------------- branches ----------------
    L_true = base_loading[None, :] * load_factor[:, None] * (1 + 0.015 * k[:, None]) + rng.normal(0, 0.03, (S, E))
    h_end = hard[:, edges[:, 0]] | hard[:, edges[:, 1]]
    err_L = np.where(h_end, np.abs(rng.normal(0.08, 0.04, (S, E))), np.abs(rng.normal(0, 0.008, (S, E))))
    # errors at hard endpoints are mostly UNDER-estimates (the dangerous, systematic failure mode)
    sign_L = np.where(h_end, np.where(rng.random((S, E)) < 0.85, -1.0, 1.0), sgn((S, E)))
    L_pred = L_true + sign_L * err_L
    L_dc = L_true + rng.normal(0, 0.1, (S, E))
    for arr in (L_true, L_pred, L_dc, err_L):
        arr[~in_service] = np.nan
    lam_e = np.minimum(lam[:, edges[:, 0]], lam[:, edges[:, 1]])
    decay_e = np.exp(-lam_e[:, None, :] * t) - np.exp(-lam_e[:, None, :] * T)
    L_traj = L_pred[:, None, :] + rng.normal(0, 0.1, (S, 1, E)) * decay_e

    # ---------------- tables ----------------
    sc = np.arange(S, dtype=np.int64)
    ls = sc // n_topology_variants
    r_final = np.sqrt(res_p[:, -1] ** 2 + res_q[:, -1] ** 2)
    bus = pd.DataFrame(
        {
            "scenario": np.repeat(sc, n_bus),
            "load_scenario": np.repeat(ls, n_bus),
            "bus": np.tile(np.arange(n_bus, dtype=np.int32), S),
            "k": np.repeat(k, n_bus),
            "bus_type": pd.Categorical(np.tile(bus_type, S), categories=schemas.BUS_TYPES),
            "vm_pred": vm_pred.ravel().astype(np.float32),
            "vm_true": vm_true.ravel().astype(np.float32),
            "va_pred": va_pred.ravel().astype(np.float32),
            "va_true": va_true.ravel().astype(np.float32),
            "res_p_final_mw": res_p[:, -1].ravel().astype(np.float32),
            "res_q_final_mvar": res_q[:, -1].ravel().astype(np.float32),
            "res_final_mva": r_final.ravel().astype(np.float32),
            "struct_p": np.tile(struct_p, S),
            "struct_q": np.tile(struct_q, S),
            "err_vm": err_vm.ravel().astype(np.float32),
            "err_va_deg": err_va.ravel().astype(np.float32),
            "hop_to_outage": np.full(S * n_bus, np.nan, np.float32),
        }
    )
    schemas.add_nan_columns(bus, schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES)

    branch = pd.DataFrame(
        {
            "scenario": np.repeat(sc, E),
            "load_scenario": np.repeat(ls, E),
            "branch": np.tile(np.arange(E, dtype=np.int32), S),
            "k": np.repeat(k, E),
            "in_service": in_service.ravel(),
            "loading_pred": L_pred.ravel().astype(np.float32),
            "loading_true": L_true.ravel().astype(np.float32),
            "loading_dc": L_dc.ravel().astype(np.float32),
            "err_loading": err_L.ravel().astype(np.float32),
        }
    )
    schemas.add_nan_columns(branch, ("branch_risk",) + schemas.BRANCH_FEATURES)

    nonstruct = ~(struct_p & struct_q)
    trust = np.full(S, np.nan, np.float32)
    if fill_oracle_trust:
        risk_b = h_end.astype(float) + rng.normal(0, 0.15, (S, E))
        risk_b[~in_service] = np.nan
        branch["branch_risk"] = risk_b.ravel().astype(np.float32)
        near = np.abs(np.nan_to_num(L_pred, nan=0.0) - 1.0) < 0.15
        scen_risk = np.where(near, np.nan_to_num(risk_b, nan=-1.0), -1.0).max(1) + rng.normal(0, 0.05, S)
        trust = (-scen_risk).astype(np.float32)  # higher = MORE trustworthy
    scenario = pd.DataFrame(
        {
            "scenario": sc,
            "load_scenario": ls,
            "k": k,
            "outaged_branches": outaged,
            "outaged_gens": [np.zeros(0, np.int32) for _ in range(S)],
            "max_loading_pred": np.nanmax(L_pred, 1).astype(np.float32),
            "max_loading_true": np.nanmax(L_true, 1).astype(np.float32),
            "max_loading_dc": np.nanmax(L_dc, 1).astype(np.float32),
            "n_overload_pred": (np.nan_to_num(L_pred) > 1).sum(1).astype(np.int16),
            "n_overload_true": (np.nan_to_num(L_true) > 1).sum(1).astype(np.int16),
            "vm_viol_pred": ((vm_pred < 0.95) | (vm_pred > 1.05)).sum(1).astype(np.int16),
            "vm_viol_true": ((vm_true < 0.95) | (vm_true > 1.05)).sum(1).astype(np.int16),
            "res_mean_final_mva": r_final[:, nonstruct].mean(1).astype(np.float32),
            "trust_score": trust,
            "genco_ms": np.full(S, 1.0, np.float32),
        }
    )

    # directed edges (both directions), as graphkit stores them
    e_from = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int32)
    e_to = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int32)
    e_branch = np.concatenate([np.arange(E), np.arange(E)]).astype(np.int32)
    traces = {
        "scenario": sc,
        "res_p": res_p.astype(np.float32),
        "res_q": res_q.astype(np.float32),
        "vm": vm_traj.astype(np.float32),
        "va": va_traj.astype(np.float32),
        "loading": np.concatenate([L_traj, L_traj], axis=2).astype(np.float32),
        "bus_type": bus_type.astype(str),
        "edge_from": e_from,
        "edge_to": e_to,
        "edge_branch": e_branch,  # directed edge -> physical branch (graphkit order: forward block, reverse block)
    }

    for df, sch in ((static_bus, schemas.STATIC_BUS), (static_branch, schemas.STATIC_BRANCH),
                    (bus, schemas.BUS), (branch, schemas.BRANCH), (scenario, schemas.SCENARIO)):
        schemas.validate(df, sch)
    schemas.validate_traces(traces)

    truth = {
        "hard": hard,
        "region_buses": region,
        "stressed_branches": stressed,
        "stressed_in_region": stressed[:n_in],
        "lambda": lam,
    }
    return SyntheticDataset(static_bus, static_branch, bus, branch, scenario, traces, truth)

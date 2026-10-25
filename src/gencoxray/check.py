"""Reusable last-step check for GENCO: one pass with recording hooks -> the solution plus per-bus and per-branch flags.

    from gencoxray.check import check_genco_solution, thresholds_from_summary
    v_thr, l_thr = thresholds_from_summary("case118")
    res = check_genco_solution(task, batch, v_thr, l_thr, base_mva=loaded.base_mva, rate_a_mva=rate, in_service=status)
    res.bus[res.bus.flag]          # buses whose voltage GENCO probably got wrong
    res.branch[res.branch.flag]    # branches whose loading GENCO probably got wrong

No AC solve, no labels, no extra GENCO pass. Scores (as in last_step_check.py / features.py):
    bus     max over in-service neighbours (or the bus itself, ``voltage_score="own"``) of |Vm_T - Vm_{T-1}| [p.u.]
    branch  std of the branch loading over the last four steps, loading = max of the two directions
Thresholds: those of the label-free last-step check (90 % recall inside the training range, kept fixed); use
:func:`thresholds_from_summary` for a system already evaluated here, or set your own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trace import TraceBatch


@dataclass
class CheckResult:
    """``bus``: graph, bus (local index), bus_type, vm, va_deg, score, flag. ``branch``: graph, branch (raw branch id),
    from_bus, to_bus (local), loading, score, flag. ``trace``: the full per-step record."""

    bus: pd.DataFrame
    branch: pd.DataFrame
    trace: TraceBatch


def _edge_pairs(tb: TraceBatch) -> tuple[np.ndarray, np.ndarray]:
    """(forward edge, reverse edge) indices of every in-service branch; graphkit stores per graph the forward block
    of the in-service branches, then the reverse block."""
    g_edge = tb.bus_graph[tb.edge_index[0]]
    fw, rv = [], []
    for g in np.unique(g_edge):  # loop over graphs of the batch, not over branches
        idx = np.flatnonzero(g_edge == g)
        half = idx.size // 2
        fw.append(idx[:half])
        rv.append(idx[half:])
    return np.concatenate(fw), np.concatenate(rv)


def branch_ids(tb: TraceBatch, in_service: np.ndarray) -> np.ndarray:
    """Raw branch id of each in-service branch of the batch (order of :func:`_edge_pairs`).

    ``in_service``: [n_graphs, n_branch] bool, raw branch order (e.g. ``runner.branch_status_matrix``)."""
    in_service = np.asarray(in_service, bool)
    return np.concatenate([np.flatnonzero(in_service[g]) for g in range(in_service.shape[0])])


def loading_from_flows(tb: TraceBatch, rate_a_mva: np.ndarray, in_service: np.ndarray) -> np.ndarray:
    """[T, Ne] loading |S_ft| / rate_a from the recorded flows. The PF mask zeroes RATE_A in the batch, so ratings must
    come from the raw branch table (``rate_a_mva`` [n_branch], never perturbed by datakit)."""
    fw, rv = _edge_pairs(tb)
    ids = branch_ids(tb, in_service)
    if ids.size != fw.size:
        raise ValueError(f"{fw.size} in-service branches in the batch, {ids.size} in ``in_service``")
    rate = np.empty(tb.edge_index.shape[1])
    rate[fw] = rate[rv] = np.asarray(rate_a_mva, np.float64)[ids]
    return np.hypot(tb.pft, tb.qft) / np.where(rate > 0, rate, np.nan)


def score_trace(tb: TraceBatch, voltage_threshold: float, loading_threshold: float,
                voltage_score: str = "neighbours", loading: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bus and per-branch last-step scores and flags (``score >= threshold``) from a recorded pass.
    ``loading`` [T, Ne] overrides ``tb.loading`` (use :func:`loading_from_flows` on real graphkit batches)."""
    if tb.n_steps < 4:
        raise ValueError("the branch score needs at least four correction steps")
    own = np.abs(tb.vm[-1] - tb.vm[-2]).astype(np.float64)  # [Nb]
    if voltage_score == "own":
        s_v = own
    elif voltage_score == "neighbours":
        src, dst = tb.edge_index
        on = tb.edge_in_service
        s_v = np.full(own.size, -np.inf)
        np.maximum.at(s_v, dst[on], own[src[on]])
        s_v[np.isneginf(s_v)] = np.nan  # isolated bus
    else:
        raise ValueError(f"voltage_score must be 'neighbours' or 'own', not {voltage_score!r}")
    bus = pd.DataFrame({"graph": tb.bus_graph, "bus": tb.bus_local, "bus_type": tb.bus_type, "vm": tb.vm[-1],
                        "va_deg": tb.va[-1], "score": s_v, "flag": s_v >= voltage_threshold})
    fw, rv = _edge_pairs(tb)
    lo = tb.loading if loading is None else loading
    L = np.fmax(lo[:, fw], lo[:, rv]).astype(np.float64)  # [T, n_branch]
    s_l = L[-4:].std(axis=0)
    g = tb.bus_graph[tb.edge_index[0, fw]]
    pos = np.concatenate([np.arange((g == k).sum()) for k in np.unique(g)])
    branch = pd.DataFrame({"graph": g, "branch": pos, "from_bus": tb.bus_local[tb.edge_index[0, fw]],
                           "to_bus": tb.bus_local[tb.edge_index[1, fw]], "loading": L[-1], "score": s_l,
                           "flag": s_l >= loading_threshold})
    return bus, branch


def check_genco_solution(task, batch, voltage_threshold: float, loading_threshold: float, *, base_mva: float,
                         rate_a_mva: np.ndarray, in_service: np.ndarray,
                         voltage_score: str = "neighbours") -> CheckResult:
    """Run GENCO once on ``batch`` with recording hooks and flag buses and branches by their last-step change.

    ``task``: graphkit task holding the GENCO model (``task.model`` = GNS_heterogeneous, not torch.compile-d).
    ``base_mva``: the checkpoint's fitted per-unit base (``normalizer_stats.pt``), for flows in MVA.
    ``rate_a_mva`` [n_branch] and ``in_service`` [n_graphs, n_branch]: ratings and status in raw branch order (the
    batch carries neither: graphkit drops outaged branches and the PF mask zeroes the ratings).
    Call before anything inverse-transforms ``batch`` in place. ``branch`` in the result is the raw branch id."""
    import torch

    from .trace import TraceRecorder, forward_genco

    rec = TraceRecorder(task.model)
    with torch.inference_mode(), rec:
        forward_genco(task, batch)
        tb = rec.to_trace_batch(batch, base_mva=base_mva)
    bus, branch = score_trace(tb, voltage_threshold, loading_threshold, voltage_score,
                              loading=loading_from_flows(tb, rate_a_mva, in_service))
    branch["branch"] = branch_ids(tb, in_service)
    return CheckResult(bus, branch, tb)


def thresholds_from_summary(grid: str, voltage_score: str = "neighbours") -> tuple[float, float]:
    """(voltage, loading) thresholds of the label-free last-step check for ``grid`` (summary.last_step_check)."""
    from . import io

    s = io.read_summary()["last_step_check"][grid]
    vkey = "last_step_vm_nbmax" if voltage_score == "neighbours" else "last_step_vm_own"
    return (float(s["bus"]["in_dist"][vkey]["threshold"]),
            float(s["branch"]["in_dist"]["last_step_loading"]["threshold"]))

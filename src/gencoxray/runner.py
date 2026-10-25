"""Batched inference over a split → BUS/BRANCH/SCENARIO tables + TRACES shards.

The pure table-assembly functions (:func:`batch_tables`, :func:`attach_raw`) are tested on CPU; :func:`e0_sanity`,
:func:`trace_overhead`, :func:`run_split` and :func:`truth_from_batch` were verified on real data (graphkit
lfe-tutorial @ 9d42d82).

Design (see CLAUDE.md):
* ONE unit conversion point: everything is computed in the model's normalised space (pred AND truth flows with the
  same graphkit ``ComputeBranchFlow``), then multiplied by the fitted ``base_mva``. We deliberately do NOT use
  ``predict_step`` tables: their shape differs between graphkit branches (paper branch: bus dict only, no branch
  table, no clamping of known values), and they inverse-transform the batch in place.
* Feature columns are left NaN here and filled offline from the TRACES shards by scripts/build_features.py
  (restartable without a GPU).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, schemas
from .trace import TraceBatch, _local_index

log = logging.getLogger(__name__)
VM_VIOL_BAND = (0.95, 1.05)
EDGE = ("bus", "connects", "bus")


@dataclass
class Truth:
    """AC ground truth for one batch, aligned with the TraceBatch (physical units)."""

    vm: np.ndarray  # [Nb] p.u.
    va_deg: np.ndarray  # [Nb]
    pft: np.ndarray  # [Ne] MW
    qft: np.ndarray  # [Ne] MVAr


@dataclass
class SplitResult:
    bus: pd.DataFrame
    branch: pd.DataFrame
    scenario: pd.DataFrame
    meta: dict = field(default_factory=dict)


# ============================================================================ pure assembly (tested)
def branch_status_matrix(branch_raw: pd.DataFrame, scenarios: np.ndarray) -> np.ndarray:
    """[len(scenarios), n_branch] bool in-service matrix (raw branch-row order) from datakit ``branch_data`` rows."""
    b = branch_raw[["scenario", "br_status"]]
    n_br = int(b.groupby("scenario").size().iloc[0])
    st = b.br_status.to_numpy().reshape(-1, n_br) == 1  # read_raw sorts by scenario, keeping element order
    order = pd.Index(b.scenario.to_numpy()[::n_br]).get_indexer(scenarios)
    if np.any(order < 0):
        raise KeyError("scenario missing from the raw branch table")
    return st[order]


def expand_edges(tb: TraceBatch, truth: Truth, status: np.ndarray, br_from: np.ndarray, br_to: np.ndarray,
                 rate_a_mva: np.ndarray) -> tuple[TraceBatch, Truth]:
    """Re-insert the out-of-service branches that graphkit drops (``RemoveInactiveBranches``, lfe-tutorial @ 9d42d82).

    In the batch, graph g holds only its in-service branches, as a forward block then a reverse block in raw branch
    order. This returns the same batch with every graph carrying all ``2 * n_branch`` directed edges (forward block =
    raw branch order, reverse block = same order reversed in direction), NaN flows/loading on outaged edges and
    ``edge_in_service`` from ``status``. Verifies that the batch's (from, to) pairs match the raw table at the mapped
    positions.

    Ratings come from the raw table, not the batch: the PF mask (``AddPFHeteroMask``) sets the edge ``RATE_A`` input to
    ``mask_value`` (0), so loading is recomputed here as |S_ft| / rate_a (datakit never perturbs ratings).

    status: [G, n_branch] bool; br_from/br_to: [n_branch] local bus ids; rate_a_mva: [n_branch] MVA.
    """
    G, n_br = status.shape
    B = int(np.bincount(tb.bus_graph, minlength=G)[0])
    full_on = np.concatenate([status, status], 1)  # [G, 2*n_br]: (graph, direction, branch) = batch edge order
    tgt = np.flatnonzero(full_on.ravel())
    if tgt.size != tb.edge_index.shape[1]:
        raise ValueError(f"batch has {tb.edge_index.shape[1]} directed edges, raw in-service branches imply {tgt.size}")
    off = np.repeat(np.arange(G) * B, 2 * n_br)
    ei = np.stack([np.tile(np.r_[br_from, br_to], G), np.tile(np.r_[br_to, br_from], G)]) + off
    if not np.array_equal(ei[:, tgt], tb.edge_index):
        raise ValueError("batch edges do not match raw branch order (from/to mismatch after re-insertion)")
    rate = np.tile(np.r_[rate_a_mva, rate_a_mva], G).astype(np.float32)

    def full(a: np.ndarray) -> np.ndarray:  # [..., Ne] -> [..., G*2*n_br], NaN where out of service
        out = np.full(a.shape[:-1] + (G * 2 * n_br,), np.nan, np.float32)
        out[..., tgt] = a
        return out

    pft, qft = full(tb.pft), full(tb.qft)
    loading = np.hypot(pft, qft) / np.where(rate > 0, rate, np.nan)  # NaN on outaged edges via the flows
    tb2 = dataclasses.replace(tb, pft=pft, qft=qft, loading=loading.astype(np.float32), edge_index=ei,
                              edge_in_service=full_on.ravel(), rate_a_mva=rate)
    return tb2, Truth(vm=truth.vm, va_deg=truth.va_deg, pft=full(truth.pft), qft=full(truth.qft))


def _edge_branch_ids(tb: TraceBatch) -> tuple[np.ndarray, np.ndarray, int]:
    """Per directed edge: graph index and physical branch index (forward block then reverse block per graph)."""
    g_edge = tb.bus_graph[tb.edge_index[0]]
    pos = _local_index(g_edge)
    n_e = np.bincount(g_edge, minlength=len(tb.scenario))
    if np.any(n_e % 2) or np.unique(n_e).size != 1:
        raise ValueError("expected the same even number of directed edges in every graph (forward + reverse blocks)")
    half = int(n_e[0] // 2)
    return g_edge, pos % half, half


def batch_tables(tb: TraceBatch, truth: Truth, genco_ms: float = np.nan) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                                  pd.DataFrame, dict]:
    """Build BUS / BRANCH / SCENARIO rows and one TRACES shard for one batch. Vectorised over the batch.

    ``k``, ``load_scenario``, ``outaged_*`` and ``loading_dc`` are placeholders here; :func:`attach_raw` fills them.
    """
    G = len(tb.scenario)
    nb = np.bincount(tb.bus_graph, minlength=G)
    if np.unique(nb).size != 1:
        raise ValueError("all graphs in a split must have the same bus set (N-k keeps every bus)")
    B, T = int(nb[0]), tb.n_steps
    sc_bus = tb.scenario[tb.bus_graph].astype(np.int64)

    # ---- bus rows
    rp, rq = tb.res_p[-1], tb.res_q[-1]
    rp_m = np.where(tb.struct_p, 0.0, rp)
    rq_m = np.where(tb.struct_q, 0.0, rq)
    r_fin = np.hypot(rp_m, rq_m)
    dva = (tb.va[-1] - truth.va_deg + 180.0) % 360.0 - 180.0
    bus = pd.DataFrame({
        "scenario": sc_bus, "load_scenario": np.full(sc_bus.shape, -1, np.int64),
        "bus": tb.bus_local.astype(np.int32), "k": np.zeros(sc_bus.shape, np.int16),
        "bus_type": pd.Categorical(tb.bus_type, categories=schemas.BUS_TYPES),
        "vm_pred": tb.vm[-1], "vm_true": truth.vm.astype(np.float32),
        "va_pred": tb.va[-1], "va_true": truth.va_deg.astype(np.float32),
        "res_p_final_mw": rp_m.astype(np.float32), "res_q_final_mvar": rq_m.astype(np.float32),
        "res_final_mva": r_fin.astype(np.float32),
        "struct_p": tb.struct_p, "struct_q": tb.struct_q,
        "err_vm": np.abs(tb.vm[-1] - truth.vm).astype(np.float32), "err_va_deg": np.abs(dva).astype(np.float32),
        "hop_to_outage": np.full(sc_bus.shape, np.nan, np.float32),
    })
    schemas.add_nan_columns(bus, schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES)

    # ---- branch rows (max over the two directions)
    g_edge, br, E = _edge_branch_ids(tb)
    gid = g_edge * E + br
    rate = tb.rate_a_mva
    true_load = np.hypot(truth.pft, truth.qft) / np.where(rate > 0, rate, np.nan)
    true_load[~tb.edge_in_service] = np.nan

    def per_branch(edge_vals: np.ndarray) -> np.ndarray:
        out = np.full(G * E, -np.inf)
        np.maximum.at(out, gid, np.where(np.isnan(edge_vals), -np.inf, edge_vals))
        out[np.isinf(out)] = np.nan
        return out

    lp, lt = per_branch(tb.loading[-1]), per_branch(true_load)
    in_srv = np.zeros(G * E, bool)
    np.logical_or.at(in_srv, gid, tb.edge_in_service)
    branch = pd.DataFrame({
        "scenario": np.repeat(tb.scenario.astype(np.int64), E), "load_scenario": np.full(G * E, -1, np.int64),
        "branch": np.tile(np.arange(E, dtype=np.int32), G), "k": np.zeros(G * E, np.int16),
        "in_service": in_srv, "loading_pred": lp.astype(np.float32), "loading_true": lt.astype(np.float32),
        "loading_dc": np.full(G * E, np.nan, np.float32), "err_loading": np.abs(lp - lt).astype(np.float32),
        "branch_risk": np.full(G * E, np.nan, np.float32),
    })
    schemas.add_nan_columns(branch, schemas.BRANCH_FEATURES)

    # ---- scenario rows
    LP, LT = lp.reshape(G, E), lt.reshape(G, E)
    nonstruct = ~(tb.struct_p & tb.struct_q)
    vm_p, vm_t = tb.vm[-1].reshape(G, B), truth.vm.reshape(G, B)
    viol = lambda v: ((v < VM_VIOL_BAND[0]) | (v > VM_VIOL_BAND[1])).sum(1).astype(np.int16)
    r_mean = np.bincount(tb.bus_graph[nonstruct], weights=r_fin[nonstruct], minlength=G) / np.maximum(
        np.bincount(tb.bus_graph[nonstruct], minlength=G), 1)
    with np.errstate(all="ignore"):
        scen = pd.DataFrame({
            "scenario": tb.scenario.astype(np.int64), "load_scenario": np.full(G, -1, np.int64),
            "k": np.zeros(G, np.int16),
            "outaged_branches": [np.zeros(0, np.int32)] * G, "outaged_gens": [np.zeros(0, np.int32)] * G,
            "max_loading_pred": np.nanmax(np.where(np.isnan(LP), -np.inf, LP), 1).astype(np.float32),
            "max_loading_true": np.nanmax(np.where(np.isnan(LT), -np.inf, LT), 1).astype(np.float32),
            "max_loading_dc": np.full(G, np.nan, np.float32),
            "n_overload_pred": (np.nan_to_num(LP) > 1).sum(1).astype(np.int16),
            "n_overload_true": (np.nan_to_num(LT) > 1).sum(1).astype(np.int16),
            "vm_viol_pred": viol(vm_p), "vm_viol_true": viol(vm_t),
            "res_mean_final_mva": r_mean.astype(np.float32),
            "trust_score": np.full(G, np.nan, np.float32), "genco_ms": np.full(G, genco_ms, np.float32),
        })
    for c in ("max_loading_pred", "max_loading_true"):
        scen.loc[np.isinf(scen[c]), c] = np.nan

    # ---- traces shard
    def per_graph(a: np.ndarray, n: int) -> np.ndarray:  # [T, G*n] -> [G, T, n]
        return a.reshape(T, G, n).transpose(1, 0, 2)

    first = np.flatnonzero(tb.bus_graph == 0)
    e0 = g_edge == 0
    traces = {
        "scenario": tb.scenario.astype(np.int64),
        "res_p": per_graph(tb.res_p, B), "res_q": per_graph(tb.res_q, B),
        "vm": per_graph(tb.vm, B), "va": per_graph(tb.va, B), "loading": per_graph(tb.loading, 2 * E),
        "bus_type": tb.bus_type[first].astype(str),
        "edge_from": (tb.edge_index[0, e0] - first[0]).astype(np.int32),
        "edge_to": (tb.edge_index[1, e0] - first[0]).astype(np.int32),
        "edge_branch": np.tile(np.arange(E, dtype=np.int32), 2),
    }
    return bus, branch, scen, traces


def truth_crosscheck(branch: pd.DataFrame, branch_raw: pd.DataFrame) -> dict:
    """Cross-check: our ``loading_true`` (graphkit physics on the AC target state) vs datakit's own AC flows
    (:func:`reference.ac_loading_raw`). Also checks that both mark the same branches out of service."""
    from . import reference

    m = branch.merge(reference.ac_loading_raw(branch_raw), on=["scenario", "branch"], validate="one_to_one")
    if len(m) != len(branch):
        raise ValueError("BRANCH rows and raw branch rows do not align")
    d = (m.loading_true - m.loading_raw).abs().dropna()
    same_nan = bool((m.loading_true.isna() == m.loading_raw.isna()).all())
    out = {"n": int(d.size), "median_abs_diff": float(d.median()), "p99_abs_diff": float(d.quantile(0.99)),
           "max_abs_diff": float(d.max()), "same_out_of_service": same_nan}
    out["passed"] = bool(out["median_abs_diff"] < 1e-3 and same_nan)
    return out


def attach_raw(bus: pd.DataFrame, branch: pd.DataFrame, scen: pd.DataFrame, load_map: pd.Series,
               outage_df: pd.DataFrame, dc_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, ...]:
    """Fill load_scenario, k (= #outaged branches + gens), outaged_* lists and loading_dc / max_loading_dc."""
    o = outage_df.set_index("scenario")
    k = (o.outaged_branches.map(len) + o.outaged_gens.map(len)).astype(np.int16)
    for df in (bus, branch, scen):
        df["load_scenario"] = df.scenario.map(load_map).astype(np.int64)
        df["k"] = df.scenario.map(k).astype(np.int16)
    scen["outaged_branches"] = scen.scenario.map(o.outaged_branches)
    scen["outaged_gens"] = scen.scenario.map(o.outaged_gens)
    if dc_df is not None and len(dc_df):
        branch = branch.drop(columns="loading_dc").merge(dc_df, on=["scenario", "branch"], how="left")
        branch["loading_dc"] = branch.loading_dc.astype(np.float32)
        mx = branch.groupby("scenario").loading_dc.max()
        scen["max_loading_dc"] = scen.scenario.map(mx).astype(np.float32)
    return bus, branch, scen


# ============================================================================ torch side
def _timed(fn, device) -> float:
    """Wall-clock seconds of ``fn()``, CUDA-synchronised before and after on a GPU."""
    import torch

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def truth_from_batch(batch, base_mva: float) -> Truth:
    """AC truth in physical units, computed in normalised space BEFORE any inverse_transform (Va stays in rad there)."""
    import torch
    from gridfm_graphkit.datasets.globals import VA_H, VM_H
    from gridfm_graphkit.models.utils import ComputeBranchFlow

    y = batch.y_dict["bus"]
    state = torch.zeros(y.shape[0], 4, device=y.device, dtype=y.dtype)
    state[:, 0], state[:, 1] = y[:, VM_H], y[:, VA_H]  # VM_OUT = 0, VA_OUT = 1
    pft, qft = ComputeBranchFlow()(state, batch.edge_index_dict[EDGE], batch.edge_attr_dict[EDGE])
    return Truth(vm=y[:, VM_H].float().cpu().numpy(), va_deg=np.rad2deg(y[:, VA_H].float().cpu().numpy()),
                 pft=(pft.float().cpu().numpy() * base_mva), qft=(qft.float().cpu().numpy() * base_mva))


def run_split(loaded, grid_cfg, exp_cfg, split: str, *, max_scenarios: int | None = None,
              write: bool = True) -> SplitResult:
    """Run GENCO with the TraceRecorder over one split; write tables (features NaN) and TRACES shards."""
    import torch

    from . import reference
    from .trace import TraceRecorder, forward_genco

    max_scenarios = max_scenarios or exp_cfg.trace.max_scenarios_per_split
    task, dev = loaded.task, loaded.device
    # graphkit drops out-of-service branches from each graph; the raw branch table restores the full layout.
    raw = loaded.raw_dir
    split_ids = np.sort(np.asarray(loaded.datamodule.test_scenario_ids[0], np.int64))
    br_raw = reference.read_raw(raw, "branch_data", split_ids)
    status_all = branch_status_matrix(br_raw, split_ids)
    one = br_raw[br_raw.scenario == split_ids[0]]
    br_from, br_to = one.from_bus.to_numpy(np.int64), one.to_bus.to_numpy(np.int64)
    br_rate = one.rate_a.to_numpy(np.float64)
    rec = TraceRecorder(task.model)
    if write:
        io.clear_traces(grid_cfg.name, split)
    parts, n_done, shard, t_fwd = [], 0, 0, 0.0
    with torch.inference_mode(), rec:
        for batch in loaded.test_loader():
            batch = batch.to(dev)
            dt = _timed(lambda: forward_genco(task, batch), dev)  # noqa: B023 — called immediately
            if rec.n_calls["node_residuals_layer"] != loaded.n_steps:
                raise RuntimeError(f"hook fired {rec.n_calls} times, expected {loaded.n_steps}")
            tb = rec.to_trace_batch(batch, loaded.base_mva)
            truth = truth_from_batch(batch, loaded.base_mva)
            rec.reset()
            status = status_all[np.searchsorted(split_ids, tb.scenario)]
            tb, truth = expand_edges(tb, truth, status, br_from, br_to, br_rate)
            G = len(tb.scenario)
            t_fwd += dt
            bus, br, sc, tr = batch_tables(tb, truth, genco_ms=1e3 * dt / G)
            if write:
                io.write_traces(tr, grid_cfg.name, split, shard)
            parts.append((bus, br, sc))
            shard += 1
            n_done += G
            if n_done >= max_scenarios:
                break
    bus, br, sc = (pd.concat([p[i] for p in parts], ignore_index=True) for i in range(3))
    scen_ids = sc.scenario.to_numpy()
    bus_raw = reference.read_raw(raw, "bus_data", scen_ids)
    gen_raw = reference.read_raw(raw, "gen_data", split_ids)
    # k counts contingencies only: elements offline in every scenario of the split are base-case state
    base_br, base_gen = reference.base_offline(br_raw, "br_status"), reference.base_offline(gen_raw, "in_service")
    br_raw = br_raw[br_raw.scenario.isin(scen_ids)]
    gen_raw = gen_raw[gen_raw.scenario.isin(scen_ids)]
    out = reference.outages(br_raw, gen_raw, base_out_branches=base_br, base_out_gens=base_gen)
    bus, br, sc = attach_raw(bus, br, sc, reference.load_scenario_map(bus_raw), out, reference.dc_loading(br_raw))
    meta = {"grid": grid_cfg.name, "split": split, "n_scenarios": int(n_done), "n_steps": loaded.n_steps,
            "base_mva": loaded.base_mva, "graphkit_commit": loaded.graphkit_commit,
            "genco_ms_per_instance_unbatched_timing": 1e3 * t_fwd / max(n_done, 1),
            "base_offline": {"branches": base_br.tolist(), "gens": base_gen.tolist()},
            "truth_vs_datakit_loading": truth_crosscheck(br, br_raw)}
    if write:
        for name, df in (("bus", bus), ("branch", br), ("scenario", sc)):
            io.write_table(df, name, grid_cfg.name, split)
        (io.results_dir(grid_cfg.name, split) / "meta.json").write_text(json.dumps(meta, indent=2))
    return SplitResult(bus, br, sc, meta)


def e0_sanity(loaded, n_scenarios: int = 500, atol_mva: float = 1e-3) -> dict:
    """Sanity checks of the recording, run before any analysis. Four checks on real batches:

    a) the residual and branch-flow hooks fire exactly ``n_steps`` times per forward, and the last hooked state equals
       the model's returned prediction ``out["bus"]`` exactly (the trajectory ends at GENCO's answer);
    b) per-step mean ‖PBRes‖ from the hook == ``model.layer_residuals[i]`` (rtol 1e-4) — the hook sees what the loss sees;
    c) the final-step residual recomputed independently from the final state with fresh graphkit layers == hooked one
       (atol ``atol_mva``);
    d) the AC *truth* state has near-zero residual (max < 1 MVA) — validates targets, units and edge attributes.
    """
    import torch
    from gridfm_graphkit.datasets.globals import QG_H, VA_H, VM_H
    from gridfm_graphkit.models.utils import (
        ComputeBranchFlow,
        ComputeNodeInjection,
        ComputeNodeResiduals,
    )
    from torch_scatter import scatter_add

    from .trace import TraceRecorder, forward_genco

    task, dev = loaded.task, loaded.device
    rec = TraceRecorder(task.model)
    a_ok, b_err, c_err, d_err, n = True, 0.0, 0.0, 0.0, 0
    with torch.inference_mode(), rec:
        for batch in loaded.test_loader():
            batch = batch.to(dev)
            out = forward_genco(task, batch)
            calls = rec.n_calls
            a_ok &= calls["node_residuals_layer"] == loaded.n_steps == calls["branch_flow_layer"]
            a_ok &= bool(torch.equal(rec._state[-1], out["bus"][:, :4]))
            res = torch.stack(rec._res)  # [T, Nb, 2] normalised
            step_mean = res.norm(dim=-1).mean(1)
            lr = torch.stack([task.model.layer_residuals[i] for i in range(loaded.n_steps)])
            b_err = max(b_err, float(((step_mean - lr).abs() / lr.abs().clamp_min(1e-12)).max()))
            ei, ea, x = batch.edge_index_dict[EDGE], batch.edge_attr_dict[EDGE], batch.x_dict["bus"]
            nb = x.shape[0]

            def resid(state):
                pft, qft = ComputeBranchFlow()(state, ei, ea)
                p_in, q_in = ComputeNodeInjection()(pft, qft, ei, nb)
                return torch.stack(ComputeNodeResiduals()(p_in, q_in, state, x), -1)

            c_err = max(c_err, float((resid(rec._state[-1]) - res[-1]).abs().max()) * loaded.base_mva)
            y = batch.y_dict["bus"]
            _, g2b = batch.edge_index_dict[("gen", "connected_to", "bus")]
            pg = scatter_add(batch.y_dict["gen"][:, 0], g2b, dim=0, dim_size=nb)  # PG_H = 0 in gen targets
            true_state = torch.stack([y[:, VM_H], y[:, VA_H], pg, y[:, QG_H]], -1)
            d_err = max(d_err, float(resid(true_state).abs().max()) * loaded.base_mva)
            n += int(batch.num_graphs)
            rec.reset()
            if n >= n_scenarios:
                break
    out = {"hook_calls_ok": bool(a_ok), "max_rel_err_step_means": b_err, "max_abs_err_final_mva": c_err,
           "max_abs_truth_residual_mva": d_err, "n_scenarios": n}
    out["passed"] = bool(a_ok and b_err < 1e-4 and c_err < atol_mva and d_err < 1.0)
    log.info("sanity checks: %s", out)
    return out


_THROTTLE_THERMAL_OR_POWER = 0x04 | 0x08 | 0x20 | 0x40 | 0x80  # SW power cap, HW slowdown, SW/HW thermal, power brake


def gpu_state() -> dict:
    """SM clock, temperature and active clock-throttle reasons from ``nvidia-smi`` (empty dict if unavailable).
    A laptop GPU that has been under load for a while can sit at a fraction of its clock (observed: 330 of 2100 MHz)."""
    import subprocess

    q = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    try:
        line = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                       text=True, timeout=10).strip().splitlines()[0]
    except Exception:  # noqa: BLE001 — no NVIDIA driver / tool
        return {}
    sm, sm_max, temp, power, reasons = (v.strip() for v in line.split(","))
    r = int(reasons, 16)
    return {"sm_mhz": float(sm), "sm_max_mhz": float(sm_max), "temp_c": float(temp), "power_w": float(power),
            "throttle_reasons": hex(r), "throttled": bool(r & _THROTTLE_THERMAL_OR_POWER)}


def genco_runtime(loaded, batch_sizes: tuple[int, ...] = (1, 4, 16, 64, 256, 512), n_instances: int = 512,
                   repeats: int = 5) -> dict:
    """Amortised GENCO inference time per instance (paper Fig. 7 protocol: inputs already on the device, warm-up,
    CUDA-synchronised wall clock, no hooks). For each batch size, ``n_instances`` test graphs are re-batched in memory
    and the full pass is timed ``repeats`` times (median). Sizes that run out of GPU memory are skipped."""
    import torch
    from torch_geometric.data import Batch

    from .trace import forward_genco

    task, dev = loaded.task, loaded.device
    ds = loaded.datamodule.test_datasets[0]
    graphs = [ds[i] for i in range(min(n_instances, len(ds)))]
    state_before = gpu_state() if dev.type == "cuda" else {}
    per_bs = {}
    with torch.inference_mode():
        for bs in batch_sizes:
            try:
                batches = [Batch.from_data_list(graphs[i:i + bs]).to(dev) for i in range(0, len(graphs), bs)]
                forward_genco(task, batches[0])  # warm-up
                times = [_timed(lambda: [forward_genco(task, b) for b in batches], dev)  # noqa: B023
                         for _ in range(repeats)]
                per_bs[bs] = 1e3 * float(np.median(times)) / len(graphs)
            except torch.OutOfMemoryError:
                log.info("batch size %d: out of GPU memory, skipped", bs)
                break
            finally:
                batches = None
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    best = min(per_bs, key=per_bs.get)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    state_after = gpu_state() if dev.type == "cuda" else {}
    valid = not (state_before.get("throttled") or state_after.get("throttled"))
    if not valid:
        log.warning("GPU was thermally/power throttled during the runtime sweep: %s -> %s", state_before, state_after)
    return {"device": name, "n_instances": len(graphs), "ms_per_instance_by_batch": per_bs,
            "best_batch_size": best, "ms_per_instance": per_bs[best], "torch_compile": False,
            "gpu_state_before": state_before, "gpu_state_after": state_after, "valid": valid}


def ac_runtime(raw_dirs: list) -> dict:
    """AC/DC per-instance solve time recorded by datakit while generating the ground truth (``runtime_data``,
    seconds per scenario, one solver process). Median and mean over all scenarios of the given splits."""
    import pyarrow.dataset as pads

    t = pd.concat([pads.dataset(Path(r) / "runtime_data.parquet", format="parquet", partitioning="hive")
                   .to_table(columns=["ac", "dc"]).to_pandas() for r in raw_dirs], ignore_index=True)
    return {"n_scenarios": len(t), "ac_ms_median": 1e3 * float(t.ac.median()),
            "ac_ms_mean": 1e3 * float(t.ac.mean()), "dc_ms_median": 1e3 * float(t.dc.median()),
            "dc_ms_mean": 1e3 * float(t.dc.mean()), "source": "datakit runtime_data (PowerModels, 1 process)"}


def check_split(grid: str, split: str, batch_size: int, expected_k: int | None = None) -> dict:
    """Acceptance check for one recorded split (reads ``results/`` only; numpy/pandas).

    Tables validate against their schemas (on read); one TRACES shard per batch, in scenario order and covering exactly
    the SCENARIO table; row counts = scenarios × buses / branches; the truth cross-check in ``meta.json`` passed; and,
    if ``expected_k`` is given (Texas ``test_k``), every scenario has exactly k outages.
    """
    bus, br, sc = (io.read_table(t, grid, split) for t in ("bus", "branch", "scenario"))
    sb, sbr = io.read_table("static_bus", grid), io.read_table("static_branch", grid)
    meta = json.loads((io.results_dir(grid, split) / "meta.json").read_text())
    shard_sc = [s["scenario"] for s in io.iter_traces(grid, split)]
    n, B, E = len(sc), len(sb), len(sbr)
    out = {
        "n_scenarios": n, "n_shards": len(shard_sc),
        "shards_ok": len(shard_sc) == -(-n // batch_size)
        and np.array_equal(np.concatenate(shard_sc), sc.scenario.to_numpy()),
        "bus_rows_ok": len(bus) == n * B, "branch_rows_ok": len(br) == n * E,
        "truth_crosscheck_ok": bool(meta["truth_vs_datakit_loading"]["passed"]),
        "k_counts": {int(k): int(c) for k, c in sc.k.value_counts().sort_index().items()},
    }
    if expected_k is not None:
        out["k_ok"] = bool((sc.k == expected_k).all())
    out["passed"] = all(v for key, v in out.items() if key.endswith("_ok"))
    return out


def trace_overhead(loaded, n_batches: int = 20, warmup: int = 3) -> dict:
    """Wall-clock cost of recording relative to a plain forward pass.

    Batches are cached on the device first, so data loading is excluded. Each batch is timed (CUDA-synchronised)
    three ways: plain forward; forward with the hooks registered (``hook_overhead``, acceptance criterion < 15 %); and
    forward + hooks + :meth:`TraceRecorder.to_trace_batch` (GPU→CPU copy and unit conversion; ``extract_overhead``,
    reported for transparency; during recording it overlaps with writing shards). Medians over batches after
    ``warmup``.
    """
    import torch

    from .trace import TraceRecorder, forward_genco

    task, dev = loaded.task, loaded.device
    batches = []
    for b in loaded.test_loader():
        batches.append(b.to(dev))
        if len(batches) >= n_batches + warmup:
            break

    rec = TraceRecorder(task.model)
    state_before = gpu_state() if dev.type == "cuda" else {}
    t_plain, t_hook, t_extract = [], [], []
    with torch.inference_mode():
        for i, b in enumerate(batches):
            def fwd(b=b):
                return forward_genco(task, b)

            def fwd_extract(b=b):
                fwd(b)
                return rec.to_trace_batch(b, loaded.base_mva)

            tp = _timed(fwd, dev)
            with rec:
                th = _timed(fwd, dev)
                rec.reset()
                te = _timed(fwd_extract, dev)
                rec.reset()
            if i >= warmup:
                t_plain.append(tp), t_hook.append(th), t_extract.append(te)
    med = {k: float(np.median(v)) for k, v in (("plain", t_plain), ("hook", t_hook), ("extract", t_extract))}
    out = {"n_batches": len(t_plain), "batch_size": int(batches[0].num_graphs),
           "forward_ms_per_batch": 1e3 * med["plain"], "hooked_ms_per_batch": 1e3 * med["hook"],
           "extract_ms_per_batch": 1e3 * med["extract"],
           "hook_overhead": med["hook"] / med["plain"] - 1, "extract_overhead": med["extract"] / med["plain"] - 1}
    out["gpu_state_before"], out["gpu_state_after"] = state_before, gpu_state() if dev.type == "cuda" else {}
    out["passed"] = bool(out["hook_overhead"] < 0.15)
    log.info("trace overhead: %s", out)
    return out


def build_static_tables(traces: dict, branch_raw_one_scenario: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STATIC_BUS / STATIC_BRANCH from one TRACES shard + the raw branch rows of one scenario (pure).

    Hop distances are written as -1 and refreshed by build_features (``confounders.hop_distance``); x/y are NaN until
    the stress and accuracy map script fills them (``viz.grid_layout.layout``). ``rate_a_mva`` comes from datakit (MVA).
    """
    bt = np.asarray(traces["bus_type"]).astype(str)
    n = bt.shape[0]
    ef, et = np.asarray(traces["edge_from"]), np.asarray(traces["edge_to"])
    half = ef.shape[0] // 2
    deg = np.bincount(np.r_[ef[:half], et[:half]], minlength=n).astype(np.int16)
    sb = pd.DataFrame({"bus": np.arange(n, dtype=np.int32),
                       "bus_type": pd.Categorical(bt, categories=schemas.BUS_TYPES), "degree": deg,
                       "hop_to_pv_ref": np.full(n, -1, np.int16), "hop_to_ref": np.full(n, -1, np.int16),
                       "x": np.full(n, np.nan, np.float32), "y": np.full(n, np.nan, np.float32)})
    b = branch_raw_one_scenario.reset_index(drop=True)
    if len(b) != half:
        raise ValueError(f"raw branch rows ({len(b)}) != directed edges / 2 ({half}); edge order assumption broken")
    sbr = pd.DataFrame({"branch": np.arange(half, dtype=np.int32), "from_bus": ef[:half].astype(np.int32),
                        "to_bus": et[:half].astype(np.int32), "rate_a_mva": b.rate_a.to_numpy(np.float32)})
    return sb, sbr

"""Hook-based recording of GENCO's intermediate solution at every correction step. Verified on graphkit
lfe-tutorial @ 9d42d82 (the sanity checks pass on case118 and Texas2k).

Never modify graphkit; hooks only. Hook points on ``GNS_heterogeneous`` (identical on ``main`` and
``genco-paper-repro``; checked 19 Oct 2026):

* ``node_residuals_layer`` (ComputeNodeResiduals) — fires once per step (N times, including the last).
  ``forward(P_in, Q_in, bus_data_pred, bus_data_orig) -> (residual_P, residual_Q)``.
  ``bus_data_pred[:, :4]`` = [Vm, Va(rad), Pg, Qg] after the physics decoder = the full per-step state.
* ``branch_flow_layer`` (ComputeBranchFlow) — fires once per step. ``forward(bus_data, edge_index, edge_attr)
  -> (Pft, Qft)`` per directed edge.
* ``physics_mlp`` (residual encoder) — fires N-1 times with input ``bus_residuals[Nb, 2]``. Cross-check only.

graphkit's own evaluation (``test_step`` / ``predict_step``) builds NEW ComputeBranchFlow/ComputeNodeResiduals
instances, so our hooks never double-fire during evaluation code.

Pitfalls:
* Don't trace a ``torch.compile``-d model: compiled graphs can bypass Python hooks.
* Everything captured is in the model's normalised space: powers / fitted ``baseMVA`` (NOT 100), angles in rad.
  Call :meth:`TraceRecorder.to_trace_batch` BEFORE anything inverse-transforms the batch in place
  (``predict_step`` / ``test_step`` do), because rate_a is read from the normalised edge attributes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import schemas

RATE_A, B_ON = 9, 10  # gridfm_graphkit.datasets.globals (edge feature indices; same on both branches)


@dataclass
class TraceBatch:
    """Per-step trajectory for one batch of graphs, physical units, numpy on CPU.

    Shapes: T = steps, Nb = buses in batch (all graphs concatenated), Ne = directed edges in batch.
    """

    res_p: np.ndarray  # [T, Nb] MW
    res_q: np.ndarray  # [T, Nb] MVAr
    vm: np.ndarray  # [T, Nb] p.u.
    va: np.ndarray  # [T, Nb] degrees
    pft: np.ndarray  # [T, Ne] MW (sending end)
    qft: np.ndarray  # [T, Ne] MVAr
    loading: np.ndarray  # [T, Ne] |S_ft| / rate_a (NaN where rate_a <= 0 or the edge is out of service)
    edge_index: np.ndarray  # [2, Ne] batch-global bus indices
    edge_in_service: np.ndarray  # [Ne] bool
    rate_a_mva: np.ndarray  # [Ne]
    bus_graph: np.ndarray  # [Nb] graph index within batch
    bus_local: np.ndarray  # [Nb] local bus index within its graph
    scenario: np.ndarray  # [n_graphs] scenario ids
    bus_type: np.ndarray  # [Nb] str in {PQ, PV, REF}
    struct_p: np.ndarray  # [Nb] bool
    struct_q: np.ndarray  # [Nb] bool

    @property
    def n_steps(self) -> int:
        return self.res_p.shape[0]


def _local_index(group: np.ndarray) -> np.ndarray:
    """Position of each element within its (contiguous, ordered) group. Same as graphkit's local_index_per_graph."""
    n = group.shape[0]
    if n == 0:
        return group.copy()
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    first = np.repeat(starts, np.diff(np.r_[starts, n]))
    return np.arange(n) - first


class TraceRecorder:
    """Context manager that registers forward hooks and collects per-step tensors.

    Usage::

        rec = TraceRecorder(task.model)
        with torch.inference_mode(), rec:
            for batch in loader:
                out = forward_genco(task, batch)
                tb = rec.to_trace_batch(batch, base_mva=loaded.base_mva)
                rec.reset()
    """

    def __init__(self, model: Any, *, record_branch: bool = True, record_encoder_input: bool = False):
        for name in ("node_residuals_layer", "branch_flow_layer", "physics_mlp"):
            if not hasattr(model, name):
                raise AttributeError(f"model has no '{name}'; is this GNS_heterogeneous? Update trace.py + CLAUDE.md.")
        self.model = model
        self.record_branch = record_branch
        self.record_encoder_input = record_encoder_input
        self._handles: list = []
        self.reset()

    # ----------------------------------------------------------------- hooks
    def _on_residuals(self, _m, args, kwargs, out):
        state = args[2] if len(args) > 2 else kwargs["bus_data_pred"]
        self._state.append(state[:, :4].detach())
        self._res.append(_stack(out, -1).detach())  # [Nb, 2]
        self._calls["node_residuals_layer"] += 1

    def _on_flows(self, _m, _args, out):
        self._flows.append(_stack(out, 0).detach())  # [2, Ne]
        self._calls["branch_flow_layer"] += 1

    def _on_encoder(self, _m, _args):
        self._calls["physics_mlp"] += 1

    def __enter__(self) -> TraceRecorder:
        m = self.model
        self._handles = [m.node_residuals_layer.register_forward_hook(self._on_residuals, with_kwargs=True)]
        if self.record_branch:
            self._handles.append(m.branch_flow_layer.register_forward_hook(self._on_flows))
        if self.record_encoder_input:
            self._handles.append(m.physics_mlp.register_forward_pre_hook(self._on_encoder))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self) -> None:
        """Clear collected tensors AND the call counters (hooks stay registered)."""
        self._res: list = []
        self._state: list = []
        self._flows: list = []
        self._calls = {"node_residuals_layer": 0, "branch_flow_layer": 0, "physics_mlp": 0}

    @property
    def n_calls(self) -> dict[str, int]:
        return dict(self._calls)

    # ----------------------------------------------------------------- assembly
    def to_trace_batch(self, batch: Any, base_mva: float, rate_a_mva: np.ndarray | None = None) -> TraceBatch:
        """Stack per-step lists to [T, ...], convert to physical units, build masks. Call before inverse_transform."""
        import torch

        if not self._res:
            raise RuntimeError("no steps recorded (StateEstimation skips the loop; torch.compile can bypass hooks)")
        res = torch.stack(self._res).float().cpu().numpy()  # [T, Nb, 2]
        st = torch.stack(self._state).float().cpu().numpy()  # [T, Nb, 4]
        ekey = ("bus", "connects", "bus")
        ei = batch.edge_index_dict[ekey].cpu().numpy()
        ea = batch.edge_attr_dict[ekey]
        n_e = ei.shape[1]
        if rate_a_mva is None:
            rate_a_mva = ea[:, RATE_A].float().cpu().numpy() * base_mva
        in_service = (ea[:, B_ON].float().cpu().numpy() > 0.5) if ea.shape[1] > B_ON else np.ones(n_e, bool)
        if self._flows:
            fl = torch.stack(self._flows).float().cpu().numpy() * base_mva  # [T, 2, Ne]
            pft, qft = fl[:, 0], fl[:, 1]
        else:
            pft = qft = np.full((res.shape[0], n_e), np.nan, np.float32)
        rate = np.where(rate_a_mva > 0, rate_a_mva, np.nan)
        loading = np.sqrt(pft**2 + qft**2) / rate
        loading[:, ~in_service] = np.nan

        md = batch.mask_dict
        bt = np.full(res.shape[1], "PQ", dtype=object)
        bt[md["PV"].bool().cpu().numpy()] = "PV"
        bt[md["REF"].bool().cpu().numpy()] = "REF"
        sp, sq = schemas.structural_zero_mask(bt)
        g = batch.batch_dict["bus"].cpu().numpy()
        return TraceBatch(
            res_p=(res[..., 0] * base_mva).astype(np.float32),
            res_q=(res[..., 1] * base_mva).astype(np.float32),
            vm=st[..., 0].astype(np.float32),
            va=np.rad2deg(st[..., 1]).astype(np.float32),
            pft=pft.astype(np.float32), qft=qft.astype(np.float32), loading=loading.astype(np.float32),
            edge_index=ei, edge_in_service=in_service, rate_a_mva=np.asarray(rate_a_mva, np.float32),
            bus_graph=g, bus_local=_local_index(g),
            scenario=batch["scenario_id"].reshape(-1).cpu().numpy(),
            bus_type=bt.astype(str), struct_p=sp, struct_q=sq,
        )


def _stack(pair, dim: int) -> Any:
    import torch

    return torch.stack(tuple(pair), dim)


_SIG_CACHE: dict[type, bool] = {}


def forward_genco(task: Any, batch: Any) -> dict:
    """Call the GENCO forward on either graphkit branch.

    ``main``: ``model(batch)``. ``genco-paper-repro``: ``model(x_dict, edge_index_dict, edge_attr_dict, mask_dict)``.
    """
    model = task.model
    t = type(model)
    if t not in _SIG_CACHE:
        params = list(inspect.signature(model.forward).parameters)
        _SIG_CACHE[t] = params[:1] == ["batch"]
    if _SIG_CACHE[t]:
        return model(batch)
    return model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)

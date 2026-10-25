"""Error-predictor features from the intermediate solutions of the correction steps.

numpy only, fully vectorised, no loops over buses/scenarios.
Feature names are fixed in ``schemas.BUS_FEATURES`` / ``schemas.BRANCH_FEATURES``.

Array conventions: a leading batch axis S is optional everywhere ([T, B] or [S, T, B]).

On real data ``bus_type`` changes between scenarios (a generator outage turns its PV bus into PQ), while a TRACES
shard stores only one ``bus_type``. The structural masks therefore come from the BUS rows (``struct_p``/``struct_q``
per scenario × bus); :func:`compute_bus_features` accepts them as [B] or [S, B].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas

EPS = 1e-6


def residual_magnitude(res_p: np.ndarray, res_q: np.ndarray, struct_p: np.ndarray,
                       struct_q: np.ndarray) -> np.ndarray:
    """r_t = sqrt(P_t^2 * (1-struct_p) + Q_t^2 * (1-struct_q)) in MVA.

    Buses where BOTH components are structural (REF in PF) -> NaN for every t.
    struct_* broadcast over the bus axis (last axis).
    """
    sp, sq = np.asarray(struct_p, bool), np.asarray(struct_q, bool)
    p = np.where(sp, 0.0, np.asarray(res_p, np.float64))
    q = np.where(sq, 0.0, np.asarray(res_q, np.float64))
    r = np.sqrt(p**2 + q**2)
    return np.where(sp & sq, np.nan, r)


def _segment_max(vals: np.ndarray, seg: np.ndarray, n_seg: int, valid: np.ndarray | None = None) -> np.ndarray:
    """NaN-aware max of ``vals[..., e]`` over the entries e with the same ``seg[e]`` -> [..., n_seg].

    ``valid`` ([E] or broadcastable to vals) excludes entries. Segments without a finite valid entry -> NaN.
    One stable sort + ``np.maximum.reduceat`` (much faster than ``np.maximum.at`` on large batches).
    """
    v = np.where(np.isnan(vals), -np.inf, vals)
    if valid is not None:
        v = np.where(valid, v, -np.inf)
    order = np.argsort(seg, kind="stable")
    s_sorted = seg[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    out = np.full(v.shape[:-1] + (n_seg,), -np.inf)
    if order.size:
        out[..., s_sorted[starts]] = np.maximum.reduceat(v[..., order], starts, axis=-1)
    out[np.isinf(out)] = np.nan
    return out


def _struct(mask: np.ndarray, lead_ndim: int) -> np.ndarray:
    """[B] or [S, B] structural mask -> broadcastable against [..., T, B]."""
    m = np.asarray(mask, bool)
    return m[..., None, :] if m.ndim == lead_ndim + 1 and m.ndim > 1 else m


def compute_bus_features(res_p: np.ndarray, res_q: np.ndarray, vm: np.ndarray, va_deg: np.ndarray,
                         struct_p: np.ndarray, struct_q: np.ndarray,
                         edge_index: np.ndarray | None = None,
                         edge_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All bus features of the spec table, each of shape [B] (or [S, B]).

    Definitions (r_t from :func:`residual_magnitude`, t = 1..T):
        f_final       = r_T
        f_logslope    = OLS slope of log(r_t + EPS) on t
        f_tail_ratio  = (r_T + EPS) / (r_{T-1} + EPS)
        f_contraction = (r_T + EPS) / (r_1 + EPS)
        f_nonmono     = #{t : r_{t+1} > r_t}  (float)
        f_auc_log     = mean_t log(r_t + EPS)
        f_overshoot   = r_T - min_t r_t
        f_drift_vm    = |vm_T - vm_{T-1}|
        f_drift_va    = |va_T - va_{T-1}|
    ``struct_p``/``struct_q``: [B], or [S, B] when the bus types differ between scenarios (real N-k data).
    If ``edge_index`` [2, E] (directed, both ways, local bus indices) is given, also return
    ``nbmax_<f>`` = max of <f> over 1-hop neighbours, ignoring NaN (all-NaN neighbourhood -> NaN).
    ``edge_mask`` [E] or [S, E] bool: only True edges count as neighbours. Pass in-service flags: results/ keeps
    out-of-service branches as edges (re-inserted by runner.expand_edges, NaN loading) and an outaged line connects
    nothing.
    Residual-based features are NaN where r is NaN (REF); drift features stay defined.
    """
    res_p = np.asarray(res_p, np.float64)
    lead = res_p.ndim - 2  # 0 or 1
    T, B = res_p.shape[-2:]
    r = residual_magnitude(res_p, res_q, _struct(struct_p, lead), _struct(struct_q, lead))  # [..., T, B]
    nan_bus = np.isnan(r).all(axis=-2)
    lr = np.log(r + EPS)

    t = np.arange(1, T + 1, dtype=np.float64)
    tc = (t - t.mean())[:, None]
    with np.errstate(invalid="ignore"):
        out = {
            "f_final": r[..., -1, :],
            "f_logslope": (tc * lr).sum(axis=-2) / (tc**2).sum(),
            "f_tail_ratio": (r[..., -1, :] + EPS) / (r[..., -2, :] + EPS),
            "f_contraction": (r[..., -1, :] + EPS) / (r[..., 0, :] + EPS),
            "f_nonmono": np.where(nan_bus, np.nan, (np.diff(r, axis=-2) > 0).sum(axis=-2).astype(np.float64)),
            "f_auc_log": lr.mean(axis=-2),
            "f_overshoot": r[..., -1, :] - r.min(axis=-2),
        }
    vm, va = np.asarray(vm, np.float64), np.asarray(va_deg, np.float64)
    out["f_drift_vm"] = np.abs(vm[..., -1, :] - vm[..., -2, :])
    out["f_drift_va"] = np.abs(va[..., -1, :] - va[..., -2, :])

    if edge_index is not None:
        src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
        stack = np.stack([out[f] for f in schemas.BUS_FEATURES])  # [F, ..., B]
        valid = None if edge_mask is None else np.asarray(edge_mask, bool)
        nb = _segment_max(stack[..., src], dst, B, valid)  # [F, ..., B]
        for i, f in enumerate(schemas.BUS_FEATURES):
            out[f"nbmax_{f}"] = nb[i]
    return out


def compute_branch_features(loading: np.ndarray, edge_branch: np.ndarray, n_branch: int,
                            branch_from: np.ndarray, branch_to: np.ndarray,
                            bus_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Branch features, each of shape [n_branch] (or [S, n_branch]).

    loading: [T, Ne] (or [S, T, Ne]) per directed edge; physical loading_t = max over the edges with
    the same ``edge_branch`` id (NaN-aware).
        g_drift_last      = |L_T - L_{T-1}|
        g_drift_tail_std  = std(L_{T-3}, ..., L_T)   (population std, 4 steps)
        g_sign_flips      = number of sign changes in diff(L_t)   (zero differences are not a change)
        g_endpoint_max_<f> = max(bus_features[f][from], bus_features[f][to]) for f in schemas.BUS_FEATURES
                             (NaN-aware: NaN only if both endpoints are NaN)
    Out-of-service branches (all-NaN loading) get NaN own features.
    """
    L = _segment_max(np.asarray(loading, np.float64), np.asarray(edge_branch), n_branch)  # [..., T, nb]
    d = np.diff(L, axis=-2)
    sgn = np.sign(d)
    flips = ((sgn[..., 1:, :] * sgn[..., :-1, :]) < 0).sum(axis=-2).astype(np.float64)
    dead = np.isnan(L).all(axis=-2)
    out = {
        "g_drift_last": np.abs(L[..., -1, :] - L[..., -2, :]),
        "g_drift_tail_std": L[..., -4:, :].std(axis=-2),
        "g_sign_flips": np.where(dead, np.nan, flips),
    }
    bf, bt = np.asarray(branch_from), np.asarray(branch_to)
    for f in schemas.BUS_FEATURES:
        x = np.asarray(bus_features[f], np.float64)
        out[f"g_endpoint_max_{f}"] = np.fmax(x[..., bf], x[..., bt])
    return out


# ----------------------------------------------------------------------------- table adapters (one shard)
def _shard_rows(df: pd.DataFrame, scen: np.ndarray, n_per: int, key: str) -> pd.DataFrame:
    """Rows of ``df`` for the scenarios in ``scen``, sorted by (scenario, key); checks completeness."""
    sub = df[df.scenario.isin(scen)].sort_values(["scenario", key], kind="stable").reset_index(drop=True)
    if len(sub) != len(scen) * n_per:
        raise ValueError(f"expected {len(scen)} x {n_per} rows, got {len(sub)}")
    return sub


def in_service_edges(loading: np.ndarray) -> np.ndarray:
    """[S, T, E] directed-edge loading -> [S, E] in-service flags (NaN at every step = out of service)."""
    return np.isfinite(loading).any(axis=1)


def _shard_order(traces: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(order, sorted scenario ids, S, B): shards are processed in sorted scenario order to match the tables."""
    order = np.argsort(traces["scenario"], kind="stable")
    scen = np.asarray(traces["scenario"])[order]
    return order, scen, len(scen), traces["res_p"].shape[2]


def add_bus_features(bus_df: pd.DataFrame, traces: dict[str, np.ndarray]) -> pd.DataFrame:
    """Fill the f_* / nbmax_* columns for the scenarios in one TRACES shard (the caller loops over shards).

    Returns the rows of ``bus_df`` whose scenario is in ``traces['scenario']``, sorted by (scenario, bus), with the
    feature columns filled (float32). Neighbour aggregation uses the in-service edges only (loading not NaN).
    Structural masks are taken per row from ``bus_df`` (bus types vary between scenarios).
    """
    order, scen, S, B = _shard_order(traces)
    sub = _shard_rows(bus_df, scen, B, "bus")
    sp = sub.struct_p.to_numpy(bool).reshape(S, B)
    sq = sub.struct_q.to_numpy(bool).reshape(S, B)
    ei = np.stack([traces["edge_from"], traces["edge_to"]])
    f = compute_bus_features(traces["res_p"][order], traces["res_q"][order], traces["vm"][order],
                             traces["va"][order], sp, sq, ei, in_service_edges(traces["loading"][order]))
    for name in schemas.BUS_FEATURES + schemas.BUS_NB_FEATURES:
        sub[name] = f[name].reshape(-1).astype(np.float32)
    return sub


def add_branch_features(branch_df: pd.DataFrame, bus_df: pd.DataFrame, traces: dict[str, np.ndarray],
                        static_branch: pd.DataFrame) -> pd.DataFrame:
    """Fill the g_* columns for the scenarios in one TRACES shard; same row contract as :func:`add_bus_features`.
    ``traces['edge_branch']`` maps directed edges to physical branches; ``bus_df`` must already hold the bus
    features of these scenarios (output of :func:`add_bus_features`)."""
    order, scen, S, B = _shard_order(traces)
    sb = static_branch.sort_values("branch")
    nb = len(sb)
    sub = _shard_rows(branch_df, scen, nb, "branch")
    bus = _shard_rows(bus_df, scen, B, "bus")
    bus_f = {f: bus[f].to_numpy(np.float64).reshape(S, B) for f in schemas.BUS_FEATURES}
    g = compute_branch_features(traces["loading"][order], traces["edge_branch"], nb, sb.from_bus.to_numpy(),
                                sb.to_bus.to_numpy(), bus_f)
    for name in schemas.BRANCH_FEATURES:
        sub[name] = g[name].reshape(-1).astype(np.float32)
    return sub

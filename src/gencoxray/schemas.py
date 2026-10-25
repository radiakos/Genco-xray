"""Data contracts between workstreams (IMPLEMENTED — this is the source of truth).

Every table written to ``results/`` must pass :func:`validate`. The human-readable mirror is
``docs/interfaces.md``; change both together.

Units in all tables are physical: MW, MVAr, MVA, Vm in p.u., Va in degrees, loading = S / rate_a.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Feature names (shared by features.py, trust.py, synthetic.py)
# --------------------------------------------------------------------------------------
BUS_FEATURES: tuple[str, ...] = (
    "f_final",
    "f_logslope",
    "f_tail_ratio",
    "f_contraction",
    "f_nonmono",
    "f_auc_log",
    "f_overshoot",
    "f_drift_vm",
    "f_drift_va",
)
BUS_NB_FEATURES: tuple[str, ...] = tuple(f"nbmax_{f}" for f in BUS_FEATURES)
BRANCH_OWN_FEATURES: tuple[str, ...] = ("g_drift_last", "g_drift_tail_std", "g_sign_flips")
BRANCH_ENDPOINT_FEATURES: tuple[str, ...] = tuple(f"g_endpoint_max_{f}" for f in BUS_FEATURES)
BRANCH_FEATURES: tuple[str, ...] = BRANCH_OWN_FEATURES + BRANCH_ENDPOINT_FEATURES

BUS_TYPES: tuple[str, ...] = ("PQ", "PV", "REF")


def structural_zero_mask(bus_type: np.ndarray, task: str = "PowerFlow") -> tuple[np.ndarray, np.ndarray]:
    """Return (struct_p, struct_q) boolean masks of residual components that are zero by construction.

    PowerFlow (paper Eq. 11 + REF recovery): Q at PV and REF, P at REF.
    OptimalPowerFlow: Q at PV and REF only.

    Parameters
    ----------
    bus_type : array of str in {"PQ","PV","REF"}, shape [B]
    task : "PowerFlow" | "OptimalPowerFlow"
    """
    bt = np.asarray(bus_type).astype(str)
    is_pv, is_ref = bt == "PV", bt == "REF"
    struct_q = is_pv | is_ref
    if task == "PowerFlow":
        struct_p = is_ref.copy()
    elif task == "OptimalPowerFlow":
        struct_p = np.zeros_like(is_ref)
    else:
        raise ValueError(f"unsupported task {task!r}")
    return struct_p, struct_q



# --------------------------------------------------------------------------------------
# Schema machinery
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Col:
    name: str
    kind: str  # "int" | "float" | "bool" | "cat" | "list"
    nullable: bool = True
    desc: str = ""


class SchemaError(ValueError):
    pass


_KIND_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "cat": lambda s: isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s) or s.dtype == object,
    "list": lambda s: s.dtype == object,
}


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind not in _KIND_CHECKS:
        raise ValueError(kind)
    return bool(_KIND_CHECKS[kind](s))


def validate(df: pd.DataFrame, schema: tuple[Col, ...], *, strict: bool = False) -> pd.DataFrame:
    """Check columns, dtypes and nullability. Returns ``df`` unchanged so it can be chained.

    strict=True also rejects extra columns.
    """
    errors: list[str] = []
    names = [c.name for c in schema]
    for c in schema:
        if c.name not in df.columns:
            errors.append(f"missing column {c.name!r}")
            continue
        s = df[c.name]
        if not _kind_ok(s, c.kind):
            errors.append(f"column {c.name!r}: dtype {s.dtype} is not {c.kind}")
        if not c.nullable and s.isna().any():
            errors.append(f"column {c.name!r}: contains nulls but is non-nullable")
    if strict:
        extra = sorted(set(df.columns) - set(names))
        if extra:
            errors.append(f"unexpected columns {extra}")
    if errors:
        raise SchemaError("; ".join(errors))
    return df


def add_nan_columns(df: pd.DataFrame, names: tuple[str, ...]) -> None:
    """Append float32 all-NaN columns in place (feature / score columns filled by a later milestone)."""
    for n in names:
        df[n] = np.float32(np.nan)


def _float_cols(names: tuple[str, ...], desc: str = "") -> tuple[Col, ...]:
    return tuple(Col(n, "float", True, desc) for n in names)


STATIC_BUS: tuple[Col, ...] = (
    Col("bus", "int", False, "local bus index 0..B-1"),
    Col("bus_type", "cat", False, "PQ|PV|REF"),
    Col("degree", "int", False),
    Col("hop_to_pv_ref", "int", False, "BFS hops to nearest bus with known Vm"),
    Col("hop_to_ref", "int", False),
    Col("x", "float", True, "layout x"),
    Col("y", "float", True, "layout y"),
)

STATIC_BRANCH: tuple[Col, ...] = (
    Col("branch", "int", False, "physical branch id"),
    Col("from_bus", "int", False),
    Col("to_bus", "int", False),
    Col("rate_a_mva", "float", True, "thermal limit; <=0 means unlimited"),
)

BUS: tuple[Col, ...] = (
    Col("scenario", "int", False),
    Col("load_scenario", "int", False, "datakit load_scenario_idx: the CV grouping key"),
    Col("bus", "int", False),
    Col("k", "int", False, "contingency order"),
    Col("bus_type", "cat", False),
    Col("vm_pred", "float", False, "p.u."),
    Col("vm_true", "float", False, "p.u."),
    Col("va_pred", "float", False, "deg"),
    Col("va_true", "float", False, "deg"),
    Col("res_p_final_mw", "float", False),
    Col("res_q_final_mvar", "float", False),
    Col("res_final_mva", "float", False, "non-structural components only"),
    Col("struct_p", "bool", False),
    Col("struct_q", "bool", False),
    Col("err_vm", "float", False, "|vm_pred - vm_true|"),
    Col("err_va_deg", "float", False),
    Col("hop_to_outage", "float", True, "BFS hops to nearest endpoint of an outaged element (filled by build_features)"),
) + _float_cols(BUS_FEATURES) + _float_cols(BUS_NB_FEATURES)

BRANCH: tuple[Col, ...] = (
    Col("scenario", "int", False),
    Col("load_scenario", "int", False),
    Col("branch", "int", False),
    Col("k", "int", False),
    Col("in_service", "bool", False),
    Col("loading_pred", "float", True),
    Col("loading_true", "float", True),
    Col("loading_dc", "float", True),
    Col("err_loading", "float", True),
    Col("branch_risk", "float", True, "error score of the fitted error predictor (higher = riskier); NaN until filled"),
) + _float_cols(BRANCH_FEATURES)

SCENARIO: tuple[Col, ...] = (
    Col("scenario", "int", False),
    Col("load_scenario", "int", False),
    Col("k", "int", False),
    Col("outaged_branches", "list", False),
    Col("outaged_gens", "list", False),
    Col("max_loading_pred", "float", True),
    Col("max_loading_true", "float", True),
    Col("max_loading_dc", "float", True),
    Col("n_overload_pred", "int", False),
    Col("n_overload_true", "int", False),
    Col("vm_viol_pred", "int", False),
    Col("vm_viol_true", "int", False),
    Col("res_mean_final_mva", "float", False),
    Col("trust_score", "float", True, "scenario accuracy score (higher = more accurate); filled by error_check"),
    Col("genco_ms", "float", True, "amortised per-instance runtime"),
)

TRACES_KEYS: tuple[str, ...] = (
    "scenario", "res_p", "res_q", "vm", "va", "loading", "bus_type", "edge_from", "edge_to", "edge_branch",
)


def validate_traces(arrs: dict[str, np.ndarray]) -> None:
    """Shape checks for one TRACES shard: res_p/res_q/vm/va [S,T,B], loading [S,T,E] (directed edges),
    edge_from/edge_to/edge_branch [E] (local bus indices; edge_branch = physical branch id)."""
    missing = [k for k in TRACES_KEYS if k not in arrs]
    if missing:
        raise SchemaError(f"traces missing {missing}")
    S, T, B = arrs["res_p"].shape
    for k in ("res_q", "vm", "va"):
        if arrs[k].shape != (S, T, B):
            raise SchemaError(f"traces[{k}] shape {arrs[k].shape} != {(S, T, B)}")
    if arrs["loading"].shape[:2] != (S, T):
        raise SchemaError("traces[loading] must be [S,T,E]")
    E = arrs["loading"].shape[2]
    if any(arrs[k].shape != (E,) for k in ("edge_from", "edge_to", "edge_branch")):
        raise SchemaError("edge_from/edge_to/edge_branch must be [E]")
    if arrs["scenario"].shape != (S,) or arrs["bus_type"].shape != (B,):
        raise SchemaError("scenario must be [S], bus_type [B]")

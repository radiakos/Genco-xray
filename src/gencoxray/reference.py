"""Ground truth and outage bookkeeping from datakit raw tables.

datakit raw layout (checked 19 Oct 2026, gridfm_datakit/utils/column_names.py), one row per scenario × element:

* ``bus_data.parquet``    scenario, load_scenario_idx, bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF, ... [, Va_dc, Pg_dc]
* ``branch_data.parquet`` scenario, load_scenario_idx, idx, from_bus, to_bus, pf, qf, pt, qt, ..., rate_a, br_status
                          [, pf_dc, pt_dc]
* ``gen_data.parquet``    scenario, load_scenario_idx, idx, bus, p_mw, ..., in_service, is_slack_gen [, p_mw_dc]

graphkit (lfe-tutorial @ 9d42d82) DROPS out-of-service branches from each graph (``RemoveInactiveBranches``); the
remaining edges are a forward block then a reverse block in branch-row order. ``runner.expand_edges`` re-inserts the
dropped ones, so in our tables and traces directed edge j and j + E/2 are physical branch j (raw row j).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- pure functions (tested)
def branch_loading(pft: np.ndarray, qft: np.ndarray, edge_branch: np.ndarray, rate_a_mva: np.ndarray,
                   n_branch: int) -> np.ndarray:
    """Physical-branch loading = max over its directed edges of |S| / rate_a.

    pft/qft: [..., Ne] MW/MVAr; edge_branch: [Ne]; rate_a_mva: [n_branch]. Returns [..., n_branch];
    NaN where rate <= 0 or no finite direction.
    """
    s = np.sqrt(pft**2 + qft**2)
    lead = s.shape[:-1]
    flat = s.reshape(-1, s.shape[-1])
    out = np.full((flat.shape[0], n_branch), -np.inf)
    vals = np.where(np.isnan(flat), -np.inf, flat)
    np.maximum.at(out.T, edge_branch, vals.T)
    out[np.isinf(out)] = np.nan
    rate = np.where(np.asarray(rate_a_mva) > 0, rate_a_mva, np.nan)
    return (out / rate).reshape(*lead, n_branch)


def element_position(data: pd.DataFrame) -> np.ndarray:
    """0-based row position of each element within its scenario (= raw row order = our branch / generator id)."""
    return data.groupby("scenario").cumcount().to_numpy()


def load_scenario_map(bus_data: pd.DataFrame) -> pd.Series:
    """scenario -> load_scenario_idx (datakit creates n_topology_variants scenarios per load scenario)."""
    return bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first().astype(np.int64)


def base_offline(data: pd.DataFrame, status_col: str) -> np.ndarray:
    """Row positions that are out of service in EVERY scenario of ``data`` — offline in the base case, not a contingency.

    Texas2k's summer-peak base case has many uncommitted generators (``in_service == 0`` everywhere); counting them
    would give k ≈ 114 for an N-1 split. Pass the result to :func:`outages`. Needs a reasonably large set of
    scenarios (a whole split); :func:`runner.check_split` verifies k == split k on Texas.
    """
    pos = element_position(data)
    on_any = np.zeros(int(pos.max()) + 1, bool)
    np.logical_or.at(on_any, pos, data[status_col].to_numpy() == 1)
    return np.flatnonzero(~on_any).astype(np.int32)


def outages(branch_data: pd.DataFrame, gen_data: pd.DataFrame | None = None, *,
            base_out_branches: np.ndarray | None = None, base_out_gens: np.ndarray | None = None) -> pd.DataFrame:
    """Per scenario: arrays of outaged branch row indices (br_status == 0) and generator row indices (in_service == 0),
    excluding elements already offline in the base case (``base_out_*``, see :func:`base_offline`).

    Branch/gen ids are the 0-based row position within the scenario (= raw row order), not datakit ``idx``.
    """
    scen = np.sort(branch_data.scenario.unique())
    empty = np.zeros(0, np.int32)

    def outaged(data: pd.DataFrame, status_col: str, base_out: np.ndarray | None) -> list[np.ndarray]:
        d = data[["scenario", status_col]].assign(pos=element_position(data).astype(np.int32))
        if base_out is not None:
            d = d[~d.pos.isin(base_out)]
        per = d[d[status_col] == 0].groupby("scenario")["pos"].apply(lambda s: s.to_numpy(np.int32))
        return [per.get(s, empty) for s in scen]

    has_gens = gen_data is not None and "in_service" in gen_data
    return pd.DataFrame({
        "scenario": scen.astype(np.int64),
        "outaged_branches": outaged(branch_data, "br_status", base_out_branches),
        "outaged_gens": outaged(gen_data, "in_service", base_out_gens) if has_gens else [empty] * len(scen),
    })


def dc_loading(branch_data: pd.DataFrame) -> pd.DataFrame:
    """(scenario, branch, loading_dc) with loading_dc = max(|pf_dc|, |pt_dc|) / rate_a (DC has no Q).

    NaN where rate_a <= 0 or br_status == 0. Empty frame (same columns) if the DC columns are absent.
    """
    cols = ["scenario", "branch", "loading_dc"]
    if not {"pf_dc", "pt_dc"} <= set(branch_data.columns):
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(cols, ("int64", "int32", "float32"))})
    b = branch_data
    rate = b.rate_a.where(b.rate_a > 0)
    ld = np.maximum(b.pf_dc.abs(), b.pt_dc.abs()) / rate
    ld = ld.where(b.br_status == 1)
    return _per_branch(b, "loading_dc", ld)


def ac_loading_raw(branch_data: pd.DataFrame) -> pd.DataFrame:
    """(scenario, branch, loading_raw) from datakit's own AC flows: max(|S_f|, |S_t|) / rate_a.

    Used in the sanity checks to cross-check the truth loading the runner computes from the target state.
    """
    b = branch_data
    sf = np.hypot(b.pf, b.qf)
    st = np.hypot(b.pt, b.qt)
    ld = (np.maximum(sf, st) / b.rate_a.where(b.rate_a > 0)).where(b.br_status == 1)
    return _per_branch(b, "loading_raw", ld)


def _per_branch(branch_data: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    """(scenario, branch, <name>) long table, branch = row position within the scenario."""
    return pd.DataFrame({"scenario": branch_data.scenario.astype(np.int64).to_numpy(),
                         "branch": element_position(branch_data).astype(np.int32),
                         name: values.astype(np.float32).to_numpy()})


# ----------------------------------------------------------------------------- loaders (thin, untested here)
RAW_COLS = {
    "bus_data": ["scenario", "load_scenario_idx", "bus"],
    "branch_data": ["scenario", "from_bus", "to_bus", "pf", "qf", "pt", "qt", "rate_a", "br_status", "pf_dc", "pt_dc"],
    "gen_data": ["scenario", "bus", "in_service"],
}


def read_raw(raw_dir: str | Path, table: str, scenarios: np.ndarray | None = None) -> pd.DataFrame:
    """Read the needed columns of one datakit table (file or partitioned directory), optionally only ``scenarios``.

    Missing optional columns (``*_dc``) are skipped. Rows are sorted by scenario, keeping element order.
    """
    import pyarrow.parquet as pq

    path = Path(raw_dir) / f"{table}.parquet"
    available = set(pq.ParquetDataset(path).schema.names)
    cols = [c for c in RAW_COLS[table] if c in available]
    filters = [("scenario", "in", list(map(int, scenarios)))] if scenarios is not None else None
    df = pd.read_parquet(path, columns=cols, filters=filters)
    return df.sort_values("scenario", kind="stable").reset_index(drop=True)

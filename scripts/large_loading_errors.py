"""Large loading-error check: the (scenario, branch) pairs with the largest |L_pred - L_true|, in loading and in MVA.
Writes results/<grid>/large_loading_errors.csv and summary.large_loading_errors.<grid>. Run after ac_resolve_selection.py.

    python scripts/large_loading_errors.py --grid texas2k

Two lists of the --top largest errors on in-service branches:
* ``all``: over every scenario (the left end, q = 0, of the bottom panels of the hybrid figure);
* ``kept_<subset>``: over the scenarios the margin_trust rule keeps with GENCO at its q* (summary.e4), i.e. the errors
  that remain after the selected share is re-solved with AC. Same ordering and tie-break as ac_resolve_selection.py.
Per pair: rating, GENCO and AC apparent flow (MVA, = loading x rate_a), error in MVA, number of outages, hops from
the branch to the nearest outage, transformer flag (tap ratio not 0/1, phase shift, or different nominal kV at the
two ends, from the raw base data). Context: the same quantities for all in-service rows with error > 1 (100 % of rating).
"""

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gencoxray import cli, hybrid, io
from gencoxray.config import grid_splits, k_fit, load_config, resolve

log = logging.getLogger("large_loading_errors")
_spec = importlib.util.spec_from_file_location("ac_resolve_selection", Path(__file__).with_name("ac_resolve_selection.py"))
ac_resolve_selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac_resolve_selection)
COLS = ["scenario", "branch", "k", "in_service", "loading_pred", "loading_true", "err_loading"]


def branch_static(gname: str, split: str) -> pd.DataFrame:
    """static_branch (rate_a_mva, from_bus, to_bus) + transformer flag from the raw base data of one split."""
    st = io.read_table("static_branch", gname).sort_values("branch").reset_index(drop=True)
    raw = next((resolve(load_config().data_dir) / gname / "data" / split).glob("*/raw"))
    br = pd.read_parquet(raw / "branch_data.parquet", columns=["scenario", "idx", "from_bus", "to_bus", "tap", "shift"])
    br = br[br.scenario == br.scenario.min()].sort_values("idx").reset_index(drop=True)
    bus = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario", "bus", "vn_kv"])
    kv = bus[bus.scenario == bus.scenario.min()].set_index("bus").vn_kv
    if len(br) != len(st) or not (br.from_bus.to_numpy() == st.from_bus.to_numpy()).all():
        raise RuntimeError("raw branch order does not match static_branch")
    tap = br.tap.to_numpy()
    st["transformer"] = (((tap != 0) & (tap != 1)) | (br["shift"].to_numpy() != 0)
                         | (kv.reindex(st.from_bus).to_numpy() != kv.reindex(st.to_bus).to_numpy()))
    st["kv_from"], st["kv_to"] = kv.reindex(st.from_bus).to_numpy(), kv.reindex(st.to_bus).to_numpy()
    return st


def kept_scenarios(gname: str, splits: list[str], k_max: int, seed: int) -> dict[str, set]:
    """Per hybrid subset: (split, scenario) pairs the margin_trust rule keeps with GENCO at its q*."""
    scale = ac_resolve_selection.fit_scale(gname, splits, k_max, seed)
    sc = ac_resolve_selection.load(gname, splits, scale)
    risk = ac_resolve_selection.gate_scores(sc)["margin_trust"]
    e4 = io.read_summary()["e4"][gname]["subsets"]
    out = {}
    for name, m in ac_resolve_selection.subsets(sc.k.to_numpy(), k_max).items():
        q = e4[name]["gates"]["margin_trust"]["q_star"]
        o = hybrid._order(risk[m], seed)  # same order and tie-break as hybrid.budget_curve
        n = int(np.ceil(q * m.sum() - 1e-9))
        sub = sc[m].iloc[o[n:]]
        out[name] = set(zip(sub.split, sub.scenario))
        log.info("%s: q* %.3f, %d of %d scenarios kept with GENCO", name, q, len(sub), m.sum())
    return out


def hop_to_branch(gname: str, split: str, rows: pd.DataFrame, st: pd.DataFrame) -> np.ndarray:
    """Hops from each row's branch (min over its two ends) to the nearest outage, from the BUS table."""
    b = io.read_columns("bus", gname, split, ["scenario", "bus", "hop_to_outage"])
    h = b.set_index(["scenario", "bus"]).hop_to_outage
    f = st.from_bus.to_numpy()[rows.branch]
    t = st.to_bus.to_numpy()[rows.branch]
    s = rows.scenario.to_numpy()
    return np.fmin(h.reindex(pd.MultiIndex.from_arrays([s, f])).to_numpy(),
                   h.reindex(pd.MultiIndex.from_arrays([s, t])).to_numpy())


def describe(d: pd.DataFrame) -> dict:
    return {"n": len(d), "median_rate_a_mva": float(d.rate_a_mva.median()),
            "median_err_mva": float(d.err_mva.median()), "max_err_mva": float(d.err_mva.max()),
            "max_err_loading": float(d.err_loading.max()), "share_transformer": float(d.transformer.mean()),
            "share_rate_below_50_mva": float((d.rate_a_mva < 50).mean()),
            "median_k": float(d.k.median()), "median_hop_to_outage": float(np.nanmedian(d.hop_to_outage))}


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    cli.setup_logging()
    cfg = load_config()
    gname = cli.grid_name(a)
    splits, k_max = grid_splits(gname), k_fit(gname)
    st = branch_static(gname, splits[0])
    kept = kept_scenarios(gname, splits, k_max, cfg.seed)
    rows, big = [], []
    for s in splits:
        r = io.read_columns("branch", gname, s, COLS)
        r = r[r.in_service & r.err_loading.notna()].assign(split=s)
        # keep this split's candidates for every list: its own top rows overall and per subset, plus all errors > 1
        cand = [r.nlargest(a.top, "err_loading"), r[r.err_loading > 1.0]]
        for name, ks in kept.items():
            mk = np.fromiter(((s, x) in ks for x in r.scenario.to_numpy()), bool, len(r))
            cand.append(r[mk].nlargest(a.top, "err_loading"))
        c = pd.concat(cand).drop_duplicates(["scenario", "branch"])
        c["hop_to_outage"] = hop_to_branch(gname, s, c, st)
        rows.append(c)
    D = pd.concat(rows, ignore_index=True)
    D = D.merge(st[["branch", "rate_a_mva", "transformer", "kv_from", "kv_to"]], on="branch", how="left")
    D["s_genco_mva"] = D.loading_pred * D.rate_a_mva
    D["s_ac_mva"] = D.loading_true * D.rate_a_mva
    D["err_mva"] = D.err_loading * D.rate_a_mva
    lists = {"all": D.nlargest(a.top, "err_loading")}
    for name, ks in kept.items():
        mk = np.fromiter(((sp, x) in ks for sp, x in zip(D.split, D.scenario)), bool, len(D))
        lists[f"kept_{name}"] = D[mk].nlargest(a.top, "err_loading")
    out = pd.concat([v.assign(list=k) for k, v in lists.items()], ignore_index=True)
    out.to_csv(io.results_dir(gname) / "large_loading_errors.csv", index=False)
    all_rated = io.read_table("static_branch", gname).rate_a_mva
    summ = {k: describe(v) for k, v in lists.items()} | {
        "errors_above_rating": describe(D[D.err_loading > 1.0]),
        "all_branches": {"median_rate_a_mva": float(all_rated.median()),
                         "share_rate_below_50_mva": float((all_rated < 50).mean()),
                         "share_transformer": float(st.transformer.mean())}, "top": a.top}
    io.update_summary(f"large_loading_errors.{gname}", summ)
    show = ["split", "scenario", "branch", "k", "hop_to_outage", "transformer", "kv_from", "kv_to", "rate_a_mva",
            "s_genco_mva", "s_ac_mva", "err_mva", "err_loading"]
    for k, v in lists.items():
        log.info("%s:\n%s", k, v[show].round(3).to_string(index=False))
    log.info("summary: %s", summ)

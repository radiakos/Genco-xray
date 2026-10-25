"""Gradient sensitivities: loading sensitivities by automatic differentiation through GENCO on IEEE 118-bus vs
finite-difference AC. Writes summary.e7.

    python scripts/gradient_sensitivities.py --grid case118 --n-scenarios 16

Protocol (docs/experiments.md, "Gradient sensitivities", preliminary):
* Splits nk/k01 (in-distribution: the checkpoint was trained on N-1), nk/k05 and nk/k10 (shifted); the first
  --n-scenarios test scenarios of each.
* Targets: per scenario, the --n-lines most loaded in-service branches under AC.
* GENCO: d loading / d Pd_k for every bus k by autograd through all 12 correction steps (sensitivity.loading_sensitivity).
* AC reference: central finite differences (±1 % of Pd_k, at least ±0.01 MW) with a numpy Newton-Raphson solver on the
  scenario's own datakit network; the script first checks that the solver reproduces datakit's AC loadings.
* Baseline: DC power-flow sensitivity (PTDF; slack = REF bus; d|S|/dPd ≈ sign(P_ft)·dP_ft/dPd / rate).
* Agreement per (scenario, line), over buses: Spearman, top-5 overlap of |sensitivity|, sign agreement on buses with
  |AC| ≥ 10 % of the line's max, relative L2 error, and the least-squares slope of GENCO on AC. Report medians per split.
"""

import logging
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoints are cached; never re-download here

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gencoxray import cli, io
from gencoxray import sensitivity as se
from gencoxray.checkpoints import download, load_genco
from gencoxray.config import load_config, load_grid
from gencoxray.runner import branch_status_matrix

log = logging.getLogger("gradient_sensitivities")
SPLITS = ("nk/k01", "nk/k05", "nk/k10")


def dc_sensitivity(case: se.ACCase, branch_raw: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    """[L, B] DC sensitivities of loading to Pd (1/MW): PTDF with the REF bus as slack, sign of the AC base flow."""
    br = branch_raw.reset_index(drop=True)
    on = case.in_service
    tap = np.where(br.tap.to_numpy() > 0, br.tap.to_numpy(), 1.0)
    bser = np.where(on, 1.0 / (br.x.to_numpy() * tap), 0.0)
    f, t = case.br_from, case.br_to
    B = case.Pd.size
    Bbus = np.zeros((B, B))
    np.add.at(Bbus, (f, f), bser)
    np.add.at(Bbus, (t, t), bser)
    np.add.at(Bbus, (f, t), -bser)
    np.add.at(Bbus, (t, f), -bser)
    keep = np.flatnonzero(~case.ref)
    X = np.zeros((B, B))
    X[np.ix_(keep, keep)] = np.linalg.inv(Bbus[np.ix_(keep, keep)])
    ptdf = bser[:, None] * (X[f] - X[t])  # [E, B]: dP_ft / d injection (p.u./p.u.)
    sgn = np.sign(br.pf.to_numpy())
    # load increase = injection decrease; MW/MW -> per-MW loading
    return -(sgn[targets, None] * ptdf[targets]) / case.rate_mva[targets, None]


def agreement(g: np.ndarray, a: np.ndarray) -> dict:
    ok = np.isfinite(g) & np.isfinite(a)
    g, a = g[ok], a[ok]
    if a.size < 5 or np.ptp(a) == 0:
        return {}
    big = np.abs(a) >= 0.1 * np.abs(a).max()
    top = lambda v: set(np.argsort(-np.abs(v))[:5])
    return {"spearman": float(spearmanr(g, a).statistic) if np.ptp(g) > 0 else np.nan,
            "top5_overlap": len(top(g) & top(a)) / 5,
            "sign_agreement": float(np.mean(np.sign(g[big]) == np.sign(a[big]))),
            "rel_l2": float(np.linalg.norm(g - a) / np.linalg.norm(a)),
            "slope": float(np.dot(g, a) / np.dot(a, a))}


if __name__ == "__main__":
    ap = cli.parser(__doc__)
    ap.add_argument("--grid", default="case118")
    ap.add_argument("--n-lines", type=int, default=10)
    ap.add_argument("--n-scenarios", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cli.setup_logging()
    exp, grid = load_config(), load_grid(a.grid)
    paths = download(grid, local_dir=exp.data_dir)
    rows, check = [], []
    for split in SPLITS:
        loaded = load_genco(grid, paths, split=split, device=a.device, batch_size=a.n_scenarios, num_workers=0)
        batch = next(iter(loaded.test_loader())).to(a.device)
        scen = batch["scenario_id"].reshape(-1).cpu().numpy().astype(np.int64)
        bus_raw = pd.read_parquet(loaded.raw_dir / "bus_data.parquet", filters=[("scenario", "in", scen.tolist())])
        br_raw = pd.read_parquet(loaded.raw_dir / "branch_data.parquet", filters=[("scenario", "in", scen.tolist())])
        br_raw = br_raw.sort_values("scenario", kind="stable")
        status = branch_status_matrix(br_raw, scen)
        cases, targets, brs = [], [], []
        for s in scen:
            b_s = br_raw[br_raw.scenario == s].reset_index(drop=True)
            c = se.ac_case(bus_raw[bus_raw.scenario == s], b_s)
            V, ok = se.ac_solve(c)
            check.append({"split": split, "scenario": int(s), "converged": bool(ok),
                          "max_abs_loading_diff_vs_datakit": float(np.nanmax(np.abs(se.ac_loading(c, V) - c.loading_ref)))})
            order = np.argsort(-np.nan_to_num(c.loading_ref, nan=-1))
            targets.append(order[: a.n_lines])
            cases.append(c)
            brs.append(b_s)
        targets = np.asarray(targets)
        rate = cases[0].rate_mva
        G_sens = se.loading_sensitivity(loaded, batch, targets, status, rate)  # [G, L, B]
        for gi, (s, c, b_s) in enumerate(zip(scen, cases, brs)):
            A = se.fd_ac_sensitivity(c, int(s), targets[gi])
            D = dc_sensitivity(c, b_s, targets[gi])
            for li, line in enumerate(targets[gi]):
                for src, M in (("genco", G_sens[gi, li]), ("dc", D[li])):
                    r = agreement(M, A[li])
                    if r:
                        rows.append({"split": split, "scenario": int(s), "line": int(line),
                                     "ac_loading": float(c.loading_ref[line]), "source": src, **r})
        log.info("%s: %d scenarios x %d lines done", split, len(scen), a.n_lines)
    df = pd.DataFrame(rows)
    out = io.results_dir(a.grid)
    df.to_csv(out / "e7_sensitivity.csv", index=False)
    chk = pd.DataFrame(check)
    med = df.groupby(["split", "source"])[["spearman", "top5_overlap", "sign_agreement", "rel_l2", "slope"]].median()
    summ = {"protocol": {"splits": list(SPLITS), "n_scenarios_per_split": a.n_scenarios, "n_lines": a.n_lines,
                         "fd_rel_step": 0.01, "reference": "numpy Newton-Raphson on datakit's per-scenario network"},
            "ac_solver_check": {"all_converged": bool(chk.converged.all()),
                                "max_abs_loading_diff_vs_datakit": float(chk.max_abs_loading_diff_vs_datakit.max())},
            "median_by_split": {f"{sp}.{src}": {c: float(v) for c, v in r.items()} for (sp, src), r in med.iterrows()},
            "n_pairs": int(len(df) // 2)}
    io.update_summary(f"e7.{a.grid}", summ)
    log.info("gradient sensitivities: %s", summ)

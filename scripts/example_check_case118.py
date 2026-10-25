"""Example: the reusable last-step check on IEEE 118-bus, on a CPU, in well under a minute.

    python scripts/example_check_case118.py

Loads the released IEEE 118-bus checkpoint, runs one GENCO pass on 16 scenarios with 5 outages, prints the flagged
buses and branches, and checks that the scores equal the ones of the feature tables in results/ (same definitions).
"""

import logging
import os
import time

import numpy as np

from gencoxray import cli, io
from gencoxray.check import check_genco_solution, thresholds_from_summary
from gencoxray.checkpoints import download, load_genco
from gencoxray.config import load_config, load_grid
from gencoxray.reference import read_raw
from gencoxray.runner import branch_status_matrix

os.environ.setdefault("HF_HUB_OFFLINE", "1")
log = logging.getLogger("example")

if __name__ == "__main__":
    cli.parser(__doc__).parse_args()
    cli.setup_logging()
    t0 = time.perf_counter()
    grid, split = load_grid("case118"), "nk/k05"
    loaded = load_genco(grid, download(grid, local_dir=load_config().data_dir), split=split, device="cpu",
                        batch_size=16, num_workers=0)
    batch = next(iter(loaded.test_loader()))
    scen_ids = batch["scenario_id"].reshape(-1).numpy().astype(np.int64)
    br_raw = read_raw(loaded.raw_dir, "branch_data", scen_ids)
    status = branch_status_matrix(br_raw, scen_ids)  # [16, n_branch]
    rate = br_raw[br_raw.scenario == scen_ids[0]].rate_a.to_numpy()  # MVA, raw order, same in every scenario
    v_thr, l_thr = thresholds_from_summary("case118")
    res = check_genco_solution(loaded.task, batch, v_thr, l_thr, base_mva=loaded.base_mva, rate_a_mva=rate,
                               in_service=status)
    scen = res.trace.scenario
    log.info("%d scenarios: %d of %d buses and %d of %d branches flagged (thresholds %.3g p.u., %.3g) in %.1f s",
             scen.size, res.bus.flag.sum(), len(res.bus), res.branch.flag.sum(), len(res.branch), v_thr, l_thr,
             time.perf_counter() - t0)
    log.info("flagged buses (scenario, bus, score):\n%s",
             res.bus[res.bus.flag].assign(scenario=lambda d: scen[d.graph])[["scenario", "bus", "score"]].head(20))
    # consistency with the feature table (nbmax_f_drift_vm)
    b = io.read_columns("bus", "case118", split, ["scenario", "bus", "nbmax_f_drift_vm"])
    b = b.set_index(["scenario", "bus"]).nbmax_f_drift_vm
    ref = b.reindex(list(zip(scen[res.bus.graph], res.bus.bus))).to_numpy()
    diff = np.nanmax(np.abs(ref - res.bus.score.to_numpy()))
    r = io.read_columns("branch", "case118", split, ["scenario", "branch", "g_drift_tail_std"])
    r = r.set_index(["scenario", "branch"]).g_drift_tail_std
    ref_b = r.reindex(list(zip(scen[res.branch.graph], res.branch.branch))).to_numpy()
    diff_b = np.nanmax(np.abs(ref_b - res.branch.score.to_numpy()))
    log.info("max |score - feature table|: bus %.2e p.u., branch %.2e", diff, diff_b)

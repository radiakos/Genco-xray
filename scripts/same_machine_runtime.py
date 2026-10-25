"""Same-machine runtime: GENCO (GPU and CPU) and sparse Newton-Raphson AC (CPU) on the same Texas 2000-bus scenarios.
Writes results/<grid>/runtime_same_machine.json and summary.runtime.<grid>. Run on an otherwise idle machine (laptop on
mains power, maximum-performance mode), after newton_check.py.

    python scripts/same_machine_runtime.py --grid texas2k

Protocol:
* GENCO: inputs already on the device, no recording hooks, no torch.compile; per batch size, --warmup batches, then
  --timed batches each timed on its own (torch.cuda.synchronize before and after). ms per scenario = batch time /
  batch size; median and interquartile range over batches. GPU clock, temperature and throttle reasons logged every 10
  batches. Batch sizes that run out of memory are skipped. Repeated on the CPU with --cpu-timed batches.
* AC: newton.solve (sparse polar NR, tol 1e-8 p.u.) on --per-split scenarios per outage count (the warm-start sample
  of newton_check.py), from flat, DC and GENCO starts; network assembly timed separately. Same machine, one process.
* Speed-up per AC re-solve share q: t_NR,cold / (t_GENCO + q t_NR,start) for start = cold (flat) and warm (GENCO);
  ceiling 1/q as t_GENCO -> 0 for cold re-solves.
"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd

from gencoxray import cli, io, newton
from gencoxray.checkpoints import download, load_genco
from gencoxray.config import grid_splits, load_config, load_grid
from gencoxray.runner import gpu_state

os.environ.setdefault("HF_HUB_OFFLINE", "1")
log = logging.getLogger("runtime")


def _stats(ms: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(ms, [25, 50, 75])
    return {"median": float(med), "q25": float(q1), "q75": float(q3), "mean": float(ms.mean()), "n": int(ms.size)}


def time_genco(loaded, batch_sizes, warmup: int, timed: int) -> dict:
    import torch
    from torch_geometric.data import Batch

    from gencoxray.trace import forward_genco

    task, dev = loaded.task, loaded.device
    ds = loaded.datamodule.test_datasets[0]
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    out = {}
    with torch.inference_mode():
        for bs in batch_sizes:
            n = (warmup + timed) * bs
            graphs = [ds[i % len(ds)] for i in range(n)]
            try:
                batches = [Batch.from_data_list(graphs[i:i + bs]).to(dev) for i in range(0, n, bs)]
                ms, states = [], []
                for j, b in enumerate(batches):
                    sync()
                    t0 = time.perf_counter()
                    forward_genco(task, b)
                    sync()
                    if j >= warmup:
                        ms.append(1e3 * (time.perf_counter() - t0) / bs)
                    if cuda and j % 10 == 0:
                        states.append(gpu_state())
                out[bs] = _stats(np.asarray(ms)) | ({"gpu_states": states,
                                                    "throttled_any": any(s.get("throttled") for s in states)}
                                                   if cuda else {})
                log.info("%s batch %d: %.2f ms/scenario (IQR %.2f-%.2f)", dev, bs, out[bs]["median"],
                         out[bs]["q25"], out[bs]["q75"])
            except torch.OutOfMemoryError:
                log.info("%s batch %d: out of memory, skipped", dev, bs)
                break
            finally:
                batches = None
                if cuda:
                    torch.cuda.empty_cache()
    best = min(out, key=lambda b: out[b]["median"])
    return {"device": torch.cuda.get_device_name(dev) if cuda else "cpu", "by_batch": out, "best_batch_size": best,
            "ms_per_scenario": out[best]["median"]}


def time_newton(gname: str, warm: pd.DataFrame) -> dict:
    """Re-time NR solves (flat, dc, genco) on the warm-start sample of newton_check.py, plus network assembly."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("newton_check", Path(__file__).with_name("newton_check.py"))
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    rows = []
    for split, d in warm.groupby("split"):
        scen = np.unique(d.scenario.to_numpy())
        raw = nc.raw_dir(gname, split)
        flt = [("scenario", "in", scen.tolist())]
        bd = pd.read_parquet(raw / "bus_data.parquet", filters=flt)
        brd = pd.read_parquet(raw / "branch_data.parquet", filters=flt)
        brd = brd.sort_values(["scenario", "idx"] if "idx" in brd else ["scenario"], kind="stable")
        bt = io.read_columns("bus", gname, split, ["scenario", "bus", "vm_pred", "va_pred"])
        bd_g, brd_g, bt_g = (dict(tuple(x.groupby("scenario"))) for x in (bd, brd, bt))
        for s in scen:
            t0 = time.perf_counter()
            net = newton.network(bd_g[s], brd_g[s])
            t_build = time.perf_counter() - t0
            b = bd_g[s].sort_values("bus")
            g = bt_g[s].sort_values("bus")
            starts = {"flat": newton.flat_start(net), "dc": newton.dc_start(net, b.Va_dc.to_numpy()),
                      "genco": (g.vm_pred.to_numpy(np.float64), np.deg2rad(g.va_pred.to_numpy(np.float64)))}
            for name, x0 in starts.items():
                r = newton.solve(net, *x0)
                rows.append({"split": split, "scenario": int(s), "start": name, "ms": 1e3 * r["seconds"],
                             "ms_build": 1e3 * t_build, "iterations": r["iterations"], "converged": r["converged"]})
    T = pd.DataFrame(rows)
    out = {name: _stats(d.ms.to_numpy()) | {"iterations_median": float(d.iterations.median()),
                                             "n_not_converged": int((~d.converged).sum())}
           for name, d in T.groupby("start")}
    out["network_assembly"] = _stats(T[T.start == "flat"].ms_build.to_numpy())
    return out


if __name__ == "__main__":
    ap = cli.parser(__doc__, grid=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=50)
    ap.add_argument("--cpu-batch-sizes", type=int, nargs="+", default=[16])
    ap.add_argument("--cpu-timed", type=int, default=10)
    a = ap.parse_args()
    cli.setup_logging()
    exp, gname = load_config(), cli.grid_name(a)
    grid = load_grid(gname)
    paths = download(grid, local_dir=exp.data_dir)
    split = grid_splits(gname)[len(grid_splits(gname)) // 2]
    res = {"protocol": {k: getattr(a, k) for k in ("batch_sizes", "warmup", "timed", "cpu_batch_sizes", "cpu_timed")}
           | {"genco_split": split}}
    res["gpu_state_start"] = gpu_state()
    loaded = load_genco(grid, paths, split=split, device="cuda", num_workers=0)
    res["genco_gpu"] = time_genco(loaded, a.batch_sizes, a.warmup, a.timed)
    loaded = load_genco(grid, paths, split=split, device="cpu", num_workers=0)
    res["genco_cpu"] = time_genco(loaded, a.cpu_batch_sizes, 3, a.cpu_timed)
    loaded = None
    warm = pd.read_csv(io.results_dir(gname) / "newton_warm.csv")
    res["newton"] = time_newton(gname, warm)
    t_g, t_cold, t_warm = (res["genco_gpu"]["ms_per_scenario"], res["newton"]["flat"]["median"],
                           res["newton"]["genco"]["median"])
    e4 = io.read_summary().get("e4", {}).get(gname, {}).get("subsets", {})
    res["speedup"] = {"genco_alone_vs_nr_cold": t_cold / t_g}
    for sub, v in e4.items():
        q = v["gates"]["margin_trust"]["q_star"]
        res["speedup"][sub] = {"q_star": q, "cold_resolve": t_cold / (t_g + q * t_cold),
                               "warm_resolve": t_cold / (t_g + q * t_warm), "ceiling_1_over_q": 1 / q}
    (io.results_dir(gname) / "runtime_same_machine.json").write_text(json.dumps(res, indent=2, default=float))
    io.update_summary(f"runtime.{gname}", res)
    log.info("GENCO GPU %.2f ms, CPU %.2f ms; NR cold %.2f ms, warm %.2f ms; speed-ups %s", t_g,
             res["genco_cpu"]["ms_per_scenario"], t_cold, t_warm, res["speedup"])

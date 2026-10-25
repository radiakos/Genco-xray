"""Run GENCO with recording hooks over the splits of one grid.

Writes results/<grid>/static_{bus,branch}.parquet and, per split, bus/branch/scenario.parquet (feature columns NaN),
traces/shard_*.npz and meta.json. Always runs the sanity checks on the first split and aborts if they fail.

    python scripts/capture_intermediate_solutions.py --grid case118 --split test --max-scenarios 500   # smoke run
    python scripts/capture_intermediate_solutions.py --grid texas2k                                       # all splits
"""

import json
import logging

from gencoxray import cli, io, reference
from gencoxray.checkpoints import data_root_for_split, download, load_genco
from gencoxray.config import k_of_split, load_config, load_grid
from gencoxray.runner import (
    ac_runtime,
    build_static_tables,
    check_split,
    e0_sanity,
    genco_runtime,
    run_split,
    trace_overhead,
)

if __name__ == "__main__":
    ap = cli.parser(__doc__)
    ap.add_argument("--grid", required=True, choices=["texas2k", "case118", "case118_small", "case118_seed1", "case118_seed42", "case2000"])
    ap.add_argument("--split", action="append", default=None, help="default: all splits in configs/grids/<grid>.yaml")
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--skip-e0", action="store_true", help="debugging only")
    ap.add_argument("--e0-only", action="store_true",
                    help="sanity checks + recording overhead on the first split, no tables")
    ap.add_argument("--num-workers", type=int, default=0, help="data-loader workers (0 is safest on Windows)")
    ap.add_argument("--allow-throttled", action="store_true",
                    help="--runtime: keep a measurement taken while the GPU was throttled (flagged valid=false)")
    ap.add_argument("--runtime", action="store_true",
                    help="GENCO + AC per-instance runtime -> results/<grid>/runtime.json (run on an idle machine)")
    a = ap.parse_args()
    cli.setup_logging()
    exp, grid = load_config(), load_grid(a.grid)
    paths = download(grid, local_dir=exp.data_dir)
    batch_size = getattr(grid, "trace_batch_size", None) or exp.trace.batch_size  # Texas: 16 on an 8 GB GPU
    for i, split in enumerate(a.split or grid.splits):
        loaded = load_genco(grid, paths, split=split, device=exp.device, batch_size=batch_size,
                            num_workers=a.num_workers)
        m2_key = f"m2.{grid.name}.{split.replace('/', '_')}"
        if i == 0 and not (a.skip_e0 or a.runtime):  # --runtime: the sanity checks are recorded by the other runs
            e0 = e0_sanity(loaded, n_scenarios=exp.trace.e0_scenarios)
            e0 |= {"split": split, "base_mva": loaded.base_mva, "n_steps": loaded.n_steps,
                   "graphkit_commit": loaded.graphkit_commit}
            if not e0["passed"]:
                raise SystemExit(f"sanity checks failed (see docs/experiments.md, \"Sanity checks\"): {e0}")
            if a.e0_only:  # reference record: sanity checks + recording overhead (run on an otherwise idle machine)
                e0["overhead"] = trace_overhead(loaded)
                io.update_summary(f"e0.{grid.name}", e0)
                if not e0["overhead"]["passed"]:
                    raise SystemExit(f"tracing overhead >= 15 %: {e0['overhead']}")
            else:  # recording runs still gate on the sanity checks but must not overwrite the reference record
                io.update_summary(f"{m2_key}.e0", e0)
        if a.e0_only:
            break
        if a.runtime:
            bss = tuple(getattr(grid, "runtime_batch_sizes", None) or (1, 4, 16, 64, 256, 512))
            raw_dirs = [p for p in (data_root_for_split(grid, paths, s) / grid.network / "raw" for s in grid.splits)
                        if (p / "runtime_data.parquet").exists()]
            rt = {"genco": genco_runtime(loaded, bss), "ac": ac_runtime(raw_dirs), "splits_used_for_ac": len(raw_dirs)}
            if not rt["genco"]["valid"] and not a.allow_throttled:
                raise SystemExit(f"GPU throttled during the runtime sweep; let it cool and re-run: {rt['genco']}")
            rt["speedup_ac_over_genco"] = rt["ac"]["ac_ms_median"] / rt["genco"]["ms_per_instance"]
            (io.results_dir(grid.name) / "runtime.json").write_text(json.dumps(rt, indent=2))
            io.update_summary(f"m2.{grid.name}.runtime", rt)
            break
        res = run_split(loaded, grid, exp, split, max_scenarios=a.max_scenarios)
        if i == 0:
            shard = next(io.iter_traces(grid.name, split))
            br_raw = reference.read_raw(loaded.raw_dir, "branch_data", shard["scenario"][:1])
            sb, sbr = build_static_tables(shard, br_raw)
            io.write_table(sb, "static_bus", grid.name)
            io.write_table(sbr, "static_branch", grid.name)
        k_split = k_of_split(split) or None  # test_k (Texas) and nk/kXX (118): exactly k; "test" -> 0 -> unchecked
        chk = check_split(grid.name, split, batch_size, expected_k=k_split)
        chk["truth_vs_datakit_loading"] = res.meta["truth_vs_datakit_loading"]
        io.update_summary(f"{m2_key}.check", chk)
        if not chk["passed"]:
            raise SystemExit(f"acceptance check failed for {grid.name}/{split}: {chk}")
        logging.info("%s/%s: %d scenarios", grid.name, split, res.meta["n_scenarios"])

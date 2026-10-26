# Convergence Checks for GENCO: The Last Update Flags Inaccurate Solutions

**GENCO's final power mismatch is a weak indicator of inaccurate solutions; the size of its last update is a strong one.**
Code and result tables for the GENCO Challenge 2026 entry (direction: hybrid solvers). Full text:
[final_submission.md](final_submission.md). Approaches that did not work:
[supplementary_negative_results.md](supplementary_negative_results.md).

A Newton–Raphson power flow is considered converged when two quantities are small: the power mismatch and the state
update of the last iteration. GENCO (the graph neural power-flow solver of
[gridfm-graphkit](https://github.com/gridfm/gridfm-graphkit)) solves AC power flow in twelve iterations; the GENCO
paper checks it by its power mismatch and never reports whether its state has stopped changing. We record all twelve
intermediate iterates of the released checkpoints, without modifying or retraining the model, and compare every
criterion with AC power flow on 44,476 contingency cases (Texas 2000-bus, N-1 to N-20; IEEE 118-bus, N-1 to N-10).

![Final mismatch against last update](docs/figures/mismatch_against_last_step_change_texas2k.png)

Main results (details, intervals and limitations in the submission):
- **The final mismatch is a weak indicator.** AUROC is 0.45–0.85; for IEEE 118-bus voltages it is no better than
  chance in two of three independently trained models.
- **The last update is a strong indicator for individual buses and branches.** AUROC is 0.87–0.97 on the training
  contingency orders, and it flags 2.4 to 20 times fewer buses and branches than the final mismatch for the same
  capture of inaccurate values. It needs no AC solution, except to set its threshold once.
- **One Newton step from GENCO's solution is a near-exact check** (AUROC ≥ 0.9996) and gives correct overload verdicts
  without AC verification: 7 incorrect verdicts in 20,000 Texas cases, against 1,806 for GENCO alone.
- **Screening cases for AC verification:** ranking cases by the margin to the thermal limit divided by the expected
  loading error detects 99.5 % of overloaded N-10 to N-20 cases on Texas after verifying 33.5 % of them (44.5 % for
  the margin alone; 67–89 % for ranking by error magnitude).
- **Critical-branch ranking:** GENCO's tail loading agrees with AC better than DC power flow does (rank correlation
  0.9999 against 0.959 on Texas 2000-bus). The map measures how hard a branch is for GENCO, not how weak the grid is.

## Use the check on your own GENCO runs

```python
from gencoxray.check import check_genco_solution, thresholds_from_summary

v_thr, l_thr = thresholds_from_summary("case118")          # fixed thresholds of the label-free check
res = check_genco_solution(task, batch, v_thr, l_thr, base_mva=loaded.base_mva,
                           rate_a_mva=rate, in_service=status)   # one GENCO pass, recording hooks
res.bus[res.bus.flag]        # buses whose voltage GENCO probably got wrong
res.branch[res.branch.flag]  # branches whose loading GENCO probably got wrong
```

`python scripts/example_check_case118.py` runs this on a CPU in under 20 s and checks the scores against the stored
feature tables. See `src/gencoxray/check.py` for what the scores are and what the thresholds mean.

## Reproduce

```bash
CUDA_TAG=cu126 bash scripts/00_setup_env.sh              # environment (see also environment-lock.txt)
python scripts/download_checkpoints_and_data.py --grid case118 --grid texas2k
python scripts/data/generate_case118_contingency_sets.py # IEEE 118-bus contingency sets (needs Julia + PowerModels)
make sanity-checks GRID=case118                          # recorded states reproduce GENCO's own output
make capture-intermediate-solutions GRID=texas2k         # record the twelve iterations (about 15 GB for both systems)
make build-features error-check ac-resolve-selection stress-maps stress-and-accuracy-map GRID=texas2k
make last-step-check newton-check GRID=texas2k           # last-update check; Newton step; figures
python scripts/write_report_numbers.py                   # every number of the submission -> results/writeup_numbers.md
pytest -q                                                # acceptance tests
```

Remaining stages, including the other IEEE 118-bus checkpoints:

```bash
for grid in case118 texas2k; do make last-step-check newton-check GRID=$grid; done
make large-loading-errors GRID=texas2k
for grid in case118 texas2k; do make scenario-check-scores ac-resolve-selection GRID=$grid; done
python scripts/uncertainty_baselines.py --grid texas2k
make gradient-sensitivities
# seeds 1 and 42 of the IEEE 118-bus base checkpoint (grids case118_seed1, case118_seed42)
for grid in case118_seed1 case118_seed42; do
  make capture-intermediate-solutions build-features error-check stress-maps stress-and-accuracy-map \
       last-step-check GRID=$grid
done
python scripts/uncertainty_baselines.py --grid case118 --seed-grids case118_seed1 case118_seed42
make figures GRID=texas2k
python scripts/example_check_case118.py                  # the reusable check, on a CPU
```

The seed grids read the same raw data as `case118`; give them the same data folder (a link: `ln -s case118
data/case118_seed1`, or a directory junction on Windows) before the capture step.

The recorded intermediate solutions (about 15 GB) are not distributed; the capture step regenerates them from the public
checkpoints and datakit data, and [docs/trace_format.md](docs/trace_format.md) documents their fields and units. The
IEEE 118-bus contingency sets are regenerated from `configs/datagen/`.

**Software.** Python 3.12.15, PyTorch 2.12.1 (CUDA 12.6), PyTorch Geometric 2.8.1, torch-scatter 2.1.2, Lightning 2.6.6,
gridfm-datakit 1.1.0, gridfm-graphkit at commit 9d42d82 (branch `lfe-tutorial`); the IEEE 118-bus sets need Julia with
PowerModels 0.21.6 and Ipopt 1.15.0. Hardware: one RTX 3070 Laptop GPU (8 GB); Texas 2000-bus recording uses batch size
16.

**Seeds and determinism.** One seed (0, in `configs/experiment.yaml`) controls row sampling, fold assignment,
resampling and tie-breaking. CPU computation is deterministic; on the GPU, scatter operations can make repeated
recordings differ at float32 round-off.

**Verification.** The acceptance tests (`pytest -q`) check, among other things, that each method recovers structure
planted in synthetic data, that the fast resampling equals its row-level definition and that the AC-verification screening curve matches
brute force.

## Layout

| Path | Content |
|---|---|
| `src/gencoxray/` | recording hooks (`trace`), features, fitted error model (`trust`), Newton–Raphson toolkit (`newton`), last-update check (`last_step_check`, `check`), screening for AC verification, critical-branch ranking (stress maps), figures (`viz/`) |
| `scripts/` | one script per stage, named after what it does; `make` targets have the same names |
| `configs/` | grid, experiment and data-generation settings |
| `results/` | `summary.json`, `writeup_numbers.md` and the per-system CSV/JSON tables behind every number and figure |
| `docs/` | figures, trace format, data contracts (`interfaces.md`), design notes (`concept.md`, `experiments.md`) |
| `tests/` | acceptance tests (synthetic data; one test needs a GPU and a checkpoint) |

## Requirements and data

Python ≥ 3.10; PyTorch, PyTorch Geometric and gridfm-graphkit at commit `9d42d82` (branch `lfe-tutorial`; see
`results/GRAPHKIT_COMMIT`) are installed by `scripts/00_setup_env.sh`. Checkpoints: `gridfm/genco-pf-contingency` (Texas
2000-bus) and `gridfm/genco-pf-datakit` (IEEE 118-bus, base seeds 0, 1, 42). IEEE 118-bus data:
`gridfm/pf_small_case118_ieee`. Do not fork or patch graphkit: the model is observed through hooks only.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)). The GENCO checkpoints and gridfm-datakit data have their own releases and terms.

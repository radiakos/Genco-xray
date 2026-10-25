# CLAUDE.md — working agreement for Claude Code (GENCO X-Ray)

Read this file, then `PLAN.md`, before writing code. `PLAN.md` holds the milestone order, owners and acceptance
checks. This file holds the rules and the verified facts. Then read `docs/concept.md`, `docs/interfaces.md` and
`docs/experiments.md`.

## What we are building
A GENCO Challenge entry (deadline 25 Oct 2026 AoE). GENCO refines its power-flow solution over **12 correction
steps**, feeding per-bus power-balance residuals back each step. We record that trajectory with forward hooks and
derive two **separate** signals:
1. **Trust** (model-intrinsic): does the trajectory predict where GENCO is wrong vs AC? → trust-gated hybrid solver.
2. **Stress** (physical): from GENCO's *final* states over N-k × load ensembles, which branches/buses are stressed?
   Validated against AC, with DC as the baseline.
The headline is the joint **Stress × Trust map**. Never present trajectory signals as grid weakness; they measure
model difficulty, and every link to physics must be checked against AC and against the confounder baselines.

## How to work here
1. Work on **one milestone at a time** (`/milestone Mx`). Its checklist and acceptance check are in `PLAN.md`; its spec
   is the `§Mx` section of `docs/specs/WS-*.md`.
2. Modules marked **STUB** have their public API and docstrings (inputs, outputs, units, edge cases). Implement to the
   docstring; don't change signatures. If one must change, update the docstring, `docs/interfaces.md` and the tests in
   the same commit and say why.
3. Modules marked **DRAFT** (`checkpoints`, `trace`, `runner` torch side) were written without a GPU or graphkit
   install. Verify them against the installed graphkit in M0/M1 and fix them; update the facts below if they are wrong.
4. Tests in `tests/` are acceptance criteria. Don't weaken a test to pass it; if a test is wrong, fix it and explain.
5. When a milestone's acceptance check passes: tick it in `PLAN.md` and add one line to `docs/log.md`.

## Hard rules
- **Never fork or patch gridfm-graphkit.** Observe the model only via hooks in `trace.py`. Pin the commit in
  `results/GRAPHKIT_COMMIT` (`lfe-tutorial` @ 9d42d82; M0 showed it reproduces the paper's Texas predictions to
  float32 precision, see the M0 facts below).
- **One unit-conversion point.** Inside the model everything is normalised; everything in `results/` is physical
  (MW, MVAr, MVA, Vm p.u., Va **degrees**, loading = S / rate_a). Conversion happens only in `trace.py`/`runner.py`
  using the **fitted** `base_mva` from `normalizer_stats.pt`.
- **Mask structural zeros** (`schemas.structural_zero_mask`): in PF, Q at PV/REF and P at REF are zero by construction.
- **Analysis code is numpy/pandas/sklearn only.** Torch only in `checkpoints.py`, `trace.py`, `runner.py`, `sensitivity.py`.
- **No leakage.** Every CV split, bootstrap and conformal calibration groups by **`load_scenario`**, never `scenario`
  and never rows. The shift experiment fits on k ≤ 2 only.
- **Always report the baselines** next to trust results: `f_final`, `k`, `hop_to_outage`, `hop_to_pv_ref`, random.
  If the trajectory does not beat `f_final`, the write-up says so.
- **Schemas are the contract** (`src/gencoxray/schemas.py` ↔ `docs/interfaces.md`). Validate every table you write.
- **Figures and numbers come from scripts**, from `results/` only. Notebooks are for exploration.
- **No investment or shadow-price claims.** Sensitivities (S1) are a validated stretch result at most.

## Verified facts (checked 19 Oct 2026 against graphkit `main` @ 36b360f and `genco-paper-repro` @ 4e7fbcf)
- GENCO = `gridfm_graphkit.models.gnn_heterogeneous_gns.GNS_heterogeneous` (registry `"GNS_heterogeneous"`).
  Base = `hidden_size 48, attention_head 8, num_layers 12`.
- `forward` differs: `main` → `forward(batch, return_embeddings=False)`; paper branch →
  `forward(x_dict, edge_index_dict, edge_attr_dict, mask_dict)`. `trace.forward_genco` handles both.
- Per step `i` (PF/OPF): `mlp_bus` → `physics_decoder(...)` → `output_temp` [Vm, Va(rad), Pg, Qg];
  `branch_flow_layer(output_temp, ei, ea)` → (Pft, Qft); `node_residuals_layer(P_in, Q_in, output_temp, x)` → (rP, rQ);
  then `h_bus += physics_mlp(stack([rP, rQ]))` for `i < num_layers - 1`. So the residual hook fires 12×, `physics_mlp` 11×.
  `model.layer_residuals[i]` stores only the scalar mean of ‖PBRes‖.
- `test_step`/`predict_step` build **new** ComputeBranchFlow/ComputeNodeResiduals instances: our hooks don't double-fire.
  They also **inverse-transform the batch in place**. Read everything you need from the batch before calling them.
- `predict_step` output differs: `main` returns `{"bus": …, "branch": …}` with known values clamped to truth;
  paper branch returns a flat bus dict, no branch table, no clamping. **Don't depend on it**; `runner.py` computes truth itself.
- Normalisation (`HeteroDataMVANormalizer`): powers / **fitted baseMVA** = 95th percentile of non-zero |Pd|, |Qd|, |Pg|, |Qg|
  in the training split (not 100); angles ×π/180 (rad). `inverse_transform` does **not** convert Va back (stays rad).
  Edge `RATE_A` is divided by baseMVA. Always pass the training run's `normalizer_stats.pt`.
- Indices (`gridfm_graphkit.datasets.globals`): bus `PD_H 0, QD_H 1, QG_H 2, VM_H 3, VA_H 4`; out `VM_OUT 0, VA_OUT 1,
  PG_OUT 2, QG_OUT 3`; gen `PG_H 0, G_ON 6`; edge `P_E 0, Q_E 1, …, RATE_A 9, B_ON 10`.
- Edges: ~~graphkit keeps every branch row as an edge (`B_ON = 0`)~~ — **wrong on lfe-tutorial @ 9d42d82 (M2)**:
  `RemoveInactiveBranches` / `RemoveInactiveGenerators` drop out-of-service branches/gens and the `B_ON` column, so the
  edge count varies per graph (forward block then reverse block of the in-service branches, raw order).
  `runner.expand_edges` re-inserts them from the raw `br_status`, so in `results/` directed edges `j` and `j + E/2`
  are physical branch `j` again (NaN loading when out of service).
- The PF mask (`AddPFHeteroMask`) also zeroes the edge inputs `P_E, Q_E, ANG_MIN, ANG_MAX, RATE_A`. **Never read
  ratings from the batch**; take `rate_a` (MVA, never perturbed) from the raw `branch_data`.
- datakit raw tables (`bus_data`, `branch_data`, `gen_data` .parquet, one row per scenario × element) carry
  `load_scenario_idx`; `branch_data` has `pf, qf, pt, qt, rate_a, br_status` (+ `pf_dc, pt_dc` if `include_dc_res`);
  `gen_data` has `in_service`. There is no thermal-rating perturbation in datakit.
- Do not trace a `torch.compile`-d model: compiled graphs can bypass Python hooks.

## Verified in M0 (19 Oct 2026) — environment actually in use
- conda env **`genco`** (Windows 11). Call `C:/Users/Giannis/anaconda30/envs/genco/python.exe` directly;
  `conda run` breaks on multi-line `-c`. Python 3.12.15, torch 2.12.1+cu126, PyG 2.8.1, torch-scatter 2.1.2,
  Lightning 2.6.6, gridfm-datakit 1.1.0; `gencoxray` is installed editable with `--no-deps`. `[ac]` extras
  (lightsim2grid, pandapower) are **not** installed: they would upgrade numpy to 2.4.6 under torch/numba.
- GPU: RTX 3070 Laptop, **8 GB**. Texas2k needs `batch_size ≤ 16` (the config's 256 OOMs).
- graphkit = editable checkout `../gridfm-graphkit` on **`lfe-tutorial` @ 9d42d82** (not `genco-paper-repro`).
  `task.model(batch)` (main-style `forward`) works. Vm at REF is always masked on this branch, which is what the
  checkpoints' `mask_ref_vm: true` means on `main`; the key itself is ignored. Equivalence evidence:
  case118 `evaluate` reproduces the model card (`results/case118/m0_model_card_agreement.csv`), and Texas2k
  `predict` matches the paper's own `mlflow/eval/test_1/predictions.parquet` (median |ΔVm| 1e-7 p.u., |ΔVa| 5e-7;
  `results/texas2k/m0_paper_prediction_agreement.csv`). M1 confirmed the per-step facts above on this commit
  (same hook targets, `forward(batch, return_embeddings=False)`, `layer_residuals[i]` = mean per-bus ‖(rP, rQ)‖,
  PF decoder returns a fresh `torch.stack` each step, so hooked states are never overwritten in place).
- Windows graphkit CLI runs need `--mp_context spawn` and few `--num_workers`. v0 configs (Texas, HF case118)
  need `--config_conversion auto_inline` (in-memory only, the file is not rewritten).
- Checkpoints: `gridfm/genco-pf-datakit` (case118/case2000; tiny/small/base; 3 seeds; ships the training run's
  `*_scenario_splits.json`) and `gridfm/genco-pf-contingency` (Texas2k, N-2-trained, raw `test_1…test_20`,
  paper predictions per split). Fitted baseMVA: case118 base 137.93, Texas2k 147.61. `normalizer_stats.pt` =
  `{network: {baseMVA_orig, baseMVA, vn_kv_max}}`.
- Raw data are Hive tables, ~200 scenarios per `scenario_partition`, ids contiguous from 0; graphkit's dataset
  index == scenario id (`data_index_<scenario>.pt`). Evaluating on a partial download must use
  `data.split_from_existing_files` built from the training split (`checkpoints.write_split_files`), never a
  fresh random split, which would leak training scenarios.
- case118 training data (`pf_small_case118_ieee`, `topology_perturbation: random, k: 1`): every scenario has
  exactly one outage (77 % branch, 23 % gen), no N-0. Texas2k has 2000 buses; ≈1000 scenarios per `test_k`.
- Contingency order `k` excludes base-case-offline elements (`reference.base_offline`: out in every scenario of the
  split). Texas2k's summer-peak case has many uncommitted generators; counting them gave k = 114 on test_1.
- datakit on Windows (M2): (i) PGLib `.m` download fails TLS in Python here → fetched once with Windows `curl.exe` into
  `site-packages/gridfm_datakit/grids/`; (ii) datakit pastes Windows paths into Julia strings (`\U` escape error) →
  `<case>_corrected.m` was produced once with the same PowerModels parse/export and forward-slash paths (datakit skips
  the step when the file exists). Both are local caches outside the repo; redo them on a new machine.
- `gridfm_datakit setup_pm` only works with `JULIA_PKG_SERVER=""` (the Julia pkg server stalls on the `General`
  registry). Installed: PowerModels 0.21.6, Ipopt 1.15.0. Set the same variable when running `gridfm_datakit generate`
  if Julia needs to resolve packages again.

## Verified in M3 (19 Oct 2026)
- `bus_type` **varies between scenarios** on real data (a generator outage turns its PV bus into PQ: 53 of 118 buses on
  case118 `nk/k03`, 303 of 2,000 on Texas `test_5`). A TRACES shard stores one `bus_type`, so trajectory features take
  `struct_p`/`struct_q` from the BUS rows (`features.compute_bus_features` accepts [S, B] masks). `static_bus.bus_type`
  is the modal type (M3).
- datakit `gen_data.bus` is the local bus index 0..B−1 (graphkit order), so `outaged_gens` map to buses directly.
- No branch has `rate_a ≤ 0` on case118 or Texas2k, so NaN loading in TRACES ⇔ out of service.

## Commands
```bash
pytest -q                          # acceptance tests (synthetic, CPU; test_trace needs torch)
GENCOXRAY_GPU_TESTS=1 pytest -q tests/test_runner_contract.py   # after M0, on the GPU box
make m1 GRID=case118               # E0 + smoke extraction (500 scenarios)
make m2 GRID=texas2k               # full extraction
ruff check src tests && ruff format src tests
```

## Code style
Python ≥ 3.10, type hints, numpy-style docstrings with shapes and units. Vectorise (no Python loops over buses or
scenarios in analysis code). `logging`, not print. Paths and thresholds from `configs/*.yaml`. Seeds from
`configs/experiment.yaml:seed`.

## When a fact above turns out wrong
graphkit changes fast. Inspect the installed source, fix the code, and update this file in the same commit.

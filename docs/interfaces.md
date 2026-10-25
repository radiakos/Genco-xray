# Data contracts

Source of truth: `src/gencoxray/schemas.py`. Validate on write with `schemas.validate(df, SCHEMA)`.
This page is the human-readable mirror. **Change both together.**

## Layout and producers
Producers are the scripts in `scripts/` (same names as the `make` targets).
```
results/
  <grid>/                         grid ∈ {texas2k, case118, case2000, synthetic}
    static_bus.parquet            STATIC_BUS     capture_intermediate_solutions writes; build_features refreshes hops;
                                                 stress_and_accuracy_map adds x/y
    static_branch.parquet         STATIC_BRANCH  capture_intermediate_solutions
    runtime.json                  capture_intermediate_solutions --runtime (GENCO and AC ms per instance; not used in
                                  the write-up)
    e1_anatomy.csv                error_check: mismatch by step (grouping, group, step, q25, median, q75, n)
    e2_metrics.csv, e2_stratified.csv, e2_risk_coverage.csv, e3_conformal.csv     error_check
    e4_curves.csv                 ac_resolve_selection (subset, gate, q, recall, precision, n_missed, max_abs_err,
                                  mean_abs_err)
    e5_{branch_stress,bus_stress,criticality,agreement}.csv                       stress_maps
    joint.parquet                 stress_and_accuracy_map: one row per branch: stress_{genco,ac,dc} (CVaR95), trust
                                  (accuracy score), quadrant (group), disagreement(_rank), from_bus, to_bus,
                                  n_stressed_scenarios, and *_p_overload twins
    <split>/                      texas2k: test_1 … test_20 · case118: test, nk/k01 … nk/k10
      bus.parquet                 BUS            capture (features NaN) → build_features fills features + hop_to_outage
      branch.parquet              BRANCH         capture → build_features fills g_* → error_check fills branch_risk
      scenario.parquet            SCENARIO       capture → error_check fills trust_score
      traces/shard_00000.npz …    TRACES         capture (one shard per batch, scenario order)
      meta.json                   capture (checkpoint, graphkit commit, fitted base_mva, n_steps, timing)
  summary.json                    every number quoted in the write-up; keys: e0 sanity checks, e1 mismatch by step,
                                  e2 error check, e3 error bands, e4 selection for AC re-solve, e5 stress maps,
                                  e6 stress and accuracy map, e7 gradient sensitivities, plus last_step_check,
                                  newton, uncertainty, large_loading_errors
  figures/                        make_figures
```
Tables are written in two stages (recording of truth and predictions, then offline enrichment), so nothing after the
capture step needs a GPU.

## Units (everything in `results/` is physical)
| Quantity | Unit |
|---|---|
| P, residual_P | MW |
| Q, residual_Q | MVAr |
| bus mismatch r = ‖(ΔP, ΔQ)‖ | MVA, over components not zero by construction |
| Vm | p.u. |
| Va | degrees (angle errors wrapped to [−180, 180]) |
| loading | max(\|S_from\|, \|S_to\|) / rate_a; NaN if rate_a ≤ 0 or the branch is out of service |

Conversion uses the **fitted** `base_mva` from the checkpoint's `normalizer_stats.pt` (95th percentile of
non-zero powers in the training split), never 100.

## Keys
- `scenario` — datakit scenario id (= graphkit `scenario_id`).
- `load_scenario` — datakit `load_scenario_idx`. datakit creates `n_topology_variants` scenarios per load scenario.
  **All folds, bootstraps and conformal splits group by `load_scenario`.**
- `bus` — local bus index 0..B−1 (graphkit order). `branch` — branch row position within the scenario
  (= raw branch-row order; directed edges `j` and `j + E/2` are branch `j`, including out-of-service branches that
  graphkit drops and `runner.expand_edges` re-inserts). Not datakit's `idx`.
- `k` — number of outaged branches + generators in the scenario, from the raw tables (not from the split name),
  **excluding elements offline in the base case** (out in every scenario of the split, `reference.base_offline`;
  Texas2k has 113 such generators). `outaged_*` lists exclude them too; `meta.json:base_offline` records them.

## Tables

### STATIC_BUS
`bus:int32`, `bus_type:category{PQ,PV,REF}`, `degree:int16`, `hop_to_pv_ref:int16`, `hop_to_ref:int16`
(−1 until build_features), `x:float32`, `y:float32` (NaN until stress_and_accuracy_map). build_features sets
`bus_type` to the bus's **most frequent** type over the grid's first split (bus types vary per scenario, see BUS),
`degree` to the number of distinct physical neighbours, and the hop distances by BFS over the base-case branches
(unreachable → number of buses).

### STATIC_BRANCH
`branch:int32`, `from_bus:int32`, `to_bus:int32`, `rate_a_mva:float32`.

### BUS (per scenario × bus)
Keys `scenario:int64`, `load_scenario:int64`, `bus:int32`, `k:int16`, `bus_type:category` (**per scenario**: a
generator outage turns its PV bus into PQ; TRACES `bus_type` is only the first scenario's, so features use the
row's `struct_p`/`struct_q`).
Final state `vm_pred, vm_true, va_pred, va_true`. Final mismatch `res_p_final_mw, res_q_final_mvar, res_final_mva`
(components zero by construction set to 0 and flagged by `struct_p, struct_q:bool`).
Labels `err_vm`, `err_va_deg`. Position variable `hop_to_outage:float32` (build_features; hops over the scenario's
in-service graph to the nearest endpoint of an outaged branch or bus of an outaged generator; NaN for k = 0 or
unreachable).
Error-predictor features (build_features, float32, NaN where undefined): `f_final, f_logslope, f_tail_ratio,
f_contraction, f_nonmono, f_auc_log, f_overshoot, f_drift_vm, f_drift_va` and `nbmax_<f>` for each (max over
in-service 1-hop neighbours).

### BRANCH (per scenario × physical branch)
Keys `scenario`, `load_scenario`, `branch`, `k`. `in_service:bool`, `loading_pred, loading_true, loading_dc`
(`loading_dc` NaN if datakit has no DC results), `err_loading`, `branch_risk` (error score from error_check; higher =
riskier). Features (build_features): `g_drift_last, g_drift_tail_std, g_sign_flips` and `g_endpoint_max_<f>` for
each bus feature.

### SCENARIO (per scenario)
`scenario`, `load_scenario`, `k`, `outaged_branches:list<int32>`, `outaged_gens:list<int32>` (row positions),
`max_loading_pred, max_loading_true, max_loading_dc`, `n_overload_pred, n_overload_true:int16`,
`vm_viol_pred, vm_viol_true:int16` (buses outside [0.95, 1.05]), `res_mean_final_mva`,
`trust_score` (scenario accuracy score from error_check; higher = more accurate = −risk), `genco_ms` (forward time per
scenario at the run's batch size).

### TRACES (npz shard)
Per scenario: `scenario[S]`, `res_p[S,T,B]`, `res_q[S,T,B]`, `vm[S,T,B]`, `va[S,T,B]` (deg), `loading[S,T,E]`
(per **directed** edge). Shared: `bus_type[B]`, `edge_from[E]`, `edge_to[E]` (local), `edge_branch[E]`.
T = 12. Out-of-service edges have NaN loading. Read with `io.iter_traces` (one shard in memory at a time).

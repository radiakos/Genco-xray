# Experiments

Protocols as declared before the results, with the later additions marked. Each experiment writes its numbers to
`results/summary.json` under the key shown (the keys keep their original short names) and its figures to
`results/figures/`. Confidence intervals: 95 % percentile cluster bootstrap over **load scenarios** (1,000 resamples,
`trust.bootstrap_ci`).

## Sanity checks of the recording → `summary.e0`
- **Protocol:** IEEE 118-bus test split, then Texas `test_1`, 500 scenarios each (`runner.e0_sanity`).
- **Pass (all four):** (a) the mismatch hook fires `num_layers` (12) times per forward pass; (b) the per-step mean bus
  mismatch from the hook equals `model.layer_residuals[i]` (rtol 1e-4); (c) the final mismatch recomputed with fresh
  graphkit layers from the recorded final state equals the recorded one (< 1e-3 MVA); (d) the **AC solution** has a
  maximum mismatch < 1 MVA (validates targets, units and edge attributes). Each recorded split is also checked: AC
  loading computed by the runner against datakit's own AC flows.
- **If it fails:** stop and diagnose (units, fitted base MVA, normaliser statistics, branch/checkpoint mismatch).

## Mismatch by step (descriptive) → `summary.e1`
Median and interquartile range of the bus mismatch $r_t$ against step $t$, by bus type, by hop-distance bin and by
outage count k (Texas k ∈ {1, 2, 5, 10, 20}). Purpose: show the decay and the position effects before any claim.

## Error check: does the record of the correction steps predict GENCO's errors? → `summary.e2`
- **Implementation:** inside the training range = k ≤ max(`in_distribution_k`) (Texas 2, IEEE 118-bus 1); headline
  threshold τ = first configured value with ≥ 200 positives inside the training range, verdict "undetermined" below
  that; folds = hash of `load_scenario` (the same in every split; Texas `test_k` splits share load scenarios), each row
  scored by the fold model that never saw its load scenario; seeded row samples per split for fitting and evaluation;
  `trust_score` (scenario accuracy score) = the pre-declared aggregate with the best AUROC inside the training range
  (see the `scripts/error_check.py` docstring).
- **Labels:** bus `err_vm > τ_vm` (τ_vm ∈ {0.002, 0.005, 0.01} p.u.); branch `err_loading > τ_L` (τ_L ∈ {0.02, 0.05});
  scenario decision error = (max_loading_pred > 1) ≠ (max_loading_true > 1) **or** max err_loading > 0.05.
- **Scores:** each single feature; neighbourhood-maximum (`nbmax_`) variants; the fitted error predictor (logistic
  regression on standardised log-features; histogram gradient boosting as a check). Five folds grouped by
  **load_scenario**.
- **Baselines:** final mismatch `f_final` (the bar to beat), `k`, `hop_to_outage` (lower = riskier), `hop_to_pv_ref`,
  |GENCO − DC| loading disagreement, random.
- **Metrics:** AUROC, AUPRC, Spearman(score, error), error rate of the kept rows against the share kept.
- **Sets:** (a) inside the training range: Texas k ≤ 2, IEEE 118-bus test split; (b) beyond: fit on Texas k ≤ 2,
  evaluate on each k ≥ 3; (c) position control: AUROC within hop-distance bins and bus-type strata, and features
  residualised on the position variables.
- **Pass (error predictor beats final mismatch):** ΔAUROC(error predictor − f_final) > 0.03 with the interval excluding
  0 in (a) **and** in pooled (b).

## Error bands for branch loading → `summary.e3`
- **Target:** branch loading. Calibrate on k ≤ 2 (load-scenario-disjoint from the test rows).
- **Fixed per branch:** stratified split-conformal per (branch, k-bin) on |y − ŷ| (following Alcántara and
  Chatzivasileiadis).
- **Scaled by expected error:** split-conformal on |y − ŷ| / s(x), with s(x) the exponential of the error predictor's
  log-odds or an isotonic map of it.
- **Metrics:** hit rate per k (target 90 %), mean band width, and the share of overload decisions whose band contains 1.
- **Pass (scaled bands are narrower):** scaled width < fixed width at a hit rate within ±2 points for k ≤ 2. The hit
  rate beyond the training range is reported as measured.

## Selecting cases for AC re-solve → `summary.e4`
- **Protocol:** rank scenarios by a rule and re-solve the top share q with AC (take the AC solution for those). Sweep
  q ∈ [0, 1].
- **Rules compared:** scenario accuracy score, mean final mismatch, k, random, oracle (true decision errors first).
- **Metrics against q:** recall and precision of overloaded scenarios, number of missed overloads, largest remaining
  |loading error|. (A runtime axis was planned and withdrawn: no wall-clock speed-up is claimed.)
- **Pass (accuracy score selects best):** q* (smallest q with recall ≥ 99.5 %) for the accuracy score < q* of every
  non-oracle rule, on Texas pooled k ≤ 5 and on k ≥ 10 separately.
- **Added 19 Oct after the IEEE 118-bus run (IEEE 118-bus = development system, Texas = held-out test; declared before
  Texas was run):** a missed overload is a scenario whose max L_pred sits just under 1, so the distance to the limit
  matters as much as the error size. Rules `margin` = −|max L_pred − 1| (distance to the limit) and `margin_trust` =
  −z_flip (distance in units of expected error), the decision-flip score with z_b = (L_pred,b − 1)/s(branch_risk_b)
  and s the isotonic error map of the error bands (fitted on load-scenario half 0 inside the training range).
  **Pass (distance in units of expected error selects best):** q*(margin_trust) < q* of every other non-oracle rule,
  on the same subsets.
- q* is computed exactly (every n/S); intervals by a paired cluster bootstrap over load scenarios (200 resamples).

## Stress maps against AC and DC → `summary.e5`
- **Per branch:** P(loading > 1), E[(loading − 1)⁺], CVaR₉₅(loading) (tail loading). **Per bus:**
  P(Vm ∉ [0.95, 1.05]). **Per outage element:** criticality = mean number of overloads over the scenarios in which that
  element is out.
- **Sources:** GENCO prediction, AC solution and DC (branches only).
- **Agreement:** Spearman ρ (all elements), precision@k and Jaccard@k for k ∈ {10, 25, 50}, Kendall τ on the top 50.
  Bootstrap over load scenarios.
- **Pass (GENCO stress map beats DC):** GENCO's ρ and precision@25 against AC exceed DC's, with non-overlapping
  intervals.
- **Implementation (19 Oct):** judged on branch tail loading (continuous, defined for every branch); P(overload) and
  E[(L−1)⁺] are zero for most branches, which biases rank statistics and their bootstrap, so they are reported as set
  agreement (top-k among positives, symmetric precision@k). A paired GENCO − DC bootstrap is reported next to the
  non-overlapping-interval rule. See `scripts/stress_maps.py`.

## Stress and accuracy map → `summary.e6`
- Branch stress = the GENCO stress map above. Branch accuracy score = −mean `branch_risk` over scenarios in which the
  branch is stressed (L_pred > 0.9; otherwise all). The scenario accuracy score is not used here: stressed branches
  occur in high-k scenarios, so every stressed branch would look error-prone.
- **Groups:** top decile of stress × bottom tercile of accuracy score (configurable).
- **Test:** per branch, disagreement = |stress_GENCO − stress_AC| in metric units (primary); rank disagreement is
  secondary, because it barely moves inside the top set. One-sided Mann–Whitney U, effect size = rank-biserial.
  **Pass (stressed error-prone branches disagree more with AC):** stressed and error-prone has significantly higher
  disagreement than stressed and accurate.
- **Figure:** Texas map, branches coloured by stress and dashed where error-prone; inset: stress against accuracy
  score.

## Gradient sensitivities (preliminary) → `summary.e7`
On IEEE 118-bus: ∂loading_ℓ/∂Pd_k by automatic differentiation through GENCO against finite-difference AC (perturb
Pd_k by ±1 %, re-solve). Report the Spearman correlation of sensitivity rankings for the 10 most loaded lines.
Ratings were never perturbed in training, so ∂/∂rating is out of scope.

## Figures (made by `scripts/make_figures.py`)
| Name | Content | Source |
|---|---|---|
| `mismatch_against_last_step_change` | Voltage error against final mismatch and last-step change (Texas) | last-step check |
| `mismatch_by_step` | Mismatch against step, by bus type / hop bin / k | mismatch by step |
| `error_check_comparison` | AUROC per check, inside vs beyond the training range | error check |
| `error_rate_and_bands` | Error rate of kept rows; hit rate of error bands against k | error check, error bands |
| `ac_resolve_selection` | Recall and largest remaining error against share re-solved | selection for AC re-solve |
| `stress_agreement` | GENCO and DC against AC tail loading; paired differences | stress maps |
| `stress_and_accuracy_map` | Texas map of stress and accuracy score | stress and accuracy map |
| `stress_disagreement_by_group` | Disagreement by stress × accuracy group | stress and accuracy map |
| `newton_step_and_conditioning` | Voltage error against one Newton step and against the mismatch | Newton check |
| `sensitivity_against_newton_raphson` | Sensitivity agreement | gradient sensitivities |
| `mismatch_animation` | Per-step mismatch map, one single-outage and one twelve-outage scenario (GIF) | recordings |

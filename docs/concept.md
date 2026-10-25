# Concept

Design note written before the results. The submission (`final_submission_genco_challenge.md`) reports what held up; where the two
differ, the submission is the reference.

## 1. Problem
Planners and operators screen thousands of contingency × load scenarios to find where the grid is weak. The GENCO
paper reports this screening as 30× faster than Newton–Raphson on grids of 2,000 buses or more (paper Tab. 6). A
screening tool is useful only if the user knows **when not to believe it**, and the paper's safeguard
("power-balance residuals and violation checks identify critical cases", §6) is left unspecified.

## 2. Key observation
GENCO is an iterative solver in latent space. At each of its T = 12 steps it decodes a full AC state
(Vm, Va, Pg, Qg), computes per-bus power mismatches (paper Eq. 13–14) and feeds them back (Eq. 17). Training rewards
early feasibility (weights λ^{T−t}, Eq. 18). The record of the correction steps is therefore a learned convergence
behaviour:

- buses that converge show geometric decay of the mismatch and vanishing state updates;
- buses where the model fails were expected to **stall** (tail ratio → 1), **oscillate** (mismatch rising between
  steps) or still **change** in Vm/Va at the last step.

This information is free at inference time, needs no labels, and is not visible in the final mismatch alone. (Result:
only the last-step change carried the signal; stalling and oscillation did not, see `study_unsuccessful.md`.)

## 3. Two signals, kept apart

**Error checks (property of the model).** Per-bus features of the correction steps (`features.py`): final mismatch,
log-decay slope, tail ratio, number of rising steps, last-step state change, and the maximum of each over in-service
neighbours. Neighbours matter because a voltage error at bus *k* appears in the mismatches of its neighbours. Branch
features use the per-step branch loading (is the predicted loading still moving?). The scenario accuracy score
aggregates over buses and branches.

**Stress (property of the network).** From final states over ensembles (`stress.py`): branch overload probability,
expected excess loading, tail loading (CVaR₉₅), bus voltage-violation probability, and **outage criticality** (which
outages cause the most overloads). Validated against AC, with DC power flow as the baseline.

**Stress and accuracy map (`joint.py`).** Each branch gets a stress value and an accuracy score, giving four groups.
The group to act on is *stressed and error-prone*: GENCO reports stress there but is likely wrong, so the user should
run AC. The test is whether GENCO's disagreement with AC on stress concentrates in that group.

## 4. Threats to validity and mitigations

| Threat | Why it matters | Mitigation (where) |
|---|---|---|
| Position effects (hop distance, receptive field) | Buses far from generators may converge later regardless of error | `confounders.py`: AUROC within hop-distance bins; features residualised on position variables |
| Components zero by construction | ΔQ at PV/slack and ΔP at slack are 0 | `schemas.structural_zero_mask`, set per scenario |
| Feedback carries power balance only | Thermal limits never enter the loop (paper fn. 7) | Branch features use the loading over the steps, not mismatches |
| Training-density effects | Mismatches are high where training data were sparse | Evaluation beyond the training range, reported per k; k as a position variable |
| Leakage | Same load scenario in fitting and test | Folds and resamples grouped by load scenario; transfer tested by k |
| Units and normalisation | Mismatches inside the model are divided by the fitted base MVA | One conversion point in `runner.py`; sanity checks against graphkit's own evaluation |
| Uniform outage sampling ≠ real outages | Paper App. G.2: Hydro-Québec outages are clustered | Stated in the limitations; results per k, not only pooled |
| Thermal ratings | PGLib `rate_a = 0` means unlimited | Excluded from loading metrics (none occur on the systems used) |
| Single seed or checkpoint | Results may be checkpoint-specific | IEEE 118-bus: three base seeds and the small checkpoint |

## 5. Related work and positioning
- **GENCO** (Puech et al., arXiv:2608.09921) proposes but does not test a mismatch-based safeguard (§6) and reports
  final-step metrics only. We use the intermediate steps.
- **Trustworthiness layer** (Alcántara and Chatzivasileiadis, arXiv:2602.07995): split-conformal bands on a GridFM
  model with per-element quantiles of |y − ŷ|, fixed per element and outage count, with no internal signals. We add an
  input-dependent error score; used as the scale of the conformal score (|y − ŷ| / s(x)), it was expected to give
  narrower bands at the same hit rate (result: not narrower inside the training range; better hit rate beyond it).
- **Iterative neural solvers** (Donon et al.): mismatch-driven updates, without using the intermediate solutions as an
  error signal.
- **The challenge's example entry** (contingency screening with a tuned threshold): we replace the threshold by
  per-element and per-scenario checks, evaluate selection rules for AC re-solve, and add contingency stress maps.

## 6. Out of scope
- Investment recommendations. We show where stress is and where GENCO is accurate; reinforcement decisions need rating
  changes and added components (paper App. G.1 lists them as future datakit features). Gradient sensitivities are
  preliminary evidence only.
- Optimal power flow and state estimation. The recording tools work for them, but they are not evaluated.

# What did not work, and what we learned

Companion to `final_submission_genco_challenge.md`, which reports the results that held up. This file records the approaches that
failed, or held on one system only, with the numbers behind each. Every number is in `results/` (file named in
brackets); where the submission states the result, its section is given. Terms follow the submission.

| # | What we tried | Outcome | In the submission? |
|---|---|---|---|
| 1 | Shape of the mismatch over the steps (stalling, oscillation, slow decay) as the error check | Below chance; the state's last-step change carries the signal | Yes (3.6) |
| 2 | Ranking scenarios by expected accuracy to choose AC re-solves | About as good as random | Texas, 3.4. **IEEE 118-bus no** |
| 3 | First version of the "distance to the limit in units of expected error" rule | Wrong definition, corrected before Texas was run | **No** |
| 4 | Overload probability as the stress statistic | Failed on synthetic data; ties or loses to DC on real grids | Reason only, no numbers |
| 5 | Stress and accuracy map: secondary tests and other accuracy scores | Not significant; IEEE 118-bus untestable | Partly (3.5) |
| 6 | Narrower error bands from scaling by expected error | No narrower inside the training range | No |
| 7 | Gradient sensitivities from GENCO | Worse than DC factors | Yes (3.6) |
| 8 | Claims on the small IEEE 118-bus checkpoint | One claim reverses | No |
| 9 | Pipeline assumptions that were wrong | Found and fixed | **No** |
| 10 | Wall-clock speed-up | Measured, found unrepresentative, withdrawn | Only as "no runtime claim" |
| 11 | "Accept only if both the final mismatch and the last-step change are small" (earlier recommendation) | Flags every scenario; per scenario no better than random | Partly (4) |
| 12 | One Newton step as a cheaper hybrid | More accurate, not cheaper in Newton iterations | Yes (3.4) |

## 1. The mismatch shape did not carry the signal

The project started from the idea that a bus where GENCO is wrong shows it in how the mismatch falls over the twelve
steps: it stalls, oscillates, or decays slowly. We computed these as features per bus (tail ratio of the last steps,
number of steps in which the mismatch rises, overshoot above the minimum, log-decay slope, contraction, area under the
log-mismatch curve) and scored each alone against the voltage-error label (`e2_metrics.csv`, AUROC inside the training
range, own bus):

| Feature | Texas 2000-bus | IEEE 118-bus |
|---|---|---|
| tail ratio | 0.437 | 0.342 |
| steps with rising mismatch | 0.460 | 0.406 |
| overshoot | 0.483 | 0.368 |
| log-decay slope | 0.649 | 0.603 |
| final mismatch | 0.659 | 0.454 |
| **last-step voltage change** | **0.842** | **0.972** |

Stalling and oscillation, the behaviours we expected, point the wrong way (below 0.5): wrong buses do not stall, they are
still moving. What works is the size of the change in the last step, not the shape of the mismatch curve. The fitted
error predictor still contains these features, but its most informative features all describe how much the state
moves at the end (3.2).

## 2. Ranking scenarios by expected accuracy does not choose re-solves

The first design ranked scenarios by their expected accuracy and re-solved the least accurate. The share needed for
99.5 % recall of overloaded scenarios is close to that of a random order. Texas is in 5.6 (0.865 against random 0.928 with
10 or more outages). **IEEE 118-bus is not in the submission** (`e4_curves.csv`, `writeup_numbers.md`):

| IEEE 118-bus, all outage counts (24,476 scenarios, 23,296 overloaded) | Share re-solved |
|---|---|
| scenario accuracy score | 0.908 |
| mean final mismatch | 0.920 |
| number of outages | 0.947 |
| random | 0.862 |
| distance to the limit | 0.164 |
| distance to the limit in units of expected error | 0.127 |

Here 95 % of scenarios are overloaded under AC and GENCO alone has 96.5 % recall, so the task is dominated by the few
cases close to the limit. All error-ranked rules are no better than random, and the distance-to-the-limit rules win by a
large factor. Scaling by expected error helps over all outage counts (difference [−0.049, −0.017]) but not in the
subsets (up to 5 outages [−0.084, 0.004]; 10 or more [−0.022, 0.002]). This is why the final design is built on the
distance to the limit, with the error score as a scale, and why the rules were developed on IEEE 118-bus and held out on Texas.

## 3. A wrong first version of the best rule

The first implementation of "distance to the limit in units of expected error" took the smallest $|z_b|$ over all
branches. That flags scenarios which are safely overloaded through one branch if some other branch sits near 1. It was
replaced by the decision-flip definition in Section 2.5 (the smallest score that would change the overload decision),
after the IEEE 118-bus run and before Texas 2000-bus was evaluated.

## 4. Overload probability failed as the stress statistic

We first compared GENCO, AC and DC by the probability that a branch overloads. Most branches never overload, so this
is exactly zero for most of them in every source. On synthetic data with a planted stressed set this broke the
statistics in three ways:
- *Rank correlation over all branches is inflated by joint zeros, and its resampling is biased*: resamples drop
  overload events and create more joint zeros, so the DC interval lay entirely above its own point estimate.
- *Restricting to branches with positive probability rewards false positives*: DC scored a rank correlation of 0.81
  against GENCO's 0.27, although GENCO found exactly the planted set.
- *Top-k precision rewards false positives*: DC scored 1.00 with 16 false positives.

We therefore switched to tail loading (the mean of the worst 5 % of loadings), which is defined for every branch. This
was decided on the synthetic data, before the real-grid results. On the real grids the original statistic gives a less
favourable picture than tail loading (`e5_agreement.csv`): top-25 precision for overload probability is 0.92 for both
GENCO and DC on Texas (0.96 against 0.92 on IEEE 118-bus); the rank correlation over all branches is 0.909 against 0.842
on Texas, and on IEEE 118-bus DC is higher, 0.965 against 0.955 (point estimates only). The advantage of GENCO over DC
is therefore a property of the tail-loading ranking and of the count of branches found, not of every stress statistic.

## 5. Stress and accuracy map: what is not significant

Section 3.5 reports the primary test (absolute disagreement with AC, $p = 3\times10^{-15}$, Texas). Two secondary
variants, defined before the results, do not reach significance (`writeup_numbers.md`, "Stress and accuracy map"):
disagreement in rank, $p = 0.156$; and the same test with overload probability as the stress measure, $p = 0.057$. Other
accuracy scores on Texas: final mismatch has no signal (rank-biserial 0.04, $p = 0.26$), predicted loading is degenerate
(every stressed branch falls in one group). On IEEE 118-bus only 1 of 19 stressed branches is error-prone, so the test
has no power, and there the DC-based score and predicted loading separate the stressed branches better than the error
predictor (rank-biserial 0.93, $p = 1.4\times10^{-4}$). The map's value is therefore tied to Texas and to one accuracy
score; the absolute differences are at most about $10^{-3}$ of the rating.

## 6–8. Reported in the submission

Briefly, with the sections that hold the detail. *Error bands* scaled by expected error are not narrower than fixed
bands inside the training range (0.0049 against 0.0045 on Texas; 0.0091 against 0.0080 on IEEE 118-bus), and neither
keeps its 90 % hit rate beyond it. *Gradient sensitivities* rank buses worse than DC factors (rank correlation
0.83 / 0.81 / 0.76 against 0.99 / 0.99 / 0.98 for 1 / 5 / 10 outages), so no sensitivity claim is made (3.6). On the
*small checkpoint*, position alone beats the error predictor for bus errors beyond the training range (0.816 against
0.793), and the final mismatch is at or below chance for two of three base seeds.

## 9. Pipeline assumptions that were wrong

Each of these would affect anyone repeating the study:
- *Out-of-service elements:* we assumed gridfm-graphkit keeps every branch as an edge. On the commit used it removes
  out-of-service branches and generators, so the edge count varies per scenario; the thermal ratings are also set to
  zero in the power-flow input. Ratings come from the raw datakit tables and the edges are re-expanded.
- *Per-unit base:* the checkpoints use a fitted base (137.9 MVA and 147.6 MVA for the base seed 0 and Texas
  checkpoints, per seed for the others), not 100 MVA.
- *Outage count:* counting every offline generator gave 114 outages on the first Texas test set, because the
  summer-peak case has many uncommitted generators; the count now excludes elements out in the base case.
- *Silent truncation and caching:* a cap of 1,000 scenarios per split silently truncated the recordings (rerun with
  5,000), and graphkit reuses an older processed-data cache even if its graph count differs from the raw data; the
  runner now checks the count.
- *Reference solver:* datakit's shunt values are already per unit (treating them as MW left a constant 0.44 p.u.
  mismatch); datakit does not enforce generator reactive limits (5 % of Texas and 20 % of IEEE 118-bus in-service
  generator rows lie outside them); GENCO predicts the voltage at the slack bus, which the reference holds at its setpoint.
- *Tooling on Windows:* datakit's grid download failed TLS in Python, its Julia path handling failed on backslashes, and
  the Julia package server stalled; each was worked around locally and is documented in `CLAUDE.md`.

## 10. The speed-up that was withdrawn

Early drafts stated a speed-up of GENCO-based screening over AC. We withdrew it: the timings came from one laptop with
a power-capped, thermally throttled GPU, compared with our own solver, and the AC reference was not timed on the same
machine. The submission states only quantities that do not depend on hardware: the share of scenarios re-solved with AC
and the resulting ceiling on the saving in AC solves.

## 11. The earlier recommendation, tested as a scenario rule

Earlier versions of the submission recommended accepting a GENCO solution only if both its final mismatch and its
last-step change are small, and re-solving the rest. Tested on every scenario (`fixed_rules.csv`,
`scenario_level_checks.csv`, `e4_curves.csv`):
- *As a fixed rule it accepts nothing.* The thresholds were set per bus and branch for 90 % recall of wrong elements,
  so they flag 5–44 % of all buses and 6–29 % of branches (depending on system and outage count); every one of the 44,476
  scenarios has at least one element above them.
- *Ranked by its largest value per scenario it does not find overloads.* The share re-solved for 99.5 % recall of
  overloaded scenarios is 0.85 (Texas, all outage counts) and 0.93 (IEEE 118-bus), against 0.94 and 0.86 for a random
  order and 0.36 and 0.13 for the distance to the limit in units of expected error. The largest last-step change alone
  does no better (0.85 and 0.92).
- *Per scenario the last-step change is weaker than the final mismatch.* For "the scenario has a wrong bus or branch"
  (73 % of Texas scenarios), the largest last-step change has AUROC 0.68–0.72, the largest final mismatch 0.86 and the
  number of outages alone 0.73; on IEEE 118-bus the voltage score falls to 0.45.

What survives: the last-step change as a scale for the distance to the limit (0.369 against 0.335 with 10 or more
outages on Texas, no AC labels), and, on Texas only, as a way to bound the worst flow error (5 % re-solved leaves 0.22
times the rating, against 1.42 for the mean final mismatch); the second was found after the observation that motivated
it, on the same data.

## 12. One Newton step is more accurate, not cheaper

One Newton step from GENCO's solution decides overloads almost exactly with no AC re-solve (7 wrong decisions in 20,000
Texas scenarios, against 1,806 for GENCO alone). We expected it to also be the cheapest hybrid. Counting a Jacobian
factorization as one Newton iteration, it costs 1.0 iteration per scenario, against 0.96 for the distance rule on Texas
and 0.30 on IEEE 118-bus, where 95 % of scenarios are overloaded and that rule re-solves only 12.7 %. One step also does
not fix the worst scenario: the largest remaining loading error is still 2.5 times the rating on Texas.

## Not attempted

Other Texas 2000-bus checkpoints (only one is released and training was out of reach), clustered or credible outage
lists, a second neural solver, an operator demo (dropped), and uploading the recorded intermediate solutions (about
15 GB, regenerated by the capture step instead).

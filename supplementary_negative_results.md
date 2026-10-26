# Negative results

Companion to `final_submission.md`. This file lists the ideas we tested for a stated reason that did not give a useful result, and what each one showed. Every number comes from `results/` (file named in brackets). Terms follow the submission.

| # | Idea | Result | In the submission? |
|---|---|---|---|
| 1 | Shape of the mismatch over the iterations as an error indicator | Below chance; the last update carries the signal | Yes (3.6) |
| 2 | Ranking cases by expected accuracy to select AC verifications | No better than random | Texas only (3.4) |
| 3 | Overload probability as the stress statistic | Equal to or worse than DC | Rationale only |
| 4 | Critical-branch map: secondary tests | Not significant | Partly (3.5) |
| 5 | Scaled error bands, gradient sensitivities, small checkpoint | No gain over simpler alternatives | Gradient sensitivities only (3.6) |
| 6 | Per-element thresholds as a case-level acceptance rule | Flags every case | Yes (3.6) |
| 7 | One Newton step as a cheaper hybrid | More accurate, not cheaper | Yes (3.4) |

## 1. The shape of the mismatch carries no signal

We expected an inaccurate bus to show stalling, oscillation or slow decay of the mismatch over the twelve iterations. Each feature alone was scored against the voltage-error label (`e2_metrics.csv`; AUROC, training contingency orders, own bus):

| Feature | Texas 2000-bus | IEEE 118-bus |
|---|---|---|
| tail ratio | 0.437 | 0.342 |
| iterations with rising mismatch | 0.460 | 0.406 |
| overshoot | 0.483 | 0.368 |
| log-decay slope | 0.649 | 0.603 |
| final mismatch | 0.659 | 0.454 |
| **last voltage update** | **0.842** | **0.972** |

Stalling and oscillation score below chance: inaccurate buses are still moving, not stalled. The signal lies in the size of the last update, not in the shape of the curve.

## 2. Ranking cases by expected accuracy does not select AC verifications

On Texas, the share of cases verified to detect 99.5 % of overloaded cases is close to random (0.865 against 0.928, N-10 to N-20; Section 3.4). The same holds on IEEE 118-bus, which the submission does not report (`e4_curves.csv`). There 95 % of the 24,476 cases are overloaded, so the task is dominated by the few cases close to the limit:

| IEEE 118-bus, all contingency orders | Share verified with AC |
|---|---|
| Case accuracy score | 0.908 |
| mean final mismatch | 0.920 |
| contingency order | 0.947 |
| random | 0.862 |
| distance to the limit | 0.164 |
| distance in units of expected error | 0.127 |

The margin to the limit matters more than the error magnitude, which is why the final design is built on the margin, with the error score as a scale.

## 3. Overload probability as the stress statistic

Most branches never overload, so their overload probability is exactly zero in every source, and rank correlation and top-k precision are dominated by ties and false positives. We therefore used the tail loading. On the real systems the overload probability also gives a less favourable picture for GENCO (`e5_agreement.csv`): top-25 precision is 0.92 for both GENCO and DC on Texas, and the rank correlation is 0.909 against 0.842 on Texas but 0.955 against 0.965 on IEEE 118-bus (point estimates). The advantage of GENCO over DC therefore depends on the tail-loading ranking.

## 4. Critical-branch map: secondary tests

Besides the primary test (Section 3.5, p = 3×10⁻¹⁵), the disagreement in rank (p = 0.156) and the variant with overload probability (p = 0.057) are not significant (`writeup_numbers.md`). The final mismatch as accuracy score shows no effect (rank-biserial correlation 0.04, p = 0.26). On IEEE 118-bus only 1 of 19 critical branches is error-prone, so the test has no power. The map is therefore limited to Texas and to the error index as accuracy score.

## 5. Error bands, gradient sensitivities and the small checkpoint

*Error bands* scaled by the expected error are not narrower than fixed bands within the training orders (0.0049 against 0.0045 on Texas; 0.0091 against 0.0080 on IEEE 118-bus). *Gradient sensitivities* rank buses worse than DC power transfer distribution factors (rank correlation 0.83 / 0.81 / 0.76 against 0.99 / 0.99 / 0.98 for N-1, N-5 and N-10). On the *small checkpoint* of `gridfm/genco-pf-datakit` (the submission uses the base size), the proximity baseline beats the fitted error model for bus errors beyond the training orders (0.816 against 0.793). Error bands and the small checkpoint are not part of the submission.

## 6. Combined thresholds as a case-level acceptance rule

We tested accepting a GENCO solution only if its final mismatch and its last update are both small (`fixed_rules.csv`, `scenario_level_checks.csv`, `e4_curves.csv`). The thresholds of Table 1 were set per element for 90 % capture, so they flag 5–44 % of buses and 6–29 % of branches, and all 44,476 cases contain at least one flagged element. Ranked by the largest last update per case, 0.849 (Texas) and 0.915 (IEEE 118-bus) of cases must be verified for 99.5 % detection of overloads, against 0.939 and 0.862 for random (Table 3). Per case, the largest last update is also a weaker indicator of an inaccurate value than the largest final mismatch (AUROC 0.68–0.72 against 0.86 on Texas). The last update remains useful as the error scale for the margin to the limit (0.369 against 0.445 for the margin alone, N-10 to N-20 on Texas).

## 7. One Newton step is more accurate, but not cheaper

One Newton step gives almost exact overload verdicts without AC verification (7 incorrect in 20,000 Texas cases, against 1,806 for GENCO alone), but it costs 1.0 Newton iteration per case, against 0.96 for the distance in units of expected error on Texas and 0.30 on IEEE 118-bus. It also leaves a largest loading error of 2.5 times the rating on Texas.

# Convergence Checks for GENCO: The Last Update Flags Inaccurate Solutions

*Reliability checks, AC verification screening and critical-branch rankings from GENCO's intermediate iterates.
Challenge direction: hybrid solvers.*

**Team:** Ioannis Karampinis (Technical University of Denmark) · Christos Doudesis (Technical University of Denmark)  
**Contact:** iokar@dtu.dk, chrisdoudesis23@gmail.com  
**Code and result tables:** https://github.com/radiakos/genco-xray  
**Models and data used:** [gridfm/genco-pf-contingency](https://huggingface.co/gridfm/genco-pf-contingency),
[gridfm/genco-pf-datakit](https://huggingface.co/gridfm/genco-pf-datakit), gridfm-datakit (arXiv:2512.14658)

![Final mismatch against last update](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/9owfqZvULFyxjRWyBpROL.png)

**Figure 1.** Texas 2000-bus system. GENCO’s voltage-magnitude error at each PQ bus, plotted against its final power mismatch (left) and against the size of its last voltage update (right). Top: N-1 and N-2 cases, the contingency orders GENCO was trained on. Bottom: N-3 to N-20. Dashed line: error tolerance of 0.005 p.u. Orange: the flagging threshold for each criterion, set once on N-1 and N-2 cases to capture 90 % of inaccurate buses and then held fixed. At this threshold, the final mismatch flags 83 % of all buses and the last update 34 %.

## Summary

A Newton–Raphson power flow is considered converged when two quantities are small: the power mismatch and the state update of the last iteration. GENCO (arXiv:2608.09921) is a graph neural network that solves AC power flow in twelve iterations, feeding each iteration’s power mismatch into the next. The GENCO paper proposes a hybrid workflow in which power mismatch and limit checks select the cases that are re-solved with an AC solver. It does not test how well the mismatch identifies inaccurate solutions. GENCO also never reports the second convergence quantity, i.e. whether its state has stopped changing.

We record all twelve intermediate iterates of the released GENCO models, without modifying or retraining them, on 44,476 contingency cases (Texas 2000-bus, N-1 to N-20; IEEE 118-bus, N-1 to N-10) and compare every criterion against AC power flow. Criteria are scored by the AUROC: the probability that an inaccurate value receives a higher score than an accurate one (0.5 is random, 1 is perfect).

| Question | Answer |
|---|---|
| Is the final power mismatch a reliable indicator of inaccurate values? | No. AUROC is 0.45–0.85. For IEEE 118-bus voltages it is no better than chance in two of three independently trained models. |
| Is the size of the last update a better indicator? | Yes, for individual buses and branches. AUROC is 0.87–0.97 on the training contingency orders. To capture 90 % of inaccurate values it flags 2.4 to 20 times fewer buses and branches than the final mismatch. It needs no AC solution, except to set its threshold once. |
| What does one Newton step from GENCO's solution give? | A near-exact error estimate (AUROC ≥ 0.9996) and correct overload verdicts without AC verification: 7 incorrect verdicts in 20,000 Texas cases, against 1,806 for GENCO alone. |
| Which cases should be verified with AC? | Rank cases by the margin to the thermal limit divided by the expected loading error. Verifying 33.5 % of N-10 to N-20 cases detects 99.5 % of overloaded cases, against 44.5 % when ranking by the margin alone. Ranking by error magnitude alone is far less efficient (67–89 % of cases). |
| Does GENCO rank critical branches like AC? | Yes, and better than DC power flow: rank correlation 0.9999 against 0.959 on Texas. Of the 364 branches that overload under AC, GENCO identifies 324 and DC identifies 265. |


**Recommendation for GENCO users.** Where the network model is available, take one Newton step from GENCO’s solution. It measures the error almost exactly, removes most of it, and decides overloads with almost no AC verification. Without a Jacobian, use the last update to flag inaccurate buses and branches. Select the cases for AC verification by their margin to the thermal limit, expressed in units of the last update or of the error index (Section 2.4). Do not rely on the final mismatch alone.

## 1. Problem Statement

GENCO produces the states x̂<sub>t</sub> = (V̂, θ̂, P̂<sup>g</sup>, Q̂<sup>g</sup>) for iterations t = 1, …, T, T = 12, and reports x̂<sub>T</sub>. At every iteration it computes the bus power mismatch (ΔP<sub>i,t</sub>, ΔQ<sub>i,t</sub>) with its own branch-flow layer. Mismatch components that are zero by construction (ΔQ at PV and slack buses, ΔP at the slack bus) are masked per case, since a generator outage can turn a PV bus into a PQ bus. The bus mismatch is

$$
r_{i,t} = \sqrt{m^P_i\,\Delta P_{i,t}^2 + m^Q_i\,\Delta Q_{i,t}^2} \quad [\text{MVA}], \qquad m^P_i, m^Q_i \in \{0,1\}.
$$

Branch loading is L<sub>b</sub> = max(|S<sub>b,ft</sub>|, |S<sub>b,tf</sub>|) / S̄<sub>b</sub>, with S<sub>b,ft</sub> the apparent power entering branch b at its from-bus and S̄<sub>b</sub> its thermal rating; a branch is overloaded if L<sub>b</sub> > 1, a case if any branch is. With V, L from AC power flow and V̂, L̂ GENCO's final values, a value is **inaccurate** if |V̂<sub>i</sub> − V<sub>i</sub>| > τ<sub>V</sub> (PQ buses) or |L̂<sub>b</sub> − L<sub>b</sub>| > τ<sub>L</sub> (in-service branches). Inaccurate values are rare: they make up 0.04–2.5 % of bus or branch values, depending on the system and on whether the contingency order was seen in training.

The question is which quantity, available from a single GENCO run, best identifies inaccurate values, and how to use it to select the cases that need an AC power flow. Criteria that require fitting are fitted only on the contingency orders in GENCO’s training data. All higher orders then test whether a criterion still holds outside the training distribution.

| | Texas 2000-bus | IEEE 118-bus |
|---|---|---|
| Buses / branches | 2,000 / 3,220 | 118 / 186 |
| GENCO model | `gridfm/genco-pf-contingency` | `gridfm/genco-pf-datakit`, base, three trained copies (seeds 0, 1, 42) |
| Contingency orders in GENCO's training | N-1, N-2 | N-1 |
| Test cases | first 1,000 of each published test set, N-1 to N-20 (20,000) | 572 held-out N-1 cases and ten generated sets, N-1 to N-10 (23,904) |
| Error tolerances τ<sub>V</sub>, τ<sub>L</sub> | 0.005 p.u., 0.02 | 0.002 p.u., 0.05 |

The IEEE 118-bus sets were generated with gridfm-datakit and PowerModels. Each tolerance is the first configured value that yields at least 200 inaccurate values on the training contingency orders, so that the statistics rest on enough errors. It is not an operational accuracy requirement. AC power flow is the reference throughout.

## 2. Method

![Pipeline](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/lFA2BoO90wdJhHfYhoHRo.png)

**Figure 2.** One GENCO run, recorded at all twelve iterations, feeds two analyses. Top: error flags per bus and branch, an error estimate, and the screening of cases for AC verification. Bottom: the ranking of critical branches over all cases, combined with the error index to show which critical branches to verify. AC power flow is used only to score the criteria. DC power flow is the baseline for the ranking.

### 2.1 Recording the intermediate solutions

Figure 2 summarises the pipeline. We attach PyTorch forward hooks to GENCO’s mismatch and branch-flow layers and record the state, bus mismatch and branch flows at every iteration. The model code is not modified. Quantities are converted to physical units with each model’s own fitted power base (147.61 MVA for Texas, 137.93 MVA for IEEE 118-bus, not 100 MVA). Thermal ratings are taken from the raw datakit tables, because GENCO’s input sets them to zero. Before any analysis we verified on held-out data that the hooks fire 12 times per run and that the last recorded state is bitwise identical to GENCO’s output. We also verified that the recorded mean mismatch matches the value GENCO stores (relative difference 2×10<sup>−7</sup>) and that the AC solution passes GENCO’s own mismatch layer with a maximum mismatch of 0.069 MVA.

### 2.2 Criteria compared

All criteria require a single GENCO run unless stated otherwise.

- **Final mismatch** r<sub>i,T</sub>: this is the criterion proposed in the GENCO paper and serves as the main baseline.
- **Last update** (no AC solution, no fitting): it is the analogue of the Newton–Raphson step-size criterion. For voltages, we define the neighbourhood-max update s<sup>V</sup><sub>i</sub> = max<sub>j ∈ N(i)</sub> |V̂<sub>j,T</sub> − V̂<sub>j,T−1</sub>|, where N(i) is the set of buses connected to bus i by an in-service branch. For branch loadings, we use the variability over the last four iterations, s<sup>L</sup><sub>b</sub> = std(L̂<sub>b,T−3</sub>, …, L̂<sub>b,T</sub>). A variant for voltages uses the own-bus update |V̂<sub>i,T</sub> − V̂<sub>i,T−1</sub>|. A value is flagged when s ≥ c. The threshold c is the largest value that captures 90 % of the inaccurate values on the training contingency orders, and it is then held fixed.
- **Fitted error model:** a logistic regression on features of the twelve iterations (final mismatch, its decay over the iterations, number of iterations in which it rises, last updates of V, θ and loading, and their maxima over neighbouring buses). Its log-odds is the **error index**. A linear model is used because it is transparent and cheap; gradient-boosted trees agree within ±0.03 AUROC.
- **One Newton step** from GENCO's state, Δx = −J(x̂)<sup>−1</sup> g(x̂), where g is the reduced power-flow mismatch and J its Jacobian. Since g(x∗) = 0 at the AC solution x∗, to first order x̂ − x∗ ≈ −Δx. The step therefore estimates GENCO’s error at the cost of one Jacobian factorization. It also explains why the local mismatch is a poor indicator: the voltage error at bus i is row i of J<sup>−1</sup> applied to the mismatch of the whole network, not the mismatch at bus i.
- **Ensemble-spread baselines:** (i) The spread of V̂ and L̂ over eight GENCO runs in which every load is scaled by 1 + 0.01 ξ, with ξ ~ N(0, 1). (ii) On IEEE 118-bus, the spread over the three independently trained copies.
- **Proximity baseline:** a model built from the contingency order, the number of hops to the outaged element, to the nearest voltage-controlled bus and to the slack bus, and the bus degree. GENCO’s errors grow near the outage and with the contingency order. On Texas, the median final mismatch is 0.190 MVA at the terminal buses of an outaged element against 0.071 MVA five or more hops away, and 0.062 MVA for N-1 against 0.125 MVA for N-20. A criterion is therefore only useful if it adds information beyond this baseline.

### 2.3 Evaluation protocol

For each load condition, datakit builds several outage sets, so cases that share a load condition are not independent. Every cross-validation fold, resample and calibration split is therefore grouped by load condition, and each value is scored by a model that has not seen its load condition. Intervals are 95 % intervals obtained by resampling load conditions. Differences between criteria use paired resamples.

### 2.4 Screening for AC verification

A screening rule assigns a score to every case, and the share q of cases with the highest scores is verified with AC power flow. The required share q<sup>∗</sup> is the smallest q at which 99.5 % of the overloaded cases are detected. Let L̂<sup>max</sup> = max<sub>b</sub> L̂<sub>b</sub> be the highest predicted loading of a case. The rules are:

- **distance to the limit**, −|L̂<sup>max</sup> − 1|: our reading of the limit check in the GENCO paper;
- **distance in units of expected error:** each branch's margin L̂<sub>b</sub> − 1 divided by its expected loading error (a monotone fit of the loading error against the error index); the case score is the smallest such value that would flip the case's overload verdict;
- **distance in units of the last update:** the same, with the last update s<sup>L</sup><sub>b</sub> as the error scale (no AC solution needed);
- **one Newton step, then margin to the limit:** cases that are not verified take their overload verdict from the loadings after the Newton step;
- **error magnitude alone:** the case accuracy score (an aggregate of the error index), the mean final mismatch or the largest last update; and, as references, the contingency order, a random order and an ideal screening that knows the AC answer.

The Newton-step and last-update rules (†) were added post hoc, after all other results had been obtained. A rule that verifies a share q<sup>∗</sup> reduces the number of AC power flows by at most a factor 1/q<sup>∗</sup>, independent of hardware. Because the Newton rule adds a Jacobian factorization to every case, we also report cost in Newton iterations per case: one per factorization, plus q<sup>∗</sup> times the mean number of iterations of an AC run started from GENCO’s solution.

### 2.5 Critical-branch ranking

For each branch and each source (GENCO, AC, DC power flow), the **tail loading** is the mean of the 5 % highest loadings over all cases, CVaR<sub>0.95</sub>(L<sub>b</sub>) = E[L<sub>b</sub> | L<sub>b</sub> ≥ VaR<sub>0.95</sub>(L<sub>b</sub>)] (conditional value-at-risk, Rockafellar and Uryasev, 2000). We use it instead of the overload probability because most branches never overload, which leaves that probability tied at zero. Agreement with AC is measured by the rank correlation across branches and by the share of AC’s 25 most loaded branches that each source also places in its own top 25.

To show where the ranking is least reliable, a branch is **critical** if its GENCO tail loading is in the top 10 %, and **error-prone** if its mean error index, over the cases in which it is loaded above 90 %, is in the worst third. A one-sided Mann–Whitney test compares the disagreement |CVaR<sub>GENCO</sub> − CVaR<sub>AC</sub>| of critical error-prone and critical accurate branches.

## 3. Results and Discussion

### 3.1 The final mismatch is a weak indicator

The final mismatch reaches an AUROC of 0.659 for Texas bus voltages on N-1 and N-2 cases and 0.783 on N-3 to N-20. On IEEE 118-bus, for N-1 voltages, it is no better than chance for two of the three trained copies (0.454 [0.389, 0.521] and 0.480 [0.420, 0.537]; third copy 0.594). Its quality therefore depends on the trained model.

![Newton step and conditioning](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/TtWOTtjOcFtAsvbxClvFB.png)

**Figure 3.** Texas 2000-bus, 100 cases per contingency order. (a) True voltage error per PQ bus against the error estimated by one Newton step from GENCO’s solution (dashed: equality; orange: error tolerance). (b) The same errors against the final mismatch at the bus. (c) Median over cases of the largest voltage error divided by the largest mismatch, per contingency order.

A small mismatch can coincide with a large voltage error. One Newton step applies J<sup>−1</sup> to the mismatch of the whole network and reproduces the true voltage error almost exactly (rank correlation 0.998 on both systems). The same errors show no pattern against the local mismatch at the bus (Figure 3a, b). The voltage error per unit of mismatch grows 2.3-fold from N-1 to N-20 on Texas (Figure 3c). Conditioning explains little of this: the smallest singular value of J varies only from 0.074 to 0.080 across quintiles of cases. A second explanation is plausible but untested. GENCO’s training loss penalises the mismatch at every iteration, so the mismatch may be small for accurate and inaccurate states alike.

### 3.2 The last update is a strong indicator for individual buses and branches

The amount by which the estimate still changes in the last iteration separates inaccurate from accurate values far better than the final mismatch (Table 1, Figure 4).

| System | Criterion | AUROC bus / branch, trained orders | Flagged bus / branch, trained orders | Higher orders, bus: captured / flagged | Higher orders, branch: captured / flagged |
|---|---|---|---|---|---|
| Texas 2000-bus | Last update | 0.871 / 0.923 | 34 % / 23 % | 97 % / 44 % | 85 % / 29 % |
| | Final mismatch | 0.659 / 0.702 | 83 % / 72 % | 97 % / 89 % | 94 % / 81 % |
| | Fitted error model (needs AC) | 0.916 / 0.945 | 23 % / 17 % | 96 % / 34 % | 89 % / 28 % |
| IEEE 118-bus | Last update | 0.967 / 0.972 | 5 % / 6 % | 48 % / 5 % | 90 % / 23 % |
| | Final mismatch | 0.454 / 0.672 | 91 % / 91 % | 99 % / 96 % | 99 % / 97 % |
| | Fitted error model (needs AC) | 0.976 / 0.988 | 5 % / 3 % | 84 % / 10 % | 88 % / 17 % |

**Table 1.** Each threshold is set on the training contingency orders (Texas N-1 and N-2, IEEE 118-bus N-1) to capture 90 % of inaccurate values and is then held fixed. “Flagged”: share of all buses or branches flagged. “Captured”: share of inaccurate values that are flagged, over all higher orders (Texas N-3 to N-20, IEEE 118-bus N-2 to N-10). Confidence intervals: results/<system>/last_step_check.csv in the repository.

![Comparison of checks](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/cu0urprtzFkZodOMvLKEk.png)

**Figure 4.** Texas 2000-bus. AUROC with 95 % intervals for bus voltage errors (left) and branch loading errors (right).
Circles: N-1 and N-2; squares: N-3 to N-20. The legend label "Position only" denotes the proximity baseline.

On Texas, the fixed thresholds keep working at every order from N-3 to N-20 (82–100 % of inaccurate values captured). On IEEE 118-bus, the neighbourhood-based voltage criterion does not: the captured share falls from 90 % for N-1 to 40 % for N-10, while the own-bus variant retains 80–91 %. This model was trained on N-1 only, so its threshold rests on the narrowest data. Precision is low because errors are rare: capturing 90 % of inaccurate Texas bus voltages requires flagging 34 % of buses, of which only 0.1 % are actually inaccurate. The share flagged is the operator’s verification workload, and the last update reduces it by a factor of 2.4 to 20 relative to the final mismatch.

The learned error model outperforms the final mismatch in every case (paired AUROC differences +0.085 to +0.522, all intervals above zero), and its most informative features all describe how much the state is still changing at the end. On the training orders, however, the proximity-to-outage baseline predicts bus voltage errors equally well (Texas 0.937 against 0.916). The iteration-level information pays off on higher orders, where the error model reaches 0.947 against 0.771 for the proximity baseline and 0.783 for the final mismatch.

### 3.3 One Newton step is a near-exact check

| Criterion | Extra cost beyond one GENCO run | AUROC over system, error type and contingency order |
|---|---|---|
| One Newton step | one Jacobian factorization; needs the network model | ≥ 0.9996 |
| Spread over three trained copies | two more trained models; IEEE 118-bus only | 0.96–0.99 |
| Last update | none | 0.79–0.97 |
| Spread under 1 % load perturbation | seven more GENCO runs | 0.47–0.83 |
| Final mismatch | none | 0.44–0.85 |

**Table 2.** Identical cases for all criteria: 100 per contingency order on Texas, and the ten generated sets on IEEE 118-bus. The ranges therefore differ slightly from the Summary and Table 1, which use all cases.

One Newton step separates inaccurate from accurate values almost perfectly (area under the precision–recall curve ≥ 0.99; rank correlation with the true error 0.998 for voltages and 0.999–1.000 for loadings). This is what Newton–Raphson theory predicts for a state close to the solution. It is also expected, because the step estimates the very error it is tested against. Its practical value is that it also corrects the state, reducing the median Texas voltage error from 1.0×10<sup>−4</sup> to 1.1×10<sup>−6</sup> p.u. Among criteria with no additional cost, the last update is the best. From a single model and a single run, it comes within 0.01–0.04 AUROC of the spread over three trained models.

### 3.4 Screening cases for AC verification

![Screening for AC verification](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/0kfNCVMkxFpH8DNyZcc4y.png)

**Figure 5.** Texas 2000-bus, N-1 to N-5 (left) and N-10 to N-20 (right). Top: share of overloaded cases detected against the share verified with AC. Dots mark the smallest share reaching 99.5 %. Bottom: largest loading error remaining among the cases not verified. Purple: one Newton step, which reaches the target without any AC verification. In the legend, "margin" means the margin to the thermal limit.

| Rule | Texas, N-1 to N-5 | Texas, N-10 to N-20 | Texas, all | IEEE 118-bus, all | Newton iterations per case, Texas / IEEE 118-bus |
|---|---|---|---|---|---|
| † One Newton step, then distance to the limit | 0.000 | 0.000 | 0.000 | 0.001 | 1.00 / 1.00 |
| Distance in units of expected error | **0.469** | **0.335** | **0.364** | **0.127** | 0.96 / 0.30 |
| † Distance in units of the last update | 0.505 | 0.369 | 0.412 | 0.141 | 1.09 / 0.34 |
| Distance to the limit | 0.588 | 0.445 | 0.517 | 0.164 | 1.36 / 0.39 |
| Case accuracy score | 0.672 | 0.865 | 0.814 | 0.908 | 2.15 / 2.16 |
| Mean final mismatch | 0.758 | 0.894 | 0.840 | 0.920 | 2.22 / 2.18 |
| † Largest last update | 0.749 | 0.888 | 0.849 | 0.915 | 2.24 / 2.17 |
| Random order | 0.955 | 0.928 | 0.939 | 0.862 | |
| Ideal screening (knows the AC answer) | 0.105 | 0.075 | 0.082 | 0.043 | |
| AC power flow for every case, flat start | 1 | 1 | 1 | 1 | 4.08 / 4.04 |

**Table 3.** Share of cases verified with AC to detect 99.5 % of overloaded cases, and cost in Newton iterations per case (Section 2.4). † Added after all other results.

**With a Jacobian.** Overload verdicts taken after one Newton step (Figure 5) are incorrect in 7 of 20,000 Texas cases, against 1,806 for GENCO alone. On IEEE 118-bus the figures are 139 against 1,211. No AC verification is needed in either case. The gain lies in correctness rather than cost. The step costs 1.0 iteration per case, about the same as the best rule without a Jacobian on Texas (0.96). On IEEE 118-bus that rule costs only 0.30, so the Newton step is more expensive there; 95 % of the cases are overloaded and few verifications are needed.

**Without a Jacobian.** The margin to the limit is the appropriate baseline. Dividing it by the expected error lowers the share verified for N-10 to N-20 from 44.5 % to 33.5 % (paired interval of the difference [−0.154, −0.058]). Dividing it by the last update, which needs no AC solution, gives 36.9 %. Ranking by error magnitude alone is far less efficient than ranking by margin. For N-1 to N-5, the error-magnitude rules (case accuracy score, mean final mismatch, largest last update) require 67–76 % of the cases, against 59 % for the margin and 96 % for random selection. For N-10 to N-20 they require 87–89 %, against 44.5 % for the margin and 93 % for random selection. A missed overload is a case whose highest loading lies just below 1, so whether GENCO’s error matters depends on the margin to the limit as much as on the error magnitude. Ideal screening needs 7.5 % for N-10 to N-20, so most of the achievable gain without a Jacobian remains open.

AC power flows warm-started from GENCO’s solution need 2.64 iterations on Texas and 2.37 on IEEE 118-bus, which is 35 % and 41 % fewer than the 4.08 and 4.04 from a flat start. Verifying a share q<sup>∗</sup> reduces the number of AC power flows by at most a factor 1/q<sup>∗</sup>. With the best rule without a Jacobian, this is a factor of 2.7 over all Texas cases and 3.0 for N-10 to N-20. We make no wall-clock speed-up claim, because it depends on the AC solver and the hardware, and we did not benchmark against production tools.

### 3.5 Critical-branch ranking

![Critical-branch ranking against AC and DC](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/bhmtq6Y5nH1Z4_r7Zg_NY.png)

**Figure 6.** Texas 2000-bus, 20,000 cases. Top: tail loading per branch from GENCO (left) and DC power flow (right) against AC. Bottom: GENCO minus DC in rank correlation and top-25 agreement with AC, per range of contingency orders.

| Texas 2000-bus, tail loading | GENCO | DC power flow |
|---|---|---|
| Rank correlation with AC (3,220 branches) | 0.9999 | 0.959 |
| Share of AC's top 25 branches found | 1.00 | 0.88 |
| Branches overloaded at least once under AC (364): found / false alarms | 324 / 24 | 265 / 5 |

GENCO is closer to AC than DC power flow is (Figure 6) in every range of contingency orders (rank correlation +0.041 [0.040, 0.041], top-25 agreement +0.12 [0.04, 0.16]). The DC rank correlation is 0.959 in every range, so its error is systematic, which is consistent with DC neglecting reactive power and voltage magnitude. Part of the gap results from the choice of measure, since DC loading uses active power only. Rank correlation across 3,220 branches is close to 1 for any good estimate, so the top-25 agreement and the counts of overloaded branches identified are the more informative comparisons. These rankings average over 20,000 cases. They show that GENCO identifies where the grid is stressed, not that individual flows are exact. On IEEE 118-bus, GENCO identifies all of AC’s top 25 and DC power flow 24 (0.96), which is close to a ceiling with only 186 branches.

![Critical and error-prone branches](https://cdn-uploads.huggingface.co/production/uploads/6ade8cc5fff25f12f5807c24/jtUhVBqUNSbPVesEXyB3n.png)

**Figure 7.** Texas 2000-bus. Branch colour: GENCO tail loading. Dashed: error-prone. Thick dashed: critical and error-prone. Inset: tail loading against the mean error index, with the 205 critical error-prone branches highlighted.

Of the 322 critical branches on Texas, 205 are error-prone (Figure 7). Their GENCO tail loading differs from AC's by a median of 9.6×10<sup>−4</sup>, against 2.1×10<sup>−4</sup> for the 117 accurate ones, a factor of 4.6 (one-sided p = 3×10<sup>−15</sup>). Loading and accuracy are correlated (rank correlation −0.38), but the two groups have similar loading (median 0.700 against 0.679) and the effect holds within every loading quartile (p ≤ 0.004). The error index comes from the fitted error model, so this is the error check aggregated per branch; the final mismatch, used the same way, shows no effect (effect size as rank-biserial correlation 0.04, against 0.52). The disagreements are small in absolute terms (median at most 10<sup>−3</sup> of the rating): the map shows where to verify with AC, not that GENCO is unusable there.

### 3.6 What did not work
- **The evolution of the mismatch over the iterations carries no signal.** We expected inaccurate buses to show stalling, oscillation or slow decay. Such features reach only 0.34–0.48 AUROC (below chance) for own-bus voltage errors on the training orders. The size of the own-bus last update reaches 0.84 on Texas and 0.97 on IEEE 118-bus. Inaccurate buses do not stall: their voltages are still changing.
- **GENCO's gradients are poor sensitivities.** On IEEE 118-bus, the load-to-flow sensitivities ∂L<sub>b</sub>/∂P<sup>d</sup><sub>i</sub> obtained by differentiating through the twelve iterations have the right sign and roughly the right magnitude, but rank buses worse than DC power transfer distribution factors (rank correlation with AC 0.83 / 0.81 / 0.76 against 0.99 / 0.99 / 0.98 for N-1, N-5 and N-10). Accurate states do not imply accurate derivatives. 
- **Accepting whole cases using the per-element thresholds fails.** Every case has at least one bus or branch above the thresholds of Table 1, so such a rule verifies every case. Aggregated as the maximum per case, the last update identifies cases containing an inaccurate value less well than the final mismatch does (AUROC 0.68–0.72 against 0.86 on Texas).


## 4. Limitations

- **Data.** We use two systems. For Texas we had a single released model, the first 1,000 of about 8,800 cases of each test set, and uniformly random outages, whereas real outages are spatially clustered. The IEEE 118-bus contingency sets were generated by us, and that model was trained on N-1 only.
- **Choices made after seeing development results:** these are the rule of at least 200 inaccurate values for setting the tolerances, the margin-to-limit rules (defined on IEEE 118-bus and then tested on Texas), the aggregation into the case error score, and the use of tail loading instead of overload probability. The Newton-step and last-update screening rules were defined after all other results had been obtained, so neither system serves as a held-out test for them. The expected-error scale is fitted on the training orders, so the shares reported for N-1 to N-5 and for all orders may be slightly optimistic.
- **The last update flags large errors but does not quantify them:** its rank correlation with the magnitude of the true error is 0.26–0.36 for Texas bus voltages, against 0.998 for one Newton step.
- **The screening rules target missed overloads, not flow accuracy.** Among the Texas cases that are not verified, loading errors of up to 5.3 times the rating remain. In one N-19 case, GENCO places 1,707 MVA on a 310 MVA, 161 kV line that carries 60 MVA under AC. The case is not selected, because its predicted loading is far from the limit. One Newton step reduces the largest remaining error to 2.5 times the rating. Applications that need accurate flows, and not only overload verdicts, must also verify the cases with the largest errors with AC power flow.
- **Cost.** The Newton-step cost estimate assumes that a Jacobian factorization costs as much as one Newton iteration. No wall-clock time was measured with production tools.
- **The critical-branch map measures how hard a branch is for GENCO, not how weak the grid is;** we draw no conclusions about network reinforcement. The analysis could not be tested on IEEE 118-bus, where only 1 of 19 critical branches is error-prone.

## 5. Reproducibility

Code, configurations and the result tables behind every number and figure: https://github.com/radiakos/genco-xray

## References

- A. Puech et al. GENCO — A unified neural solver embedded in a development framework for steady-state grid analysis.
  arXiv:2608.09921, 2026.
- gridfm-datakit. arXiv:2512.14658.
- R. T. Rockafellar and S. Uryasev. Optimization of conditional value-at-risk. *Journal of Risk*, 2(3):21–41, 2000.
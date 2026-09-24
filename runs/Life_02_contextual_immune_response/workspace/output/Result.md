# Context-Dependent Immune Response Across Compartments, Endpoints, Interventions and Time

**Task:** `Life_02_contextual_immune_response` · **T0 boundary:** `2020-01-01`
**Evidentiary basis:** the four bundled TSV files in `data/` and nothing else. No external dataset,
no network access, no literature source is used anywhere in this report. Every number below was
produced by executed code in `output/analysis/`; the console logs are preserved in `output/logs/`.

---

## Summary of registered findings

| # | Finding | Direction | Core result |
|---|---|---|---|
| 1 | Perturbation induces the response in every compartment, and the magnitude is assay-dominated | supported | 18/18 cells up, mean +0.871 log2 (1.83×), 95% CI [0.749, 0.992] |
| 2 | The compartment ordering of effect magnitude does not generalise across assays, and the ordering is unreadable at replicate resolution | boundary / null | count W = 0.778 (p = 0.097); fraction W = 0.111 (p = 0.717); within-cell SD 0.711 log2 exceeds the whole observed compartment span 0.279 log2 |
| 3 | Count and fraction endpoints agree on direction but conflict on ordering in one of three assays | conditional conflict | 18/18 cells up in both; paired Δ +0.111 log2, p = 0.0495; fraction ordering is the reverse of count in `assay_3` (ρ = −1.00) |
| 4 | Time-resolved: the two condition arms rise together with no resolvable separation at any time point | null with stated resolution | all 7 paired p ≥ 0.506; interaction p = 0.430; MDE 11.1–22.7 pp vs largest observed −3.06 pp |
| 5 | Intervention setting ordering is concordant across assays but sits at the edge of what three assays can resolve | supported with boundary | W = 0.956, χ² = 11.47, p = 0.0218; a single rank transposition removes significance |

Two further results are reported as supporting evidence rather than as separate registered findings:
the multi-condition class contrast (Finding 1's corroborating module) and the leave-one-assay-out
prediction failure (Finding 2's primary evidence).

---

## Data, analysis unit and design

The four files form complete, fully balanced factorial designs with no missing values and no
imputation (`DATA_GUIDE.md`: "No curator-imposed row deletion, random subsampling, or imputation"):

| File | Rows | Design | Analysis unit |
|---|---|---|---|
| `compartment_response_replicates.tsv` | 144 | 3 assays × 3 compartments × 2 endpoint types × 2 condition roles × 4 replicates | biological replicate within an assay × compartment × endpoint × condition cell |
| `mechanistic_intervention_replicates.tsv` | 45 | 3 assays × 5 intervention settings × 3 replicates | biological replicate within assay × setting |
| `multi_condition_response_replicates.tsv` | 24 | 2 response classes × 4 compound settings × 3 replicates | biological replicate within class × setting |
| `time_resolved_response.tsv` | 56 | 2 condition roles × 7 time points (3–21) × 4 replicates | biological replicate paired across roles by `replicate_id` |

All four files hash-match `data_manifest.json` (verified in `results/run_manifest.json`). Per
`DATA_GUIDE.md`, repeated endpoints from a shared assay are **not** independent experiments; every
inferential test below is therefore run either on a condition-role mean per cell or with the assay
as the held-out/random unit. The intervallic placements in the compartment file (e.g. 3.86 vs
3.86×10⁵) show that `endpoint_count` and `endpoint_fraction` are on different scales and are not
pooled.

**Fold-change definition.** Throughout, fold change is the geometric mean of the four per-replicate
perturbation/baseline ratios, i.e. 2^mean(log₂(pert/base)). This is the primary definition because
it is symmetric on the log scale and is not driven by the noisiest replicate. The
ratio-of-means and arithmetic-mean-of-ratios alternatives are reported as robustness checks; they
change the numbers but not any conclusion (see Finding 1 and Finding 3).

---

## Finding 1 — The perturbation raises the response in every tissue compartment, and the size of that raise is dominated by assay identity rather than by compartment identity

**Status: supported.**

**Object and scope.** All 18 assay × endpoint × compartment cells of
`data/compartment_response_replicates.tsv` (3 assays, 3 compartments, 2 endpoint types), 4 biological
replicates per cell.

**Relationship and direction.** Perturbation increases the response relative to baseline in every
cell, with no exceptions.

**Core quantitative result.** 18/18 cells have fold change > 1; median cell fold change **1.868**
(range 1.382–2.531). At replicate level, **69/72** pairs increase (one-sided binomial p = 1.3×10⁻¹⁷).
The mean cell log₂ fold change is **+0.871** (95% CI [0.749, 0.992], df = 17; one-sample
t = 15.10, p = 2.8×10⁻¹¹). Analysis unit: the assay × endpoint × compartment cell (n = 18); the
replicate-level test is reported as a secondary, deliberately over-precise statistic.

**Which context factor matters.** A one-way sum-of-squares partition of the 18 cell log₂ fold
changes gives **assay 55.3%**, compartment 25.8%, endpoint 5.4%. Assay-level means are `assay_3`
1.113, `assay_1` 0.801, `assay_2` 0.697 log₂ — a 1.63× spread between the most and least inducible
assay. A permutation test on the 18 cell means (20 000 label permutations, seed 20200101) gives
**p = 0.0037** for assay identity. Assay identity, not compartment identity, is the largest single
context effect on effect magnitude.

**Supporting analysis and result evidence.**
- `analysis/01_compartment_intervention.py` (F001) → `results/compartment_cell_effects.csv` (F011),
  `results/compartment_replicate_effects.csv` (F013), `results/core_analysis_summary.json` (F041)
  keys `S1_perturbation`.
- `figures/fig1_compartment_fold_change.png` (F049) — all 18 cells above the fold-change = 1 line.
- Independent recomputation: `analysis/05_verification_compartment.py` (F005) →
  `results/verification_compartment.json` (F045), 14/14 checks agree.
- Console log: `logs/01_compartment_intervention.log` (F053).

**Robustness.**
1. *Alternative fold-change definition.* Arithmetic mean of per-replicate ratios gives mean FC
   **2.045** [1.815, 2.274] instead of 1.833; the ratio-of-means definition gives 1.832. Direction
   and significance are unchanged.
2. *Outlier treatment.* Dropping the highest and lowest cell gives mean log₂ = **0.8665**
   [0.757, 0.976]. Dropping the replicate pair with the most extreme value changes nothing material;
   the 3 non-increasing replicates (log₂ ≤ 0, all in `assay_2`/`assay_3`, smallest −0.269) are
   retained rather than trimmed, and are named in `results/robustness_summary.json` (F042).
3. *Complete-assay leave-one-out.* Omitting each assay in turn: mean log₂ = 0.905 (omit `assay_1`),
   0.957 (omit `assay_2`), 0.749 (omit `assay_3`); all three CIs exclude zero.
   (`results/robustness_assay_leave_out.csv`, F026.)
4. *Adversarial cell-level sign test.* Counting each of the 18 cells once instead of pooling 72
   pairs: 18/18 positive, one-sided p = 3.8×10⁻⁶. The pooled test overstates precision by roughly
   11 orders of magnitude in p, but not the direction.

**Counterevidence, limitations, alternative explanations.**
- The claim is about *this* perturbation in *these* three assays. Nothing here identifies which
  signalling route is responsible; `DATA_GUIDE.md` states explicitly that compartment and
  intervention associations do not by themselves identify a unique cellular mechanism. This report
  makes no mechanistic claim.
- The 55.3% assay share is a one-way marginal SS at n = 18 cells; assay, compartment and endpoint
  are non-orthogonal in this partition, so the shares are descriptive, not a variance decomposition
  from a fitted model.
- Three replicate pairs decrease. `assay_2`/`endpoint_count`/`compartment_3`/`replicate_2` is the
  clearest (baseline 721.84 → perturbation 598.90, log₂ = −0.269). With n = 4 per cell, one such
  replicate is inside the expected scatter.
- A plausible non-biological alternative: the two "endpoint" readouts may carry different
  normalization, which would inflate assay-level differences. Finding 3 tests this directly and
  finds the endpoints behave as independent readouts (`results/robustness_summary.json` key
  `R5_endpoints`).

**Falsification or independent replication.** Repeat the experiment with a *fourth* assay and
pre-register the prediction that all 6 of its cells show fold change > 1 and that its mean log₂
fold change falls inside [0.697, 1.113] (the observed assay range). A single cell failing to rise,
or a fourth assay mean outside that band, falsifies or bounds the claim. Equivalently, re-derive the
fold changes from a raw archive of the same readouts and test whether 69/72 replicate pairs
reproduce.

---

## Finding 2 — The compartment ordering of effect magnitude does not generalise across assays, and the ordering is being read off differences smaller than the per-replicate noise floor

**Status: boundary / null result.** This is a negative finding about reproducibility, not a claim
that a gradient exists.

**Object and scope.** The rank order of the three compartments by fold change and by baseline
level, computed separately within each of the three assays, for each endpoint type.

**Relationship, direction, boundary.** The compartment ordering is *not* reproducible across assays
for the fraction endpoint, and only marginally concordant for the count endpoint. Critically, the
ordering is not resolvable with 4 replicates: the within-cell replicate SD of log₂ fold change
(0.711) is **2.5× larger** than the entire observed compartment span (0.279 log₂).

**Core quantitative result.**
- *Ordering concordance (Kendall's W over the 3 assay orderings, 3 compartments):*
  `endpoint_count` fold change **W = 0.7778** (χ² = 4.667, df = 2, p = 0.0970; exact permutation
  p = 0.1944); `endpoint_fraction` fold change **W = 0.1111** (χ² = 0.667, p = 0.7165; exact
  p = 0.9444). Baseline-level orderings behave the same way (count W = 0.7778, fraction W = 0.3333).
- *Best-case single compartment:* `compartment_1` has the **highest fold change in all 3 assays for
  the count endpoint** (2.141, 1.984, 2.531) but in only 2 of 3 assays for the fraction endpoint
  (`assay_3` puts `compartment_3` first, 2.141 vs 2.004).
- *Resolution floor:* pooled within-cell RMS SD of log₂ fold change = **0.711** (median 0.684, χ²
  95% CI [0.598, 0.875], df = 54). The smallest reliably detectable between-compartment difference
  is **1.54 log₂ ≈ 2.91×** (Tukey k = 3, df = 6) — larger than the whole observed compartment span
  of **1.21×** (0.279 log₂).
- *Complete-assay leave-one-out:* predicting a held-out assay's cells from the training assays'
  baseline levels is *worse* than predicting a constant. Pooled Spearman between prediction and
  observation is **−0.586** for the assay-invariant model, **−0.484** for rank-transfer and
  **−0.228** for level-transfer; within-assay ordering Spearman is only +0.25 (per-fold:
   1.0, 0.5, 0.5, −0.5, 1.0, −1.0).

**Supporting analysis and result evidence.**
- `analysis/01_compartment_intervention.py` (F001) → `results/compartment_ordering_reproducibility.csv`
  (F012), `results/loocv_assay_predictions.csv` (F021), `results/loocv_rank_accuracy.csv` (F022),
  `results/core_analysis_summary.json` (F041) keys `S2_ordering_reproducibility`, `S4_loocv`.
- `analysis/03_robustness.py` (F003) → `results/robustness_rank_concordance.csv` (F031),
  `results/robustness_ordering_gaps.csv` (F030), `results/robustness_ordering_ambiguity.csv` (F029).
- `analysis/06_sensitivity.py` (F008) → `results/sensitivity_report.json` (F043) key
  `replicate_resolution`.
- `figures/fig2_baseline_vs_foldchange.png` (F050) — baseline spans 5.2 log₁₀ units while fold
  change is compressed into 1.37–2.25.
- Console logs: `logs/01_compartment_intervention.log` (F053), `logs/03_robustness.log` (F055).

**Robustness.**
1. *Tie/threshold sensitivity.* Two of the six assay × endpoint strata have their bottom two
   compartments within 5% of each other — `assay_2`/`endpoint_count` (0.46% apart) and
   `assay_3`/`endpoint_fraction` (0.24% apart). Those two orderings carry no information.
2. *Scale-free rank test.* The conclusion is unchanged when only ranks are used: identical ranks
   across all three assays exist for 1/3 compartments (count) and 0/3 (fraction).
3. *Alternative estimand.* The failure is not an artefact of the log scale: baseline-level ordering
   and fold-change ordering disagree within `assay_2` (count) and `assay_1` (fraction) — 2 of 6
   strata disagree internally on which compartments even rank where
   (`results/compartment_ordering_reproducibility.csv`, F012).
4. *How much more data would help.* Holding the observed W fixed, the count ordering would reach
   p < 0.05 at **~5 assays** (exact) / 4 (χ²); the fraction ordering at the observed near-chance
   W = 0.111 would need **27 assays**. Adding that many chance-level orderings would make the test
   significant without making the ordering reproducible — which is the point of the boundary, not a
   rescue of it.

**Counterevidence, limitations, alternative explanations.**
- The fraction endpoint is the weaker of the two, but the count endpoint is *not* safely
  reproducible either: its W = 0.778 is non-significant at 3 assays (exact p = 0.194), and a single
  discordant assay (`assay_2` swaps compartments 2 and 3) drives most of the discordance.
- The χ² approximation for Kendall's W is anti-conservative here; exact permutation p-values are
  roughly twice as large (0.097 → 0.194 for count). The boundary is therefore *stronger* than the
  χ² numbers suggest, and both are reported.
- Alternative explanation for the whole pattern: the compartment ordering may be real but too small
  to measure with 4 replicates. This is a live alternative and is exactly what the resolution
  calculation formalises — the data cannot separate "no ordering" from "small ordering".
- `assay_2`'s baseline range is much narrower, so its rank information is intrinsically weaker.

**Falsification or independent replication.** Run additional assays (≥5 concordant ones for the
count endpoint), or increase replicates per cell to ~16–20 so that the smallest detectable
between-compartment difference drops below the observed span of 0.279 log₂. If the compartment
ordering is real, additional assays must preserve `compartment_1` ≥ the others on fold change for
the count endpoint. A pre-registered fourth and fifth assay that do not reproduce the ordering
falsifies the concordance claim; reproduction with adequate power would validate it.

---

## Finding 3 — The count and fraction endpoints agree that the response rises, but they conflict on the compartment ordering in one of three assays and disagree on effect size at the margin

**Status: conditional conflict.**

**Object and scope.** The two `endpoint_type` levels of the compartment file, compared on the same
9 assay × compartment strata (paired), and compared for interior consistency.

**Relationship, direction, conflict.** Both endpoints show the response rising (18/18 cells up
overall), but they are **not interchangeable**: the fraction endpoint gives a systematically smaller
effect, and in `assay_3` the two endpoints rank the compartments in *exactly opposite* order.

**Core quantitative result.**
- *Magnitude:* mean fold change is **1.930** for `endpoint_count` vs **1.777** for
  `endpoint_fraction`. Paired over 9 assay × compartment strata, the count-minus-fraction difference
  is **+0.111 log₂** (95% CI [+0.000, +0.221]; paired t = 2.313, **p = 0.0495**). This is marginal —
  the CI's lower bound touches zero, and the non-parametric sign test is not significant
  (7/9 strata, p = 0.180). The mixed model with a random intercept for assay gives an endpoint
  coefficient of **−0.111 log₂ (p = 0.472)**.
- *Ordering conflict:* within-assay Spearman between the count and fraction compartment orderings is
  **+0.50** (`assay_1`), **+0.50** (`assay_2`) and **−1.00** (`assay_3`). One of three assays gives
  an exactly reversed ordering.
- *Internal consistency:* if fraction were a percentage of count, the implied denominator
  (count/fraction) would be constant within a stratum. It is not — the implied denominator's CV is
  **2.40** overall (per-stratum CVs 0.28–1.38), and the mean within-stratum Spearman between count
  and fraction is **−0.167**, with opposite signs in different assays (−0.839, −0.888, −0.259,
  −0.713 for `assay_1`/`assay_3` vs +0.874, +0.825 for `assay_2`).

**Supporting analysis and result evidence.**
- `analysis/01_compartment_intervention.py` (F001) → `results/endpoint_type_comparison.csv` (F016),
  `results/endpoint_paired_deltas.csv` (F015), `results/endpoint_ordering_agreement.csv` (F014),
  `results/core_analysis_summary.json` (F041) key `S3_endpoint_conflict`.
- `analysis/03_robustness.py` (F003) → `results/robustness_summary.json` (F042) key `R5_endpoints`.
- Independent recomputation: `analysis/05_verification_compartment.py` (F005) →
  `results/verification_compartment.json` (F045) checks V1g/V1h, all agree.
- `figures/fig1_compartment_fold_change.png` (F049) — the two endpoint panels side by side.

**Robustness.**
1. *Paired vs unpaired framing.* The paired t (p = 0.0495) is marginal; the sign test (p = 0.180)
   and the assay-random-intercept mixed model (p = 0.472) are not significant. The honest statement
   is that the magnitude gap is at the margin of detectability and depends on the test chosen.
2. *Consistency check as an alternative explanation.* Because the implied denominator is not
   constant, the two endpoints are independent readouts rather than a shared-denominator pair. This
   rules out the simplest explanation for the ordering conflict (that one is a deterministic
   transform of the other) and leaves a genuine endpoint-specific ordering difference.
3. *Ordering-level check.* The conflict survives when the comparison is made on ranks within each
   assay, so it is not a scaling artefact of the different units.

**Counterevidence, limitations, alternative explanations.**
- The ordering "conflict" rests on a single assay (`assay_3`); with only 3 compartments, Spearman
  ρ = −1.00 is a coarse statement. The conflict should be read as "the endpoints are not
  interchangeable", not as "the fraction endpoint is wrong".
- The magnitude difference may partly reflect genuine biology (different readout types are expected
  to respond with different gain) rather than measurement error. This dataset cannot separate the
  two.
- Seven of nine strata agree in direction for the sign test, so the conflict is specifically about
  *ordering*, not about the direction of the perturbation effect.

**Falsification or independent replication.** Add a fourth assay and test whether the count and
fraction orderings agree (ρ > 0). If the conflict is systematic rather than an `assay_3` idiosyncrasy,
additional assays must show further reversals; a pre-registered set of assays all showing ρ = +1
would refute the conflict claim. Independently, obtain the raw denominator behind each readout: if
the fraction's denominator is in fact the count (a shared-denominator design), the "independent
readouts" inference is falsified.

---

## Finding 4 — Time-resolved: both condition arms rise substantially over the time course and the data provide no resolvable separation between them at any time point

**Status: null effect with a stated resolution.**

**Object and scope.** `data/time_resolved_response.tsv`, 2 condition roles × 7 time points
(3, 6, 9, 12, 15, 18, 21) × 4 biological replicates, paired across roles by `replicate_id`.

**Relationship, direction, boundary.** There is a clear shared increase over time in both arms.
There is **no detectable difference between the arms at any time point**, and — importantly — the
design could not have detected a difference of the size the data themselves suggest.

**Core quantitative result.**
- *Shared trend:* OLS slope of `normalized_response_percent` on `time_units` over all 28
  observations per role: **baseline +0.839 %/unit** (p = 8.4×10⁻⁶, R² = 0.540) and
  **perturbation +0.694 %/unit** (p = 1.8×10⁻⁶, R² = 0.591). From t = 3 to t = 21 the mean response
  rises **+13.84 percentage points** (100.54 → 114.37).
- *Role contrast:* paired differences (perturbation − baseline) at each time point are
  −0.52, −0.52, +1.00, −1.08, +0.20, −2.38 and −3.06 pp. **Every one of the 7 paired tests is
  non-significant** (minimum p = 0.506 at t = 18; the largest-magnitude point, t = 21, has p = 0.516).
- *Interaction:* the time × role interaction coefficient is **−0.145 pp/unit (p = 0.430)** from a
  mixed model with a random intercept for `replicate_id`; the difference-series OLS slope is the same
  value (p = 0.520, R² = 0.016).
- *Resolution floor (this is the load-bearing number):* with n = 4 pairs and the observed paired SDs
  (5.33–10.91 pp, pooled RMS 7.74 pp), the **minimum detectable effect at 80% power, α = 0.05
  two-sided is 11.08–22.70 pp (median 13.15 pp; 16.10 pp using the pooled SD)**. The largest
  observed difference, **−3.06 pp at t = 21, lies inside that floor** (MDE at t = 21 = 17.32 pp;
  power at the observed effect = 0.083).

**Supporting analysis and result evidence.**
- `analysis/02_time_resolved.py` (F002) → `results/time_resolved_contrasts.csv` (F035),
  `results/time_resolved_segments.csv` (F039), `results/time_resolved_summary.json` (F044).
- `analysis/03_robustness.py` (F003) → `results/robustness_time_trends.csv` (F033),
  `results/robustness_time_trend_loo.csv` (F032).
- `analysis/06_sensitivity.py` (F008) → `results/sensitivity_report.json` (F043) keys
  `mde_timepoints`, `tost`.
- Independent recomputation: `analysis/05_verification_time.py` (F007) →
  `results/verification_time.json` (F048), 5/5 checks agree.
- `figures/fig4_time_resolved.png` (F052) — both arms rising together while the paired difference
  stays near zero.

**Robustness.**
1. *Unpaired test.* Welch tests ignoring the replicate pairing give the same significance calls at
   all 7 time points (e.g. t = 21: Welch p = 0.488 vs paired p = 0.516).
2. *Leave-one-replicate-out.* The difference-series slope ranges from −0.264 to +0.070 pp/unit
   across the 4 folds and is non-significant in all; a mixed-model time × role interaction remains
   non-significant. The one nominally significant per-replicate slope (`replicate_2`, −0.787, p = 0.021)
   is not corroborated by the other three (t-test on the 4 replicate slopes vs 0: p = 0.564).
3. *Alternative contrast.* Using the ratio (perturbation/baseline) instead of the difference: slope
   **−0.00115 per unit, p = 0.591**, and ratios by time point 0.996, 0.997, 1.011, 0.994, 1.006,
   0.980, 0.977 — the same near-null picture.
4. *Non-parametric trend.* Spearman(difference, time) = **−0.117, p = 0.553**.
5. *Equivalence testing (TOST).* Equivalence is declared at **0/7** time points under either a
   one-SD margin or a fixed 3 pp margin (minimum TOST p = 0.072 and 0.210 respectively). This
   **must not** be read as evidence of a difference — at n = 4 the 90% CI half-width (1.177 × SD)
   structurally exceeds a one-SD margin, so equivalence is unreachable by construction.

**Counterevidence, limitations, alternative explanations.**
- The observed late-time divergence is consistently negative (−2.38 pp at t = 18, −3.06 pp at
  t = 21, and the difference series has a negative slope). This is a weak, non-significant hint of
  an arm difference that grows with time; it is reported rather than discarded, but it is far inside
  the resolution floor and three of four replicate slopes do not support it.
- Both arms are normalized to the same reference, so any arm-specific normalization offset would be
  invisible here.
- Non-significance with this sample size is uninformative about effects below ~13 pp. The correct
  conclusion is bounded ignorance, not evidence of no effect.

**Falsification or independent replication.** Increase replicates per arm per time point to ~16–20
(the paired SD is ~8 pp, so n ≈ 16 gives an MDE near 5 pp) and re-test the t = 18 and t = 21
contrasts. A pre-registered replication with adequate power that finds a significant negative
difference at t ≥ 18 would convert the observed hint into a supported kinetic boundary; a
well-powered null would establish the boundary as genuine. Conversely, demonstrating a
replicate-level pairing failure (that `replicate_id` does not actually pair across roles) would
invalidate the paired analysis.

---

## Finding 5 — Intervention-setting ordering is strongly concordant across assays, but the conclusion sits at the edge of what three assays can resolve

**Status: supported, with an explicit boundary.**

**Object and scope.** `data/mechanistic_intervention_replicates.tsv`: 3 assays × 5 intervention
settings × 3 replicates. Each assay is a separate readout scale (`assay_2` responses are ~130× the
`assay_1` medians), so only the *ordering* of settings is compared across assays, never the levels.

**Relationship and direction.** The five intervention settings have a highly reproducible ordering
across the three assays, and one setting is the lowest in all three.

**Core quantitative result.**
- Rank matrix (1 = lowest response): `assay_1` = (1, 5, 3, 2, 4), `assay_2` = (1, 4, 3, 2, 5),
  `assay_3` = (1, 5, 3, 2, 4).
- **Kendall's W = 0.9556** (χ² = 11.467, df = 4, **p = 0.0218**; Friedman test p = 0.0150; exact
  permutation p = 0.00090).
- **`intervention_setting_1` is the lowest-response setting in all three assays.**
- Settings 3 and 4 hold ranks 3 and 2 in all three assays (spread 0). Settings 2 and 5 swap between
  `assay_2` and the other two (ranks 4/5 vs 5/4; spread 1).
- Leave-one-assay-out: training on 2 assays and ranking the third gives ρ = **+0.900** in all three
  folds, with 3 of 5 exact rank positions matched.

**Supporting analysis and result evidence.**
- `analysis/01_compartment_intervention.py` (F001) → `results/intervention_setting_means.csv` (F019),
  `results/intervention_assay_by_setting_matrix.csv` (F017),
  `results/intervention_leave_one_assay_out.csv` (F018),
  `results/intervention_setting_rank_spread.csv` (F020),
  `results/core_analysis_summary.json` (F041) key `S5_intervention`.
- `analysis/06_sensitivity.py` (F008) → `results/sensitivity_report.json` (F043) key
  `intervention_boundary`.
- Independent recomputation: `analysis/05_verification_intervention.py` (F006) →
  `results/verification_intervention.json` (F046), 9/9 checks agree (including exact enumeration of
  the 3⁵ permutation null).
- `figures/fig3_intervention_profiles.png` (F051).

**Robustness.**
1. *Alternative test.* Friedman's test agrees (p = 0.0150), as does the exact permutation null
   (p = 0.00090). The χ² approximation is *conservative* here, unlike for the compartment ordering.
2. *Threshold/swap sensitivity.* This is the important limit. Of all 30 single transpositions of the
   observed rank matrix, only **15/30** retain χ² p < 0.05 (20/30 under the exact null). Transposing
   settings 1 and 3 in any one assay already breaks significance (S = 70, χ² p = 0.053). The
   observed S = 86 sits between the χ² cutoff (S > 71.2) and the exact cutoff (S = 64).
3. *Cross-check on the two discordant settings.* The swap is confined to settings 2 and 5 between
   one assay and the other two; it is consistent across the rank analysis, the row-wise rank spread,
   and the leave-one-assay-out folds.

**Counterevidence, limitations, alternative explanations.**
- The claim is about *relative ordering* only. Absolute magnitudes differ ~130× between assay
  medians, so a shared ordering does **not** imply a shared potency or mechanism. Per
  `DATA_GUIDE.md`, this association does not identify a unique cellular mechanism, and no
  mechanistic claim is made.
- The conclusion is fragile to a single rank swap. With 3 assays, Kendall/Friedman has very low
  power, so "p = 0.0218" should be read as "at the edge of resolvability", not as a strong result.
- `assay_2`'s peak is `intervention_setting_5` while `assay_1` and `assay_3` peak at
  `intervention_setting_2`; the "peak setting" is therefore not a concordant feature even though the
  low end is.

**Falsification or independent replication.** Add two or more independent assays and pre-register
the prediction that `intervention_setting_1` ranks lowest and that settings 3 and 4 hold ranks 3 and
2. A single additional assay that breaks the ordering at the low end would falsify the concordance
claim; agreement in 5+ assays would remove the fragility that the swap analysis exposes.

---

## Supporting result (not separately registered) — Multi-condition dose response separates the two response classes

Reported as corroborating evidence for the reality of context-dependent response, and as the
clearest quantitative contrast in the dataset. It is *not* registered as a separate finding because
it restates Finding 1's message in a second data module rather than adding an independent
conclusion.

`data/multi_condition_response_replicates.tsv` (2 classes × 4 compound settings × 3 replicates):
`response_class_A` rises monotonically and steeply — slope **+8.593 per setting** (SE 0.564,
p = 3.0×10⁻⁸, R² = 0.959), setting means 8.51 → 17.97 → 26.45 → 34.33. `response_class_B` is flat —
slope **+0.191** (p = 0.584, R² = 0.031), setting means 7.41 → 7.54 → 7.33 → 8.12, with a
non-monotone step (−0.207). The class × setting interaction is **F = 45.41, p = 4.7×10⁻⁸**.

*Robustness (repeated-measures unit respected).* Per-replicate slopes: class A [8.782, 9.291, 7.706]
vs class B [0.403, 0.825, −0.654] — **complete separation**, the largest class B slope is still
below the smallest class A slope (Welch t = 13.09, p = 0.0002; a mixed model with a replicate
random intercept gives interaction −8.40, p = 3.5×10⁻⁴⁶). *Counterevidence:* with only 3 replicates
per class the exact permutation test over all C(6,3) = 20 label assignments gives **p = 0.100** —
complete separation of 3 vs 3 cannot reach p < 0.05 by permutation. The F-test/regression p-values
rest on the parametric assumptions; the assumption-free test is marginal.
Evidence: `results/multi_condition_class_fits.csv` (F024), `results/multi_condition_anova.csv`
(F023), `results/multi_condition_setting_means.csv` (F025),
`results/robustness_multicondition_replicate_slopes.csv` (F028),
`results/core_analysis_summary.json` (F041) key `S6_multi_condition`,
`results/robustness_summary.json` (F042) key `R1_multicondition`.

---

## Cross-cutting limitations

1. **Neutralized identities.** Biological identities are replaced by neutral roles
   (`compartment_N`, `assay_N`, `intervention_setting_N`, `response_class_X`). Every conclusion is
   therefore phrased over these roles; no biological interpretation of *which* compartment,
   compound or cell type is intended, and none can be made from this dataset.
2. **No mechanism.** As `DATA_GUIDE.md` warns, compartment and intervention associations do not by
   themselves identify a unique cellular mechanism. This report contains no mechanistic claim.
3. **Small n at the assay level.** Three assays is the entire population of independent
   experimental units for cross-assay generalisation. Both ordering analyses (Findings 2 and 5) are
   powered by 3 units, which is why both are reported with explicit fragility analyses.
4. **Compound is not dose.** `compound_setting_id` is treated as an ordinal setting index only.
   Whether settings 1–4 are a dilution series is unknown, so "dose response" language is descriptive.
5. **Marginal results are flagged as marginal.** Finding 3's gap (p = 0.0495) and Finding 5's W
   (p = 0.0218) are both near the significance boundary and should not be over-read. Non-parametric
   and exact tests are reported alongside every parametric one.
6. **χ² vs exact disagreement.** For Kendall's W the χ² approximation is anti-conservative for the
   compartment ordering and conservative for the intervention ordering. Both are reported
   throughout; neither alone is treated as the answer.
7. **Time units are unitless.** `time_units` carries no unit label, so the trend is reported as
   percent-per-time-unit and no biological timescale is claimed.
8. **All tests are two-sided** and at α = 0.05 unless a one-sided alternative is explicitly named
   (the pooled sign test in Finding 1).

---

## Methods

### Data and units
Four TSV files read directly with pandas; no resampling, imputation, or row deletion. All four match
the SHA-256 digests in `data_manifest.json` (`results/run_manifest.json`, F-written by script
F009). The analysis unit is the biological replicate nested in its cell, per `DATA_GUIDE.md`;
repeated endpoints from a shared assay are never treated as independent experiments.

### Estimators
- **Primary fold change:** 2^mean(log₂(perturbation/baseline)) over the replicates in a cell
  (geometric mean of ratios).
- **Confidence intervals:** t-based on the stated unit; `scipy.stats.t.ppf(0.975, df)`.
- **Paired contrasts:** `scipy.stats.ttest_rel`; non-parametric confirmation by binomial sign test;
  TOST for equivalence against explicit margins.
- **Mixed models:** `statsmodels.formula.api.mixedlm` with explicit random intercepts
  (`assay_id` for the compartment endpoint model; `replicate_id` for the time and multi-condition
  models). REML fit; convergence checked and reported.
- **Ordering concordance:** Kendall's W from the definition
  `W = 12S / (k²(n³ − n))` with `S = Σ(rank_sum − mean rank_sum)²`, plus Friedman's test and, via
  the sensitivity module, exact enumeration of the permutation null.
- **Cross-validation:** leave-one-complete-assay-out (the assay is the held-out unit, never a
  random replicate row, per `DATA_GUIDE.md`), with three competing prediction models.
- **Minimum detectable effect:** closed form `MDE = (t_crit + t_β) · SD / √n` verified against a
  noncentral-t inversion solving for 80% power.

### Validation design
Every test that carries a finding is run with the assay or condition group as the complete held-out
unit; leave-one-assay-out and leave-one-replicate-out folds are reported for Findings 1, 2, 4 and 5.
Alternative fold-change definitions, alternative contrasts (ratio vs difference), unpaired vs paired
tests, precision weighting, outlier trimming, and tie/threshold sensitivity are reported alongside
the primary results, including where they weaken or qualify them.

---

## Reproduction

Full instructions, expected console output, and the complete execution order are in
[`reproduce.md`](reproduce.md), which is the companion reproduction document for this report.

**What was executed.** Nine Python scripts under `output/analysis/` were run to completion on
Python 3.11 with numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, statsmodels 0.15.0 and matplotlib 3.11.2.
Their real console output is preserved verbatim in `output/logs/` (F053–F056 for scripts 01, 02, 03
and 06; scripts 04, 05 and 07 report to stdout and are captured in the same tree). No script
imports from the network and no external dataset was consulted.

**Quick re-run.**
```bash
cd /data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project
python3 output/analysis/01_compartment_intervention.py
python3 output/analysis/02_time_resolved.py
python3 output/analysis/03_robustness.py
python3 output/analysis/04_figures.py
python3 output/analysis/05_verification_compartment.py
python3 output/analysis/05_verification_intervention.py
python3 output/analysis/05_verification_time.py
python3 output/analysis/06_sensitivity.py
python3 output/analysis/07_registry_and_delivery.py
python3 validate_output.py output
```

**Fidelity notes.** Scripts 04 and 06 were authored and executed by delegated sub-agents and their
artifacts were then re-run and audited independently: `04_figures.py` regenerates byte-identical
PNGs, and its captions were corrected after the audit identified three mislabelled statements (the
naming of the assay pair behind the 173× separation, "three orders of magnitude" for a 2.2-order
spread, and a peak-setting claim that the data contradict). `06_sensitivity.py` regenerates a
byte-identical JSON; one wrong number in its explanatory text (29 assays where the computed value is
27) was corrected in place. Two earlier analysis bugs of mine were found and fixed by re-running:
a variable named `C` shadowing patsy's `C()` in the ANOVA call, and a transposed
`friedmanchisquare` argument in script 03 (the corrected manual χ² is 4.667 for the count endpoint
and 0.667 for the fraction endpoint). A third issue — an incorrect Kendall χ² formula that inflated
p to 3×10⁻¹⁶⁶ — was caught in review and replaced with `χ² = k(n−1)W`. No `StructuredOutput` tool
was available to the delegated sub-agents; their findings were returned as text and their artifacts
verified directly on disk.

**All external sources used:** none. No literature, database, or other external information was
consulted for this report, so no citation or first-public-date record is required under the `T0`
boundary.

---

## Registered finding index and evidence traceability

| Finding | Primary analysis | Primary result evidence | Figure | Independent verification |
|---|---|---|---|---|
| 1 | `analysis/01_compartment_intervention.py` (F001) | `results/compartment_cell_effects.csv` (F011), `results/core_analysis_summary.json` (F041), `results/robustness_summary.json` (F042) | `figures/fig1_compartment_fold_change.png` (F049) | `results/verification_compartment.json` (F045) 14/14 |
| 2 | `analysis/01_compartment_intervention.py` (F001), `analysis/03_robustness.py` (F003), `analysis/06_sensitivity.py` (F008) | `results/compartment_ordering_reproducibility.csv` (F012), `results/loocv_rank_accuracy.csv` (F022), `results/sensitivity_report.json` (F043) | `figures/fig2_baseline_vs_foldchange.png` (F050) | `results/verification_compartment.json` (F045) |
| 3 | `analysis/01_compartment_intervention.py` (F001), `analysis/03_robustness.py` (F003) | `results/endpoint_type_comparison.csv` (F016), `results/endpoint_ordering_agreement.csv` (F014), `results/robustness_summary.json` (F042) | `figures/fig1_compartment_fold_change.png` (F049) | `results/verification_compartment.json` (F045) checks V1g/V1h |
| 4 | `analysis/02_time_resolved.py` (F002), `analysis/06_sensitivity.py` (F008) | `results/time_resolved_contrasts.csv` (F035), `results/time_resolved_summary.json` (F044), `results/sensitivity_report.json` (F043) | `figures/fig4_time_resolved.png` (F052) | `results/verification_time.json` (F048) 5/5 |
| 5 | `analysis/01_compartment_intervention.py` (F001), `analysis/06_sensitivity.py` (F008) | `results/intervention_assay_by_setting_matrix.csv` (F017), `results/intervention_setting_rank_spread.csv` (F020), `results/sensitivity_report.json` (F043) | `figures/fig3_intervention_profiles.png` (F051) | `results/verification_intervention.json` (F046) 9/9 |

Identifiers `F0xx` resolve through the native artifact registry at
`results/file_registry.json`, which records each artifact's path, byte size and SHA-256. The
consolidated independent-verification record (28 checks, all agreeing) is at
`results/verification_report.json` (F047).

**Where this analysis failed or was qualified.** (i) The χ² approximation for Kendall's W was found
to be anti-conservative for the compartment ordering (p = 0.097 vs exact 0.194) and conservative for
the intervention ordering (p = 0.022 vs exact 0.0009); no single p-value is treated as definitive.
(ii) The assay-level leave-one-out exercise failed to predict held-out assays better than a constant
— reported as the primary evidence for Finding 2 rather than suppressed. (iii) A mixed model in
script 01 initially failed with a `PatsyError` because a string column name was used where the
design matrix needed a computed column; the failure is preserved in the console log and the model
was refit successfully. (iv) Three delegated sub-agents were unable to call `StructuredOutput` and
one verification agent failed twice on API errors; the work was re-split into smaller units that
succeeded, and every artifact was verified directly on disk rather than taken from a self-report.
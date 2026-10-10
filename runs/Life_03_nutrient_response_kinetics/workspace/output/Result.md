# Nutrient-Response Kinetics Across System States and Tissues

Task ID: `Life_03_nutrient_response_kinetics`. This report is the authoritative
statement of findings for a frozen bundled dataset of four TSV panels compared
across two system states (`background_state` vs `perturbed_state`). All numbers
below are direct observations from the computed `output/results.json` and
`output/robustness.json`; non-significant results are reported as evidence-backed
null/unresolved, not as proof of no effect. Directional structure is internally
consistent but underpowered (small n).

## Methods

The analysis is implemented in `output/analysis.py` (core kinetics, group means,
correlations, and two-sample tests) and the robustness checks in
`output/robustness.py` (checks a-f). `analysis.py` loads four TSV files with
`pandas` (`sep="\t"`); no missing values in any file (verified at load time):

- `nutrient_response_timecourse.tsv` (40 rows): 2 states × 5 time points
  (0, 5, 10, 20, 30 min) × 4 replicates.
- `circulating_input_measurement.tsv` (16 rows): 2 input roles (`pre_input`,
  `nutrient_input`) × 2 states.
- `tissue_uptake_measurement.tsv` (12 rows): 2 tissues × 2 states × 3 replicates.
- `baseline_system_phenotype.tsv` (76 rows): 4 anatomical measures × 2 states
  (background: 12 replicates; perturbed: 7 replicates).

**Analysis unit:** one biological replicate within a system state / tissue /
input condition / time course; repeated measures sharing a state are nested.
State comparison is `perturbed_state` vs `background_state`.

**Statistical methods** (as implemented in `analysis.py`, section "statistical_tests"):
independent two-sample Student t-test (`scipy.stats.ttest_ind`, `equal_var=True`);
Mann-Whitney U (`scipy.stats.mannwhitneyu`, `alternative="two-sided"`); and
Cohen's d = (mean_perturbed − mean_background)/pooled SD. Significance threshold
α = 0.05 (two-sided). Pearson correlations use `scipy.stats.pearsonr`; robustness
check (f) adds per-measure Kruskal-Wallis (`scipy.stats.kruskal`); robustness
check (e) uses a paired one-sample t-test on per-replicate tissue differences
(`scipy.stats.ttest_1samp`).

## Reproduction

1. Ensure the project root is `/data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project`
   and that `data/` contains the four TSV files and `output/` contains
   `analysis.py` and `robustness.py`.
2. Verify the Python environment (Python 3.11+ with `pandas`, `numpy`, `scipy`
   available; the frozen run used pandas 3.0.5, numpy 2.4.6, scipy 1.17.1):
   `python3 -c "import pandas, numpy, scipy; print(scipy.__version__)"`
3. Run the core analysis: `python3 output/analysis.py` → writes
   `output/results.json`.
4. Run the robustness checks: `python3 output/robustness.py` → writes
   `output/robustness.json`.
5. (Optional) Validate the output contract:
   `python3 validate_output.py output`.

See `output/reproduce.md` for prerequisites and expected outputs.

## Finding 1: The perturbed state has a lower baseline anatomical phenotype, strongest for `anatomical_measure_3` and `anatomical_measure_4`

- **Object and scope.** Baseline `measurement_value` per anatomical measure
  (`anatomical_measure_1..4`), comparing `perturbed_state` vs `background_state`.
  The analysis unit is one biological replicate within a state (7 replicates in
  the perturbed state, 12 in the background state).
- **Relationship, direction, boundary.** The perturbed state has significantly
  *lower* values for `anatomical_measure_2`, `anatomical_measure_3`, and
  `anatomical_measure_4` (all significant at α=0.05). `anatomical_measure_1`
  is also lower in direction but is *not* significant — the phenotype effect is
  measure-specific (boundary), not uniform across all measures.
- **Core quantitative result** (analysis unit = replicate; values from
  `results.json` → `baseline_phenotype.state_difference_by_measure` and
  `statistical_tests.tests.baseline_anatomical_measure_*`):
  - `anatomical_measure_3`: perturbed 119.878571 vs background 172.715000;
    diff −52.836429; t-stat −6.417628, t-p = 6.37e-06; Mann-Whitney U = 0.0,
    p = 3.97e-05; Cohen's d = −3.052.
  - `anatomical_measure_4`: perturbed 40.682857 vs background 52.584167;
    diff −11.901310; t-stat −4.279268, t-p = 5.07e-04; Mann-Whitney U = 7.0,
    p = 1.79e-03; Cohen's d = −2.035.
  - `anatomical_measure_2`: perturbed 8.345714 vs background 9.948333;
    diff −1.602619; t-stat −2.713252, t-p = 0.014761; Mann-Whitney U = 9.0,
    p = 0.005997; Cohen's d = −1.290.
  - `anatomical_measure_1`: perturbed 7.432857 vs background 8.039167;
    diff −0.606310; t-stat −0.799338, t-p = 0.435121; Mann-Whitney U = 38.0,
    p = 0.773240; Cohen's d = −0.380 (not significant).
- **Supporting analysis and result evidence.** Computed in `analysis.py`
  (section "(f) Statistical tests (two-sample)", loops over
  `sorted(bp["anatomical_measure_id"].unique())` and calls `stat_pair`);
  stored in `results.json` (`baseline_phenotype` and
  `statistical_tests.methods`). The nonparametric cross-check is implemented in
  `robustness.py` (check f, `f_baseline_kruskalwallis`) and stored in
  `robustness.json`.
- **Robustness result.** Per `robustness.json` → `f_baseline_kruskalwallis`,
  Kruskal-Wallis agrees with the t-test for *all* four measures
  (`all_agree = 1`). Measures significant by both tests: `anatomical_measure_2`
  (t-p 0.014761, KW-p 0.005267), `anatomical_measure_3` (t-p 6.37e-06,
  KW-p 3.86e-04), `anatomical_measure_4` (t-p 5.07e-04, KW-p 0.003096);
  non-significant by both: `anatomical_measure_1` (t-p 0.435121, KW-p 0.735317).
  The Mann-Whitney U also agrees with the t-test on all three significant
  measures, supporting robustness to non-normality.
- **Counterevidence, limitation, alternative explanation.** Counter-evidence:
  `anatomical_measure_1` shares the (negative) direction but is non-significant
  with a small effect (d = −0.38), so the reduction is measure-specific rather
  than a uniform phenotype shift. Limitation: small and unequal n (7 vs 12
  replicates), replicates are not shared across states, and the two-sample
  t-test assumes independence of the replicate observations. Alternative
  explanation: the measured reduction could partly reflect the perturbation
  protocol rather than a trait inherent to the state; cross-panel association
  does not by itself establish molecular causality.
- **Executable new observation / independent replication.** Measure a fresh set
  of independent biological replicates of `anatomical_measure_3` and
  `anatomical_measure_4` (and `_2`) in both states, using the same independent
  two-sample t-test and Mann-Whitney U. A null or reversed difference
  (perturbed ≥ background, p ≥ 0.05) would falsify the reduction claim.

## Finding 2: The acute signaling response at peak (20 min) is not significantly different between states, and baseline phenotype does not establish a cross-panel association with peak response

- **Object and scope.** (a) `normalized_signaling_response` at the peak time
  point (20 min in both states), and (b) the Pearson correlation between
  replicate-level baseline phenotype (mean across the 4 anatomical measures) and
  peak response. The analysis unit is one matched biological replicate (4 per
  state, 8 pooled pairs).
- **Relationship, direction, boundary.** No significant state difference at
  peak (positive point estimate but a null). No reliable cross-panel association:
  within-state Pearson r is unstable and non-significant in both states, and the
  pooled correlation reverses sign and is non-significant.
- **Core quantitative result** (analysis unit = replicate; values from
  `results.json` → `time_course_kinetics`, `correlations`, and
  `statistical_tests.tests.time_course_peak_response`):
  - Peak state difference: perturbed 12.308190 vs background 10.849082; diff
    +1.459108; t-stat 0.658057, t-p = 0.534919; Mann-Whitney U = 10.0,
    p = 0.685714; Cohen's d = 0.465 (not significant).
  - Correlation: background r = 0.270553 (p = 0.729447); perturbed r =
    0.602767 (p = 0.397233); pooled r = −0.099919 (p = 0.813894, n = 8).
    All non-significant.
- **Supporting analysis and result evidence.** Computed in `analysis.py`
  (sections "(a) Time-course kinetics", "(e) Correlations", "(f) Statistical
  tests"); stored in `results.json` (`time_course_kinetics`, `correlations`,
  `statistical_tests`). The leave-one-out, alternative-test, and threshold
  sensitivity robustness checks are implemented in `robustness.py` (checks a, b,
  c) and stored in `robustness.json`
  (`a_loo_peak_response`, `b_alternative_test_peak`, `c_threshold_sensitivity`).
- **Robustness result.** Per `robustness.json` → `a_loo_peak_response`, the
  matched leave-one-replicate-out state difference is non-significant at every
  drop (t-p 0.8426, 0.9507, 0.4197, 0.1932; `n_leave_outs_significant_t = 0`),
  but the *sign* is not stable: dropping `replicate_1` flips the difference to
  negative (−0.4620) while the other three drops are positive
  (`sign_changes = 1`), with the full-data difference positive (+1.4591).
  Check b (`b_alternative_test_peak`) confirms the t-test and Mann-Whitney U
  give the same non-significant conclusion (t-p 0.5349, MW-p 0.6857,
  `significance_differs = 0`). Check c (`c_threshold_sensitivity`) shows the
  perturbed-minus-background difference is positive at all three thresholds
  (t=0: +0.3749; peak: +1.4591; t=30: +0.9757; `sign_changes = 0`).
- **Counterevidence, limitation, alternative explanation.** Counter-evidence:
  the perturbed-state within-state correlation (r = 0.60) is directionally
  positive but underpowered (n = 4) to reach significance; the pooled
  correlation reverses sign (negative) because the two states' within-state
  correlations are weighted oppositely, so a single cross-panel association is
  not established and is unstable under leave-one-out. Limitation: only 4
  matched replicates per state, so power to detect either a state difference or
  a correlation is very low. Alternative explanation: the null is an
  evidence-backed unresolved result, not proof of no relationship; the peak
  response may depend on covariates (e.g., circulating input) not modeled here.
- **Executable new observation / independent replication.** Recruit additional
  independent matched biological replicates so each state has ≥ 8–10 matched
  pairs. A sustained significant correlation (|r| ≥ 0.5, p < 0.05) between
  baseline phenotype and peak response would validate the association, and a
  significant state difference at peak (p < 0.05 across leave-one-out) would
  validate the state effect; a persistent null across replicates would support
  the evidence-backed null.

## Finding 3: Tissue uptake is tissue-structured (tissue_1 > tissue_2) but its state effect is not statistically supported; circulating input rises after `nutrient_input` in both states without a significant state difference

- **Object and scope.** (a) `uptake_measurement_au` by tissue (`tissue_1`,
  `tissue_2`) and state; (b) `measured_abundance_au` by input role
  (`pre_input`, `nutrient_input`) and state. The analysis unit is one biological
  replicate (3 per tissue–state group; 3–5 per input-role–state group).
- **Relationship, direction, boundary.** Tissue_1 uptake exceeds tissue_2 in
  both states (robust, paired-test significant). No significant *state* difference
  in tissue uptake or circulating input. Circulating input is higher after
  `nutrient_input` than `pre_input` in both states, but the between-state
  differences are non-significant (evidence-backed null).
- **Core quantitative result** (analysis unit = replicate; values from
  `results.json` → `tissue_uptake`, `circulating_input`, and
  `statistical_tests.tests.tissue_uptake_*` / `circulating_input_*`):
  - Tissue uptake state difference: `tissue_1` perturbed 1092.3755 vs background
    1099.2288 (diff −6.8534; t-p = 0.9626, MWU-p = 1.0, d = −0.041);
    `tissue_2` perturbed 296.0242 vs background 406.7789 (diff −110.7547;
    t-p = 0.2395, MWU-p = 0.40, d = −1.127). Neither is significant. Full-data
    overall state difference −58.8041 (t-p = 0.8163), dominated by tissue_2.
  - Tissue structure: tissue_1 − tissue_2 = 692.4499 in background (paired
    t-p = 0.0038) and 796.3513 in perturbed (paired t-p = 0.0074), both
    significant; overall 744.4006.
  - Circulating input: `pre_input` perturbed 1071.835 vs background 1082.647
    (diff −10.812; t-p = 0.9749, MWU-p = 0.6286, d = −0.025); `nutrient_input`
    perturbed 10857.932 vs background 8164.693 (diff +2693.240; t-p = 0.3686,
    MWU-p = 0.5556, d = 0.645). Neither state comparison is significant.
- **Supporting analysis and result evidence.** Computed in `analysis.py`
  (sections "(b) Circulating input", "(c) Tissue uptake", "(f) Statistical
  tests"); stored in `results.json` (`circulating_input`, `tissue_uptake`,
  `statistical_tests.methods`). The tissue holdout and state holdout checks are
  implemented in `robustness.py` (checks d and e) and stored in
  `robustness.json` (`d_tissue_holdout`, `e_state_holdout`).
- **Robustness result.** Per `robustness.json` → `e_state_holdout`, the tissue
  ordering tissue_1 > tissue_2 holds in both states and overall
  (`consistent_tissue_1_gt_tissue_2 = 1`), with both state-specific paired
  differences significant (background p = 0.0038; perturbed p = 0.0074) —
  tissue structure is robust to state holdout. Per `robustness.json` →
  `d_tissue_holdout`, the per-tissue state differences are unchanged by holding
  out the other tissue (by construction): tissue_2 only diff −110.7547 (t-p =
  0.2395, non-significant); tissue_1 only diff −6.8534 (t-p = 0.9626,
  non-significant); full-data overall diff −58.8041 (t-p = 0.8163). The
  directional structure is therefore internally consistent but the state-level
  differences are not statistically supported.
- **Counterevidence, limitation, alternative explanation.** Limitation: very
  small n (3 per tissue–state group; 3–5 per input-role–state group), so power
  is low. The tissue_2 state drop (d = −1.13) is suggestive but non-significant,
  and the `nutrient_input` state difference (d = 0.645) is non-significant,
  so both are evidence-backed null/unresolved rather than claims of a state
  effect. Alternative explanation: the observed tissue and input patterns may
  reflect assay sensitivity or sampling design rather than a true state-driven
  change.
- **Executable new observation / independent replication.** Recruit additional
  independent biological replicates for tissue uptake (both tissues, both states)
  and circulating input (both roles, both states), applying the same
  two-sample t-test and Mann-Whitney U. A significant state difference in tissue
  uptake or circulating input (p < 0.05) would validate the respective state
  claim, and a reversal of the tissue_1 > tissue_2 ordering would falsify the
  tissue-structure claim.

## Limitations

- Small sample sizes throughout (4 matched time-course replicates per state; 3
  per tissue–state group; 3–5 per input-role–state group; 7 vs 12 baseline
  replicates), limiting statistical power and inflating leave-one-out instability.
- Two-sample t-tests assume independence and, for normality-sensitive conclusions,
  near-normality; Mann-Whitney U and Kruskal-Wallis are used as nonparametric
  cross-checks, and they agree with the t-test on all tested comparisons.
- Baseline phenotypic replicates differ across states (replicate_1..7 vs
  replicate_1..12), so cross-state association is not a perfectly matched design.
- Directional patterns in the null findings (positive peak difference, tissue_1
  > tissue_2, nutrient_input > pre_input) are internally consistent but are
  underpowered; they are reported as evidence-backed null/unresolved, not as
  proof of no effect, and no causal mechanism is inferred.

## Validation

The output contract is checked by the validator at the project root. Command:

```bash
python3 validate_output.py output
```

Actual result (validator semantic receipt, key fields):

```json
{
  "schema": "truthinsightbench-semantic-output-validation",
  "passed": true,
  "errors": [],
  "warnings": [],
  "registered_findings": {
    "count": 3,
    "numbers": [1, 2, 3],
    "traceability": [
      {"finding_number": 1, "analysis_sources": ["analysis.py", "robustness.py"], "result_evidence": ["results.json", "robustness.json"], "passed": true},
      {"finding_number": 2, "analysis_sources": ["analysis.py", "robustness.py"], "result_evidence": ["results.json", "robustness.json"], "passed": true},
      {"finding_number": 3, "analysis_sources": ["analysis.py", "robustness.py"], "result_evidence": ["results.json", "robustness.json"], "passed": true}
    ]
  },
  "reproduction": {"mode": "companion_markdown", "document": {"path": "reproduce.md"}},
  "analysis_sources": ["analysis.py", "robustness.py"],
  "result_artifacts": ["environment_summary.json", "results.json", "robustness.json"]
}
```

The validator confirms `passed: true` with an empty `errors` list, three
consecutive findings (1, 2, 3) each resolving an analysis source
(`analysis.py`, and `robustness.py` where applicable) and machine-readable
result evidence (`results.json`, `robustness.json`), and a companion
reproduction document (`reproduce.md`). The full receipt (including per-file
sha256/byte sizes) is printed by the command above.

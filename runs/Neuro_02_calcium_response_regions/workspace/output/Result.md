# Region-, Condition-, and Batch-Dependent Calcium Response Structure

## Scope, data, and analysis unit

This analysis uses the frozen per-cell calcium-response summaries in
`data/cellular_calcium_response_summaries.tsv` (819 rows, 12 columns) as the sole
direct evidence. The independent analysis unit is **one cell** (`observation_id`),
nested within a recording slice, recording batch, and biological source; cells from
the same slice are not independent animals. Measures (per-cell means) are
`mean_amplitude_dff0`, `mean_time_to_peak_s`, `mean_peak_width_s`,
`mean_peak_prominence_dff0`, `mean_peak_frequency_hz`, and `mean_auc_au`.
Factors are `condition` (baseline | neural_activity_blocked | receptor_stimulated),
`anatomical_region_id` (region_1–4), `animal_age_days`, and `recording_batch_id`
(9 batches).

Two structural confounds dominate this dataset and shape every finding:

1. **Recording batch is 1:1 with age.** All 9 batches contain a single age
   (`batch_age.json`: `all_one_to_one = 9`), so batch and age are perfectly
   confounded and cannot be separated in this data.
2. **Region coverage is uneven across conditions.** `region_3` (n=189) was
   recorded only under `receptor_stimulated`; `region_4` (n=50) only under
   baseline and neural_activity_blocked (no receptor_stimulated); `region_1` and
   `region_2` cover all three conditions. This makes region–condition associations
   partially unidentifiable (see Finding 2 and Finding 4).

All models are OLS (statsmodels) with the treatment coding shown in the
`formula` fields; "controlled for" means the named factors were included as
covariates. Every finding below cites the executable source
`output/analysis.py` and a machine-readable result file in `output/`.

## Finding 1: Receptor stimulation produces a large, robust increase in AUC, peak width, and amplitude, and a decrease in time-to-peak, after controlling for region and batch.

- **Object and scope.** Per-cell calcium-event measures across the full dataset
  (n=819) and the balanced region_1/2 subset (n=580), comparing
  `receptor_stimulated` against `baseline`, controlling for `region` and `batch`.
- **Relationship/direction.** Receptor stimulation increases `mean_auc_au`,
  `mean_peak_width_s`, and `mean_amplitude_dff0`; it decreases
  `mean_time_to_peak_s`. It has no reliable effect on `mean_peak_prominence_dff0`
  or `mean_peak_frequency_hz`. `neural_activity_blocked` is essentially null on
  most measures.
- **Core quantitative result (analysis unit = cell).** Full-data OLS
  `condition + region + batch` (`output/cond_region_batch.json`): receptor vs
  baseline — AUC +85.95 (p=3.19e-35), width +30.29 (p=5.94e-32), amplitude +0.33
  (p=3.06e-13), time-to-peak −38.40 s (p=1.16e-10). Balanced region_1/2 subset
  (`output/clean_condition.json`): AUC +87.93 (p=1.13e-35), width +30.85
  (p=1.45e-27), amplitude +0.34 (p=1.18e-13), time-to-peak −29.29 s
  (p=1.32e-05). Direction and significance are identical across the two
  specifications.
- **Supporting analysis and result evidence.** `output/analysis.py` (model 1 and
  model 3); `output/cond_region_batch.json`, `output/clean_condition.json`,
  `output/raw_condition_means.json`, `output/median_clean.json`;
  `output/fig_condition_effect.png`.
- **Robustness.** (a) Leave-one-batch-out: with each of the 9 batches removed,
  the receptor AUC, width, and amplitude contrasts remain significant in all 9
  folds (`output/lbo_condition.json`: `all_significant = true`). (b) Outlier
  treatment (trim 2% tails): AUC +86.72 (p≈0) and width +30.58 (p≈0)
  (`output/outlier_clean.json`), unchanged in magnitude. (c) Median instead of
  mean in the region_1/2 subset preserves the same sign on every measure
  (`output/median_clean.json`). (d) **Permutation null test**: permuting
  condition labels within (region, batch, slice) strata and refitting the OLS
  model yields empirical two-sided p≈0 for AUC (obs +87.93), width (+30.85),
  amplitude (+0.34), and time-to-peak (−29.29), while prominence (p=0.112) and
  frequency (p=0.584) are not significant — matching the parametric model
  (`output/permutation_condition.json`). (e) **Stratified nonparametric bootstrap**
  (resample cells with replacement within region, batch, slice strata; refit each
  time): the bootstrap is centered on the observed effect and its 95% percentile
  CIs exclude 0 for AUC [80.66, 94.39], width [27.95, 34.21], amplitude
  [0.251, 0.438], and time-to-peak [−47.58, −10.85]; prominence [−0.012,
  0.162] and frequency [−0.0005, 0.0023] include 0
  (`output/bootstrap_condition.json`).
- **Counterevidence, limitation, alternative explanations.** Prominence and
  frequency are null (borderline prominence p≈0.06); neural_activity_blocked is
  null, so the effect is specific to receptor stimulation, not to any perturbation.
  Because batch=age, the controlled "batch" term cannot itself be interpreted
  as an independent biological factor (see Finding 3). An alternative explanation
  is that receptor stimulation changes the number/size of events counted per cell
  rather than a single-event property; the data are per-cell *means*, so this is
  not directly testable here.
- **Falsifying/replicating observation.** Re-record a new set of batches covering
  receptor-stimulated and baseline in the same regions; a true receptor effect
  should reproduce with the same sign and magnitude after controlling for the
  new batch/age structure. If the effect collapses when batch and age are fixed,
  or if it is confined to region_1/2, the finding would be revised.

## Finding 2: Region identity is strongly associated with morphological measures (width, prominence, frequency, amplitude, time-to-peak), but the region–AUC association is not robust to the condition-coverage confound.

- **Object and scope.** Per-cell measures across regions, controlling for
  `condition` and `batch`; full data (n=819) and the balanced region_1/2 subset.
- **Relationship/direction.** Region_2 (vs region_1) increases width, prominence,
  amplitude, and decreases time-to-peak and frequency. Region_3 increases width
  and prominence. Region_4 increases width. For AUC, only region_3 is elevated
  relative to region_1.
- **Core quantitative result (analysis unit = cell).** Full-data OLS
  `region + condition + batch` (`output/region_condition_batch.json`), ref
  region_1: width region_2 +10.46 (p=4.80e-10), region_3 +12.39 (p=2.15e-10),
  region_4 +14.09 (p=4.57e-05); prominence region_2 +0.321 (p=1.78e-38);
  frequency region_2 −0.0040 (p=1.60e-12); amplitude region_2 +0.302
  (p=7.06e-23), region_3 +0.338 (p=2.17e-21), region_4 +0.097 (p=0.117,
  not significant); AUC region_3 +20.38 (p=8.74e-05) but region_2 −0.23
  (p=0.958) and region_4 −7.38 (p=0.424, not significant).
- **Supporting analysis and result evidence.** `output/analysis.py` (model 2 and
  model 8); `output/region_condition_batch.json`, `output/region_clean.json`,
  `output/coverage.json`.
- **Robustness.** In the balanced region_1/2 subset (the only subset where both
  regions share all three conditions), `output/region_clean.json` shows region_2
  vs region_1: width +10.72 (p≈0), prominence +0.322 (p≈0), amplitude +0.312
  (p≈0), time-to-peak −25.17 (p≈0), frequency −0.0038 (p≈0), **but AUC +1.40
  (p=0.738, not significant)**. Thus the morphological region effects survive the
  balanced subset, while the region–AUC association does not. (a) **Permutation
  null test**: permuting region_1/region_2 labels within (batch, condition)
  strata and refitting (partial-out M1, fixed condition+batch) keeps amplitude
  (p=0.0), time-to-peak (p=0.0), width (p=0.0), prominence (p=0.0), and
  frequency (p=0.0) significant, but AUC is not (p=0.797, CI [−4.45, 11.94]
  includes 0) — confirming the AUC region effect is a coverage artifact
  (`output/region_effect_permutation.json`). (b) **Leave-one-batch-out**: with
  each of the 9 batches removed, the region_2-vs-region_1 contrast keeps the same
  sign and remains significant for amplitude, time-to-peak, width, prominence,
  and frequency (9/9), but AUC is not robust (1/9 significant, sign changes)
  (`output/lbo_region.json`). (c) **Confounding quantification**: within
  receptor-stimulated cells, region_2-vs-region_1 AUC is −0.48 (essentially
  zero) whereas the full-model region_3-vs-region_1 AUC contrast is +20.38,
  demonstrating the full-data "region_3 AUC effect" is driven by region_3 being
  sampled only under receptor stimulation; 9/9 batches are single-age and the
  `AUC ~ age + batch + condition + region` model is rank-deficient (rank 14 of
  15), and batch terms show high VIF (max ≈31.4)
  (`output/confounding_quantification.json`). The coverage matrix is shown in
  `output/fig_region_coverage.png`.
- **Counterevidence, limitation, alternative explanations.** Region_4 amplitude is
  not significant (p=0.117); region_3 and region_4 AUC are not significant.
  The limitation is the uneven condition coverage (Finding 2 boundary), which
  makes the AUC region main effect unidentifiable. An alternative explanation is
  that the apparent region–AUC difference is entirely a condition effect (Finding 1)
  filtered through region coverage.
- **Falsifying/replicating observation.** Collect cells in region_3 and region_4
  under baseline and neural_activity_blocked conditions. If region_3 AUC rises
  relative to region_1 only when receptor-stimulated, the region–AUC association
  is conditional on condition; if AUC differences persist across conditions,
  region is an independent AUC driver.

## Finding 3: Recording batch (equivalent to age in this data) explains a large fraction of AUC variance independent of condition and region — a batch-driven structure.

- **Object and scope.** Per-cell `mean_auc_au` (and morphological measures),
  across the 9 recording batches, controlling for `condition` and `region`.
- **Relationship/direction.** Batch is a strong predictor of AUC; 8 of 9 batch
  contrasts are significant, and most batches have lower AUC than the reference
  batch_02.
- **Core quantitative result (analysis unit = cell).** OLS
  `batch + condition + region` (`output/batch_ctrl.json`), AUC: F-test p=1.03e-100,
  R²=0.467. Batch contrasts vs batch_02: batch_03 −77.17 (p=1.91e-25),
  batch_06 −88.30 (p=6.58e-20), batch_08 −60.04 (p=6.06e-17), batch_05 −60.74
  (p=3.30e-09), batch_04 −44.69 (p=2.40e-11), batch_09 −42.38 (p=3.02e-10),
  batch_07 −36.69 (p=5.29e-08); batch_01 +39.52 (p=0.138, not significant).
- **Supporting analysis and result evidence.** `output/analysis.py` (model 4);
  `output/batch_ctrl.json`, `output/batch_age.json`, `output/batch_means.json`,
  `output/fig_batch_age.png`.
- **Robustness.** The batch effect is structurally confirmed by the mapping in
  `output/batch_age.json`: every batch has exactly one age (`all_one_to_one = 9`),
  so the batch contrast is also an age contrast. The batch effect on AUC is the
  only measure where this batch/age structure is large and consistent (R²≈0.47).
  `output/confounding_quantification.json` confirms the two confounds quantitatively:
  (i) 9/9 batches are single-age and the `AUC ~ age + batch + condition + region`
  design is rank-deficient (rank 14 of 15 parameters), so age and batch cannot be
  separated; (ii) batch terms show high multicollinearity (VIF up to ≈31.4).
- **Counterevidence, limitation, alternative explanations.** batch_01 is not
  significant (p=0.138), and the effect is "lower than batch_02" rather than a
  monotonic function of age. Crucially, because batch=age (1:1), this is a **batch/age
  confound, not a separable biological age effect** (see Finding 4); the data
  cannot tell whether the variance is a technical batch artifact or a true age
  effect.
- **Falsifying/replicating observation.** Record new batches spanning multiple
  ages within the same technical batch, or the same age across multiple technical
  batches. If AUC variance tracks the technical batch but not age, it is a
  batch artifact; if it tracks age regardless of batch, it is an age effect.

## Finding 4: The age trend is modest and measure-specific, and the region × condition interaction is significant for peak width only.

- **Object and scope.** (a) Linear age trend across all cells, controlling for
  `condition` and `region`. (b) Region × condition interaction in the balanced
  region_1/2 subset, controlling for `batch`.
- **Relationship/direction.** (a) Significant positive linear age trends for
  AUC, amplitude, and prominence only; time-to-peak, width, and frequency are
  null or near-null. (b) The receptor-driven width increase is larger in region_2
  than region_1; the receptor AUC increase is region-independent.
- **Core quantitative result (analysis unit = cell).** (a) OLS `age + condition +
  region` (`output/age_trend.json`): AUC slope +0.427 d⁻¹ (p=0.013), amplitude
  +0.0072 d⁻¹ (p=2.72e-08), prominence +0.0074 d⁻¹ (p=2.47e-12); time-to-peak
  +0.199 d⁻¹ (p=0.165), width −0.029 d⁻¹ (p=0.652), frequency +0.000005
  d⁻¹ (p=0.805). (b) OLS `region*condition + batch` in region_1/2
  (`output/interaction_clean.json`): width region_2:receptor p=5.92e-04
  (coef +18.98); AUC region_2:receptor p=0.195 (coef −17.78, not significant);
  amplitude region_2:receptor p=0.056 (borderline, coef +0.176).
- **Supporting analysis and result evidence.** `output/analysis.py` (model 5 and
  model 9); `output/age_trend.json`, `output/interaction_clean.json`,
  `output/batch_age.json`.
- **Robustness.** The age model is exactly equivalent to the batch model because
  batch=age (1:1 mapping, `output/batch_age.json`); the measure-specific pattern
  (AUC/amplitude/prominence significant, width/frequency null) is the same
  signal as the batch structure in Finding 3. The width interaction is the only
  interaction that reaches significance and it is specific to a single measure.
- **Counterevidence, limitation, alternative explanations.** The amplitude
  interaction is only borderline (p=0.056); the AUC interaction is null. The age
  finding is limited by the perfect batch–age confound, so it is an *age/batch*
  observation rather than an independent age effect. The width interaction may
  reflect residual confounding (region_2 has different batch/age composition)
  rather than a true biological interaction.
- **Falsifying/replicating observation.** (a) For age: obtain cells whose age
  varies within a technical batch; a genuine age effect should persist. (b) For
  interaction: re-record region_1 and region_2 receptor-stimulated data with
  balanced batch/age composition; the width interaction should vanish if it was
  a batch/age artifact and remain if it is a true region-by-condition effect.

## Methods and interpretation notes

- **Direct observations vs. inference.** The signed, significant coefficients in
  Findings 1–3 are direct statistical observations from OLS on the frozen data.
  The statements "batch=age" and "region coverage is uneven" are structural
  observations from `output/batch_age.json` and `output/coverage.json`.
  Mechanistic claims (e.g., *why* receptor stimulation changes event number) are
  **not** supported by this dataset and are only offered as alternative
  explanations.
- **Null effects reported.** neural_activity_blocked is null on most measures
  (Finding 1); region_4 amplitude is null; the AUC region main effect and the
  AUC region×condition interaction are null (Finding 2, Finding 4); width,
  frequency, and time-to-peak age trends are null (Finding 4).
- **Confounds and boundaries.** The perfect batch–age confound and the uneven
  region–condition coverage are the two principal boundaries of this dataset
  (see the Scope section). They are the main reason several candidate
  relationships (independent age effect, independent region–AUC effect) are
  reported as unidentifiable rather than refuted.
- **External sources.** None used. All findings rely solely on the workspace
  data.

## Reproduction

Run the analysis from the project root (Python 3.11; `pandas`, `numpy`,
`statsmodels`):

```
python3 output/analysis.py               # fits all models, writes the JSON result files
python3 output/make_fig.py               # fig_condition_effect.png, fig_batch_age.png
python3 output/fig_region_coverage.py    # fig_region_coverage.png
python3 output/permutation_condition.py  # permutation null test (Finding 1)
python3 output/bootstrap_condition.py    # stratified bootstrap (Finding 1)
python3 output/confounding_quantification.py  # region coverage, batch=age, VIF
python3 output/region_effect_permutation.py   # region-effect permutation (Finding 2)
python3 output/lbo_region.py             # leave-one-batch-out region effect (Finding 2)
```

`output/analysis.py` is the primary executed analysis source; the robustness
sub-steps (permutation, bootstrap, confounding quantification, region-effect
permutation, leave-one-batch-out, and the coverage figure) are the six
independent reference scripts above. During this run the six robustness
sub-steps were delegated in parallel to general-purpose sub-agents via the native
Workflow (run ID `wf_75db2f45-dac`); each sub-agent ran its reference script and
verified the resulting output file. The scripts are deterministic (fixed seed)
so re-running reproduces the same JSON artifacts.

The machine-readable results that the findings cite are:

- `output/cond_region_batch.json`, `output/clean_condition.json` — Finding 1
- `output/region_condition_batch.json`, `output/region_clean.json`,
  `output/coverage.json` — Finding 2
- `output/batch_ctrl.json`, `output/batch_age.json`, `output/batch_means.json`
  — Finding 3
- `output/age_trend.json`, `output/interaction_clean.json` — Finding 4
- `output/lbo_condition.json`, `output/outlier_clean.json`,
  `output/median_clean.json` — Finding 1 robustness checks
- `output/permutation_condition.json`, `output/bootstrap_condition.json` —
  Finding 1 robustness checks
- `output/region_effect_permutation.json`, `output/lbo_region.json`,
  `output/confounding_quantification.json` — Finding 2 robustness / confound
  quantification
- `output/slice_effect.json` — slice-level variance (context)
- `output/raw_condition_means.json` — unadjusted condition means (context)

Figures: `output/fig_condition_effect.png`, `output/fig_batch_age.png`,
`output/fig_region_coverage.png`.

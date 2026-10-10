# Parameter Geometry and Identifiability Boundaries in ODE Fits

**Task ID:** `Math_03_ode_parameter_geometry`  ·  **T0:** `2022-01-01`
**Report language:** English. Raw data values and file names preserved.

This report is the sole authoritative account of the registered findings (4). It was produced
from the three frozen data files under `data/` and from executed analysis code. External
sources were not used for any core claim; no external literature was required for the T0
boundary.

## Data inventory and integrity

Three files constitute the complete task dataset (verified against `data_manifest.json`):

| File | Rows | Fields | SHA-256 status |
|---|---|---|---|
| `data/parameter_trajectories.tsv` | 25,392 | `parameter_index`, `solver_family_id`, `progress_index`, `normalised_distance_like_value` | full match |
| `data/singular_value_spectrum.tsv` | 4,232 | `mode_index`, `log_singular_value` | full match |
| `data/measurement_simulation_pairs.tsv` | 13,000 | `observation_index`, `measurement_like_value`, `simulation_like_value` | full match |

- 4,232 parameters × 2 solver families × 3 progress points (0,1,2) = 25,392 complete
  trajectory rows; no missing values (manifest: no curator-imposed deletion/subsampling/imputation).
- The spectrum has 4,232 modes; `mode_index` and `parameter_index` both span 0–4,231, so the
  join `parameter_index == mode_index` is used (i-th parameter direction ↔ i-th singular value).
- **Integrity correction (this run):** during an early manual hash check a trailing `5c5` was
  accidentally omitted from a copied manifest string, which falsely suggested a mismatch. Re-reading
  `data_manifest.json` programmatically and recomputing hashes confirmed **all three files match
  the manifest's full 64-character SHA-256 exactly**. No data is corrupted or truncated.
  See `output/correction_evidence.md`.

**Analysis unit** (per `DATA_GUIDE.md`): one complete parameter trajectory, solver family, spectrum,
or paired observation series. Progress points within a run are correlated, so whole-trajectory and
whole-family holdouts are used rather than point-level splits.

## Finding 1: The singular-value spectrum is extremely ill-conditioned

- **Object and scope:** the 4,232-mode singular-value spectrum `log_sv` (natural log), indexed
  by `mode_index`.
- **Relationship/direction/boundary:** the spectrum decays over ~36 orders of magnitude
  (`log_sv` from +1.18 to −82.45; `span_ln` 83.63; `span_orders10` ≈ 36.32). The best fit is a
  linear-in-index (exponential) model, not a power law: exponential `R²` ≈ 0.642 vs power-law
  (log-log) `R²` ≈ 0.332 vs raw-scale linear `R²` ≈ 0.225. The tail is steep and non-linear:
  tail slope (last 20% of modes) ≈ −0.060 vs early slope (first 10%) ≈ −0.004. Effective
  dimension is threshold-sensitive (≈9 modes above `log_sv` 0, ≈4,220 above −80).
- **Core quantitative result & unit:** `span_orders10 = 36.32`; best-model `R² = 0.642`;
  tail_slope = −0.060, early_slope = −0.004 (`nonlinear_decay = true`); analysis unit = the
  4,232-mode spectrum.
- **Supporting evidence:** analysis source `analysis.py` (spectrum fits) and `output/figures/spectrum.png`;
  result evidence `results.json` (`spectrum` block), `output/robustness_a.json` (`spectrum_models`
  block), `output/result_tables_spectrum.tsv`.
- **Robustness:** alternative model families (linear-in-index, power-law, exponential) all agree
  that the exponential/linear-in-index fit is best and the tail is steep; `effdim` grows with a
  laxer threshold, confirming a long tail.
- **Counter-evidence / limitation / alternatives:** the R² of the best fit (≈0.64) is not 1, so
  the spectrum is not a pure exponential; a single spectrum is a summary (not per-run), so this
  characterizes the fitted operator, not any single parameter. This is a boundary statement, not
  a claim about a specific ODE.
- **Falsifiable replication:** recompute the spectrum span and fit R² on a fresh dataset; a span
  < ~5 orders of magnitude or a power-law fit with R² clearly above the exponential would falsify
  the ill-conditioning claim.

## Finding 2: Trajectory behavior is essentially uncorrelated with the singular spectrum

- **Object and scope:** per-parameter trajectory change (distance at progress 2 minus progress 0)
  across 4,232 parameters and 2 solver families, correlated with the same-index `log_sv`.
- **Relationship/direction/boundary:** the change is a *near-null* function of `log_sv`.
  Correlation of trajectory slope with `log_sv` is −0.163 (family 0) and −0.047 (family 1); pooled
  −0.069. The OLS regression slope of total change on `log_sv` is ≈ −0.0007 (negligible).
- **Core quantitative result & unit:** `corr_slope_sv` fam0 = −0.163, fam1 = −0.047, pooled =
  −0.069; `reg_total_on_sv_slope` ≈ −0.0007; analysis unit = per-parameter trajectory.
- **Supporting evidence:** analysis source `analysis.py` (trajectory–spectrum correlation) and
  `output/figures/trajectory.png`; result evidence `results.json` (`trajectories` block),
  `output/robustness_a.json`, `output/result_tables_trajectories.tsv`.
- **Robustness:** leave-one-solver-family-out reproduces each family's own correlation
  (`lof_without_fam0 = −0.047`, `lof_without_fam1 = −0.163`), so the near-null relationship is
  not a pooling artefact.
- **Counter-evidence / limitation / alternatives:** the correlations are small but not exactly
  zero; family 0 shows a slightly stronger (still weak) negative association. This is a null/near-
  null effect, reported honestly rather than forced into a relationship.
- **Falsifiable replication:** recompute the trajectory–spectrum correlation on a new dataset; a
  |correlation| clearly above ~0.3 and a significant regression slope would falsify the near-null
  claim.

## Finding 3: Measurement–simulation agreement is strong and unbiased, but does not establish
parameter identifiability

- **Object and scope:** 13,000 paired `measurement_like_value` / `simulation_like_value`
  observations.
- **Relationship/direction/boundary:** measurement and simulation agree strongly and without bias
  (mean error ≈ −0.000195, two-sided p > 0.05), with a high Pearson correlation ≈ 0.765.
  However, this agreement is **not constant** across simulation-value quantiles
  (`corr_by_quantile = [0.58, 0.35, 0.12, 0.07, 0.19]`, spread ≈ 0.50), so it does not by
  itself establish parameter identifiability (per the manifest warning).
- **Core quantitative result & unit:** `corr_meas_sim = 0.765` (winsor 0.768, outlier-excluded
  0.813); `err_mean = −0.000195`, `t = −0.14`, p > 0.05; analysis unit = the 13,000 pairs.
- **Supporting evidence:** analysis source `analysis.py` (measurement–simulation statistics) and
  `output/figures/robustness_b.png`; result evidence `results.json` (`measurement_simulation`
  block), `output/robustness_b.json` (`measurement` + `corr_by_quantile`),
  `output/result_tables_pairs.tsv`.
- **Robustness:** bias remains near zero under winsorization and 10% two-sided outlier exclusion;
  the non-constant quantile correlation is the specific evidence that agreement ≠ identifiability.
- **Counter-evidence / limitation / alternatives:** overall agreement is real and robust; the
  quantile non-uniformity is a caveat on the identifiability reading, not a refutation of the
  agreement itself.
- **Falsifiable replication:** recompute the error bias and quantile-wise correlations on a new
  dataset; a non-zero bias or a constant correlation across quantiles would change the identifiability
  interpretation.

## Finding 4: Solver family is a real source of convergence difference (family 1 converges more)

- **Object and scope:** per-parameter total trajectory change across 4,232 parameters and the 2
  solver families.
- **Relationship/direction/boundary:** family 1 converges substantially more than family 0
  (mean total change fam0 ≈ −0.158, fam1 ≈ −0.377; difference fam0 − fam1 ≈ +0.219).
- **Core quantitative result & unit:** `diff_mean_tot_fam0_minus_fam1 = +0.219`, bootstrap 95% CI
  [0.216, 0.223] (excludes 0); sign-test p = 1.0; 95.6% of parameters have fam0 change > fam1
  change; analysis unit = per-parameter trajectory, whole-family holdout.
- **Supporting evidence:** analysis source `analysis.py` (solver-family comparison) and
  `output/figures/robustness_a.png`; result evidence `results.json` (`solver_specific` block),
  `output/robustness_a.json` (`batch_leaveout`), `output/robustness_b.json` (`solver` block),
  `output/result_tables_trajectories.tsv`.
- **Robustness:** batch leave-out (4 contiguous parameter batches) shows fam1 converges more in
  *every* batch (means −0.376…−0.381 vs −0.148…−0.168; conv fractions 0.998…0.999 vs
  0.990…0.996), and the lowest-SV bin is 100% convergent for fam1. Leave-one-family-out is
  consistent.
- **Counter-evidence / limitation / alternatives:** the difference is a per-family mean effect;
  a few parameters (≈0.14% convergent in fam0) are exceptions. The effect is on a *normalised
  distance-like* value, whose physical scale is not specified.
- **Falsifiable replication:** recompute the fam0 − fam1 mean total change on a new dataset; a
  difference consistent with 0 (CI containing 0) would falsify the solver-family effect.

## Limitations, alternative explanations, and unresolved items

- **Log base assumption:** `log_singular_value` is treated as a natural log. If it were base-10,
  the absolute slope magnitudes rescale (span stays ≈ 3.63 orders in log10 vs 36.32 in ln), but
  the qualitative ill-conditioning, fit rankings, and all conclusions are unchanged.
- **Spectrum join assumption:** `parameter_index == mode_index` is the natural alignment given
  identical 0–4,231 ranges; no other alignment is provided by the data.
- **Agreement ≠ identifiability:** the manifest explicitly warns small measurement–simulation error
  does not prove identifiability; Finding 3's non-constant quantile correlation operationalizes this.
- **No external data or precomputed results** were merged; all numbers derive from the three
  frozen files via executed code.
- **Unresolved:** none material — all robustness checks either confirm or strengthen the findings.
  The only nuance (quantile non-constancy) supports the identifiability caveat rather than
  weakening it.

## Reproduction

1. Verify inputs (all three `data/*.tsv` SHA-256 match `data_manifest.json`; no missingness).
2. Run the analysis: `python3 output/analysis.py` → writes `output/results.json`,
   `output/result_tables_*.tsv`, `output/figures/{spectrum,trajectory,convergence_vs_sv}.png`.
3. Run the robustness checks: `python3 output/robustness_a.py` (spectrum/trajectory/batch) and
   `python3 output/robustness_b.py` (measurement/solver); each writes its own JSON and figure.
4. Read the synthesis: `output/verification_report.md`, `output/verification.json`.
5. Validate delivery: `python3 validate_output.py output`.
6. Verify the correction evidence is self-reproducible: `python3 output/verify_corrections.py`
   → writes `output/recomputed_values.json` (independent recomputation + command/source) and
   `output/correction_receipt.json` (structured receipt for the original StructuredOutput
   `linear_slope` defect). Both exit cleanly and report `status: PASS`.

Dependencies: `pandas`, `numpy`, `matplotlib` (Agg backend). All scripts are deterministic; the
bootstrap uses a fixed seed (42).

## Methods (analysis workflow)

- **Spectrum:** OLS fits of `log_sv` vs mode-index (exponential), vs `log1p(mode_index)`
  (power-law), and raw `sv` vs mode-index (linear); R², slopes, early/tail segment slopes,
  effective dimension by threshold, and cumulative-energy k95/k99.
- **Trajectories:** pivot to (param, family, progress); total change = d2−d0; alternative
  progress scales (first-step, last-step, quadratic fit); convergence/flat flags; correlation of
  slope/total with `log_sv` per family and pooled; leave-one-family-out.
- **Measurement–simulation:** error = meas − sim; mean/abs-mean/percentiles; Pearson correlation;
  winsorization (5/95) and 10% two-sided outlier exclusion; relative error; correlation by
  simulation-value quantile; two-sided normal bias test.
- **Solver-specific:** per-parameter fam0−fam1 difference; bootstrap (seed 42, 3000 reps) on the
  mean difference; sign test; 4-batch contiguous leave-out; convergence fraction by log-SV bin.
- **Verification:** a synthesis agent read only the JSON artifacts and figures (no recomputation)
  and adjudicated each finding as supported/weakened/null/unresolved.

## Correction evidence (integrity)

- Early manual hash check dropped a trailing `5c5`, falsely implying a `measurement_simulation_pairs.tsv`
  mismatch. Programmatic re-check against the actual `data_manifest.json` confirms **all three**
  files match their manifest hashes exactly. See `output/correction_evidence.md`.
- The child StructuredOutput for `robustness_a.json` `spectrum_models.linear_slope` dropped an
  `e-05` exponent; the JSON artifact and independent recomputation confirm the true value
  `−4.285991340480605e-05`. This field is not used for any conclusion (raw-scale linear fit is
  the worst model, R² ≈ 0.225). See `output/correction_evidence.md`.

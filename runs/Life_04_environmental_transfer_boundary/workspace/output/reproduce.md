# Reproduction — `Life_04_environmental_transfer_boundary`

Companion reproduction document for `output/Result.md`. Everything below was executed in this
workspace; no step is proposed-but-unrun.

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11.2 (GCC 12.2.0) |
| Available libraries | `numpy` 2.4.6, `pandas` 3.0.5 |
| **Unavailable** libraries | `scipy`, `statsmodels`, `sklearn`, `matplotlib`, `seaborn`, `networkx`, `pytest` (all `ModuleNotFoundError`) |
| Consequence | Every test is implemented from first principles in `output/analysis/ti_lib.py`: exact Fisher, exact binomial, exact Wilcoxon rank-sum via dynamic programming, Monte-Carlo permutation with the +1 correction, IRLS logistic regression (plain / ridge / Firth), cluster-robust sandwich covariance, ANOVA ICC. Figures are hand-rendered SVG. |
| Randomness | `numpy.random.default_rng` with seeds `20200101` (main), `+1`, `+2`, `+3` (sensitivity, supplementary, checks). |

No external data, package, or network resource was used. The only scientific inputs are the three
files in `data/`.

## 2. Inputs and integrity

```
data/unit_level_observations.tsv           30 x 5   723 B
data/site_level_observations.tsv          212 x 3  3868 B
data/symptom_assay_surface_summary.tsv     30 x 5   737 B
```

SHA-256 of all three files is recomputed inside `run_analysis.py` and written to
`analysis/results/results.json → payload.files`; the values match `data_manifest.json`. The run also
asserts cross-file consistency (`day_columns_identical_between_files`,
`assay_identical_between_files`: both `true`) and zero missing cells.

## 3. Exact command sequence

Run from the project root, in this order:

```bash
python3 output/analysis/run_analysis.py       # -> analysis/results/results.json + 7 CSVs
python3 output/analysis/run_sensitivity.py    # -> analysis/results/sensitivity.json
python3 output/analysis/run_supplementary.py  # -> analysis/results/supplementary.json
python3 output/analysis/run_checks_figures.py # -> analysis/results/checks.json + figures/*.svg
python3 validate_output.py output             # -> semantic delivery receipt
```

Every script is standalone, reads only `data/`, and rewrites its own artifact. Observed runtimes
on this host (one core, Python 3.11.2 / numpy 2.4.6 / pandas 3.0.5) are:

| Script | Runtime |
|---|---|
| `output/analysis/run_analysis.py` | ~193 s |
| `output/analysis/run_sensitivity.py` | ~25 s |
| `output/analysis/run_supplementary.py` | ~6 s |
| `output/analysis/run_checks_figures.py` | <1 s |
| `python3 validate_output.py output` | <1 s |

The cost is dominated by permutation replicates (200,000 for unit-level rank tests, 20,000 for
logistic/changepoint and within-unit label nulls, 1,000 for the two LOO increment nulls, 20,000
for the unit-level bootstrap). Permutation steps that build a full `n_rep × n_rows` array are
chunked (5,000 rows at a time) so peak memory stays near `chunk × n_rows` rather than
`n_rep × n_rows`; the replicate values are identical to a single un-chunked pass under the same
seed.

## 4. Method-to-file map

| Analysis step used in the report | Code | Result artifact |
|---|---|---|
| Payload integrity and cross-file consistency | `run_analysis.py` §0 | `results.json → payload` |
| Rotating `site_role_id` detection | `run_analysis.py` §0b, `run_sensitivity.py` §A | `results.json → site_label_structure`, `sensitivity.json → A_label_rotation` |
| Within-unit all-or-nothing positivity, ICC, design effect | `run_analysis.py` §1 | `results.json → within_unit_positivity`, `→ design_effect_sites` |
| Naive site-level vs unit-blocked contrast | `run_analysis.py` §1 | `results.json → naive_vs_blocked` |
| Overdispersion of unit positive counts | `run_analysis.py` §1 | `results.json → overdispersion_positive_counts` |
| Illness-day association (threshold-free) | `run_analysis.py` §2a | `results.json → F_day` |
| Exact Wilcoxon null by dynamic programming | `run_analysis.py` §2a | `results.json → F_day.p_exact_all_assignments` |
| Illness-day threshold grid | `run_analysis.py` §2a, `run_sensitivity.py` §E | `day_threshold_sensitivity.csv`, `sensitivity.json → E` |
| Assay association and gradient | `run_analysis.py` §2c, `run_sensitivity.py` §D | `results.json → F_assay`, `sensitivity.json → D_assay_robustness` |
| Changepoint grid and search-adjusted null | `run_supplementary.py` §1–2 | `supplementary.json → changepoint_grid`, `→ best_cut_2x2` |
| Adjusted models (symptom, day) | `run_supplementary.py` §3 | `supplementary.json → adjusted_models` |
| Symptom and surface linkage | `run_analysis.py` §2d | `results.json → F_symptom` |
| Unit-level logistic models and LOO | `run_analysis.py` §3 | `results.json → models_unit_level` |
| Whole-unit holdouts (LOO, leave-2, blocks) | `run_analysis.py` §4 | `results.json → robustness_whole_unit_holdouts`, `leave_one_unit_out.csv`, `block_holdouts.csv` |
| Multiplicity control | `run_sensitivity.py` §C | `sensitivity.json → C_multiplicity` |
| Complete-unit leave-out of 8-row units | `run_checks_figures.py` §G1 | `checks.json → G1_drop_eight_row_units` |
| Containment asymmetry | `run_checks_figures.py` §G2 | `checks.json → G2_containment` |
| Exact binomial on the low-assay cluster | `run_checks_figures.py` §G3 | `checks.json → G3_low_assay_cluster` |

## 5. Verification embedded in the run

The scripts contain assertions that fail loudly rather than degrade silently:

- `run_analysis.py` asserts the DP-based exact rank-sum p is computed over all
  119,759,850 label assignments and that the closed-form rho-from-rank-sum transform reproduces the
  observed Spearman rho to `1e-9`.
- `run_supplementary.py` asserts that the vectorised permutation arithmetic reproduces the
  fitted best-cut log-likelihood to `1e-6` before it is trusted for 20,000 replicates.
- `run_analysis.py` asserts the assay column is identical in `unit_level_observations.tsv` and
  `symptom_assay_surface_summary.tsv`.
- `run_sensitivity.py` verifies the rotation rule holds in 30/30 units rather than assuming it.

## 6. Known deviations and failure handling

- **`scipy`/`statsmodels` absence** was discovered before any test was written, so no test was
  silently replaced by a weaker surrogate. The substitutions (exact DP null in place of
  `scipy.stats.mannwhitneyu`, IRLS in place of `statsmodels.Logit`) are exact rather than
  approximate where the null is finite, and Monte-Carlo where it is not.
- **Separation.** The illness-day model separates positivity under leave-one-out and the best-cut
  model separates a full cell (`rate_below = 1.0`). A weak ridge (`penalty=0.05`) is applied
  identically to observed and permuted fits, and Firth fits are reported alongside; the separation
  itself is reported as a limitation, not hidden.
- **Runtime.** Two first-draft permutation loops (a 200,000 × 212 within-unit shuffle and a
  20,000 × 27 changepoint search calling `logistic_fit`) exceeded the 600 s command timeout and were
  killed; both were replaced by matrix formulations that produce the same statistics with identical
  seeds. The vectorised versions were validated against the fitted likelihood by assertion. Full-scale
  permutation loops were additionally chunked (5,000 rows at a time), which keeps peak memory bounded
  and yields identical replicate values under the same seed.
- **LOO replicate count and an abandoned optimisation.** The two leave-one-out permutation nulls in
  `run_analysis.py` were originally run at 2,000 replicates. An optimisation that cached IRLS fits
  on the sufficient statistic `X.T @ y` was introduced and then *removed*: that key is valid only
  for a fixed design, but leave-one-out changes the design per held-out unit, so it is an invalid
  cache key there (the embedded determinism assertion caught it, returning 0.0828 instead of the
  true −0.4276 increment). With the cache removed, each LOO evaluation costs 30 IRLS solves
  (≈0.05 s), so the LOO nulls were reduced to 1,000 replicates. This is the *only* place the
  replicate count differs from the original plan, and it is recorded in
  `results.json → models_unit_level.loo_increment_* .n_rep`. All other permutation studies retain
  their intended counts (200,000 unit-level rank/label tests, 20,000 logistic/changepoint and
  bootstrap nulls, 20,000 unit-level overdispersion bootstrap).

## 7. Determinism

Given the same `numpy`/`pandas` versions and seeds, re-running the four commands regenerates the
same JSON artifacts. Monte-Carlo p-values are reported with the +1 correction, so the smallest
attainable p is `1/(n_rep+1)` (e.g. `5.0e-05` at 20,000 replicates) rather than 0.
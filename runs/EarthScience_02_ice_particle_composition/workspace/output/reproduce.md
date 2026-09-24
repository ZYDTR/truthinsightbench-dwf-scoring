# Reproduction guide

Task `EarthScience_02_ice_particle_composition` — "Depth-Resolved Particle Abundance and
Composition in Ice Samples". T0 = 2018-01-01.

Everything reported in `Result.md` is produced by the scripts listed below, run from the
project root with `python3`. No network access, no external data and no precomputed task
result was used for any number in this work.

## Environment

- Python 3.11, `numpy` 2.4.6 and `pandas` 3.0.5 (both preinstalled).
- `scipy` 1.17.1, `statsmodels` 0.15.0 and `matplotlib` 3.11.2 were installed from the
  configured PyPI mirror during the run
  (`pip install scipy statsmodels matplotlib`).
- The main analysis deliberately implements its own Spearman, Student-t and
  cluster-robust OLS routines in `output/analysis/common.py` so the pipeline does not
  depend on the optional packages. Section "Cross-validation" below shows these agree
  with `scipy`/`statsmodels` to floating-point precision.
- Matplotlib figures were rendered with the `Agg` backend.

## Data

Read-only, hashes verified against `data_manifest.json` by step 1:

| file | rows | cols | sha256 |
|---|---|---|---|
| `data/particle_composition_profiles.tsv` | 30 | 83 | `c8c8dbe2b0518e2f24f2d709ba6ffaf6991349ee4e354427f50a9c89553c3ebf` |
| `data/sample_environment_profiles.tsv` | 30 | 15 | `2e371aa337c1a076c8055a1f4c7b4390a7a8bcec16561dc671ee1e0f660f33a8` |

Both files list the same 30 `sample_id` values in the same order.

## Pipeline (run in order)

| step | script | writes |
|---|---|---|
| 1 | `output/analysis/01_data_audit.py` | `output/results/01_data_audit.json` |
| 2 | `output/analysis/02_explore.py` | console screen (no artifact) |
| 3 | `output/analysis/03_candidates.py` | `output/results/03_candidates.json` |
| 4 | `output/analysis/04_resolution.py` | `output/results/04_resolution.json` |
| 5 | `output/analysis/05_quantisation_null.py` | `output/results/05_quantisation.json` |
| 6 | `output/analysis/06_robustness.py` | `output/results/06_robustness.json` |
| 7 | `output/analysis/07_resolution_and_spectrum.py` | `output/results/07_resolution_spectrum.json` |
| 8 | variance partitioning + anomaly scan (inline, reproduced in `output/analysis/08_variance_partition_note.md`) | `output/results/08_variance_partition.json` |

Shared helpers: `output/analysis/common.py`.

Independent adversarial verification (separate code, written from scratch by sub-agents,
each told to refute rather than confirm):

| finding | verification | code | result artifact |
|---|---|---|---|
| F1 | independent re-derivation + exact sign-flip, Theil–Sen, cluster bootstrap | `output/results/verify_F1.py`, `verify_F1_part2.py`, `verify_F1_part3.py` | `output/results/verify_F1.json` |
| F1, F2, figures | figure agent's own adversarial checks | `output/analysis/10_figures_adversarial.py`, `output/verify/verify_figs_adversarial2.py` | `output/results/10_figures_adversarial.json`, `output/verify/verify_figs_adversarial2.json` |
| F2 | independent recomputation + deepest-interval confound test + group sign-flip + cluster jackknife/bootstrap | `output/verify/verify_F2.py`, `output/verify/verify_F2_adversarial.py`, `output/verify/verify_F2_cluster_inference.py` | `output/verify/verify_F2.json`, `output/verify/verify_F2_adversarial.json` |
| F3 | 8-part adversarial programme (abundance restriction, covariate specificity, compositional transforms, `group_05` jackknife, depth-partial permutation, quantisation, cluster inference) + 2 addenda (13 further tests) | `output/verify_F3/verify_F3_adversarial.py`, `verify_F3_addendum.py`, `verify_F3_addendum2.py` | `output/verify_F3/verify_F3.json` (logs `run.log`, `addendum.log`, `addendum2.log`) |
| F4 | chance-agreement null, uniqueness, bound direction, falsification check | `output/results/verify_F4_maincheck.json` (checks run inline by the main session; a delegated re-implementation `output/verify_F4/verify_F4.py` was written but did not finish within the run, so its JSON does not exist) | `output/results/verify_F4_maincheck.json` |
| F5 | independent bin-range table, injection power analysis, second counting-null with seed 777, binary-pattern test, core-bin constraint | `output/verify_F5/verify_F5.py` | `output/verify_F5/verify_F5.json` |
| figures 1-4 | rendered from the TSVs | `output/analysis/09_fig3_fig4_size_quantisation.py` | `output/figures/fig1..fig4*.png`, plot data in `output/results/fig*.csv` |

Machine-readable cross-reference of every registered finding to its analysis code and result
artifacts: `output/results/findings_index.json`.

## Analysis conventions that must be reproduced exactly

1. **Analysis unit.** One sampled depth interval. 30 intervals in 5 collection groups
   (`group_01` n=5, `group_02` n=7, `group_03` n=7, `group_04` n=5, `group_05` n=6).
   Intervals inside a group are repeated observations, never independent sites.
2. **Within-group ("fixed-effect") transform.** Every regression subtracts the group mean
   from both the response and every regressor (function `demean` in
   `output/analysis/06_robustness.py`). The intercept is therefore identically zero.
3. **Cluster-robust standard errors.** CR1 with the collection group as the clustering
   dimension and `G-1 = 4` degrees of freedom (`ols_cluster` in `common.py`). With only
   5 clusters these are the binding constraint on every p-value; classical
   homoskedastic errors would be far smaller.
4. **Log transforms.** `log10` for concentration, chlorophyll a, POC, PON, standard
   deviation and bin midpoints; `log(PE/varnish)` (natural log) for the PE/varnish ratio.
   The minimum total concentration is 33 per L and the minimum chlorophyll a is
   0.02 ug/L, so no zero/negative values are transformed.
5. **Depth.** `depth_mid_m = (depth_min_m + depth_max_m)/2`. The "deepest interval of a
   group" is the one with the largest `depth_mid_m` in that group.
6. **Missing data.** Only `temperature_C` for `sample_023` and `sample_024` is missing;
   every other column is complete and no imputation is performed. Analyses using
   temperature therefore have n=28.

## Cross-validation of the hand-rolled statistics

`output/results/04_resolution.json` → `cross_validation_hand_rolled_vs_scipy`:
Spearman rho agrees with `scipy.stats.spearmanr` exactly (difference 0.0) and the
two-sided p-values agree to <1e-15. `cross_validation_cluster_se` compares
`ols_cluster` against `statsmodels.OLS(...).fit(cov_type="cluster", ...)`: the
p-values match, but see `Result.md` §"Limitations" — the two implementations differ in
the tiny-`G` scaling factor, and the report quotes the `statsmodels`-compatible
p-value where they disagree. `output/results/06_robustness.json` additionally reports
the exact group sign-flip null (32 configurations) and 20 000-draw within-group
permutation nulls, which do not depend on any variance estimator.

## Figures

```
output/figures/fig1_depth_abundance.png      output/results/fig1_depth_abundance.csv
output/figures/fig2_chlorophyll_abundance.png output/results/fig2_chlorophyll_abundance.csv
output/figures/fig3_size_spectrum.png        output/results/fig3_size_spectrum.csv
output/figures/fig4_quantisation.png         output/results/fig4_quantisation.csv
```

Every plotted value is computed from the two TSVs (and, for figure 4, the reconstructed
tallies in `output/results/07_resolution_spectrum.json`). No plotted value is hand-entered.

## Semantic output validation

```
python3 validate_output.py output
```

## Failed / unavailable steps (retained, not hidden)

- A planned external literature check did **not** complete: the sub-agent's API call was
  rejected upstream (`400 ... DataInspectionFailed`) before returning any source. Because
  the contract makes external search optional and requires a verified first-public date
  strictly before T0 for any source used, **no external source is cited anywhere in this
  work**. All background statements in `Result.md` are labelled as interpretation, not
  as literature-supported fact.
- A first multi-agent verification workflow failed because the `StructuredOutput` tool was
  rejected for that agent role, causing agents to loop; it was stopped (`TaskStop`) and
  relaunched without structured output. Its completed products are retained and used.
- The multi-agent run was interrupted more than once by process teardown. Verification
  products that did land are used as-is; the F4 delegated re-implementation
  (`output/verify_F4/verify_F4.py`) was written but its JSON never completed, so the F4
  adversarial checks were run inline by the main session instead and are recorded in
  `output/results/verify_F4_maincheck.json`.
- `output/analysis/04_resolution.py` contains an early quantum-search routine
  (`best_quantum`) that is ill-posed because the count columns are derived from rounded
  percentages. Its output is retained in `04_resolution.json` but the corrected
  reconstruction in step 7 (`consistent_N` in `07_resolution_and_spectrum.py`) supersedes
  it; `Result.md` uses only the step-7 numbers.
- Three counting-null simulations of the size-spectrum artefact exist and **disagree**
  (≈58 %, ≈88 % and ≈96 % of the association explained). All three are retained in
  `output/results/05_quantisation.json`, `output/results/07_resolution_spectrum.json` and
  `output/verify_F5/verify_F5.json`; `Result.md` reports the disagreement rather than
  selecting one number.
- `output/results/10_figures_adversarial.json` and several verification JSONs contain
  `null` entries where a sub-agent's Python `None` was serialised mid-computation. No
  reported quantity in `Result.md` depends on a `null` field.
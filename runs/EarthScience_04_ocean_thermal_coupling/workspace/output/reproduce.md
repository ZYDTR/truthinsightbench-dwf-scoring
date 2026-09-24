# Reproduction Guide

Task `EarthScience_04_ocean_thermal_coupling`. This document records how every
number in `output/Result.md` was produced, so the analysis can be re-executed
end to end from the frozen input.

## 1. Input

| Item | Value |
|---|---|
| Data file | `data/observations.tsv` |
| Rows | 621 data rows + 1 header row |
| Bytes | 15150 |
| SHA-256 | `dd1954d79d70fff765e91be1d1778eecb0b7f5bd3afe75a4169960e8ae46c358` |
| Declared hash in `data_manifest.json` | identical (verified in step 01) |
| Columns | `c01_depth_sed_m_merged`, `c02_age_ka_bp_1_see_reference_s`, `c03_age_ka_bp_2_see_reference_s`, `c04_c_wuellerstorfi_18o_pdb_vs_vpdb_analytical_error_0` |

The analysis uses only the four columns present in the frozen table. No
external dataset, precomputed result, or substitution is used anywhere.

## 2. Environment

```
python3 3.11
numpy   2.4.6
pandas  3.0.5
scipy   1.17.1
statsmodels 0.15.0
matplotlib  3.11.2
```

`scipy`, `statsmodels` and `matplotlib` were not present in the base
environment and were installed with `python3 -m pip install scipy matplotlib
statsmodels` before the analysis ran. All scripts use relative paths resolved
from the repository root via `Path(__file__).resolve().parents[2]`, so they can
be run from any working directory.

## 3. Column mapping used by every script

```python
df = pd.read_csv("data/observations.tsv", sep="\t")
df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
```

The renamed labels are used only for readability; the underlying values are
unmodified.

## 4. Execution order

Run from the repository root:

```bash
python3 output/analysis/01_data_profile.py        # schema, integrity, analysis unit
python3 output/analysis/02_spectral_phase.py      # period, AR(1) null, phase, holdout
python3 output/analysis/03_age_coordinate_leverage.py
python3 output/analysis/04_depth_domain_spectrum.py
python3 output/analysis/05_lsr_excursion.py
python3 output/analysis/06_transition_structure.py
python3 output/analysis/07_coupling_scale.py
python3 output/analysis/08_smoothing_sensitivity.py
python3 output/analysis/09_confirmations.py       # reconciliation of headline numbers
python3 output/analysis/10_figures.py             # figures
python3 output/analysis/11_shared_denominator_adjudication.py  # adjudication of the contested coupling
```

Steps 01 and 02 are the primary-evidence scripts. Steps 03-08 are independent
robustness modules. Step 09 recomputes every headline number from the raw table
in one self-contained script and adjudicates the disagreements that were found
between modules. Step 10 regenerates the figures.

## 5. Modules, their questions, and their artifacts

| Step | Question | Result artifact |
|---|---|---|
| 01 | What is the schema, integrity, sampling grid and missing-data boundary? | `output/results/01_data_profile.json`, `01_depth_series.csv` |
| 02 | Is there a periodic component, at what period, and is its phase stable? | `output/results/02_spectral_phase.json`, `02_periodogram.csv`, `02_sliding_window.csv`, `02_block_holdout.csv` |
| 03 | How much does the choice between the two age coordinates matter? | `output/results/03_age_coordinate_leverage.json` |
| 04 | Is the periodicity in the proxy or in the age axis? | `output/results/04_depth_domain_spectrum.json` |
| 05 | Is the ~241 ka accumulation excursion unique, and what causes it? | `output/results/05_lsr_excursion.json` |
| 06 | Is there a mean-level transition beyond trend and periodicity? | `output/results/06_transition_structure.json` |
| 07 | Is the accumulation/proxy coupling real or a shared-denominator artefact? | `output/results/07_coupling_scale.json`, `07_block_means.csv` |
| 08 | How sensitive are the conclusions to smoothing, detrending and gap treatment? | `output/results/08_smoothing_sensitivity.json`, `08_scan_curves.csv` |
| 09 | Do the headline numbers survive an independent single-script recomputation? | `output/results/09_confirmations.json` |
| 10 | Figures | `output/figures/fig1..fig4*.png` |
| 11 | Is the LSR/\|d18O\| coupling arithmetic, and does a residual survive the artefact? | `output/results/11_shared_denominator_adjudication.json` |

## 6. Key computational choices

**Analysis unit.** Rows are not replicates. Median depth step is 0.01 m, median
age step 0.46 ka, lag-1 autocorrelation of `d18o_pdb` at that step is 0.79. All
significance statements therefore use either (a) an AR(1) Monte-Carlo null built
on the irregular axis with correlation `rho**|Δx|/step`, (b) contiguous
non-overlapping blocks, or (c) an effective sample size. Naive per-row p-values
are reported in step 07 only to show how misleading they are.

**Period search.** Design matrix `[1, t, cos(wt), sin(wt), cos(2wt), sin(2wt)]`
with `w = 2π/P`, scanned over `P ∈ [40, 300]` ka in 0.065 ka steps unless a
different grid is stated. Fundamental amplitude is `hypot(b_cos, b_sin)` and
phase is `atan2(-b_sin, b_cos)`.

**AR(1) surrogates.** Cholesky factorisation of the correlation matrix
`R_ij = rho**|x_i - x_j| / median_step`, diagonal inflated by `1e-10`, scaled to
the standard deviation of the detrended series, sampled on the same irregular
axis as the observation. Every surrogate is passed through the SAME period scan
as the data, so the empirical p-value pays for the look-elsewhere effect.

**Shared denominator.** `np.gradient(depth, age)` and `np.gradient(d18o, age)`
both divide by the same age increment. Note that `np.gradient(age, age)` is
identically 1 and useless; the meaningful increment is `np.gradient(age)`
(range 0.20-2.53 ka). The permutation null that breaks this coupling is
described in step 07.

## 7. Corrections applied during the run

Five items were found and corrected; they are recorded here because the
reproducibility trace must show them rather than hide them.

1. **Step 02 persistence baseline.** The original gradient-matched baseline used
   an unclamped linear extrapolation at the training-range edges, producing
   empty-segment RMSE values of 9.95 per mille for the first and last blocks.
   Replaced with a nearest-neighbour persistence baseline clamped at the
   training edges; results were regenerated.
2. **Step 07 `dAge` selection.** The module initially reported
   `r(dAge, LSR) = -0.014`, which contradicts a direct computation. The cause
   was an incorrect age-increment variable. Re-running the module after the fix
   gives `r(dAge, LSR) = -0.820`, matching `09_confirmations.py`. The step 07
   JSON as delivered is the corrected version.
3. **Step 03 constant-rate coordinate.** Step 03 reported 84.07 ka for the
   constant-sedimentation-rate coordinate, using
   `mean(gradient(depth, age)) = 0.022345 m/ka`. The record-mean rate over the
   full span is `span_depth / span_age = 0.017592 m/ka`. Step 09 re-derives the
   constant-rate result two independent ways and gets 106.73 ka from the
   coordinate scan and 106.69 ka from the depth-domain peak divided by the mean
   rate. The report uses the step 09 values and treats the 84.07 ka figure as a
   defect of step 03.

4. **Step 11 added to adjudicate a mid-run disagreement.** Two independent
   reviews of the LSR vs |d18O/dage| relationship reached opposite conclusions
   because they permuted different objects (d18O *values* vs d18O *increments*),
   and one of the two review files was rewritten during the run with a partly
   reversed verdict. Rather than cite either version, step 11 constructs the
   arithmetic-only null directly from resampled numerators and reports the
   conservative bound (70% of the observed correlation is denominator
   arithmetic). Both reviews' numbers are retained verbatim in
   `output/results/verify_C2.json`; the report follows step 11.

5. **Verifier return values vs verifier files.** The workflow that ran the
   verification agents reported four of the five as failing to emit their
   structured return value, and one API error. All five verifier scripts and all
   five result JSON files were nevertheless written and executed successfully.
   The report was sourced from the files on disk, never from the agents' return
   messages. Two verifier files (`verify_C2.json`, `verify_C4.json`) were
   rewritten by their agents late in the run; every number cited in the report
   was re-checked against the final on-disk versions, and one threshold
   (11 ka vs 12 ka in C4) and the C2 correlation nulls were corrected as a
   result.

## 8. Figure regeneration

```bash
python3 output/analysis/10_figures.py
```

Writes `fig1_record_and_ages.png`, `fig2_spectra_and_null.png`,
`fig3_phase_and_holdout.png`, `fig4_lsr_and_change_point.png` to
`output/figures/`. Figures are regenerated from the raw table plus the step 02
CSV artifacts; none of them is hand-edited.

## 9. Determinism

All Monte-Carlo steps use explicit seeds (`20220101` in step 02, `20260919` in
step 07, `913` in step 09). Re-running the pipeline reproduces the quoted
numbers bit-for-bit apart from the Monte-Carlo p-values, whose resolution is set
by the surrogate count (300-2000 draws) and which are quoted to that resolution.
# w2_variance — variance decomposition and explanatory-power ranking

Workstream: rank candidate explanations for `log10(Erosion rate)` on the pooled
glacial+nonglacial table, and test the "no stable relation" baseline.
Script: `w2_variance.py` (run with `python3 output/analysis/w2_variance/w2_variance.py` from the
project root). Everything below is from that single execution against the frozen TSVs.
Estimator: explicit design-matrix OLS (normal equations), asserted in-script against
`statsmodels.formula.api.ols` for the small models and against the one-way ANOVA
identity for `C(Reference)` — all matched to <1e-10 on R² and <1e-6 on AIC/BIC.

## Filters / samples

| sample | rule | n |
|---|---|---|
| `full_logER` | `logER` non-missing (rate>0) | 6711 |
| `logTI_subset` | `logER` and `logTI` non-missing (interval>0) | 6555 |
| `common_nested` | the above + `logS,logP,logA,Latitude,Methodology,Process` | 6450 |

`logTI` is missing for 156 of the 6711 `logER` rows (all `Cosmogenic surface`,
`Marrero et al., 2018`). The 105-row gap between `logTI_subset` and `common_nested`
comes from `logA` missing in glacial rows; `Relief` methods collapse to n=1 in the
common sample. AIC/BIC are only compared within a sample; the `full_logER` table is
available-case per model and is reported for completeness only.

## Headline ranking (common sample n=6450, k = rank of design matrix)

| model | n | k | R² | adj R² | AIC | BIC |
|---|---|---|---|---|---|---|
| C(Reference) | 6450 | 365 | 0.6844 | 0.6655 | 9428.1 | 11899.8 |
| logTI * C(Methodology) + C(Process) | 6450 | 10 | 0.5934 | 0.5928 | 10351.4 | 10419.1 |
| logTI * C(Methodology) | 6450 | 9 | 0.5670 | 0.5665 | 10754.8 | 10815.7 |
| logTI + logS + logP + logA + C(Methodology) | 6450 | 9 | 0.2468 | 0.2458 | 14325.9 | 14386.9 |
| logS + logP + logA + Latitude | 6450 | 5 | 0.2127 | 0.2122 | 14603.2 | 14637.1 |
| logTI + C(Methodology) + C(Process) | 6450 | 7 | 0.1784 | 0.1776 | 14882.5 | 14929.9 |
| C(Methodology):C(Process) cells | 6450 | 8 | 0.1439 | 0.1429 | 15150.0 | 15204.2 |
| C(Compilation) | 6450 | 16 | 0.1134 | 0.1113 | 15391.6 | 15500.0 |
| C(Process) | 6450 | 2 | 0.0653 | 0.0651 | 15704.6 | 15718.2 |
| logTI + C(Methodology) | 6450 | 6 | 0.0580 | 0.0573 | 15762.3 | 15802.9 |
| C(Methodology) | 6450 | 5 | 0.0297 | 0.0291 | 15951.7 | 15985.6 |
| logTI | 6450 | 2 | 0.0060 | 0.0058 | 16101.1 | 16114.7 |
| null (intercept only) | 6450 | 1 | 0.0000 | 0.0000 | 16137.9 | 16144.6 |

Ranking by R² and by AIC/BIC agree except that AIC/BIC disfavour the 365-parameter
`C(Reference)` model (BIC 11899.8, worse than `ti_x_method_proc` at 10419.1): Reference
explains the most variance but at a crippling parameter cost. Among parsimonious
models, **method-stratified timescale (`logTI * C(Methodology)`, ± Process) dominates
everything else by a wide margin**.

## Incremental R² (common sample, each term added last)

Timescale term (`logTI`), added last:

| base model | ΔR² | F | p |
|---|---|---|---|
| C(Methodology) | 0.0284 | 194.1 (1, 6444) | 1.7e-43 |
| C(Methodology)+C(Process) | 0.0602 | 472.1 (1, 6443) | 4.3e-101 |
| saturated Method:Process cells | 0.0435 | 345.1 (1, 6441) | 4.5e-75 |
| logS+logP+logA+Latitude | 0.0170* | 146.7 (1, 6440) | 2.1e-33 |
| full `logTI*method` block | 0.4752 | 1881.7 (4, 6440) | ~0 |

Process term (`C(Process)`), added last:

| base model | ΔR² | F | p |
|---|---|---|---|
| C(Methodology) | 0.0885 | 646.9 (1, 6444) | 4.5e-136 |
| C(Methodology)+logTI | 0.1204 | 943.8 (1, 6443) | 1.5e-193 |
| C(Methodology)*logTI | 0.0264 | 417.7 (1, 6440) | 5.2e-90 |

Method term added last after logTI+Process: ΔR² = 0.1029 (F=201.7, df 4, 6443, p≈3e-163).
Geology block added last after logTI+Method: ΔR² = 0.1888 (F=538.0, df 3, 6441, p≈5e-312).

Sequential ladder (ΔR² per step): null → method 0.0297 → +logTI 0.0284 → +Process 0.1204
→ method×logTI 0.4150 → (replacing with the additive geology model drops R² by 0.3466,
i.e. geology excluding the interaction is far weaker).

Interpretation: timescale alone is almost worthless (R²=0.006) and the additive
timescale term is small (ΔR²≈0.06). **The explanatory power lives in the
method × timescale interaction**, which is a measurement-regime signature, not a
smooth scale law.

## "No stable relation" baseline

Pooled `logER ~ logTI`, n=6555: slope = **-0.0221 dex/dex** (se 0.0041, 95% CI
[-0.0302, -0.0141], R²=0.0044, p=7e-08 vs 0; p≈0 vs -1). So a single pooled slope is
statistically non-zero but practically negligible and nowhere near the
fixed-thickness value of -1: a single stable ER~TI relation is rejected.

With Reference-clustered SE the marginal slope is **not** significant:
CI [-0.0676, +0.0233], p=0.34.

Single pooled slope vs method-specific slopes: interaction F = 1884.4 (df 4, 6545),
p ≈ 0, ΔR² = 0.499. Method-stratified slopes:

| Methodology | n | slope | se | 95% CI | R² |
|---|---|---|---|---|---|
| Cosmogenic detrital | 3479 | **-0.9939** | 0.0013 | [-0.9963, -0.9914] | 0.9945 |
| Cosmogenic surface | 163 | -0.8772 | 0.0360 | [-0.9482, -0.8062] | 0.7870 |
| Thermochronometric | 72 | -0.6141 | 0.1146 | [-0.8427, -0.3856] | 0.2909 |
| Relief | 25 | -0.2738 | 0.1097 | [-0.5008, -0.0469] | 0.2131 |
| Volumetric | 2816 | **+0.1336** | 0.0086 | [+0.1166, +0.1505] | 0.0785 |

So the answer is not one slope: two methods sit at ≈-1 (mechanical coupling,
ER = thickness/interval), and Volumetric has a *positive* slope (+0.13).

## Simpson-style reversal check

Marginal pooled slope -0.0221 → conditional-on-method slope **-0.1017**
(se 0.0076, CI [-0.1166, -0.0868]). It does not flip sign; it becomes ~4.6× more
negative because method groups have very different `logTI` means (Volumetric mean
-0.33 dex ≈ 0.5 yr; Thermochronometric +6.26 dex ≈ 1.8 Myr) and different rate
levels. With Reference-clustered SEs: marginal p=0.34 (n.s.), conditional
-0.1017 CI [-0.1892, -0.0142], p=0.023. So conditioning on method rescues a
weak, small negative slope, but the reversal is an *attenuation*, not a sign flip;
the apparent pooled relation is a between-method composition artefact.

## Clustering / effective sample size (n=6711 `logER` rows)

- 417 unique `Reference` (max cluster 512 rows; 120 singletons; median 7; p90 37)
- 17 unique `Compilation`, 1938 unique `Locality` (1604 singletons, max 154)
- 5114 unique (Latitude, Longitude) pairs (5052 after rounding to 3 dp)

One-way random-effects (ANOVA, n0-corrected) ICC of `logER` within `Reference`:
**ICC = 0.726** (MSB 10.198, MSW 0.237, mean cluster size 16.09, n0 15.88),
design effect **11.95**, effective n = 6711/11.95 ≈ **561** (se inflation ×3.46).
Independent ML cross-check (`MixedLM` `logER ~ 1`): REML ICC = 0.847, ML ICC = 0.847
(τ²=1.316, σ²=0.237) — same conclusion, larger. Within-Locality ICC = 0.828
(deff 3.04, effective n ≈ 2208). Within-Compilation ICC = 0.356 (deff 141,
effective n ≈ 48!). Residual ICC after removing the pooled `logTI` fit: 0.677 by
Reference; after method-conditioning: 0.666. **Clustering is severe and barely
reduced by either timescale or method terms**, so nominal p-values on 6711 rows
overstate evidence by roughly 3-12×.

## Mechanical-coupling / unit audit (verified independently)

92 rows have `Exhumation (m)`, `ER`, `TI`: 81 glacial, 11 nonglacial.
- `ER*TI == Exhumation*1000` to rtol 1e-6 for 20 rows; rtol 1e-3 for 28 rows.
- All 28 are glacial; **0 of 11 nonglacial rows satisfy it** (median relative error
  ≈ 0.999 — for those, Exhumation is a different quantity, e.g. min/max-derived).
- Median relative error among glacial rows: 0.0029 (i.e. essentially exact for a
  subset, small rounding elsewhere; max 9.0 for outliers).
So the identity holds exactly only for a minority (≈35%) of the exhumation-carrying
glacial rows, but it confirms the ER=thickness/interval construction is present in
this compilation — consistent with the -1 slopes seen in the detrital cosmogenic and
Volumetric-nonglacial strata.

Compilation-level slopes expose two data-quality facts:
- **Chen et al., 2022 (n=1521, 22.7% of all rows) has zero within-compilation variance
  in `logTI`** (all rows share one interval, var(logTI)=1.2e-32); its slope is not
  identified (the naive fit returns a garbage se ≈ 4e5). Any "pooled" slope is
  therefore partly driven by a cluster with no internal timescale information.
- `OCTOPUS` (n=3457) slope -0.9952 (R²=0.998); `Marrero et al., 2018` -1.419
  (R²=0.983); `GloRiSe` -0.083; `This study` -0.157; `Hallet et al., 1996` -0.056.
- Sensitivity dropping Chen et al., 2022: pooled slope -0.0182
  (CI [-0.0267, -0.0098]), conditional-on-method slope -0.9939
  (CI [-1.0127, -0.9750]) — i.e. with that cluster removed the conditional slope
  collapses onto -1, exactly the fixed-thickness signature.

Within process×method cells (n≥10): Glacial/Volumetric -0.073, Glacial/Cosmogenic
surface -0.342, Glacial/Thermochronometric -0.652, Nonglacial/Volumetric +0.117,
Nonglacial/Cosmogenic detrital -0.995, Nonglacial/Cosmogenic surface -0.982.
Note Glacial/Volumetric is nearly flat while Nonglacial/Cosmogenic detrital is
essentially -1: method, not process, is what separates the regimes.

## What failed / unresolved

- No failures in the final run; all models converged. Two script iterations were
  required: (a) `smf.ols("logER ~ C(Reference)")` with 417 levels took ~183 s per
  fit under machine load, so fits were moved to explicit normal-equation OLS (a
  `lstsq` on the 417-column design was still ~82 s; `solve` on X'X ≈26 s);
  (b) one incremental-R² pair (`combined` vs `geology+method`) was non-nested and
  returns `"error": "not a valid nested pair"` — kept as an honest error record, and
  re-run as the valid `combined_plus_lat` vs `geology_plus_method` pair.
- The ANOVA ICC (0.726) and MixedLM ICC (0.847) disagree; the ANOVA estimator with
  strongly unbalanced clusters (max 512) is the more conservative one and is what the
  design-effect numbers use. Both imply effective n in the hundreds, not thousands.
- Latitude is unwinsorized (range -85.9 to 80.5, 1634 southern-hemisphere rows); it adds
  only ΔR²=0.0047 when added last to the combined model, so results are not sensitive to it.
- These are observational decompositions: "explained variance" here includes the
  mechanical ER=thickness/TI coupling and compilation-level reporting conventions,
  not only geological process.
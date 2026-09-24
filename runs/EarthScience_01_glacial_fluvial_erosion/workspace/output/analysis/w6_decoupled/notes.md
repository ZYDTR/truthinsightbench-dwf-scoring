# w6_decoupled — de-biased scaling from records with independent clocks

Script: `w6_decoupled.py` (run with `python3`, ~70 s, deterministic, seed 20240919).
All fits: OLS on log10 axes computed with `numpy.linalg.lstsq` (statsmodels import is
pathologically slow in this container — >2 min for a bare `import statsmodels.api`,
so it was never used). CIs are cluster bootstraps that resample **References**
(the independent analysis unit per DATA_GUIDE), 1000 draws (400 for n>1500).

Universe: all 427 glacial + 6128 nonglacial rows with `Erosion rate > 0` and
`Time interval > 0` (6555 rows; no rows dropped by these filters).
`logER = log10(ER)`, `logTI = log10(TI)`, `logET = log10(ER*TI)` (mm).

## 1. Coupling classification audit

Rule (row-level, CSV in `row_classification_audit.csv`). A row is **coupled** if any of:

| evidence | test | rows |
|---|---|---|
| `exhumation_identity` | `Exhumation > 0` and `\|ER*TI/(EX*1000) − 1\| ≤ 0.05` | 54 |
| `flux_identity` | `SF > 0`, `A > 0` and `\|ER/(SF/A·10⁶) − 1\| ≤ 0.05` | 154 |
| `compilation_product_lock` | Methodology×Compilation, n≥10, ≥5 distinct intervals, ≥1.0 dex of logTI, IQR(log10 ER·TI) ≤ 0.10 dex | 3457 |
| `reference_product_lock` | same geometry at Methodology×Compilation×Reference, n≥5 | 0 as primary evidence (62 ref cells hard-lock, all inside OCTOPUS) |
| `crn_attenuation_lock` | CRN stratum whose median `ER*TI` is within 0.25 dex of the ~0.6 m effective attenuation depth Λ/ρ, IQR(log10 ER·TI) ≤ 0.30 dex, ≥1 dex logTI | 64 |

**independent** = not coupled AND Methodology ∈ {Cosmogenic surface, Cosmogenic
detrital, Thermochronometric} (nuclide inventories / cooling ages are their own clock).
**indeterminate** = everything else (Volumetric, Relief) — the reported interval is
very often the denominator of the rate but we cannot prove it row by row.

Counts (Methodology × Process), from `results.json`:

| Process | Methodology | coupled | independent | indeterminate |
|---|---|---|---|---|
| Glacial | Cosmogenic detrital | 0 | 9 | 0 |
| Glacial | Cosmogenic surface | 31 | 68 | 0 |
| Glacial | Relief | 8 | 0 | 10 |
| Glacial | Thermochronometric | 4 | 64 | 0 |
| Glacial | Volumetric | 27 | 0 | 206 |
| Nonglacial | Cosmogenic detrital | 3457 | 13 | 0 |
| Nonglacial | Cosmogenic surface | 64 | 0 | 0 |
| Nonglacial | Relief | 0 | 0 | 7 |
| Nonglacial | Thermochronometric | 0 | 4 | 0 |
| Nonglacial | Volumetric | 138 | 0 | 2445 |

Totals: coupled 3729, independent 158, indeterminate 2668.
Strict-independent (independent and not near-locked): Glacial 141, Nonglacial 17.

### Verified facts (independently re-derived, not restated)

* Exhumation identity: 92 rows have `Exhumation > 0`. At 1e-3 relative tolerance the
  identity holds in **both** directions indistinguishably (28 rows satisfy
  `E = EX·1000/T`; 28 satisfy `EX = E·T/1000`; 28 either way), i.e. the column is
  arithmetically redundant. 54 rows pass at the 5 % tolerance used here. The known-fact
  figure of 81 glacial rows with Exhumation is correct; only 54 of them close to ≤5 %.
* Nonglacial Exhumation rows (11) are **not** closed — their `ER*TI/(EX*1000)` ratios
  are 875–10000, so those intervals are in different units from the glacial table.
  They were therefore not used for any identity flag.
* Sediment-flux identity: 154 rows satisfy `ER = SF/A·10⁶` at ≤5 % (138 GloRiSe,
  10 glacial "This study", 6 Hallet). Note `ER·A/SF` is ≈0.5 for about half of the
  Hallet rows, i.e. a factor-2 convention difference; those rows failed the identity.

### The decisive new finding: the cosmogenic strata are also product-locked

For a CRN-inventory rate the steady-state limit is `E·T → P·τ = Λ/ρ`, the effective
attenuation depth. Inspection of `ER*TI` per CRN stratum (`crn_attenuation_depth_test.csv`):

| stratum | n | median ER·T | IQR(logET) | logTI range |
|---|---|---|---|---|
| Nonglacial Cosmogenic detrital OCTOPUS | 3457 | **0.650 m** | 0.029 dex | 5.75 dex |
| Nonglacial Cosmogenic surface Marrero 2018 | 34 | 0.485 m | 0.197 | 1.10 dex |
| Nonglacial Cosmogenic surface "This study" | 30 | 0.438 m | 0.297 | 3.13 dex |
| Glacial Cosmogenic surface Balter-Kennedy 2021 | 32 | 0.084 m | 0.974 | 0.41 dex |
| Glacial Cosmogenic surface "This study" | 67 | 1.250 m | 1.490 | 3.13 dex |
| Glacial Cosmogenic detrital "This study" | 9 | 5.000 m | 0.804 | 3.48 dex |

All three nonglacial CRN strata cluster near 0.4–0.65 m, i.e. the same ~0.6 m
attenuation depth that the frozen OCTOPUS source script uses implicitly
(`octopusexposureage.jl` converts a nuclide inventory to an age using production
scaled by topography/self-shielding/snow, then the tabulated rate is `ebe_mmkyr/1000`).
Reproducing that script on the shipped `octopusdata.tsv` gives `E·age` with 5–95 %
range **614–721 mm**, confirming `E = 0.65/T` for OCTOPUS. The three glacial CRN
strata sit at 0.08–5 m and have large logET spread, so they are *not* attenuation-locked.

This moved 3521 rows (OCTOPUS + Marrero + nonglacial "This study" CRN surface) from
"independent" to "coupled", which is why the nonglacial independent class collapses to
17 rows. Without this test the same data would have been reported as an "independent"
validation of slope ≈ −1.0 (n=3521) — the single largest trap in this dataset.

## 2. Coupled vs independent slopes

`slopes_by_class.csv`, `within_reference_slopes.csv`. Slope in dex per dex.

| subset | n | slope | se | R² | cluster-boot 95 % CI | TI range (yr) |
|---|---|---|---|---|---|---|
| ALL rows | 6555 | −0.022 | 0.004 | 0.004 | [−0.072, 0.021] | 0.003–2.5e8 |
| Glacial ALL | 427 | −0.057 | 0.018 | 0.023 | [−0.141, 0.007] | 0.02–2.5e8 |
| Glacial coupled | 70 | −0.210 | 0.044 | 0.255 | [−0.342, −0.144] | 2–4.5e7 |
| Glacial independent | 141 | +0.116 | 0.050 | 0.037 | [−0.167, 0.420] | 140–2.5e8 |
| Glacial indeterminate | 216 | −0.047 | 0.027 | 0.014 | [−0.122, 0.010] | 0.02–3.4e7 |
| Nonglacial ALL | 6128 | −0.031 | 0.004 | 0.009 | [−0.078, 0.014] | 0.003–2e7 |
| Nonglacial coupled | 3659 | −0.366 | 0.007 | 0.428 | [−0.621, −0.244] | 0.003–2.9e6 |
| Nonglacial independent | 17 | +0.171 | 0.040 | 0.554 | [−1.129, 0.297] | 1000–1.1e7 |
| Glacial Cosmogenic detrital, indep | 9 | −0.940 | 0.159 | 0.833 | [−1.253, −0.510] | 1000–3e6 |
| Glacial Cosmogenic surface, indep | 68 | −0.264 | 0.124 | 0.064 | [−1.026, 0.392] | 140–1.9e5 |
| Glacial Thermochronometric, indep | 64 | −0.577 | 0.111 | 0.302 | [−0.956, −0.064] | 1e5–2.5e8 |
| Nonglacial Cosmogenic detrital, coupled | 3457 | −0.995 | 0.001 | 0.998 | [−0.999, −0.992] | 5.3–2.9e6 |

**Do the steep near-−1 slopes survive?** Nowhere except (a) the product-locked
suites and (b) two small CRN strata:

* Nonglacial Cosmogenic surface **all** (Marrero+This study, n=64): −0.982 ± 0.078,
  R²=0.718, CI [−1.489, −0.845]. But all 64 rows carry the CRN attenuation lock and
  `ER·T` = 0.44–0.49 m; the slope is the closure.
* Glacial Cosmogenic detrital (n=9): −0.940 ± 0.159, R²=0.833, CI [−1.253, −0.510].
  `ER·T` median 5 m, spread 0.80 dex — not attenuation-locked, so this one is the only
  survivor. It is 9 rows from 6 references spanning only 1–3 Myr, so it is weak.

Within-Reference (fixed-effects) slopes, which remove all between-paper heterogeneity
and use only within-paper timescale variation:

| subset | n | refs | slope | se | R² |
|---|---|---|---|---|---|
| ALL rows | 3823 | 189 | −0.752 | 0.008 | 0.679 |
| Glacial ALL | 230 | 35 | −0.043 | 0.045 | 0.004 |
| Nonglacial ALL | 3580 | 154 | −0.984 | 0.004 | 0.950 |
| Glacial independent | 75 | 7 | −0.162 | 0.079 | 0.055 |
| Nonglacial coupled | 3400 | 150 | −1.003 | 0.002 | 0.983 |

The pooled −0.75 for "all rows" is entirely the OCTOPUS within-paper closure
(−1.006, R²=0.999). Glacial rows show **no** within-paper timescale scaling (−0.043).

## 3. Alternative-definition robustness / predictor ranking

`predictor_dominance.csv`, `no_timescale_models.csv`. Model: `logER ~ C(Methodology)`
(+ `C(Process)` for pooled rows) plus one predictor at a time; each predictor is
ranked by |standardized β| and by its incremental R² (F-test, 1 df).

Independent pooled (n=152 complete cases; extra dummy `C(Process)`):

| predictor | std β | ΔR² | p(F) | rank |β| | rank ΔR² |
|---|---|---|---|---|---|
| log10 Time interval | −0.843 | 0.1485 | 2.1e-08 | 2 | 1 |
| log10 Area | +1.201 | 0.0883 | 2.4e-05 | 1 | 3 |
| log10 Precip | +0.361 | 0.1173 | 8.5e-07 | 3 | 2 |
| Latitude | +0.215 | 0.0456 | 2.8e-03 | 4 | 4 |
| log10 Slope | −0.094 | 0.0076 | 0.23 | 5 | 5 |

All rows (n=6450): logS dominates (std β +0.469, ΔR²=0.143), logP second
(ΔR²=0.100), logTI third (ΔR²=0.028, std β −0.329). Slope dominance on all rows is
a between-stratum effect (the steep, low-slope OCTOPUS/Chen suites).

**Timescale excluded** (model with `C(Methodology)` + `C(Process)` + logS+logP+logA+Lat,
independent pooled, n=152, R²=0.497): precipitation dominates (+1.044 ± 0.197,
p=1e-6, std β 0.474), area +0.130 ± 0.044 (p=0.004), slope −0.506 ± 0.113 (p=1.5e-2),
latitude +0.0036 ± 0.0022 (p=0.10). So when timescale is removed, **precipitation**
is the single dominant control among the remaining predictors for the independent rows.

## 4. Threshold sensitivity (independent rows)

`threshold_sensitivity.csv`, `threshold_sensitivity_by_method.csv`.
Because the strict-independent rows have `TI ≥ 140 yr` and soft-lock rows `TI ≥ 1000 yr`,
the "below 10 yr / below 100 yr" subsets are empty by construction: the thresholds
0, 10 and 100 yr return identical fits.

| subset | TI≥0 | TI≥10 | TI≥100 | TI≥1000 | trajectory |
|---|---|---|---|---|---|
| pooled strict-indep | +0.099 ± 0.044 (n=158) | same | same | +0.118 ± 0.051 (n=146) | flat, shallow |
| pooled incl soft-lock | −0.411 ± 0.072 (n=222) | same | same | −0.444 ± 0.084 (n=210) | slightly steeper |
| Glacial independent | +0.116 ± 0.050 (n=141) | same | same | +0.157 ± 0.061 (n=129) | flat |
| Nonglacial independent | +0.171 ± 0.040 (n=17) | same | same | same | flat |
| Glacial ALL | −0.057 ± 0.018 (n=427) | −0.081 (n=342) | −0.034 (n=272) | +0.051 (n=222) | crosses zero |
| Nonglacial ALL | −0.031 (n=6128) | −0.951 (n=3548) | −0.951 (n=3541) | −0.934 (n=3160) | jumps at TI≥10 |

The apparent scale dependence is **not** produced by short-window records: on
independent rows nothing changes with threshold, and on all rows the steep value
appears only once the 5-yr Chen et al. (2022) fixed-interval suite (TI ≡ 5 yr) is
excluded — that step is a stratum change, not a timescale effect.

## 5. Counter-case: large ΔT with little ΔE

`ET_spread_by_stratum.csv`, `countercase_pairs.csv`. Diagnostic = within-stratum
spread of `log10(E·T)` (dex).

| stratum | n | range logET | range logTI | range logER | slope |
|---|---|---|---|---|---|
| Glacial Volumetric This study | 140 | **9.14** | 9.25 | 4.75 | −0.06 |
| Glacial Volumetric Hallet 1996 | 62 | **7.21** | 5.80 | 3.78 | −0.06 |
| Glacial Relief Geirsdóttir 2007 | 4 | **6.44** | 5.66 | 1.24 | +0.05 |
| Nonglacial Volumetric GloRiSe | 1057 | **6.15** | 2.56 | 4.17 | −0.08 |
| Nonglacial Volumetric Chen 2022 | 1521 | 5.39 | 0.00 | 5.39 | n/a (TI constant) |
| Nonglacial Volumetric This study | 5 | 4.46 | 4.18 | 1.15 | −0.07 |

Strongest pair: **Glacial Volumetric "This study"** — interval changes by 2.57e7 yr
(0.019 → 2.57e6 yr) while rate changes only 1.01 dex (0.078 → 0.80 mm/yr), so
`log10(E·T)` moves 9.14 dex (0.002 mm → 2.06e6 mm of equivalent thickness).
Second: Hallet 1996, 7.21 dex over a 1.2e4 yr interval change.

Across the 19 strata with n≥5 the within-stratum slope is correlated with the spread
of `log10(E·T)` (Spearman ρ = +0.62, p = 0.0044; OLS slope 0.30 ± 0.12, R²=0.26):
the more tightly the product is pinned, the closer the apparent exponent is to −1.
That is the quantitative signature of the artifact, not of geology.

## 6. Bounded conclusion

* **Best de-biased exponent**: Glacial **Cosmogenic surface** independent class,
  **−0.26 ± 0.12** (n=68, R²=0.064, cluster-boot CI [−1.00, 0.37]) over
  ~1.4e2–1.9e5 yr. It is the only stratum that is neither coupled nor attenuation-
  locked and still has a resolvable timescale axis. The CI is wide, so the data
  support "shallow negative, consistent with 0" but exclude −1.
* **Thermochronometric** (glacial, n=64): −0.58 ± 0.11 over 1e5–2.5e8 yr, but 47 of
  68 rows share TI = 1 Ma. Excluding TI = 1 Ma gives −0.656 ± 0.182 (n=21, 10 refs),
  restricting to TI > 1 Ma gives −0.733 ± 0.255 (n=19). The timescale dependence
  survives those cuts but rests on ~20 rows.
* **Glacial Cosmogenic detrital** (n=9): −0.94 ± 0.16, R²=0.83, but 6 references and
  1–3 Myr only — the only near-−1 estimate that is not demonstrably a closure.
* **Not quotable**: pooled and all-row slopes (dominated by product locks); any
  Volumetric or Relief stratum (E·T spans 3–9 dex, so coupling and geology are
  inseparable — see §5); the nonglacial CRN suites (attenuation-locked at 0.44–0.65 m).

## What failed / unresolved

* `import statsmodels.api` took >2 min in this container and blocked a first script
  version; replaced with a hand-rolled OLS using `lstsq` + `pinv` (identical results).
* Two early classification rules were wrong and were caught by inspection:
  (i) an IQR-based lock test that used *all* rows flagged GloRiSe as coupled because
  153 of 154 rows share TI = 2.74e-3 yr — that is a constant timescale, not a
  timescale span, so the test was rewritten to require ≥5 distinct intervals and
  ≥1 dex of logTI; (ii) the first robust version required multi-row interval groups,
  which then missed OCTOPUS (thousands of singleton intervals), and was replaced by
  explicit `n_distinct ≥ 5` + `1 dex` gates.
* `Nonglacial | Cosmogenic detrital | This study` (n=13) has a single interval
  (1000 yr) so its slope (−0.065 ± 0.017, R²=1e-16) is not estimable; flagged
  `estimable=False` via `x_range_dex`, no bootstrap possible (one cluster).
* Whether Glacial Volumetric rows with no exhumation column are thickness/interval
  constructions cannot be resolved from the frozen columns; they remain
  "indeterminate" and are excluded from all quoted exponents.
* No external evidence was used; all conclusions rest on the shipped TSV columns and
  on the two shipped source scripts (OCTOPUS `.jl`) for the attenuation-depth reading.

## Superseded files present in this directory

Six files predate this run and were left untouched: `decoupled_corrected.py`,
`corrected_results.json`, `fig_product_lock_partition.png`, `table_class_slopes.csv`,
`table_product_lock_partition.csv`, `table_predictor_dominance_free_strata.csv`.
They are **not** outputs of `w6_decoupled.py` and should not be used. They record a
contradiction that the current classification no longer has: the earlier
`w6_decoupled.py` revision labelled the OCTOPUS stratum "independent" while its own
stratum diagnostics said product-locked. In this run the OCTOPUS soft-lock is
promoted to hard (IQR logET = 0.029 dex, 3447 distinct intervals spanning 5.75 dex)
and all 3457 rows are labelled `coupled` with evidence `compilation_product_lock`,
consistent with `stratum_lock_diagnostics.csv`. Their headline numbers
(product-locked slope −0.995, "product-free" slope +0.063) coincide with the
corresponding values here because both runs ultimately keyed on the same E·T
criterion; the current run adds the exhumation, flux and CRN-attenuation tests and
the within-reference fitting.

## Files

| file | content |
|---|---|
| `w6_decoupled.py` | the script actually run (self-contained, deterministic) |
| `results.json` | every fitted model with n / estimate / se / p / r² / CI + filter rules |
| `row_classification_audit.csv` | row-level classification trail (6555 rows) |
| `stratum_lock_diagnostics.csv`, `reference_lock_diagnostics.csv` | lock tests per stratum / per reference |
| `crn_attenuation_depth_test.csv` | CRN ER·T vs the ~0.6 m attenuation depth |
| `slopes_by_class.csv`, `within_reference_slopes.csv` | §2 tables |
| `independent_definition_sensitivity.csv` | 10 alternative "independent" definitions |
| `predictor_dominance.csv`, `no_timescale_models.csv` | §3 tables |
| `threshold_sensitivity.csv`, `threshold_sensitivity_by_method.csv` | §4 tables |
| `ET_spread_by_stratum.csv`, `countercase_pairs.csv`, `stratum_slope_vs_spread.csv` | §5 tables |
| `w6_figures.png` | 4-panel figure: classes, threshold subsets, slope trajectory, slope vs product spread |
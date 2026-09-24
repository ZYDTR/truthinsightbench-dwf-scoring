# Scale and Measurement Boundaries of Glacial and Nonglacial Erosion Rates

**Task ID:** `EarthScience_01_glacial_fluvial_erosion`
**Evidence boundary (T0):** 2024-10-17
**Data:** the frozen workspace payload only — `data/` (21 files, 6,189,779 bytes, all sha256-verified against `data_manifest.json`)

---

## Summary

The compilation contains 6,715 published erosion-rate estimates: 427 glacial and 6,288 nonglacial.
Reported erosion rate varies by more than five orders of magnitude (glacial 0.0004–1000 mm/yr,
nonglacial 0–118.9 mm/yr), and the obvious candidate explanation — that the observation timescale
governs the reported rate — is **not supported as a general relation**. The pooled slope of
log₁₀(rate) against log₁₀(interval) is −0.022 dex/dex and explains 0.4 % of the variance; under
clustering by source it is not distinguishable from zero (95 % CI [−0.068, +0.023], p = 0.34).

What *does* govern the reported rate is **how the rate was constructed**. The compilation splits
into two populations that must not be pooled:

1. **Product-locked strata**, where the reported rate was formed by dividing a measured thickness
   by the very interval stored beside it. Here rate and interval are not independent observations,
   and the regression between them is an algebraic restatement of the construction. The largest
   such stratum (n = 3,470, nonglacial cosmogenic detrital) has a slope of −0.9951 (R² = 0.9979)
   with the implied eroded thickness pinned at 650.4 mm (sd of log₁₀ = 0.033 dex).
2. **Product-free strata**, where thickness and interval vary independently. Here the same
   regression yields +0.063 dex/dex (R² = 0.024), with a cluster-bootstrap interval that includes
   zero. Climate, slope and latitude carry more explanatory power than timescale does.

Glacial rates are higher than nonglacial rates for the same method and the same timescale — by
2.9× pooled across methods, and by ~2–4× in the two strata whose comparator pools are actually
populated — but this contrast **reverses** in the Relief and Thermochronometric strata, where
nonglacial rates exceed glacial ones. That reversal is data-warranted and is reported as a boundary
on the contrast rather than smoothed away. A much larger ratio (74×) appears in the
cosmogenic-surface stratum, but that comparator pool is 100 % Antarctic and is reported with an
explicit latitude caveat rather than as the headline number.

Five findings are registered below. Two report a supported relationship, one reports a null/boundary
result, one reports a conditional reversal, and one reports a design limit that constrains all the
others.

---

## Data and analysis unit

**Analysis unit.** One published erosion-rate estimate per row. This is *not* an independent
replicate structure: 6,711 rows carrying a rate come from 417 distinct `Reference` values, 17
`Compilation` values and 5,114 distinct coordinate pairs. A single reference contributes up to 512
rows. The intraclass correlation of log₁₀(rate) within `Reference` is **ICC = 0.726**, giving a
design effect of 11.95 and an **effective sample size of ≈ 560** from 6,711 rows. Every inferential
statement below is therefore reported with both its row count and its cluster count, and the large
strata are bootstrapped by `Reference` cluster.

**Filter rule (explicit, retained in audit trails).** `Erosion rate > 0` and `Time interval > 0`.
This removes 4 non-positive rates and 156 rows with a missing interval (all nonglacial). No
winsorization is applied to any headline number; winsorized and trimmed variants appear only as
labelled robustness rows in
`output/analysis/w3_scaling/tables/outlier_treatments.csv` and
`output/analysis/w3_scaling/tables/threshold_sensitivity.csv`.

**Missingness.** Glacial: `Exhumation (m)` present for 81/427, `Sediment flux` for 51/427,
`Area` for 392/427. Nonglacial: `Exhumation (m)` for 11/6,288, `Time interval` for 6,128/6,288.
Row-level audit: `output/analysis/w5_audit/table_row_anomaly_audit.csv`.

**Not erosion data.** `data/raw/beryllium.tsv` (70 rows) holds dissolved and colloidal beryllium
concentrations (nmol/kg) with salinity, pH and DOC in estuaries and groundwater. It has no rate,
interval or denudation column and is excluded.

**Auxiliary files are not row-aligned.** `slope_glacial.tsv` (461 lines), `slope_nonglacial.tsv`
(6,271), the CHELSA `precipitation_*.tsv` lookups and the `lithology_*.tsv` tables are keyed to
*locations*, not rows. Positional comparison against the main tables fails outright (strict match
fraction 0.000 for both slope files). The lithology tables settle the question: all four are
exactly one row per coordinate key, their key sets equal the main tables' key sets, and an inner
join returns every main row (229 keys → 427 glacial rows; 4,900 keys → 6,288 nonglacial rows).
The precipitation lookups carry 233 and 4,896 keys against 229 and 4,900 in the main tables, so
they are near-complete rather than exact. They were used only as coordinate-keyed joins, never as
row-aligned companions — which matters, because a naive positional join would silently misassign
slope and precipitation values across rows.
Evidence: `output/analysis/w5_audit/table_auxiliary_alignment.csv`,
`output/analysis/w5_audit/companion_alignment.csv`.

---

## Finding 1 — The dominant rate–timescale signal in this compilation is a construction artefact, not a scaling law

**Object and scope.** The pooled glacial + nonglacial compilation; the relationship between
`Erosion rate (mm/yr)` and `Time interval (yr)`.

**Relationship and direction.** Where a rate was computed as a measured thickness divided by the
interval reported beside it, the rate–interval relation is exactly `log E = log D − log T`: slope
−1, and R² → 1 when the thickness `D` is roughly fixed. This is the single strongest structure in
the dataset and it is algebraic, not geological.

**Core quantitative result** (analysis unit: one published estimate; `logER ~ logTI`, OLS):

| Stratum | n | slope (dex/dex) | R² | sd of log₁₀(E·T) |
|---|---|---|---|---|
| Nonglacial, Cosmogenic detrital | 3,470 | **−0.9950** (se 0.0008) | **0.9979** | 0.033 dex |
| Nonglacial, Cosmogenic surface | 64 | −0.9818 (se 0.0782) | 0.718 | 0.331 dex |
| Glacial, Cosmogenic detrital | 9 | −0.9405 (se 0.1590) | 0.833 | 0.606 dex |
| Nonglacial, Volumetric | 2,583 | **+0.1174** (se 0.0098) | 0.053 | 1.928 dex |
| Glacial, Volumetric | 233 | −0.0729 (se 0.0253) | 0.035 | 2.299 dex |

Four independent lines of evidence establish the artefact:

1. **Exact unit identity.** Where `Exhumation (m)` is populated on the *glacial* table,
   `Erosion rate × Time interval` reproduces `Exhumation × 1000` within 0.1 % for 28 of 81 rows
   and within 1 % for 48 of 81 (median ratio 1.0000); by stratum, 80 % of `Relief` and 57 % of
   `Thermochronometric` rows (`output/analysis/w1_coupling/table_A_unit_identity.csv`). The
   tolerance matters and is reported at four levels (1e-9: 11/81; 1e-6: 19/81; 0.1 %: 28/81;
   1 %: 48/81), so "exact" here means agreement to a stated relative tolerance, not bit-equality.
   **The 11 nonglacial rows with this column cannot be used for the test at all**: their ratio is
   ~1000 rather than ~1, i.e. the nonglacial `Exhumation (m)` column is offset from its rate
   column by a factor of 1000. Taken literally that would make the stored nonglacial rates m/yr
   rather than the documented mm/yr, or the stored exhumations km rather than m — either way the
   two columns are mutually inconsistent on that table, and the identity test rests on the 81
   glacial rows only.
2. **Thickness collapse.** The nonglacial cosmogenic-detrital stratum's implied eroded thickness
   has median 650.4 mm, IQR [630.3, 674.7] mm, and only 0.033 dex of spread across 5.75 dex of
   timescale. A physically free thickness would spread over orders of magnitude — as it does in
   the volumetric strata (1.93 and 2.30 dex).
3. **Bottom-up reconstruction.** Rebuilding the reported rate from the raw nuclide columns in
   `octopusdata.tsv` with the attenuation relation `E = P·Λ/(ρ·N)` and the exposure ages in
   `octopusages.tsv` reproduces the published rate with slope 1.008 and R² = 0.9976 (n = 3,221),
   confirming `E × t = Λ/ρ = 64.95 cm` — a material constant, independent of site
   (`output/analysis/w1_coupling/table_C_density_sensitivity.csv`).
4. **Within-study slopes.** Re-fitting the slope *inside each individual study* of that stratum
   gives a median of −1.0037 across 115 references (IQR [−1.0161, −0.9949], median R² = 0.9996).
   The −1 is reproduced study by study, which is the signature of a shared arithmetic construction
   rather than of a shared landscape process
   (`output/analysis/w0_main/table_within_reference_slopes.csv`).

**Robustness.** Theil-Sen robust slope −0.9954 (95 % CI [−0.9963, −0.9945]); 2,000-draw row
bootstrap CI [−0.9963, −0.9938]; **Reference-cluster** bootstrap CI [−0.9985, −0.9915] (computed
independently in two workstreams: w3 gives [−0.99853, −0.99177], w6 gives [−0.99847, −0.99154]);
HC3 robust SE 0.00064. The conclusion is unchanged under every estimator
(`output/analysis/w3_scaling/results.json`, `large_strata_robustness`).

**Counterevidence and limitation.** The `Nonglacial Volumetric` stratum (n = 2,583) has a
thickness spread of 1.93 dex and a slope of **+0.117**, i.e. the opposite sign. This is important
and is not a defect: it demonstrates that a positive, geology-like relation is recoverable from
this compilation *when the construction does not force the slope*. The claim is therefore bounded —
it is about specific product-locked strata, not about the compilation as a whole. Two caveats:
the identity test is available for only 92 rows because `Exhumation (m)` is largely unpopulated;
and for the `Balter-Kennedy et al., 2021` and `Young et al., 2016` subsets the implied thickness
ratio sits at ~1002 rather than 1000, a 0.2 % unit-rounding offset that does not change any
conclusion but does mean "exact" is a 10⁻³ relative tolerance, not bit-equality.

**Executable falsification.** Take the 1,057-row `Chen et al., 2022` subset, whose rates are
reach-scale and independently calculated. If the artefact reading is right, its implied thickness
should not collapse: the measured spread is 1.156 dex with no estimable slope because the interval
is constant at 5 yr. A sharper test: obtain the ungated sub-basin thicknesses behind the OCTOPUS
basins and check whether `E × t` stays at 65 cm once thickness is allowed to vary freely. If it
does not, Finding 1 is refuted as an artefact claim.

**Evidence.**
`output/analysis/w1_coupling/coupling_audit.py` ·
`output/analysis/w1_coupling/results.json` ·
`output/analysis/w1_coupling/table_A_unit_identity.csv` ·
`output/analysis/w1_coupling/table_B_thickness_collapse.csv` ·
`output/analysis/w1_coupling/table_C_density_sensitivity.csv` ·
`output/analysis/w0_main/table_within_reference_slopes.csv` ·
`output/analysis/w3_scaling/results.json`

---

## Finding 2 — Once product-locked strata are removed, no general rate–timescale scaling survives

**Object and scope.** The 6,555 rows with both a positive rate and a positive interval; the
question of whether any single or method-stratified timescale relation is recoverable.

**Relationship and direction.** Null. The pooled relation is effectively flat and does not
strengthen when the coupling-affected rows are excluded — it *disappears*.

**Core quantitative result.** Partitioning strata by the thickness diagnostic
(`sd(log₁₀ E·T) ≤ 0.15` dex with a timescale spread ≥ 1.0 dex ⇒ product-locked):

| Class | n | References | slope | R² | cluster-bootstrap 95 % CI |
|---|---|---|---|---|---|
| Product-locked | 3,470 | 166 | −0.9950 | 0.9979 | [−0.9985, −0.9915] |
| **Product-free** | **3,081** | **232** | **+0.0631** | **0.0242** | **[−0.0107, +0.1408]** |
| All pooled | 6,555 | 401 | −0.0221 | 0.0044 | [−0.0690, +0.0256] |

The product-free slope's interval contains zero. The single sharpest structure in the entire
dataset, the −1 slope, exists **only** in the class where it is arithmetically entailed
(`output/analysis/w6_decoupled/corrected_results.json`).

**Model comparison** (`logER` against 13 competing specifications,
`output/analysis/w0_main/table_model_comparison.csv`):

| Model | n | k | R² | AIC |
|---|---|---|---|---|
| `~ 1` (null) | 6,555 | 1 | 0.0000 | 16,475.7 |
| `~ logTI` | 6,555 | 2 | 0.0044 | 16,448.7 |
| `~ C(Methodology)` | 6,555 | 5 | 0.0420 | 16,202.4 |
| `~ C(Process)` | 6,555 | 2 | 0.0689 | 16,009.7 |
| `~ logS + logP + logA + Latitude` | 6,450 | 5 | 0.2127 | 14,603.2 |
| `~ logTI * C(Methodology)` | 6,555 | 10 | **0.5666** | **11,013.4** |
| `~ C(Reference)` | 6,555 | 401 | 0.6907 | 9,584.0 |

Timescale alone improves the null by 0.44 % of variance. Method × timescale reaches 0.567 but does
so almost entirely through the product-locked strata (Finding 1). Geology-only predictors
(slope, precipitation, area, latitude) reach 0.213 with no timescale term at all. Adding `logTI`
last to a model already containing method, process and geology raises R² by only 0.0124
(`output/analysis/w2_variance/incremental_r2.csv`). The largest single explanatory factor is which study the row came from
(`C(Reference)` alone: R² = 0.691; ICC within `Reference` = 0.726).

**Counterevidence and limitation.** The naive pooled slope is nominally "significant"
(p = 7 × 10⁻⁸) under classical standard errors. That significance is an artefact of treating 6,555
clustered rows as independent: the same coefficient under `Reference`-clustered errors gives
p = 0.34. Within the product-free class, precipitation (marginal R² = 0.174) and slope
(0.097) both outrank timescale (0.018). A true scale dependence could exist and be masked here,
because the product-free class is a residue of heterogeneous methods rather than a clean
experimental design. The claim is a bounded null: *over the timescale range this compilation
actually samples, with method held fixed and clustering respected, no stable rate–timescale scaling
is recoverable.*

**Executable falsification.** Fit a hierarchical model with study-level random intercepts and
measurement-error weighting using the populated `Erosion rate min`/`max` columns — not done here.
If a nonzero population-level timescale exponent emerges with a cluster-robust interval excluding
zero, Finding 2 is refuted.

**Evidence.**
`output/analysis/w6_decoupled/decoupled_corrected.py` ·
`output/analysis/w6_decoupled/corrected_results.json` ·
`output/analysis/w6_decoupled/table_class_slopes.csv` ·
`output/analysis/w0_main/main_analysis.py` ·
`output/analysis/w0_main/table_model_comparison.csv` ·
`output/analysis/w2_variance/w2_variance.py` ·
`output/analysis/w2_variance/results.json` (`no_stable_relation_baseline`, `simpson_reversal_check`)

---

## Finding 3 — Measurement method, not process category, is the primary control on reported rate

**Object and scope.** All 6,711 rows carrying a rate; the competing explanations of no relation,
smooth scale dependence, method stratification, and process category.

**Relationship and direction.** Method-stratified. Median reported rate differs by ~2 orders of
magnitude across methods (glacial `Thermochronometric` 1.50 mm/yr vs `Cosmogenic surface`
0.40 mm/yr; nonglacial `Relief` 19.0 vs `Cosmogenic surface` 0.0006 mm/yr), and method carries
more explanatory weight than process category at every level of adjustment.

**Core quantitative result.** Variance decomposition on log₁₀(rate), n = 6,711
(`output/analysis/w2_variance/model_comparison_primary.csv`):

| Grouping | R² |
|---|---|
| `C(Methodology)` alone | 0.1495 |
| `C(Process)` alone | 0.0649 |
| `C(Methodology) + C(Process)` | 0.2476 |
| `C(Methodology) × C(Process)` | 0.2932 |
| `C(Compilation)` | 0.2672 |
| `C(Reference)` | 0.6907 |

Method alone explains 2.3× what process alone does. Adding method last to a model containing
`logTI` and process raises R² by 0.103; adding process last to a model containing `logTI` and
method raises R² by 0.120 — both large, but method is the larger single categorical factor once
the interaction is accounted for, and the method × process interaction (R² = 0.293) exceeds their
additive combination (0.248), meaning the effect of process depends on which method was used.

**Robustness.** Under HC3 robust errors, three of nine strata change their classification against
the −1 and 0 references; `Glacial Relief` moves from "intermediate" to "consistent with 0". The
method ranking itself is unchanged under all robustness variants
(`output/analysis/w3_scaling/results.json`, `consistency_robust_estimators`).
Leave-one-compilation-out refits are in
`output/analysis/w3_scaling/tables/leave_one_compilation_out_summary.csv`.

**Counterevidence and limitation.** Method is confounded with timescale, and the confounding is
severe: `Cosmogenic detrital` has median interval 10⁴·⁰ yr while `Volumetric` has 10⁰·⁷ yr, and
`Thermochronometric` 10⁶·⁰ yr. Across methods, median log₁₀(interval) spans 5.9 dex. So "method"
partly encodes *which timescales that method can access*. Method and compilation are also nearly
collinear in places — the entire `GloRiSe` and `Chen et al., 2022` contributions are `Volumetric`,
and the entire `OCTOPUS` contribution is `Cosmogenic detrital` — so `C(Methodology)` and
`C(Compilation)` are partly the same factor under two names. The defensible claim is the ranking
*method > process*, not a claim that method is a clean causal factor.

**Executable falsification.** Restrict to the one stratum where several distinct compilations and
both processes are represented at overlapping timescales — nonglacial `Volumetric` contains
`Chen et al., 2022` (1,521), `GloRiSe` (1,057), `This study` (5) — and re-test the method ranking
with compilation as a nested random effect. If process recovers primacy, Finding 3 is refuted.

**Evidence.**
`output/analysis/w2_variance/w2_variance.py` ·
`output/analysis/w2_variance/model_comparison_primary.csv` ·
`output/analysis/w2_variance/incremental_r2.csv` ·
`output/analysis/w3_scaling/results.json` ·
`output/analysis/w0_main/table_coverage_by_stratum.csv`

---

## Finding 4 — The glacial–nonglacial contrast is real at matched timescale but reverses in two method strata

**Object and scope.** 427 glacial and 6,288 nonglacial estimates; whether glacial processes
erode faster, and under what conditions that statement holds.

**Relationship and direction.** Conditional. Glacial rates exceed nonglacial rates at matched
method and matched timescale in three of five method strata, and **reverse** in two.

**Core quantitative result.**

*Marginal:* glacial median 0.55 mm/yr vs nonglacial 0.061 mm/yr.

*Scale-matched* (each glacial row matched to its 5 nearest nonglacial neighbours in log₁₀ interval
within the same method, difference in dex, one-sample t-test and cluster bootstrap):

| Methodology | n glacial | n comparator pool | mean difference (dex) | ratio | p | cluster-bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| Cosmogenic surface | 99 | 64 | +1.867 | **73.7×** | 3.0 × 10⁻⁴³ | [+1.715, +2.006] |
| Cosmogenic detrital | 9 | 3,470 | +0.582 | 3.8× | 0.024 | [+0.209, +0.968] |
| Volumetric | 233 | 2,583 | +0.252 | 1.8× | 2.2 × 10⁻⁴ | [+0.122, +0.382] |
| Thermochronometric | 68 | 4 | −0.482 | 0.33× | 4.6 × 10⁻⁸ | [−0.635, −0.340] |
| **Relief** | 18 | 7 | **−1.019** | **0.096×** | 0.0022 | [−1.497, −0.481] |

Pooled across methods: +0.463 dex (**2.9×**), p = 1.2 × 10⁻¹³
(`output/analysis/w4_process/scale_matched_knn5.csv`).

**The 73.7× ratio is the least trustworthy number in this table, and the reason is checkable.**
The cosmogenic-surface comparator pool is only 64 rows and **100 % of them lie poleward of 60°**
(median |latitude| 77.3°, Region `Antarctica` for all 64: 34 from `Marrero et al., 2018`, 30 from
`This study`), whereas only 37 % of the matched glacial rows do (median |latitude| 49.0°). Antarctic
cosmogenic-surface rates are near-zero by construction — cold, arid, polar-desert landscapes with
almost no denudation, median 0.00006 mm/yr. The ratio therefore contrasts polar nonglacial sites
with mid-latitude glacial sites, and latitude rather than process may carry most of it. For the same
reason the `Thermochronometric` reversal rests on just 4 nonglacial comparators. The two ratios
that rest on a genuinely populated comparator pool — `Cosmogenic detrital` (9 vs 3,470) and
`Volumetric` (233 vs 2,583) — are 3.8× and 1.8×, an order of magnitude smaller than the headline
figure. The defensible summary is a ratio of **roughly 2–4×** at matched scale and method, not 74×.

*Common-support check.* Restricting to the interval range both processes actually sample
(10⁻¹·⁷²–10⁴·²⁰ yr), the volumetric comparison gives glacial median 0.518 vs nonglacial
0.087 mm/yr, a 5.96× ratio, Mann-Whitney p = 1.3 × 10⁻³²
(`output/analysis/w0_main/table_common_support_process.csv`). The direction is unchanged.

**The reversal.** In `Relief` (nonglacial faster by 10.4×) and `Thermochronometric` (nonglacial
faster by 3.0×), the sign flips. Breaking the glacial set by `Type`
(`output/analysis/w4_process/reversal_strata.csv`):

| Method | Glacial Type | n | difference (dex) | direction |
|---|---|---|---|---|
| Relief | Alpine | 6 | −0.716 | nonglacial faster |
| Relief | Continental | 10 | −1.767 | nonglacial faster |
| Relief | Outlet ice stream | 2 | +1.813 | glacial faster |
| Thermochronometric | Alpine | 57 | −0.429 | nonglacial faster |
| Thermochronometric | Continental | 5 | −1.615 | nonglacial faster |
| Volumetric | Alpine | 59 | +0.775 | glacial faster |
| Volumetric | Continental | 50 | −0.110 | null (p = 0.30) |
| Volumetric | High-latitude | 67 | +0.038 | null (p = 0.76) |

So the contrast is not simply "glacial > nonglacial". Within `Volumetric`, `Continental` and
`High-latitude` glacial types show no difference from fluvial nonglacial rates at all, while
`Alpine` shows a clear excess. The pooled ratio is carried by a subset of method/type combinations.

**Robustness.** Cluster-robust standard errors (by glacial `Reference`) are reported alongside the
classical ones and are 2–3× larger; the `Volumetric` result survives (p = 0.070 clustered vs
0.0002 classical — a material weakening), while `Cosmogenic surface` (p ≈ 2 × 10⁻¹⁹ clustered)
and `Thermochronometric` (p = 0.056 clustered) bracket the significance boundary from opposite
sides. `Thermochronometric`'s reversal is therefore **not** robust to clustering and should be
treated as unresolved. The `Relief` reversal involves only 18 glacial rows from 15 references and
should be treated as a boundary flag rather than an established effect.

**Counterevidence and limitation.** (i) The comparator pools are thin and geographically biased:
apart from the two large strata, they are 64 rows (100 % Antarctic) and 7 rows. (ii) The
comparison is confounded with compilation: `This study` contributes 294 of 427 glacial rows and
its median log₁₀(rate) is 0.349 dex *lower* than the inherited glacial rows, while its 48
nonglacial rows are 1.091 dex lower than inherited nonglacial rows — a compiler effect large
enough to move any marginal contrast. (iii) Glacial and nonglacial timescale distributions differ
(Mann-Whitney p = 1.3 × 10⁻¹⁴). (iv) `Type` is nested within process by construction, so type-level
results cannot be compared across processes without the matched design used above.
(v) Alternative controls (area, slope, precipitation, latitude) are reported in
`output/analysis/w4_process/results.json` under `part6_controls`; the contrast survives them only
in the strata with populated comparator pools.

**Executable falsification.** The `Nonglacial Cosmogenic surface` stratum (n = 224) is largely
Antarctic (Region `Antarctica`, 224 rows) while the glacial cosmogenic-surface rows are
mid-latitude; re-run the matched comparison restricted to |latitude| > 60° on both sides. If the
73.7× ratio collapses, the headline contrast is a latitude confound rather than a process effect.

**Evidence.**
`output/analysis/w4_process/w4_process_analysis.py` ·
`output/analysis/w4_process/scale_matched_knn5.csv` ·
`output/analysis/w4_process/reversal_strata.csv` ·
`output/analysis/w4_process/results.json` ·
`output/analysis/w0_main/table_common_support_process.csv` ·
`output/analysis/w0_main/table_compiler_effect.csv`

---

## Finding 5 — Non-independence, not sample size, is the binding constraint on every claim above

**Object and scope.** The whole compilation; the inferential design rather than the geology.

**Relationship and direction.** Bounding. The effective information content is an order of
magnitude below the nominal row count, and this limit propagates into every finding above.

**Core quantitative result.** From 6,711 rows carrying a rate:

| Quantity | Value |
|---|---|
| Nominal rows | 6,711 |
| Distinct `Reference` | 417 |
| Distinct `Compilation` | 17 |
| Distinct coordinate pairs | 5,114 |
| ICC of log₁₀(rate) within `Reference` | **0.726** |
| Design effect (mean cluster size 16.09) | **11.95** |
| **Effective sample size** | **≈ 560** |
| SE inflation factor | 3.46 |
| ICC within `Compilation` | 0.356 |

(`output/analysis/w2_variance/results.json`, `clustering`.)

Three consequences that change conclusions:

1. The pooled rate–interval slope is significant classically (p = 7 × 10⁻⁸) but **not** under
   `Reference`-clustered errors (p = 0.34).
2. The marginal glacial/nonglacial difference, significant at p = 1.2 × 10⁻¹³ with row-level
   bootstrap, has its strongest stratum (`Thermochronometric`) fall to p = 0.056 under clustering.
3. `C(Reference)` is the single best predictor in the dataset (R² = 0.691), exceeding every
   physical variable. Between-study variance dominates between-site variance.

**Robustness.** The design effect is computed two ways (`m̄` and `n₀` estimators): 11.95 and 11.80,
agreeing to 1.3 %. Cluster bootstraps (2,000 Reference-resamples) are reported for all headline
slopes in `w3_scaling` and `w6_decoupled`.

**Counterevidence and limitation.** This is a property of the compilation as received, not a
defect introduced by the analysis, and it cuts against *all* findings including the null in
Finding 2 — a null under low effective n is weak evidence of absence. It does not invalidate
Finding 1, whose evidence is arithmetic (an exact identity, reproducible study-by-study) rather
than statistical. It also does not invalidate Finding 3's method ranking, which is a large
between-stratum effect visible within single references.

**Executable falsification.** Re-run the pooled regression with the `C(Reference)` grouping
replaced by a nested random intercept for `Locality` within `Reference` at multiple coordinate
rounding scales (±3 decimal places). The effective-n estimate of ≈ 560 should be stable; if
spatial clustering within studies is negligible, the design effect should fall toward 1 and this
finding is refuted.

**Evidence.**
`output/analysis/w2_variance/w2_variance.py` ·
`output/analysis/w2_variance/results.json` (`clustering`, `simpson_reversal_check`) ·
`output/analysis/w3_scaling/results.json` (`large_strata_robustness`, cluster bootstraps) ·
`output/analysis/w5_audit/table_independence_structure.csv` ·
`output/analysis/common/parse_audit.json`

---

## Alternative explanations considered and their disposition

| Candidate explanation | Disposition | Decisive evidence |
|---|---|---|
| **No stable relation** | **Supported** as the pooled description | Pooled slope −0.022, R² = 0.004, clustered CI [−0.068, +0.023] |
| **Smooth scale dependence** (a single exponent) | **Not supported** | A single slope is rejected against method-specific slopes (ΔR² = 0.475 when the method interaction is added). The apparent −1 applies only to product-locked strata |
| **Method-stratified dependence** | **Supported, and the strata split into two populations** | Method × interval R² = 0.567 vs 0.004 for interval alone; after removing product-locked strata the slope is +0.063 with a CI spanning zero |
| **Process-category dependence** | **Conditional** — supported at matched scale in 3 of 5 methods, reversed in 2 | Matched ratios from 73.7× down to 0.096× across strata |
| **Mathematical coupling** | **Confirmed as the dominant specific mechanism** | Exact identity in 28/92 testable rows; thickness pinned at 650.4 mm (0.033 dex) across 5.75 dex of timescale; median within-study slope −1.0037 over 115 references |
| **Compiler/study identity** | **Confirmed as a large competing factor** | `C(Reference)` R² = 0.691; ICC = 0.726; "This study" rows differ from inherited rows by 0.349 (glacial) and 1.091 dex (nonglacial) |
| **Geological controls (slope, precipitation, area, latitude)** | **Real but secondary to study identity** | Geology-only R² = 0.213; precipitation marginal R² = 0.174 in product-free strata |

The three-way separation requested by the research objective — geological process, observation
window, and mathematical coupling — resolves as follows: the **observation window** contributes
almost nothing once method is fixed; **mathematical coupling** accounts for the single largest
apparent scaling signal; **geological process** produces a real but conditional contrast
(Finding 4) that is confounded by compilation and latitude.

---

## Reproduction

A full reproduction guide is in **`output/analysis/REPRODUCE.md`**, covering environment, exact
execution order, seeds, and the analysis-unit conventions. In brief, from the project root:

```bash
python3 output/analysis/common/prepare_data.py       # shared consolidated frame
python3 output/analysis/w0_main/main_analysis.py     # coverage, model comparison, stratum slopes
python3 output/analysis/w0_main/decisive_tests.py    # falsification tests
python3 output/analysis/w1_coupling/coupling_audit.py
python3 output/analysis/w2_variance/w2_variance.py
python3 output/analysis/w3_scaling/w3_scaling.py
python3 output/analysis/w4_process/w4_process_analysis.py
python3 output/analysis/w5_audit/audit_data.py
python3 output/analysis/w6_decoupled/decoupled_corrected.py
```

All scripts are deterministic (seeds recorded in each `results.json`). Failed and superseded
attempts, including a sub-agent transport failure and a mislabelled classification that was
detected and corrected, are recorded in **`output/analysis/FAILURES.md`** rather than discarded.

## Provenance and external sources

Data provenance, file hashes and the integrity statement are in
**`output/provenance/DATA_SOURCE_LOG.md`**. No external scientific dataset was downloaded,
substituted, or merged; no literature search was performed; the analysis is self-contained within
the frozen payload. The one interpretive relation applied (`E = P·Λ/(ρ·N)` for spallation
equilibrium) is derived inside this report from workspace columns and tested empirically against
them, so no pre-`T0` external source needs to be cited for it.

## Limitations

1. **Effective sample size ≈ 560**, not 6,711. Every interval reported here is explicitly row-level
   or cluster-level; the cluster-level intervals are the honest ones.
2. **The coupling test is available for only 92 rows** because `Exhumation (m)` is populated for
   81 of 427 glacial and 11 of 6,288 nonglacial rows. The artefact conclusion rests mainly on the
   thickness-spread and reconstruction evidence, not on that column.
3. **`Erosion rate min`/`max` are unused**; all fits are unweighted. A measurement-error-weighted
   exponent remains an open extension.
4. **The reversing strata are small** (`Relief`: 18 glacial rows / 15 references;
   `Thermochronometric`: 68 glacial rows / 18 references, with only 4 nonglacial rows). They are
   reported as boundaries on Finding 4, not as established effects.
5. **Method and compilation are partly collinear**, so Finding 3's ranking should be read as
   "method-as-instantiated-in-these-compilations" rather than as a clean causal partition.
6. **The null in Finding 2 is a bounded null.** It says no scaling is recoverable from *this*
   compilation with *these* methods — not that erosion rates are timescale-independent in nature.
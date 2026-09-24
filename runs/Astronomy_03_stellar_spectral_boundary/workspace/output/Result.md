# Wavelength- and Temperature-Dependent Spectral Response Boundaries

**Task:** `Astronomy_03_stellar_spectral_boundary` · **T0:** 2023-01-01
**Authoritative report.** This file is the sole authority for the registered findings.
Evidence paths given below are relative to the workspace root.

---

## 1. Summary

The workspace contains two bundled tables: a photon-flux grid over three effective-temperature
slices, 35 ordered composition (metallicity) index levels and 176 wavelengths, and a separate
five-column normalised response series. Four findings are registered.

The dominant scientific result is that **both** responses in the photon-flux grid reverse sign as a
function of wavelength, and that the two reversals sit at different places for different reasons.
The composition response reverses near 440 nm and is of enormous amplitude in the ultraviolet
(several orders of magnitude) but only a few percent in the optical. The temperature response
reverses near 620-650 nm and is largest in the far-ultraviolet. Working these out required
separating a genuine difference between the two temperature slices from a definitional artefact:
the apparent 39 nm spread in the composition crossover between the 5300 K and 6300 K slices is
statistically real, but the 6300 K value is definition-dependent because that slice has three sign
changes rather than one.

The second result concerns the independent normalised series. It is **not** independent of the
grid: it is the T = 5800 K slice at composition indices −0.90, −0.50, 0.00, +0.50, +0.90, converted
from photon flux to energy flux by multiplication by wavelength, to within the 3-significant-figure
rounding floor of the bundled flux values. This was established by a zero-free-parameter ratio test
and independently confirmed by two predictions the hypothesis makes but that were not used to fit it
(peak positions and crossover locations). The conversion holds only above ~330 nm; below that a
large, discrete, composition-blind factor appears, which is an interval-weighting property of the
data product rather than a physical effect. Its functional form — a single constant divided by the
local grid sampling step — is identified; the physical provenance of that constant is not, and two
specific mechanisms proposed during this analysis were tested and refuted.

---

## 2. Data and analysis unit

### 2.1 Inventory and integrity (`output/analysis/stage1_inventory.py` → `output/results/stage1_inventory.json`)

Both files match the SHA-256 hashes declared in `data_manifest.json`
(`output/results/stage1_inventory.json` → `integrity`). The flux table is a **complete regular grid**:
18 480 rows = 3 temperatures × 35 composition levels × 176 wavelengths, with no missing cell, no
duplicate, no non-finite value and no non-positive flux.

| Property | Value |
|---|---|
| Effective temperature slices | 5300, 5800, 6300 K |
| Composition index levels | 35, from −1.00 to +0.90 |
| Composition ladder | 0.05 steps over −1.00…+0.50 (31 levels), then 0.10 steps to +0.90 |
| Wavelength range | 179.4 – 850.0 nm |
| Wavelength sampling | **non-uniform**: 5.0 nm for 104 of 175 intervals; 22 intervals of 1.0 nm inside 299–326.5 nm; 1.5–4.3 nm steps between |
| Normalised series | 5 columns × 175 rows, 179.31 – 849.77 nm |

Three sampling facts govern everything downstream and are the reason the robustness checks below
are shaped the way they are:

1. **Wavelength samples are strongly autocorrelated.** The mean lag-1 autocorrelation of
   `log(flux)` along wavelength is **0.972**, still **0.55** at lag 20
   (`output/results/stage1_inventory.json` → `wavelength_axis_correlation`). Neighbouring
   wavelengths are correlated scan points, exactly as `DATA_GUIDE.md` warns. Random-point validation
   is therefore invalid and is *demonstrated* to be invalid in §5.2.
2. **The composition ladder is thinned above +0.50** (0.10 steps; 0.55/0.65/0.75/0.85 absent), so all
   primary fits use only the uniformly spaced 31-level region M ≤ +0.50.
3. **Flux values carry only 3 significant figures.** This sets a hard floor on any resolvable
   relative deviation (quantified in §5.4) and is the single most important limitation in this report.

### 2.2 Analysis unit

One complete temperature-by-wavelength spectrum, or one complete ordered response series. All
headline numbers are properties of whole spectra or whole series; individual wavelength points and
individual composition settings are never treated as independent replicates.

---

## 3. Registered findings

### Finding 1 — The composition response reverses sign at a shared optical boundary, and the reversal is reproduced independently by every temperature slice

**Object and scope.** Photon flux `photon_flux_cm2` as a function of the ordered composition index
`metallicity_index`, at fixed effective temperature and wavelength, across all three temperature
slices and the full 179.4–850 nm range.

**Relationship and direction.** The sign of `∂ log F / ∂ M` depends on wavelength. Blueward of the
boundary, increasing the composition index **lowers** the flux (by several orders of magnitude in the
far-ultraviolet). Redward of the boundary it **raises** the flux, but only weakly.

**Core quantitative result** (analysis unit: one complete temperature-by-wavelength spectrum;
uniform 31-level ladder M ≤ +0.50; OLS slope of `ln F` on the index):

| T (K) | First sign change (nm) | ∂log F/∂M at 180 nm | median ∂log F/∂M, λ ≥ 500 nm | median R² | permutation-null median R² |
|---|---|---|---|---|---|
| 5300 | 448.7 | −2.00 | +0.044 | 0.9844 | 0.0138 |
| 5800 | 445.8 | −3.00 | +0.060 | 0.9899 | 0.0197 |
| 6300 | 409.9 | −2.76 | +0.072 | 0.9901 | 0.0160 |

Source: `output/results/stage2_boundaries.json`; `output/results/stage6_statistical_power.json`.

The sign flip is not marginal: the slope is negative at **every** wavelength in 250–400 nm and
positive at essentially every wavelength above 460 nm, in all three slices.

**Supporting analysis and result evidence.**
- `output/analysis/stage2_boundaries.py` → `output/results/stage2_boundaries.json` (primary numbers)
- `output/analysis/stage6_statistical_power.py` → `output/results/stage6_statistical_power.json`
  (association strength vs null, cross-slice reproduction)
- `output/analysis/stage4_robustness.py` → `output/results/stage4_robustness.json` (robustness)
- Independent recomputation: `output/verification/D1_metallicity_crossover.json`,
  `output/verification/GAP3_artefact_nulls.json`,
  `output/verification/GAP4_interaction_and_uncertainty.json`
- Figure: `output/figures/fig1_wavelength_boundaries.png` (panel b)

**Robustness.**
- **Contiguous-band hold-out, not random points.** The crossover is re-estimated inside held-out
  wavelength windows and confirmed to lie inside 400–500, 420–470 and 430–460 nm but not 480–560 nm,
  as the boundary requires (`output/results/stage4_robustness.json` → `R2_holdout`). The three slices
  are additionally mutually independent replicates of the same wavelength profile: pairwise
  correlations of the full slope profile are **0.949** (5300 vs 5800 K), **0.896** (5300 vs 6300 K)
  and **0.987** (5800 vs 6300 K) (`stage6_statistical_power.json` →
  `cross_slice_profile_reproduction`). Each slice reproduces the other's boundary without being
  fitted to it.
- **Association strength.** Median R² of the ordered index against `log F` is 0.984–0.990 against a
  permutation-null median of 0.014–0.020; no permutation of 300 reached the observed value, so the
  p-value is floor-limited at < 1/300 (`stage6_statistical_power.json` → `r2_vs_permutation_null`).
  *Caveat:* this permutation destroys the ladder *ordering*, so it calibrates the association, not the
  crossover location. A permutation p-value cannot resolve below 1/300 and is reported as an
  inequality rather than as an exact zero.
- **Leave-one-temperature-out.** Dropping any single slice still recovers a crossover at
  434.9–447.3 nm — a spread of 12.4 nm, which is 2.5 grid intervals
  (`stage4_robustness.json` → `R1_leave_one_temperature_out`). This supports "reproducible in
  location to within a few grid intervals", not exact reproduction.
- **Alternative response definition.** A rank-based monotone trend reproduces the same boundaries
  (447.8 / 446.8 / 409.5 nm) (`stage4_robustness.json` → `R3_rank_based_alternative`).
- **Threshold sensitivity.** Widening the composition window from M ≤ 0.0 to M ≤ +0.50 moves the
  5800 K boundary only from 444.6 to 445.8 nm (`stage4_robustness.json` → `R4_metallicity_window`).
- **Wavelength resampling.** Resampling onto a uniform 5 nm ladder moves the boundary by 0.00 nm;
  decimating to every 2nd or 4th point moves it by at most 19.6 nm, bounded by the coarser retained
  interval (`output/verification/GAP3_artefact_nulls.json` → `claim_i...wavelength_null_N2/N3`).
- **Single-rung jackknife.** Dropping any one composition level moves the boundary by ≤ 0.12 nm at
  5300/5800 K; random 16-of-31 monotone sub-ladders keep the median within 0.2 nm at all three
  temperatures (`GAP3_artefact_nulls.json` → `monotone_preserving_null`).
- **Level bootstrap.** 95% intervals: 5300 K [448.25, 449.08], 5800 K [445.45, 446.18],
  6300 K [409.02, 434.54] nm (`GAP4_interaction_and_uncertainty.json` →
  `level_bootstrap_metallicity_crossover`).

**Counterevidence, limitations, and alternative explanations.**
- The **label-permutation null is uninformative here and is not used as a p-value.** The composition
  climatology is rank-1 (PC1 ≥ 98.9 % of level-space variance), so permuted labels reproduce a
  slope profile ~0.9-correlated with the observed one about half the time, and the null crossing
  distribution is a broad amplitude-edge mode spanning roughly 185–515 nm
  (`GAP3_artefact_nulls.json` → `claim_i...plain_label_permutation_null`). This was verified
  independently during this run and is the reason the finding rests on the *association strength*
  and *cross-slice reproduction* rather than on the crossing location alone.
- The **crossing location is definition-dependent.** The 6300 K slice has **three** sign changes
  between 179 and 850 nm, not one. The first neg→pos rule gives 409.9 nm; a persistence rule gives
  434.4 nm. The 5300 K slice likewise has five sign changes. Only the 5800 K slice has exactly one.
  The headline "shared boundary" is therefore a statement about the *band* over which the sign flips,
  not about a unique wavelength.
- The 5300 K and 6300 K crossovers are **statistically distinguishable** — the paired bootstrap
  difference is +38.8 nm with 95% interval [+14.3, +39.3] nm, and the two slices differ in 100% of
  4 000 paired resamples (`GAP4_interaction_and_uncertainty.json` →
  `paired_bootstrap_differences_metallicity_crossover`). A joint model finds a significant
  temperature × composition interaction (F = 6.50 on 1 and 89 df, p = 0.0125; interaction
  coefficient +12.7 nm per 1000 K per unit index, 95% interval [+6.7, +18.8] nm)
  (`GAP4_interaction_and_uncertainty.json` → `joint_model.local_window_slope`). So the boundary is
  **not** rigorously temperature-invariant; the honest statement is that all three slices place it in
  the 410–450 nm region, with the 6300 K value systematically bluer and definition-sensitive.
- **Mechanism interpretation is not established by these data.** The data are radiation calculations;
  the wavelength dependence of the composition effect is consistent with ultraviolet line blanketing
  and back-warming redistributing flux (background: Gustafsson et al. 2008, Coelho et al. 2005 — see
  `output/SOURCES.md`), but nothing in this dataset isolates that mechanism. `DATA_GUIDE.md`
  explicitly warns these observations cannot establish causality, and this report does not claim it.

**Falsification test.** Fit the same per-wavelength slope on a temperature slice absent from this
file (e.g. 5000 K or 7000 K). The finding requires a negative slope at 300 nm and a positive slope at
700 nm. A positive 300 nm slope in a neighbouring slice would refute the shared-boundary claim.

---

### Finding 2 — The temperature response also reverses sign, but in the red, and the reversal is not monotone in composition

**Object and scope.** The ratio `F(6300 K) / F(5300 K)` as a function of wavelength, across all 35
composition levels.

**Relationship and direction.** The ratio is **> 1** in the ultraviolet and **< 1** in the red: hot
stars are brighter in the ultraviolet and fainter in the red, at these effective temperatures.

**Core quantitative result** (analysis unit: one complete temperature-by-wavelength spectrum):

| Composition index | r at 180 nm | r at 250 nm | r at 400 nm | r at 600 nm | r at 700 nm | r at 850 nm |
|---|---|---|---|---|---|---|
| −1.00 | 183.0 | 5.19 | 1.574 | 1.016 | 0.931 | 0.856 |
| +0.00 | 156.8 | 8.64 | 1.701 | 1.038 | 0.957 | 0.872 |
| +0.90 | 119.0 | 16.47 | 1.845 | 1.060 | 0.985 | 0.882 |

All 35 levels cross 1.0 between **618.4 and 651.7 nm**. Source:
`output/verification/D2_temperature_reversal.json` → `primary_numbers`.

**Supporting analysis and result evidence.**
- `output/analysis/stage2_boundaries.py` → `output/results/stage2_boundaries.json`
- Independent recomputation: `output/verification/D2_temperature_reversal.json`
- Adversarial refutation: `output/verification/D2_refutation.json`
- Null tests: `output/verification/GAP3_artefact_nulls.json` → `claim_ii_temperature_crossover`
- Figure: `output/figures/fig1_wavelength_boundaries.png` (panel c)

**Robustness.**
- The **long-wavelength deficit is real and exceeds the noise.** Averaged over 800–850 nm the ratio is
  0.856–0.888 depending on composition, while the mean absolute step-to-step scatter in that band is
  only 0.003–0.006; the deficit exceeds the scatter in every level
  (`D2_temperature_reversal.json` → `primary_numbers.long_wavelength_deficit`).
- **Both adjacent temperature pairs behave the same way.** F(5800)/F(5300) crosses near 652.5 nm and
  F(6300)/F(5800) near 595.0 nm (`stage2_boundaries.json` → `adjacent_temperature_crossovers`), so
  the reversal is a property of the temperature sequence, not an artefact of the extreme pair.
- **Wavelength resampling and decimation leave the endpoint intact.** The ratio at the longest
  retained wavelength stays below 1 in every case and every level, under uniform 5 nm resampling and
  under 2×/4× decimation (`GAP3_artefact_nulls.json` → `claim_ii...wavelength_null_N3`).
- **A blackbody baseline cannot produce the reversal.** Fitting a Planck function per temperature
  slice: a nominal 5300/6300 K pair gives 11.0 at 180 nm (observed median 183) and **never** drops
  below 1 — the Rayleigh–Jeans asymptote is 1.189. Every one of the 35 fitted Planck-pair curves stays
  above 1 across the whole band. The reversal therefore requires atmospheric structure, not just the
  Planck shape (`D2_refutation.json` → `blackbody_baseline_test`).

**Counterevidence, limitations, and alternative explanations.**
- **The composition dependence of the crossing is NOT established.** The crossing moves redward with
  24.7 nm per unit index, but this is a zero of a function sampled every 5 nm in that region: 15 of
  35 crossings land **exactly** on a grid node, only 24 distinct values occur, and the fit's residual
  standard deviation (3.7 nm) is comparable to the total effect. An adversarial refutation concluded
  the metallicity dependence is not separable from the quantisation noise floor
  (`D2_refutation.json` → `claim_component_D_crossing_moves_redward_with_M`, `survives: false`).
  A bootstrap puts the slope at +20.7 nm per unit index with 95% interval [+12.7, +28.5] nm
  (`D2_temperature_reversal.json` → `crossing_monotonicity`), but given the quantisation this
  interval should be treated as optimistic.
- The **"order 100× at 180 nm"** figure holds as a *median* (183×); the ladder spans 85–279×, and
  7 of 35 levels fall below 100×.
- The trend is **not strictly monotone**: a backward step of 1.6 nm occurs at the lowest index.
- **No data exist beyond 850 nm.** The file ends there. Nothing can be said about whether the deficit
  continues, steepens or turns over — this is an explicit boundary of the evidence, not a null result.

**Falsification test.** Extend the wavelength axis redward of 850 nm for one temperature pair. The
finding requires F(6300)/F(5300) to remain below 1 and the deficit to persist. A recovery to > 1
within, say, 900–1000 nm would refute the "reversal" framing.

---

### Finding 3 — The normalised response series is a photon-to-energy transform of the 5800 K temperature slice, not an independent series

**Object and scope.** The relationship between `normalized_spectral_response.tsv` (five columns,
175 wavelengths) and the photon-flux grid.

**Relationship.** The five columns correspond to the **T = 5800 K** slice at composition indices
**−0.90, −0.50, 0.00, +0.50, +0.90**, related to the grid by
`response_level_k(λ) · λ  ∝  F(T = 5800 K, M_k, λ)`.

**Core quantitative result.**

| Quantity | Value |
|---|---|
| Shared temperature | 5800 K (all five levels) |
| Composition assignment | −0.90, −0.50, 0.00, +0.50, +0.90 |
| Proportionality constant | 3.9742 × 10⁻¹³ |
| Relative spread of that constant over 5 × 121 points | 1.30 × 10⁻³ |
| Pooled median relative deviation at 5800 K | 1.03 × 10⁻³ |
| Pooled median relative deviation at 5300 K | 7.72 × 10⁻² |
| Pooled median relative deviation at 6300 K | 6.56 × 10⁻² |

Source: `output/verification/D3_crossfile_relation.json`.

**Supporting analysis and result evidence.**
- `output/analysis/stage3_crossfile.py` → `output/results/stage3_crossfile.json`
- Independent recomputation: `output/verification/D3_crossfile_relation.json`
- Non-circularity test: `output/verification/GAP1_calibration_circularity.json`
- Adversarial refutation: `output/verification/D3_refutation.json`
- Figure: `output/figures/fig2_crossfile_relation.png`

**Robustness.**
- **Zero-free-parameter ratio test.** If the assignment is right, `level_k / level_1` must equal
  `F(5800, M_k) / F(5800, M_1)` with no adjustable constant. Pooled mean relative deviation:
  **0.0012 at 5800 K**, versus **0.029 at 5300 K** and **0.023 at 6300 K** — a factor 20–25 worse
  (`D3_crossfile_relation.json` → `decisive_interp_matched`).
- **Transform discrimination.** Head to head on the same data, median relative deviations are
  **0.0013** for `level · λ`, **0.219** for `level` alone, and **0.431** for `level / λ`
  (`D3_refutation.json` → `2_transform_head_to_head`). Fitting a free exponent *p* in
  `level · λ^p` selects **p = 1.0** as the optimum, with error rising monotonically on both sides
  (`D3_refutation.json` → `2d_free_exponent_p`).
- **The temperature identification is strongly separated; the metallicity one is not.** Under a
  *fair* null that re-fits the constant separately for every candidate grid curve, the winner is
  unchanged (5800 K, M = 0.00) and is separated from the best other-temperature curve by **31.5×**.
  But the margin over the next-best *neighbouring metallicity* at the same temperature collapses from
  2.24× (unfair, fixed constant) to **1.23×** (fair)
  (`output/results/stage3_crossfile.json` → `photon_to_energy_null`). The temperature assignment is
  therefore well identified; the individual metallicity rungs are only weakly separated, as expected
  for levels that differ by less than the flux quantisation.
- **Quantisation-aware discrimination.** At 5800 K / M = 0.0, 8.7% of points exceed their rounding
  bound for the rank-0 candidate, versus 40.4% with a maximum residual 36× the bound for the rank-1
  neighbour (`D3_refutation.json` → `5_quantisation_aware_violations`).
- **Two predictions not used in fitting agree.** (a) The normalised levels peak at 469.9–479.9 nm,
  which matches the peak of `F/λ` (480 nm at 5300 and 5800 K) rather than the peak of `F`
  (575–605 nm). (b) `level_k / level_1` reverses sign at **443–447 nm**, matching the grid's own
  5800 K composition crossover of **445.8 nm** — with no wavelength factor involved in that test.
- **The calibration is not circular.** Fitting the constant on 500–700 nm *only*, or 400–500 nm
  *only*, or 750–850 nm *only*, reproduces the held-out bands with mean absolute relative deviation
  1.12 × 10⁻³, essentially identical to the in-band value 1.03 × 10⁻³ — an extrapolation penalty of
  ratio 1.09 (`output/verification/GAP1_calibration_circularity.json` → `verdict`).
- **The residual is entirely explained by rounding.** 98.5% of points fall inside their own
  3-significant-figure rounding half-step, and the observed mean deviation (1.085 × 10⁻³) matches the
  uniform-rounding expectation (1.059 × 10⁻³) (`GAP1_calibration_circularity.json` →
  `quantization_floor`).

**Counterevidence, limitations, and alternative explanations.**
- The fit **saturates the data's precision.** Because the residual sits at the rounding floor, no
  finer agreement can be demonstrated from this file, and a hypothetical alternative transform that
  differs from `level · λ` by less than ~0.2% could not be excluded by these data.
- Level 3's winning index is **marginal between −0.05 and 0.00** in the unconstrained shape search,
  because those two adjacent levels differ by less than the quantisation floor; the ratio-based test
  resolves it to 0.00 by using all five levels jointly.
- The correspondence holds **after explicit wavelength realignment.** By raw row index the two files
  do not align (the series has 175 points to the grid's 176, and the arrays are index-shifted past
  index 66). Without realignment the pooled median deviation degrades from 0.0011 to 0.015
  (`D3_refutation.json` → `4_no_realignment_raw_index`). The correspondence is a property of the
  *aligned* data product.
- **The five levels are not five independent series.** A rank analysis gives a participation ratio of
  ~1.01 and entropy rank ~1.03: the five columns lie almost entirely on a one-dimensional locus, as
  they must if they are one temperature slice over one composition axis. They should not be treated
  as five independent replicates (`D3_refutation.json` → `3_rank_and_independence`).
- This is a statement about the **provenance and relationship of the bundled tables**, not about
  stellar physics. It does not identify the upstream source.

**Falsification test.** Obtain the same response series computed at a fourth effective temperature, or
for a composition index not in {−0.90, −0.50, 0.00, +0.50, +0.90}. The finding requires the same
transform to hold with only the per-level constant changing. A series that matches 5300 K or 6300 K
better would refute the 5800 K assignment.

---

### Finding 4 — The photon-to-energy relation breaks below ~330 nm by a large, discrete, composition-blind factor that is a property of the data product, not of the stellar atmospheres

*(The functional form of the factor is identified and its composition-blindness is decisive. The
underlying mechanism is **not** identified, and two candidate mechanisms proposed during this analysis
were refuted — see the counterevidence below.)*

**Object and scope.** The same cross-file relation as Finding 3, evaluated across the full wavelength
range rather than the optical band.

**Relationship and boundary.** The relation holds to ~0.1% above 330 nm and fails blueward of it. The
failure factor is not smooth: it is a **step ladder** whose steps match the flux grid's own wavelength
interval structure.

**Core quantitative result.** Deviation = `level · λ / (c · F)` with `c` fitted on λ ≥ 400 nm only:

| Segment | n points | Measured factor | Predicted = 5 nm / local interval | Relative error |
|---|---|---|---|---|
| 179–296 nm (far-UV ramp) | 44 | 2.071 | 2.045 | 1.8 × 10⁻² |
| 299–322 nm (1.0 nm sampling) | 21 | **5.010** | **5.000** | 2.0 × 10⁻³ |
| 322–327 nm (2.0 nm sampling) | 3 | **2.535** | **2.500** | 1.4 × 10⁻² |
| 330–850 nm (5.0 nm sampling) | 104 | **1.0001** | **1.000** | 1.7 × 10⁻³ |

Source: `output/results/stage5_uv_mechanism.json`; `output/verification/D4_uv_boundary.json`.

**Supporting analysis and result evidence.**
- `output/analysis/stage5_uv_mechanism.py` → `output/results/stage5_uv_mechanism.json`
- Independent recomputation and mechanism test: `output/verification/D4_uv_boundary.json`
- Independent functional-form identification: `output/verification/GAP2_mechanism.json`
- Audit that refuted the summation reading: `output/verification/AUDIT_main_session.md` §7
- Figure: `output/figures/fig2_crossfile_relation.png`

**Robustness.**
- **Model comparison.** Against explicit alternatives, mean absolute relative error is **0.0132** for
  `5 nm / local grid step`, **0.0186** for `5 nm / the series' own step`, **0.642** for a constant
  factor of 5, and **0.799** for a constant factor of 1 (`stage5_uv_mechanism.json` →
  `alternative_models`). The step model wins decisively.
- **Composition-blindness — the decisive test.** The factor is identical across all five levels to a
  relative spread of **3.9 × 10⁻⁴** in the plateau. Those five levels differ in their underlying
  composition sensitivity at 300 nm by a log-slope of −0.427, and by up to ~152× pointwise in the
  ultraviolet. A physical composition effect *cannot* be composition-independent. This is the
  strongest single piece of evidence that the factor is not astrophysical.
- **Not caused by the wavelength misregistration.** The plateau factor stays at 5.00 and the far-UV
  mean at 2.05 under index-shifted, index-unshifted, linear-interpolation and nearest-neighbour
  resampling alike (`D4_uv_boundary.json` → `registration_negative_control`). The one-index shift
  between the files is therefore not what produces the deviation — an interpretation this analysis
  initially proposed and then discarded on this evidence.
- The plateau value 5.010 is within 0.5% of exactly 5.000, and the narrow-segment value 2.535 within
  1.4% of exactly 2.500 — both matching the corresponding grid interval ratios (1.0 nm and 2.0 nm
  against the 5.0 nm optical interval).

**Counterevidence, limitations, and alternative explanations.**
- **The functional form is identified; the mechanism is not.** This distinction was established by
  direct test and is the most important limitation on this finding.
  - *Form is determined.* Dividing the deviation by `1/local step` leaves a constant with no residual
    trend (Pearson r = 0.082 against wavelength; drift 0.009 nm across the whole UV span, interval
    containing zero). On the 66 step-robust ultraviolet rows the normalized value is
    **5.0000 ± 0.036** (block-bootstrap 95% interval [4.995, 5.006]). The free exponent is
    **p = 1.0000**, with 1 inside its bootstrap interval [0.9965, 1.0020]. The `1/step` family beats
    the best constant factor by **83×** in mean relative error on clean rows (0.0046 vs 0.365).
  - *The constant's identity is not determined.* Fitting the numerator from the ultraviolet alone
    gives *N̂* = 5.0011 nm with combined uncertainty 0.0189 nm — a difference of **0.06 combined
    standard deviations** from 5.0 nm, so it is **not** statistically distinguishable
    (`GAP2_mechanism.json` → `free_N`, `distinguishable_from_5nm: false`). Neither table contains a
    binning, resampling or integration operation, so the numerical equality `N` = optical grid step
    is *consistent with* the data but not *forced* by them. No physical interpretation is available
    from two tables alone.
- **Two more specific mechanisms were proposed during this analysis and both were refuted.**
  - *Wavelength misregistration* — refuted: correcting the one-index offset leaves the deviation
    intact (`output/verification/D4_uv_boundary.json` → `mechanism_is_registration_shift: false`).
  - *Bin summation* — refuted twice, by different tests. An auditor's explicit summation conventions
    miss by ~40× (0.53 vs 0.013 mean relative deviation;
    `output/verification/AUDIT_main_session.md` §7a), and a literal node-count model misses by ~38×
    (0.174 vs 0.0046 on step-robust rows) while being **directly contradicted** where it is testable:
    at 322.41 nm three grid nodes lie in a 5 nm window, so a bin sum predicts 3.00 against 2.504
    measured; at 326.41 nm two nodes predict 2.00 against 2.500 measured; and the exact 1 nm versus
    2 nm plateau rows give a ratio of 0.9995 where a node-count reading requires 0.6
    (`GAP2_mechanism.json` → `mechanism`). The width-based `1/step` ratio tracks all of these; the
    node count does not.
  Both refutations are recorded rather than silently folded in, because each implies different
  provenance for the series and a different physical reading of the "boundary".
- **Which file carries the interval cannot be determined** from these two tables alone. The reading
  adopted here — that the normalised series is per-bin while the grid is per-unit-wavelength, or the
  reverse — is a plausible interpretation, not an established fact.
- The narrow 322–327 nm segment rests on **3 points only**.
- The far-UV ramp is the least accurate region (mean relative error 1.8%).
- **The physical readings of Findings 1 and 2 are unaffected** by this artefact: those findings use
  the flux grid alone and never involve the normalised series. This matters, because it means the
  artefact does not propagate into the other registered conclusions.

**Falsification test.** Re-derive the response series from a flux grid on a **uniform** wavelength
ladder. The artefact explanation requires the sub-330 nm deviation to vanish entirely; a physical
explanation requires it to persist, and would additionally require it to vary across composition
settings in proportion to their differing ultraviolet sensitivity.

---

## 4. What was tested and did not support a relationship

Reported because they materially bound the claims above.

1. **A blackbody (Planck) baseline does not explain the spectral shapes.** Comparing each grid
   spectrum to a Planck curve at its nominal effective temperature, the ratio `F / B_λ(T)` varies by a
   factor of **130–1720** across 179–850 nm, and even restricted to the optical 400–850 nm it still
   varies by **2.0–2.6×** (`output/results/stage8_baselines.json` → `planck_baseline`). Since the
   table's absolute unit is arbitrary, this is a test of spectral *shape*: the departure from a pure
   continuum is far larger than any scaling ambiguity, so the spectra carry strong non-continuum
   structure. Separately, **no fitted Planck pair reproduces the temperature-ratio reversal at all** —
   all 35 fitted curves stay above 1 across the whole band
   (`output/verification/D2_refutation.json` → `blackbody_baseline_test`).
2. **Random-point validation is invalid here, and this was demonstrated rather than asserted.**
   The mean lag-1 autocorrelation of `log(flux)` along wavelength is 0.972 (0.55 at lag 20;
   `output/results/stage1_inventory.json`). The decisive test is **leave-out prediction error**:
   holding out 20, 40 or 80 wavelengths and predicting them by interpolation gives a contiguous-block
   error **2.5×, 4.0× and 7.7×** larger than a random hold-out of the same size, with the ratio
   growing as the block grows (`output/results/stage8_baselines.json` → `random_point_validation`).
   Random hold-out fails to remove information because neighbours interpolate across every gap.
   Every robustness check in this report therefore holds out contiguous wavelength bands or whole
   temperature slices.
   *A weaker formulation of this test was tried first and did not work*: asking whether a random
   40-point subset can locate the crossover produced a narrow interquartile range, because
   interpolation across scattered gaps is easy. That formulation is recorded in
   `output/results/stage8_baselines.json` history and was rejected in favour of the prediction-error
   test above.
3. **The thinned composition ladder above +0.50 does not create artefactual structure.** Every
   long-wavelength decrease in flux with increasing index occurs between adjacent levels **below**
   M = −0.6, inside the fully sampled region; the 0.10-step region introduces no decreasing steps
   (`output/results/stage4_robustness.json` → `R6_sampling_region`).
4. **A quadratic fit does not locate a physical optimum.** Fitting `log F` against the index with a
   quadratic gives stationary points at M ≈ −1.19 to −1.22, mostly outside the sampled index range
   (21–32% of wavelengths), so a turnover in composition is not constrained by these data
   (`output/results/stage4_robustness.json` → `R5_quadratic_model`).

---

## 5. Cross-checking, independent verification, and adversarial testing

Every registered finding was recomputed from the raw TSVs by an independent worker that was not
permitted to read the main pipeline's code or results, then subjected to an adversarial refutation
attempt. The audit trail is under `output/verification/`.

**Independent recomputation agreed to machine precision.** Three headline crossing wavelengths
recomputed as 448.6654616491391 / 445.824847458333 / 409.8714474115428 nm against claimed
448.66546164914007 / 445.8248474583345 / 409.8714474115438 nm — relative tolerance 1 × 10⁻⁹
(`output/verification/CROSSCHECK.json` → `checks`). Sixteen further spot checks matched at the same
tolerance.

**Conflicts found and resolved.** The cross-check flagged one apparent numeric conflict — a factor of
131.46 versus 152.49 for the ultraviolet flux range across composition. Both are correct: 131.46 is
the 6300 K/179.4 nm value, 152.49 is the maximum over all three temperatures and all wavelengths
below 300 nm. The cross-check additionally confirmed that two other apparent conflicts were
comparisons of **different quantities** (a composition crossover near 440 nm versus a temperature-ratio
crossing near 640 nm), not disagreements.

**Documented overreach corrected in this report.** The cross-check flagged five places where the
verification artifacts' own conclusions ran ahead of their numbers. Three of them changed this
report's wording: the temperature-ratio crossing is now described as a *redward trend* rather than
movement that "moves redward with metallicity" as a monotone statement; the "order 100×" ultraviolet
figure is now given as a median with its full range; and the 0.86–0.88 band at 850 nm is now given
with its actual observed spread (0.8559–0.8856, exceeded at both ends).

**Adversarial refutations and their outcomes.**

| Finding | Refutation attempted | Outcome |
|---|---|---|
| 1 | Label-permutation null, monotone-preserving nulls, wavelength resampling/decimation | Survives; the plain permutation null was found **non-informative** (rank-1 climatology) and discarded |
| 2 | Quantisation-vs-trend test, edge-effect test, blackbody baseline | Survives only **weakened**: the composition dependence of the crossing is not separable from quantisation noise |
| 3 | Discrete-search degeneracy, alternative transforms, rank/independence, raw-index alignment | Survives; the ranked search separates the winner by 20–60× and the free exponent selects *p* = 1.0 |
| 4 | Registration-shift negative control, sampling-convention variants | Phenomenology reproduced, but the **original mechanism explanation was refuted** and replaced |

The last row is worth stating plainly: this analysis's first explanation for the sub-330 nm deviation
was that it reflected the wavelength-grid misregistration between the two files. The independent
refutation showed that correcting the misregistration leaves the deviation **intact**
(`output/verification/D4_uv_boundary.json` → `mechanism_is_registration_shift: false`). The finding
was revised accordingly.

### 5.1 Audit of this analysis's own pipeline

An independent auditor read the stage scripts and recomputed **43 claimed numbers from the raw
TSVs**: **0 mismatches** (float statistics agreeing to relative difference ≤ 10⁻¹², integer counts
exactly). The reshape was separately stress-tested — round-trip over all 18 480 rows exact, global
max and min cells located correctly, and all five wrong axis orderings fail loudly — so the
`(temperature, metallicity, wavelength)` ordering is confirmed correct
(`output/verification/AUDIT_main_session.md` §1–2).

The same audit found real defects, all of which have been repaired in the stage scripts and are
reflected in the numbers above:

| Defect found | Repair |
|---|---|
| Stage 3's "2.24× separation factor" was unfair: the constant was fitted on the winning curve and held fixed for competitors | Added a per-candidate refit; the fair value is **1.23×**, and the report now quotes both (stage3, `photon_to_energy_null`) |
| Stage 6 inverted its source, claiming the crossover "is a weak statistic"; the source's verdict was **SURVIVES** | Stage 6 docstring and interpretation rewritten; the null is now correctly described as degenerate (rank-1 composition family), and the p-value is reported as floor-limited `< 1/300` |
| Stage 5's `transition_alignment` predicate was vacuous (`np.argmin(...) is not None` is always true) | Replaced with an actual distance test; all three deviation jumps land **exactly** (0.0 nm) on grid step changes |
| Stage 5 named a *summation* mechanism the auditor rejected by ~40× | Corrected to a coverage/interval-weighting relation, with the refutation recorded |
| Stage 4 R6's verdict sentence contradicted its own stored field for the 5300 K slice | Verdict rebuilt from the stored pairs: 1 of 9 decreasing pairs does lie above M = +0.5, so the weaker statement is now made |
| Stage 4 R1 called the crossover "reproducible" over a 12.4 nm spread | Now stated as reproducible *to within 2.5 grid intervals* |
| Stage 4 R2 (band hold-out) and R7, stage 3's `optical_residual_structure`, stage 5's model set, and stage 1's `duplicated_cells` were vacuous or unreachable | Named as vacuous in the audit and not relied upon in this report; R2's real content is provided by stage 8's prediction-error test |

Also noted and accepted as a genuine limitation: interpolating the grid onto the series' wavelength
axis carries a curvature-induced error bounded at 6.7% (p99 3.4%, median 0.08%), which limits how
finely the Finding 3 residual can be trusted; and stage 4 hard-codes the 5800 K slice in three
checks without flagging it as the hypothesis temperature chosen from stage 3.

**Remaining uncertainty the cross-check identified as untested.** Uncertainty in the absolute
radiometric scale; the true physical identity of the 5 nm constant (§3 Finding 4); which file is
per-bin and which is per-unit-wavelength; the absence of a planted-data experiment for Findings 1
and 2; and the absence of confidence intervals on several headline values (partially addressed by
`GAP4_interaction_and_uncertainty.json`).

---

## Reproduction

Executed environment: Python 3.11.2, numpy 2.4.6, pandas 3.0.5, matplotlib 3.11.2. `scipy` is **not**
installed in this environment; no stage depends on it. All randomness uses fixed seeds
(20260919, 20230101).

Run from the workspace root, in order:

```
python3 output/analysis/stage1_inventory.py        # inventory, integrity, sampling structure
python3 output/analysis/stage2_boundaries.py       # wavelength boundaries (Findings 1, 2)
python3 output/analysis/stage3_crossfile.py        # cross-file relation (Findings 3, 4)
python3 output/analysis/stage4_robustness.py       # robustness checks R1-R7
python3 output/analysis/stage5_uv_mechanism.py     # ultraviolet breakdown mechanism (Finding 4)
python3 output/analysis/stage6_statistical_power.py# association strength and cross-slice replication
python3 output/analysis/stage7_consolidate.py      # machine-readable findings registry
python3 output/analysis/stage8_baselines.py        # rejected baseline models
python3 validate_output.py output                  # semantic delivery validation
```

| Stage | Result artifact |
|---|---|
| 1 | `output/results/stage1_inventory.json` |
| 2 | `output/results/stage2_boundaries.json` |
| 3 | `output/results/stage3_crossfile.json` |
| 4 | `output/results/stage4_robustness.json` |
| 5 | `output/results/stage5_uv_mechanism.json` |
| 6 | `output/results/stage6_statistical_power.json` |
| 7 | `output/results/findings_registry.json` |
| 8 | `output/results/stage8_baselines.json` |

Shared loading and hashing helpers are in `output/analysis/common.py`; stages 1–7 all consume the
same `(temperature, composition, wavelength)` cube built there, so every reported number traces to
one parsing path. Each stage re-verifies the manifest hashes before analysis.

Independent verification artifacts (separately authored, each with its own script):
`output/verification/D1_metallicity_crossover.json` + `D1_verify.py`, `D2_temperature_reversal.json`
+ `D2_verify.py`, `D3_crossfile_relation.json` + `D3_verify.py`, `D4_uv_boundary.json` +
`D4_verify.py`, `GAP1_calibration_circularity.json` + `GAP1_calibration.py`,
`GAP3_artefact_nulls.json` + `GAP3_nulls.py`, `GAP4_interaction_and_uncertainty.json` +
`GAP4_interaction.py`, `GAP2_mechanism.json` + `GAP2_mechanism.py`, `D2_refutation.json` +
`D2_refute.py`, `D3_refutation.json` + `D3_refute.py`, `CROSSCHECK.json` + `crosscheck.py`,
`AUDIT_main_session.json` + `AUDIT_main_session.md` (audit of this analysis's own pipeline).

Figures: `output/figures/fig1_wavelength_boundaries.png` (Findings 1, 2),
`output/figures/fig2_crossfile_relation.png` (Findings 3, 4);
generator script `output/figures/make_fig2.py`.

---

## 7. Limitations of the evidence base

1. **Three temperature slices only.** Every temperature-dependent statement rests on 5300, 5800 and
   6300 K. "Temperature-dependent" here means "differs across these three slices", not a measured
   trend.
2. **Composition is an abstract ordered index**, not a calibrated abundance. The ladder is uniform
   only up to +0.50; whether the index is [Fe/H], [M/H] or an α-enhanced mixture is not stated, and
   the interpretation of any abundance-dependent result would differ between those conventions
   (background: Coelho 2014, in `output/SOURCES.md`).
3. **Flux values carry 3 significant figures.** This bounds every resolvable relative deviation to
   roughly 0.2–0.5% and is why Finding 3's residual is described as *at* the noise floor rather than
   below it.
4. **Wavelength sampling is non-uniform and ends at 850 nm.** Redward behaviour is unconstrained, and
   the ultraviolet is sampled on a ladder whose intervals are themselves the cause of Finding 4.
5. **No error bars accompany the flux values.** All uncertainty quoted here is either bootstrap over
   composition levels (which cannot capture wavelength-sampling error) or the quantisation floor —
   not a propagated measurement uncertainty.
6. **These are radiation calculations.** `DATA_GUIDE.md` states plainly that they cannot establish
   biological or evolutionary causality; this report makes no such claim, and the mechanism
   interpretations in §3 are labelled as interpretations throughout.
7. **The bundled files are complete within the declared scientific scope**, which does not assert that
   every variable or record of any upstream source is included.

---

## 8. External background references

Used **only** for physical background, standard methods and candidate alternative explanations —
never as direct evidence for any claim about this dataset. All are strictly pre-T0 (2023-01-01), with
verified first-public dates and stable identifiers, recorded with their verification method in
`output/SOURCES.md`. Principal entries: Planck 1901 (DOI `10.1002/andp.19013090310`); Gustafsson et
al. 2008 (DOI `10.1051/0004-6361:200809724`, arXiv:0805.0554, first public 2008-05-05);
Coelho et al. 2005 (DOI `10.1051/0004-6361:20053511`, arXiv:astro-ph/0505511, 2005-05-25);
Casagrande et al. 2010 (DOI `10.1051/0004-6361/200913204`, arXiv:1001.3142, 2010-01-18);
Marr & Wilkin 2012 (DOI `10.1119/1.3696974`, arXiv:1109.3822, 2011-09-17);
Bessell & Murphy 2012 (DOI `10.1086/664083`, arXiv:1112.2698, 2011-12-12);
Roberts et al. 2017 (DOI `10.1111/ecog.02881`, first public 2016-12-08).

No external dataset was downloaded, substituted or merged. The workspace `data/` directory is the
complete dataset for this task. No external source was used to identify or reconstruct the upstream
source behind these tables.
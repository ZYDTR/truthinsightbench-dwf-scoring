# State- and Composition-Dependent Long-Wavelength Emission Structure

## Background and Objective

This report analyzes three frozen, complete data sets bundled in `data/` under
`/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project`:
(1) `aggregate_state_emission_spectra.tsv` (12,770 rows; 5 aggregate states per
series across 4 design series), (2) `dilute_state_emission_spectra.tsv` (2,253 rows;
one complete dilute spectrum per real series), and (3) `long_wavelength_response.tsv`
(20 rows; 5 composition points per series). The objective is to describe, with
real measured numbers, how (a) the long-wavelength response varies with composition
fraction, (b) the aggregate emission spectrum varies with aggregate state, and
(c) the dilute-state peak relates to the aggregate-state peaks — and to flag where
the metrics (peak position, FWHM) are method- or band-sensitive. All figures below
are computed by `output/analysis/core_analysis.py` and `output/analysis/robustness.py`,
which write machine-readable TSVs to `output/evidence/`. No external data were used.
`matplotlib` was unavailable in the analysis environment, so `figure_1_emission_structure`
(aggregate_spectra.png) and `figure_2_response_composition` were skipped; this is
recorded in `output/evidence/figure_availability.tsv` and is a reporting gap, not a
data gap.

---

## Finding 1: Long-wavelength response increases monotonically and nearly linearly with composition fraction for every design series

**Object and scope.** The relationship `long_wavelength_response_au ~ composition_fraction`
was fitted separately for each of the four design series (`reference_series`,
`series_A`, `series_B`, `series_C`) using the 5 composition points per series in
`data/long_wavelength_response.tsv`. The independent analysis unit is one complete
composition series (5 ordered composition fractions).

**Relationship, direction, and boundary.** The relationship is positive, monotonic, and
very strongly linear for all four series: OLS `R²` is ≥ 0.99 and Spearman ρ = 1.0
for every series, and the number of sign changes in the first differences is 0 for
every series (i.e. the response never decreases as composition fraction increases).
The boundary of the claim is the measured composition window per series, not a broader
composition range.

**Core quantitative result and analysis unit.** Per-series OLS slope (au per unit
fraction), intercept (au), R², and quadratic curvature coefficient (from
`composition_effects.tsv`):

| series | n | composition range | response range | OLS slope | intercept | R² | Spearman | sign_changes | quad curvature |
|---|---|---|---|---|---|---|---|---|---|
| series_A | 5 | [0.027, 0.091] | [10.59, 20.89] | 164.010801 | 6.215010 | 0.990978 | 1.0 | 0 | -429.171757 |
| series_B | 5 | [0.017, 0.091] | [9.24, 18.87] | 130.573732 | 7.126838 | 0.999119 | 1.0 | 0 | -167.797885 |
| series_C | 5 | [0.025, 0.084] | [9.42, 16.47] | 120.889846 | 6.642304 | 0.990298 | 1.0 | 0 | -621.396640 |
| reference_series | 5 | [0.010, 0.118] | [5.35, 6.05] | 6.350313 | 5.293740 | 0.998468 | 1.0 | 0 | +1.641695 |

The three real series (A/B/C) show much steeper positive slopes (≈121–164 au per unit
fraction) than the reference (≈6.35 au per unit fraction), and their response levels are
higher (≈9–21 au vs ≈5–6 au). The curvature coefficient is negative for A, B, C
(quadratic term bending downward) and essentially zero/positive for the reference,
so the real-series response is concave within the measured window while the reference
is (nearly) linear.

**Supporting analysis and result evidence.**
- Analysis: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/analysis/core_analysis.py` (section "1) COMPOSITION EFFECT ON LONG-WAVELENGTH RESPONSE").
- Evidence: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/composition_effects.tsv`
- Raw input: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/data/long_wavelength_response.tsv`

**Robustness result.** There is no dedicated `robustness_*.tsv` targeting the composition
fit (all four robustness checks in `robustness.py` operate on the aggregate emission
spectra). The robustness evidence available for this finding is internal consistency:
0 sign changes and Spearman ρ = 1.0 across all four series (a monotonic relationship
that is insensitive to the choice of linear vs rank-based fit), with OLS R² > 0.99
for every series. This is the full extent of the robustness support; see the
limitation below on sample size.

**Counterevidence, limitation, and alternative explanations.** The main limitation is
that each series has only 5 composition points, so a per-series leave-one-out is
not statistically meaningful and the slope is driven by the full 5-point span. The
negative curvature coefficients for A/B/C indicate the linearity is approximate and
could reverse at higher composition fractions outside the measured window.
Alternative explanations: the response may be genuinely composition-driven, or it may
reflect a correlated change in aggregation/state that is not captured by
composition fraction alone (the data do not separate these). The reference series'
much smaller slope is a real contrast but reflects a different composition window
[0.010, 0.118] and a lower baseline response.

**Executable new observation / replication to falsify or validate.** Measure a new
composition series with ≥ 8–10 evenly spaced fractions extending beyond the current
window (e.g. below 0.010 and above 0.091) and refit OLS + a quadratic model.
A falsification of the "positive, monotonic" claim would be any sign change in the
first differences or a reversal of the slope's sign; validation would be a
reproducible positive monotonic slope with R² > 0.9 across the extended range.

---

## Finding 2: From state_1 to state_5 the aggregate peak shifts to shorter wavelength (negative/blue shift), peak intensity drops, and (where computable) the band broadens — across all series

**Object and scope.** For each design series, the aggregate-state spectra in
`data/aggregate_state_emission_spectra.tsv` (5 states: state_1 … state_5) were
summarized into peak wavelength, max intensity, FWHM, integrated intensity, and
band_ratio per state. The comparison is state_5 vs state_1. The independent analysis
unit is one complete design-series aggregate spectrum set (all 5 states).

**Relationship, direction, and boundary.** Direction is consistent: the state_5→state_1
peak shift (state_5 minus state_1) is negative (shorter wavelength) for every series,
max intensity decreases from state_1 to state_5 for every series, and the FWHM increases
from state_1 to state_5 wherever both FWHM values are defined. The boundary is that
the FWHM claim is only supported where FWHM is defined (see Finding 4); peak-position
magnitude is method-sensitive (see Finding 4).

**Core quantitative result and analysis unit.** From `state_effects.tsv` (peak shift,
max-intensity ratio state_5/state_1, FWHM change state_5 − state_1) and
`state_by_state.tsv`:

| series | state_1 peak | state_5 peak | shift (5−1) | state_1 max | state_5 max | ratio 5/1 | state_1 FWHM | state_5 FWHM | FWHM change |
|---|---|---|---|---|---|---|---|---|---|
| reference_series | 1132 | 1117 | -15.0 nm | 0.02019 | 0.0142 | 0.703318 | NaN | NaN | NaN |
| series_A | 1028 | 1020 | -8.0 nm | 0.07519 | 0.0289 | 0.384360 | 308.949686 | 405.403509 | +96.453823 nm |
| series_B | 1030 | 1028 | -2.0 nm | 0.06439 | 0.02249 | 0.349278 | 324.176329 | NaN | NaN |
| series_C | 1029 | 1023 | -6.0 nm | 0.05334 | 0.02305 | 0.432133 | 333.78125 | NaN | NaN |

Integrated intensity and band_ratio (from `spectral_summary.tsv`) move in the same
direction (decrease): series_A 25.80511→12.78031 (band_ratio 3.7695→2.0064);
series_B 23.08576→11.000555 (3.3126→1.6365); series_C 19.930675→11.199855
(2.9246→1.6484); reference 6.054015→5.34722 (1.1619→1.1321).

**Supporting analysis and result evidence.**
- Analysis: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/analysis/core_analysis.py` (sections "2) STATE EFFECT" and "3) DILUTE vs AGGREGATE PEAK WAVELENGTHS" for per-state values).
- Evidence: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/state_effects.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/state_by_state.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/spectral_summary.tsv`.

**Robustness result.** Cite `robustness_leaveout.tsv` (leave-one-series-out on the state
effect): full A+B+C mean shift = -5.333 nm, all_negative = True. Each subset
(A+B = -5.0 nm, A+C = -7.0 nm, B+C = -4.0 nm) keeps all_negative = True,
sign_consistent_vs_full = True, and magnitude within 50% of the full mean
(rel_mean_vs_full 0.9375, 1.3125, 0.7500, 1.0), with per-series shifts unchanged
(A=-8, B=-2, C=-6). Cite `robustness_peak.tsv` (peak definition): Gaussian-center vs
argmax shift mean 16.2 nm (range -4.0 to +28.2 nm), so the *sign* of the shift is
robust but its magnitude is method-sensitive. Cite `robustness_band.tsv` (band
validation): 20/20 peaks unchanged under a 900–1250 nm restriction (peak_shift 0.0),
so the peak-position comparison is band-robust. Cite `robustness_fwhm.tsv` (FWHM
threshold): only 1/20 states has a valid 30%-of-max width (series_A state_1,
rel_change_30_vs_50 = +0.376771, i.e. +37.7% wider than the 50% FWHM); 0/20 valid
at 20% — FWHM is threshold-sensitive.

**Counterevidence, limitation, and alternative explanations.** The FWHM "broadens"
claim is only supported for series_A (state_5 FWHM defined); for series_B and
series_C state_5, and all five reference states, the 50% FWHM is undefined (NaN)
because the lower-wavelength half-crossing falls outside the measured band. The
exact peak-shift magnitude (−2 to −15 nm) is uncertain because the argmax peak is
systematically ~16 nm (up to 28 nm) shorter than a local Gaussian-fit center for the
broad A/B/C peaks (asymmetric/shouldered peaks). The reference series sits in a
different magnitude regime (peaks 1117–1132 nm vs 1020–1034 nm for A/B/C) and its
FWHM is entirely undefined, so it cannot support the broadening claim.
Alternative explanations: the state-dependent change may reflect a change in the
emitting species/aggregation rather than a simple physical broadening; the
method-sensitivity of peak position means part of the "blue shift" may be an
artifact of peak asymmetry.

**Executable new observation / replication to falsify or validate.** (i) Measure a new
series with the full emission band extended below 900 nm so the lower half-crossing
is captured, and recompute FWHM for state_5 of B and C — a falsification of the
"broadening" claim would be a non-increasing FWHM; (ii) re-estimate peaks with both
argmax and Gaussian fit and report the difference — if the state_1→state_5 shift
remains negative and of the same sign under both definitions, the finding is validated
as method-robust in sign.

---

## Finding 3: The dilute-state peak is not consistently offset from the aggregate-state peaks; the dilute-minus-aggregate sign and magnitude flip across series and states (conditional conflict)

**Object and scope.** Each real series has a single complete dilute spectrum
(`data/dilute_state_emission_spectra.tsv`), summarized to a single dilute peak
(`dilute_summary.tsv`). This peak is compared against each of the 5 aggregate-state
peaks per series (`dilute_vs_aggregate.tsv`). The independent analysis unit is one
design series (its dilute peak vs its 5 aggregate-state peaks).

**Relationship, direction, and boundary.** There is no single consistent direction.
For series_A the dilute peak is at *longer* wavelength than every aggregate-state
peak (Δ = dilute − aggregate ∈ [+3, +5, +5, +3, +11] nm for states 1–5).
For series_C it is also at longer wavelength (Δ ∈ [+5, +7, +0, +6, +11] nm).
For series_B the sign is mixed: Δ = −1 nm for states 1 and 2 (dilute at *shorter*
wavelength), then +3, +2, +1 nm for states 3–5. Magnitudes range from −1 to +11 nm.
The boundary of the claim is the 3 real series (A/B/C); the reference series has no
dilute spectrum in the data, so it is excluded.

**Core quantitative result and analysis unit.** Dilute peaks (from `dilute_summary.tsv`):
series_A 1031 nm (max 0.99354, FWHM 274.178329, integrated 292.717520);
series_B 1029 nm (max 0.99501, FWHM 266.677711, integrated 287.580990);
series_C 1034 nm (max 0.99631, FWHM 275.451760, integrated 294.497510).
Dilute-minus-aggregate offsets (from `dilute_vs_aggregate.tsv`):

| series | state_1 | state_2 | state_3 | state_4 | state_5 |
|---|---|---|---|---|---|
| series_A (dilute 1031) | +3 | +5 | +5 | +3 | +11 |
| series_B (dilute 1029) | −1 | −1 | +3 | +2 | +1 |
| series_C (dilute 1034) | +5 | +7 | +0 | +6 | +11 |

**Supporting analysis and result evidence.**
- Analysis: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/analysis/core_analysis.py` (section "3) DILUTE vs AGGREGATE PEAK WAVELENGTHS").
- Evidence: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/dilute_vs_aggregate.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/dilute_summary.tsv`.

**Robustness result.** No dedicated `robustness_*.tsv` was run for the dilute-vs-aggregate
comparison (the four robustness checks target the aggregate spectra). The honest
robustness statement is that the *inconsistency itself is the result*: the sign is
not even within-series consistent (series_B flips from negative to positive across
states), so no robustness check would convert this into a uniform offset. The dilute
peaks are stable single argmax values per series (max intensity ≈ 0.993–0.996),
so the comparison is not an artifact of a noisy dilute peak.

**Counterevidence, limitation, and alternative explanations.** The reference series is
excluded because it has no dilute spectrum (a data boundary, not a null finding).
Because the sign flips within series_B and the magnitudes span −1 to +11 nm, the
data do not support any single "dilute peak is always X nm from the aggregate peak"
claim. Alternative explanations: (a) dilute vs aggregate emission arise from
different emitters (genuinely different species/energy levels), so a fixed offset
is not expected; (b) the peak-position method-sensitivity (Finding 4) means each
offset carries ~10–25 nm uncertainty in the A/B/C peaks, which is comparable to or
larger than many of these offsets — so the small offsets (±1 to +3 nm) are within
the method uncertainty.

**Executable new observation / replication to falsify or validate.** Measure the dilute
and aggregate spectra of a new series (or the same series under a second preparation
batch) and recompute the dilute-minus-aggregate offsets using the same argmax and
a Gaussian fit. A falsification of the "no consistent offset" claim would be a
uniform sign and magnitude (e.g. always +X nm ± small) across states and series;
validation would be a repeat of the mixed-sign pattern.

---

## Finding 4: Spectral-width (FWHM) and peak-position metrics are method- and band-sensitive; FWHM is undefined for 7 of 20 states and fails the 900–1250 nm band validation (applicability boundary)

**Object and scope.** This finding characterizes the reliability of the FWHM and
peak-position metrics computed in `spectral_summary.tsv`, `state_by_state.tsv`,
`state_effects.tsv`, and the dilute/aggregate summaries. Scope is all 4 series × 5
states = 20 aggregate states (plus the 3 dilute peaks). The independent analysis
unit is one state spectrum.

**Relationship, direction, and boundary.** The "relationship" is a robustness/applicability
statement: peak *position* is robust to the 900–1250 nm band restriction (20/20 peaks
unchanged), but FWHM is *not* robust to (a) the intensity threshold, (b) the peak
definition, or (c) the wavelength band. The boundary of the claim is the measured
wavelength coverage (900–1600 nm for aggregate; 850–1600 nm for dilute).

**Core quantitative result and analysis unit.** From `robustness_fwhm.tsv`,
`robustness_band.tsv`, and `robustness_peak.tsv`:
- FWHM (50% of max) is undefined (NaN) for 7/20 states: reference_series
  state_1–5, series_B state_5, series_C state_5. The defined FWHMs range from
  308.949686 nm (series_A state_1) to 405.403509 nm (series_A state_5).
- Band validation (900–1250 nm vs full band): peak unchanged for 20/20 states
  (peak_shift_full_minus_restricted = 0.0), but the restricted-band FWHM is NaN
  for 18/20 states — the broad peaks extend below 900 nm, so the lower
  wavelength half-crossing is lost (only series_A state_1 and state_2 survive the
  restriction).
- Peak definition (argmax vs local Gaussian fit, 211 points for A/B/C): Gaussian
  center minus argmax mean = 16.2 nm, range −4.0 to +28.2 nm; for the broad A/B/C
  peaks the argmax is systematically shorter (shifts +18.2 to +28.2 nm), while
  reference-series shifts are small (−4.0 to +2.6 nm).
- FWHM threshold sensitivity: width at 30% of max is +37.7% wider than at 50%
  (rel_change_30_vs_50 = +0.376771, the only valid 30% case, series_A state_1);
  no valid 20% cases — indicating non-Gaussian, broad peaks.

**Supporting analysis and result evidence.**
- Analysis: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/analysis/robustness.py`
  (CHECK 1 FWHM threshold, CHECK 2 peak definition, CHECK 4 band validation).
- Evidence: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/robustness_fwhm.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/robustness_peak.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/robustness_band.tsv`,
  `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/output/evidence/figure_availability.tsv` (matplotlib unavailable → figures skipped).

**Robustness result.** This finding is itself a robustness result; its supporting
checks are the four `robustness_*.tsv` files above, executed by `robustness.py`.
The peak-position claim (Finding 2) is band-robust (20/20) but the FWHM claims
are not (18/20 restricted NaN, 7/20 full NaN, threshold-sensitive).

**Counterevidence, limitation, and alternative explanations.** The undefined FWHMs
are a data/coverage limitation (half-crossing outside the measured band), not a
measurement error; they mean FWHM-based comparisons (e.g. the broadening claim in
Finding 2) apply only to series_A. The argmax-vs-Gaussian discrepancy shows that
the "peak" of these broad, asymmetric/shouldered spectra is not uniquely defined.
Alternative explanations: the apparent band-broadening may partly reflect the
wavelength sampling and the band edge rather than a true physical broadening.

**Executable new observation / replication to falsify or validate.** Acquire spectra
with wavelength coverage extended below 900 nm (and, ideally, a finer wavelength
grid) for the states currently missing FWHM, then recompute the 50% FWHM. A
falsification of the "FWHM undefined because the half-crossing lies outside the
band" statement would be a defined FWHM at the current band edge; validation would
be a defined, consistent FWHM for the previously-NaN states and a stable
argmax-vs-Gaussian gap.

---

## Limitations and Alternative Explanations

1. **Sample size.** Each composition series has only 5 points; each aggregate/dilute
   comparison uses 5 states or 1 dilute spectrum per series. Effect estimates (e.g.
   OLS slopes, FWHM) are therefore not statistically generalizable beyond the
   measured windows, and per-series leave-one-out is not meaningful for the 5-point
   composition fits.
2. **Peak-position method-sensitivity.** For the broad series_A/B/C peaks, the
   argmax peak is ~16 nm (up to 28 nm) shorter than a local Gaussian-fit center
   (`robustness_peak.tsv`). Any claim about absolute peak wavelength or a small
   peak offset (Finding 3) carries this uncertainty.
3. **FWHM coverage gap.** FWHM is undefined for 7/20 aggregate states (all 5
   reference states, series_B state_5, series_C state_5) and fails the 900–1250 nm
   band validation (18/20 NaN), so FWHM-based broadening claims apply only to
   series_A.
4. **Dilute-vs-aggregate is not a uniform effect.** The sign and magnitude of the
   dilute-minus-aggregate offset flip across series and states (Finding 3), so no
   single offset model is supported.
5. **Reference series is a different regime.** Its peaks (1117–1132 nm) and
   intensities (≈0.014–0.020) are in a different range from A/B/C (1020–1034 nm,
   ≈0.022–0.075), and its FWHM is entirely undefined; it should not be averaged
   with A/B/C for width claims.
6. **Mechanism is not identified.** As the data guide states, spectral intensity and
   peak position alone do not uniquely identify an emission mechanism. All findings
   above are empirical descriptions of the measured spectra, not mechanism
   assignments.
7. **Reporting gap (not a data gap).** `matplotlib` was unavailable, so both
   figures were skipped (`figure_availability.tsv`, `matplotlib_available = False`);
   `aggregate_spectra.png` was not created. The quantitative conclusions above
   are unaffected and are fully reproducible from the TSVs.

---

## Reproduction

To rerun all analyses and regenerate every evidence TSV, run the two executed
scripts below with `python3` from the project root. They overwrite the TSVs in
`output/evidence/`. (No network access or `pip install` is required; only
`numpy` and `pandas` are imported.)

```bash
cd /data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project
python3 output/analysis/core_analysis.py
python3 output/analysis/robustness.py
```

Optional exploratory and figure scripts (also executed; figures were skipped because
`matplotlib` is unavailable):

```bash
python3 output/analysis/explore_data.py
python3 output/analysis/make_figures.py
```

**Expected inputs** (read-only, unchanged):
- `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/data/aggregate_state_emission_spectra.tsv`
- `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/data/dilute_state_emission_spectra.tsv`
- `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project/data/long_wavelength_response.tsv`

**Evidence files produced** (machine-readable, under `output/evidence/`):
- `composition_effects.tsv` (Finding 1)
- `state_effects.tsv`, `state_by_state.tsv`, `spectral_summary.tsv` (Finding 2)
- `dilute_vs_aggregate.tsv`, `dilute_summary.tsv` (Finding 3)
- `robustness_peak.tsv`, `robustness_band.tsv`, `robustness_fwhm.tsv`, `robustness_leaveout.tsv` (Findings 2 and 4)
- `figure_availability.tsv` (figure-skip record)

**Finding-to-evidence traceability map:**

| Finding | Analysis script | Evidence TSV(s) |
|---|---|---|
| Finding 1 (composition effect) | `output/analysis/core_analysis.py` | `output/evidence/composition_effects.tsv` |
| Finding 2 (state effect) | `output/analysis/core_analysis.py` | `output/evidence/state_effects.tsv`, `output/evidence/state_by_state.tsv`, `output/evidence/spectral_summary.tsv`; robustness: `output/evidence/robustness_leaveout.tsv`, `output/evidence/robustness_peak.tsv`, `output/evidence/robustness_band.tsv`, `output/evidence/robustness_fwhm.tsv` |
| Finding 3 (dilute vs aggregate) | `output/analysis/core_analysis.py` | `output/evidence/dilute_vs_aggregate.tsv`, `output/evidence/dilute_summary.tsv` |
| Finding 4 (FWHM/peak metric sensitivity) | `output/analysis/robustness.py` | `output/evidence/robustness_fwhm.tsv`, `output/evidence/robustness_peak.tsv`, `output/evidence/robustness_band.tsv`, `output/evidence/figure_availability.tsv` |

All numbers quoted in this report are taken directly from the TSVs above and were
re-verified by re-running the scripts (stdout shown in the run log).

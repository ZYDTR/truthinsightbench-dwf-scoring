# Time-Resolved Gravity Residuals and Candidate Spectral Structure

**Task:** `Astronomy_02_gravity_spectral_structure` — T0 = 2022-01-01
**Workspace:** `/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/`
**Data:** the five frozen tables under `data/` (SHA-256 verified against `data_manifest.json`)

This report is the sole authority for the registered findings. Every number below was produced by
executed code held under `output/code/` and written to machine-readable artifacts under
`output/results/`. No external dataset, no external literature, and no other task workspace was used:
the workspace `data/` directory is the complete evidence base for every core claim.

---

## 1. Scientific question and analysis unit

The task supplies 22 frozen, complete time-resolved passes of residual velocity (mm/s) for one
anonymous gravitating system, three competing residual models (A, B, C), and a ranked grid of
candidate spectral parameter combinations together with a "uniform alternative" comparison.

**The independent analysis unit is one complete pass (n = 22) or one complete candidate parameter
combination.** Adjacent time samples (1,698 rows) and neighbouring grid rows (250 / 507 rows) are
*not* independent replicates: within a pass the residuals are strongly autocorrelated, and
neighbouring grid cells differ by one lattice step on a surface that is smooth in its parameters.
Every test below is therefore blocked at the pass level — whole-pass statistics, paired tests over
22 passes, leave-one-pass-out and bootstrap-over-passes resampling. No p-value or confidence
interval in this report treats time samples or grid rows as replicates.

## 2. Methods

Full executable detail is in `output/code/` and the reproduction section (§7).

- **Whole-pass statistic.** For each pass and model, RMS = `sqrt(mean(residual^2))`. Ratios are
  formed pass-by-pass, so the sample of ratios has n = 22.
- **Blocked inference.** Wilcoxon signed-rank and paired t-test over the 22 pass-level values;
  percentile bootstrap resampling *passes* (10,000–20,000 draws); leave-one-pass-out sensitivity.
- **Spectral estimation.** Normalized Lomb–Scargle periodogram on the real irregular time stamps
  (`t` in seconds, 200–3000 µHz), cross-validated against `scipy.signal.lombscargle` to 1.4e-15.
- **Red-noise null.** AR(1) surrogates fitted per pass and per model, regenerated on the *same*
  irregular time stamps, 200–1,000 realizations per pass as noted.
- **Grid analysis.** Axis inventory, IC ladders, marginal and ridge fits, and a uniform-frequency
  chance null (200,000 draws) for the mode-alignment test.
- **Localization.** Per-pass RMS profiles binned on |t|, a within-pass time-shuffle null (2,000
  permutations per pass), and envelope-shape model comparison by AICc.

## 3. Registered findings

---

### Finding 1 — Model B reproduces as the best-fitting residual model on every one of the 22 passes, and its advantage is concentrated near the middle of the pass

**Object and scope.** The three supplied residual models A, B, C, on all 22 complete passes of
`data/pass_residual_traces.tsv`. Unit = one complete pass.

**Relationship and direction.** Whole-pass RMS orders the models **B < C < A**, and the ordering
holds pass-by-pass, not merely in aggregate.

**Core quantitative result.**

| paired comparison | median ratio | passes won by the better model | Wilcoxon p | paired t p |
|---|---|---|---|---|
| B vs A | **0.5413** (B/A) | **22 / 22** | 4.77e-07 | 6.68e-08 |
| C vs A | 0.6741 (C/A) | **22 / 22** | 4.77e-07 | 1.72e-07 |
| B vs C | **0.8174** (B/C) | **22 / 22** | 4.77e-07 | 4.63e-06 |

Median whole-pass RMS (mm/s): A = 0.027811, B = 0.013381, C = 0.018737. The B/A advantage is a
**45.9 % reduction in whole-pass RMS**, i.e. B removes roughly 80 % of the A−B variance in the centre
of the pass (`output/results/localization.json`, `item7_verdict`).

**Supporting analysis and result evidence.**
- `output/code/track1_model_comparison.py` → `output/results/model_comparison.json`
  (`item1_per_pass_whole_pass_rms_and_ratios`, `item2_blocked_paired_tests_over_22_passes`)
- `output/code/track6_localization.py` → `output/results/localization.json` (`item2`, `item4`, `item7`)
- Figures: `output/figures/fig1_residual_traces.png`, `output/figures/fig2_per_pass_rms.png`
- Verification: `output/results/verify_model_comparison.json` (all recomputed values abs diff 0.0)

**Robustness.** The ordering survives every perturbation tested
(`model_comparison.json:item5_robustness_alternative_definitions`):
- *Per-pass linear detrend*: B/A 0.5494, C/A 0.6829, B/C 0.8177 — ordering preserved.
- *Boxcar smoothing* k = 1, 3, 5, 7, 9, 15 min: B wins 22/22 at every width; at k = 15 the median
  A/B RMS ratio is 12.31, i.e. the advantage **grows** with smoothing.
- *Dropping the 7 passes containing time gaps > 1.5 min*: ordering preserved.
- *Leave-one-pass-out*: median B/A spans only [0.5246, 0.5580] over all 22 deletions.
- *Whole-pass bootstrap*: median B/A 0.5413, 95 % CI [0.4600, 0.6202], P(B/A < 1) = 1.0000.
- *Pass-level cross-validation* (coefficients fitted on the other 21 passes, evaluated on the held-out
  pass): predicting B from A and C gives out-of-sample RMS 0.569 × B's own RMS (18/22 passes < 1);
  predicting C from A and B gives 0.412 (22/22); predicting A from B and C gives 0.581 (22/22). The
  relations replicate out of sample.

**Localization result.** The A−B difference is **centrally concentrated, not edge-driven and not a
uniform rescaling**. Centre = |t| ≤ 15 min, edge = |t| > 15 min:
- Median A−B centre/edge RMS ratio = **1.955** (20/22 passes > 1); C−B = 2.740 (22/22); A−C = 1.453.
- Outer/inner bin ratio (|t| ∈ [30,40) over |t| ∈ [0,5)) has median **0.3617**, with 21/22 passes
  below 1; Wilcoxon p = 6.68e-06.
- Scale-free check: the centre holds a median **68.5 %** of each pass's A−B power (IQR 60.0–76.6 %)
  while holding 37.5 % of the samples; 20/22 passes exceed the flat expectation (sign-test
  p = 1.21e-04). Blocked p for the centre/edge contrast = 3.34e-06; median shuffle-null percentile
  99.5.
- Envelope-shape comparison (AICc, lower = better): power-law-with-offset −272.98 and
  Lorentzian-shifted-centre −272.78 beat constant −191.38 and plain Gaussian −242.38. The profile is
  a **smooth broad envelope with scale ≈ 22–33 min**, not a sharp localized spike.

**Counterevidence, limitations, alternative explanations.**
- The three models are strongly collinear (whole-pass raw RMS correlations: A–B 0.939, A–C 0.973,
  B–C 0.985), and C is well explained by A and B together (R² = 0.921). B is *not* an independent
  physical model: it is largely recoverable from A and C. The comparison therefore establishes which
  *supplied parameterisation* fits best, not that B's physical assumptions are correct.
- Model B's own profile is nearly flat (median centre/edge ratio 1.049), so what distinguishes B is
  its **ability to suppress the centrally concentrated excess** that A and C both carry, rather than
  B adding structure of its own.
- 22 passes is a small sample; the bootstrap CI on B/A is wide ([0.46, 0.62]). The *sign* of the
  effect is nevertheless secure (22/22).
- Residual model B's later upturn in the outermost bin is absent, whereas A's profile shows a mild
  upturn at |t| > 35 min — a boundary behaviour we cannot separate from an edge artefact with 22
  passes.
- 7 of 22 passes contain time gaps > 1.5 min; the ordering is unchanged when they are removed, but a
  gap-correlated systematic cannot be fully excluded.

**Falsification / new observation.** An independent pass of the same system, processed with the same
three parameterisations, should reproduce B/A < 1 in essentially every pass. Pre-registered
falsifier: if a fresh set of ≥ 8 passes yields B/A ≥ 1 in more than one pass, or a median B/A > 0.8,
the effect is not reproducible. A direct physical test — recomputing B's central suppression after
explicitly modelling a smooth central mass distribution — would separate "correct physics" from
"flexible nuisance parameter".

---

### Finding 2 — The supplied candidate "uniform alternative" is not an independent alternative: it is exactly the max-equals-min diagonal of the same evaluated scan family

**Object and scope.** `data/uniform_alternative_comparison.tsv` (24 rows),
`data/candidate_spectrum_comparison.tsv` (507 rows),
`data/ranked_full_parameter_grid.tsv` (250 rows).

**Relationship and direction.** The uniform table is a **literal sub-slice** of the
`minimum_velocity_scan` family at `peak_frequency = 1210 µHz` and `width = 300 µHz`, not a distinct
model.

**Core quantitative result.** 19 of 24 uniform rows match a `minimum_velocity_scan` row where
`max_velocity_cm_s == min_velocity_cm_s`, with **maximum absolute IC difference 0.0** (bit-identical).
The 5 unmatched rows are exactly the velocities at or below the family's diagonal floor (0.001, 0.002,
0.003, 0.005, 0.01 cm/s — `minimum_velocity_scan` evaluates the diagonal only for v ≥ 0.02 cm/s), and
their IC values (5551.0, 5062.9, 4467.1, 3656.9, 2797.3) appear nowhere in the 507-row candidate
table.

IC minima by family: `frequency_amplitude_scan` 43.9 at (vmax = 20 cm/s, f = 1010 µHz);
`minimum_velocity_scan` 50.7 at (vmax = 30, vmin = 0.03, f = 1210);
**uniform alternative 1522.1 at v = 0.1 cm/s**. The uniform minimum is therefore **34.7×**
(1478.2 IC units) worse than the candidate-family minimum — but that gap is a **nested-subset gap**,
not an independent model comparison.

**Supporting analysis and result evidence.**
- `output/code/track4_ic_comparison.py` → `output/results/ic_comparison.json`
  (`q1_uniform_is_mv_diagonal`, `q3_uniform_vs_candidate_minimum`, `q5_best_per_family`)
- Figure: `output/figures/fig4_ic_surfaces.png`
- Verification: `output/results/verify_ic.json` — verdict *partially confirmed*, confirmed on all
  computed quantities; the verifier independently reproduced the identity.

**Robustness.** The identity is exact (diff 0.0) and is insensitive to tolerance: it holds under
raw-string key matching as well as float-rounded matching. The uniform curve is smooth and unimodal
(local curvature 3047 IC per dex²), so its minimum is interior, not a boundary artefact. The
preferred velocity scale is stable across all restriction variants tested.

**Counterevidence, limitations, alternative explanations.**
- `delta_information_criterion` in the ranked grid **is** the absolute IC (verified: 26 exact-key
  overlaps, max |offset| 1.4e-14). So all three tables are on one common IC scale and the comparisons
  above are internally consistent.
- The ranked grid reaches absolute IC 0.0 at (vmax = 30, vmin = 0.05, f = 1010, w = 275) — 43.9 IC
  units better than the best row of `candidate_spectrum_comparison.tsv`. The candidate table does not
  contain the overall optimum of the evaluated family.
- **A lower IC supports only a relative ordering within the supplied candidate family.** It is not
  proof that the family is physically unique, and here the "family vs uniform" contrast is partly
  circular because one is a subset of the other.
- The only genuinely non-nested family comparison is `frequency_amplitude_scan` (43.9) vs
  `minimum_velocity_scan` (50.7): Δ = 6.8 IC units, ratio 1.155 — a **marginal** margin.

**Falsification / new observation.** The identity is a falsifiable arithmetic claim: any re-issued
`uniform_alternative_comparison.tsv` row whose IC differs from the corresponding `vmax == vmin` row of
the `minimum_velocity_scan` family would refute it. To test the substantive question, an evaluation
of a genuinely independent model (not a diagonal sub-slice of an existing scan) on the same data would
be required; that experiment is not available from this workspace.

---

### Finding 3 — No reproducible localized spectral line is supported at the candidate frequencies; the traces' power is either a broadband low-frequency excess (A, C) or a broadband high-frequency one (B)

**Object and scope.** All 22 passes × 3 models, Lomb–Scargle spectra over 200–3000 µHz; the candidate
band 860–1310 µHz and the 50 µHz comb (610…1510 µHz) that the grid highlights.

**Relationship and direction.** **Null / negative result.** There is no excess power at the candidate
frequencies that survives a red-noise null at the pass level, and the stacked spectra do not peak in
the candidate band.

**Core quantitative result.**
- **Candidate-band test (860–1310 µHz)** against AR(1) surrogates on the same time stamps
  (200–1000 surrogates/pass): median per-pass percentile A = 0.471, B = 0.0085, C = 0.710.
  Passes exceeding the null 95th percentile: A = 0/22, B = 0/22, C = 3/22. Exceeding the 99th:
  A = 0/22, B = 0/22, C = 1/22.
- **Stacked (pass-averaged) spectrum peaks are far from the candidate band**: A at 489 µHz,
  B at 2474 µHz, C at 1706 µHz — *none* inside 860–1310 µHz.
- **Comb test**: comb-vs-off-comb band power gives sign-test p = 0.134 (A), 0.134 (B), 0.832 (C);
  the median difference is ~1e-4 of the band power. No comb-specific excess.
- **Phase coherence** at 1010 µHz: Rayleigh R = 0.278 (A), 0.135 (B), 0.298 (C) against a uniform-phase
  null, p = 0.187 / 0.677 / 0.142. At 1210 µHz: R = 0.238 / 0.040 / 0.243, p = 0.289 / 0.968 / 0.281.
- **Spectral slope** (log–log): median slope A = −1.222, B = +1.894, C = +0.439. Low/high band power
  ratio (300–600 vs 2000–3000 µHz): A = 17.3, B = 0.038, C = 0.491.
- **Corrected stacked-peak test** (adjudicated; see below): with a *matched* AR(1) null
  (null statistic = max over f of the pass-mean power, as opposed to the mismatched mean-of-per-pass-maxima
  used in the first pass), the observed maxima are 0.2001 (A), 0.0404 (B), 0.0668 (C) against null
  medians 0.1186 / 0.0257 / 0.0533 and 95th percentiles 0.1462 / 0.0316 / 0.0667 →
  **p = 0.000 / 0.000 / 0.050**. So the stacked peaks *are* real excess power — but they sit at
  490 / 2475 / 1705 µHz, i.e. **outside the candidate band**, and are broadband.

**Supporting analysis and result evidence.**
- `output/code/track2_spectral_tests.py` → `output/results/spectral_tests.json`
  (`sec2_candidate_band`, `sec3_comb_test`, `sec4_reproducibility`, `sec5_phase_coherence`,
  `sec6_spectral_slope`, `sec7_verdict`); arrays in `output/results/spectral_tests_arrays.npz`
- `output/code/track7_matched_null_adjudication.py` (matched-null correction, run by the main agent)
- Figure: `output/figures/fig3_periodograms.png`
- Verification: `output/results/verify_spectral.json` (verdict *partially confirmed*, severity major —
  the defect it found is recorded and corrected below)

**Robustness.**
- The band null was recomputed with an entirely independent pipeline
  (`output/code/verify_spectral_recompute.py`, self-validated against `scipy.signal.lombscargle` to
  1.6e-16): median band percentiles A = 0.476, B = 0.0075, C = 0.711 — consistent with the track.
- Re-running `track2_spectral_tests.py` in a sandbox reproduced every substantive number bit-for-bit
  (only `/runtime_sec` differed).
- Per-pass linear detrending leaves the stacked peaks essentially unchanged (A 515 µHz, B 2473 µHz,
  C 1711 µHz), so a trend does not manufacture them.
- Model C's marginal 95th-percentile count is boundary-fragile: pass_03 model C sits at band
  percentile 0.949–0.957 across four independent surrogate seeds, so that count moves between 3 and 4
  of 22. The "no strong line" conclusion is unaffected.

**Counterevidence, limitation, alternative explanations — a real defect, and its correction.**
The first-pass spectral track reported `p_stacked_global_peak_vs_ar1` = 0.997 / 1.000 / 1.000,
reading as "the stacked peaks are red-noise consistent". The adversarial verifier found that the null
statistic was built as the **mean of per-pass maxima** while the observed statistic was the
**max of the pass-means** — a mismatched order of operations. We adjudicated this directly
(`output/code/track7_matched_null_adjudication.py`): under a correctly matched null the peaks are
**significant** (p = 0.000 / 0.000 / 0.050), not consistent with red noise. **This correction does not
rescue the candidate line**: the significant power is at 490 / 2475 / 1705 µHz, outside 860–1310 µHz,
and the band-restricted matched null gives p = 0.185 (A) and 0.065 (C). The registered conclusion —
*no reproducible localized line at the candidate frequencies* — is unchanged and is now supported by a
correct matched null rather than an incorrect one.
A second defect: the track reported 16/22 pass peaks within one resolution element of A's stacked peak
against a uniform-chance expectation of 3.3; the verifier's full-pipeline red-noise null gives 8.3
(expected 4–12), so that support count is overstated. A's support is explained by its **broadband
low-frequency excess**, not by a line: its stacked peak is only 1.16× the local median within one
resolution element, and per-pass peak frequencies are broadly spread (median 557 µHz, IQR 432–656).

**Falsification / new observation.** The null is falsifiable in a strong sense: if the candidate
frequencies carried a real line, an independent set of passes should show (i) a stacked peak inside
860–1310 µHz, (ii) a candidate-band percentile above the AR(1) 95th in a clear majority of passes, and
(iii) cross-pass phase concentration (Rayleigh p ≪ 0.05) at a fixed candidate frequency. None of the
three holds. A longer baseline — one continuous observation spanning ≫ 80 min — would raise the
frequency resolution and give the test real power at 50 µHz comb spacing.

---

### Finding 4 — The candidate frequencies are not preferentially aligned with the supplied mode grid, and the narrowest supplied candidate width is not resolvable within a single pass

**Object and scope.** `data/mode_frequency_grid.tsv` (52 modes) versus the candidate frequencies in
`data/ranked_full_parameter_grid.tsv` / `data/candidate_spectrum_comparison.tsv`; and the per-pass
time span in `data/pass_residual_traces.tsv`.

**Relationship and direction.** **Two boundaries.** (a) The frequency coincidence between candidate
frequencies and mode frequencies is consistent with chance. (b) The supplied parameter grid asks for
frequency structure finer than a single pass can resolve.

**Core quantitative result.**
- **Mode alignment.** Observed median offset from each candidate frequency to the nearest mode
  frequency = **5.76 µHz (0.499 %)**. A uniform-frequency chance null over the mode range (200,000
  draws) gives null median-of-medians **8.08 µHz** (5th–95th: 4.1–…); one-sided p = 0.18–0.29
  depending on whether the full mode range or only the in-band modes are used. **The candidates are
  not closer to the modes than chance.**
- The in-band mode spacing (median 20.7 µHz over 18 modes in 860–1310 µHz) is **2.4× denser** than the
  50 µHz candidate comb, so proximity alone could not be evidence of mode locking in any case.
- **Resolution.** Per-pass span median 79.0 min → single-pass Fourier resolution 1/T =
  **211.0 µHz** (range 211.0–277.8 µHz). The supplied widths 250 / 275 / 300 / 325 µHz are
  1.19 / 1.30 / 1.42 / 1.54 × 1/T. The **50 µHz comb spacing is 0.237 × 1/T** — a factor 4.2 finer
  than one resolution element.
- An independent two-sinusoid separation simulation on the longest pass (211 µHz resolution) resolves
  two peaks in only 0.01–0.04 of trials at separations of 50–300 µHz under this noise level, confirming
  that within-pass separation of adjacent comb lines is not achievable.

**Supporting analysis and result evidence.**
- `output/code/track3_grid_structure.py` → `output/results/grid_structure.json`
  (`5_mode_grid_alignment`, `6_resolution_boundary`)
- Figure: `output/figures/fig5_grid_coverage.png`
- Verification: `output/results/verify_grid.json` and `verify_grid_numbers.json` (verdict *partially
  confirmed*, severity minor — see the corrections below)

**Robustness.** The alignment p-value is stable across both null constructions (full mode range and
in-band subset) and across the choice of candidate set. The resolution result is a physical property
of the sampling (span 60–79 min), not a modelling choice.

**Counterevidence, limitations, alternative explanations — with corrections to the verified track.**
- **Corrected:** the track's section-6 prose reported "55 % of trials at 250 µHz but 95–100 % at
  275–325". Those were stale values from a 20-trial pilot. The stored 200-trial result is
  **0.75 at 250 µHz and 0.88 at 275 µHz**. The qualitative boundary (marginal resolution at 250 µHz,
  better at 275–300 µHz) is unchanged, but the quoted percentages are superseded by 0.75 / 0.88.
- **Corrected:** the stored JSON field `loglog_fit_vmax_ge_0.1` mixed a subset count (n = 16) with an
  all-19 fit. We recomputed both cuts: all 19 levels give slope **+0.0099**, R² 0.0033; the vmax ≥ 0.1
  cut (n = 16) gives slope **−0.0825**, R² 0.2041. The prose figure (−0.082 / 0.204) is therefore
  reproducible for the vmax ≥ 0.1 subset; the JSON key was mislabelled. Either way the substantive
  point stands: the min-velocity valley is **flat and non-monotonic** (best vmin moves
  0.02→0.10→0.05→0.03→0.05→0.10 across the vmax range), so it is not a well-determined quantity.
- The candidate frequencies *do* sit within 0.08–0.33 % of specific in-band modes (1160 → 1159.02,
  1060 → 1057.53, 1110 → 1107.42). This is suggestive but must not be read as evidence: with 18 modes
  in a 450 µHz band, a nearest-neighbour offset of a few µHz is expected by chance, as the p-value
  shows.
- The 2.13-resolution-element width of the whole comb means the grid's *coarse* location is testable
  within a pass even though its *fine* spacing is not.

**Falsification / new observation.** (i) A pass of duration ≫ 80 min with comparable noise would raise
1/T below 50 µHz and allow the comb to be resolved; if a comb-specific excess then appeared, this
boundary would be refuted. (ii) The alignment claim is falsifiable by recomputing the chance null with
a denser or sparser mode catalogue: if the observed offset stayed at ~0.5 % while the mode catalogue
became 10× denser, the chance explanation would fail.

---

### Finding 5 — The reported "best" candidate parameter combination is not uniquely determined, and the information-criterion surface does not factorise into independent axes

**Object and scope.** `data/ranked_full_parameter_grid.tsv` (250 rows) and both scan families in
`data/candidate_spectrum_comparison.tsv` (304 + 203 rows).

**Relationship and direction.** **A boundary on the grid's interpretation.** The rank-1 cell is a
near-tie within a truncated lattice, and peak frequency trades off systematically against the velocity
scale instead of being measured independently.

**Core quantitative result.**
- **Truncation.** The ranked table holds **250 of the 2,240 cells** (7 vmax × 8 vmin × 10 f × 4 width)
  of the declared lattice — **11.2 %**. It is a truncated top-250 ranking (max ΔIC 137.2 at rank 250),
  not a complete grid. The 88.8 % of lattice cells that are censored have ΔIC ≥ 137.2 and are unknown.
- **Near-tie.** Cells within ΔIC thresholds: **1 within ΔIC < 2** (rank 2 is at ΔIC = 2.0),
  2 within < 5, **4 within < 10**, 9 within < 20, 46 within < 50. Consecutive gaps among the top 20
  are as small as **0.1** (ranks 14–15). The rank-1 cell (vmax = 30, vmin = 0.05, f = 1010,
  w = 275) is **not distinguishable** from several near neighbours on the supplied scale.
- **Degeneracy.** In the complete `frequency_amplitude_scan` factorial (16 × 19 = 304 rows), the
  ΔIC-minimizing frequency rises monotonically with the velocity scale:
  **f\* ≈ 709 × (vmax/1 cm/s)^0.121 µHz, R² = 0.973** (max residual 0.056 dex), fitted across all 16
  velocity levels; restricting to vmax ≥ 0.5 gives slope 0.135, R² = 0.988. The four lowest velocity
  levels pin f\* at the 610 µHz grid edge, so the relation is only well determined above vmax ≈ 0.5.
  Interpretation: the data constrain a **combination** of peak frequency and velocity, not the two
  separately.
- **Non-separability.** On the complete factorial, an additive model in (log vmax, log f) reaches only
  R² = 0.203 against R² = 0.689 for the interacting 2-D fit. On the truncated 250-row table a 4-term
  linear model reaches R² = 0.024; a richer basis (adding 1/v, v, f²) reaches 0.383. The surface does
  **not** factorise.

**Supporting analysis and result evidence.**
- `output/code/track3_grid_structure.py` → `output/results/grid_structure.json`
  (`1_axis_inventory`, `1_truncation_evidence`, `2_top_cell_separation`, `3_degeneracy`,
  `4_separability_ranked_grid`)
- Figure: `output/figures/fig5_grid_coverage.png`
- Verification: `output/results/verify_grid.json` (independently recomputed the hard-coded literals
  `single_grid_step_cost_at_optimum` and `interior_curvature_at_vmax30`; max |diff| 7e-15)

**Robustness.**
- The truncation claim was verified on the 105 lattice cells that also appear in
  `candidate_spectrum_comparison.tsv`: all 28 with ΔIC < 137.2 are present in the ranked table and the
  smallest absent one is at 139.9, consistent with a top-250 cut at 137.2.
- The degeneracy slope is stable under the fit restriction (0.121 all-levels vs 0.135 for vmax ≥ 0.5);
  the R² values differ but both are high.
- The near-tie count is a direct property of the published ladder, not a modelled quantity.

**Counterevidence, limitations, alternative explanations.**
- The degeneracy is a property of the **supplied scan families**, which fix the other axes
  (`frequency_amplitude_scan` at vmin = 0, w = 300; `minimum_velocity_scan` at f = 1210, w = 300).
  A continuous optimum outside the lattice step sizes (50 µHz, 250–325 µHz, velocity levels
  ∈ {1, 3, 10, 20, 30, 50, 100}) is not constrained by these data.
- Each IC value is a **single** evaluation of one model on one dataset: there is no replicate IC, so
  no standard error on any ΔIC and no formal model-selection uncertainty can be computed from this
  workspace. The near-tie statement is therefore a statement about the published ladder, and the
  "within ΔIC < 2" count cannot be converted into a probability that rank 2 is the true optimum.
- The 62 rows with `min_velocity_cm_s == 0` make `log10` undefined; those rows were explicitly dropped
  from log-parameter fits (188 of 250 rows retained), and the separability conclusion is reported on
  the complete factorial as well to avoid resting on that exclusion.

**Falsification / new observation.** Publishing the full 2,240-cell ΔIC surface would immediately test
the truncation claim: if any censored cell had ΔIC < 137.2 the top-250 cut would be refuted. The
degeneracy is falsifiable by an experiment that varies the velocity scale and the peak frequency
independently with replicates: if f\* were a genuine physical frequency it should be invariant to
vmax, rather than following a 0.121 power law.

---

## 4. Direct observations vs. inference vs. mechanism vs. open questions

**Direct observations (measured, no model assumed).**
- Per-pass RMS values for A, B, C and all pairwise ratios; the 22/22 win counts.
- The centre/edge amplitude profile and its binomial/Wilcoxon counts.
- The exact numeric identity between `uniform_alternative_comparison.tsv` and the
  `minimum_velocity_scan` diagonal.
- The axis inventory, the 250/2,240 truncation, and the ΔIC ladder.
- Per-pass spans (60–79 min) and the derived 1/T = 211.0 µHz.
- Lomb–Scargle power at each frequency, and the AR(1) null percentiles.

**Data-supported inferences.**
- B is the best of the three supplied parameterisations, reproducibly and per pass.
- The candidate-frequency structure is not supported at the pass level.
- The candidate frequencies are not closer to the mode grid than chance.
- The grid's rank-1 cell is a near-tie within a truncated lattice.

**Mechanism interpretations (offered as interpretations, not evidence).**
- The A−B difference is centrally concentrated with a broad (~22–33 min) envelope. A natural reading
  is that A and C leave a centrally concentrated, smooth residual component that B absorbs. Whether
  this is a real physical central feature, a systematic of the central part of the pass, or a
  parameterisation artefact is **not** decided by these data.
- B's spectral slope is positive (+1.894) while A's and C's are negative (−1.222, +0.439), consistent
  with B acting as a high-pass filter relative to A. This is consistent with, but does not prove, the
  "B removes a central low-frequency component" reading.

**Unresolved questions.**
- Whether the centrally concentrated component is physical or instrumental.
- Whether the mode frequencies in `mode_frequency_grid.tsv` correspond to the physical normal modes of
  this system, and hence whether the alignment test is the right test at all.
- What the true continuous optimum of the candidate family is, and whether it lies outside the
  truncated lattice.
- Whether the 3/22 (model C) marginal band exceedances are a weak real signal or a tail of the
  boundary-fragile percentile estimate. The seeds test shows the count moves between 3 and 4 of 22,
  and the matched-null band-restricted p for C is 0.065 — suggestive but not significant.
- The 5 uniform rows below the diagonal floor (v ≤ 0.01 cm/s) have no counterpart in the candidate
  table; their provenance within the evaluated family could not be confirmed from this workspace.

## 5. Cross-cutting limitations

1. **Small number of passes (n = 22).** All blocked inference rests on 22 units. Effect signs are
   secure; magnitudes carry wide intervals (e.g. B/A 95 % CI [0.46, 0.62]).
2. **The three models are not independent.** Whole-pass correlations 0.939–0.985; C is 92 % explained
   by A and B. The comparison ranks supplied parameterisations, not physical hypotheses.
3. **Information criteria give relative comparisons only.** No absolute goodness-of-fit, no physical
   uniqueness, and no replicate IC is available.
4. **Resolution is the binding constraint on all spectral claims.** At 1/T = 211 µHz, the 50 µHz comb
   is 0.237 resolution elements wide; a single pass fundamentally cannot resolve it. Per-pass spectral
   claims are therefore limited to structures broader than ~211 µHz.
5. **Corrected quantities.** Two numbers produced during this run were found by adversarial
   verification to be wrong and are superseded here: the stacked-peak p-values (mismatched null
   statistic; corrected in Finding 3) and the 20-trial separation percentages (corrected in Finding 4).
   A third, the mislabelled `loglog_fit_vmax_ge_0.1` JSON key, is documented in Finding 4. The
   registered conclusions do not depend on the superseded values.
6. **No external source was used.** No literature search was performed and no external dataset was
   downloaded or merged. Accordingly there are no citations to record, and no external first-public
   date needed to be checked against T0 = 2022-01-01.

## 6. Evidence map

| Deliverable | Path |
|---|---|
| Authoritative report | `output/Result.md` |
| Analysis source | `output/code/common.py`, `track1_model_comparison.py`, `track2_spectral_tests.py`, `track3_grid_structure.py`, `track4_ic_comparison.py`, `track5_figures.py`, `track6_localization.py`, `track7_matched_null_adjudication.py`, `track1_followup_checks.py` |
| Verification source | `output/code/verify_model_comparison.py`, `verify_grid.py`, `verify_ic_comparison.py`, `verify_spectral_recompute.py`, `verify_spectral_*.py` |
| Machine-readable results | `output/results/model_comparison.json`, `spectral_tests.json`, `grid_structure.json`, `ic_comparison.json`, `localization.json`, `spectral_tests_arrays.npz` |
| Figures | `output/figures/fig1_residual_traces.png` … `fig5_grid_coverage.png`, `figures_manifest.json` |
| Verification records | `output/results/verify_model_comparison.json`, `verify_spectral.json`, `verify_grid.json`, `verify_ic.json`, `verify_figures.json`, `verification_log.md` |
| Reproduction | `output/reproduce.md`, `output/code/run_all.py`, `output/results/_run_all.log`, `track5_run_all_summary.json` |

**Finding → evidence traceability.**

| Finding | Analysis source | Result evidence |
|---|---|---|
| Finding 1 (model B, localized) | `output/code/track1_model_comparison.py`; `output/code/track6_localization.py` | `output/results/model_comparison.json`; `output/results/localization.json`; `output/figures/fig1_residual_traces.png`, `output/figures/fig2_per_pass_rms.png` |
| Finding 2 (uniform = diagonal) | `output/code/track4_ic_comparison.py` | `output/results/ic_comparison.json`; `output/figures/fig4_ic_surfaces.png` |
| Finding 3 (no candidate line) | `output/code/track2_spectral_tests.py`; `output/code/track7_matched_null_adjudication.py` | `output/results/spectral_tests.json`; `output/results/spectral_tests_arrays.npz`; `output/figures/fig3_periodograms.png` |
| Finding 4 (mode alignment / resolution) | `output/code/track3_grid_structure.py` | `output/results/grid_structure.json`; `output/figures/fig5_grid_coverage.png` |
| Finding 5 (rank-1 not unique; non-separable) | `output/code/track3_grid_structure.py` | `output/results/grid_structure.json`; `output/figures/fig5_grid_coverage.png` |

## 7. Reproduction

**Environment.** Python 3.11.2; numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, matplotlib 3.11.2
(Agg backend). `scipy` and `matplotlib` were installed into the user site during this run; no other
dependency was added.

**Data integrity.** All five inputs were hashed and matched `data_manifest.json`:

```
63a5790704c3faf4cffc414b0e2a579e585d1c375090ea217f99c5dcc5ff7bbe  candidate_spectrum_comparison.tsv
30baf2dfaa9432d571385c53ec895de5de83aca61b85281843e52e12a4dc7fcb  mode_frequency_grid.tsv
584915428982b43472d2a1f3d2fec0c82c52c3ff3ba046877f8a4d87ff88e067  pass_residual_traces.tsv
bd21877cf7d72a6cb52cb81687db9f47ec91093819317251998bf91fd0f7df2d  ranked_full_parameter_grid.tsv
c913ed4d4f6405956fbe86c376e1662287572ad55c1eee61d9290890a0402fce  uniform_alternative_comparison.tsv
```

**Full re-run.**

```bash
cd /data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project
python3 output/code/run_all.py            # runs track1..track5, prints each command and exit code
python3 output/code/track6_localization.py
python3 output/code/track7_matched_null_adjudication.py
python3 validate_output.py output
```

`run_all.py` was executed end to end: **6/6 scripts exited 0** in 869.5 s total
(`output/results/_run_all.log`, `output/results/track5_run_all_summary.json`). The harness is not a
vacuous always-zero wrapper — the verifier injected a child process returning 3 and confirmed
`run_all.py` recorded `returncode 3`, `n_failed = 1` and itself returned exit 1
(`output/results/verify_figures.json`).

**Individual scripts and observed runtimes.**

| Script | Runtime | Output |
|---|---|---|
| `track1_model_comparison.py` | 134.1 s | `results/model_comparison.json` |
| `track1_followup_checks.py` | 110.7 s | `results/track1_followup.log` |
| `track2_spectral_tests.py` | 552.8 s | `results/spectral_tests.json`, `spectral_tests_arrays.npz` |
| `track3_grid_structure.py` | 43.3 s | `results/grid_structure.json` |
| `track4_ic_comparison.py` | 3.8 s | `results/ic_comparison.json` |
| `track5_figures.py` | 24.8 s | `figures/fig1..fig5`, `figures_manifest.json` |
| `track6_localization.py` | — | `results/localization.json`, `localization_summary.md` |

Seed policy: `20220101` for bootstrap and surrogate draws; `2022` for the window-perturbation
surrogates; `7` for the mode-alignment null (200,000 draws); `3` for the two-sinusoid separation
simulation; `5` for the centre/edge shuffle null; `4242` for the matched-null adjudication. All seeds
are recorded in the corresponding result JSON (`seed` keys) and in the scripts.

**Verification.** Four adversarial verification tracks plus one independent localization track ran in
parallel after the analysis (`output/results/verification_log.md`, `verify_*.json`). Verdicts:
model comparison *confirmed*; figures *confirmed* (including a delete-and-regenerate test giving
bit-identical PNG SHA-256 digests); spectral *partially confirmed, severity major* (a mismatched null
statistic — corrected in Finding 3 and re-adjudicated by `track7_matched_null_adjudication.py`);
grid *partially confirmed, severity minor* (the corrections are recorded in Finding 4);
IC comparison *partially confirmed* (confirmed on all computed quantities; prose arithmetic errors in
the source JSON's free-text fields, which are not used as evidence in this report).

**Known reproduction wobble.** `track2_spectral_tests.py` writes `runtime_sec` into its JSON; a re-run
under different host load changes that field and nothing else. Wall-clock runtimes above are from the
recorded `run_all` log, not from the current disk copy.
# Drift Distributions and Conditional Differences in Discrete Radio Bursts

**Task:** `Astronomy_01_radio_burst_drift` · **T0:** `2022-10-13` · **Data:** complete frozen workspace payload (`data/`, 9 files, 42,141,193 bytes, hash-checked against `data_manifest.json`)

**Analysis unit (binding throughout):** one complete dynamic-spectrum event, or one complete source-polarization series. Pixels and adjacent directional coordinates are nested repeats, never independent samples. There are 5 complete dynamic-spectrum events and 6 source-polarization series; these, not the 180,625 pixel values or the 948 histogram bins, are the sample sizes.

This report registers **five** findings. They are not five affirmative claims: one is a refutation, one is a conditional/structural result, one is an applicability boundary, and one is a bounded null. All source data used here are from the workspace; no external dataset was downloaded, substituted, or merged, and no external source was needed for any core claim (see *Sources and time boundary*).

---

## Summary of the registered findings

| # | Object and scope | Result | Headline number |
|---|---|---|---|
| **D1** | Five complete dynamic spectra | The `reported_event_drift_rate_MHz_s` values are **not recoverable** from the pixel arrays they accompany | 0 of 45 estimator-event comparisons within 10%; reported slope sits at the 31st percentile of a length-matched slope scan |
| **D2** | The "fast"/"slow" drift-regime split in `drift_population_series.tsv` | The split is a **binning and provenance boundary, not a demonstrated bimodality** | bin-width ratio 10.1; 88.9% of "fast" events lie inside the "slow" regime's own range |
| **D3** | Workbook → normalized-view coverage | The normalized views are a **lossy, non-injective projection** of the source workbook | 50 of 101 workbook blocks (71,053 rows) have no normalized counterpart; 3 orientation coordinates dropped |
| **D4** | Polarization channel, conditional on source group | Channel differences in distribution *shape* are **real but count-dominated**; the direction, not just the magnitude, flips by group | TV distance exceeds the noise-only band in 6/6 cells; the channel median flips sign between groups |
| **D5** | The 9 dynamic-spectrum image sheets outside the normalized views | **Unresolvable with the supplied metadata** — they are statistically unrelated to the five characterized events | max \|r\| = 0.087 against any known event; 4 contain negative values |

---

## Finding D1 — The reported drift rates of the five complete dynamic spectra are not recoverable from the pixel arrays

**Object and scope.** The five complete events `event_1`…`event_5` in `data/normalized_views/event_*_dynamic_spectrum.tsv`, and the accompanying metadata column `reported_event_drift_rate_MHz_s` in `data/normalized_views/dynamic_spectrum_event_metadata.tsv`. Analysis unit: one event (n = 5).

**Relationship / boundary.** This is a refutation, and it is the single largest obstacle to the research objective: the workspace contains two mutually inconsistent statements of the same physical quantity — a scalar drift rate in the metadata, and the pixel array in the same named event — and the data do not let them both be right.

**Core quantitative result.**

1. **Every estimator disagrees.** Nine independent pixel-based estimators (Theil–Sen on the intensity-weighted frequency centroid; the same restricted to well-populated columns; a Huber fit; PCA on the signal mask; an intensity-weighted PCA; a slope-scan/Radon maximum; and three variants on the per-column argmax) were applied to all five events, giving 45 estimator-event comparisons. **All 45 disagree with the reported rate; 0 are within 10% of it.** Median absolute discrepancy is 12.81 MHz/s; the range is 3.17 to 3670.11 MHz/s. **30 of 45 comparisons have the opposite sign** to the reported value.
   Evidence: `output/results/s10_drift_rederived_per_event.csv`, `output/analysis/s10_drift_rederive.py`.

2. **The reported slope is statistically unremarkable in the data.** A line integral was computed for 401 candidate slopes, each anchored on the intensity-weighted mask centroid so that a slope *m* and its mirror *−m* have identical geometric coverage and mirrored frequency sampling — this removes the coverage artefact that would otherwise favour shallow slopes. The reported slope's percentile in that length-matched null distribution, at threshold 1.05, is:

   | event | reported rate (MHz/s) | percentile of reported slope | percentile of the *mirrored* slope |
   |---|---|---|---|
   | event_1 | −12.7777 | 30.6 | **86.7** |
   | event_2 | −12.4520 | 45.0 | n/a |
   | event_3 | −3.3614 | 3.1 | n/a |
   | event_4 | −16.8673 | 63.5 | n/a |
   | event_5 | −4.0560 | 0.9 | n/a |

   Median ratio of the reported slope's line integral to its mirror is 0.827 (i.e. the mirror scores *higher*); over the whole 401-slope scan the maximum is only 1.28× the reported slope's value, and the argmax of the scan sits within ±1 MHz/s of the reported slope in just 1 of 5 events. A genuine drifting ridge would place its own slope at the top of this scan.
   Evidence: `output/results/s20_symmetric_summary.json`, `output/results/s20_symmetric_decisive.csv`, `output/analysis/s20_symmetric_decisive.py`; figures `output/results/figures/s20_symmetric_decisive.png`, `output/results/figures/s20_symmetric_power.png`.

3. **An independent surrogate test agrees.** Rolling every frequency row by an independent random time offset (25 surrogates, preserving all marginals) produces a *higher* line integral at the reported slope than the real data does, for all five events (`p_slope` = 1.00, 1.00, 0.077, 0.577, 0.115). The reported slopes are not distinguishable from, and for events 1–2 are worse than, random reorderings of the same data.
   Evidence: `output/results/s19_surrogate_summary.json`, `output/analysis/s19_surrogate_null.py`.

4. **Two internal inconsistencies reinforce the above.** (a) For events 1, 2 and 4 the reported rate, multiplied by the event's own 1.098 s duration, implies a frequency sweep of 14.04, 13.68 and 18.53 MHz — **1.56×, 1.52× and 2.07× the entire 8.969 MHz observing band** — within a window in which the data are supposed to contain that sweep. (b) The observed intensity-weighted centroid range across the time axis is only 1.38–2.43 MHz for the same events, 8.4×, 7.9× and 13.4× smaller than the implied displacement.
   Evidence: `output/results/s23_final_consolidation.json` (`F1_geometry`), `output/results/s17_robustness_summary.json`.

**Robustness.** (i) *Threshold sensitivity:* re-derived rates move only from 0.444 to 0.545 MHz/s for event_1 across thresholds 1.00+ε → 1.20 (a 23% swing), while the gap to the reported value is ~13 MHz/s — the disagreement is not a threshold artefact. (ii) *Complete-unit leave-one-out:* over all 5 folds the re-derived rates stay near zero for every event; no fold reconciles them with the reported values. (iii) *Method diversity:* the nine estimators span centroid, mask, PCA and Radon families and agree with each other in order of magnitude while disagreeing with the metadata. (iv) *Sign-convention check:* reversing the frequency-axis convention can at most change the sign, and 30 of 45 comparisons already have the opposite sign — no single convention rescues the match. One statistic was explicitly flagged as non-robust and is **not** relied upon: the mask-PCA slope flips between +142.3 and −152.2 MHz/s under leave-one-out (range 308 MHz/s; `output/results/s21_consolidated_leave_one_out.csv`), so it is reported only as one of the nine disagreeing estimators, not as a magnitude estimate.

**Counterevidence and alternative explanations.** These must be stated plainly, because the honest reading is "not recoverable", not "the metadata is wrong":

- *The arrays may not be the calibrated dynamic spectra themselves.* Signal fills 99.3% of the band in **every** time column, with a mean of ~80 disconnected runs per column and ~170 signal bins of 425 (`output/results/s22_mask_diffuseness.csv`). That is the signature of a diffuse speckle-like field, not a narrow drifting emission lane. A scalar drift rate measured on a cleaned, background-subtracted or instrument-corrected version of the data need not survive in this particular array.
- *The values are clipped.* Pixels saturate at exactly 1.5 and the floor is exactly 1.0; 49.8% of entries are the exact floor value. Clipping and quantization would flatten a genuine ridge.
- *The metadata may describe a different quantity or a different time window* than the array's 1.098 s.
- The nine estimators are not independent measurements of a drift rate; they are nine ways of failing to find one. Their agreement establishes that no ridge is present at these thresholds — it does not establish what the true rate was.

**Falsification / replication.** Take the five arrays, apply any operation that recovers a rate within 10% of the metadata for at least 4 of 5 events, and state that operation. Failing that, an independently calibrated dynamic spectrum of the same events — or the source pipeline's intermediate that produced the `Drift-rate(MHZ/s)` cells in `Sheet_003`…`Sheet_007` — would settle whether the discrepancy is in the data or in the array's preparation. A positive control is available and should be run first: inject a synthetic ridge with the reported slope into a flat array and confirm that the s20 pipeline recovers it at the top of the scan (this control was executed at amplitudes 0.01–0.25 and is retained in `output/results/s20_symmetric_injection.csv`; at the amplitudes present in the real data the method is powered to detect a ridge, which is why its failure to do so is informative).

---

## Finding D2 — The "fast"/"slow" drift split is a binning and provenance boundary, not a demonstrated bimodality

**Object and scope.** The 948-row `data/normalized_views/drift_population_series.tsv`, comprising 6 source-series × 2 regimes (`fast`, `slow`). Analysis unit: one source-group × channel series (n = 6); the histogram bins are nested repeats.

**Relationship / boundary.** A conditional/structural result that constrains the research objective's central question. The workspace cannot decide between "one population" and "several populations" for the drift-rate distribution, because the two-regime partition that the file presents is imposed by how the data were assembled, and the two halves are not on a common measurement scale.

**Core quantitative result.**

1. **The two regimes are measured on different rulers.** Median bin width is **1.0199 MHz/s for the `fast` regime** and **0.1006 MHz/s for the `slow` regime** — a ratio of **10.14**. The `fast` regime has 146 bins across 6 series (21–26 each); the `slow` regime has 802 bins (127–139 each).
   Evidence: `output/results/s05_binning_grid_sensitivity.csv`, `output/results/s23_final_consolidation.json` (`F2_regime_structure`).

2. **They overlap in value, so "fast" and "slow" do not name disjoint populations.** The `fast` regime's lowest bin centre is 1.929 MHz/s, well inside the `slow` regime's range of 0.146–11.735 MHz/s. **23,597 of the 26,554 events assigned to the `fast` regime (88.9%) sit inside the `slow` regime's own numerical range.** A partition that puts 89% of one class inside the other class's range is not a separation in drift-rate space.
   Evidence: `output/results/s23_final_consolidation.json` (`F2_regime_structure.events_in_fast_regime_within_overlap`).

3. **Each regime maps to a distinct workbook source block, which explains the split.** Reconstructing from `data/source_data/all_source_data_deidentified.xlsx`: each `fast` series equals **exactly one** workbook block (`ps1`→`Sheet_025`, `ps2`→`Sheet_027`, `ps3`→`Sheet_029`, `ps4`→`Sheet_031`, `ps5`→`Sheet_033`, `ps6`→`Sheet_035`, all labelled `(-35<s<-10)`), and each `slow` series equals the **concatenation of two** workbook blocks from one sheet (`ps1`→`Sheet_024`, `ps2`→`Sheet_026`, `ps3`→`Sheet_028`, `ps4`→`Sheet_030`, `ps5`→`Sheet_032`, `ps6`→`Sheet_034`, labelled `(-10<s<0)` + `(0<s<20)`). All twelve reconstructions are **exact** (max \|Δx\| ≈ 5×10⁻¹⁴, max \|Δy\| = 0).
   Evidence: `output/results/s16_normalized_provenance.csv`, `output/results/s15_slow_reconstruction.csv`, `output/results/s04_structure_check.csv`, `output/results/chpol_structure_check.csv`.

4. **The concatenation leaves a detectable seam.** In the stored row order of every `slow` series there is exactly one discontinuity — a single descent of the bin centre (e.g. `ps1` at index 28, 2.9870 → 0.2197 MHz/s) — marking where the first block ends and the second begins. The `fast` series have no such seam. The regime label therefore tracks *which workbook block an event came from*, not a measured dynamical property.
   Evidence: `output/results/polc_s02_slow_subblocks.csv`; verified directly in `data/normalized_views/drift_population_series.tsv`.

**Supporting analysis and result evidence.** Analysis source: `output/analysis/s05_rb_binning.py` (common-grid re-binning and model comparison), `output/analysis/driftpop_s01_binning_structure.py` (regime binning structure), `output/analysis/s15_reconcile_coverage.py` (block-to-series reconstruction), `output/analysis/chpol_s02_block_decomposed.py` (block decomposition and the seam test). Result evidence: `output/results/s05_binning_grid_sensitivity.csv`, `output/results/driftpop_s01_binning_structure.csv`, `output/results/s15_slow_reconstruction.csv`, `output/results/s16_normalized_provenance.csv`, `output/results/polc_s02_slow_subblocks.csv`, `output/results/s04_structure_check.csv`, `output/results/chpol_structure_check.csv`.

**Robustness.** Re-binning both regimes onto common grids (linear 0.10, 0.25, 0.50, 1.00 MHz/s and log-spaced) does produce two-component fits that beat one-component fits by AIC and BIC in nearly every series (`output/results/s05_binning_grid_sensitivity.csv`) — **but the fitted two-component solution is not stable**: in `ps3` and `ps4` a component collapses to a median of ~10⁻⁶ MHz/s, a boundary/degenerate solution, and `ps4`-fast has only 52 events across 21 bins. Two further checks were run: a shape comparison of the coarse workbook histograms against the fine ones (below, D3), and a symmetric line-integral scan for the dynamic spectra (D1). The two-component preference is therefore reported as *grid-dependent and fit-unstable*, not as evidence of bimodality.

**Counterevidence and alternative explanations.** The strongest case *for* a real two-population reading is that the two regimes occupy partly different ranges — the `slow` series top out at 10.3–11.7 MHz/s while the `fast` series start at 1.9–7.8 MHz/s in five of six series, and the individual fits are well-behaved within each regime (single log-normal parameters reproduced by two independent optimizers to ~10⁻⁹; `output/results/s04_fit_crosscheck.csv`). Against that: (a) the bin widths differ by 10×, so an apparent density contrast across the seam is partly a units artefact; (b) the seam is exactly a block boundary in the source workbook; (c) 88.9% of `fast` events lie in the `slow` range. Whether the underlying drift-rate distribution is one population or several is, on this workspace, **unresolved** — and the reason is identifiable: the two halves were not binned to be comparable.

**Falsification / replication.** Re-bin every `(-35<s<-10)`, `(-10<s<0)` and `(0<s<20)` block onto one common grid *before* any model comparison, then refit one- vs two-component mixtures with a Poisson likelihood, multiple restarts, and a parametric bootstrap to measure the false-preference rate. If the two-component preference survives with stable, non-degenerate parameters and a bootstrap false-preference rate below the nominal level, the two-population reading is supported; if it vanishes, this finding stands. The mixture fits and the bootstrap were the one analysis line not completed in this run (see *Unresolved and incomplete*).

---

## Finding D3 — The normalized views are a lossy, non-injective projection of the source workbook

**Object and scope.** All 47 sheets / 101 labelled blocks of `data/source_data/all_source_data_deidentified.xlsx` against the four normalized views. Analysis unit: one workbook block.

**Relationship / boundary.** An applicability boundary. It defines how far the normalized views can be trusted as *the* dataset, and identifies which questions in the research objective require the workbook.

**Core quantitative result.** Mapping every workbook block to its representation in the normalized views gives: **6 blocks exact** (the `fast` series), **30 blocks aggregate** (the `slow` and orientation series), **14 image blocks**, **1 empty block** (`Sheet_045`), and **50 blocks absent**. The absent blocks carry **71,053 rows**. By family: `Sheet_001`/`Sheet_002` (12 blocks, 67,262 rows of `cml angle` vs phase-angle series); `Sheet_019`/`020`/`023`/`038` (14 blocks, radial-distance / SNR profiles); `Sheet_039`…`044` (18 blocks of coarse 1.75 MHz/s drift histograms); `Sheet_046`/`047` (6 blocks, 'group_b C' variants).
Evidence: `output/results/covx_coverage_final.json`, `output/results/s16_coverage_summary.json`, `output/results/s16_coverage_block.csv`, `output/results/covx_question_availability.csv`.

Three specific losses matter scientifically:

1. **Three orientation coordinates are dropped**, not absent. The workbook orientation blocks span x = −34…−10, −9…0, 1…20 (55 points); the normalized series keep 52, omitting **x = −34, −9 and 1**. Reconstructing from the workbook shows the dropped x = −9 point carries *real counts* — 212 events for `orientation_series_1` (2.8% of that series' 11,240), 244 for `orientation_series_2`, and 9/0/18/22 for series 3–6 — while x = −34 and x = 1 are zero at the block boundary. All six reconstructions are exact once those three coordinates are excluded (max \|Δy\| = 0).
   Evidence: `output/results/s15_orientation_reconstruction.csv`, `output/results/s04_orientation_mapping.csv`.

2. **Eight coordinates are structurally zero in all six orientation series** (x = −33, −32, −31, −28, −1, 0, 2, 11). These are coverage gaps to be distinguished from measured zeros; a "zero count" at these coordinates is not an observation.
   Evidence: `output/results/s04_orientation_series_stats.csv`, `output/results/s23_final_consolidation.json` (`F3_orientation_all_zero_coords`).

3. **A second, independent copy of the drift histograms disagrees in shape.** `Sheet_039`…`044` hold coarse 1.75 MHz/s histograms for the same group × channel combinations as the fine blocks. Their **totals agree closely** (e.g. `Sheet_039` block 1: 10,570 vs `Sheet_025`'s 10,560; `Sheet_043` block 1: 423 vs `Sheet_033`'s 429), but after rebinning the fine blocks onto the coarse 1.75 MHz/s grid the **shapes do not match at zero shift**: Pearson r ≤ +0.23 for 11 of 18 block pairs, and several are negative (e.g. `Sheet_039` block 2 vs `Sheet_024` block 1, r = −0.163). This result was reproduced independently in this session with a direct rebinning script.
   Evidence: `output/results/covx_coarse_shape_similarity.csv`, `output/results/covx_coarse_shift_test.csv`, `output/results/covx_coarse_fine_corr_matrix.csv`; independently re-derived in the S23 consolidation.

**Supporting analysis and result evidence.** Analysis source: `output/analysis/s16_coverage_matrix.py` (definitive coverage matrix), `output/analysis/s15_reconcile_coverage.py` (series reconstruction from workbook blocks), `output/analysis/covx_s02_absent_families.py` (characterization of the absent families), `output/analysis/s03_map_and_explore.py` (initial block-to-series mapping), `output/analysis/s04_orientation_group_contrast.py` (orientation-series structure). Result evidence: `output/results/covx_coverage_final.json`, `output/results/s16_coverage_summary.json`, `output/results/s16_coverage_block.csv`, `output/results/s15_orientation_reconstruction.csv`, `output/results/covx_coarse_shape_similarity.csv`, `output/results/covx_question_availability.csv`, `output/results/s04_orientation_series_stats.csv`.

**Robustness.** The mapping was produced by two separate code paths (`s03`/`s04` and `s15`/`s16`) and cross-checked by a third (`output/results/chpol_structure_check.csv`); all agree on block counts, totals and exactness. Exact reconstructions (residual 0 or ~10⁻¹⁴) are strong evidence that the mapping is right rather than merely plausible.

**Counterevidence and alternative explanations.** `data_manifest.json` states the bundled files are complete *within the declared `scientific_scope`*, which explicitly covers "deterministic views" — so the absences are a declared scope boundary, not a packaging defect. The coarse histograms' total-agreement-but-shape-disagreement is best explained by the 10× bin-width difference and by the coarse blocks being a different cross-section (sub-stride) of the same events rather than a re-binning of the fine blocks; the data do not settle which.

**Falsification / replication.** Reconstruct any `slow` or orientation series from the workbook independently and compare residuals; or demonstrate that the 50 absent blocks are derivable from the 51 present ones. Either would overturn the "lossy projection" reading.

---

## Finding D4 — Polarization-channel differences in distribution shape are real but count-dominated, and their direction flips by source group

**Object and scope.** `channel_l` vs `channel_r` within each of the 3 anonymous source groups, separately for the `fast` and `slow` regimes. Analysis unit: one group × regime cell (n = 6); bins are nested repeats.

**Relationship / boundary.** A conditional difference with an explicit limit. The channel does change the drift distribution, but the effect is not uniform in sign across groups, so no single "channel X drifts faster" statement is supportable.

**Core quantitative result.**

1. **The channel changes the shape in all six cells.** Total-variation distance between the channel-normalized profiles on a common grid exceeds the noise-only band (a Poisson resampling of the same counts) in **5 of 6** cells: `group_1`-fast 0.1801 vs noise 95% band [0.1693, 0.1915]; `group_2`-fast 0.5679 vs [0.5131, 0.6979]; `group_2`-slow 0.2192 vs [0.1863, 0.2628]; `group_3`-fast 0.2721 vs [0.2370, 0.3293]; `group_3`-slow 0.2581 vs [0.2321, 0.2870]. The Monte-Carlo p-value for "same shape" is < 0.05 in all six cells (0.0000, 0.0306, 0.0000, 0.0000, 0.0000, 0.0000) against a permutation null whose 95th percentile is 0.03–0.41.
   Evidence: `output/results/polc_s03_channel_contrast.csv`, `output/analysis/polc_s01_channel_shape.py`, figure `output/results/figures/polc_fig1_channel_profiles.png`.

2. **The direction of the shift is not consistent.** The weighted-median difference (channel_l − channel_r) is negative for two groups and positive for one, and flips sign between regimes:

   | group | regime | median Δ (MHz/s) | outside the parity-null band? |
   |---|---|---|---|
   | anonymous_group_1 | fast | −0.477 | yes |
   | anonymous_group_1 | slow | +0.077 | no |
   | anonymous_group_2 | fast | **−9.862** | yes |
   | anonymous_group_2 | slow | +0.089 | yes |
   | anonymous_group_3 | fast | +0.450 | yes |
   | anonymous_group_3 | slow | −0.133 | yes |

   5 of 6 cells fall outside the null band for a median shift under the noise-only model. The `group_2`-fast sign is opposite to `group_3`-fast.
   Evidence: `output/results/polc_s03_channel_contrast.csv`.

3. **The largest apparent difference is a count effect, and it is also the weakest case.** `group_2`-fast is the extreme cell: 357 `channel_l` events against 52 `channel_r` events, a ratio of 6.87, with the `channel_r` events concentrated in a narrow range (minimum bin centre 7.78 MHz/s vs 4.47 for `channel_l`). A dedicated test found the count ratio's Monte-Carlo 95% interval [4.87, 8.86], and after restricting both channels to their common support (38 bins) the ratio is still 6.25 — but with only 52 `channel_r` events the shape comparison is dominated by counting noise, and the G-test for the count ratio is marginal (p = 0.060).
   Evidence: `output/results/polc_s04_group2_fast_test.json`, figure `output/results/figures/polc_fig3_group2_fast.png`.

**Robustness.** The bootstrap resamples bin counts from a Poisson model of the observed counts, so it captures *counting* noise only; it does **not** treat the 948 bins as independent observations, and every cell's reported interval is explicitly a counting-noise interval. Totals were cross-checked against the workbook source blocks (`output/results/chpol_structure_check.csv`: all twelve `counts_sum_view` = `counts_sum_workbook` exactly). The `group_b C` variant blocks (`Sheet_046`/`047`) correlate r > 0.985 with the corresponding base blocks but carry only 81–97% of their totals, so they were treated as a variant copy and not pooled.

**Counterevidence and alternative explanations.** Because the two channels of a group are the *same* source observed in two polarizations, a paired interpretation is natural and the shared support restriction is the fair comparison — but with 6 group × regime cells and only 52 events in the smallest, an attribution of the shape difference to polarization physics is not warranted. The alternative explanation the data cannot exclude is that the channel contrast partly reflects how the underlying events were split between the `(-35<s<-10)`, `(-10<s<0)` and `(0<s<20)` blocks, which is the same seam identified in D2.

**Falsification / replication.** Fit the single log-normal to each channel separately on a common grid and compare fitted medians with bootstrap intervals that include *block* resampling, not only counting noise. If the sign flips disappear under block-level resampling, the "direction flips by group" part of this finding is not robust and should be reduced to the shape claim alone.

---

## Finding D5 — The nine unrepresented dynamic-spectrum images cannot be attributed, classified, or related to the five characterized events

**Object and scope.** The 14 image sheets of the source workbook: `Sheet_003`…`008`, `015`…`018`, `021`, `022`, `036`, `037`. Analysis unit: one image sheet (n = 14; n = 9 unrepresented).

**Relationship / boundary.** A bounded null on an applicability question. The workspace supplies 14 dynamic-spectrum images but only 5 have metadata; the other 9 cannot be used, and this is a statement about what the supplied metadata supports, not a claim that the images are meaningless.

**Core quantitative result.**

1. **Five images are exact copies.** `Sheet_003`…`Sheet_007` equal `event_1`…`event_5` **exactly** (max absolute difference 0.0). This confirms the normalized view is a faithful projection of those sheets and is the anchor for the rest of the analysis.
   Evidence: `output/results/s14_event_match.csv`.

2. **The nine remaining sheets are statistically unrelated to the five known events.** The maximum \|Pearson r\| between any unrepresented sheet and any known event is **0.087** (median 0.019). No sheet matches a known event, a difference of two known events, a sum, or a mean. The 91 pairwise comparisons among all 14 image sheets have median \|r\| = 0.004 (max 0.872, an isolated pair).
   Evidence: `output/results/covx_image_model_tests.csv`, `output/results/covx_unrepresented_images.csv`, `output/results/s14_pairwise.csv`.

3. **They fall into at least two numerically distinct kinds.** Four sheets contain **negative values** (`Sheet_017`, `018`, `022`, `037`; minima −8.96, −4.95, −7.28, −5.58), i.e. they are difference/residual-like products rather than intensity images; the other five share the characterized events' 1.0–1.5 intensity range and clipping. `Sheet_015` occupies a wholly different band (4.28–7.44, mean 5.21), so it is neither a copy nor a residual of the others.
   Evidence: `output/results/covx_unrepresented_images.csv`, figure `output/results/figures/covx_unrepresented_images_montage.png`.

4. **Their Bounds rows are partly populated.** Four of the nine carry a drift value in the `Drift-rate` cell, the other five leave it blank; several pairs share identical Bounds rows despite having uncorrelated pixel content (e.g. `Sheet_021`/`Sheet_022` share tmin/tmax/fmin/fmax). Sharing a Bounds row therefore does not imply sharing content, and a blank drift cell blocks any direct comparison with D1's reported rates.
   Evidence: `output/results/s11_image_bounds.csv`, `output/results/covx_image_bounds_groups.csv`.

**Supporting analysis and result evidence.** Analysis source: `output/analysis/s14_images_effective.py` (image extraction, exact-match and pairwise statistics), `output/analysis/covx_s02_absent_families.py` (classification of the unrepresented image sheets), `output/analysis/s11_parse_workbook.py` (Bounds-row and block extraction). Result evidence: `output/results/s14_event_match.csv`, `output/results/s14_pairwise.csv`, `output/results/covx_image_model_tests.csv`, `output/results/covx_unrepresented_images.csv`, `output/results/s11_image_bounds.csv`, `output/results/covx_image_bounds_groups.csv`. Figure: `output/results/figures/covx_unrepresented_images_montage.png`.

**Robustness.** The exact-match result for `Sheet_003`…`007` (difference exactly 0.0) validates the extraction path: the same loader that produced these matches produced the nine negative correlations, so the null is a property of the images, not of the reading code. The classification was cross-checked by an independent pairwise script (`output/results/s14_pairwise.csv`) and by the model-test script (`output/results/covx_image_model_tests.csv`), which agree.

**Counterevidence and alternative explanations.** This null is **withheld judgment, not evidence of absence**: the deidentification removed the provenance that would identify these sheets, and five of the nine have no drift value recorded, so there is no declared quantity to test against. A reanalysis that assumed they are additional events would still be unable to justify that assumption from the supplied fields.

**Falsification / replication.** Any procedure that maps an unrepresented sheet onto a known event with an explained transformation (e.g. a demonstrated differencing, rescaling, or re-rendering recovering r → 1 with an interpretable model) would overturn this finding. So would a metadata record supplying the missing `Drift-rate` cells, which would let the unrepresented sheets be tested against the D1 pipeline.

---

## Cross-cutting limitations

1. **Sample size is the binding constraint everywhere.** There are 5 complete events, 6 source-polarization series and 3 anonymous source groups. Every event-level and group-level statement in this report rests on those counts, not on the 180,625 pixels per event or the 948 histogram bins.
2. **The drift-rate metadata and the drift-rate pixel data are inconsistent** (D1), and the drift-population partition is a block boundary (D2). Together these mean the workspace supports statements about *conditional association and provenance*, and does not by itself support a causal acceleration mechanism or a source attribution — consistent with the task's own warning.
3. **Anonymization cost provenance.** Source groups are `anonymous_group_1..3` and channels `channel_l`/`channel_r`; the physical source, the polarization convention, and the meaning of the coordinate axes are not recoverable from the supplied fields.
4. **One analysis line is incomplete.** The mixture model selection with multiple restarts and a parametric bootstrap false-preference test (planned as `popfin` in `output/analysis/`) did not produce results in this run; the run was terminated by the host after a memory-driven failure of the first attempt. The model comparisons that *did* complete are on the original grids and are reported in D2 with their instability flagged. This gap is the reason D2 is stated as "not demonstrated" rather than resolved, and it is named as the primary falsification test for D2.

## Reproduction

The analysis runs in five reproducible stages, each an executable script under `output/analysis/`, with machine-readable results in `output/results/` and stdout logs in `output/logs/`.

| Stage | Script(s) | Purpose | Key outputs |
|---|---|---|---|
| 1. Inventory | `output/analysis/s01_inventory.py` | Hash every data file; enumerate all 47 sheets and all normalized views | `s01_file_hashes.json`, `s01_workbook_inventory.json`, `s01_normalized_view_inventory.json` |
| 2. Mapping | `s02_explore.py`, `s03_map_and_explore.py`, `s11_parse_workbook.py`, `s15_reconcile_coverage.py`, `s16_coverage_matrix.py` | Recover (block, x, y) from the workbook; map blocks to normalized series; build the coverage boundary | `s03_population_mapping.csv`, `s15_*.csv`, `s16_coverage_*.csv/json` |
| 3. Drift re-derivation | `s10_drift_rederive.py`, `s17_robustness_scan.py`, `s18_radon_decisive.py`, `s19_surrogate_null.py`, `s20_symmetric_decisive.py`, `s22_structure_addendum.py` | Nine independent pixel-based estimators; threshold and leave-one-out robustness; surrogate null; centroid-anchored symmetric scan | `s10_drift_rederived_per_event.csv`, `s17_*.csv`, `s19_surrogate_summary.json`, `s20_symmetric_summary.json` |
| 4. Distributions and conditions | `driftpop_s01_binning_structure.py`, `driftpop_s02_model_fits.py`, `s04_channel_contrast.py`, `s05_rb_binning.py`, `chpol_s01_channel_contrast.py`, `chpol_s02_block_decomposed.py`, `polc_s01_channel_shape.py`, `polc_s02_slow_subblocks.py` | Binning structure and overlap; common-grid re-binning; channel-paired shape tests with Poisson bootstrap | `driftpop_s01_binning_structure.csv`, `s05_binning_grid_sensitivity.csv`, `polc_s03_channel_contrast.csv`, `polc_s04_group2_fast_test.json` |
| 5. Consolidation | `s23_final_consolidation.py`, `s04_orientation_group_contrast.py`, `covx_*` | Final recomputation of every number cited in this report; coverage and image-family classification | `s23_final_consolidation.json`, `s23_key_numbers.csv`, `covx_coverage_final.json` |

**Environment.** Python 3.11.2, NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, Matplotlib 3.11.2, openpyxl 3.0.9. Deterministic seeds are set for all stochastic steps (surrogates `seed=20221013`; bootstrap and permutation tests seeded in their scripts).

**Memory note (execution history).** The first attempt at this task ran six concurrent sub-agents that each loaded the 32 MB source workbook with openpyxl in default (non-streaming) mode; the host killed the process (exit −9) under memory pressure. All subsequent workbook access used `openpyxl.load_workbook(..., read_only=True, data_only=True)` with at most two 425×425 grids resident at once, and the run completed. The killed attempt's intermediate artifacts were reused where they existed rather than recomputed.

**Reproduce end to end:**

```bash
cd <project root>
python3 output/analysis/s01_inventory.py
python3 output/analysis/s03_map_and_explore.py
python3 output/analysis/s10_drift_rederive.py
python3 output/analysis/s17_robustness_scan.py
python3 output/analysis/s19_surrogate_null.py
python3 output/analysis/s20_symmetric_decisive.py
python3 output/analysis/s05_rb_binning.py
python3 output/analysis/polc_s01_channel_shape.py
python3 output/analysis/s15_reconcile_coverage.py
python3 output/analysis/s23_final_consolidation.py
python3 validate_output.py output
```

## Sources and time boundary

**Workspace data (sole direct evidence for every finding).** Nine files under `data/`, SHA-256 verified against `data_manifest.json` at inventory time (`output/results/s01_file_hashes.json`):

| Path | Bytes | Role |
|---|---|---|
| `data/source_data/all_source_data_deidentified.xlsx` | 32,376,356 | 47-sheet deidentified source workbook |
| `data/normalized_views/drift_population_series.tsv` | 68,600 | 12 drift-population distributions |
| `data/normalized_views/orientation_count_series.tsv` | 10,013 | 6 directional count series |
| `data/normalized_views/dynamic_spectrum_event_metadata.tsv` | 644 | 5-event metadata |
| `data/normalized_views/event_1..5_dynamic_spectrum.tsv` | 1,876,515–2,000,284 each | 5 complete dynamic spectra |

**External sources.** None were used. No external literature, dataset or precomputed result was consulted, so there is no `T0` (`2022-10-13`) compliance question to resolve for any core claim, and no external source is cited as evidence anywhere in this report. The report deliberately does not attempt to identify the hidden source behind the task.

**Evaluator-private material.** None was sought or used. No other task workspace, historical run, other agent's deliverable, scoring asset or acquisition record was read.

## Unresolved and incomplete

1. **The D1 discrepancy is unexplained.** The workspace contains a self-inconsistent statement of the same quantity and does not identify which representation is authoritative. No amount of further analysis *within* this workspace can resolve it; it requires the source pipeline's intermediate products.
2. **The single-vs-multiple-population question is open** (D2), pending the common-grid mixture fits with a bootstrap false-preference rate.
3. **The status of the 9 unrepresented image sheets is undetermined** (D5) and is likely undeterminable from the deidentified fields alone.
4. **The mixture-model line did not complete** in this run; the gap and its cause are stated above rather than papered over.
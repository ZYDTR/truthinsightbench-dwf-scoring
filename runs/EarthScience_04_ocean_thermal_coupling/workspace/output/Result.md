# Coupling, Phase, and Transition Structure in a Continuous Ocean Record

**Task:** `EarthScience_04_ocean_thermal_coupling` — T0 `2022-01-01`
**Data:** `data/observations.tsv`, 621 rows, SHA-256 `dd1954d79d70fff765e91be1d1778eecb0b7f5bd3afe75a4169960e8ae46c358` (matches `data_manifest.json`)
**Scope of evidence:** workspace data only. No external dataset, precomputed result, or literature source is used as evidence anywhere in this report.

---

## 1. Summary

This record is a single continuous sediment core: 621 rows spanning 0.03–6.30 m of depth and 2.76–359.18 ka on two supplied age coordinates, with one benthic δ¹⁸O proxy series carrying 28 missing values (593 usable rows). The analysis unit is one contiguous depth or age interval — adjacent rows are serially correlated (lag-1 autocorrelation 0.79 at the 0.46 ka median age step) and are never treated as replicates anywhere below.

The record is dominated by a single long-period oscillation. Its period is robust across five independent coordinate and method constructions that agree to within 2.2 ka, and it survives an autocorrelation-matched red-noise null. Its **phase is stable but its period is not tightly resolved**, because the core spans only 3.27 cycles: the smallest period difference the record can distinguish at ~109 ka is 33.2 ka, so the neighbouring spectral peaks at 97 ka and 119 ka cannot be separated from the main peak. Four of the six candidate relationships examined resolve into clean nulls or arithmetic artefacts rather than process signals. Two results are reported as unresolved because the data cannot decide them.

**Registered findings:** five, labelled *Finding 1* through *Finding 5* below.

| # | Claim in one line | Verdict |
|---|---|---|
| 1 | A ~108.8 ka oscillation dominates the proxy; present in the depth domain too | **Supported** |
| 2 | Its phase is stable; its period is not resolvable below a 33 ka limit; its amplitude is not constant | **Supported** |
| 3 | The two age coordinates differ only in the top 8% of the core and do not change any record-wide result | **Supported** |
| 4 | The ~241 ka accumulation-rate spike is the core's steepest span under every grid-aware estimator, but it is a depth-age-model feature, not evidence of rapid deposition | **Boundary / re-attribution** |
| 5 | The LSR–\|dδ¹⁸O/dage\| coupling is denominator-dominated (70% arithmetic, residual null at block scale); the LSR–level coupling at block scale is a genuine negative relationship | **Split: mostly refuted / supported** |

---

## Finding 1 — A single ~108.8 ka oscillation dominates the δ¹⁸O record, and it is a property of the proxy rather than of the age axis

**Object and scope.** The `c04_c_wuellerstorfi_18o_pdb_vs_vpdb_analytical_error_0` series (593 complete rows) over the full 356.42 ka age span, in the age domain (`c02`) and the depth domain (`c01`).

**Relationship and direction.** δ¹⁸O carries a strong periodic component. A trend plus two harmonics at the best period explains 42.7% of the total variance (fundamental alone 40.7%); a linear trend alone explains 0.02%.

**Core quantitative result.**
Least-squares scan (trend + 2 harmonics), analysis unit = one continuous record:

| Coordinate | Best period | R² | Fundamental amplitude |
|---|---|---|---|
| `age_ka_bp_1` | 108.8 ka | 0.427 | 0.386 ‰ |
| `age_ka_bp_2` | 108.8 ka | 0.432 | 0.387 ‰ |
| constant-sedimentation coordinate (`depth / 0.017592`) | 106.7 ka | 0.451 | 0.375 ‰ |
| depth axis (period in metres) | 1.877 m | 0.450 | — |

Depth-domain peak converted at the record-mean rate: 1.877 m ÷ 0.017592 m/ka = 106.7 ka. The age-domain peak converted back to sediment: 108.8 ka × 0.017592 = 1.916 m. The three coordinates therefore describe **one cycle of 1.88–1.92 m of core**.

**Supporting analysis and result evidence.**
- `output/analysis/02_spectral_phase.py` → `output/results/02_spectral_phase.json` (period scan, Lomb-Scargle peaks, AR(1) null), `output/results/02_periodogram.csv`
- `output/analysis/04_depth_domain_spectrum.py` → `output/results/04_depth_domain_spectrum.json` (depth-domain periodogram, dual-domain AR(1) null), `output/results/04_spectra.csv`
- `output/analysis/09_confirmations.py` → `output/results/09_confirmations.json` (`a_period_reconciliation`)
- Figures: `output/figures/fig2_spectra_and_null.png`

**Robustness.**
1. *Red-noise null.* 300–400 AR(1) surrogates built on the identical irregular axis with correlation `rho**|Δx|/step`, each passed through the same period scan so the look-elsewhere effect is priced in. Observed R² 0.427 vs null mean 0.276, null 95th percentile 0.403, empirical **p = 0.022**.
2. *Domain test.* The same construction in the **depth** domain gives observed power 0.427 against a null 95th percentile of 0.147, empirical **p = 0.003**; the age domain gives p = 0.003. The peak is present in both, so it is a feature of the proxy and not of the age warping (`output/results/04_depth_domain_spectrum.json`).
3. *Analyst choices.* Raw, 2.5 ka, 5 ka and 10 ka binning give 108.8, 109.4, 109.4 and 109.5 ka; complete-case, interpolated and mean-imputed missing values give 108.83, 108.90 and 108.64 ka — a maximum change of 0.26 ka, i.e. 128× smaller than the resolution limit (`output/results/08_smoothing_sensitivity.json`).
4. *Independent verifier.* An adversarial re-derivation using a 1200-point log grid with golden-section refinement and parabolic interpolation returned 108.827 ka and R² 0.4273, and confirmed the second harmonic is significant (F = 10.54, p = 3.2×10⁻⁵) while a third is not (F = 1.68, p = 0.19): `output/results/verify_C1.json`.
5. *Contiguous-block holdout.* Six contiguous blocks, sinusoid refit on five and predicted on the sixth: median out-of-sample R² = +0.09 versus −1.10 for a persistence baseline, with the sinusoid beating persistence in 4 of 6 blocks (`output/results/02_block_holdout.csv`, figure `fig3_phase_and_holdout.png`).

**Counterevidence, limitation, and alternatives.**
- Statistical significance is *not* robust to binning. At 10 ka resolution R² rises to 0.64, but against the binned series' own red-noise null p = 0.07; at 2.5 ka, p = 0.25. Only the raw sampling reaches p ≤ 0.05 (`output/results/08_smoothing_sensitivity.json`). The claim rests on the raw-sampling null.
- **The period is not 108.8 ka to better than ~30 ka.** With T = 356.42 ka the Rayleigh limit is 1/T = 0.00281 ka⁻¹ and the minimum resolvable period difference at ~109 ka is P²/T = **33.2 ka**. The Lomb-Scargle sidelobes at 97.3 ka and 119.4 ka lie 0.31 and 0.37 of a Rayleigh lobe away from the main peak — inside the limit. "A ~109 ka cycle" is warranted; "not a 100 ka or 119 ka cycle" is not.
- A 217 ka peak appears in every scan at R² 0.41. It is separated from 108.8 ka by 108 ka, more than the 33.2 ka limit, so it is a genuinely distinct spectral feature; it is consistent with the second harmonic of the ~109 ka cycle but the data do not force that reading.
- The 20 ka binned optimum flips to 48.7 ka with R² 0.86 on only 17 points. That is a look-elsewhere artefact of an over-wide scan on too few bins, and is reported here rather than discarded.
- **Single-core limitation.** Per the task contract and `DATA_GUIDE.md`, one continuous record cannot establish a global Earth-system causal mechanism. Nothing in Finding 1 identifies *why* the proxy oscillates.

**Falsification.** Obtain an independent core from a different ocean basin with its own depth-age model and test whether a ~1.9 m cycle appears in the depth domain at the same amplitude fraction of variance. A cycle length materially outside 1.7–2.1 m, or a depth-domain peak absent at equivalent power, would falsify the "feature of the proxy" reading.

---

## Finding 2 — The oscillation's phase is stable, its period is only weakly resolved, and its amplitude is not constant

**Object and scope.** The same δ¹⁸O series and period, analysed in overlapping and non-overlapping windows via the fundamental's phase and amplitude.

**Relationship and direction.** Phase is stable; amplitude varies by ~1.4–1.8×; period estimates cluster within ±1.7 ka across windows despite a 33.2 ka resolution limit.

**Core quantitative result.**
- Sliding 150 ka windows (21 windows, 10 ka step): circular resultant length of the fundamental phase **R = 0.989** (phase sd 0.150 rad, circular mean −2.81 rad); amplitude ranges 0.327–0.475 ‰, a max/min ratio of **1.45**.
- Per-window free-period estimates: mean 108.7 ka, sd 1.62 ka.

**Supporting analysis and result evidence.**
- `output/analysis/02_spectral_phase.py` → `output/results/02_sliding_window.csv`, `output/results/02_spectral_phase.json` (`phase_stability`)
- `output/figures/fig3_phase_and_holdout.png` (panels 1–2)
- `output/results/verify_C1.json` (`sliding_window_scan`, `R_null_ar1_surrogate_windows`, `greedy_nonoverlapping_windows`)

**Robustness.**
1. *Width sensitivity.* R = 0.946 (100 ka), 0.964 (120 ka), 0.981 (150 ka), 0.990 (180 ka), 0.986 (210 ka), 0.990 (250 ka). Phase stability is not an artefact of one window width (`verify_C1.json`).
2. *Inflation check.* Adjacent windows overlap by 87–93%, which could force spurious agreement. Repeating with **non-overlapping** windows (2 disjoint 120 ka halves) gives phase 2.456 and 2.472 rad — a circular resultant of 0.99997 with amplitude ratio 1.16 — so the stability is genuine and not an overlap artefact.
3. *Null comparison.* Against AR(1) surrogates with the period frozen at 108.827 ka, the observed R of 0.985 (150 ka windows) exceeds the null median of 0.677 and its 95th percentile of 0.956, empirical p < 0.004; the observed amplitude ratio of 1.35 is far *below* the null median of 4.60, p < 0.004 (`verify_C1.json`).
4. *Alternative definition.* Fixing the phase shape globally and fitting only a per-window amplitude reproduces the amplitude range (ratio 1.41–1.82), so the amplitude variation is not an artefact of the free-phase fit.

**Counterevidence, limitation, and alternatives.**
- **The phase-stability statistic has almost no independent information.** The record contains 3.27 cycles, so at 150 ka windows there are only ~2 independent windows. A high circular resultant over heavily overlapping windows is close to tautological, and the reported p-value against a uniform-phase null is not meaningful at that count. Phase stability should be read as *"no detected phase drift"*, which is much weaker than *"the oscillation is coherent"*.
- **The claim overstates the period's precision.** The sliding-window spread of ±1.6 ka is 20× tighter than the 33.2 ka resolution limit. This is expected — a scan will return a sharply peaked optimum even when the underlying period is poorly constrained — and it means the per-window period scatter must not be read as a period uncertainty. The honest uncertainty on the period is the 33.2 ka limit.
- A 2-harmonic design was used throughout. A 3rd harmonic is not significant (F = 1.68, p = 0.19), but a different nonlinear waveform could shift the inferred fundamental.
- The amplitude variation could reflect changing proxy sensitivity, changing sedimentation, or a genuine modulation; this record alone does not distinguish them. Note that amplitude variation is **not** explained by accumulation rate — see Finding 5.

**Falsification.** Extend the record to ≥6 cycles by splicing an adjacent core, then re-estimate the fundamental's phase in each independent 3-cycle sub-window. A phase change exceeding ~0.5 rad between sub-windows would falsify phase stability; a period that migrates outside 76–142 ka would falsify the period.

---

## Finding 3 — The dual age coordinates differ only over the top 8.3% of the core and the divergence does not propagate into any record-wide result

**Object and scope.** `c02_age_ka_bp_1_see_reference_s` versus `c03_age_ka_bp_2_see_reference_s` across all 621 rows.

**Relationship and boundary.** The two coordinates are identical to within 0.01 ka for every row below 0.55 m and diverge only in the top 0.03–0.55 m.

**Core quantitative result.**
- 52 of 621 rows (8.4%) differ by more than 0.005 ka; all lie at depths 0.03–0.55 m (8.3% of the 6.27 m sampled span).
- Maximum absolute difference **1.63 ka** (at 0.31 m); signed differences run from −1.63 ka to +0.78 ka.
- Beyond 0.55 m the two coordinates are bit-identical.
- All 52 divergent rows carry δ¹⁸O, so they are not excluded by the proxy's missingness.

**Supporting analysis and result evidence.**
- `output/analysis/01_data_profile.py` → `output/results/01_data_profile.json` (`dual_age_coordinate_divergence`)
- `output/analysis/03_age_coordinate_leverage.py` → `output/results/03_age_coordinate_leverage.json`
- `output/analysis/09_confirmations.py` → `output/results/09_confirmations.json` (`a_period_reconciliation`)

**Robustness.**
1. *Both coordinates fitted separately.* Best period 108.8 ka on each; R² 0.427 vs 0.432; amplitude 0.386 vs 0.387 ‰; phase −2.585 vs −2.592 rad. No record-wide quantity moves.
2. *Divergent zone removed entirely.* Restricting to rows deeper than 0.55 m (541 rows) gives 104.5 ka with R² 0.500 — a shift of 4.3 ka, still well inside the 33.2 ka resolution limit.
3. *Sedimentation-rate field.* `np.gradient(depth, age)` is identical for the two coordinates on all interior rows (maximum absolute difference 0.0 below 0.56 m); the only nonzero difference is 4.6×10⁻⁴ at the single transition row.
4. *Adversarial perturbation.* Additive smooth perturbations of the divergent zone's ages with amplitude up to 12 ka — 7× the observed 1.63 ka divergence — move the record-wide best period by only 0.7 ka (median over 40 random draws; `output/analysis/verify_C4.py` and my own replication agree). The verifier's threshold analysis puts the required amplitude at **11 ka** for a deterministic Rayleigh-unit period shift and 50 ka for a random-shift median: a perturbation 7–30× larger than the actual divergence is needed before the record-wide period moves by the resolution limit.
5. *Independent verifier.* `output/analysis/verify_C4.py` re-derived the differing-row set independently, reproduced every descriptive fact (52 rows, 0.03–0.55 m, 1.63 ka), and quantified the shipped-pair difference in resolution units: the period shift between the two coordinates is **0.030 ka = 0.0009 Rayleigh units**; R² shifts by 0.0046, amplitude by 0.0046 relative, phase by 0.0018°. An adversarial synthetic coordinate built by translating `age_ka_bp_2` onto `age_ka_bp_1`'s span shifts the period by 1.82 ka = 0.055 Rayleigh units (`output/results/verify_C4.json`).

**Counterevidence, limitation, and alternatives.**
- The top 8.3% of the core is where the record's youngest, most rapidly deposited sediment sits, and it is also where the amplitude is largest. A coordinate error there is therefore not inconsequential *locally*, even though it is inconsequential *globally*.
- The conclusion is a statement about **record-wide** quantities. The two coordinates are not interchangeable for any question about the interval 2–33 ka: a 1.63 ka divergence is one third of the 0.46 ka median step many times over and exceeds the local cycle's resolution there.
- The 104.5 ka value from the reduced record is a different dataset (a truncated span, T = 300 ka, hence a coarser resolution limit of P²/T = 39.5 ka), not evidence that the true period is 104.5 ka.

**Falsification.** If a task-relevant quantity is defined over the top 0.55 m only — for example a Holocene-to-LGM transition age — the two coordinates would give different answers. Computing such a quantity on both coordinates and finding it stable would falsify the boundary as stated; finding it to differ would extend the divergence into a materially important result.

---

## Finding 4 — The ~241 ka accumulation-rate maximum is a real feature of the supplied depth-age model, but it is not evidence for a sedimentation event

**Object and scope.** The sedimentation rate `LSR = d(depth)/d(age)` over the record, and specifically the extreme episode near 241 ka.

**Relationship and boundary.** Sedimentation rate is highly variable (median 0.0220 m/ka, max/min ratio 12.7) but it is **not periodic**, and its one extreme episode is produced by an unusually fine age increment rather than by unusually rapid deposition. The observation is robust; the depositional interpretation is not supported.

**Core quantitative result.**
- The 99th percentile of LSR is 0.0433 m/ka. Six consecutive rows (row indices 390–395; 4.22–4.27 m; 240.87–241.88 ka) sit at or above it, forming one contiguous episode. The peak value is 0.0500 m/ka, 2.27× the record median.
- **The cause is the age grid, not the depth grid.** Across the episode the depth increment is the standard 0.01 m for every interval, while the age increment is 0.20–0.21 ka against a record median of 0.46 ka. LSR is therefore elevated because a thin slice of core is assigned an unusually short time span.
- The episode is not accompanied by a δ¹⁸O anomaly. Its mean δ¹⁸O is 2.94‰ against a record median of 3.37‰, but a block-permutation test that preserves serial correlation gives p = 0.22 for the episode mean and p = 0.15 for the local contrast — not distinguishable from noise (`output/results/verify_C3.json`).
- Removing the episode does not change the record-wide result: period 108.68 ka vs 108.90 ka, a shift of 0.22 ka.

**Supporting analysis and result evidence.**
- `output/analysis/05_lsr_excursion.py` → `output/results/05_lsr_excursion.json`
- `output/analysis/09_confirmations.py` → `output/results/09_confirmations.json` (`d_accumulation_episode`)
- `output/analysis/verify_C3.py` → `output/results/verify_C3.json` (differentiator-, grid- and recombination-null attacks)
- `output/figures/fig1_record_and_ages.png` (lower panel), `output/figures/fig4_lsr_and_change_point.png`

**Robustness.**
1. *Sedimentation-rate periodicity is a null — but the null is rho-sensitive.* The maximum Lomb-Scargle power of detrended LSR is 0.180 against an AR(1) null mean of 0.291 and 95th percentile of 0.451, empirical **p = 0.90** (`output/results/07_coupling_scale.json`). An independent review reproduced 0.180006 exactly and located the global maximum at ~40 ka, not near 100 ka, with the 90–110 ka band maximum still lower (`output/results/verify_C2.json`). However, that same review swept the assumed autocorrelation and showed the verdict turns on it: assuming rho = 0.96 (LSR's own lag-1 autocorrelation, the correct matched value) gives p = 0.89; rho = 0.90 gives p = 0.28; rho ≤ 0.75 gives p ≤ 0.01. **The correct choice is LSR's own rho (0.9585 on the row grid, 0.976 on a uniform grid), which is an order of magnitude higher than d18O's 0.788** — using the proxy's autocorrelation for the rate series would have produced a spurious "significant accumulation cycle". The matched-rho null is the one reported, and it is a null. The 0.180 global maximum also is not in the orbital band: inside 90–110 ka the LSR power is only 0.054, so the record shows no accumulation cycle at the proxy's own dominant period even before the null is applied.
2. *Differentiator sensitivity, resolved.* Five derivative estimators were compared. Central differences by hand reproduce `np.gradient` exactly (1 episode, 7 rows above p99, peak at row 413). A 3-point Savitzky-Golay derivative breaks it into 5 episodes and a 5-point SG derivative into 3, both moving the maximum to row 63. **This apparent fragility is itself an artefact, and the diagnosis is decisive:** the SG estimator assumes a uniform sampling grid, but this core has five enlarged depth steps (rows 45, 64, 313, 371, 412 — the only steps exceeding 0.01 m). Recomputing every estimator on an **interpolated uniform depth grid** puts the maximum slope at 241.26 ka for *all four* estimators tested (central, SG-3, SG-5, SG-9), at percentile rank 99.86. The SG failure is an unmodelled irregular grid, not evidence against the episode (`output/results/verify_C3.json`, `attackB_differentiators.uniform_grid`).
3. *Sampling-grid attack.* Only row 412 of the five enlarged-step rows lies inside the episode, so an enlarged depth step does **not** create it. Slope estimated over windows of 1, 2, 4 and 8 ka centred on 241 ka is rank 99.86, 99.93, 99.86 and 98.92 respectively; the episode remains the record's steepest span.
4. *Recombination null.* Shuffling depth increments over permuted age increments yields 7 rows above p99 spread across 7 separate episodes where the expectation is 6.2. An isolated extreme LSR value is therefore expected roughly once per record by chance — this bounds the strength of the claim but does not relocate it, since the observed episode is contiguous and 7 rows long.

**Counterevidence, limitation, and alternatives.**
- **The correct reading is a re-attribution, not a discovery of a sedimentation pulse.** The observation (an extreme accumulation-rate value at ~241 ka) is real and reproducible under every grid-aware estimator. The mechanism normally implied by such a value — a pulse of rapid deposition — is **not** supported, because the depth increments across the episode are the standard 0.01 m while the age increments are compressed. Whether the age model is *wrong* there is not decidable from this file alone: the record supplies ages, not the radiocarbon or tuning constraints behind them.
- Under 10 ka block-averaged LSR, 2 bins exceed the 95th percentile, so the episode is not the standout under that coarser definition.
- Three of the row-level estimators place the global maximum at 240.87–241.07 ka, but SG-3 and SG-5 place it at row 63 (42.59 ka) even on the uniform grid check applied to their own row-level output. The uniform-grid test supersedes this, but it is recorded that the estimators do not agree in their raw form.
- The episode lies where both age coordinates agree exactly (below 0.55 m), so it is not a Finding 3 issue.

**Falsification.** Obtain the original age-model tie points. If the 0.20 ka increments at 240.87–241.88 ka are honoured by real dated horizons, the episode is a genuine accumulation event and this finding is wrong. If they are interpolation artefacts between sparse tie points, the re-attribution is confirmed. A second falsifier: find an interval elsewhere in the core with equally fine age increments but median-range LSR — that would show fine age increments alone do not produce such episodes, and would point back toward a real depositional origin.

---

## Finding 5 — The proxy-variability/accumulation coupling is an arithmetic artefact, but accumulation rate and proxy *level* are genuinely coupled at block scale

**Object and scope.** Relationships between sedimentation rate and the δ¹⁸O series, at per-row and at inter-block scale.

**Relationship, direction, boundary.** Two superficially similar correlations have opposite status. `LSR` vs `|dδ¹⁸O/dage|` is **denominator-dominated and does not survive serial-correlation-aware testing**. `LSR` vs δ¹⁸O **level** is a **supported negative** relationship at block scale.

**Core quantitative result.**

| Relationship | Pearson r | n | Verdict |
|---|---|---|---|
| LSR vs `\|dδ¹⁸O/dage\|` (per row) | +0.475 | 593 | **mostly artefact** — 70% reproduced by the shared denominator alone |
| numerator-level: `\|Δdepth\|` vs `\|Δδ¹⁸O\|` | +0.106 | 593 | weak, p = 0.0095 raw — but **null** at block scale |
| LSR vs δ¹⁸O level (per row) | −0.407 | 593 | survives null, p = 0.0005 |
| LSR vs δ¹⁸O level (10 ka blocks) | −0.508 | 35 blocks | supported, block-permutation p = 0.0014 |
| LSR vs δ¹⁸O level (20 ka blocks) | −0.594 | 18 blocks | supported, block-permutation p = 0.010 |
| LSR vs δ¹⁸O level (40 ka blocks) | −0.798 | 9 blocks | supported, block-permutation p = 0.022 |
| LSR vs δ¹⁸O **variability** (block sd) | +0.174 | 18 blocks | **null**, p = 0.49 |

The mechanism: `np.gradient(depth, age)` and `np.gradient(d18o, age)` both divide a between-row difference by the *same* age increment. This was quantified directly (`output/results/11_shared_denominator_adjudication.json`) by pairing the **real age grid** with depth and |δ¹⁸O| increments resampled from their own marginals, so the denominator structure is untouched while no real pairing survives. That null gives r = 0.330 ± 0.036 — **70% of the observed 0.475 from arithmetic alone**. The position-wise mean `|grad|` of pure-noise surrogates, which depends only on the age grid, correlates with LSR at r = +0.972.

**A weak residual survives the artefact, and it is an unresolved quantity.** The numerator-level correlation — which shares no denominator at all — is r = +0.106 (raw p = 0.0095). But once the analysis unit is respected it disappears: across contiguous 10, 20 and 40 ka blocks the numerator-level correlation is +0.084, +0.160 and +0.324 with block-permutation p = 0.62, 0.53 and 0.38 respectively. **Zero of three block widths reach significance.** The defensible statement is that the per-row coupling is denominator-dominated, with a raw numerator-level residual that does not survive serial-correlation-aware testing.

**Supporting analysis and result evidence.**
- `output/analysis/07_coupling_scale.py` → `output/results/07_coupling_scale.json`, `output/results/07_block_means.csv`
- `output/analysis/09_confirmations.py` → `output/results/09_confirmations.json` (`b_inter_block_coupling`)
- `output/figures/fig4_lsr_and_change_point.png` (panel 1)

**Robustness.**
1. *Permutation null for the artefact.* 2000 age-order permutations of δ¹⁸O give a null r(LSR, |grad|) of +0.470 ± 0.037 versus an observed +0.475 (n = 593; the n = 547 variant that differentiates the full 621-row series before subsetting gives a null of +0.468 against an observed +0.468). Either way the observed value sits on the null mean — two-sided p = 0.75 (n = 593) or 0.50 (n = 547). Pure noise reproduces the entire correlation.
2. *Partial correlation is not the right test, and this is reported as a failed test.* Controlling linearly for the age increment gives a partial r of +0.391, i.e. it does **not** collapse. This is because the artefact is proportional rather than additive: the shared scale factor sits inside a product, so linear conditioning cannot remove it. The task-specified diagnostic fails; the permutation null and amplitude correction are the valid tests.
3. *Confound test for the surviving coupling.* Both series carry the ~109 ka cycle, which could manufacture an inter-block correlation. Removing the trend plus the 108.8 ka fundamental and second harmonic from **both** series leaves the coupling intact at all three block widths (−0.549, −0.376, −0.518 at 10/20/40 ka). The relationship is not merely the shared cycle.
4. *Coordinate and zone robustness.* Repeating with `age_ka_bp_2` gives −0.447 / −0.597 / −0.807; excluding the divergent top-of-core zone (0.03–0.55 m) gives −0.428 / −0.356 / −0.654. Sign and magnitude are stable.
5. *Leave-one-block-out.* At 40 ka the correlation ranges from −0.861 to −0.731 across the nine leave-one-out refits — sign-stable, not driven by one interval (`09_confirmations.json`).
6. *Amplitude is not coupled.* Block-level δ¹⁸O standard deviation against LSR gives r = +0.174 (p = 0.49), and the partial correlation controlling for block mean level is −0.047. This is a clean **null** and it rules out the most natural explanation of Finding 2's amplitude variation.

**Counterevidence, limitation, and alternatives.**
- **This relationship was genuinely contested during the run and the disagreement is recorded rather than hidden.** Two independent reviews reached opposite conclusions because they permuted different objects. Permuting the δ¹⁸O *values* (which also inflates the marginal spread of the resulting increment by ~40%) gives a null of 0.470–0.472 and declares the relationship fully explained (p ≈ 0.47–0.78). Permuting the *increments* — preserving the numerator's own marginal distribution — gives a null of 0.355 and p = 0.00025. `output/analysis/11_shared_denominator_adjudication.py` settles this by constructing the arithmetic-only null **directly from resampled numerators**, which yields 0.330 and is the most conservative defensible bound. The value-permutation null overstates how much the denominator explains; the increment-based null is the faithful one.
- **The most important caveat is that the reported effect size is construction-dependent**: "the coupling is entirely arithmetic" and "70% of it is arithmetic" are both defensible readings of the same data depending on the null. The report states the conservative reading (70%) and gives the raw numerator-level residual (+0.106) alongside its block-scale null result.
- An extreme-case check confirms the arithmetic is sufficient in principle: with a heavy-tailed independent numerator and the denominator held fixed, the null correlation exceeds the observed value. However, that construction does not preserve the numerator's marginal spread, so it bounds the mechanism rather than the hypothesis.
- The per-row LSR–level correlation (r = −0.407) is *not* trustworthy on its own, because rows are not independent. The block-scale results carry the claim; the per-row number is quoted only to show the direction is consistent.
- Block correlations across 9–35 blocks have wide intervals, and Pearson and Spearman diverge at 40 ka (−0.798 vs −0.283), signalling non-linearity or leverage by one or two blocks. The Spearman value at 40 ka is not significant (p = 0.46). The 10 ka and 20 ka block widths, where Pearson and Spearman agree closely, are the strongest evidence.
- **No causal direction is established.** A negative association between accumulation rate and δ¹⁸O level is equally consistent with (i) faster accumulation preserving a different signal, (ii) the age model absorbing a real climate signal (the circularity direction), or (iii) both series responding to a third factor. This record cannot separate them, and note that the same circularity that creates the artefact in rows 1–2 of the table could also contribute to the surviving block-scale relationship.
- The residual covariance is small: after dividing out the grid-only amplitude channel, the correlation falls from 0.468 to 0.131, i.e. ~7.8% of the variance on the n = 547 basis. Two denominators appear in this finding for a legitimate reason and both are stated: differentiating the full 621-row series and then subsetting to finite differences gives n = 547 and r = 0.468, whereas differentiating the 593-row complete-case series across the bridged gaps gives n = 593 and r = 0.475. The gap-bridging changes the median age increment from 0.450 to 0.460 ka and so shifts the correlation slightly. No conclusion depends on which is used.

**Falsification.** Measure an independent accumulation-rate proxy — excess ²³⁰Th flux, or a constant-initial-activity model — and test whether it correlates negatively with δ¹⁸O level at block scale. If an independently measured accumulation rate does **not** reproduce the negative relationship, the coupling is an age-model artefact of the same family as the one documented in the first two rows of the table. For the artefact claim itself: the falsifier is a null construction that preserves both the denominator and the numerator's marginal increment distribution while destroying only the pairing **and** yields a null near 0.475; the increment-preserving null gives 0.355 and the direct arithmetic null gives 0.330, so that falsifier was not found. For the residual: extending the record to ≥6 cycles would test whether the block-scale numerator residual (+0.084 to +0.324, never significant) becomes resolvable.

---

## 2. Additional tested relationships that did not become registered findings

These are reported so that failed and null tests are not silently discarded (contract §Required research actions 6).

**The δ¹⁸O mean-level "change point" at ~336.6 ka does not survive adversarial testing.**
A single mean-level split reduces the two-segment sum of squares from 92.94 to 83.19 (a 10.5% reduction) with a mean shift of +0.50‰, and it is preferred by BIC when the sample size is taken as the raw 593 rows (ΔBIC = −59.4). It does not survive once serial correlation is respected:

- Four independently derived effective sample sizes give n_eff of 28.0 (Sokal integrated autocorrelation), 2.98 (AR(1) MLE), 38.8 (40-row block bootstrap) and 76.5 (10-row block). Across that span BIC prefers the split at some values and not others; the break-even is n_eff ≈ 33.0, which lies inside the estimated range. The verdict is therefore **indeterminate on BIC alone** (`output/results/verify_C5.json`).
- Against an AR(1) surrogate null the observed 10.5% variance reduction is unremarkable: the null mean is 41.4% and the null 95th percentile is 72.1%, empirical **p = 0.99**. Against phase-randomised surrogates p = 0.49, and against a moving-block bootstrap on the smooth residual p = 0.51–0.55.
- **The split is not edge-stable.** Dropping the last 5% of rows moves the best split from 336.9 ka to 74.0 ka — a 263 ka shift. Requiring 60 or 80 rows per segment moves it to 74.0 ka as well. The "change point" is the last 44 rows, a high-δ¹⁸O tail in a record whose final age steps are 1.82 and 2.53 ka (4–6× the median step), and it dissolves when that tail is trimmed.
- Under a common model comparison the periodic model is overwhelmingly preferred: BIC with n_eff = 91.7 gives segmented-mean 4.59, linear trend 10.22, no structure 5.72, and the 2-harmonic sinusoid **−22.82**. Contiguous 6-block cross-validation agrees: the sinusoid is the only model with positive mean out-of-sample R² (+0.020) versus −0.44 (segmented mean), −0.65 (linear trend) and −0.45 (no structure). The optimum flips to 336.6 ka under *every* choice only because that split maximises the mean difference; it does not survive resampling. (`output/results/09_confirmations.json`; `output/results/06_transition_structure.json`.)

**A second, weaker transition candidate exists and is also not registered.** A three-segment model adds a split near 309 ka, but its middle segment is only 24 ka long (well inside the 33.2 ka resolution limit) and R² rises only to 0.201 against 0.427 for the periodic model. Reported in `output/results/06_transition_structure.json`; not promoted to a finding.

**Other nulls.**
- Sedimentation rate is not periodic: max Lomb-Scargle power 0.180 vs a matched-rho AR(1) null 95th percentile of 0.451, p = 0.90 (rho-sensitivity documented under Finding 4).
- LSR against *signed* dδ¹⁸O/dage: r = +0.067, p = 0.017 — a weak relationship whose sign is not stable enough to report.
- δ¹⁸O block variability against LSR: r = +0.174, p = 0.49; partial given block mean level, −0.047.

---

## 3. Limitations of the whole analysis

1. **One core, one proxy.** Every result here describes a single continuous sediment record and one δ¹⁸O series. `DATA_GUIDE.md` states plainly that one continuous record cannot establish a global Earth-system causal mechanism. No finding claims one.
2. **Serial correlation governs everything.** With 3.27 cycles in 356 ka and a 33.2 ka resolution limit, the record supports statements about *one* long oscillation and about the mean level, and supports very little about sub-resolution spectral structure or about any transition sharper than the resolution limit.
3. **The age coordinates are model output, not measurement.** Findings 4 and 5 both show results whose interpretation depends on whether one trusts the depth-age model in specific intervals. The file does not carry the tie points, so that trust cannot be audited from within the workspace.
4. **33.2 ka resolution limit.** Stated once here and applied consistently: the neighbouring peaks at 97.3 and 119.4 ka are inside one Rayleigh lobe of the 108.8 ka peak and are not resolvable alternatives.
5. **Provenance.** No external source was used, so no citation log is required. Per the time boundary, no literature search was performed and no external data was downloaded, substituted, or merged.

---

## 4. Reproduction

Full instructions, exact commands, environment versions, the column mapping, the seeds, the key computational choices, and **the three defects found and corrected during the run** are in `output/reproduce.md`. Analysis source is in `output/analysis/`, machine-readable results in `output/results/`, figures in `output/figures/`.

Quick start:

```bash
python3 output/analysis/01_data_profile.py
python3 output/analysis/02_spectral_phase.py
python3 output/analysis/03_age_coordinate_leverage.py
python3 output/analysis/04_depth_domain_spectrum.py
python3 output/analysis/05_lsr_excursion.py
python3 output/analysis/06_transition_structure.py
python3 output/analysis/07_coupling_scale.py
python3 output/analysis/08_smoothing_sensitivity.py
python3 output/analysis/09_confirmations.py
python3 output/analysis/10_figures.py
```

## 5. Finding-to-evidence index

| Finding | Analysis source | Result evidence |
|---|---|---|
| 1 | `output/analysis/02_spectral_phase.py`, `04_depth_domain_spectrum.py`, `09_confirmations.py` | `output/results/02_spectral_phase.json`, `04_depth_domain_spectrum.json`, `09_confirmations.json`, `02_periodogram.csv`, `04_spectra.csv`; `output/figures/fig2_spectra_and_null.png` |
| 2 | `output/analysis/02_spectral_phase.py` | `output/results/02_spectral_phase.json`, `02_sliding_window.csv`, `02_block_holdout.csv`; `output/figures/fig3_phase_and_holdout.png` |
| 3 | `output/analysis/01_data_profile.py`, `03_age_coordinate_leverage.py`, `09_confirmations.py` | `output/results/01_data_profile.json`, `03_age_coordinate_leverage.json`, `09_confirmations.json` |
| 4 | `output/analysis/05_lsr_excursion.py`, `09_confirmations.py` | `output/results/05_lsr_excursion.json`, `09_confirmations.json`; `output/figures/fig1_record_and_ages.png`, `fig4_lsr_and_change_point.png` |
| 5 | `output/analysis/07_coupling_scale.py`, `09_confirmations.py`, `11_shared_denominator_adjudication.py` | `output/results/07_coupling_scale.json`, `07_block_means.csv`, `09_confirmations.json`, `11_shared_denominator_adjudication.json`; `output/figures/fig4_lsr_and_change_point.png` |

Independent adversarial verification artifacts (each written by a separate agent instructed to refute rather than confirm): `output/analysis/verify_C1.py` → `output/results/verify_C1.json`; `verify_C3.py` → `verify_C3.json`; `verify_C4.py`; `verify_C5.py` → `verify_C5.json`.
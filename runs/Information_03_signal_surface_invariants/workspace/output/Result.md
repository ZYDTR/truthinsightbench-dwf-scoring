# Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces

**Task:** `Information_03_signal_surface_invariants`  ·  **T0:** 2022-01-01  ·  **Report language:** English

This is the authoritative report. It registers five findings for scoring. All numeric evidence comes
from the frozen task tables under `data/` and from code that was executed in this workspace; every
number is traceable to a result file listed in the Finding-to-evidence index at the end.

---

## 1. Data, analysis unit and scope

Three frozen tables, all complete factorials with no missing cells, no imputation and no curator-imposed
row deletion (`output/results/t01_inventory.json`):

| File | Rows | Design | Role |
|---|---|---|---|
| `data/replicated_response_panel.tsv` | 8232 | 7 `input_axis_1_index` × 7 `input_axis_2_index` × 8 `signal_bin_index` × 7 `output_channel_index` × 3 `replicate_id` | primary replicated surface |
| `data/independent_response_panel.tsv` | 4116 | 7 × 7 × 7 `signal_bin_index` × 4 `output_channel_index` × 3 `replicate_id` | independent panel |
| `data/independent_fit_error.tsv` | 21 | 7 `input_axis_index` × 3 `replicate_id` | supplied independent fit-error matrix |

SHA-256 of each file matches `data_manifest.json` exactly, and every analysis script re-verifies those
hashes at load time (`output/analysis/common.py::load_panels`; hashes echoed in every
`output/logs/step*.log`). Values are strictly positive (replicated range 2447.83 – 29 337 190.41;
independent range 4741.76 – 9 785 051.56), so `log(response_value)` is well defined; logarithms are
natural throughout and the response is analysed on the log scale unless stated otherwise.

**Analysis unit.** One complete input-grid condition *with* its replicate structure — i.e. a cell
`(input_axis_1_index, input_axis_2_index, signal_bin_index, output_channel_index)` together with its three
replicates. Adjacent grid cells and repeated measurements are not independent systems, and no result below
treats them as such. Replicate means are intermediate quantities; replicate dispersion is analysed as its
own signal (Findings 4 and 5). Every holdout, bootstrap and permutation resamples **complete** rows,
columns, channels or grid regions, never random cells.

**Y is defined on a magnitude scale, not a clean scale-free one**, and this governs how every result must
be read. `response_value` spans four orders of magnitude; the per-channel dynamic range on the raw scale
goes from 1× (channel 0) to 3642× (channel 6) but only 9.6× on the log scale. This is why Findings 1 and 3
report scale-explicit statements rather than scale-free invariants (`output/results/t05_...json`,
`t20_robustness.json`).

**Boundary with external information.** No external source was consulted or used. T0 = 2022-01-01 was
therefore not a binding constraint, and no external dataset was downloaded, substituted or merged: the
workspace `data/` directory is the complete evidentiary basis. No literature citation is claimed.

---

## Finding 1 — The replicated surface is a single latent gain axis: one coordinate organises the entire (a1, a2) × channel grid

**Object and scope.** The replicate-averaged log-response surface of `replicated_response_panel.tsv`,
arranged as 49 complete `(input_axis_1_index, input_axis_2_index)` grid conditions × 7
`output_channel_index` columns (and, for the confirmation variant, as 392 `(a1, a2, signal_bin)` rows × 7
channels).

**Relationship and direction.** After centring each channel, the 49×7 matrix is close to rank 1: a single
latent coordinate `g(a1, a2)` carries the grid variation, and each output channel expresses it with its own
scalar gain. The leading singular value is 13.559 against 2.604 for the second.

**Core quantitative result.** Rank-1 R² = **0.9550** (rank-2 = 0.9902) on the 49×7 surface; rank-1
R² = **0.9556** on the 392×7 surface. This equals 95.5% of the *within-channel* variance, since channel
means were removed before the decomposition. The channel loadings
`[0.009, 0.005, −0.016, −0.081, −0.269, −0.599, −0.750]` are monotone in channel index and correlate
|r| = 0.9978 with the channels' grid standard deviations `[0.1259, 0.1731, 0.2058, 0.2492, 0.5467, 1.1639,
1.4579]`. The gain of channel 6 is 11.6× that of channel 0.

**Supporting evidence.** `output/analysis/step1_surface_geometry.py` → `output/results/t03_rank1_gain_axis.json`,
`t04_channel_sensitivity.json`, `t04_channel_sensitivity.csv`;
independently recomputed from scratch in `output/analysis/step6_verification.py` →
`output/results/t21_verification.json` section `A_rank1_r2` (both a spectral and an explicit reconstruction
formula, agreeing to 1e-9);
figure `output/results/figures/fig2_rank1_spectrum.png`.

**Robustness.** (i) *Null simulation:* 500 synthetic surfaces that preserve each channel's mean and grid
standard deviation but destroy the cross-channel correspondence give rank-1 R² = 0.559 ± 0.020 (range
0.541–0.712), with 0/500 at or above the observed value — empirical p = 0.002. The structure is genuine
shared gain, not an artefact of unequal channel magnitudes (`t20_robustness.json`). (ii) *Complete-factor
holdouts* (`t09_holdout_validity.json`): holding out each full `a1` row and refitting the rank-1 basis on
the remaining six gives R² 0.795 mean / 0.403 worst against the honest per-channel-mean baseline, rising to
0.899 / 0.775 at rank 2; holding out each full `a2` column gives 0.964 mean / 0.938 worst; holding out whole
grid regions (3×3 corners, middle strips) gives 0.726–0.972 at rank 1 and 0.878–0.989 at rank 2.
(iii) *Complete-unit leave-out:* removing any one replicate or any one input level moves the rank-1 R² only
within 0.945–0.964 (`t20_robustness.json`). (iv) *Outlier treatments:* dropping the top-1%-CV cells,
winsorising at the 1st/99th percentile within channel, or dropping each channel's global min and max all
leave rank-1 R² within 0.9541–0.9560. (v) *Bootstrap* over complete grid conditions: rank-1 R² = 0.9556,
95% CI [0.9387, 0.9682].

**Counterevidence, limitation and alternative explanations.**
(a) The alignment of channel magnitudes (the near-constant, slightly decreasing channel *offsets*,
variance-decomposition channel main effect 58.92%) makes a shared additive input term a *tenable*
alternative to independent per-channel scaling responses — the two models make identical predictions for
the three input axes once a channel offset is fitted, and the data cannot separate them. The saving
evidence is the amplitude spread (11.6× in grid standard deviation), which an offset-only model cannot
produce, and the per-channel slope ordering, not merely the offsets.
(b) The latent basis transfers weakly to held-out *low-gain* channels: leave-one-channel-out R² is 0.019,
0.003, 0.023 on channels 0–2 but 0.895, 0.982, 0.962 on channels 4–6, because the latent coordinate is
dominated by the high-gain channels (`t20_robustness.json`). The rank-1 description is therefore an
amplitude statement about the high-gain channels rather than a basis that predicts the quiet channels.
(c) Rank 2 reaches 0.9902, so a genuinely two-dimensional structure remains beside the leading axis.

**Falsification test.** The claim would be refuted by a replicated panel in which the per-channel grid
surfaces are not scalar multiples of one another — specifically by an experiment in which the channel with
the smallest grid standard deviation shows a *reliably different* grid pattern from the channel with the
largest. Concretely: fit the rank-1 basis on channels 4–6, predict channel 0, and repeat at higher replicate
counts so that channel 0's residual is measured above its own noise floor. This is an executable protocol
on the same measurement scale.

---

## Finding 2 — The single-gain-axis structure is a magnitude-scale property, not a scale-free invariant

**Object and scope.** The same 49×7 surface as Finding 1, re-expressed under seven defensible
normalisations: raw linear, natural log, log10, square root, pooled rank, per-channel z-score, per-channel
min-max, and within-channel rank.

**Relationship and direction.** The rank-1 structure is preserved by magnitude-preserving transforms and
destroyed by per-channel calibration. Rank-1 R² is 0.9987 (linear), 0.9938 (square root), 0.9565 (log10),
0.9550 (log) — but only 0.5688 (pooled rank), 0.5681 (per-channel z-score), 0.5588 (per-channel min-max),
0.5841 (within-channel rank). The per-channel z-scored value of 0.5681 sits inside the null distribution of
Finding 1 (0.559 ± 0.020): after per-channel calibration the single-gain-axis signal is **indistinguishable
from noise**.

**Core quantitative result.** The destruction is explained by the normalisation itself: standardising each
channel removes the per-channel amplitude, which is exactly what the latent axis encodes. Loading agreement
with the log solution falls from 1.000 (log) to 0.272 (per-channel z-score) and 0.798 (linear). The
amplitude being removed is large and is itself an intrinsic property of the construct: the per-channel
dynamic range of the raw surface runs from 1.0× (channel 0) to 3642.3× (channel 6) on the linear scale, but
only 0.562 to 5.393 log units on the log scale.

**Supporting evidence.** `output/analysis/step5_robustness.py` → `output/results/t20_robustness.json`
(see `normalisation_battery`); earlier variants in `output/results/t05_normalisation_perturbation.json`;
recomputed in `output/results/t21_verification.json` section `A_rank1_r2.variants_not_the_claim`;
figure `output/results/figures/fig2_rank1_spectrum.png` (right panel).

**Robustness.** The ordering is stable across all seven transforms, and the boundary is sharp rather than
gradual: every transform that preserves the per-channel magnitude keeps rank-1 R² ≥ 0.93, and every
transform that rescales each channel to unit spread drops it to ≈0.56–0.58. This is confirmed by an
independent recomputation (`t21_verification.json`, no shared code with `common.py`).

**Counterevidence, limitation and alternative explanations.** This is a *boundary* result and it constrains
Finding 1: a reader who prefers a scale-free notion of structure should treat the single-gain-axis claim as
unsupported, because the shared-gain signal is carried by the amplitude differences that per-channel
calibration removes. Conversely, the linear-scale rank-1 R² of 0.9987 is *easier* to achieve than the log
one — raw-scale variance is dominated by the largest values — so neither extreme is the "correct"
normalisation, and both are reported. The normalisation choice must be declared in any downstream use.

**Falsification test.** Apply the analysis to an input-output surface whose channels have been
instrument-calibrated to a common reference level *before* measurement: the boundary predicts that the
rank-1 account should then be recoverable only after an explicit, documented re-introduction of the channel
gains, and that a per-channel-standardised analysis of the calibrated data will show no dominant axis. A
single-gain-axis structure surviving per-channel calibration would refute this finding.

---

## Finding 3 — On the shared input×signal×channel grid the two panels do not reproduce each other's response levels; the disagreement is not a domain artefact

**Object and scope.** The 1372 cells common to both panels: `input_axis_1_index` 0–6,
`input_axis_2_index` 0–6, `signal_bin_index` 0–6, `output_channel_index` 0–3, three replicates each. The
independent panel is a **strict sub-domain** of the replicated panel — it contains no cell that the
replicated panel lacks — so the domain mismatch cannot *create* this disagreement.

**Relationship and direction.** Null. Replicate-averaged log-responses on the common grid are essentially
uncorrelated between panels; no relabelling, reflection or axis swap recovers agreement.

**Core quantitative result.** Cell-level Pearson **r = +0.0168** (Spearman +0.0324, n = 1372 cells;
permutation p = 0.54; row-level bootstrap 95% CI [−0.026, +0.037]). The replicated model of §"structural
transfer" below adds nothing. The level offset is real and large: the independent panel sits 0.651 log
units (≈1.92×) above the replicated panel on the same cells, with far greater spread (log SD 1.3603 vs
0.3753 within the shared region).

**Supporting evidence.** `output/analysis/step3_independent_panel.py` →
`output/results/t11_direct_replication.json`, `t12_alignment_search.json`,
`t16_structural_transfer.json`; `output/analysis/step7_domain_boundary.py` →
`output/results/t22_domain_boundary.json`; recomputed from scratch in
`output/results/t21_verification.json` sections `C_cross_panel_correlation` and `R1_scale_artefact`;
figure `output/results/figures/fig4_cross_panel_noise.png` (left panel).

**Robustness.** (i) All 8 combinations of axis reflection and axis swap were tested: the strongest
absolute match is −0.158 (both axes reversed), still null in magnitude. (ii) All 24 permutations of the four
independent channels were tested: the best gives +0.071. (iii) Scale-variant refutation attempts
(`t21_verification.json::R1_scale_artefact`): per-channel centring +0.069 (p = 0.011), global
variance-standardisation +0.017, pooled ranks +0.032 — all immaterial except two within-channel variants
(within-channel z +0.101, within-channel ranks +0.115, both p = 0.0001), which are addressed under
counterevidence. (iv) A model fitted on the replicated panel (`channel` + `channel×a1` + `channel×a2` +
`channel×bin`, in-sample R² = 0.5751) scores **R² = −0.2768** when applied unchanged to the independent
panel; the best channel permutation gives −0.2436 and the axis-swapped form −0.2510, while the same model
family fitted on the independent panel reaches 0.7657 on its own data. The failure is a transfer failure,
not an under-fitting failure.

**Counterevidence, limitation and alternative explanations.** Two qualifications must be stated plainly.
First, a **weak but real** shared component exists at within-channel scale: within-channel z-scored
r = +0.101 and within-channel ranks r = +0.115 (both p = 0.0001). The near-zero headline applies to
panel-level cell agreement, not to within-channel shape agreement, and a reader looking for *any* shared
signal will find a small one. Second, the residual tracks the `(a1, a2)` marginals: collapsing both panels
over bin and channel raises the common-domain correlation to +0.2862, and removing each panel's own
(bin, channel) marginal and re-standardising gives −0.105 (p = 0.0002) — the shared component lives in the
input-axis marginals, not in the cell-level surface. Among alternative explanations, the two panels may
simply measure non-identical things under anonymous axis labels; the workspace provides no construct
identifiers that would settle this, and the T0 boundary forbids using external search to reconstruct them.
What the data *do* establish is that the disagreement is measurable entirely inside the shared region and
therefore cannot be attributed to the domain mismatch.

**Falsification test.** Re-acquire the two panels under a common reference: a single calibration condition
measured in both campaigns, or an explicit replicate-level audit that pairs a replicated-panel measurement
with an independent-panel measurement of the *same* physical condition. If the 1372 shared cells then agree
at anything like their within-panel replicate precision, this finding is refuted. If they remain
uncorrelated, the panels describe different response surfaces.

---

## Finding 4 — The two panels share a within-cell relative noise scale, and that sharing is multiplicative rather than additive

**Object and scope.** Within-cell replicate dispersion on the same 1372 shared cells: for every cell, the
standard deviation and coefficient of variation (CV) of `log(response_value)` across its three replicates,
computed separately in each panel and matched cell-by-cell. The signal is the *relative* dispersion,
independent of response level — explicitly disjoint from Finding 3's near-zero level correlation.

**Relationship and direction.** Positive and monotone: cells that are noisy in the replicated panel are
noisy in the independent panel. This is the one clean cross-panel invariant found in this task.

**Core quantitative result.** Cell-level CV Pearson **r = +0.3822** (n = 1372 cells; permutation p < 5e-5;
all three replicate-wise values agree at ≈0.38). Removing both panels' `(a1, a2)` marginal means leaves
**r = +0.1993** (p < 5e-5), so the sharing is not merely a marginal coincidence. The panel-level marginal
CV profiles correlate at +0.845 (`a1`) and +0.915 (`a2`). Cross-frequency noise ordering (mean CV over
channels 0–3): channels 1–2 are the noisiest in *both* panels (1.18/1.16 and 1.31/1.17), while channel 0 is
quiet in the replicated panel (0.72) yet loud in the independent panel (1.15).

**Supporting evidence.** `output/analysis/step3_independent_panel.py` →
`output/results/t14_noise_scale_sharing.json`; domain audit in
`output/results/t22_domain_boundary.json`; recomputed from scratch in
`output/results/t21_verification.json` sections `D_cv_sharing` and `R2_multiplicative_artefact`;
the supplied fit-error matrix with its explicit power caveat in `output/results/t15_fit_error.json`;
figure `output/results/figures/fig4_cross_panel_noise.png` (right panel).

**Robustness.** (i) *Multiplicative-vs-additive:* the sharing survives on the CV scale (+0.3822), on the
log-scale SD (+0.3692, p < 5e-5) and on log(CV) (+0.3164, p < 5e-5), but the **linear-scale** SD gives only
+0.1108 and turns **−0.0482 (p = 0.068)** after `(a1, a2)` means are removed (−0.0821, p = 0.008, after
channel means are also removed). The invariant is a *relative* (multiplicative) scale property; on an
absolute scale it does not survive. (ii) *Alternative conditioning:* removing `(a1, a2)` means keeps
+0.1993 (p < 5e-5); removing `(a1, a2, bin)` means keeps +0.1364. (iii) *Replicate-deviation control:* the
cross-panel correlation of raw replicate deviations is −0.0917 (p < 5e-5), i.e. shared structure is carried
by dispersion magnitude, not by a shared per-replicate random draw.

**Counterevidence, limitation and alternative explanations.** (i) The supplied `independent_fit_error.tsv`
does **not** validate against this signal at the required standard: its axis-level mean fit-error correlates
with replicated-panel CV by `a1` at only r = +0.209 (t = 0.48 vs critical 2.571 at df = 5, permutation
p = 0.66) and with independent-panel CV by `a1` at r = +0.374 (t = 0.90, p = 0.40). With n = 7 axis levels
these tests have very little power (permutation-p floor 2/5040; a t of 2.571 would be needed), so this is a
*null result on an underpowered matrix*, not evidence against the shared noise axis — and it is reported
rather than discarded. (ii) The per-channel noise *ordering* differs between panels (channel 0 is quiet in
one and loud in the other), so the shared quantity is a per-input-condition relative noise level, not an
identical channel ranking. (iii) A modelled-detection-capacity component accounts for about half of the
observed similarity, so the residual physical sharing is roughly the size of the conditioned value
(+0.1993).

**Falsification test.** The most direct test is per-condition: take a subsample of the 1372 cells with the
*lowest* replicated-panel CV and repeat both measurements with more replicates. If the independent panel's
dispersion in those cells converges toward the replicated panel's (as multiplicative noise sharing
predicts), the finding holds; if the two panels' dispersions remain unrelated on re-measurement, it is
refuted. The additive alternative predicts that raw-scale standard deviations, not CVs, should be the
shared quantity — and the data already contradict that (r = −0.048 after conditioning).

---

## Finding 5 — The signal-bin-0 boundary feature is not reproduced by the independent panel; the boundary holds inside the shared domain and reverses in sign

**Object and scope.** The lowest `signal_bin_index` level, present in both panels (bin 0 of 0–7 in the
replicated panel, bin 0 of 0–6 in the independent panel). The feature is evaluated per panel against the
panel's own trend fitted to the higher bins, and then re-evaluated on the shared bin range 0–6 so that the
comparison cannot be attributed to the replicated panel's extra bin 7.

**Relationship and direction.** The replicated panel's bin 0 sits *above* the smooth trend through bins 1–7
by +0.948 log units. The independent panel's bin 0 sits *below* its own trend through bins 1–6 by
−0.4539 log units. The two panels disagree in **sign**, and this reversal persists inside the shared
domain — the replicated excess on bins 0–6 only is still **+0.2800** while the independent excess is
**−0.4539**.

**Core quantitative result.** Replicated panel: observed bin-0 mean log-response 11.1171 versus a quadratic
trend through bins 1–7 of 10.1689 → excess **+0.9482** (fold change ×2.58). The same bin 0 is also the
**global maximum of the bin marginals** (11.1171 vs 10.561 at bin 7). Independent panel: observed 8.9324
versus trend 9.3864 → excess **−0.4539** (×0.64). The excess is positive in **all 49** `(a1, a2)` grid
cells and in all 7 channels of the replicated panel.

**Supporting evidence.** `output/analysis/step1_surface_geometry.py` →
`output/results/t06_bin0_boundary.json`; `output/analysis/step3_independent_panel.py` →
`output/results/t19_bin0_notch_panels.json`; domain-matched comparison in
`output/results/t22_domain_boundary.json`; recomputed from scratch in
`output/results/t21_verification.json` sections `E_bin0_excess` and `R3_trend_model_artefact`;
figure `output/results/figures/fig3_bin0_notch.png`.

**Robustness.** (i) *Trend-model sensitivity* (11 variants): excess estimates span **+0.2287 to +1.1552**
(×1.26 to ×3.17) and are **never negative**; the direct mean of bins 1–2 gives +0.9390. The sign is robust
while the magnitude is model-dependent, so the report quotes the canonical quadratic value with its range.
(ii) *Leave-one-bin-out* trend fits give +0.8811 to +0.9876. (iii) *Per-cell*: the excess over a quadratic
trend is positive in all 49 cells (100% for the canonical trend, 47–49 of 49 across variants).
(iv) *Channels and bins:* the feature is present in both the magnitude-dominant channel 6 (global maximum of
the entire replicated panel, +2.200) and in the quiet channel 0 (+0.157), and is not concentrated in any
single replicate.

**Counterevidence, limitation and alternative explanations.** (i) The excess magnitude is trend-model
dependent (+0.23 to +1.16), so only its sign and its status as the bin-marginal maximum are robust; the
fold-change ×2.58 should not be quoted without the range. (ii) An earlier automated variant of the trend
check raised `TypeError: cannot unpack non-iterable NoneType object` at
`output/analysis/step6_verification.py:617`; the defect was repaired and the final verification run reports
`errors: []` with 0 discrepancies. The error is recorded here rather than hidden.
(iii) **Mechanism is unresolved and no mechanism is claimed.** The width of bin 0 (1 of 8 levels) is
consistent with a surface/edge artefact such as a boundary reflection, a region-of-interest edge effect, or
a curvature singularity near the lowest signal coordinate — but the workspace supplies no construct
identifiers, calibration record or acquisition metadata that could discriminate among these, and the T0
boundary forbids reconstructing them externally. An *apparent* geometric signature (the excess grows roughly
with `input_axis_2` index, from ×1.1 at `a2` = 0 to ×4.4 at `a2` = 6, peaking near the low-`a1`/high-`a2`
corner) is noted as a description of where the feature is concentrated, **not** as an identified mechanism.
(iv) The feature is defined by the panel's own trend, so it is a statement about that panel's bin profile
rather than about an absolute reference level.

**Falsification test.** Measure `signal_bin_index` 0 in the same campaign as bins 1–7 with additional
replicates and a nominal blank or zero-signal control: the finding predicts a positive, reproducible offset
at bin 0 in the replicated construct, and it is refuted if bin 0 falls on the extrapolated trend within
replicate precision. The cross-panel claim is falsified if the independent construct is shown to be a
different quantity measured on a different axis rather than a genuine re-measurement of the same boundary —
which would also resolve the sign reversal reported here.

---

## 2. Null and conflicting results retained

Reported because they materially bear on the findings above.

1. **`independent_fit_error.tsv` does not validate the noise-scale story.** Its axis-level correlations with
   both panels' CV profiles are non-significant (r = +0.209, p = 0.66 against the replicated panel;
   r = +0.374, p = 0.40 against the independent panel) at n = 7 axis levels. Power is the binding
   constraint, and the null is reported as such (Finding 4).
2. **The fit-error matrix's own structure.** Axis explains 59.8% of its variance and replicate 31.0%, with
   40.2% left after removing axis means — replicate 3 is systematically lower (mean 0.1194 vs 0.1352/0.1352),
   which is consistent with a fixed replicate-level offset rather than a purely random error term.
3. **The 3-way interaction in the fitted offset model.** A saturated model of the channel offsets (channel
   main + channel×a1 + channel×a2 + channel×bin) reaches in-sample R² = 0.5751; adding the channel×a1×a2
   interaction is significant (ΔR² = 0.0088, F = 2.54 on 36 and 180 df, p = 1.96e-5). Together with 3-way
   interactions carrying 6.00% of the total variance (main effects are only 70.67%), this is *why* the
   anchor-and-baseline families are not additive and why channel 6 is a poor anchor against the noisy low
   channels.
4. **No shared input-gain law on the overlap.** Within channels 0–3, replicated a1 slopes are
   [0.0162, 0.0311, 0.0563, 0.0946], strictly increasing and spanning a factor of 5.8; the independent
   panel's are [0.1865, 0.2026, 0.0169, 0.1457], non-monotone and up to 11.5× steeper at channel 0. The two
   panels do not share an input sensitivity law even where they overlap.
5. **The channel-regime boundary is not uniquely identified.** The best of all 63 two-block partitions of
   the 7 channels is the nominal 0–2 | 3–6 cut (gap 0.5644, permutation p = 0.0269), but the runner-up
   0–3 | 4–6 (gap 0.4849, p = 0.0558) is statistically indistinguishable (gap difference 0.0795 against a
   null SD of 0.1610), and only 2 of 63 partitions reach p < 0.05. A boundary in the channel dimension is
   plausible; its exact location is not pinned down, so no finding is registered on it.
6. **The channel amplification ratio is near-singular.** The headline chan6/chan0 slope ratio of 29.2 is
   correctly computed, but its bootstrap 95% CI is unusable ([−264, +313]; 7.5% of resamples give a
   non-positive channel-0 slope) because channel 0's slope (0.0166, bootstrap SD 0.0111) sits at the noise
   floor. The robust statements are the *ordering* (strictly increasing slopes in 99.1% of resamples,
   channel 6 maximal in 100%) and the peak/sub-grid ratio chan6/chan5 = 1.250 [1.156, 1.379]. This is why
   the ratio does not appear in any registered finding.

---

## 3. Reproduction

A companion reproduction document with environment, data hashes, exact commands and expected outputs is
provided at `output/reproduce.md`. In outline: every script under `output/analysis/` is standalone
(`python3 <script>.py` from `output/analysis/`), depends only on Python 3.11 with numpy, pandas and
matplotlib (no scipy), verifies the SHA-256 of all three input files at load, and writes its machine-readable
results to `output/results/` and a text log to `output/logs/`. Nothing was downloaded; no result was
hand-entered.

---

## 4. Finding-to-evidence index

| Finding | Analysis source | Result evidence | Figure |
|---|---|---|---|
| Finding 1 | `output/analysis/step1_surface_geometry.py`, `step2_holdout_validity.py`, `step5_robustness.py` | `output/results/t03_rank1_gain_axis.json`, `t04_channel_sensitivity.json`, `t09_holdout_validity.json`, `t20_robustness.json`, `t21_verification.json` (`A_rank1_r2`) | `fig2_rank1_spectrum.png`, `fig5_gain_axis_loadings.png` |
| Finding 2 | `output/analysis/step5_robustness.py`, `step6_verification.py` | `output/results/t20_robustness.json`, `t05_normalisation_perturbation.json`, `t21_verification.json` (`A_rank1_r2.variants_not_the_claim`) | `fig2_rank1_spectrum.png` |
| Finding 3 | `output/analysis/step3_independent_panel.py`, `step7_domain_boundary.py`, `step6_verification.py` | `output/results/t11_direct_replication.json`, `t12_alignment_search.json`, `t16_structural_transfer.json`, `t22_domain_boundary.json`, `t21_verification.json` (`C_cross_panel_correlation`, `R1_scale_artefact`) | `fig4_cross_panel_noise.png` |
| Finding 4 | `output/analysis/step3_independent_panel.py`, `step7_domain_boundary.py`, `step6_verification.py` | `output/results/t14_noise_scale_sharing.json`, `t15_fit_error.json`, `t22_domain_boundary.json`, `t21_verification.json` (`D_cv_sharing`, `R2_multiplicative_artefact`) | `fig4_cross_panel_noise.png` |
| Finding 5 | `output/analysis/step1_surface_geometry.py`, `step3_independent_panel.py`, `step6_verification.py` | `output/results/t06_bin0_boundary.json`, `t19_bin0_notch_panels.json`, `t22_domain_boundary.json`, `t21_verification.json` (`E_bin0_excess`, `R3_trend_model_artefact`) | `fig3_bin0_notch.png` |

**Full artifact list.** Analysis source: `common.py`, `step1_surface_geometry.py`,
`step2_holdout_validity.py`, `step3_independent_panel.py`, `step4_figures.py`, `step5_robustness.py`,
`step6_verification.py`, `step7_domain_boundary.py`. Result artifacts: `t01`–`t22` under
`output/results/` (JSON plus two CSVs) and five PNGs under `output/results/figures/`. Process logs:
`output/logs/step1.log`–`step7.log`.

**Independent verification.** `step6_verification.py` re-derives all six headline numbers from `data/`
using code that does not import `common.py` (rank-R² computed by both a spectral and an explicit
reconstruction formula; permutations 20 000 draws each). Final run: 6/6 MATCH at 5e-5 tolerance,
`n_discrepancies = 0`, `errors = []`. A separate consistency audit parsed all 22 result JSON files and ran
66 cross-file numeric checks with 0 real mismatches.
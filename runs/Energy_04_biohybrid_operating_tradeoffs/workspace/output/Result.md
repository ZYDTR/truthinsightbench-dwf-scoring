# Dose, Time, and Operating Tradeoffs in a Biohybrid Energy System

**Task ID:** `Energy_04_biohybrid_operating_tradeoffs`
**T0 (evidence boundary):** `2023-01-01`
**Authoritative report.** This file is the sole authority for the registered findings.
**External sources used:** none. No literature search was performed, so the `T0` boundary is not engaged; every number below derives from the frozen workspace data.

---

## 1. Summary

The workspace contains three small tabular files describing a biohybrid/energy
operating system: a dose scan (`dose_response.tsv`, 3 rows), a perturbation
time response (`perturbation_time_response.tsv`, 4 rows), and a much larger
"time response curves" table (`time_response_curves.tsv`, 1367 rows).

The central result of this analysis is structural and it conditions every other
claim: **`time_response_curves.tsv` is not one homogeneous trajectory.** It is a
concatenation of three regimes that differ in time resolution, dynamic range,
noise model, and treatment of the fitted columns. The column named `time` is a
usable time axis for only ~2.6% of the rows. Treating the file as a single
time series — the natural reading — produces a strong apparent "decay over
time" that is an artefact of regime concatenation, and it produces a strong
apparent `condition_a`–`condition_b` coupling that is almost entirely a shared
smooth trajectory rather than a within-condition relationship.

Five findings are registered. Two are affirmative (a conditional coupling
boundary; a hysteresis/path-dependence result), one is a structural
prerequisite, one is a negative/derived-column result, and one reports that a
requested relationship is **unresolved** because the data cannot support it.

---

## 2. Data inventory, schema, and integrity

All three files were hashed and compared against `data_manifest.json`.

| File | Rows x Cols | Bytes | sha256 match |
|---|---|---|---|
| `data/time_response_curves.tsv` | 1367 x 7 | 45747 | yes |
| `data/perturbation_time_response.tsv` | 4 x 7 | 285 | yes |
| `data/dose_response.tsv` | 3 x 3 | 83 | yes |

Integrity receipt: `output/results/findings.json` → `data_integrity`.
Every declared hash matched the observed hash; no substitution or truncation
was detected.

**Columns.** `time_response_curves.tsv`:
`time`, `promoter_like_signal`, `condition_a`, `condition_b` (all integer-valued),
`condition_c`, `fit_a`, `fit_b` (float-valued).
`perturbation_time_response.tsv`: `time` plus `{control,condition_a,condition_b}_{mean,sd}`.
`dose_response.tsv`: `dose`, `response_mean`, `response_sd`.

**Missingness.** There are no `NaN` values in `time_response_curves.tsv` or
`perturbation_time_response.tsv`. `dose_response.tsv` has `response_sd`
missing for **all three** rows (`response_mean` and `dose` are complete). This
is the only missingness in the task, and it is material to Finding 5.

**Analysis unit.** Per `DATA_GUIDE.md` and `data_manifest.json`, the independent
unit is *one complete condition-specific trajectory or one complete dose setting
with its uncertainty*, and *fitted columns are supporting representations, not
independent observations*. This analysis honours that: raw and fitted columns
are never pooled as independent evidence (quantified in Finding 4), and the
perturbation file is treated as **three** independent trajectories (control,
`condition_a`, `condition_b`) of four timepoints each, not as twelve
independent observations.

**A label collision worth recording.** The string `condition_a` / `condition_b`
appears in both the curves file and the perturbation file, but the two files
are on incompatible scales (curves: integer counts in the thousands;
perturbation: fractional means near 0.32 with SDs ~0.003). `condition_c`
appears **only** in the curves file, where it behaves as a fitted column
(Finding 4), not as a fourth experimental perturbation. Any cross-file
comparison of these like-named columns would be invalid.

---

## 3. Registered findings

### Finding 1 — `time_response_curves.tsv` is three concatenated regimes, and the `time` column is a valid axis for only 2.6% of rows

**Object and scope.** All 1367 rows of `data/time_response_curves.tsv`, and the
`time` column's role as an independent variable.

**Relationship / boundary.** Row index and the `time` column are decoupled
outside a narrow band. Segmentation by *time-value support* (contiguous runs of
rows whose `time` belongs to the multi-point sweep set `{t : t >= 2}`) yields
exactly three regimes:

| Regime | Rows | n | % of file | Distinct `time` values | `promoter_like_signal` range | Fitted columns |
|---|---|---|---|---|---|---|
| **A — baseline** | 0–258 | 259 | 18.9% | {0, 1} | 0 – 7 | all exactly 0 |
| **B — sweep** | 259–293 | 35 | 2.6% | 25 values, 0 … 10083 | 0 – 10009 | nonzero from row 259 |
| **C — tail** | 294–1366 | 1073 | 78.5% | {0, 1, 2} | 10 – 3866 | nonzero, smoothly declining |

Regime B is the only region carrying the 23 multi-point sweep times (2, 6, 7,
8, 16, 47, 71, 108, 154, 212, 264, 464, 561, 1052, 2052, 2823, 4380, 5708,
6611, 8076, 9015, 9643, 10083). Outside rows 259–293 there is **exactly one**
row with `time >= 2` (row 910). In regime A the fitted columns `condition_c`,
`fit_a`, `fit_b` are *exactly* zero for all 259 rows and first become nonzero
at rows 263, 263, and 259 respectively.

**Core quantitative result.** The `time` column explains a large share of
variance in the pooled file but essentially none within regime C:
`R²(promoter_like_signal ~ time as a categorical factor)` = **0.6674** pooled,
versus **0.0002** restricted to regime C. The pooled value is carried entirely
by regime B.

**Supporting analysis and result evidence.**
- Source: `output/analysis/analysis.py`, function `part1_regimes` (`segment`).
- Results: `output/results/findings.json` → `finding1_regimes`;
  `output/results/table_regimes.tsv`;
  `output/figures/fig1_regime_structure.png` (panel a shows the three regimes
  against row index on a symlog axis; panel b shows the same values against the
  labelled `time`).

**Robustness.** (i) *Boundary sensitivity*: the A|B and B|C cut points were
moved over a 7 x 7 grid (±40 rows, 37 admissible variants). Regime C's
`condition_a`–`condition_b` correlation stayed in [0.679, 0.973] and its trend
R² stayed high in every variant, so the *existence* of a distinct third regime
is not a cut-point artefact — though the baseline null in Finding 3 *is*
(see that finding). (ii) *Alternative sweep definition*: counting every row
with `time >= 2` anywhere gives 28 rows; taking the contiguous run containing
the maximum `time` gives 25 rows; the segmentation used gives 28 rows in
sweep-supported runs of which 25 are contiguous. All three definitions put the
sweep at 1.8–2.6% of the file, so the conclusion is not definition-dependent.
(iii) *Leave-one-regime-out*: the pooled linear trend of
`promoter_like_signal` on `time` over all 1367 rows is slope **0.9101**,
R² **0.24875**. Dropping regime B raises the slope to **15.9754** per time unit
while collapsing R² to **0.000034** (n = 1332). The apparent pooled time trend
is therefore carried by the 35 sweep rows: remove them and the trend explains
essentially nothing. For completeness, the other two folds give slope 0.9000 /
R² 0.24966 (drop regime A) and slope 0.8925 / R² 0.30844 (drop regime C), i.e.
removing the sweep is the only fold that changes the picture
(`findings.json` → `finding1_regimes.leave_one_regime_out_trend`).

**Counterevidence, limitations, alternative explanations.** The regime
boundaries are inferred, not declared: the manifest describes the file as a
single entity and no column marks the regimes. An alternative reading is that
the whole file is one long experiment whose early rows are a low-signal
warm-up; that reading does not survive the *exact zero* of all three fitted
columns across all 259 regime-A rows, nor the clean value separation. For
`promoter_like_signal` regime A spans [0, 7] and regime C spans [10, 3866] —
**no overlap at all** (0 of 1073 regime-C rows fall at or below regime A's
maximum). The same holds for `condition_a` (A: [0, 4]; C: [11, 4500]; 0 of 1073
overlapping). The separation is **not** clean for `condition_b`, however:
regime A spans [0, 3] while regime C spans [0, 801], and 255 of 1073 regime-C
rows (23.8%) fall at or below regime A's maximum of 3. The regime boundary
therefore rests on the two high-amplitude channels, and a reader who weights
`condition_b` alone would see a much weaker boundary. Regime B could also be
read as an instrument calibration sweep stored inline; the data cannot
distinguish "calibration" from "biological response", and this report does not
assume either.

**Falsification / independent replication.** If regime A and regime C are the
same physical state, then a fresh run of the same protocol should produce
regime-A-scale integer values (0–7 for `promoter_like_signal`) in tail rows.
In these data **none** of the 1073 regime-C rows does so (the minimum is 10),
and the only 2 file-wide rows at that scale sit at rows 259–260, i.e. at the
start of the sweep regime, not in the tail. A new observation of a resampled
tail with a material fraction (>10%) of `promoter_like_signal` values <= 7
would refute the regime separation; on the `condition_b` channel alone that
threshold is already met, which is why the claim is carried by the
`promoter_like_signal` and `condition_a` channels. Equivalently, an independent
run that supplies an explicit `regime` or `cycle_id` column would settle it
directly.

---

### Finding 2 — The sweep regime shows path dependence (hysteresis) that is strongly channel-dependent and absent from one channel

**Object and scope.** Rows 259–293 of `data/time_response_curves.tsv`, the only
region where the labelled `time` axis is informative; channels
`promoter_like_signal`, `condition_a`, `condition_b`.

**Relationship / direction / boundary.** Inside regime B the rows trace an
ascending branch (rows 259–267, `time` climbing 0 -> 10083) followed by a
descending branch (rows 268–283, `time` falling 9643 -> 2) and a short tail
(rows 284–293). The measured signal is therefore **not a single-valued function
of the labelled `time` within the sweep**: at a matched nominal time the
descending branch sits far above the ascending branch for two of the three
channels.

**Core quantitative result.** At the exactly matched labelled time `t = 2`:

| Channel | Ascending (row 260) | Descending (rows 284/286/288) | Ratio (desc/asc) | Jackknife range of ratio |
|---|---|---|---|---|
| `promoter_like_signal` | 3 | 5248, 4942, 4628 | **1543 – 1749** | 1595 – 1698 |
| `condition_a` | 4 | 5642, 5364, 5084 | **1271 – 1411** | 1306 – 1376 |
| `condition_b` | 43 | 1312, 1171, 1086 | **25.3 – 30.5** | 26.2 – 28.9 |

On a common log-time grid (50 … 9000) the loop area
`∫ (down − up) d(log10 t)` is **16682** (`promoter_like_signal`) and **16838**
(`condition_a`) against a global channel maximum of ~10000, i.e. the loop
height reaches **0.838** of the full observed range. For `condition_b` the loop
area is only **1097**, i.e. **0.179** of range.

The hysteresis is also *channel-specific in timing*: `promoter_like_signal` and
`condition_a` both apex at row 270, `time = 5708`, while `condition_b` apexes
earlier at row 267, `time = 10083` — a difference of 4375 time units. The
descending-branch peak exceeds the ascending-branch peak by **1.74x** for
`promoter_like_signal` and **1.68x** for `condition_a`, but the
`condition_b` ratio is **0.87x** (the descending branch peaks *below* the
ascending branch).

**Supporting analysis and result evidence.**
- Source: `output/analysis/analysis.py`, function `part2_pathdependence`.
- Results: `output/results/findings.json` → `finding2_path_dependence`
  (keys `matched_time_branches`, `loop_on_logtime_grid`, `apex_location`,
  `jackknife_matched_t2_ratio`, `branch_effect_after_logtime`);
  `output/results/table_matched_time_branches.tsv`;
  `output/figures/fig2_hysteresis.png`; `output/figures/fig1_regime_structure.png`
  (panels c and d contrast the same rows keyed by row index vs by `time`).

**Robustness.** (i) *Complete-unit leave-out*: dropping each of the three
matched `t = 2` descending replicates one at a time moves the ratio only within
1595–1698 (`promoter_like_signal`), 1306–1376 (`condition_a`) and 26.2–28.9
(`condition_b`); no single row drives the loop. (ii) *Alternative model*:
regressing the channel on log10(`time`) using **only** the ascending branch and
then predicting the descending branch leaves a mean residual of **+6170** for
`promoter_like_signal`, **+6352** for `condition_a` and **+649** for
`condition_b` — the same ordering and roughly the same relative magnitudes as
the raw matched-time ratios, so the effect is not an artefact of the
interpolation grid.

**Counterevidence, limitations, alternative explanations.** The strongest
counter-consideration is that branches can only be matched at `t = 2`, `t = 1`
and `t = 6`, because those are the only times appearing on both branches; the
log-time-grid loop therefore rests on interpolation across nine grid points.
The ascending branch has just nine rows, so its shape is coarsely sampled. An
alternative explanation for the loop is **instrument settling or
response lag** rather than a physical hysteresis: the ascending branch is
sampled at rapidly increasing `time` and may not have equilibrated, which would
push the descending branch above it for *any* monotone loading process. These
data cannot separate true hysteresis from slow equilibration, because there is
no independent replicate of the sweep and no wait-time column. A third
alternative — that the descending rows simply come from a different experiment
— cannot be excluded either, but the smooth continuity of the descending branch
into regime C at row 293/294 argues against a hard break there.

**Falsification / independent replication.** Re-run the same sweep with
dwell times at each `time` setting (or a reversal at `t = 10083` back to
`t = 2` *without* the apex) and measure the same channel. True hysteresis
predicts the loop persists at arbitrarily long dwell; slow equilibration
predicts the loop collapses toward zero as dwell increases. Concretely: if the
`promoter_like_signal` descending/ascending ratio at `t = 2` does not stay
above, say, 10 after dwell is at least doubled, the hysteresis reading is
refuted.

---

### Finding 3 — The `condition_a`–`condition_b` coupling is regime-conditional: a genuine null in the baseline regime, and almost entirely a shared trajectory in the tail regime

**Object and scope.** Association among `promoter_like_signal`,
`condition_a`, `condition_b` in each regime of
`data/time_response_curves.tsv`, at the level of the whole trajectory.

**Relationship / direction / boundary.** The pairwise association changes by
two orders of magnitude across regimes, and the pooled estimate is an artefact
of pooling. In the baseline regime the two conditions are **uncorrelated**; in
the tail regime they are strongly correlated, but that correlation disappears
under first-differencing. The boundary is the regime boundary of Finding 1.

**Core quantitative result.**

| Regime (n) | `prom`~`a` | `prom`~`b` | **`a`~`b`** |
|---|---|---|---|
| A baseline (259) | +0.1095 | −0.0474 | **+0.0062** |
| B sweep (35) | +0.9967 | +0.4151 | +0.3938 |
| C tail (1073) | +0.9834 | +0.9899 | **+0.9523** |
| Pooled (1367) | +0.9835 | +0.7301 | +0.6789 |

The baseline `a`~`b` correlation is **+0.0062**, 95% CI **[−0.116, +0.128]**,
`r² = 0.00004`. In the same regime the OLS slope of `b` on `a` is +0.00428, so
spanning the *entire* observed `condition_a` range (0–4) predicts a change in
`condition_b` of only **+0.0171** against a `condition_b` SD of **0.5310** —
about 3% of one SD.

In the tail regime the same correlation is **+0.9523**, but the correlation of
**first differences** falls to **+0.1551**, and the correlation after removing
a 15-row centred rolling median from both channels falls to **+0.3051**. The
share of the pooled `a`~`b` association attributable to the shared smooth
trajectory is `1 − (0.0062² / 0.6789²)` = **0.99992**.

**Supporting analysis and result evidence.**
- Source: `output/analysis/analysis.py`, function `part3_coupling`.
- Results: `output/results/findings.json` → `finding3_coupling` (keys
  `levels_by_regime`, `first_differences`, `spearman`,
  `detrended_rolling_median`, `trend_share_of_pooled_ab_r2`) and
  `fano_by_regime`; `output/figures/fig3_coupling_by_regime.png`.

**Robustness.** Four checks were run. (i) *First differences* remove the shared
trajectory (tail: 0.9523 -> 0.1551). (ii) *Rank correlation* reproduces the
level pattern. (iii) *Detrending* by a rolling median gives tail 0.3051.
(iv) *Boundary sensitivity* — and this is the important one — re-cutting the
regimes over 37 admissible variants shows the baseline `a`~`b` correlation
ranges from **+0.0062 to +0.9522** depending on where the A|B boundary is
placed. Every variant that places the boundary at row >= 268 visibly starts to
include sweep rows, and the correlation jumps immediately. This is direct
evidence that the null result is a property of the *baseline-regime rows*, not
of the file as a whole; it is reported as a warning in the run receipt
(`output/results/findings.json` → `_meta.warnings`).

A supporting noise-model check (`fano_by_regime`) shows why the two regimes
must not be pooled: after local detrending, regime A's channels have Fano
factors of **0.98, 1.02, 0.95** (consistent with unit-variance counting noise),
while regime C's are **1.94, 1.11, 1.50**. Regime A therefore behaves like a
quiet baseline under a simple counting-noise model, and regime C does not.

**Counterevidence, limitations, alternative explanations.** The baseline null is
a *bounded* null, not proof of independence: with n = 259 and 80% power the
smallest reliably detectable `|r|` is about **0.174**, so a true coupling
smaller than that would be missed. The `condition_b` channel in regime A is
largely zeros (182 of 259 rows exactly 0), which limits its information content;
and `promoter_like_signal`~`condition_a` in regime A is +0.1095, i.e. weakly
positive and not distinguishable from zero at this n. The alternative
explanation for the tail correlation is mundane and, on this evidence, correct:
two smooth monotone series will correlate strongly even when the underlying
processes are unrelated, and first-differencing is the appropriate diagnostic
— the drop from 0.952 to 0.155 supports that reading rather than a mechanistic
coupling. The lead-lag scan in the tail (`finding3_coupling.leadlag_C_tail_best`)
found `r` increasing monotonically up to the edge of the searched window
(best `r = 0.978` at lag 25 vs `r = 0.952` at lag 0); because the correlation
increases with `|lag|` rather than peaking, **no genuine lead-lag delay is
identified** and the result is reported as an unresolved non-result, not as a
delay estimate.

**Falsification / independent replication.** Under the null there is no
`a`–`b` relationship in the baseline regime. A new baseline run of ~260
records from the same instrument should reproduce `|r| < 0.13`; an observed
`|r| > 0.17` with the same n would refute the null. For the tail-regime claim,
the decisive test is to interleave an independent, genuinely randomized driver
into the tail regime: if first-difference correlation remains near 0.15 under
randomization, the coupling is a trajectory artefact.

---

### Finding 4 — The fitted columns are derived and strictly monotone; they are not independent observations and cannot be counted twice

**Object and scope.** The columns `condition_c`, `fit_a`, `fit_b` of
`data/time_response_curves.tsv` and their relationship to
`promoter_like_signal`, `condition_a`, `condition_b`.

**Relationship / direction / boundary.** Each fitted column is a near-perfect
deterministic function of the matching raw channel in **row order**, with the
same measurements. They therefore carry no independent information. The
conditional boundary is that this derivation is only expressed where the fit is
defined: in regime A all three fitted columns are exactly 0.

**Core quantitative result.** Correlations between each raw channel and its
fitted column (rows >= 259, n = 1108) are **0.99956** (`condition_a` vs
`fit_a`), **0.99878** (`condition_b` vs `fit_b`) and **0.99950**
(`promoter_like_signal` vs `condition_c`), all at **lag 0** (no lag improves
them), with residual SDs of 38.1, 35.3 and 36.3 respectively against channel
ranges of ~10000.

The decisive evidence is a **monotonicity contrast**. Over the 1072 consecutive
steps of regime C:

| Column | Fraction of steps strictly descending | Sign changes |
|---|---|---|
| `condition_c` | **1.0000** | **0** |
| `fit_a` | **1.0000** | **0** |
| `fit_b` | **0.9450** | 74 |
| `promoter_like_signal` | 0.4991 | 743 |
| `condition_a` | 0.5159 | 735 |
| `condition_b` | 0.4776 | 802 |

The raw channels reverse direction about half the time, exactly as noisy data
should; `condition_c` and `fit_a` **never** do. A permutation test (2000
shuffles) gives a permutation mean of 0.4997 with SD 0.0089 against an observed
value of 1.0000, `p < 0.001` (0 of 2000 permutations reached the observed
value) for all three fitted columns. The fitted columns are therefore smoothed
representations, not measurements.

The smoothing operator is identifiable but **the axis it was applied along is
not**. A centred moving average with a short window (w = 3–7) reproduces each
fitted column with RMSE 10.2–11.1 and `r = 0.9966–0.9999`. However, applying
the *same* smoother to the raw channel **after a descending sort** fits at
least as well (best overall: descending sort, w = 3–7, RMSE 10.2–11.1) as
applying it in row order (fit-vs-raw RMSE 19.6 / 22.9 / 12.3). This ambiguity
exists because regime C is itself already very close to a descending sort of
its own values (`r` between raw and its own descending sort = 0.9996 / 0.9997 /
0.9978), so the two candidate axes are nearly indistinguishable here. The
honest conclusion is: **the fit is a short-window centred moving average, but
whether the underlying sequence is row order or a magnitude-sorted order is
unresolved by these data.**

**Supporting analysis and result evidence.**
- Source: `output/analysis/analysis.py`, function `part4_fitted`.
- Results: `output/results/findings.json` → `finding4_fitted_columns` (keys
  `association`, `window_scan`, `monotonicity_permutation`, `sortedness`,
  `operator_test`, `double_counting`, `sign_changes_C_tail`);
  `output/results/table_fit_vs_raw.tsv`;
  `output/figures/fig4_raw_vs_fitted.png`.

**Robustness.** (i) *Complete-unit leave-out / alternative definition*: a
12-point window scan over two smoothing modes (`centred` vs `trailing`) was run
for all three pairs; the centred short-window form wins in every case, so the
"smoothed representation" reading does not depend on the window chosen.
(ii) *Alternative model*: the monotonicity result is invariant to the window
parameter, because it is a count of direction reversals rather than a fit.
(iii) *Double-counting quantification*: pooling raw and fitted columns would
inflate the nominal sample from 1108 to 2216 rows — a factor of **2.0** — while
the residuals (fit − raw) retain lag-1 autocorrelation of **0.48 / 0.47 /
0.74**, so even the *residuals* are not independent across rows.

**Counterevidence, limitations, alternative explanations.** Fitted columns are
not worthless — they are a legitimate supporting representation, which is why
`DATA_GUIDE.md` names them as such. The finding here is about *evidentiary
status*, not about deleting them. One genuine limitation: without the
generating code or a declared functional form, the moving-average
characterisation is an empirical description fitted to these data, not a
recovered formula; a different smoother (e.g. a Savitzky–Golay filter or a
spline) could match these columns similarly. A second limitation is the axis
ambiguity described above, which is a real unresolved question rather than a
settled result, and is reported as such.

**Falsification / independent replication.** If raw and fitted columns are
independent observations, the fitted column should contain noise the raw column
does not. A direct test with new data: compute the residual `fit − raw` and
check whether it correlates with an *independent* replicate of the same
condition. If it does not (as these data imply), the fitted column is derived.
Concretely, an independent run that records raw and fitted channels and whose
`fit` column has a nonzero fraction of direction reversals (> 5%) in a monotone
stretch would refute the smoothing reading.

---

### Finding 5 — The dose scan cannot identify an optimum (unresolved), while the perturbation trajectories give a clear, well-separated and control-stable response ordering

**Object and scope.** `data/dose_response.tsv` (3 settings) and
`data/perturbation_time_response.tsv` (three trajectories x 4 timepoints). These
are separate experiments and are reported separately; they are linked only in
that both describe the same operating system.

**Relationship / direction / boundary.** Two different answers with two
different evidence standards:

**(5a) Dose response — UNRESOLVED, by design of the data.** The scan has
**3 settings**, `response_sd` is **missing for all 3 rows**, and the response is
**non-monotone** (3.4919 at dose 21.69508; 6.98379 at 31.77532; 6.04814 at
47.90370). The interior apex is visible but unsupported:

- linear fit: slope +0.0831, `R² = 0.369`;
- log-log fit: slope +0.6803 (Hill-like exponent), `R² = 0.543`;
- second difference: **−4.428** (concave);
- a quadratic in dose has 3 free parameters for 3 points, so it fits exactly
  (`R² = 1` by construction) and carries **zero** independent evidence for an
  optimum;
- leave-one-setting-out is unstable: dropping setting 1 gives slope **−0.0580**
  (sign flip), dropping setting 2 gives **+0.0975**, dropping setting 3 gives
  **+0.3464**; the residuals at the dropped setting are −4.077, +2.509 and
  −6.523 respectively.

Because no `response_sd` is available, no confidence interval can be computed
for any dose-response estimate. Both warnings in `data_manifest.json` are
therefore operative and correct: *a universal optimum cannot be inferred*.

**(5b) Perturbation trajectories — SUPPORTED.** All twelve supplied SDs (4
timepoints x 3 trajectories) are present, with none missing and a range of
0.00135 to 0.00850. Over the four timepoints (0, 2, 4, 6):

| Trajectory | t=0 | t=6 | Change | % change | Strictly decreasing | log-linear `R²` | rate | half-life | effect / pooled SD |
|---|---|---|---|---|---|---|---|---|---|
| control | 0.32243 | 0.32397 | **+0.00154** | +0.48% | no | 0.278 | +0.00087 | — | 0.27 |
| `condition_a` | 0.32137 | 0.27817 | **−0.04320** | −13.44% | **yes** | 0.9609 | −0.02351 | 29.48 | 11.43 |
| `condition_b` | 0.32387 | 0.22950 | **−0.09437** | −29.14% | **yes** | 0.9697 | −0.05526 | 12.54 | 24.05 |

The control is flat (drift 0.27 pooled SD, non-monotone), while both
conditions decay strictly monotonically with effects of 11.4 and 24.1 pooled
SDs. `condition_b` is the stronger perturber: **2.18x** the total change of
`condition_a` (2.35x on the log-linear rate). Normalising each condition to the
matched control at each timepoint gives `condition_a`
[0.9967, 0.9310, 0.8943, 0.8586] and `condition_b`
[1.0045, 0.8526, 0.7911, 0.7084]. A log-linear (exponential) description beats
a linear one for both (`condition_a` 0.9609 vs 0.9524; `condition_b` 0.9697 vs
0.9517), so the decay is at least as consistent with a constant fractional
loss rate as with a constant absolute rate — with only 4 timepoints this
distinction is weak and is reported as a preference, not a determination.

**Supporting analysis and result evidence.**
- Source: `output/analysis/analysis.py`, function `part5_dose`.
- Results: `output/results/findings.json` → `finding5_dose_and_perturbation`
  (keys `dose_scan`, `perturbation`, `bootstrap_from_supplied_sd`,
  `dose_alternative_summaries`); `output/figures/fig5_dose_perturbation.png`.

**Robustness.** (i) *Alternative model*: for the dose scan, three functional
forms (linear, log-log, quadratic) plus a complete leave-one-setting-out sweep
were compared; all three framings agree that the apex is unsupported.
(ii) *Threshold/uncertainty sensitivity*: a 20 000-draw bootstrap that perturbs
**only the two endpoint timepoints** (t = 0 and t = 6) by their supplied SDs
gives `condition_a` change = −0.0432, 95% CI [−0.0546, −0.0315];
`condition_b` change = −0.0943, 95% CI [−0.1058, −0.0832]; and the
difference `b − a` = −0.0512, 95% CI [−0.0673, −0.0348] with
`P(diff <= 0) = 1.000` (0 of 20 000 draws). The `condition_b` >
`condition_a` ordering is therefore not marginal **under the sampled SDs**.
This interval is deliberately narrow because it holds the intermediate
timepoints fixed and treats the supplied SDs as known; it quantifies
endpoint-measurement precision, not experiment-to-experiment reproducibility.
Independent verification quantified how optimistic: resampling **all four**
timepoints of both trajectories instead of only the endpoints widens the
`b − a` interval to [−0.0676, −0.0348] — essentially unchanged — and a
timepoint-wise paired check of the `b < a` separation over t = 2, 4, 6 gives a
combined Stouffer z of **−16.28** with `b` below `a` at every post-baseline
timepoint, so the ordering does not depend on the endpoint-only construction.
The honest caveat is therefore about *experiment-to-experiment* reproducibility
across only 4 timepoints, not about the interval arithmetic.

**Counterevidence, limitations, alternative explanations.** The perturbation
result rests on **4 timepoints per trajectory**, so the half-lives (29.5 and
12.5) are extrapolations from a short window and should be treated as ordering
statistics, not calibrated constants. The experiment is also not
concentration-matched to the dose scan: the perturbation file carries no dose
column and the dose file carries no condition label, so no dose-response curve
can be attached to `condition_a` or `condition_b` from these data. A plausible
alternative explanation for the perturbation ordering is simply that the two
conditions were applied at different amplitudes; because the applied amplitude
is not recorded, **the 2.18x ratio is a difference in observed response, not
evidence of a difference in potency per unit dose**. This distinction is the
main interpretive limit of 5b.

**Falsification / independent replication.** For 5a, the unresolved status is
falsified (i.e. resolved) by re-running the dose scan with at least 6 settings
spanning the same range plus `response_sd` at each; a monotone or
single-interior-peak curve surviving leave-one-out would then establish the
optimum. For 5b, re-running the perturbation at three different applied
amplitudes for each condition would test whether the ordering is a potency
difference or an amplitude difference; if the curves superpose after
normalisation by applied amplitude, the "different potency" reading is refuted.

---

## 4. Cross-cutting limitations

1. **No replicate structure.** The curves file contains no replicate, batch,
   subject, or session identifier. Exact-duplicate rows exist (209), but nothing
   marks them as replicates rather than coincidences, so no batch leave-out or
   hierarchical variance decomposition is possible. This bounds every finding
   above.
2. **`condition_c` is not a fourth condition.** It appears only in the curves
   file and tracks `promoter_like_signal` as a fitted column (Finding 4). The
   manifest's variable dictionary lists it under
   `time_response_curves.tsv` measures, but its behaviour is that of a derived
   column. It is not treated as an experimental arm anywhere in this report.
3. **External literature boundary.** No external source was consulted, so the
   `T0 = 2023-01-01` boundary is satisfied trivially and there are no citations
   to record. All mechanism language above (e.g. "loading", "equilibration")
   is used as a description of the observed trajectory shape, not as an appeal
   to published mechanism.
4. **`scipy` was unavailable in this runtime.** All statistics — OLS,
   Pearson/Spearman correlation, permutation tests, and the bootstrap — were
   implemented directly with `numpy`/`pandas`. The run receipt records this
   (`output/results/findings.json` → `_meta.scipy_available = false`) so the
   numerical provenance is explicit.
5. **Reported nulls are bounded nulls.** Finding 3's baseline result is a null
   *at the achieved sensitivity* (`|r|` >= 0.174 detectable at 80% power), not
   an absence of any relationship.

---

## 5. What was tested and did not work as expected

Recorded here rather than discarded, because they bear on the registered claims.

- **A block bootstrap intended as a null for the tail decline failed.** The
  first version resampled 25-row blocks and concatenated them in random order,
  which mixes regions of very different local level and destroys the trend by
  construction (mean `R²` = 0.024 against an observed 0.393). That procedure is
  not a valid null and is retained in the results only as a documented failure
  (`output/results/findings.json` → `tail_decline_order_null.block_bootstrap_note`).
  It was replaced with a plain row-shuffle null, under which the observed
  `R² = 0.393` is far outside the null distribution (null mean 0.00094, 95% CI
  [5.4e-7, 0.0046], `p < 0.001`, 0 of 1000 shuffles reached it), and with a
  segment-persistence check in which **all 20** consecutive 54-row segments of
  regime C have negative slope (median −0.272, range −43.74 to −0.034).
- **The apparent lead-lag in the tail did not replicate as a delay.** The
  cross-correlation between `condition_a` and `condition_b` rises monotonically
  for `lag >= 0`, reaching the edge of the searched window (0.9523 at lag 0,
  0.9784 at lag +25) instead of peaking, and falls monotonically as `|lag|`
  grows on the negative side (0.9510 at lag −1 down to 0.9309 at lag −25).
  No interior peak exists anywhere in lags −25…+25
  (`findings.json` → `finding3_coupling.leadlag_C_tail_shape`). A genuine delay
  produces an interior peak; this shape is the signature of two smooth series.
  Reported as an unresolved non-result.
- **The "smoothed representation" description is only partly resolved.** The
  operator (short centred moving average) is identifiable; the axis along which
  it was applied is not, because regime C is nearly a descending sort of its own
  values. This ambiguity is stated in Finding 4 rather than resolved.
- **The dose scan did not support any optimum.** Three settings with no SD and a
  sign-flipping leave-one-out cannot distinguish a peak from a plateau or from
  noise. This is the requested "unresolved" outcome, not a missing analysis.

---

## 6. Reproduction

**Data.** Frozen inputs at `data/` were verified against `data_manifest.json`
by sha256 inside the analysis run; all three matched
(`output/results/findings.json` → `data_integrity`).

**Environment.** Python 3.11, `numpy` 2.4.6, `pandas` 3.0.5, `matplotlib`
3.11.2, all recorded in `_meta`. `scipy` was **not** installed (see
limitation 4); `matplotlib` and its dependencies were installed into the user
site at run time. No network access was used for data.

**Command.**

```
python3 output/analysis/analysis.py
```

The script is self-contained and deterministic. It uses a fixed RNG seed
(`RNG_SEED = 20230919`, defined at the top of the script) for every permutation
and bootstrap draw, so the reported intervals are exactly reproducible. It reads
only `data/` and writes only under `output/`.

**Produced artifacts.** `output/results/findings.json` (complete machine-readable
result tree), `output/results/table_regimes.tsv`,
`output/results/table_matched_time_branches.tsv`,
`output/results/table_fit_vs_raw.tsv`,
`output/results/table_headline_numbers.tsv`, and
`output/figures/fig1_regime_structure.png` …
`output/figures/fig5_dose_perturbation.png`.

**Independent verification.** After the first analysis pass, eight independent
verifier agents were run through the native workflow tool
(`output/verification/`), one per registered finding plus three artifact-level
checks (tables/`findings.json` consistency, figure fidelity including PNG
pixel read-back, and cross-cutting report claims). Each verifier wrote and ran
its own scripts from the raw TSVs without executing `analysis.py`. Across the
run, **632 individual numeric claims** were recomputed; the verifiers reported
three discrepancies, all of which are corrected above and none of which changed
a registered conclusion:

1. **Finding 1 text/value pairing (corrected).** The leave-one-regime-out
   sentence quoted 0.900 / 0.250 as the baseline for the "dropping regime B"
   comparison. Those are the *drop-regime-A* values; the correct pooled baseline
   is 0.9101 / 0.24875. The pooled reference is now reported alongside all folds,
   and `findings.json` gained an explicit `POOLED_all_rows` entry.
2. **Finding 5 SD count (corrected).** The report said "all six supplied SDs";
   the file contains **12** (4 timepoints x 3 trajectories). Corrected, with the
   range 0.00135–0.00850 now stated and an `sd_inventory` record added.
3. **Figure 4a title (corrected).** Panel (a) is titled "full row range" but
   plots rows 259–1366, because the fitted columns are exactly zero for all 259
   regime-A rows. The title now states the plotted range and the reason.

Two further verifier observations strengthened rather than changed the report
and are now recorded in `findings.json`: resampling **all** perturbation
timepoints widens the `b − a` interval only to [−0.0676, −0.0348] (from
[−0.0673, −0.0348]), and a per-timepoint paired check gives a Stouffer
`z = −16.28` with `b < a` at every post-baseline timepoint. The verifier also
confirmed `fig3`'s 24 bars match recomputation to within 1.9 px, and that
`fig5(b)`'s error bars match the supplied SDs to within 1 px. Verification
scripts and outputs are preserved unedited in `output/verification/`.

**Finding-to-evidence map.**

| Finding | Analysis source | Primary result evidence |
|---|---|---|
| Finding 1 | `output/analysis/analysis.py` (`part1_regimes`) | `output/results/findings.json`, `output/results/table_regimes.tsv`, `output/figures/fig1_regime_structure.png` |
| Finding 2 | `output/analysis/analysis.py` (`part2_pathdependence`) | `output/results/findings.json`, `output/results/table_matched_time_branches.tsv`, `output/figures/fig2_hysteresis.png` |
| Finding 3 | `output/analysis/analysis.py` (`part3_coupling`) | `output/results/findings.json`, `output/figures/fig3_coupling_by_regime.png` |
| Finding 4 | `output/analysis/analysis.py` (`part4_fitted`) | `output/results/findings.json`, `output/results/table_fit_vs_raw.tsv`, `output/figures/fig4_raw_vs_fitted.png` |
| Finding 5 | `output/analysis/analysis.py` (`part5_dose`) | `output/results/findings.json`, `output/figures/fig5_dose_perturbation.png` |
| (verification, all findings) | `output/verification/verifier_finding*.py`, `output/verification/verifier_artifact_consistency.py`, `output/verification/verifier_figures.py`, `output/verification/independent_verifier.py` | `output/verification/verifier_artifact_consistency_results.json`, `output/verification/verifier_figures_out.json`, `output/verification/verifier_finding1_results.json`, `output/verification/verifier_finding1_extra_results.json`, `output/verification/verifier_finding3_results.json`, `output/verification/verifier_finding5_results.json`, `output/verification/verifier_full_output.txt` |

A standalone reproduction note with the same commands is provided at
`output/reproduce.md`.
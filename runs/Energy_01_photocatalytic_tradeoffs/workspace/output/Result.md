# Cross-Modal Performance and Kinetic Tradeoffs in Photocatalytic Formulations

**Task:** `Energy_01_photocatalytic_tradeoffs` · **T0:** 2022-01-01
**Evidence base:** the four frozen files in `data/` only. No external dataset was downloaded,
substituted, or merged; no literature was searched for the core claims.
**Reproduction:** see [`reproduce.md`](reproduce.md). Analysis source is in `analysis/`; machine-readable
results are in `results/`; an independent verification and alternative-explanation battery is in
`verification/`; figures are in `figures/`.

The four modalities are *not* a single joined table — this is the structural fact that governs most
of what follows. `conversion_timecourses.tsv` and `ultrafast_transient_traces.tsv` carry a
`formulation_id`; `optical_action_spectra.tsv` and `cofactor_loading_response.tsv` carry **no**
formulation identifier at all, and the loading scan's absolute rate scale is separated from the
time-course rate scale by at least two orders of magnitude (pairwise ratios span 113× to 28 139×). The independent analysis unit is one complete curve, spectrum,
trace, or loading condition; adjacent points along a curve are a serial sweep, not replicates.

---

## Finding 1 — Product evolution separates into two opposed kinetic-shape groups, but the winning functional form is not identifiable at five points per curve

**Object and scope.** The four product-evolution time courses in `data/conversion_timecourses.tsv`
(`formulation_id`, `illumination_time_h`, `gaseous_product_umol`; 4 formulations × 5 times, 0–4 h).

**Relationship and direction.** Final yield at 4 h spans 54.8× across formulations: 9.48, 4.13,
0.636, and 0.173 µmol for formulations 1–4. More interesting than the spread is that the *shape*
splits into two mutually opposed groups. Formulations 1 and 2 are near-linear to slightly
decelerating — successive segment rates (µmol/h) of [2.40, 2.40, 2.44, 2.24] and [1.21, 0.95, 1.00,
0.97], giving last/first segment-rate ratios of **0.933** and **0.802**. Formulations 3 and 4 are
clearly accelerating — [0.124, 0.132, 0.189, 0.191] and [0.0329, 0.0395, 0.0486, 0.0520], giving
ratios of **1.540** and **1.581**. The two groups are separated in the power-law exponent
`y = a·t^b`: b = 0.984 and 0.906 for the first pair, 1.249 and 1.230 for the second.

**Core quantitative result.** Analysis unit = one complete curve (n = 4 curves, 5 points each). The
single most robust statement is the sign of the shape difference, not the identity of the model.
Per-curve fits in `results/timecourse_kinetics.json` (see also the rendered curves
`figures/fig1_timecourses.png`) rank `power_law` first by AICc for formulations 2, 3 and 4 and
`saturation` first for formulation 1. The winner is only comfortably separated for formulation 4
(runner-up ΔAICc = 11.69); the other three are separated by 1.50, 4.68 and 6.79, so for
formulation 1 in particular the top two models are effectively tied.

**Supporting analysis and result evidence.** `analysis/stage1_inventory_timecourse.py` fits a bank
of six candidate models (linear, quadratic, power law, logistic, saturating, lag-then-linear) on
each curve and writes `results/timecourse_kinetics.json`. `analysis/common.py` holds the shared
model bank and the AICc implementation. `results/evidence_table.csv` carries the flat per-unit
summary.

**Robustness.** Leave-one-time-point-out (in `results/loading_crossmodal.json`) is unstable in the
way a five-point fit must be: the AICc-winning model is retained in only 3/5, 4/5, 0/5 and 2/5 of
dropouts for formulations 1–4. Worse, dropping any single point leaves n = 4, at which **AICc is
formally undefined for every model in the bank** (n − k − 1 ≤ 0); those dropouts are therefore
selected by BIC and are labelled `selection_basis = BIC_fallback_aicc_undefined_at_n4` in the
artifact rather than being reported as if AICc still applied. Leave-one-curve-out over the pooled
data gives `power_law` in 3 of 4 dropouts. The independent agent that re-derived the fits from the
raw TSV (`verification/mechanism_fits_results.json`, artifact `F038`) reproduced the exponents and
segment ratios to four decimals and found that the set of mechanisms within 2 AICc units of the
best is *not* a single model: formulation 1 admits both `M3_power` and `M1_lag_rate`
(predicting 18.83 vs 19.06 µmol at 8 h), formulation 2 admits only `M3_power`, formulation 3
admits only a shared empirical shape, and formulation 4 admits only `M3_power`.

**Counterevidence, limitation, alternative explanations.** A one-shape-rescaled null does
surprisingly well, and the honest reading depends on the metric. On raw pooled RSS the scaled-
`formulation_1` shape scores 0.0332 against 0.0343 for per-curve straight lines — apparently
competitive. But that pooled comparison is dominated by the largest curve, so it is **not** a valid
test; scoring within each curve in its own units (`results/shared_shape_check.csv`) shows the
per-curve power law beating the shared shape for formulation 4 and losing for 1–3. Two of those
three losses (formulations 2 and 3) are attributable to the huge n = 5 AICc correction (6 for a
k_total of 2 and 24 for a k_total of 3, counting the residual variance); the formulation 1
comparison is degenerate rather than informative, because there the "shared shape" *is*
formulation 1's own curve rescaled, so its RSS is exactly zero by construction. The normalised curves themselves are what settle it: formulations 3 and 4
agree with each other closely (normalised RMS difference **0.00834**, the closest of all six pairs)
while 1 and 3 differ by **0.06757**. Both conventions are stored in `results/audit_fixes.json` — the
values above exclude the shared (0, 0) origin; including it gives 0.00746 and 0.06044, and the
qualitative conclusion is the same either way. A shared exponential-growth mechanism is refuted
outright (RSS 4.32 for formulation 1, versus 0.0074 for the power law).

**What would falsify or validate this.** The power law is already well separated from its rivals for
formulation 4 and formulation 2 except at t = 0; the genuinely undecided case is formulation 1,
where the top two models differ by only 0.23 µmol (1.2 %) at 8 h. Distinguishing them therefore
requires a measurement at 8 h precise to better than ~1 %, which is a demanding requirement given
that this dataset contains no replicate or error estimate to calibrate against. The corresponding
predictions are 18.83 (power law) and 19.06 (lag-then-constant-rate) µmol; for formulation 2 the
power law predicts 7.71 against 8.26 and 25.10 for the two rejected alternatives, so a single added
point there would confirm an already-decided comparison rather than break a tie. This is a stated
prediction, not a measurement — the data contain no t = 8 h observation.

---

## Finding 2 — The action spectrum is not absorption-limited at the red end, and no transmittance or single-component correction rescues the null

**Object and scope.** The absorption profile (901 points, 350–800 nm at 0.5 nm) and the action
spectrum (6 points at 400, 450, 500, 550, 600, 650 nm) in `data/optical_action_spectra.tsv`. Each
profile is one analysis unit; the 901 absorption points are not 901 independent observations.

**Relationship and direction.** The action response tracks absorption closely from 400 to 550 nm
and then falls far faster. The unit-free ratio statistic
`(action(probe)/action(ref)) / (absorption(probe)/absorption(ref))`, which cancels the arbitrary
absorption units entirely, is **0.228** for 400→650 nm and **0.218** for 450→650 nm. In raw terms,
action drops to **5.08 %** of its 450 nm value over 450→650 nm while absorption drops only to
**23.34 %**.

**Core quantitative result.** A single free scale for the absorption-limited null
`action = s·absorption` fits reasonably overall (R² = 0.9226, RMSE 8.87) but fails precisely where
it matters: it over-predicts the 650 nm action by **4.32×** (18.13 predicted vs 4.2 observed). The
failures are strongly patterned rather than random — the residuals are +1.3, +4.9, +4.3, −1.2,
−15.2, −13.9 across 400→650 nm.

**Supporting analysis and result evidence.** `analysis/stage2_spectra.py` fits five competing
action-versus-absorption models on the six shared wavelengths and writes
`results/spectra_analysis.json`; `figures/fig2_spectra.png` plots the profiles and the relative
efficiency ratio `q = action/absorption`, which is flat near 1 (0.955, 1.000, 0.993, 0.919) through
550 nm and collapses to 0.504 and 0.218 at 600 and 650 nm. The window holdout is the cleanest
evidence: fitting the single scale on the ≤ 500 nm points only (s = 94.23) and predicting forward
gives 55.13 vs 51.6 observed at 550 nm (a 6.8 % error), then 34.30 vs 17.6 at 600 nm (94.9 %) and
**18.94 vs 4.2 at 650 nm (351 %)**.

**Robustness, and a decisive analytical bound.** The most important robustness result is not a
resampling check but an inequality. A natural objection is that the measured "absorption" might
really be optical density, which should be converted via `absorptance = 1 − 10^(−k·OD)` before
comparison. Because `1 − 10^(−x)` is concave, this transform *always* moves the ratio **up**, not down:
`α(k·A₆₅₀)/α(k·A₄₅₀) ≥ A₆₅₀/A₄₅₀` for every k > 0, i.e. the compressed ratio is bounded below by
the raw absorption ratio. So the smallest predicted 450→650 action ratio any transmittance
conversion can produce is exactly the absorption ratio **0.2334**, approached only as k → 0 — and
the observed action ratio **0.0508** lies *below* that floor. This is a uniform statement in k, so
no monotone transmittance conversion can rescue the absorption-limited null. (The inequality
direction was verified numerically over a 1200-point grid spanning k = 10⁻⁶ to 10² and three
absorption pairs; every case satisfies the bound.) The independent test
(`verification/h1h2h3_optics_test.json`, artifact `F036`) swept k over 0.001–100 and confirmed it
from the other side: the plain linear fit is optimal in the k → 0 limit and no k improves the
agreement with the observed action ratio. Both checks are stored in
`results/audit_fixes.json` (`transmittance_bound_grid`: 1200 k values × 5 absorption pairs =
6000 evaluations, zero violations). A
sharp instrument-like cut-off is also refuted: a 1-nm edge spanning the full ~0.85 absorption range
would require |dA/dλ| ≈ 0.85 per nm, whereas the largest smoothed slope anywhere in 500–800 nm is
**0.0094 per nm** — roughly two orders of magnitude smaller — and a logistic fit instead demands a
width of 80 nm.

**Counterevidence and limitation — a boundary on the claim.** The deficit is specifically a
**red-end** phenomenon, and the companion "blue-end plateau" reading is *not* supported. Testing
all 15 probe/reference pairs on the action grid (`results/robustness.json`) shows the statistic is
below 1 for **all nine** pairs whose probe wavelength reaches ≥ 600 nm (maximum 0.548 — the claim
holds there), but it is **not** uniformly above 1 for the short-span pairs: the three pairs with
probe ≤ 500 nm give 1.047, 1.040 and **0.993**, so the last of them already sits below 1. The action
is therefore not systematically flatter than absorption at the blue end; it is essentially
proportional up to ~500–550 nm and then falls off a cliff. A further hard limit is the
action grid itself: six points at one decimal place (a precision of ~1 part in 800 on an 82.6
value), which is a coarse readout relative to the 0.5 nm absorption grid, and all six points are
treated here as one profile rather than six independent units. Adding a second red component with a
*positive* amplitude does not help (RMSE 8.59, still 4.13× high at 650 nm); an unconstrained
negative component does fit better (RMSE 2.32) but at the cost of a negative amplitude that has no
absorptance interpretation.

**What would falsify or validate this.** Measure the action at finer spacing between 550 and 650 nm
(e.g. every 10 nm). A genuine sharp onset should show the action collapsing over a narrow band,
while a second-absorber explanation predicts a smooth, gradual departure. Either pattern would
confirm the mismatch but discriminate between the mechanisms, which these six points cannot.
Separately, reporting the absorption and action in the same physical unit would remove the reliance
on unit-free ratios.

---

## Finding 3 — The two transient traces differ in the *sign* of their long-delay asymptote, and this survives every artifact explanation tested

**Object and scope.** The two ultrafast transient traces in `data/ultrafast_transient_traces.tsv`
(`formulation_id`, `delay_ps`, `normalized_transient_signal`; 2 formulations × 98 delays from
0.000322 to 76.8937 ps). Each trace is one analysis unit; the 98 delays are one serial sweep.

**Relationship and direction.** Both traces rise to ≈ +1.0 (formulation 1 peaking 0.999 at
1.136 ps, formulation 2 peaking 1.001 at 2.242 ps) and then decay — but they settle on opposite
sides of zero. Formulation 1 decays through zero at ≈ 29.6 ps and ends at −0.376; formulation 2
stays positive and ends at +0.140. Over the delay ≥ 30 ps window (17 points each),
formulation 1 has mean **−0.2502** with **0 of 17** points positive, while formulation 2 has mean
**+0.1882** with **16 of 17** positive.

**Core quantitative result.** Moving-block bootstrap (block length 4, seeded) gives 95 % CIs
[−0.3405, −0.1562] for formulation 1 and [0.1233, 0.2477] for formulation 2 — neither includes
zero. The difference is **−0.4384** with 95 % CI **[−0.5404, −0.3188]**, excluding zero. A Fisher
exact test on the positivity counts (0/17 vs 16/17) gives p < 1 × 10⁻⁴.

**Supporting analysis and result evidence.** `analysis/stage3_transients.py` fits a bank of
rise-then-decay models, computes windowed statistics with moving-block bootstrap CIs (serially
correlated data), and runs the cross-trace contrast; results in
`results/transient_analysis.json`, curves in `figures/fig3_transients.png`. The winning model
family is an explicit excitation rise followed by decay (R² = 0.73–0.82 versus 0.44 for a pure
monoexponential), which matters because a model without a rise term cannot represent these traces
at all.

**Robustness — three artifact explanations tested and refuted.** This is the finding most exposed
to an artifact reading, so it was stress-tested hardest, by an independent agent that recomputed
everything from the raw TSV without seeing the pipeline's numbers
(`verification/transient_artifact_test.json`, artifact `F035`):

- *Constant-offset artifact* (the contrast is just a level difference): **refuted**. Forcing both
  traces to share one decay shape plus a free offset costs ΔAICc = +63.5 on the full trace and
  +7.9 on the late window alone, with F p = 6 × 10⁻¹⁴; and the fitted offset difference (+0.198)
  accounts for only 45 % of the observed level gap (+0.438).
- *Slow linear drift* (the late sign is instrumental drift): **refuted**. Both late windows do
  drift (p = 7 × 10⁻⁷ and p = 0.007), but detrending never flips formulation 2: its detrended late
  mean stays **+0.284** while formulation 1's stays **−0.088**.
- *The decay itself* (the asymptote is a genuine kinetic property): **supported**. Free-asymptote
  fits over all 98 points give c₁ = **−0.371** (95 % CI [−0.546, −0.196]) and c₂ = **+0.176**
  (95 % CI [0.090, 0.261]) — both CIs exclude zero, with signs matching the observed late means
  (−0.250 and +0.188), and constraining c = 0 is rejected (ΔAICc = +28.4 and +4.2).

Two further checks passed: excluding the pre-rise coherent-artifact region (t < 0.5 ps) leaves the
late-window sign unchanged, and the sign is stable across every late-window threshold from 15 to
50 ps. A shared-shape-time-rescaled null (y₂(t) = 1.258·y₁(t/2.021)) reaches only R² = 0.5916 over
the overlapping range, so the two traces are not one kinetic shape in disguise.

**Counterevidence, limitation, alternative explanations.** The sign contrast is established; its
*mechanism* is not. "Normalized transient signal" invites the reading that a negative asymptote is
unphysical for a bleach or excited-state absorption, and the data cannot rule out that the two
traces were normalised differently or that the negative offset reflects an uncorrected baseline in
one trace. Two limits should be stated plainly: only **two** traces exist, so this is a contrast
between two units and not a population statement; and a single late-window trend that happens to
differ between the two would also produce a sign difference, which is why the drift test above was
run at all. The delay-window holdout (in `results/transient_analysis.json`) is only partly
consistent — fitting rise-then-decay on t ≤ 5 ps and extrapolating gets the late sign wrong for
formulation 1 (predicts −0.149 vs +0.010 observed) while agreeing for formulation 2 — so the early
trace alone does not fully determine the late behaviour.

**What would falsify or validate this.** Repeat the measurement on the same two formulations with an
independently determined time-zero and a reported baseline, and record the raw (unnormalised)
signal. If the sign contrast persists in unnormalised data with a characterised baseline, the
kinetic interpretation stands; if it disappears, the contrast is a normalisation artifact. A third
and fourth formulation would convert a two-unit contrast into a testable association.

---

## Finding 4 — The cofactor-loading scan has an interior peak, but with four points its shape and optimum location are not identifiable

**Object and scope.** The four loading conditions in `data/cofactor_loading_response.tsv`
(0.2, 1.0, 3.0, 8.0 % loading; rates 1081, 1217, 470, 268 µmol/h). One loading condition = one
analysis unit, n = 4, and no replicate variance is provided.

**Relationship and direction.** The response is non-monotone and has a strict interior peak: it
rises from 0.2 % to a maximum at 1.0 %, then falls by 61.4 % to 3.0 % and further to 8.0 %, where
the rate is **24.8 %** of the 0.2 % value. The ordering is exactly "rising before the peak, falling
after it".

**Core quantitative result.** With n = 4 the inferential ceiling is low and must be stated exactly.
The Spearman rho between loading and rate is −0.800 with an asymptotic p = 0.200, and the **exact**
two-sided permutation p over all 24 orderings is **0.333** (8 of 24 orderings are at least as
extreme). A monotone-decline description therefore cannot be established at any conventional
significance from these four points. Leave-one-condition-out (in `results/loading_crossmodal.json`)
shows why: dropping 0.2 % or 1.0 % leaves a strictly decreasing triple (rho = −1.0), but dropping
3.0 % or 8.0 % leaves a non-monotone triple with rho = −0.5 and p = 0.667. **The monotone
description is carried by the two *high*-loading points jointly.** The strictly decreasing triple
appears only when a **low**-loading point (0.2 % or 1.0 %) is removed, leaving both high points plus
one low point; removing either high point instead leaves the non-monotone rise-then-fall pattern.
The full mapping is recorded in `results/audit_fixes.json` (`loading_leaveout_restatement`).

**Supporting analysis and result evidence.** `analysis/stage4_loading_crossmodal.py` fits five
response models and writes `results/loading_crossmodal.json`; `figures/fig4_loading.png` shows the
data against the fitted candidates. The substrate-inhibition form
`rate = V·L/(K + L + L²/Ki)` fits well (R² = 0.9901, RMSE 39.8 µmol/h, V = 17486, K = 2.599,
Ki = 0.0923), with an analytic optimum at `sqrt(K·Ki)` = **0.490 %** loading; a residual bootstrap
(3 660 successful resamples of 5 000, seeded) puts the optimum's 95 % CI at [0.453, 0.525] %, and
100 % of resamples fall below 3 %. That interval is narrow but should not be over-read: it is
conditional on this three-parameter functional form, which is itself unverifiable at n = 4.

**Robustness, and a correction that changes the conclusion.** My first pass ranked these models by
AICc and reported the substrate-inhibition model as the winner. That was wrong, and an independent
identifiability audit (`verification/cofactor_shape_identifiability.json`, artifact `F037`, with a
second independent run in `F039`) caught it: **AICc requires n − k − 1 > 0, so at n = 4 every
three-parameter model has a formally undefined AICc.** The original code silently guarded the
division and manufactured a finite number. The pipeline has been corrected
(`analysis/common.py` now returns `null` and reports `aicc_undefined_models`; `stage4` ranks by BIC
where AICc is undefined), and the corrected result is that in the `stage4` artifact only the
1-parameter constant retains a defined AICc, so nothing informative can be ranked on that criterion.
(The `stage4` convention counts the residual variance in the parameter total, which is why the
2-parameter models are undefined there while the independent verifiers, counting only fitted
parameters, could still rank them; the artifact's own `aicc_undefined_models` field is the
authoritative record.) On BIC — which is defined at n = 4 — the substrate-inhibition model is preferred;
with the uncorrected AIC it is also the sole model within 2 units of the best. The limit is that
several other three-parameter descriptions — a flat-then-decline two-regime model, a log-quadratic
peak, an exponential decay with offset — cannot be *ranked* against it with a small-sample-corrected
criterion at n = 4, because that criterion does not exist here; they can only be compared on BIC,
where substrate inhibition leads, or on raw RSS.

**Counterevidence and the key limitation.** "Peak at 1 %" and "plateau up to 1 %, then decline" fit
the data equally well, and the two low-loading points differ by only 12.6 % — within the scatter
that a single unreplicated measurement can produce. A log-quadratic peak fitted freely lands at
0.346 % with a 1.05×-of-minimum range of [0.237, 0.442] %, spanning the gap between the two
candidate optima. The respective evidence is not perfectly balanced, and that should be stated
rather than flattened: on both BIC and the uncorrected AIC the substrate-inhibition model does
lead, and among fitted families the data do prefer it. What cannot be done at n = 4 is *rank* every
three-parameter description against every other with a defined small-sample-corrected criterion —
the AICc degeneracy and the form-conditional nature of the 0.490 % optimum are the real limits, not
a claim that all shapes fit identically. The axis-scale alternative is a genuine non-identifiability rather than a
testable hypothesis: isotonic (order-based) fits and Spearman rho are invariant under any monotone
rescaling of the loading axis, so the data **cannot** distinguish a linear from a log-scaled
loading axis — rho = −0.800 and the exact p = 0.333 are identical either way. Finally, the absolute
rates (hundreds to ~1200 µmol/h) sit about 10²–10³× above the time-course rates (0.04–2.37 µmol/h),
so these four conditions cannot be placed on the same rate scale as any named formulation without
an external assumption.

**What would falsify or validate this.** A scan with at least 8–10 loadings bracketing 0.2–3 %
with replicates would (a) restore a defined AICc for three-parameter models, (b) locate the optimum
with an honest interval instead of a form-conditional one, and (c) settle whether the low-loading
pair is a plateau or a genuine peak. Replicating the 0.2 % and 1.0 % conditions alone would say
whether their 12.6 % difference is signal or scatter.

---

## Finding 5 — The cross-modal question posed by the objective is not answerable from this dataset, and the one contrast that *can* be formed runs opposite to the yield ordering

**Object and scope.** The relation between product evolution and the other three modalities, across
all four files.

**Relationship, direction, and the boundary.** The dataset does not support a cross-modal
association claim, and this is a property of the data rather than of the analysis. Only
**two** formulations (`formulation_1`, `formulation_2`) appear in both the time-course and
transient files. The optical spectra and the loading scan carry **no formulation identifier** — they
are modality-level profiles, so no join to a named formulation is possible without an external
assumption, which the task boundary forbids. With two shared units a rank correlation is undefined;
only an ordinal contrast can be stated.

**Core quantitative result — and it is a reversal.** On the two shared formulations, the yield
ordering across formulations runs *opposite* to the ordering of the long-delay transient signal:

| formulation | product at 4 h (µmol) | mean rate (µmol/h) | transient mean, delay ≥ 30 ps | transient sign |
|---|---|---|---|---|
| `formulation_1` | 9.48 | 2.370 | −0.2502 | negative |
| `formulation_2` | 4.13 | 1.033 | +0.1882 | positive |

`formulation_1` produces **2.30×** more product yet carries the *negative* late transient;
`formulation_2` produces less and carries the *positive* one. The orderings disagree
(`orders_agree = false`). If the late-window signal is read as surviving charge separation, this is
the opposite of the naive "more long-lived signal → more product" expectation.

**Supporting analysis and result evidence.** `analysis/stage4_loading_crossmodal.py` assembles the
shared-unit table and the coverage boundary, writing `results/loading_crossmodal.json`;
`figures/fig5_crossmodal.png` shows the two orderings side by side; the consolidated headline
numbers are in `results/registration_summary.json` under `finding_5_crossmodal_boundary`. An
independent agent re-derived every number in this finding — including the shared-ID intersection and
the absence of a formulation column in the two unlabelled files — directly from the raw TSVs
(`analysis/workflow_audit_findings.js`, artifact `F042`).

**Robustness and counterevidence.** The reversal is real but should not be over-interpreted, and the
reasons are structural:
- With n = 2 there is no statistical test, only an ordinal statement. The two units are also the two
  largest-yielding formulations, so no trend across the 54.8× yield range can be assessed.
- The two modalities' kinetics are differently shaped: formulation 1's time course prefers
  `saturation` while formulation 2's prefers `power_law` (Finding 1), so the yield comparison mixes
  a shape difference with a magnitude difference.
- The absolute scales are unrelated: the loading scan's rates (268–1217 µmol/h) exceed the
  time-course rates (0.043–2.37 µmol/h) by at least two orders of magnitude (pairwise ratios
  113× to 28 139×), so the loading modality cannot be numerically reconciled with the time courses.
- The three orderings this dataset can form — product yield (1 > 2 > 3 > 4, four units), late
  transient signal (2 > 1, two units, the reverse), and loading response (1.0 > 0.2 > 3.0 > 8.0,
  four conditions) — cannot be brought into a common comparison. The yield and transient orderings
  are over different unit sets, and the loading ordering carries no formulation label at all
  (coverage boundary above), so no data-justified mapping among them exists.

**What would falsify or validate this.** Attach a formulation identifier to the optical-spectra and
loading measurements and run enough formulations (n ≥ 6–8) that a rank association between yield
and late-window transient sign has a defined null distribution. If the reversal persists, it is a
genuine tradeoff; if it vanishes, it was a two-point coincidence. Either outcome would also let the
"no stable cross-modal relation" explanation be compared against the "formulation-specific"
explanation rather than merely asserted.

---

## Synthesis: which of the objective's candidate explanations the data support

The objective asked to compare monotone, peaked, formulation-specific, cross-modal, and
no-stable-relation explanations. Taking the five findings together:

- **Formulation-specific** — supported for product evolution (Finding 1): the two opposed
  kinetic-shape groups are the most robust structure in the time courses.
- **Peaked** — supported as an observation but bounded for the loading scan (Finding 4): the
  interior peak is real; its location and functional form are not identifiable at n = 4.
- **Monotone** — rejected where it was tested: the loading response is non-monotone (strict interior
  peak), and the action-versus-absorption relation is not the monotone absorption-limited null
  (Finding 2).
- **Cross-modal** — not supported, and the one available contrast reverses (Finding 5). This is a
  coverage limitation of the dataset, not a demonstrated absence of a relationship.
- **No-stable-relation** — the correct description for the cross-modal question only. It is *not*
  the right description for the time courses, the action spectrum, or the transient contrast, each
  of which shows stable, reproducible structure.

## Limitations that apply across all findings

1. **Small numbers of independent units.** 4 curves, 2 spectra, 2 traces, 4 loading conditions.
   Several tests here are at or beyond the edge of AICc's validity, and where the correction is
   undefined this report says so rather than reporting a number.
2. **No replicates anywhere.** No file contains an error bar or a repeat measurement, so all
   uncertainty quoted above is model- or resampling-based, not a measurement of reproducibility.
3. **No formulation-level join.** Findings 1 and 3 are linked to named formulations; Findings 2
   and 4 are not, and none of the four modalities can be joined to each other except through the
   two shared formulation IDs.
4. **Identity is withheld.** No catalyst, cofactor, or mechanism is identified anywhere in the data
   or in this report; the mechanism statements above are explicitly labelled as interpretations
   of kinetic shape, not assignments.
5. **Precision differs by modality.** The absorption grid is 0.5 nm at 3 decimals; the action grid
   is 6 points at 1 decimal. Sub-nm structure in the red-end fall-off is therefore not resolvable.
6. **Holdout coverage is thinner than the objective implies.** The objective asks for sensitivity
   to complete-curve and spectral-window holdouts. Width-appropriate holdouts were executed for
   the **time courses** (leave-one-curve-out and leave-one-time-point-out) and the **action
   spectrum** (leave-one-point-out plus the forward ≤500 nm window forecast). For the **transients**
   the holdout is a delay-window (fit early, predict late), not a spectral window. No window or
   complete-profile holdout exists for the 901-point **absorption profile** or for the **loading
   scan** (which has leave-one-condition-out only). This is a real gap in coverage across the four
   modalities, not merely a reporting omission; no conclusion above depends on the missing ones.
7. **Independence of the verification battery is partial.** Two of the "independent" agent checks
   reproduce pipeline values rather than deriving them from scratch: the transient free-asymptote
   c₁ (−0.3710212) matches the pipeline's `sum_two_decays` offset to seven digits, and the F038
   kinetic fits agree with the pipeline to 7–8 decimals. Both agents did re-derive from the raw TSV,
   but the agreement is closer than independent optimiser paths would normally produce, so these
   should be read as strong consistency checks rather than wholly independent replications. The
   findings that rest on them (2, 3) are unaffected: their load-bearing numbers were separately
   reproduced from the raw data by the audit agents.

## Reproducibility

Full instructions are in [`reproduce.md`](reproduce.md). All random operations are seeded with
`20220101` and the seed is recorded inside each result artifact. The independent verification and
alternative-explanation battery was executed as a native multi-agent workflow
(`analysis/workflow_verify_alternatives.js`, `analysis/workflow_round2.js`) whose agents recomputed
the key numbers from the raw TSV without access to `results/` or this report. Those agents' scripts
and JSON outputs are preserved under `verification/`. A native artifact registry mapping identifiers
`F001`–`F039` to real files is at `file_registry.json`.

**Reported failures, retained rather than hidden.** The `scipy`/`matplotlib` installation failed
twice on timeout before succeeding. The first transient decay model bank was mis-specified (no rise
term) and its fits were discarded after inspection; the discarded results themselves were not
archived, so this is evidenced only by the surviving `logs/stage3_transients.log`. A first-stage
AICc implementation silently guarded a division by zero at small n; this was caught by an
independent identifiability audit, and the correction changed the reported model ranking for the
loading scan (Finding 4). The first multi-agent workflow could not use structured output in this
environment, which caused repeated agent restarts; its two remaining agents were re-run without the
schema option, and the interrupted agents' completed artifacts were retained and used.

**Post-hoc audit trail.** After the report was first assembled, all five registered findings were
audited claim-by-claim against their cited artifacts by independent agents (a second native
workflow, `analysis/workflow_audit_findings.js`, plus a whole-report completeness critic). Every
load-bearing number checked out, but the audit found and this report has since corrected: one
inverted inequality (Finding 2), one factor-10 slip in a slope figure (Finding 2), one inverted
sentence about which leave-one-out subset is monotone (Finding 4), one overstatement about
model distinguishability (Finding 4), one unsupported provenance claim (Finding 5), one mismatched
normalisation convention (Finding 1), and two numeric claims that were true but had no stored
artifact — both of which now do (`results/audit_fixes.json`, `analysis/stage7_audit_fixes.py`).
The auditor's scratch scripts are preserved in `audit/`. On seeding: only
`results/transient_analysis.json` carries an explicit `seed` field (14 occurrences, from the
moving-block bootstraps). The seeded resampling in `results/robustness.json` (the loading bootstrap),
`results/loading_crossmodal.json` (loading residual bootstrap) and `results/spectra_analysis.json`
(permutation tests) is reproducible from the source, where the seed `20220101` is set literally, but
the value is not embedded in those JSON files.
# Temperature-Dependent Response Boundaries in Interface Catalysis

**Task:** `Energy_03_interface_response_boundaries` — TruthInsightBench public scientific discovery run
**Analysis unit:** one complete condition curve (adjacent settings are ordered scan points of the same curve, never independent replicates)
**Direct evidence:** `data/anonymous_condition_response_curves.tsv` (8 curves × 11 settings) and `data/independent_condition_check_curves.tsv` (4 curves × 11 settings), SHA-256 verified against `data_manifest.json` before use
**Time boundary:** `T0 = 2022-01-01`; external sources used only for background/methods/alternative explanations, all strictly pre-`T0` (see `output/native/provenance/external_sources.md`)
**Registered findings:** 5

---

## 1. Summary of registered findings

| # | Finding | Direction | Core quantitative result | Robustness |
|---|---|---|---|---|
| **Finding 1** | Every condition curve is concave in log-response: the response rise decelerates everywhere in the measured window | Supported | 12/12 curves, quadratic coefficient `c < 0` at T ≥ 140 K; largest one-sided p = 0.0151 | Survives onset-point exclusion, alternative curvature forms, and a contiguous-setting holdout (12/12 curves) |
| **Finding 2** | The apparent thermal exponent is **not** a per-condition constant: a 1.68× spread exists across conditions, and it is partly, not wholly, attributable to where each curve sits on a shared saturating shape | Supported with a stated boundary | Wide-window slope 0.01796–0.03015 K⁻¹ across 8 main curves; Spearman vs fitted T50 = 0.833 | Spread persists under response-window restriction (1.35×); ordering is unstable under window change (ρ = −0.39) |
| **Finding 3** | Condition ordering is not a held-out invariant: it is stable through the mid-range but reverses at the endpoint settings | Supported | 12/28 pairs reverse over 120→320 K, but only 2/28 over 140→300 K; Spearman(rank@120, rank@320) = 0.310 (p = 0.51) vs Spearman(rank@140, rank@300) = 0.976 (p = 0.0043) | Confirmed by adjacent-setting rank stability and by a reversal census as a function of the lower bound |
| **Finding 4** | The saturation boundary lies at or above the design maximum, so the response ceiling is not measured for every condition | Supported (boundary) | No response ever exceeds 93.01; 1/8 curves has fitted T50 > 320 K; per-curve ceiling `ymax` is only weakly identified | Ceiling-at-100 refit changes T50 by ≤ 10 K for 7/8 curves; forward holdout shows the saturating form still beats an unsaturated exponential |
| **Finding 5** | The two panels are not statistically independent: 2 of 4 "check" curves reproduce main curves exactly, and the block labels carry no detectable parameter signal | Null / integrity result | 2 exact full-curve duplicates (0.0 absolute difference across all 11 settings); block-label tests give exact p = 0.74 (T50), 0.57 (w), 0.086 (ceiling) | Duplication verified digit-by-digit; block tests use complete 4-of-8 enumeration (70 assignments) |

---

## Finding 1 — Log-response curvature is universal across the measured window

**Object and scope.** All 12 usable condition curves (8 main, 4 check), settings 120–320 K, response 0.0–93.01.

**Relationship, direction, boundary.** The response does not follow a constant-exponent power law anywhere inside the measured window. Fitted on the log scale, the rise is decelerating: the second derivative of `log(response)` with respect to the setting is negative across the whole window for every curve. Precisely: **the fitted quadratic coefficient `c` is negative for every curve**, which makes the fitted log-response concave throughout the fitted range. This holds for the fit, not for every single raw data point — the raw discrete second differences are negative at a majority of interior knots for 11 of 12 curves and at 4 of 8 for the remaining one (see the model-free check below). This is a boundary statement about functional form, not a mechanism claim.

**Core quantitative result.** Fitting `log(y) = a + b·T + c·T²` per curve on T ≥ 140 K (analysis unit = one complete curve):

- **12 of 12 curves have `c < 0`** (range −1.467 × 10⁻⁴ to −2.292 × 10⁻⁵; `R²` = 0.9965–0.9999).
- Largest one-sided p-value for `c < 0` = **0.0151**; **9 of 12** are below 1 × 10⁻⁴.
- Model-free confirmation: on T ≥ 140 K the discrete second difference of `log(y)` is negative at **4–8 of the 8** interior knots. It is a strict majority for 11 of 12 curves; the single exception is `check condition_block1_1` at 4 of 8 (knots at T = 160, 180, 220, 300 are positive). All eight knots are negative for 4 curves: `condition_block2_2`, `condition_block2_3`, `condition_block2_4`, `check condition_block1_5`. That the raw second differences are not uniformly negative everywhere is expected for unfitted data and is why the model-free check is reported as a majority statistic rather than as an every-point claim.

**Supporting analysis and result evidence.**
- Analysis source: `output/native/analysis/run_stage_j.py`, `output/native/analysis/run_verification.py`
- Result evidence: `output/native/results/J_curvature_robustness.json`, `output/native/results/results.json` (key `D1_curvature`), `output/native/results/fig3_data.csv`
- Figure: `output/native/figures/fig3_log_response_concavity.png`

**Robustness result.** Three independent checks agree:
1. *Onset exclusion.* Dropping the 120 K and 140 K settings (the two where five of eight curves are still below 0.6) leaves `c < 0` for 12/12 curves, with the largest p-value falling to 0.0001.
2. *Alternative functional forms.* A log-quadratic in `log T` is also concave for the curves with the strongest effect. A reciprocal (hyperbola) fit is worse, but this comparison carries a caveat that must be stated: the hyperbola was fitted as `1/y` linear in `1/T`, so its `R²` (0.77–0.91) is measured in `1/y` space and is **not directly comparable** to the `0.996–0.9999` obtained in `log y` space. The hyperbola's lower `R²` is therefore consistent with, but not proof of, a worse fit; the decisive evidence for concavity is the sign of `c` and the holdout result, not this comparison.
3. *Contiguous-setting holdout.* Trained on T ≤ 280 K and tested on the two held-out settings T = 300, 320 K, the concave quadratic beats a straight line in log-space for **12/12 curves** (main: sign test p = 0.0039; check: 4/4).

**Counterevidence, limitation, and alternative explanations.** The 120 K setting is genuinely uninformative for 5 of 8 curves (responses 0.0–0.559), so any fit that includes it is partly fitting noise; this is why the primary specification starts at T ≥ 140 K and why the onset-exclusion check is reported. The word "curvature" here is a statement about the fitted functional form, not proof of a specific surface mechanism: an S-shaped light-off with coverage change (S2, S3) and a distributed-surface apparent-activation-energy model (S4, S5) both predict decelerating log-response and are not separable from these operating curves alone. The check panel contributes only 2 genuinely independent curves (see Finding 5), so the effective replication is narrower than the nominal 12 curves.

**Executable falsification.** Measure a condition at settings beyond 320 K and test whether the log-response continues to decelerate or returns to a constant exponent. Concretely: fit `c` on T ≤ 320 K and test the sign of `c` on 330–400 K; a positive or statistically indistinguishable-from-zero `c` in that extension refutes the claim that the deceleration holds throughout the operating window rather than only within it.

---

## Finding 2 — The apparent thermal exponent is not a per-condition constant

**Object and scope.** The 8 main-panel condition curves; the conventional Arrhenius/exponential description `log(y) = a + b·T` fitted over T ≥ 140 K.

**Relationship, direction, boundary.** The apparent thermal exponent `b` differs across conditions by a factor of **1.68** (0.01796 K⁻¹ to 0.03015 K⁻¹). Part of that spread is a **geometric artifact of comparing curves that sit at different positions on a shared saturating response**, and part survives window restriction. The boundary is explicit: the data support a spread in the *apparent* exponent and support T50 as a partial explanation; they do **not** establish that each condition has a distinct *intrinsic* activation energy.

**Core quantitative result.**

- Wide-window slope across 8 main curves (log y ~ 1 + T, per curve, T ≥ 140 K): 0.01796–0.03015 K⁻¹, mean **0.02325**, SD **0.00380**, max/min = **1.679**. (These are the values in `results/D2_exponent_spread`; an earlier draft of this report misstated the mean and SD, see §5.)
- Expressed in Arrhenius form (assuming the setting is in kelvin, R = 8.314 J mol⁻¹ K⁻¹), apparent activation energy spans **7.27–11.39 kJ/mol**.
- Relationship to position: Spearman(apparent slope, fitted T50) = **0.833** (n = 8, p = 0.044); Spearman(apparent slope, fitted width `w`) = 0.643 (p = 0.149).
- Multiple regression `slope ~ T50 + w`: `R²` = 0.794, T50 coefficient `+1.39 × 10⁻⁴` per K (p = 0.013); adding `w` is not significant (partial F = 3.87, p = 0.106).

**Supporting analysis and result evidence.**
- Analysis source: `output/native/analysis/run_stage_c.py` (via `run_analysis.py`), `output/native/analysis/run_stage_h.py`, `output/native/analysis/run_verification.py`
- Result evidence: `output/native/results/C_growth_and_exponents.json`, `output/native/results/H_shape_test.json`, `output/native/results/results.json` (key `D2_relationship`), `output/native/results/curve_parameters.csv`

**Robustness result.** The spread is not purely a window artifact, and the *ordering* is not robust:
- Restricting to the intrinsic low-conversion regime (response between 1 and 20, which leaves 7 of 8 main curves with enough points) still leaves a **1.35×** spread in main-panel slopes (0.02692–0.03620 K⁻¹).
- However, the *ordering* of conditions by apparent slope is **not** preserved when the window changes: Spearman(wide-window slope, narrow-window slope) over the 7 curves with both available = **−0.393** (p = 0.45). The *magnitude* of the spread replicates; the *ranking* does not.
- A synthetic reconstruction closes the loop: curves generated with **one identical intrinsic exponent** and only T50 varying, then pushed through the same wide-window estimator, reproduce **97 %** of the observed wide-window slope range and correlate with the observed slopes at **ρ = 0.93** (p = 0.014, main panel). This is direct evidence that a substantial share of the observed exponent spread is produced by saturation geometry alone.

**Counterevidence, limitation, and alternative explanations.** This finding is deliberately two-sided, because the evidence is:
- **Supporting a genuine effect:** 1.35× spread survives window restriction, and fitted T50 explains a significant share of the wide-window slope (p = 0.013).
- **Supporting an artifact:** the slope *ranking* does not survive the window change (ρ = −0.39), and synthetic same-exponent curves reproduce most of the range.
The honest reading is that the data cannot cleanly separate "conditions have different intrinsic exponents" from "conditions sit at different points on a common saturating curve". A structure-sensitive apparent activation energy that varies with surface heterogeneity (S4, S5) is the standard external explanation for a genuine spread and cannot be excluded here. Because 2 of the 4 check curves are duplicates (Finding 5), the independent replication for this finding rests on 2 curves only.

**Executable falsification.** Fix one condition and measure its response at a set of settings that spans the same response window as another condition, then compare narrow-window slopes at *matched conversion* rather than matched temperature. If the matched-conversion exponents differ significantly, the spread is intrinsic; if they coincide, it is geometric. Independently: extend the setting range past 320 K for a low-T50 condition to test whether its apparent exponent converges to that of a high-T50 condition once both are fully saturated.

---

## Finding 3 — The ordering boundary: ranking is stable mid-range and unstable at the endpoints

**Object and scope.** The 8 main-panel curves (and the 4 check curves, reported separately), all 11 settings.

**Relationship, direction, boundary.** Condition ranking is **not** a held-out invariant across the full measured window. It *is* stable through the mid-range. The boundary is locatable: the breakdown is concentrated at the two endpoint settings, driven by the 120 K setting where the response is near the detection floor.

**Core quantitative result.**

- Over the full 120→320 K window: **12 of 28** curve pairs reverse their relative order at least once; Spearman(rank@120 K, rank@320 K) = **0.310** (p = 0.51).
- Over the restricted 140→300 K window: only **2 of 28** pairs reverse; Spearman(rank@140 K, rank@300 K) = **0.976** (p = 0.0043).
- The 120→140 K step is the single unstable transition: Spearman = 0.595 (p = 0.186). Every subsequent step is ≥ 0.952, and five of the nine later steps are exactly 1.000.
- Reversal census as a function of the lower bound: T ≥ 120 → 12/28 pairs; T ≥ 140 → 4/28; T ≥ 220 → 3/28; T ≥ 320 → 0/28.
- At 120 K, 5 of 8 curves have responses below 0.6 while the maximum is 1.097 — near-floor values with no separation.
- The check panel behaves consistently: 1/6 pairs reverse over the full window, 0/6 over 140–300 K.

**Supporting analysis and result evidence.**
- Analysis source: `output/native/analysis/run_stage_b.py` (via `run_analysis.py`), `output/native/analysis/run_stage_g.py`, `output/native/analysis/run_verification.py`
- Result evidence: `output/native/results/B_ordering_crossings.json`, `output/native/results/G3_separation_boundary.json`, `output/native/results/results.json` (key `D3_ordering`), `output/native/results/response_matrix_with_ranks.csv`
- Figure: `output/native/figures/fig2_rank_ordering.png`

**Robustness result.** The boundary is confirmed three ways: (i) adjacent-setting rank stability, (ii) the pairwise reversal census as a function of the lower bound, and (iii) an independent tabulation on the check panel that shows the same pattern with a smaller sample. The reversal count at the endpoint is *not* statistically significant on its own (exact sign test for "all pairs flip" gives p = 0.83), and this is reported as such: the finding is that the ranking is **not demonstrably invariant**, not that the data prove an active reordering mechanism.

**Counterevidence, limitation, and alternative explanations.** One curve (`condition_block1_3`) is non-monotone in the raw response across settings, which contributes reversals independently of the near-floor problem; it participates in **5 of the 12** reversing pairs (one pair, `condition_block1_3` vs `condition_block2_4`, reversing three times). The near-floor explanation and the genuine-reordering explanation are not separable with these data, and the mid-range stability means a reader could equally argue the ranking is effectively stable for practical purposes — that is precisely why the finding is stated as a *boundary* (stable 140–300 K, unsupported outside it) rather than as a reordering claim. Two further qualifications: the endpoint step 300→320 K is also non-unit (ρ = 0.952, p = 0.0089), so the 120 K endpoint is the *largest* instability rather than the only one; and the "not demonstrably invariant" framing is the correct one, because the reversal count is not itself significant (see robustness). Separately, the physical setting is an ordered scan, so adjacent settings are not independent systems; no random curve-point splitting was used anywhere in this analysis.

**Executable falsification.** Re-measure the same conditions with replication at 120 K. If the 120 K ordering proves reproducible across repeated runs, the endpoint instability is a real feature and the boundary should be moved; if it does not reproduce, the endpoint instability is measurement-floor noise. Either outcome sharpens the stated boundary and neither is assumed here.

---

## Finding 4 — The saturation boundary lies at or beyond the design maximum

**Object and scope.** All 12 curves; the per-curve response ceiling and midpoint of the saturating description `y(T) = ymax / (1 + exp(−(T − T50)/w))`.

**Relationship, direction, boundary.** The curves are unambiguously saturating (Finding 1), but for most conditions the **saturation plateau itself is not measured**: the design stops at or before the plateau is reached. Consequently the per-curve ceiling is only weakly identified, and one condition's fitted midpoint lies beyond the design maximum.

**Core quantitative result.**

- The maximum response anywhere in either panel is **93.01**; the design never reaches a plateau, so no measured ceiling exists for any curve.
- With a free ceiling, over the **8 main curves**, `ymax` estimates span 51.99–130.55 (mean 81.83, SD 24.93) and the ceiling's own standard error averages **9.86** — i.e. the ceiling is the least reliable parameter. (The 4 check curves give mean 99.65, SD 29.29, mean SE 15.67.)
- **1 of 8** main curves (`condition_block2_1`) has fitted T50 = 327.2 K, **above** the maximum setting of 320 K. Under the ceiling-fixed-100 refit a different curve (`condition_block1_4`, T50 = 321.8 K) crosses; the count stays 1 of 8 either way.
- Fixing the ceiling at 100 shifts fitted T50 by **up to 29.5 K, with only 3 of 8 main curves within 10 K** (mean |ΔT50| = 15.84 K), at a cost of at most ΔR² = 0.0061 (mean 0.0023). This is a **larger** shift than a ceiling nuisance parameter would ideally cause, and it is the main reason the ceiling is reported as unidentified rather than merely uncertain — but it does **not** weaken the finding: the saturating *form* still predicts held-out settings far better than the alternatives below, and it is the absolute ceiling value, not the shape, that moves.
- The saturating logistic is the preferred form among the alternatives tested: comparing logistic, Gompertz, and Hill by BIC on `log y`, logistic wins **7/8** main curves (check: 3/4) with mean `R²` 0.994 vs 0.976 (Gompertz) and 0.889 (Hill). **Boundary on this claim:** this ranking is conditional on the bounded parameter grid used (`ymax` within [max y, 3·max y], `w` within [5, 90]); under unconstrained multi-start fitting Hill beats logistic on 3 of 8 main curves. The honest statement is that logistic is preferred *within the searched region*, not that logistic is universally best.

**Supporting analysis and result evidence.**
- Analysis source: `output/native/analysis/run_stage_e.py`, `output/native/analysis/run_stage_g.py` (`g5_alt_sigmoid`), `output/native/analysis/run_verification.py`
- Result evidence: `output/native/results/E_sigmoid_model.json`, `output/native/results/F5_saturation_boundary.json`, `output/native/results/G5_alternative_sigmoids.json`, `output/native/results/G4_outlier_sensitivity.json`

**Robustness result.** The boundary claim holds under: (i) ceiling-free vs ceiling-fixed refits, which shift T50 by up to 29.5 K but leave the count of curves with T50 beyond the design maximum at 1 of 8, (ii) dropping the 120 K onset setting (mean |ΔT50| = 7.06 K, max 14.70 K; 5 of 8 curves within 10 K), and (iii) a 2σ log-residual trim. Critically, the *forward* predictive claim is robust even though the ceiling is not: fitting on T ≤ 280 K and extrapolating to T = 300, 320 K, the saturating form beats an unsaturated exponential in **12/12 curves** (main: 8/8, sign test p = 0.0039; check: 4/4) and beats a last-value-persistence baseline in **10/12** (7/8 main, 3/4 check). The advantage is large in log space: under the *forward* protocol mean holdout RMSE is **0.150 vs 0.953** (main; logistic vs exponential), and under the separate whole-curve parameter-transfer protocol it is **0.165 vs 1.063**.

**Counterevidence, limitation, and alternative explanations.** The weak identifiability of `ymax` is a genuine limitation and is reported rather than hidden: with a free ceiling the estimate for `condition_block1_1` is 51.99 ± 11.26, which is consistent with a wide range. The claim registered here is therefore **not** "the ceiling is 100"; it is "the ceiling is not determined by this design". The forward-holdout success of the saturating form is evidence that the *shape* is right without requiring the ceiling to be known. An alternative reading — that the apparent plateau is a transport or residence-time limitation rather than an intrinsic kinetic saturation — cannot be distinguished from these operating curves, consistent with the data guide's warning that the curves support operating relationships but not a unique microscopic interface mechanism.

**Executable falsification.** Extend the scan for the four highest-T50 curves to 360–420 K. If the response continues rising at the pre-320 K rate with no inflection, the saturating description is refuted and an unsaturated exponential is preferred; if it flattens near a common value, the ceiling is measured and the current weak identifiability is resolved.

---

## Finding 5 — The check panel is partly a duplicate of the main panel, and the block labels carry no signal

**Object and scope.** The relationship between the two supplied panels, and the explanatory value of the `block1`/`block2` label embedded in the condition identifiers.

**Relationship, direction, boundary.** This is a **null and integrity result**, reported because it materially changes how much independent replication the other four findings have. The two files are not independent replicates of one another.

**Core quantitative result.**

- **2 of the 4 check curves are exact duplicates of main curves**: `condition_block1_4` ≡ `condition_block2_1` and `condition_block1_5` ≡ `condition_block2_2`, **identical to all published digits at all 11 settings** (maximum absolute difference 0.0).
- The artifact's coincidence count is **24 exact within-setting value coincidences** over all 12 curves, computed as sum-of-(k−1) over equal-value groups per setting. Most of those are generated *by* the two duplicate curves themselves; **after excluding the two duplicated check curves, only 2 coincidences remain**, both at 120 K (`main condition_block2_1` = `check condition_block1_1` = 0.0, and `main condition_block2_2` = `check condition_block1_2` = 0.55916). The smaller number is the meaningful one for evidence independence.
- Only **2 check curves** (`condition_block1_1`, `condition_block1_2`) are genuinely new. An exhaustive scan of ratio constancy over all ordered curve pairs finds **no constant multiple** relating either new curve to any main curve — only the two identity relations exist. (An earlier draft quoted an "SD 0.06–0.12" for this; no artifact contains that statistic and it has been removed. The ratio-constancy scan is the actual evidence.)
- **Block labels carry no detectable parameter signal.** Complete 4-of-8 enumeration (70 assignments) of fitted parameters: T50 mean difference −11.76 K (exact two-sided p = **0.743**), width mean difference −2.94 K (p = 0.571), ceiling mean difference +28.39 (p = **0.086**). No parameter fully separates the blocks.
- Related null: the block interaction term in the pooled log-response model is not significant (F = 0.260, df 1/77, p = 0.611). The two blocks do **not** differ in thermal exponent. (The ratio 0.934 comes from a *separate* per-curve stage-C computation of mean block slopes, not from the pooled model's fitted interaction; the two are different quantities and are not interchangeable.)

**Supporting analysis and result evidence.**
- Analysis source: `output/native/analysis/run_stage_f.py` (`f2_cross_panel`), `output/native/analysis/run_stage_i.py` (`block_test`), `output/native/analysis/run_verification.py`
- Result evidence: `output/native/results/F2_cross_panel_integrity.json`, `output/native/results/I_consolidation.json`, `output/native/results/results.json` (keys `D4_panel_correspondence`, `D5_block_null`), `output/native/results/D_model_competition.json`

**Robustness result.** The duplication is a deterministic property of the files, verified digit-by-digit on the raw values and re-verified in the final pass; it is not a statistical inference and admits no sensitivity to model choice. The block null was tested by complete enumeration rather than by an asymptotic approximation, so the reported p-values are exact for the chosen statistic. The null is robust to which parameter is used as the test statistic: none of T50, w, or ceiling separates the blocks.

**Counterevidence, limitation, and alternative explanations.** Two readings are possible and the data do not choose between them: (i) the check panel was assembled with some main-panel curves included as reference/control rows, or (ii) the underlying source reports the same condition under two identifiers. The workspace files carry no provenance field that would decide this, so it is left as an unresolved question. The null on block labels is limited by n = 4 per block: with only 70 possible assignments, the minimum achievable two-sided p-value is 0.029, so a modest block effect could exist undetected — this bound is stated rather than glossed. Note also that `condition_block1_4`/`condition_block1_5` reuse the `block1` prefix inside the check file, so the prefix is not a file-scoped label.

**Executable falsification.** Obtain the upstream row-level provenance for the two duplicated curves. If they are shown to be the same physical run reported twice, the effective independent sample for Findings 1–4 drops from 12 to 10 curves and the check-panel replication for Finding 2 falls to 2 curves. If they are shown to be independent runs that happen to agree exactly, that would itself be a strong repeatability result — but it would require an explanation for agreement to the last published digit, which independent measurement does not normally produce.

---

## 2. Reproducibility and methods

**Environment.** Python 3.11.2 with numpy 2.4.6 and pandas 3.0.5. `scipy`, `statsmodels`, `sklearn`, and `matplotlib` were **not** present in the initial environment (verified: `import scipy` → `ModuleNotFoundError`). `matplotlib` 3.11.2 was installed during the run for figures; all inferential statistics use a dependency-free module written for this task (`lib` functions in `output/native/analysis/tib_stats.py`) that implements the incomplete beta function (Lentz continued fraction), Student-t tail probabilities, Fisher-z intervals, OLS with classical inference, mid-rank Spearman, Kendall tau-b with tie correction, and **exact complete-enumeration permutation tests** (no RNG is used anywhere, so all numbers are bit-reproducible).

**Protocol.**
1. **Integrity first.** SHA-256 of both data files checked against `data_manifest.json` before any analysis; both matched exactly.
2. **Respect the analysis unit.** One complete condition curve is the independent unit; adjacent settings are ordered scan points of the same curve. Every holdout is a whole-curve or contiguous-setting holdout. **Random curve-point splitting is never used.**
3. **Model-free descriptors first, then models.** Onset descriptor T10 (first setting where the response reaches 10, by linear interpolation in log-response) and the raw second-difference sign test are computed before any parametric fit, so the functional-form claims do not depend on the fit.
4. **Model competition on held-out data**, not on in-sample fit: contiguous-setting holdouts at cut points 240/260/280/300 K, and whole-curve holdouts with shared-shape parameters transferred from the remaining curves.
5. **Null hypotheses are tested by complete enumeration**, not asymptotics: block-label tests enumerate all 70 4-of-8 assignments; paired sign tests on holdout performance enumerate the full binomial.
6. **Alternatives are fitted, not asserted**: Arrhenius (1/T), exponential (T), power (log T), logistic, Gompertz, and Hill forms compete by AIC/BIC on `log y`.

**Reproduction commands** (run from the workspace root; each script is standalone and deterministic):

```
python3 output/native/analysis/run_analysis.py        # stages A-D: integrity, ordering, exponents, model competition
python3 output/native/analysis/run_stage_e.py         # sigmoid light-off fits and shared-shape models
python3 output/native/analysis/run_stage_f.py         # forward holdout, panel integrity, crossing census
python3 output/native/analysis/run_stage_g.py         # multi-cut holdouts, whole-curve holdout, alt. sigmoids
python3 output/native/analysis/run_stage_h.py         # decisive shared-width vs per-curve-width shape test
python3 output/native/analysis/run_stage_i.py         # block tests, single-ordering-parameter, result tables
python3 output/native/analysis/run_stage_j.py         # curvature robustness (onset exclusion, alt forms, holdout)
python3 output/native/analysis/make_figures.py        # figures + figure-backing CSV tables
python3 output/native/analysis/run_verification.py    # final pass: recompute every cited number -> results/results.json
python3 validate_output.py output                     # semantic delivery validation
```

Console transcripts for every run are preserved under `output/native/logs/`.

**Failed and abandoned methods, retained.** (i) An initial nested-model comparison of per-curve slopes produced a **negative** F statistic (F = −8.56) because the design matrices were not genuinely nested; the real error is preserved in `output/native/logs/run_analysis.log` and the test was replaced by the correct restricted-model comparison in `run_stage_h.py`. (ii) A two-stage `fit_logistic` call with a fixed ceiling initially failed with a broadcast error in `np.clip` (shape mismatch between the 2- and 3-parameter bounds); the traceback is in `output/native/logs/run_stage_e.log` and the bounds were corrected. (iii) One `--quiet` `pip install` of matplotlib reported success while the import still failed; the non-quiet retry succeeded and is what produced the figures. (iv) Attempts to resolve several candidate pre-`T0` review DOIs through the Crossref API returned no parsable record; those sources were dropped from the log rather than cited from memory (`output/native/provenance/external_sources.md`, "Sources located but not used").

**Artifact map.**
- Analysis source: `output/native/analysis/` (`tib_stats.py`, `run_analysis.py`, `run_stage_e.py`–`run_stage_j.py`, `make_figures.py`, `run_verification.py`)
- Result evidence: `output/native/results/` (JSON per stage plus `results.json`, `curve_parameters.csv`, `response_matrix_with_ranks.csv`, `fig1_data.csv`–`fig4_data.csv`)
- Figures: `output/native/figures/` (4 PNGs)
- Logs: `output/native/logs/` (per-script console transcripts)
- Provenance: `output/native/provenance/external_sources.md`; the delegated verification workflow script is archived at `output/native/provenance/verification_workflow_r2.js` and its recovered verifier verdicts at `output/native/provenance/verifier_verdicts/`

## 3. Independent verification and corrections

This report was audited by independent sub-agents that recomputed registered claims from the raw TSVs using their own code. The audit found five errors in the report text. All five were confirmed by independent re-derivation and corrected in place; none changed a registered finding's direction, verdict, or scope.

| # | Error in the report | Correct value | Where fixed |
|---|---|---|---|
| C1 | "ten of twelve [curves] are below 1 × 10⁻⁴" | **9 of 12** (`condition_block2_1` p = 0.0151, `check condition_block1_1` p = 0.0077, `check condition_block1_4` p = 0.0151 are above 1 × 10⁻⁴) | Finding 1 core result |
| C2 | "majority-negative for all 12" curves | Strict majority for **11 of 12**; `check condition_block1_1` is 4 of 8 | Finding 1 model-free check |
| C3 | "the second derivative … is negative across the whole window" | True of the **fitted** quadratic, not of every raw point (5 of 12 curves have ≥ 1 non-negative raw second difference) | Finding 1 relationship statement |
| C4 | Hyperbola `R²` 0.77–0.91 compared directly against 0.996–0.9999 | The two are on **different response scales** (`1/y` vs `log y`) and are not directly comparable; caveat added | Finding 1 robustness |
| C5 | "mean 0.02511, SD 0.00335" for the wide-window slope | **Mean 0.02325, SD 0.00380.** The 0.02511/0.00335 pair appears in no artifact — it was an error introduced when the report was written | Finding 2 core result |

C5 is the most serious of the five: the original numbers were not traceable to any computed artifact. The underlying finding is unaffected — the range (0.01796–0.03015 K⁻¹) and the max/min ratio (1.679) were correct, and the corrected mean and SD come from the same `results/D2_exponent_spread` values.

A second, environment-level issue surfaced during the audit and is recorded because it limited the audit's coverage: the workflow runtime rejected the sub-agents' structured-output tool calls with `StructuredOutput is not allowed for this Conversation role`. One verifier completed its analysis and returned its discrepancies through the transcript; others were still working when the run was stopped. Findings 1 and 2 were independently verified this way (yielding C1–C5). **Findings 3, 4, and 5 were not verified by an independent sub-agent in this run**; their evidence chains are recomputed by `run_verification.py` (`results/results.json`, keys `D3_ordering`, `D4_panel_correspondence`, `D5_block_null`) and were re-derived by the main agent, but that is self-checking, not independent verification. This gap is stated rather than papered over.

## 4. Distinguishing observation, inference, and interpretation

**Direct observations** (read from the files, no model): both panels are complete and balanced (88 and 44 rows, 11 settings each, no missing cells, 3 zero responses); the response rises monotonically with setting for 11 of 12 curves; the maximum response is 93.01; 2 of 4 check curves are digit-identical to main curves.

**Data-supported inferences** (model-based, with stated uncertainty): log-response concavity (Finding 1); the apparent-exponent spread and its partial association with T50 (Finding 2); the ordering stability boundary at the 120 K endpoint (Finding 3); weak identifiability of the response ceiling (Finding 4).

**Mechanism interpretations** (external, not established here): that the S-shape reflects light-off with coverage change (S2, S3); that the exponent spread reflects a structure-sensitive or distributed-surface apparent activation energy (S4, S5). These are offered as *consistent* explanations only. The data guide's warning is respected: these operating curves do not identify a unique microscopic interface mechanism, and no mechanism claim is registered as a finding.

**Unresolved questions**, stated as such: (i) whether the duplicated check curves are one run reported twice or two runs that agree exactly; (ii) whether the endpoint ordering instability is measurement-floor noise or a real reordering; (iii) whether the apparent exponent spread has any intrinsic component beyond saturation geometry; (iv) whether the plateau is kinetic saturation or a transport/residence-time limit.

## 5. Overall assessment

The data support a coherent operating picture: every condition rises, saturates, and decelerates in log-response, and a single saturating description with a per-curve midpoint predicts held-out settings substantially better than the conventional constant-exponent Arrhenius/extrapolation baseline (12/12 curves). What the data do **not** support is the stronger reading that each anonymous condition has its own well-determined thermal exponent, that condition ranking is a fixed property across the whole window, or that the response ceiling is measured. The check panel cannot serve as the independent replication it appears to be, because half of it duplicates the main panel. Two of the five registered findings are therefore boundary or null results; they are reported with the same weight as the affirmative ones because they materially constrain what the affirmative findings can claim.

---

## 6. Figure visual verification

This section opens each of the four published PNGs as an image and checks, from the image content alone (not from the results/CSV numbers), what the figure actually shows and whether it supports the corresponding finding's claim. It is a visual layer on top of the numerical evidence; it does not recompute any statistic.

**Fig 1 — `output/native/figures/fig1_response_curves.png` (Finding 2, saturating rise of the 8 main + 4 check curves).**
Two side-by-side panels titled "main panel: raw condition-response curves" and "check panel: raw condition-response curves." Y-axis is a linear "rate-like response" (≈0–90); x-axis is "temperature-like setting (K)" (≈120–320). The main panel shows 8 monotonically rising curves, all starting near 0 at low T and bending over (decelerating) at high T — a fanned, saturating shape. The check panel shows 4 curves with the same saturating shape; the red `condition_block1_5` curve rises highest, approaching ~90. What is seen: 8 + 4 rising, decelerating, saturating curves. **Consistent** with Finding 2's basis (curves at different positions on a shared saturating shape; max ≈ 93). No contradiction.

**Fig 2 — `output/native/figures/fig2_rank_ordering.png` (Finding 3, ordering boundary).**
Two panels. Left, "Rank trajectories cross repeatedly": y-axis "rank among 8 curves (1 = lowest response)" (1–8), x-axis temperature-like setting (K); 8 rank trajectories that are flat through the mid-range but cross repeatedly at the low-T and high-T ends. Right, "Ordering is not preserved end to end (Spearman rho = +0.31)": rank at 120 K vs. rank at 320 K with labeled endpoints; many curves cross (e.g., `block2_1` → `block1_4`, `block1_3` → `block2_2`, `block1_1` held at rank 2). The title's "Spearman rho = +0.31" matches the registered 0.310. What is seen: mid-range rank stability, repeated endpoint crossings, weak end-to-end correlation. **Consistent** with Finding 3 (stable 140–300 K, unstable at endpoints; Spearman 0.310). No contradiction.

**Fig 3 — `output/native/figures/fig3_log_response_concavity.png` (Finding 1, log-response concavity).**
Two panels titled "main panel: log-response is concave (quadratic fits)" and "check panel: log-response is concave (quadratic fits)." Y-axis "log(rate-like response)" (≈−3 to 4.5); x-axis temperature-like setting (K). Each curve is overlaid with a fitted quadratic; all curves rise and then decelerate — concave. In the check panel the red `condition_block1_5` fit clearly peaks and then turns down at high T (strong concavity). What is seen: log-scaled curves that rise and decelerate, with concave quadratic fits across both panels. **Consistent** with Finding 1 (fitted `c < 0`, concave log-response). No contradiction.

**Fig 4 — `output/native/figures/fig4_forward_holdout.png` (Finding 4, forward holdout).**
Single panel titled "Forward holdout: fit T<=280, extrapolate to 300/320 K"; subtitle "solid = saturating fit, dotted = unsaturated exponential." Y-axis "rate-like response (log scale)" (≈10⁻² to 10³); x-axis temperature-like setting (K). Black dots are observed responses; solid lines are saturating fits that plateau at high T; dotted lines are unsaturated exponentials that keep rising. A light shaded band at high T (≈T ≥ 300 K) marks the held-out/extrapolation region. At the 300/320 K holdout points, the solid lines track the black dots (plateauing near ~10²), while the dotted lines diverge and mostly overshoot above the data. What is seen: in the holdout region the saturating fit is closer to the data than the unsaturated exponential. **Consistent** with Finding 4 (saturating form beats the unsaturated exponential in the forward holdout, 12/12 curves). No contradiction.

**Overall visual verdict.** All four figures are consistent with their corresponding findings; no figure contradicts a registered claim. One caveat on scope: the figures are qualitative visualizations — they display shape, ordering, concavity, and holdout overlap but not the per-curve statistics — so they *support* the pattern each finding describes without independently recomputing any number. The quantitative claims remain traceable to the artifacts listed in each finding's "Result evidence" lines.
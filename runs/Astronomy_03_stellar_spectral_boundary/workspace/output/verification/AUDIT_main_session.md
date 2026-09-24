# Independent audit of the main-session analysis pipeline

Auditor: independent worker. Every number below was recomputed from the raw TSVs with
my own code (`numpy`/`csv`, no `scipy`). The main session's `analysis/*.py` and
`results/stage*.json` were read as the object of the audit, and the other
`verification/*.json` files as context only. JSON companion:
`AUDIT_main_session.json`.

Spot checks: 43 claimed numbers re-derived from the raw files; **0 mismatches**
(float statistics agree to rel. diff <= 1e-12, integer counts exactly).

---

## 1. `common.load_flux_grid` reshape — CORRECT, no transpose

`load_flux_grid` does **not** use `np.reshape`. It builds three sorted unique axes and
scatters with `cube[ti, mi, wi] = f`, where `ti/mi/wi` come from `np.searchsorted`.
That is the safest possible construction.

Tests I built to make it fail if the axes were transposed:

| test | result |
|---|---|
| round-trip `cube[ti,mi,wi]` vs raw `f`, all 18 480 rows | max abs diff **0.0**, exact match |
| global-max cell from the TSV text: (T=6300, M=0.9, W=575.0, F=3.09e15) vs `np.unravel_index(np.argmax(cube))` => (2, 34, 120) | match **True** |
| global-min cell: (T=5300, M=0.9, W=179.4, F=1.79e9) vs (0, 34, 0) | match **True** |
| four spot cells (5300,-1.0,179.4), (6300,0.9,179.4), (5800,0.0,300.0), (5800,0.5,850.0) | all exact |
| all five wrong orderings (M,T,W), (T,W,M), (W,T,M), (M,W,T), (W,M,T) | every one raises `IndexError`; the axis sizes 3/35/176 differ so a transpose cannot silently succeed |
| `f.reshape(3,35,176)` positional equivalence | **False** — the TSV row order is not a valid C-order flatten, which is exactly why the explicit index scatter is required |
| duplicate-cell detection claimed by the docstring | **NOT IMPLEMENTED**. `cube[ti,mi,wi]=f` silently overwrites duplicates. All 18 480 triples are unique in this frozen file (verified), so no harm here — but the docstring's guarantee and `stage1`'s `duplicated_cells` statistic (`(per_cell.shape[0] != expected_rows)*1`) are both unreachable code. |

Axis ordering is (T, M, W) = (3, 35, 176) as intended.

## 2. Wavelength-axis alignment — handled, but never validated or recorded

- The two axes share **zero common nodes** (`min |nw - ws| = 0.01 nm`, `0 of 175` exact).
- The normalised axis is a *constant small offset* below the grid nodes: about
  **-0.10 nm** in the blue, drifting to **-0.23 nm** in the red. Median nearest-node
  distance 0.12 nm, max 0.23 nm, 146/175 within 0.2 nm, 175/175 within 0.25 nm.
- It is **not** an index shift in any meaningful sense. The apparent "rows 0-66 index
  i, rows 67-174 index i+1" rule in the task brief is an artefact of the grid packing
  extra nodes into the dense band (grid has 176 nodes at 179.4..850; the file has 175
  at 179.31..849.77). The rank-preserving nearest-node correspondence holds across the
  whole axis.
- **No stage compares the two files node-for-node.** Stage 3 and stage 5 resample the
  grid onto the normalised axis with `np.interp`, and stage 4 R7 does the same. The
  alignment is therefore *sidestepped*, not *tested*.
- Residual risk, quantified: my curvature bound (quadratic through 3 nodes minus linear
  interpolation, divided by the value) is **max 6.7 %, p99 3.4 %, median 0.08 %,
  max 1.7 % inside the 299-322 nm plateau, max 1.3 % over 330-850 nm**. So quantities
  computed by interpolating the grid onto the normalised axis cannot be trusted below
  the percent level — this bounds stage 3's quoted 0.13-0.19 % photon-to-energy
  residual from below. Substituting nearest-node values for `np.interp` moves the
  plateau deviation from 5.0100 to 5.0008 and the optical mean from 1.00014 to 1.00007.
- `stage1`'s `n_wavelengths_shared_with_flux_grid_at_0p2nm = 146` reads like a
  co-sampling count; the true exact-node count is **0**. The 0.2 nm tolerance is
  unstated and unjustified (175/175 pass at 0.25 nm).

## 3. Recomputations of claimed numbers (43 checked, 0 mismatches)

Selected, claimed vs recomputed (all rel. diff <= 1e-12 unless noted):

| statistic | claimed | recomputed |
|---|---|---|
| stage1 rows / expected | 18480 / 18480 | 18480 / 18480 |
| stage1 flux min / max | 1.79e9 / 3.09e15 | same |
| stage1 densest band | 299.0-326.5 nm | 299.0-326.5 nm |
| stage1 lag-1 / lag-20 autocorr (mean over 105 spectra) | 0.972011 / 0.563693 | 0.972011 / 0.563693 |
| stage2 crossover T5300 | 448.66546164913956 | 448.66546164913956 |
| stage2 crossover T5800 | 445.8248474583328 | 445.8248474583328 |
| stage2 crossover T6300 | 409.87144741154367 | 409.87144741154367 |
| stage2 slope@180 nm T5800 | -2.9964133923155076 | -2.9964133923155076 |
| stage2 ratio M0 @180 / @850 nm | 156.78391959798995 / 0.8722466960352423 | same |
| stage3 constant C | 3.974745242072792e-13 | 3.974745242072792e-13 |
| stage3 zero-parameter ratio test @5800 K | 0.0034386096049503385 | 0.0034386096049503385 |
| stage3 photon-to-energy separation factor | 2.237425367331456 | 2.237425367331456 |
| stage4 R1 spread | 12.438939349373982 nm | 12.438939349373982 nm |
| stage4 R3 @5300 K | 447.7872723184169 | 447.7872723184169 |
| stage4 R5 frac within range @5800 K | 0.21022727272727273 | 0.21022727272727273 |
| stage5 plateau 299-322 nm | 5.010022971930054 | 5.010022971930054 |
| stage5 whole-spectrum mean abs rel err | 0.013203303637940032 | 0.013203303637940032 |
| stage6 median R^2 @5800 K | 0.989851856952804 | 0.989851856952804 |
| stage6 null median R^2 @5800 K | 0.019651966407900807 | 0.019651966407900807 |
| stage6 lag-1 slope autocorr 250-400 nm | 0.5613014544621181 | 0.5613014544621181 |

**The arithmetic of the pipeline is sound.** Everything below is about interpretation.

## 4. VACUOUS robustness checks

Named explicitly, worst first:

1. **`stage4` R2 as a whole** (`contiguous band hold-out`). No hold-out occurs. `full`
   is the slope profile fitted on *all* wavelengths; `full[s].mean()` merely re-averages
   the same full fit inside a band, and `contains_nominal_crossover` is a tautology of
   the sign profile that defined the nominal crossover. The `random_point_holdout` arm
   resamples the same full fit and also cannot fail. The honest content ("random subsets
   are uninformative because adjacent wavelengths are correlated") is asserted, not
   tested.
2. **`stage5.transition_alignment`**. The reported predicate
   `all(int(np.argmin(np.abs(step_jumps - x))) is not None for x in jumps)` is `True`
   for any non-empty input — `argmin` always returns an integer. Dead code that reads as
   a passed falsification test. (The underlying alignment is in fact partly real:
   deviation jumps at 296.23 / 319.91 / 326.41 nm; grid-step jumps at 291.9 / 296.23 /
   319.91 / 324.41 / 326.41 nm.)
3. **`stage4` R7** (`C` across temperature slices). Three numbers, no null, no tolerance,
   no acceptance criterion; `rel_spread = 0.0318` cannot fail. It is also numerically
   redundant with samples 2-4 of stage 3's own per-curve null.
4. **`stage4` R4** (metallicity-window sensitivity). Same estimator, same data, nested
   windows overlapping 21/26/31 points; "444.57 / 445.27 / 445.82 nm" is arithmetic,
   not robustness.
5. **`stage3.optical_residual_structure`**. Four correlations, no null. The reported
   lag-1 = -0.323 and `corr(residual, grid log-slope) = 0.830` are what fitting a
   straight line in log to a smooth monotone curve produces by construction.
6. **`stage5` "best model" selection**. `{5/dw, 5/dw_ownstep, constant 5, constant 1}` —
   the two constant alternatives are straw men (0.64 / 0.80 mean abs rel err); nothing
   in the set constitutes a falsification.
7. **`stage1.duplicated_cells`** — unreachable statistic (always 0).
8. **`stage4` R2's `random_point_holdout_mean_slope_std = 0.115`** — a standard
   deviation with no reference value.

Not vacuous but circular: `stage5.exact_plateaus` "predicts" 5.0 in 299-322 nm because
the grid step there is exactly 1.0 nm at all 21 points — which is how the 5/dw rule was
derived in the first place.

## 5. Reported statistics lacking a null or chance baseline

- **`stage3` photon-to-energy "2.24x separation factor" — invalid as stated.** `C` is
  fitted once on the *winning* curve (`level_3` vs F(5800 K, M=0)) over 400-850 nm and
  then held fixed while every other (T,M) candidate is scored with a constant it did not
  choose. Re-fitting `C` separately for each candidate over the same band keeps the
  winner (5800 K, M=0) but the separation over the second-best collapses from **2.237x
  to 1.234x**. Over the best other temperature it is 31.5x (0.0436 at 6300 K), so the
  temperature identification is robust; the *1.2x* margin among neighbouring
  metallicities is not a discrimination.
- `stage2.monotone_decreasing_180_to_850` — a two-point comparison, no null.
- `stage2.metallicity_monotonicity` — compared against an unstated uniform-shuffle
  expectation; no shuffle performed.
- `stage6` p-value — see below; the R^2 *levels* are fine, the calibration is not.
- `stage6.cross_slice_profile_reproduction` (r = 0.897-0.987) — no null; one smooth
  curve will correlate with itself across slices by construction.
- `stage5.composition_independence` (rel spread 3.85e-4) — no null. Five monotone
  rescalings of one underlying spectrum agree by construction; this is evidence *for*
  a shared artefact, which is the opposite of how it is phrased.
- `stage5.prediction_accuracy` residuals (1.3-1.8 %) — no competing model, and the
  residual is at the flux table's own quantisation floor (below).

## 6. Stage 6's "crossing statistic is weak" — REFUTED as framed

Stage 6 opens by citing `verification/GAP3_artefact_nulls.json` as having found the
crossing weak. GAP3's own verdict text says the opposite:
**"SURVIVES THE NULLS, with the label-permutation null explicitly declared
NON-INFORMATIVE (rank-1 degeneracy)."** Stage 6 inverted its source, then built a
replacement statistic on the inverted premise.

What is right: under the **plain whole-ladder label permutation**, the null is wide. I
re-ran it independently (500 permutations, seed 7, M <= 0.50 ladder):

| slice | observed | null q025 / q50 / q975 | frac within 5 nm | within 25 nm |
|---|---|---|---|---|
| 5300 K | 448.665 nm | 200.93 / 448.38 / 479.16 | 0.596 | 0.788 |
| 5800 K | 445.825 nm | 184.19 / 445.59 / 448.34 | 0.752 | 0.804 |
| 6300 K | 409.871 nm | 183.20 / 409.86 / 447.10 | 0.408 | 0.722 |

What is wrong: **that null is degenerate.** The composition family is rank-1 — I
reproduced GAP3 N1a's eigen-decomposition myself: PC1 = **99.44 % / 98.99 % / 99.71 %**
of the level-space variance at 5300 / 5800 / 6300 K. A random relabelling of the 31
monotone levels still traces essentially the same amplitude profile. The null's width
measures the spectrum's amplitude edge, not the plausibility of the boundary. Under the
correctly designed **monotone-preserving** nulls the crossing is sharp:

| null (my recomputation, 500 draws) | T5300 | T5800 | T6300 |
|---|---|---|---|
| random 16-of-31 monotone sub-ladder, max abs deviation from observed | **0.916 nm** | **0.764 nm** | 24.78 nm (64.2 % within 5 nm) |
| jackknife, drop any single rung, max shift | **0.111 nm** | **0.119 nm** | 24.55 nm |
| decimation every 2nd / 4th rung, max shift | <= 0.70 nm | <= 0.70 nm | <= 4.7 nm (one 24.7 outlier) |

The 6300 K spread is a genuine caveat (the 405-410 nm bracket sits on a steep step) and
the crossing is not common across slices — the three values span **38.79 nm**, which is
2.5 grid intervals. But the claim "the crossing location alone is not significantly
distinguishable from a shuffled label" does not imply "the crossing is a weak
statistic". It is pinned to better than 1 nm at two of three temperatures.

Mechanistic note: stage 6's stated reason for the weakness ("zero of a noisy, strongly
autocorrelated function") does not drive the null width. The width comes from destroying
the *order* of the ladder. The observed crossing sits at 445.8 nm on the 5 nm ladder
(bracket 445-450 nm) where the slope profile is smooth and locally linear; a zero of a
locally linear monotone function is well localised. The quoted lag-1 autocorrelation
(0.561) has no bearing on a crossing-location null.

**Bug in stage 6's implementation:** `stage6_statistical_power.py` line 90 permutes
`ms[u]` once per permutation and reuses that single permuted labelling for all 176
wavelengths, then reports `p_value_null_median_ge_observed = 0.0` from 300 permutations.
A p-value floor of 1/300 is being reported as "p < 1/300" and used to support
"the association is overwhelming". The R^2 levels themselves reproduce exactly
(median obs 0.9844 / 0.9899 / 0.9901 vs null median 0.0138 / 0.0197 / 0.0160), so the
conclusion is not in danger — but the calibration does not support the wording.

## 7. Stage 5's UV mechanism: right relation, wrong mechanism, over-precise

Recomputed exactly as claimed: C = 3.974745242072792e-13, plateau 299-322 nm =
**5.01002** (predicted 5.0), optical 330-850 nm = **1.000138** (predicted 1.0), ramp
179-296 nm = 2.0708, whole-spectrum mean abs rel err = 0.013203.

Two independent complaints:

**(a) The stated mechanism is refuted by direct test that stage 5 never ran.**
Stage 5 asserts the normalised file was produced by *re-binning the grid onto a uniform
5 nm ladder by summation*. If so, `level_3` should be proportional to the sum of grid
flux over the 5 nm window at each normalised wavelength. I tested four summation
conventions:

| convention | mean abs rel dev, all 175 | plateau 299-322 |
|---|---|---|
| exact 5 nm window `(ws>=nw) & (ws<nw+5)` | 0.534 | 0.834 |
| fixed 5-node sum from nearest node | 2.023 | 7.978 |
| trapezoid integral over `[nw, nw+5)` | 2.188 | 7.850 |
| stage 5's `5/dw` rule | **0.0132** | **0.00795** |

**Summation is rejected by a factor of 40.** What the data actually support is a
per-wavelength *normalisation/coverage* artefact — the grid's local sample density is
folded directly into the value — plus a ~0.1-1.5 nm phase lag in the transition region.
That distinction matters: it changes what the "boundary" means physically.

**(b) The failures are all in one 27-point window, and stage 5's step convention is an
arbitrary choice.** Of the 1.32 % headline error, **58 % of the total absolute error
comes from the 27 points in 296-330 nm**. Three different, equally defensible local-step
conventions give:

| convention | mean abs rel err (all) | transition 296-330 | max |
|---|---|---|---|
| stage 5's `np.interp` blend (left-edge) | 0.0132 | 0.050 | 0.737 |
| containing grid interval | **0.0177** | **0.081** | **1.449** |
| next grid interval | 0.0139 | 0.054 | 0.771 |

The "third decimal place" precision stage 5 quotes is not meaningful. The individual
failures are exactly the points whose containing interval is coarser than the file's own
sub-0.25 nm offset: at `nw = 298.92` the containing interval is [296.3, 299.0) (2.7 nm)
but the implied step is 1.10 nm — the deviation has already switched to the dense band's
~1.1 nm coverage *before* the grid node at 299.0.

**(c) The residual is at the data's own noise floor.** The flux table carries 3
significant figures (relative ulp 0.4 %, max 1.0 %). I ran a quantisation null
(+-0.5 ulp on every flux value, 120 draws): the plateau deviation has ~0.8 % rms
scatter, essentially equal to the 0.80 % residual the 5/dw model achieves there. At
T5300/T5800 the crossing itself moves only 0.037/0.040 nm (std); at T6300, 0.233 nm
(range 2.6 nm). **The 5/dw rule cannot be verified better than ~1 % with this data**, so
it is not the sharp, zero-parameter success the stage presents.

## 8. Other correctness issues

- **stage 4 R6's verdict sentence is false for one slice.** The stored
  `worst_pair_at_low_index` is `False` for 5300 K, and the decreasing pairs I recomputed
  are `(-0.85,-0.8), (-0.8,-0.75), (0.8,0.9)` at 5300 K — the pair `(0.8, 0.9)` is
  *inside* the thinned M > 0.5 region. But `R6_verdict` asserts "every long-wavelength
  decrease occurs between adjacent index levels below M = -0.6". That sentence
  contradicts stage 4's own stored field.
- **stage 4 R5's median is over a censored 21-32 % of the spectrum.** The filter
  `(-1.5, 1.0)` discards 118/103/103 of 176 wavelengths. Unfiltered medians are
  -1.83 / -1.93 / -1.85; filtered -1.22 / -1.21 / -1.19. Either way the stationary point
  is outside the index range, i.e. R5 is a null result reported as a number.
- **stage 4 R1's verdict says "crossover reproducible from any two temperature slices"
  over a 12.44 nm spread = 2.5 grid intervals.** Dropping 6300 K moves it 1.5 nm
  (445.8 -> 447.3); dropping 5800 K moves it -11.0 nm (to 434.9). "Reproducible" is
  overstated, and the three slices do **not** share one boundary (they span 38.79 nm).
- **stage 4 hard-codes the 5800 K slice** (`cube[1, ...]`) in R2/R4/R7 without flagging
  it as the hypothesis temperature chosen from stage 3.
- `stage3.unconstrained_shape_search_verdict` claims the blue-band distortion is
  "quantified in tests 4-5 below"; test 4 is a ratio test and test 5 is on the optical
  band. Neither quantifies the blue-band distortion.
- No stage records an explicit joint registration check between the two files' wavelength
  axes anywhere in `results/`.

## 9. Bottom line

**Correct:** the reshape and all axes; all 43 spot-checked numbers; stage 2's crossover
values and slopes; stage 3's zero-free-parameter ratio test at 5800 K (the normalised
ladder is a linear rescaling of the grid ladder at M = -0.9, -0.5, 0.0, 0.5, 0.9 to
0.34 %); stage 5's plateau/optical values; stage 6's R^2 levels.

One result that I expected to break under the quantisation floor and did not: stage 3's
`joint_ratio_search` winner — 5800 K, reference M = -0.90, implied ladder
[-0.5, 0.0, 0.5, 0.9], mean abs log dev 0.0034335, separation 3.4515x over the next
candidate. Re-running the whole search under +-0.5 ulp at 3 significant figures (60
draws) keeps the same winner in **60/60** draws, separation median 3.137 (range
3.022-3.278). The winner's residual sits just under the median relative ulp (0.004), so
it is not resolved beyond "the 5800 K ladder is a monotone, evenly spaced subset of the
grid ladder" — but that much is solid.

**Wrong or overstated:** stage 3's 2.24x separation factor (fair value 1.23x); stage 6's
"crossing is weak" (refuted under monotone-preserving nulls: <= 0.92 nm at 2 of 3
temperatures); stage 5's summation mechanism (refuted by 40x) and its 4-5
significant-figure precision (quantisation floor ~1 %); stage 4 R1's "reproducible"
(12.44 nm spread); stage 4 R6's verdict sentence (contradicts its own stored field).

**Vacuous:** stage 4 R2 (entire), stage 5 `transition_alignment` (dead predicate),
stage 4 R7, stage 4 R4, stage 3 `optical_residual_structure`, stage 5's "best model"
selection, stage 1 `duplicated_cells`.

**Missing:** a fair per-candidate refit of `C` in the photon-to-energy null; a
quantisation-chance baseline for every statistic quoted to more than ~1 %; a
monotone-preserving null (not a label permutation) for the crossing; an explicit,
tested cross-file wavelength registration; duplicate-cell detection that actually
detects duplicates; and a recorded null for `composition_independence` and
`cross_slice_profile_reproduction`.
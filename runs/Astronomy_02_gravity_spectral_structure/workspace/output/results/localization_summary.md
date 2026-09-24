# Localization of the between-model residual differences in time

Track: TRACK6. Data: `data/pass_residual_traces.tsv`. Independent unit: one complete time-resolved pass (n = 22). No time sample is ever treated as a replicate.

## What was measured

Centre = |relative_time_min| <= 15 min; edge = |relative_time_min| > 15 min. Primary time bins: eight 5-min bins of |t| over [0,40] min (all 22 passes contribute to every bin; 202-217 samples per bin). Envelope fits use twenty 2-min bins. The leading model, established in TRACK1, is B.

## Item 1 - binned amplitude profile (pass-averaged RMS, mm/s)

8 x 5-min bins of |t|, edges 0,5,...,40 min. Values are the mean over the 22 passes of each pass's within-bin RMS.

| bin (min) | A | B | C | A-B | A-C | C-B |
|---|---|---|---|---|---|---|
| 0-5 | 0.0443 | 0.0202 | 0.0343 | 0.0350 | 0.0175 | 0.0233 |
| 5-10 | 0.0399 | 0.0210 | 0.0281 | 0.0301 | 0.0196 | 0.0161 |
| 10-15 | 0.0326 | 0.0193 | 0.0230 | 0.0236 | 0.0180 | 0.0111 |
| 15-20 | 0.0323 | 0.0197 | 0.0238 | 0.0209 | 0.0168 | 0.0086 |
| 20-25 | 0.0291 | 0.0202 | 0.0219 | 0.0180 | 0.0148 | 0.0075 |
| 25-30 | 0.0243 | 0.0175 | 0.0185 | 0.0153 | 0.0119 | 0.0060 |
| 30-35 | 0.0220 | 0.0167 | 0.0174 | 0.0121 | 0.0112 | 0.0049 |
| 35-40 | 0.0256 | 0.0208 | 0.0215 | 0.0098 | 0.0092 | 0.0045 |

All three model profiles and all three difference profiles fall essentially monotonically from the centre to the rim: A-B goes from 0.0350 mm/s (|t| < 5) to 0.0098 mm/s (35 <= |t| < 40), a factor 3.55; C-B from 0.0233 to 0.0045; A-C from 0.0175 to 0.0092. A and C also show a mild upturn in the outermost bin (A: 0.0220 -> 0.0256), which B does not (0.0167 -> 0.0208), but the outermost bin stays below every bin inside |t| < 25 min for all three models.

The signed 16 x 5-min profile shows the difference is NOT mirror-symmetric, even though it is centre-dominant: A-B RMS is 0.0352 in [-10,-5) vs 0.0203 in [5,10), and C-B is 0.0055 in [35,40) vs 0.0028 in [-40,-35). Blocked on passes, per band ratio = RMS(right)/RMS(left): A-B median 0.557 in the 5-10 min band (17/22 passes right<left, Wilcoxon p = 0.00367) and 1.368 in the 35-40 min band (8/19, p = 0.0546); C-B median 1.796 in the 35-40 min band with 7 passes right>left. The asymmetry is therefore sign-matched between A-B and C-B (left-heavy just off centre, right-heavy at the rim) and is significant only in the 5-10 min band, i.e. it is a property of the inner envelope, not a global tilt.

Omnibus asymmetry over the inner window (RMS over t in [-20,-5) vs [5,20), one ratio per pass): A-B median ratio 0.679 with 15/22 passes right<left (Wilcoxon p = 0.147, sign test p = 0.134) and A-C 0.662 with 16/22 (p = 0.0854). This is the only place where the 'broad dome' description is incomplete, and it does not change the centre-vs-edge verdict.

Scale-free check (each pass normalised to unit (A-B)^2 before averaging): the centre (|t| <= 15, 37.5% of samples) holds a median 68.5% of the pass's A-B power (IQR 60.0-76.6%), against 37.5% if the power were spread flat in time; 20 of 22 passes exceed the flat expectation (sign-test p = 0.00012).

## Item 2 - centre-vs-edge RMS ratio, per pass

| quantity | median | q25 | q75 | min | max | passes > 1 |
|---|---|---|---|---|---|---|
| A | 1.573 | 1.350 | 1.926 | 0.744 | 2.971 | 19/22 |
| B | 1.049 | 0.921 | 1.361 | 0.662 | 1.705 | 14/22 |
| C | 1.493 | 1.241 | 1.845 | 0.713 | 2.778 | 20/22 |
| A-B | 1.955 | 1.581 | 2.336 | 0.763 | 3.300 | 20/22 |
| A-C | 1.453 | 1.073 | 1.739 | 0.635 | 2.670 | 18/22 |
| C-B | 2.740 | 1.759 | 3.373 | 1.068 | 4.957 | 22/22 |

Model B is the only model whose amplitude is nearly flat in time (median ratio 1.049, 14/22 above 1). A and C are strongly centre-loaded (medians 1.573 and 1.493; 19/22 and 20/22 above 1). The DIFFERENCES are even more centre-loaded than the models themselves: A-B median 1.955 (20/22 above 1), C-B median 2.740 (all 22 passes above 1), A-C median 1.453.

## Item 3 - within-pass time-shuffle null

For each pass, its own values are permuted over its own irregular time stamps (2000 permutations per pass per model/pair) and the centre-vs-edge ratio is recomputed. The null median is about 1.0 for every pass, so the ratio statistic is essentially unbiased under exchangeability.

| quantity | median percentile | passes >= 95th | passes whose observed ratio beats their own null median |
|---|---|---|---|
| A | 99.9 | 16/22 | 19/22 |
| B | 63.1 | 6/22 | 14/22 |
| C | 98.8 | 13/22 | 20/22 |
| A-B | 100.0 | 19/22 | 20/22 |
| A-C | 99.8 | 13/22 | 18/22 |
| C-B | 100.0 | 19/22 | 22/22 |

Why this is the right null: shuffling within a pass preserves that pass's exact amplitude distribution (its histogram of values) and its exact irregular sampling times, and destroys only the assignment of amplitudes to times. The ratio therefore measures whether the amplitude ordering in time is structured (centre-loaded) rather than exchangeable; any excess over the null cannot be produced by a heavy-tailed marginal distribution, only by real time structure. It is not a surrogate for the whole-pass amplitude level, and no p-value here treats samples as independent.

The A-B difference is the most strongly time-structured quantity in the data (median percentile 100.0; 19/22 passes at or above the 95th percentile of their own null), followed by C-B (100.0, 19/22) and A-C (99.8, 13/22). Model B is the least time-structured of the three: median percentile 63.1, with 6/22 passes at or above the 95th percentile and 1/22 at or below the 5th, and its observed per-pass ratio beats its own null median in only 14/22 passes (Wilcoxon p = 0.07). B's amplitude is therefore the closest to exchangeable in time, consistent with its centre/edge ratio being the only one near 1.

## Item 4 - blocked aggregate tests over the 22 passes

| quantity | median ratio | n above 1 | n below 1 | Wilcoxon p vs 1.0 | sign-test p |
|---|---|---|---|---|---|
| A | 1.573 | 19 | 3 | 9.06e-06 | 0.000855 |
| B | 1.049 | 14 | 8 | 0.0984 | 0.286 |
| C | 1.493 | 20 | 2 | 2.05e-05 | 0.000121 |
| A-B | 1.955 | 20 | 2 | 3.34e-06 | 0.000121 |
| A-C | 1.453 | 18 | 4 | 0.000809 | 0.00434 |
| C-B | 2.740 | 22 | 0 | 4.77e-07 | 4.77e-07 |

The A-B localization is significant on the pass unit: Wilcoxon p = 3.34e-06, sign test 20/22 (p = 0.000121). C-B is unanimous: 22/22 passes, Wilcoxon p = 4.77e-07. A-C is the weakest of the three (Wilcoxon p = 0.000809). Model B on its own fails the test (Wilcoxon p = 0.0984), so B is not significantly centre-loaded; A and C are (p = 9.06e-06 and 2.05e-05).

## Item 5 - envelope shapes fitted to the pass-averaged A-B profile

20 x 2-min bins of |t| over [0,40] min; `curve_fit`, bounded, maxfev = 4e5. AICc differences are relative to the best fit; lower is better.

| family | fitted parameters (mm/s, min) | on a bound? | profile residual RMS (mm/s) | dAICc |
|---|---|---|---|---|
| constant | 0.01998 | no | 0.00791 | 81.60 |
| gaussian | 0.03047, 22.51 | no | 0.00208 | 30.61 |
| lorentzian | 0.03251, 23.61 | no | 0.00130 | 11.76 |
| sech2 | 0.03113, 28.97 | no | 0.00180 | 24.86 |
| power_law_in_abs_t | 0.04734, 0.3248 | no | 0.00353 | 51.86 |
| power_law_shifted_t0_bounded | 0.03766, 100, 3.972 | no | 0.00103 | 5.26 |
| power_law_t0_unbounded | 0.03688, 1728, 60 | yes | 0.00090 | 0.00 |
| gaussian_shifted_center | 0.04109, 32.66, -20 | no | 0.00115 | 9.76 |
| lorentzian_shifted_center | 0.04035, 27.5, -9.933 | no | 0.00091 | 0.20 |

A constant amplitude is emphatically excluded (dAICc = 81.6; its residual RMS is 0.00791 vs 0.00091 for the best shape). Gaussian, Lorentzian, sech^2 and a canonical power law in |t| are all viable; among fits with no parameter pinned to a bound, the Lorentzian with a free centre offset wins (amplitude 0.0404 mm/s, half-width 27.5 min, offset -9.9 min; residual RMS 0.00091 mm/s), and the next-best well-identified forms (the shifted-centre Gaussian at dAICc = 9.8 and the plain sech^2 at 24.9) trail it. The Lorentzian/power-law forms edge out the Gaussian (dAICc = 30.6 for the plain Gaussian): the A-B envelope has tails heavier than a Gaussian. The power-law fits are not identifiable - a*(1+|t|/t0)^-p runs to the t0 bound and collapses to exponential-like decay with p pinned at 60, so its parameters must not be quoted even though its AICc is nominally best. The same AICc ranking holds for C-B (power-law/Lorentzian, Gaussian 30.3 AICc worse); for A-C the families are nearly tied (Gaussian, sech^2 and Lorentzian all within 1.1 AICc) because the A-C profile is flatter, and its best non-bound fit is the Lorentzian with a small offset.

Shape checks on the A-B profile: it falls outward at 17 of the 19 adjacent bin steps and rises at 2, both times by less than 0.00045 mm/s and both in the outer third (near |t| = 31 and 37 min); innermost/outermost bin ratio 3.53; the outermost 2-min bin lies 0.0005 mm/s above the second-outermost (ratio 1.048); this is 1.3% of the innermost bin and is one bin wide, i.e. a hint of an outer-edge term that is ~76.9x smaller than the central amplitude. The innermost 8 min are nearly flat (0.03494, 0.03311, 0.03084, 0.02973 mm/s), so the envelope is a broad dome, not a spike.

A model-free blocked version of the same statement: per pass, RMS(A-B) in 30-40 min divided by RMS(A-B) in 0-5 min has median 0.362, with 21 of 22 passes below 1 (Wilcoxon p = 6.68e-06, sign-test p = 1.1e-05). The A-B difference decays by roughly a factor 2.8 from centre to rim, pass by pass.

## Item 6 - whole-pass robustness

(a) Drop the 7 passes containing a sample gap > 1.5 min (pass_02, pass_03, pass_09, pass_10, pass_15, pass_18, pass_19), leaving 15 passes. The conclusion is unchanged: A-B median ratio 2.127 with 13/15 above 1 (Wilcoxon p = 0.000305); C-B median 2.825 with 15/15 above 1 (p = 6.1e-05). Model B again is the only flat one (median 1.075, p = 0.073).

(b) Leave-one-pass-out: dropping each pass in turn moves the A-B median ratio only within [1.782, 2.127] (full-sample 1.955); the Wilcoxon p stays below 0.05 in all 22 of the 22 leave-one-out aggregates (worst case p = 6.68e-06). For C-B the median stays in [2.724, 2.756] and the sign count stays 21/21 (drops to 20/21 when the single smallest pass is removed) in every leave-one-out aggregate.

(c) Replacing the mean with the median inside the RMS: the localization weakens slightly but survives - A-B median ratio 2.125, 21/22 passes above 1, Wilcoxon p = 9.54e-07; C-B 2.886, 20/22, p = 4.77e-06; B alone 1.115, p = 0.0425. The ordering (C-B most localized, then A-B, then A-C) and the flatness of B are unchanged. A time-shuffle null rebuilt on the median aggregation gives the same picture (A-B median percentile 99.9, 18/22 passes at or above the 95th percentile).

## Item 7 - verdict

The leading model's advantage is CONCENTRATED IN THE CENTRE, and it is not an edge artefact. The numbers that decide it:

1. Per pass, the centre/edge RMS ratio of the A-B difference has median 1.955 (IQR 1.581-2.336), with 20 of 22 passes above 1 (Wilcoxon p = 3.34e-06). C-B: median 2.740, all 22 passes above 1, p = 4.77e-07.
2. Binned profile: A-B RMS falls from 0.0350 mm/s at |t| < 5 min to 0.0098 mm/s at 35-40 min (factor 3.55). Outside |t| > 25 min it averages 0.0124 mm/s, against 0.0296 mm/s inside |t| <= 15 min.
3. Replacing A by B removes a median 3.95 x more variance in the centre than at the edges; the centre holds only 37.5% of the samples but absorbs 79.6% of the variance reduction.
4. The centre (|t| <= 15 min, 37.5% of samples) carries a median 68.5% of the per-pass A-B power, a concentration factor of 1.83 over a flat-in-time profile, significant on the pass unit (20/22 passes above the flat expectation, sign-test p = 0.00012).
5. The shape is a broad dome, not an edge spike and not a broad re-scaling: the envelope falls outward at 17 of 19 adjacent bin steps, the innermost 8 min vary by only a few per cent, and the outermost bin is 0.2813x the innermost (equivalently, innermost/outermost = 3.55). A constant is excluded by dAICc = 81.6; Lorentzian-like heavy-tailed forms fit best among well-identified fits (amplitude 0.0404 mm/s, half-width 27.5 min, offset -9.9 min). One qualification: the envelope is centre-dominant but not mirror-symmetric - just off centre the left half dominates (A-B median right/left = 0.557 in the 5-10 min band, 17/22 passes, Wilcoxon p = 0.00367), with the sign matched in C-B; that 0.557 median ratio is a 44% swing, much smaller than the factor 3.6 centre-to-rim decay, and it does not change the verdict.
6. There is no evidence that the model ranking is driven by edges. B is the only model whose own amplitude is time-flat (median centre/edge ratio 1.049, 14/22 passes above 1, Wilcoxon p = 0.0984); A and C (the losing models) are the strongly centre-loaded ones (1.573 and 1.493). The models differ most where the residual signal is largest - the centre - and the outermost bins of all models are small in absolute terms (<= 0.0256 mm/s), so an edge-suppression explanation would require the opposite pattern from the one observed.
7. All of the above survives dropping gappy passes, leave-one-pass-out aggregation and the median-instead-of-mean RMS.

Caveat that the numbers force: the 22 passes are the units, and the A-B localization test is significant but not unanimous (20/22, not 22/22 as the whole-pass RMS ordering is); one pass (pass_11) reverses it, and the A-C pair is the weakest of the three (median ratio 1.453, Wilcoxon p = 0.000809).

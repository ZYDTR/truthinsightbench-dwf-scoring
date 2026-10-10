# TruthInsightBench Report — Material_03 Membrane Selective Permeation

This report is the complete and only authoritative scientific account of the analysis of the
Material_03 membrane dataset. Six TSV data files
(`/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/data/*.tsv`)
were analyzed with the script `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/analysis_qa.py`
(analysis source) producing `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/results.json`
(result evidence), and robustness checks with
`/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/robustness_qa.py`
producing `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/robustness.json`.
Exactly five findings are registered below. Findings 4 and 5 are null effects, which is a valid and expected outcome.

## Finding 1: Modified membrane shows ~34x higher proton conductivity than reference membrane

(a) **Object and scope.** Proton conductivity (unit mS·cm⁻², column `proton_conductivity_mS_cm2` in
`data/conductivity_replicates.tsv`) for 10 specimens: 4 reference membrane replicas and 6 modified membrane replicas.

(b) **Relationship, direction, and boundary.** The modified membrane has higher conductivity than the reference membrane with no overlap:
the minimum modified value (36.0) exceeds the maximum reference value (3.58417). The pairwise modified/reference ratio
spans 10.044 (min modified / max reference) to 80.258 (max modified / min reference).

(c) **Core quantitative result and analysis unit.**
- Reference membrane (n=4): mean 3.155245, median 3.17275, std 0.400372, range [2.69131, 3.58417].
- Modified membrane (n=6): mean 109.833333, median 108.333335, std 70.406992, range [36.0, 216.0].
- Mean ratio (modified mean / reference mean) = **34.81**; median ratio (modified median / reference median) = **34.14**.
- **Analysis unit: one conductivity specimen.**

(d) **Supporting analysis and result evidence.** `analysis_qa.py` (F1 block) computes per-class mean/median/std/min/max,
`ratio_mean = mean(modified)/mean(reference) = 34.8098`, `ratio_median = median(modified)/median(reference) = 34.1449`,
`min_pairwise_ratio = min(modified)/max(reference) = 10.0442`, `max_pairwise_ratio = max(modified)/min(reference) = 80.2583`,
and `all_modified_gt_all_reference = true`. These values are stored in `results.json` under `finding_1_conductivity`.

(e) **Robustness result.** `robustness.json` (`F1_conductivity_robustness`): leave-one-out over the 6 modified specimens
gives ratio medians of {40.9739, 27.3159} and ratio means of {39.4898, 33.5315, 36.2782, 28.0802, 39.4053, 32.0736};
the leave-one-out minimum pairwise ratio is 10.4162 or 10.0442 in every case (never below ~10). Removing any one of the
4 reference specimens leaves the ratio mean in [33.1834, 36.462]. The ~34x separation is robust to dropping any single specimen.

(f) **Counterevidence, limitation, and alternative explanations.** There is no counterevidence to the ordering — no modified
value is below any reference value. Limitations: small n (4 vs 6) limits inferential power, and the modified class is highly
dispersed (std 70.4; range 36–216), so "34x" is a ratio of central values, not a uniform per-specimen effect (pairwise
ratios span 10.0–80.3). Alternative explanations: differences in measurement campaign, calibration, or hydration/thickness
between reference and modified batches could contribute to the offset.

(g) **Executable new observation / independent replication.** Measure at least 6 new reference and 6 new modified specimens
in an independent (blinded if possible) setup under matched conditions and test the no-overlap criterion:
min(new modified) > max(new reference). A single new modified specimen falling below a reference specimen, or a new mean
ratio materially below ~10x, would falsify the finding; replication of a ~30–35x ratio with no overlap would validate it.

## Finding 2: Modified membrane shows a bimodal local-current distribution with a large high-current fraction (~50% > 100 pA)

(a) **Object and scope.** Class-level local-current vectors (column `current_pA` in
`data/local_current_distributions.tsv`): 806 observations for the modified membrane and 476 for the reference membrane.
A supporting 841-pixel spatial current map (`data/spatial_current_map.tsv`) was also computed.

(b) **Relationship, direction, and boundary.** The modified membrane distribution is bimodal: it has both a low-current tail
(28.78% of observations < 1 pA; p10 = 0.076 pA) and a large high-current fraction (50.50% > 100 pA, 53.10% > 50 pA,
60.42% > 10 pA; p90 = 294.60 pA, max 456.22 pA). The reference membrane is entirely low-current: max = 23.739 pA,
p90 = 6.37 pA, and 0% of observations exceed 100 pA (or 50 pA). The two classes do overlap at the low end
(reference max 23.739 pA > modified p10 0.076 pA), so the separation is in the high-current fraction.

(c) **Core quantitative result and analysis unit.**
- Modified membrane (n=806): median **106.59 pA**, mean 136.12 pA, std 132.79 pA, frac > 100 pA = **0.5050**.
- Reference membrane (n=476): median **1.44 pA**, mean 2.56 pA, std 3.26 pA, frac > 100 pA = **0.0**.
- Median ratio (modified vs reference) = 74.07; mean ratio = 53.11.
- **Analysis unit: class-level local-current vector (descriptive distribution; not bootstrappable as independent specimens).**

(d) **Supporting analysis and result evidence.** `analysis_qa.py` (F2 block) computes per-class n, mean, median, p10/p90,
max, std, `frac_gt_100`, `frac_lt_1`, the median ratio, reference max, modified min, and threshold fractions
`frac_gt_50_pA` / `frac_gt_10_pA`; results are stored in `results.json` under `finding_2_local_current`.
The supporting spatial map (in `robustness.json`, `supporting_spatial_map`) is consistent: 841 pixels, mean 70.00 pA,
max 316.78 pA, 5.35% negative pixels, current-vs-radial-distance correlation −0.423, with high-current block means
(60–217 pA) in the central/lower-right region.

(e) **Robustness result.** `robustness.json` (`F2_local_current_robustness`): threshold sensitivity of the high-current
fraction — > 1 pA: 71.22% (modified) vs 60.50% (reference); > 5 pA: 63.77% vs 13.45%; > 10 pA: 60.42% vs 5.67%;
> 50 pA: 53.10% vs 0.00%; > 100 pA: 50.50% vs 0.00%; > 200 pA: 45.04% vs 0.00%. The log1p ratio of medians is 3.37.
The low-tail overlap is flagged explicitly (`reference_max > modified_p10`), bounding the claim to the high-current
fraction rather than a global shift.

(f) **Counterevidence, limitation, and alternative explanations.** Counterevidence: the low tails overlap (reference max
23.739 pA exceeds the modified p10 0.076 pA), so the classes are not separated at the low end. Limitation: observations
are class-level vectors, not independent specimens, so the distributions are descriptive and cannot be bootstrapped as
independent units; no formal mixture-model test of bimodality was run. Alternative explanations: differences in trace
selection/imaging between classes, probe-position or measurement artifacts, or differences in total current rather than
in local distribution shape.

(g) **Executable new observation / independent replication.** Acquire fresh local-current vectors for a newly fabricated
modified membrane and a newly fabricated reference membrane (independent lab preferred) and test: (i) fraction of
modified observations > 100 pA ≈ 0.5 or higher, (ii) reference max < 100 pA, and (iii) modified median / reference
median ratio > 50. A new modified membrane with median near ~1 pA and < 10% of observations > 100 pA, or a reference
membrane with high-current observations, would falsify the finding.

## Finding 3: Both membranes show strongly linear I–V behavior; modified membrane has ~10.4x the conductance of reference

(a) **Object and scope.** Ordered bias-current curves (columns `bias_V`, `current_A` in
`data/bias_current_curves.tsv`): 437 points for the modified membrane and 10 points for the reference membrane.

(b) **Relationship, direction, and boundary.** Both classes show strongly linear I–V behavior (R² ≈ 0.9997 for modified,
≈ 1.0 for reference), with near-zero intercepts (~1.2e-12 A reference, ~2.5e-12 A modified). The modified membrane's
slope (conductance) is ~10.4x the reference slope. There is no sign asymmetry: mirror asymmetries are ~2.5e-12 A
(reference) and ~4.4e-12 A (modified), both negligible.

(c) **Core quantitative result and analysis unit.**
- Modified membrane (n=437): slope 1.0091351e-09 S, intercept 2.4629381e-12 A, R² = 0.999659, mean I/V = 1.0813663e-09 S.
- Reference membrane (n=10): slope 9.7449258e-11 S, intercept 1.2319313e-12 A, R² = 0.9999999996, mean I/V = 9.7450709e-11 S.
- **Conductance ratio (modified slope / reference slope) = 10.36 (~10.4x).**
- **Analysis unit: one complete ordered bias-current curve.**

(d) **Supporting analysis and result evidence.** `analysis_qa.py` (F3 block) fits a line per class with `np.polyfit`,
computes R² = 1 − SSE/SST and the mean I/V excluding zero-bias points, and the slope ratio; results are stored in
`results.json` under `finding_3_bias_current`.

(e) **Robustness result.** `robustness.json` (`F3_bias_current_robustness`): under alternative conductance definitions,
the mean I/V ratio is 11.10 and the median I/V ratio is 10.77 (median S: 1.0493823e-09 vs 9.745025e-11), i.e. the
~10x conductance ratio is stable under alternative definitions. `both_linear = true`; mirror asymmetries are
4.4396657e-12 A (modified) vs 2.4638626e-12 A (reference), confirming negligible nonlinearity asymmetry.

(f) **Counterevidence, limitation, and alternative explanations.** Limitation: the reference curve has only 10 bias points,
so its linearity estimate rests on a small number of points, and its R² (0.9999999996) is actually higher than the
modified membrane's (0.999659) — linearity is not what differentiates the classes. Alternative explanations: a narrower
bias range for the reference could mask nonlinearity at larger biases; series resistance or contact effects could shift
apparent slopes.

(g) **Executable new observation / independent replication.** Measure a new reference membrane over a wider bias range
(e.g., ≥ 50 points spanning a symmetric bias range) and a new modified membrane, then test: (i) reference linearity with
R² > 0.99 over the extended range, and (ii) slope ratio modified/reference within [8, 13]. If the extended-range reference
curve shows strong nonlinearity (R² < 0.95) or the slope ratio collapses below ~5x, the finding is falsified.

## Finding 4: Gas-condition response is a null effect (no statistically significant mean deviation from zero)

(a) **Object and scope.** Dimensionless condition response (column `dimensionless_delta` in
`data/condition_response.tsv`) for four noble gases — Xe, Kr, Ne, Ar — each tested on 11 devices (n=11 per gas, 44 total
observations).

(b) **Relationship, direction, and boundary.** For each gas, the mean dimensionless_delta is within the 95% CI of zero;
no gas's 95% CI excludes zero (`any_gas_significant = false`). The direction of the (non-significant) means is weakly
negative for all four gases.

(c) **Core quantitative result and analysis unit.** Per gas (n=11 each):
- Xe: mean −0.1936, 95% CI [−0.4970, +0.1097] (includes zero)
- Kr: mean −0.0727, 95% CI [−0.4217, +0.2762] (includes zero)
- Ne: mean −0.0500, 95% CI [−0.3798, +0.2798] (includes zero)
- Ar: mean −0.1182, 95% CI [−0.3554, +0.1190] (includes zero)
- Overall: mean −0.1086, 95% CI [−0.2579, +0.0407] (includes zero), n=44, std 0.5053.
- **Analysis unit: one device under one test condition.**

(d) **Supporting analysis and result evidence.** `analysis_qa.py` (F4 block) computes per-gas mean and std, a z-based 95% CI
(mean ± 1.96·SD/√n), and the `ci_excludes_0` flag; per-device means and CIs are also computed. Results are stored in
`results.json` under `finding_4_condition_response`.

(e) **Robustness result.** `robustness.json` (`F4_condition_response_robustness`): leave-one-out gas means (Xe −0.0803,
Kr −0.1206, Ar −0.1055, Ne −0.1282) remain within the same near-zero range; per-device means span −0.4875 (device_01)
to +0.23 (device_10) with all 11 per-device 95% CIs including zero (e.g., device_01 CI [−1.0218, +0.0468];
device_07 CI [−1.0598, +0.2598]; device_10 CI [−0.2443, +0.7043]).

(f) **Counterevidence, limitation, and alternative explanations.** Limitation: low power (n=11 per gas, SD ≈ 0.4–0.6)
means a null is not proof of absence of effect — the overall mean (−0.109) is negative but its CI spans zero.
Alternative explanations: device-specific behavior (device_01: −0.4875 and device_07: −0.4 are the most negative) could
bias the average; unmeasured confounds (temperature, humidity) correlated with gas delivery; or the dimensionless_delta
scale may have practical floors/ceilings.

(g) **Executable new observation / independent replication.** Run a new independent condition-response experiment with ≥ 11
independent devices per gas (or a larger replication) and a pre-registered test of the null (e.g., one-way ANOVA or a
fixed-effects device model) on dimensionless_delta against zero. A gas whose pre-registered 95% CI excludes zero
(p < 0.05) would falsify the null; continued inclusion of zero for all gases would validate it.

## Finding 5: Long-term device time response is a null effect (no stable linear trend)

(a) **Object and scope.** Long-term dimensionless response over time (columns `time_day`, `dimensionless_delta` in
`data/device_time_response.tsv`) for 9 devices (device_01–device_09), 4–6 time points each (n=41 total, spanning 0–31 days).

(b) **Relationship, direction, and boundary.** Per-device linear slopes range from −0.02545 (device_09) to +0.02783
(device_04) per day; the mean slope (+0.0019/day) is close to zero — smaller than one standard error of the slope
mean (std 0.0203, SE ≈ 0.0068) — and leave-one-out mean slopes span −0.0013 to +0.0053. No device shows a strong
linear trend (linear R² ≤ 0.338; quadratic fits fit better for several devices). Five of nine slopes are positive and
four negative.

(c) **Core quantitative result and analysis unit.** Per-device slopes (per day): device_01 +0.00727, device_02 −0.02302,
device_03 −0.01108, device_04 +0.02783, device_05 +0.02594, device_06 +0.02105, device_07 +0.00256, device_08
−0.00797, device_09 −0.02545. Overall: mean slope 0.00190, std 0.02025, frac positive 0.556, frac negative 0.444;
leave-one-out mean slopes [0.00123, 0.00502, 0.00353, −0.00134, −0.0011, −0.00049, 0.00182, 0.00314, 0.00532].
**Analysis unit: one device's time series.**

(d) **Supporting analysis and result evidence.** `analysis_qa.py` (F5 block) fits a linear (and quadratic) model per
device, computes R²_linear and R²_quadratic, overall mean/std of slopes, positive/negative fractions, and leave-one-out
mean slopes. Results are stored in `results.json` under `finding_5_time_response`.

(e) **Robustness result.** `robustness.json` (`F5_time_response_robustness`): leave-one-out mean slopes span about
−0.0013 to +0.0053 (values: −0.00134 to +0.00532), none significant. Mean quadratic R² is 0.625 with individual
values up to 0.999 (device_01) and 0.973 (device_09), while linear R² is low for most devices (0.03–0.34) —
indicating any per-device shape is nonlinear rather than a stable linear trend.

(f) **Counterevidence, limitation, and alternative explanations.** Limitation: short observation windows (4–6 points over
0–31 days) and per-device n give low power to detect drift; the null does not rule out nonlinear aging. Counterevidence
to a stable trend: quadratic fits fit well for several devices (e.g., device_01 R²_quad 0.999, device_09 R²_quad 0.973,
device_08 R²_quad 0.921), suggesting nonlinear (e.g., conditioning/stabilization) evolution rather than linear drift.
Alternative explanations: initial transient/conditioning effects, periodic measurement artifacts, or environmental drift
correlated with time.

(g) **Executable new observation / independent replication.** Extend monitoring for a new cohort of ≥ 9 devices to ≥ 30 days
with ≥ 8 time points per device and pre-register a test of the mean slope against zero (e.g., mixed model with device as
random effect) with a threshold of |mean slope| > 0.01/day. A significant overall or consistently per-device slope would
falsify the null; continued non-significance (LOO mean slopes within ~±0.005/day) would validate it.

## Reproduction

Run from the project root `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project`:

```bash
cd /data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project
python3 output/analysis/analysis_qa.py      # F1–F5 + supporting spatial map
python3 output/analysis/robustness_qa.py    # robustness checks
```

No environment variables are required for the commands above. Each script resolves its default
`MEMBRANE_DATA_DIR` to `<project>/data` (two levels up from `output/analysis/`, the directory the
scripts live in) and its default `MEMBRANE_OUT_DIR` to `<project>/output/analysis`. The defaults were
verified by running the plain commands from the project root with no environment variables set
(both commands exit 0 and produce the files listed below). If you need to point at a different
dataset, set `MEMBRANE_DATA_DIR` (and optionally `MEMBRANE_OUT_DIR`) before running the commands.

Output files written:
- `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/results.json` (structured results)
- `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/results.tsv` (flat results)
- `/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project/output/analysis/robustness.json` (robustness metrics)

Input data: `data/bias_current_curves.tsv`, `data/condition_response.tsv`, `data/conductivity_replicates.tsv`,
`data/device_time_response.tsv`, `data/local_current_distributions.tsv`, `data/spatial_current_map.tsv`.
Requirements: Python 3 with `numpy` and `pandas`.

## Limitations

- Finding 1: small specimen counts (4 reference vs 6 modified); the 34x figure is a ratio of central values, with pairwise
  ratios spanning 10.0–80.3, so it should not be read as a uniform per-specimen effect.
- Finding 2: local-current observations are class-level vectors, not independent specimens; the distributions are
  descriptive and cannot be bootstrapped as independent units; bimodality is not confirmed by a formal mixture-model test.
- Finding 3: the reference bias-current curve contains only 10 points; a wider bias range would be needed to confirm
  linearity and conductance over a larger operating window.
- Finding 4 (null): low statistical power (n=11 per gas) means a null is not evidence of a true zero effect; small
  real effects could be missed.
- Finding 5 (null): short time windows (4–6 points, 0–31 days) and low linear R² limit detection of drift; nonlinear
  aging patterns (quadratic fits, R² up to 0.999) are possible and not characterized.
- The supporting spatial map is descriptive only (841 pixels, mean 70.00 pA, max 316.78 pA, radial correlation −0.423)
  and is not a primary registered finding.
- All quantitative claims above are drawn directly from `results.json` and `robustness.json` generated by `analysis_qa.py`
  and `robustness_qa.py`; no external or fabricated data were used.

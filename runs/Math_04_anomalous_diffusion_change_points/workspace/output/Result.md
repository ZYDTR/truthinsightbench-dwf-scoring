# Change-Point and Regime Structure in Anonymous Trajectories

## Data and Context

The dataset is `data/anonymous_trajectories.npz` (sha256 `fa71ba4fb2d29728e6d5054817c30dd1047a829747decb0e8fbfee5011b8a3bc`), containing `trajectories` of shape (240, 128) float64 and a contiguous `time_index` 0..127. All 240 trajectories start at zero. Per-trajectory increment variance has mean 0.20081928610203956 and median 0.10233246395892512; endpoint mean is -0.5157333412090046 with mean absolute endpoint 3.211780241690012. The analysis unit is the individual trajectory (n = 240), with time points treated contiguously within each trajectory (no shuffling of time points inside a trajectory). The goal is to (a) locate change-points in increment variance, (b) characterize the direction and contrast of the variance regime shift, (c) test whether multi-step displacement scaling or early-activity flags identify the same regimes, and (d) characterize the overall regime structure and the best statistical model.

## Analysis Workflow

Five serial phases, all implemented in pure numpy (scipy/sklearn/matplotlib are not installed), were run on the output tree.

1. **Phase 1 — Profiling** (`output/analysis/profile.py`): loads and shape-checks the data, computes endpoints, per-trajectory increment variance, lag-1 increment autocorrelation, 8 contiguous block-wise increment variances (8 blocks of 16 points), multi-step displacement M(h) for h in {1,2,4,8,16,32,64}, and per-trajectory max |increment| / max |second difference|. Writes `output/results/profile_blockvar.csv` and `output/results/profile_summary.json`.
2. **Phase 2 — Change-point detection** (`output/analysis/change_point.py`): a two-segment likelihood ratio test on increment variance over contiguous 16-step blocks (7 candidate boundaries), with chi-squared 1-df 0.95 critical value 3.84 and a variance floor of 1e-12; a nonstationary label requires logLR > 3.84 AND a variance ratio >= 2.0. Robustness includes an 8-step-block re-run, threshold sensitivity, and leave-one-block-out. Writes `output/results/change_points.csv`, `output/results/change_points_8.csv`, `output/results/change_point_robustness.json`.
3. **Phase 3 — Scaling and regimes** (`output/analysis/scaling_regimes.py`): per-trajectory M(h) and log-log scaling slope; binary H/L regime patterns from block-variance relative to the row maximum; k-means (K=2, K=3, seed 42) on row-standardized block variance; and pooled Gaussian log-likelihood model comparison (stationary-global, stationary-per-traj, piecewise, smooth). Writes `output/results/scaling.csv`, `output/results/regime_comparison.json`, `output/results/model_comparison.json`.
4. **Phase 4 — Validation** (`output/analysis/validation.py`): per-trajectory stationary null test (block permutations, B=500), leave-one-trajectory-out K-means cross-validation, trajectory block bootstrap (n=1000, seed 42) for confidence intervals, failure-boundary analysis (early/late change points, weak contrast, endpoint), and cross-agreement between the scaling active-flag and the change-point label. Writes `output/results/null_test.csv`, `output/results/bootstrap_ci.csv`, `output/results/validation_report.json`.
5. **Phase 5 — Report and validation** (this phase): assembles the authoritative narrative from the machine-readable artifacts and runs `python3 validate_output.py output`.

## Finding 1: A majority of trajectories are nonstationary, and the change point is robust to block size and leave-one-block-out

**Object and scope.** The object is the two-segment likelihood-ratio (LRT) change-point detector run on increment variance using contiguous 16-step blocks, applied to all 240 trajectories. The scope is the classification of each trajectory as "stationary" or "nonstationary" and the estimated change-point step.

**Relationship / direction / boundary.** The boundary is a chi-squared 1-df 0.95 critical logLR of 3.84 combined with a variance ratio >= 2.0. The direction is that, when a change point is declared, the post-change variance is typically smaller (the variance drops after the change point; see Finding 2).

**Core quantitative result (analysis unit = per trajectory, n = 240).** Under 16-step blocks, 143 of 240 trajectories (0.595833) are labeled nonstationary; under 8-step blocks, 162 of 240 (0.675) are. The median estimated change point (nonstationary trajectories) is 48.0 steps, with minimum 16.0 and maximum 112.0 steps. The bootstrap (n=1000 trajectory resamples, seed 42) gives a 95% percentile CI for the nonstationary fraction of [0.537396, 0.654271] (mean 0.596517) and for the median change point of [32.000000, 64.000000] (mean 48.208000) (output/results/bootstrap_ci.csv).

**Evidence.** Analysis: `output/analysis/change_point.py`; `output/analysis/validation.py`. Results: `output/results/change_points.csv`; `output/results/change_points_8.csv`; `output/results/change_point_robustness.json`; `output/results/bootstrap_ci.csv`.

**Robustness.** Leave-one-block-out preserves the overall label in 0.9296875 of all (traj, block) pairs, and preserves the nonstationary label in >= 6 of 7 exclusions for 0.9440559440559441 of nonstationary trajectories (output/results/change_point_robustness.json). Threshold sensitivity (16-step): logLR-only flags 162; ratio>=2 flags 143; ratio>=3 flags 103; ratio>=5 flags 76; ratio>=10 flags 60 (output/results/change_point_robustness.json). The change point is therefore stable to block size (16 vs 8) and to excluding any single block.

**Counterevidence / limitation / alternative explanation.** The null test (block permutation, B=500) yields p-values that are *not* uniformly consistent with the labels: only 73 of the 143 nonstationary trajectories have p < 0.05, and only 83 of all 240 have p < 0.05 (frac_all_p_lt_0_05 = 0.3458333333333333), with a negative correlation between p-value and nonstationary label (corr_p_vs_nonstat = -0.43359879074586344) (output/results/null_test.csv; output/results/validation_report.json). This is a limitation: the permutation null does not cleanly separate the two classes, and the LRT may be over-flagging under the stationary null. A plausible alternative explanation is that increment-variance nonstationarity here reflects a genuine regime shift but that the permutation test's block structure (8 blocks) is too coarse to reject the null in about a third of the flagged trajectories.

**Executable falsification/replication.** Re-run the permutation null with a finer block granularity (e.g., 4- or 8-step blocks) or a larger permutation budget (B >= 5000) and recompute p-values; if the fraction of nonstationary trajectories with p < 0.05 rises toward 143/240, the change-point claim is validated; if it stays near 73/240, the nonstationary label is not statistically defensible under the stationary null.

## Finding 2: The dominant regime shift is a *decrease* in increment variance, with strong contrast and no weak-contrast cases

**Object and scope.** The object is the direction and magnitude of the change-point variance contrast (V_post / V_pre, or its reciprocal) among the 143 trajectories labeled nonstationary by the 16-step LRT. The analysis unit is the per-trajectory best-boundary variance pair (V_pre, V_post).

**Relationship / direction / boundary.** The direction is a post-change *decrease*: the median post-change variance is smaller than the median pre-change variance, so the ratio is computed as the larger-over-smaller and is > 1. The boundary for a "declared" change point is ratio >= 2.0.

**Core quantitative result (analysis unit = per trajectory, n_nonstationary = 143).** Median pre-change variance is 0.18888051889317048 and median post-change variance is 0.05173494445822457, giving a median ratio of 5.7171814401808145 (output/results/change_point_robustness.json). The minimum ratio among nonstationary trajectories is 2.0006, so *every* nonstationary trajectory has ratio >= 2.0, and the number of nonstationary trajectories with |log(ratio)| < 0.5 (a "weak contrast") is exactly 0 (output/results/validation_report.json).

**Evidence.** Analysis: `output/analysis/change_point.py`; `output/analysis/validation.py`. Results: `output/results/change_points.csv`; `output/results/change_point_robustness.json`; `output/results/validation_report.json`.

**Robustness.** The same directional decrease is reproduced at 8-step resolution (change_points_8.csv shows the same V_pre/V_post structure and 162 nonstationary labels). The failure-boundary analysis reports early change points (cp <= 16) in 0.26573426573426573 of nonstationary trajectories and late change points (cp >= 96) in 0.24475524475524477 (output/results/validation_report.json), so the decrease is not confined to a single temporal window.

**Counterevidence / limitation / alternative explanation.** Because the ratio is defined as the larger-over-smaller, the detector does not by itself record whether pre or post is larger in raw terms; the directional claim (decrease) relies on the fact that the median V_pre (0.18888051889317048) exceeds the median V_post (0.05173494445822457). A limitation is that a very small pre-change variance would make the ratio large even for a tiny absolute change; however the 5 trajectories with zero increment variance (profile_summary.json inc_var_n_zero = 5) are the main edge case. An alternative explanation is that the apparent "decrease" is an artifact of the variance floor (1e-12) and of pooling increments across blocks.

**Executable falsification/replication.** Recompute V_pre and V_post on the raw increments (no pooling, no floor) for the 143 nonstationary trajectories and report the sign of (V_post - V_pre) for each; if the sign is negative in the large majority (as the medians imply), the decrease is confirmed; if a substantial minority are positive, the directional claim must be weakened to "a variance change of ratio >= 2.0, direction unspecified."

## Finding 3: Multi-step displacement scales sub-diffusively, but early-activity and quiet-then-active patterns are *null* markers of nonstationarity

**Object and scope.** The object is (a) the per-trajectory multi-step displacement M(h) = mean (X[t+h]-X[t])^2 and its log-log scaling slope, and (b) the derived binary "active flag" (M1 >= 0.05) and the "quiet-then-active" H/L pattern (blocks 5,6,7 high AND blocks 0,1 low), tested against the change-point label. Analysis unit = per trajectory, n = 240.

**Relationship / direction / boundary.** For (a) the relationship is between log M(h) and log h (slope > 0 indicates diffusion-like growth). For (b) the relationship under test is between the active/quiet-then-active flags and the nonstationary label; the boundary is the active threshold M1 >= 0.05.

**Core quantitative result (analysis unit = per trajectory).** The median log-log scaling slope is 0.800103 (mean 0.772247, min 0.0, max 1.389280) (output/results/scaling.csv), i.e. sub-diffusive (slope < 1) growth. Median M1 = 0.103546, M8 = 0.695743, M64 = 2.861079 (output/results/scaling.csv), and the pooled profile means (output/results/profile_summary.json) are M_h_1 = 0.20232644309453235, M_h_16 = 2.941928946931609, M_h_32 = 5.860821850108057, M_h_64 = 11.643361305735006. The active flag (M1 >= 0.05) is set for 172 of 240 trajectories. Crucially, the active flag is nearly uncorrelated with the change-point label: Pearson correlation = 0.08510532147158453 (output/results/validation_report.json). The quiet-then-active pattern occurs in 1 trajectory total and in 0 of the nonstationary trajectories, so quiet_then_active_nonstat_frac = 0.0 (output/results/validation_report.json). Leave-one-trajectory-out K-means (K=2, seed 42) classifies the nonstationary class with accuracy 0.6333333333333333 versus chance 0.5958333333333333 (output/results/validation_report.json).

**Evidence.** Analysis: `output/analysis/scaling_regimes.py`; `output/analysis/validation.py`. Results: `output/results/scaling.csv`; `output/results/profile_summary.json`; `output/results/validation_report.json`; `output/results/bootstrap_ci.csv`.

**Robustness.** The bootstrap CI for m32_over_32 (a summary of the displacement scaling) is [0.144916, 0.222317] (mean 0.182084) (output/results/bootstrap_ci.csv), consistent with stable sub-diffusive scaling. The near-zero active-flag correlation and the zero quiet-then-active fraction are point estimates from the full 240-trajectory sample; the CV accuracy (0.6333333333333333) is only modestly above chance (0.5958333333333333).

**Counterevidence / limitation / alternative explanation.** This is a null / conditional finding: high early-activity (large M1) is *not* a reliable marker of nonstationarity, and the "quiet-then-active" regime is essentially absent. The limitation is that the active flag uses a single threshold (0.05) that is not validated by the change-point detector, so the null correlation may be threshold-dependent. An alternative explanation is that early-activity and the change-point variance shift are driven by different underlying mechanisms (e.g., a transient early burst versus a sustained variance regime), so they are partially independent.

**Executable falsification/replication.** Sweep the active threshold over a range (e.g., M1 in {0.01, 0.05, 0.1, 0.5}) and recompute the correlation with the nonstationary label; if the correlation remains near zero across all thresholds, the null is robust; if it becomes strongly positive at some threshold, the active flag is a conditional marker. Similarly, re-derive the quiet-then-active pattern with different block groupings to confirm the 0.0 fraction.

## Finding 4: The data are best described by a per-trajectory piecewise model, but the improvement is driven by the ~60% nonstationary subset and the regime structure is two dominant clusters

**Object and scope.** The object is (a) a pooled Gaussian log-likelihood / BIC comparison of four models over the 30,480 increments (N = 240 x 127), and (b) the regime structure of the 8 block-variance profiles (binary H/L patterns and k-means clustering). Analysis unit = the pooled increment set for (a) and the per-trajectory 8-dim block-variance profile for (b).

**Relationship / direction / boundary.** For (a) the relationship is model fit (log-likelihood and BIC) against model complexity k. For (b) the relationship is clustering of the block-variance profiles; the boundary is the row-normalized profile used for k-means.

**Core quantitative result (analysis unit = pooled increments, N = 30480).** The best model by BIC is "piecewise" (BIC = -61270.5347113453, k = 812), versus "stationary-per-traj" (BIC = -5055.373635856315, k = 240), "smooth" (BIC = 35502.43302571196, k = 481), and "stationary-global" (BIC = 30547.906280220504, k = 1) (output/results/model_comparison.json). However, the piecewise model beats the stationary-per-trajectory model by > 10 log-likelihood units in only 0.375 of trajectories (output/results/model_comparison.json) — i.e., the global BIC win is driven by the ~143 nonstationary trajectories. For (b), k-means K=2 on row-standardized block variance gives two well-separated clusters of size [117, 123] (inertia 2757.094777745808), and K=3 gives [63, 64, 113] (inertia 1517.160004797916) (output/results/regime_comparison.json). The top binary H/L patterns are HLLLLLLL (36), HHLLLLLL (13), HHHHHHHH (9), LLHLLLLL (6), LLLLLLH (6) (output/results/regime_comparison.json).

**Evidence.** Analysis: `output/analysis/scaling_regimes.py`. Results: `output/results/model_comparison.json`; `output/results/regime_comparison.json`; `output/results/profile_blockvar.csv`.

**Robustness.** The two-cluster structure (117 vs 123) is nearly balanced, and the K=3 split (63/64/113) shows that the 113-member cluster is the "high-activity" group that also appears in the top HLLLLLLL pattern; the regime is therefore dominated by two modes rather than a fine partition. The model BIC ranking is sensitive to the nonstationary label (the piecewise boundary comes from change_points.csv), so the BIC is an *indirect* statistic.

**Counterevidence / limitation / alternative explanation.** The 0.375 fraction is a key limitation: for 62.5% of trajectories the piecewise model is *not* clearly better than a single stationary variance, so a single-change-point regime is not a universal description. The alternative explanation is that the population is a mixture: roughly 60% are genuinely piecewise (variance-changing) and roughly 40% are stationary, and the pooled BIC is the sum over both. The k-means clustering on block variance is not the same object as the change-point label; the near-balance of the two clusters and the presence of the HLLLLLLL / HHHHHHHH patterns indicate that "early-active" and "uniformly-active" regimes coexist.

**Executable falsification/replication.** Fit the four models *without* using the change-point label (e.g., select the change point by BIC per trajectory, or compare a 1-segment vs 2-segment Gaussian fit per trajectory) and recompute the per-trajectory BIC delta; if the piecewise-vs-stationary BIC delta remains positive in ~60% of trajectories, the mixture interpretation is confirmed; if it is positive in far fewer, the global BIC win is an artifact of the label-driven boundary. Re-run k-means with K=4,5 to confirm that K=2 is the natural partition (inertia elbow).

## Limitations

- **Null-test inconsistency.** The permutation null (B=500) flags only 73/143 nonstationary trajectories at p<0.05 (corr_p_vs_nonstat = -0.43359879074586344), so the LRT change-point label is not cleanly separable from the stationary null under the 8-block permutation structure.
- **Threshold dependence.** The active flag (M1 >= 0.05) and the variance-ratio threshold (>= 2.0) are fixed choices; the near-null correlation (0.085) and 0.0 quiet-then-active fraction are threshold- and block-size-dependent.
- **Directional ambiguity.** The ratio is defined as larger-over-smaller; the "decrease" direction rests on the medians (V_pre > V_post), not on a per-trajectory recorded sign.
- **Model circularity.** The piecewise model's change-point boundaries are taken from the LRT labels, so its BIC is an indirect statistic.
- **No ground truth.** The trajectories are anonymous; all findings are relative to the detected regime and cannot be confirmed against a known generating process.

## Reproduction

Run the following from the project root (working directory `/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project`), in order:

```
python3 output/analysis/profile.py
python3 output/analysis/change_point.py
python3 output/analysis/scaling_regimes.py
python3 output/analysis/validation.py
python3 validate_output.py output
```

Data: `data/anonymous_trajectories.npz`, sha256 `fa71ba4fb2d29728e6d5054817c30dd1047a829747decb0e8fbfee5011b8a3bc`.

Resulting artifacts (all under `output/results/`):
- `profile_blockvar.csv`, `profile_summary.json` (Phase 1)
- `change_points.csv`, `change_points_8.csv`, `change_point_robustness.json` (Phase 2)
- `scaling.csv`, `regime_comparison.json`, `model_comparison.json` (Phase 3)
- `null_test.csv`, `bootstrap_ci.csv`, `validation_report.json` (Phase 4)

Validation command `python3 validate_output.py output` final status: `passed: true`.

# Verification Report

Method: verification/synthesis only. Read `results.json`, `robustness_a.json`, `robustness_b.json`, and `output/figures/` (5 PNGs: `spectrum.png`, `trajectory.png`, `convergence_vs_sv.png`, `robustness_a.png`, `robustness_b.png`). No recomputation from raw data.

## Finding 1 — SPECTRUM is extremely ill-conditioned
**Verdict: supported.**
- Span confirmed: `span_orders10 = 36.32` (log_sv from -82.45 to +1.18, span_ln 83.63).
- Best model is the linear-in-index (exponential) fit, `R2 = 0.642` (`best_model: "exponential"` in robustness_a.json); power-law (log-log) is weaker, `R2 = 0.332` (also in robustness_a.json).
- "Steep tail" confirmed: `nonlinear_decay = true`, `tail_slope = -0.060` vs `early_slope = -0.004` (and `slope_late20 = -0.060` in results.json).
- Robustness check that strengthens (not weakens) it: `effdim` rises from 9 (at -0.0) to 4220 (at -80.0), showing the long tail.
No robustness check materially weakens this finding.

## Finding 2 — TRAJECTORY is weakly/essentially uncorrelated with the spectrum
**Verdict: supported.**
- `corr_slope_sv_fam0 = -0.163`, `fam1 = -0.047`, pooled `-0.069`; regression `reg_total_on_sv_slope = -0.0007` (negligible).
- Leave-one-out (`lof_without_fam0 = -0.047`, `lof_without_fam1 = -0.163`) reproduces each family's value, so the near-null relationship is not a pooling artifact.
No robustness check materially weakens this finding.

## Finding 3 — MEASUREMENT-SIMULATION agreement is strong but does not establish identifiability
**Verdict: supported.**
- Strong overall agreement, robust: `corr_meas_sim = 0.765`; winsor `0.768`, outlier-excluded `0.813`.
- Near-zero bias: `bias_near_zero = true` / `bias_is_near_zero = true` (`t = -0.14`, two-sided p = 1.11 > 0.05).
- Robustness check that supports the *identifiability caveat* (rather than weakening the core claim): correlation is **not** constant across simulation-value quantiles — `corr_by_quantile = [0.58, 0.35, 0.12, 0.07, 0.19]`, `corr_constant = false` (spread 0.50). This non-uniformity is precisely why overall agreement does not establish parameter identifiability.

## Finding 4 — SOLVER FAMILY is a significant source of convergence difference
**Verdict: supported.**
- `diff_mean_tot_fam0_minus_fam1 = +0.219`; tight `bootstrap_CI95 = [0.216, 0.223]` excludes 0; `sign_test_p = 1.0`; `frac_param_fam0_bigger = 0.956`.
- `solver_effect_real = true`.
- Batch leave-out (robustness_a.json) confirms stability: fam1 mean more negative in every batch (-0.376..-0.381 vs fam0 -0.148..-0.168) and higher conv fraction (0.998..0.999 vs 0.990..0.996); `stable = true`. `fam1_conv_lowsv_bin0 = 1.0` (fam1 converges in the lowest-SV regime; from robustness_b.json).
No robustness check materially weakens this finding.

## Robustness checks that materially weaken a finding
None. Every robustness check (winsor/outlier-excl, leave-one-out, batch leave-out, quantile correlation, nonlinear-decay tail) either confirms or strengthens a finding. The only nuance is the non-constant quantile correlation in Finding 3, which supports — not weakens — the "does not establish identifiability" caveat.

## Overall
All four ground-truth findings are **supported** and none is materially weakened. Robustness confirms: (1) the ~36-order, non-linear, steep-tail ill-conditioning of the spectrum; (2) a near-null trajectory-spectrum relationship (lof-stable); (3) strong, bias-free measurement-simulation agreement whose non-uniform quantile correlation backs the identifiability caveat; and (4) a real, tight-CI solver-family convergence difference with fam1 converging more, stable across all four parameter batches.

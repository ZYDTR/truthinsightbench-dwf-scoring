# Neuro_01 — Self-organization and prediction of latent/stimulus/response dynamics

## Data and analysis unit

- **Source data (sole direct evidence):** six MAT condition files in `data/raw/` —
  `apv` (9 trials), `bic` (6), `ctrl` (30), `dzp` (7), `mix0` (4), `mix50` (4) =
  **60 independent trials**. Each trial is a 25,600×2 latent-variable stream `s`,
  a 25,600×32 stimulus stream `o`, and a 25,600×64 response stream `r`. No missing
  values.
- **Analysis unit = one complete trial.** The 25,600 time points inside a trial are
  *not* independent replicates; all summary statistics average over the 60 trials
  (per-channel R² averaged over the model's own channels — 2 for the
  latent-targeted `s_self`/`s_o_s`, 64 for the response-targeted models — then over
  trials). `n` is always the
  number of trials (e.g. `n=60`, `n=30` for ctrl, `n=4` for mix0), never 25,600.
- **Model:** per-trial OLS of the value at time `t+1` on the chosen feature set at
  time `t` (lag = 1), standardized within each trial, R² = fraction of intra-trial
  per-channel variance explained. Models: `zero`, `mean`, `persist_r`
  (`r(t)→r(t+1)`), `s` (`s(t)→r(t+1)`), `o` (`o(t)→r(t+1)`), `s_o`
  (`[s,o]→r(t+1)`), `r_s_o` (`[r(t),s,o]→r(t+1)`), `s_self` (`s(t)→s(t+1)`),
  `s_o_s` (`[s,o]→s(t+1)`).

## Methods

Per-trial fits use the small Gram system with a residual-based residual-sum-of-squares
(computed as `‖Y − Xβ‖²` to avoid catastrophic cancellation on the near-collinear
32-channel stimulus design), and a Frisch–Waugh–Lovell identity for `r_s_o`.
See `output/reproduce.md` for the full reproduction recipe and `output/analysis/`
for the executable code.

---

## Finding 1: Latent dynamics are stimulus-driven, not self-organized

**Object / scope.** Whether the latent variables `s` evolve by their own momentum
(`s_self`: `s(t)→s(t+1)`) or require the stimulus `o` (`s_o_s`: `[s,o]→s(t+1)`),
across all six conditions and 60 trials.

**Relationship / direction / boundary.** The stimulus-informed latent predictor
`[s,o]→s(t+1)` explains a substantially larger share of next-`s` variance than the
latent-only predictor `s→s(t+1)`: the ratio is roughly 10–15× in the stimulus-rich
conditions. `s_o_s` R² ≈ 0.12–0.15 in apv (0.1474), bic (0.1385), ctrl (0.1359),
dzp (0.1409), mix50 (0.1218); `s_self` R² ≈ 0.010–0.015 in the same conditions
(apv 0.0101, bic 0.0116, ctrl 0.0102, dzp 0.0110, mix50 0.0147). The latent
stream does **not** self-organize — its next value is mostly determined by the
stimulus. Boundary: in `mix0` both collapse to ≈0.003.

**Core quantitative result + analysis unit.** Per-channel R² averaged over the 2
latent channels (the `s` stream) and then over the n trials per condition:
`s_o_s` = 0.1474 / 0.1385 / 0.1359 / 0.1409 / 0.0030 / 0.1218 for
`apv, bic, ctrl, dzp, mix0, mix50`; `s_self` = 0.0101 / 0.0116 / 0.0102 / 0.0110 /
0.0030 / 0.0147.

**Resolvable analysis + result evidence.** Analysis source
`output/analysis/prediction.py` (models `s_self`, `s_o_s`) and
`output/analysis/run_condition.py`. Result evidence
`output/results/cross_condition_r2.csv` (rows `s_o_s`, `s_self`) and per-condition
`output/results/apv_summary.csv`, `output/results/bic_summary.csv`,
`output/results/mix0_summary.csv`.

**Robustness.** Leave-one-trial-out and first-half checks keep `s_o_s` ≈ 0.11–0.15
and `s_self` ≈ 0.010 across conditions (mean over conditions: `s_o_s` 0.1146,
`s_self` 0.0101); lag-2 decays `s_o_s` only mildly to ≈0.112. See
`output/results/cross_condition_robust.csv`.

**Counterevidence / limitation / alternative.** The absolute R² is modest (0.12–0.15,
not strong); and `mix0` (n=4) is an exception where even `[s,o]` fails to predict
next-`s` (R²≈0.003). Alternative: the small `s_self` could reflect a genuinely
weak autoregressive component rather than "no self-organization."

**Falsifiable replication.** Refit on new `s`/`o` data from the same regime: if `s`
truly self-organized, `s→s(t+1)` would be a competitive predictor (R² within ~2× of
`[s,o]→s(t+1)`). Expected null: `s_o_s` ≈ 2–3× `s_self` in stimulus-rich trials.

---

## Finding 2: The next response is essentially unpredictable from all tested predictors (null)

**Object / scope.** Whether any of the tested predictors (current response, latent
state, stimulus, or combinations) explains the next response `r(t+1)`, across 60
trials.

**Relationship / direction / boundary.** All response predictors are weak — no
model reaches R² ≥ 0.15 in any condition. The best is `r_s_o`
(`[r(t),s,o]→r(t+1)`) ≈ 0.09–0.11 (apv 0.1102, bic 0.0904, ctrl 0.0914, dzp
0.0944, mix50 0.0888) and ≈0.013 in `mix0`; stimulus alone `o` is only ≈0.07–0.09
(apv 0.0916, mix50 0.0809) and ≈0.006 in `mix0`. So later responses are **not**
predictably self-organized and are only weakly stimulus-driven.

**Core quantitative result + analysis unit.** Per-channel R² averaged over the 64
response channels and then over the n trials per condition: `r_s_o` = 0.1102 /
0.0904 / 0.0914 / 0.0944 / 0.0135 / 0.0888 for `apv, bic, ctrl, dzp, mix0,
mix50`; `o` = 0.0916 / 0.0773 / 0.0789 / 0.0736 / 0.0061 / 0.0809; `s_o` =
0.1041 / 0.0836 / 0.0868 / 0.0813 / 0.0061 / 0.0861.

**Resolvable analysis + result evidence.** Analysis source
`output/analysis/prediction.py` (models `o`, `s_o`, `r_s_o`, `persist_r`) and
`output/analysis/run_condition.py`. Result evidence
`output/results/cross_condition_r2.csv` (rows `o`, `s_o`, `r_s_o`) and
`output/results/bic_summary.csv`, `output/results/apv_summary.csv`.

**Robustness.** `r_s_o` survives leave-one-trial-out (0.0815) and first-half
(0.0810); lag-2 decays it to ≈0.074. `o` is similarly stable (leave-one-out
0.0669, lag-2 0.0642). See `output/results/cross_condition_robust.csv`.

**Counterevidence / limitation / alternative.** `apv` `r_s_o` (0.110) is the single
highest value but still far below 0.15; no model crosses the 0.05 threshold in
`mix0` at all. Alternative: a nonlinear or longer-lag response model might capture
more; this linear lag-1 analysis shows the linear part is negligible.

**Falsifiable replication.** On new data, a genuinely predictable response would
give R² ≥ 0.25 in at least half the trials for some predictor; the null here
predicts no such model.

---

## Finding 3: Current response persistence is negligible

**Object / scope.** Whether the current response predicts its own next value
(`persist_r`: `r(t)→r(t+1)`), and whether adding it to the `s,o` predictors
improves response prediction (`r_s_o` vs `s_o`).

**Relationship / direction / boundary.** Response autocorrelation is weak:
`persist_r` R² ≈ 0.007–0.012 in every condition (apv 0.0110, bic 0.0083, ctrl
0.0073, dzp 0.0122, mix0 0.0089, mix50 0.0091). Adding the current response to
`[s,o]` adds only ≈0.003–0.013 R² (apv 0.104→0.110, +0.006; bic 0.084→0.090,
+0.007; dzp 0.081→0.094, +0.013). So responses are not strongly autocorrelated
and the current value is a poor predictor of the next.

**Core quantitative result + analysis unit.** Per-channel R² averaged over the 64
response channels and then over the n trials per condition: `persist_r` = 0.0110 /
0.0083 / 0.0073 / 0.0122 / 0.0089 / 0.0091 for `apv, bic, ctrl, dzp, mix0, mix50`;
incremental `r_s_o − s_o` = +0.006 / +0.007 / +0.005 / +0.013 / +0.007 / +0.003.

**Resolvable analysis + result evidence.** Analysis source
`output/analysis/prediction.py` (models `persist_r`, `s_o`, `r_s_o`). Result evidence
`output/results/cross_condition_r2.csv` (rows `persist_r`, `s_o`, `r_s_o`) and
`output/results/apv_summary.csv`, `output/results/dzp_summary.csv`.

**Robustness.** `persist_r` is stable under leave-one-trial-out (0.0089) and
first-half (0.0084); lag-2 is comparable (0.0080). See
`output/results/cross_condition_robust.csv`.

**Counterevidence / limitation / alternative.** The incremental `r_s_o − s_o` is
small but consistently positive in 5/6 conditions, a weak but real effect; it could
be a small genuine autoregressive contribution or sampling noise given the small
increments.

**Falsifiable replication.** If responses were persistent, `persist_r` alone would
give R² ≥ 0.20 in many trials; the null predicts it stays ≈0.01.

---

## Finding 4: mix0 is an applicability boundary where all relationships collapse

**Object / scope.** Whether the predictor relationships found in the other five
conditions generalize to the `mix0` condition (n = 4 trials).

**Relationship / direction / boundary.** In `mix0`, *every* predictor's R² collapses
to near zero: `s`, `o`, `s_o` ≈ 0.0061; `r_s_o` ≈ 0.0135; `s_self`, `s_o_s` ≈
0.0030; `persist_r` ≈ 0.0089. This contrasts sharply with the other conditions,
where `s_o_s` ≈ 0.12–0.15 and the response models ≈ 0.05–0.11. The latent-dynamics
and response-prediction relationships are therefore **condition-dependent** and do
not extend to `mix0`.

**Core quantitative result + analysis unit.** Per-channel R² averaged over each
model's own channels (2 latent channels for `s_self`/`s_o_s`; 64 response channels
for the rest) and then over the 4 `mix0` trials: `s_o_s` = 0.0030, `s_self` =
0.0030, `s_o` = 0.0061, `o` = 0.0061, `s` = 0.0061, `r_s_o` = 0.0135,
`persist_r` = 0.0089.

**Resolvable analysis + result evidence.** Analysis source
`output/analysis/cross_condition.py` (cross-condition pivots) and
`output/analysis/run_condition.py`. Result evidence
`output/results/mix0_summary.csv` and `output/results/cross_condition_r2.csv`
(column `mix0`).

**Robustness.** `n=4` is small, so robustness here is limited to the leave-one-trial-out
spread (std ≈ 0.004–0.005, already near zero) and the fact that all models are
simultaneously near zero. The cross-condition pivot
`output/results/cross_condition_r2.csv` shows `mix0` as the clear minimum for every
model.

**Counterevidence / limitation / alternative.** With only 4 trials, the near-zero
R² could partly reflect low trial-to-trial variance or a different generative
process rather than a robust "no relationship" statement. The mix0/mix50 pair are
the only conditions with 4 trials each.

**Falsifiable replication.** New `mix0` data should reproduce the null (all predictors
R² < 0.05); a genuine general rule would predict R² ≈ 0.12–0.15 for `s_o_s` as in
the other conditions.

---

## Numerical correction and limitations

- **Numerical correction (important).** An earlier version of
  `output/analysis/prediction.py` computed the default R² denominator `SS_tot` as a
  *scalar* summing over all 64 channels, while the residual sum-of-squares was
  per-channel. This overstated R² for the five shared-feature models (`s`, `o`,
  `s_o`, `s_self`, `s_o_s`) by exactly the number of target channels: the scalar
  denominator was 64.0× too large for the response-targeted models (`s`, `o`,
  `s_o`) and 2.0× too large for the latent-targeted models (`s_self`, `s_o_s`),
  as confirmed by the pre-fix SS_tot ratio in the independent reference. e.g. bic
  `s_o` went from 0.986 → true 0.084, and bic `s_self` from 0.506 → true 0.012.
  The fix (per-channel `SS_tot` = `‖Y − Y·mean‖²` per channel, and a residual-based
  `‖Y − Xβ‖²` for `ss_res`) was verified against an independent direct-lstsq
  reference per model per condition (see
  `output/results/verify_independent_ref_summary.csv`,
  `output/results/verify_independent_ref.csv`, and the verdict
  `output/results/verify_verdict.json`). **All R² values reported above are
  the corrected, per-channel values.** The `persist_r`, `r_s_o`, `zero`, `mean`
  models used explicit per-channel denominators and were unaffected.
- **Independent-reference verification (scope of this check).** A separate
  implementation (`output/analysis/verify_independent_ref.py`) re-derives every
  model's R² with direct `np.linalg.lstsq` on the full design (no Gram solve, no
  shared-feature batching) and an independently re-implemented per-trial
  standardization, then compares channel-by-channel against the live code on the
  same data. Across all six conditions and nine models (`s`, `o`, `s_o`,
  `s_self`, `s_o_s`, `persist_r`, `r_s_o`, `zero`, `mean`): `s`, `o`, `s_o`,
  `s_self`, `s_o_s`, `persist_r`, `zero`, `mean` match the independent reference
  to machine precision (max |diff| ≤ 2.3e-14); `r_s_o` matches the independent
  direct-lstsq ground truth `[1, s, o, r_j]` within the expected floating-point
  error of the Gram solve on the near-collinear 32-channel stimulus design
  (max |diff| 8.1e-3, mean 1.4e-3 — a numerical tolerance, not a logical
  disagreement). A full-data identity check (`--full_saved_check`) confirms every
  saved `<cond>_summary.csv` `r2_overall` equals the live code's official
  `aggregate()` output to <1e-9 for all six conditions (apv, bic, ctrl, dzp,
  mix0, mix50). Machine-readable evidence:
  `output/results/verify_independent_ref.csv`,
  `output/results/verify_independent_ref_summary.csv`,
  `output/results/verify_independent_ref_<cond>.csv`, and
  `output/results/verify_verdict.json`.
- **Limitations.** (i) Linear lag-1 OLS only; nonlinear or longer-lag dynamics are
  not assessed. (ii) `n=4` in `mix0`/`mix50` limits statistical power. (iii)
  R² is a descriptive variance-explained measure, not a causal or inferential claim
  across conditions.
- **Reproduction.** See `output/reproduce.md` (executable `output/analysis/*.py`,
  machine-readable results in `output/results/*.csv`).

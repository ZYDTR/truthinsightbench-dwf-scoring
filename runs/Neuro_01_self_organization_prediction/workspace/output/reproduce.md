# Reproduction

## How to reproduce the analysis

All data come from the workspace `data/raw/*.mat` condition files. No external
dataset is downloaded, substituted, or merged. The only dependencies are
`numpy`, `pandas`, and `scipy`.

All commands assume the **project root** as the starting directory, i.e. the
directory containing `output/` and `data/` (the same directory that holds
`validate_output.py`). The scripts resolve their own data/output paths from their
own location (`__file__`), so they do **not** depend on the current working
directory; the `output/analysis/` prefixes below are written for readability.

1. **Per-condition analysis** (one condition at a time, or all together):

   ```bash
   python3 output/analysis/run_condition.py apv bic ctrl dzp mix0 mix50
   ```

   For each condition this writes:
   - `output/results/<cond>_summary.csv` — per-model trial-averaged R², median
     per-trial R², MSE, and counts of trials above 0.05 / 0.5.
   - `output/results/<cond>_robustness.csv` — leave-one-trial-out, lag-2, and
     first-half robustness checks.
   - `output/results/<cond>_per_trial.csv` and `<cond>_per_channel.csv` —
     granular per-trial / per-channel R².
   - `output/results/<cond>_summary.json` — compact JSON summary.

2. **Cross-condition aggregation** (run after step 1):

   ```bash
   python3 output/analysis/cross_condition.py
   ```

   Writes `output/results/cross_condition_r2.csv`,
   `cross_condition_median.csv`, `cross_condition_nabove.csv`,
   `cross_condition_across.csv`, and `cross_condition_robust.csv`.

3. **Independent-reference verification** (optional cross-check of the numerical
   pipeline). This is a *separate* implementation using direct `lstsq` and its
   own standardization, compared against the live code:

   ```bash
   python3 output/analysis/verify_independent_ref.py --cond all --nrows 4000
   python3 output/analysis/verify_independent_ref.py --full_saved_check
   ```

   The first command writes `output/results/verify_independent_ref.csv` (plus
   per-condition `verify_independent_ref_<cond>.csv` and
   `verify_independent_ref_summary.csv`); the second confirms every saved
   `<cond>_summary.csv` `r2_overall` equals the live code's full-data
   `aggregate()` output. Machine-readable verdict:
   `output/results/verify_verdict.json`.

4. **Validator** (run from the project root):

   ```bash
   python3 validate_output.py output
   ```

## Key analysis files
- `output/analysis/data_loader.py` — loads the six MAT condition files into the
  60 independent trials (9 apv, 6 bic, 30 ctrl, 7 dzp, 4 mix0, 4 mix50).
- `output/analysis/prediction.py` — per-trial OLS models and aggregation.
- `output/analysis/run_condition.py` — per-condition driver + robustness checks.
- `output/analysis/cross_condition.py` — cross-condition pivots.
- `output/analysis/verify_independent_ref.py` — independent-reference verification
  (direct-lstsq comparison of the live code against a separate implementation).

## Numerical notes (important for exact reproduction)
- **Analysis unit is the trial.** Within a trial the 25,600 time points are
  *not* independent replicates; results are aggregated across the 60 trials.
- **R² is per-channel.** Each channel's R² uses that channel's own intra-trial
  SS_tot as the denominator; the per-channel R² values are then averaged over
  channels and then over trials. (An earlier scalar SS_tot that summed over all
  channels overstated R² by ~n_channels; this has been fixed — see the
  "numerical correction" section of `Result.md`.)
- **Gram solve uses the residual** `||Y - X β||²` rather than the closed-form
  Gram scalar, to avoid catastrophic cancellation on the near-collinear 32
  stimulus-channel design (mix0/mix50).
- `r_s_o` uses the Frisch–Waugh–Lovell identity with a per-channel SS_tot
  taken from the *original* response, so it is exactly the full OLS fit.

## Debug artefacts
The `output/analysis/debug_*.py` scripts document the numerical investigation
that identified and fixed the scalar-SS_tot bug; they are not part of the
reporting pipeline.

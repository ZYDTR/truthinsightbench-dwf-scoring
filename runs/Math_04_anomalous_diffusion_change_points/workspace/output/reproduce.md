# Reproduction Log

## Data
- Path: `/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project/data/anonymous_trajectories.npz`
- sha256: `fa71ba4fb2d29728e6d5054817c30dd1047a829747decb0e8fbfee5011b8a3bc`
- Contents: `trajectories` (240, 128) float64; `time_index` (128) int64, contiguous 0..127.

## Exact commands (run from the project root)
1. `python3 output/analysis/profile.py`
2. `python3 output/analysis/change_point.py`
3. `python3 output/analysis/scaling_regimes.py`
4. `python3 output/analysis/validation.py`
5. `python3 validate_output.py output`

## Resulting artifacts (all under `output/results/`)
- `profile_blockvar.csv`, `profile_summary.json` (Phase 1, `profile.py`)
- `change_points.csv`, `change_points_8.csv`, `change_point_robustness.json` (Phase 2, `change_point.py`)
- `scaling.csv`, `regime_comparison.json`, `model_comparison.json` (Phase 3, `scaling_regimes.py`)
- `null_test.csv`, `bootstrap_ci.csv`, `validation_report.json` (Phase 4, `validation.py`)

## Validation
- Command: `python3 validate_output.py output`
- Final status: `passed: true`

## Environment
- python3 3.11.2; numpy 2.4.6. All scripts use pure numpy (scipy/sklearn/matplotlib not installed).

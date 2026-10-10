# Reproduction

This document is the step-by-step reproduction trace for every registered finding in `Result.md`.
All scripts are self-contained, read **only** from the bundled `data/` directory, and write their
machine-readable evidence into `output/`. No external data or literature were used (T0 boundary respected).

## Prerequisites

- Python 3 (tested with 3.11)
- `pandas`, `numpy` (required)
- `matplotlib` (optional, only for `figures.png`)

## Files (input)

| File | Role | Rows |
|---|---|---|
| `data/network_node_summary.tsv` | node strength / homophily / distance | 800 |
| `data/network_profile_a.tsv` | coords + 5 annotations | 800 |
| `data/network_profile_b.tsv` | coords + annotation_1 | 213 |
| `data/network_profile_c.tsv` | coords + 3 annotations | 39 |
| `data/trimming_sensitivity.tsv` | trim sensitivity | 28 |

## Step 1 — Per-dimension analyses

Each script writes one machine-readable JSON artifact. Run them from `output/`:

```bash
python3 f1_strength_distance.py    # -> f1_strength_distance_result.json
python3 f2_spatial_gradient.py     # -> f2_spatial_gradient_result.json
python3 f3_homophily_null.py       # -> f3_homophily_null_result.json
python3 f4_trimming.py             # -> f4_trimming_result.json
python3 f5_profiles_annotations.py # -> f5_profiles_annotations_result.json
```

## Step 2 — Supporting figure (optional)

```bash
python3 make_figure.py             # -> figures.png
```

## Step 3 — Consolidate

```bash
python3 aggregate.py               # -> final_results.json (loads all five artifacts)
```

## Finding → evidence map

| Finding | Analysis source | Result evidence |
|---|---|---|
| 1 | `f1_strength_distance.py` | `f1_strength_distance_result.json` |
| 2 | `f2_spatial_gradient.py` | `f2_spatial_gradient_result.json` |
| 3 | `f3_homophily_null.py` | `f3_homophily_null_result.json` |
| 4 | `f4_trimming.py` | `f4_trimming_result.json` |
| 5 | `f5_profiles_annotations.py` | `f5_profiles_annotations_result.json` |

All five per-dimension artifacts are also loaded into the single consolidated
`final_results.json` (see its `findings_to_evidence` block).

## Key output values (from the artifacts)

- F1: overall `r = 0.648` (p < 1e-300); per-system `r = [0.672, 0.656, 0.565, 0.643]`; robust = true.
- F2: overall strength–`coord_y` `r = 0.260`; per-system sign pattern `[-, +, -, +]`; sign_conflict = true.
- F3: max `|r|` = 0.145; all `|r| < 0.15` → null.
- F4: 14 pairs, `all_assort_z_decline = true`; never-significant = [system_01/annotation_04, system_02/annotation_04, system_03/annotation_06, system_04/annotation_02].
- F5: n = [800, 213, 39]; annotation pairwise e.g. 2_4 ≈ 0.755, 2_5 ≈ −0.740, 4_5 ≈ −0.636.

## Notes on methodology

- Analysis unit is the **system** (200 contiguous nodes); per-system values are the robustness
  check for the pooled value.
- Pearson `r` and t-statistics are computed with `numpy`/`math` (no `scipy`). Two-sided p-values
  use the normal CDF (F1, F3) or a Lanczos/continued-fraction Student-t CDF (F2).
- No imputation, subsampling, or row deletion was performed.

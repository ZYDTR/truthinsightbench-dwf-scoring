# figures notes (TASK ID: figures)

## Method
- Script: `output/analysis/plot_figures.py` (matplotlib, Agg backend, PNG at dpi=150).
- Self-contained; re-runnable via `python3 WS/output/analysis/plot_figures.py`.
- Branches split at U/W drops (control_ratio decreases); order = Jh/U 0.05..0.40.
- Crossings: linear interpolation of gap through zero; black x markers where a sign change exists.
- Topological: per-control ground-state degeneracy (sum of degeneracy of levels within 1e-9 of the minimum) and gap to 2nd distinct level (2nd distinct value minus min); log scale.
- Correlation heatmap: 40x17 grid (U/W 0.1..4.0 x Jh/U 0.0125..0.4); black x marks cells with value < 1e-6.

## Data provenance
- `WS/data/normalized_views/control_vs_gap_case_A.tsv` (dSz=1)
- `WS/data/normalized_views/control_vs_gap_case_B.tsv` (dSz=2)
- `WS/data/normalized_views/control_vs_topological_proxy.tsv` (48 controls)
- `WS/data/normalized_views/control_vs_local_observables.tsv` (49 points)
- `WS/data/normalized_views/control_vs_correlation.tsv` (680 cells)

## Key numbers
| Figure | Path | Bytes | Notes |
|---|---|---|---|
| gap_vs_u | output/figures/gap_vs_u.png | 306983 | 8 branches each panel; 6 crossings/panel (Jh/U 0.25,0.40 no sign change) |
| topological_proxy | output/figures/topological_proxy.png | 72708 | deg 1 -> 2 at U/W 1.4; gap_to_2nd in [8e-6, 1.372] |
| local_observables | output/figures/local_observables.png | 76200 | moment 0.750->1.970; charge 1.000->0.023 |
| correlation_heatmap | output/figures/correlation_heatmap.png | 80581 | 680 cells; 14 critical (<1e-6); range [-3.9e-8, 0.1422] |

## Caveats
- Correlation grid only spans U/W 0.1-4.0 (not the full 0.1-8.0 range of other figures).
- One correlation cell is slightly negative (-3.9e-8); treated as critical (<1e-6).
- Topological degeneracy uses a 1e-9 tolerance for "lowest" level grouping.
- All four PNGs verified non-empty after execution (exit code 0).

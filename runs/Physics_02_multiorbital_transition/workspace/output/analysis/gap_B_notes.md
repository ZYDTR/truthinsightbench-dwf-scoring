# gap_B notes
## Method
- Source: control_vs_gap_case_B.tsv (U/W, gap_like_observable); 8 contiguous branches, Jh/U = 0.05..0.40.
- Split at every U/W drop; per branch: n_points, u_start/end, neg/pos counts, linear-interp zero crossing (null if no sign change), gap max/min + U/W.
- Robustness: crossing interval (u_lo,u_hi) = the two grid points bracketing the sign change.
- Cross-check: re-parse table_008.dat (zip) and compare crossings (rel_diff).
- Shape: monotonic_positive = (min of successive diffs of positive gap) >= 0.
## Provenance
- TSV: /data/user/conversations/conv-af56a8cc-3c22-4b90-8a53-8142e2d7b40e/project/data/normalized_views/control_vs_gap_case_B.tsv
- Raw: table_008.dat extracted via python zipfile (no external data).
## Key numbers
| Jh/U | crossing U/W | interval | gap_max | gap_min | mono_pos |
|---|---|---|---|---|---|
| 0.05 | 2.6402 | [2.6,2.7] | 0.33999 | -0.00219 | True |
| 0.10 | 1.6609 | [1.6,1.7] | 0.37587 | -0.00438 | True |
| 0.15 | 1.3033 | [1.3,1.4] | 0.38601 | -0.00039 | True |
| 0.20 | 1.1109 | [1.1,1.2] | 0.39061 | -0.00148 | True |
| 0.25 | none | n/a | 0.39325 | 0.00116 | True |
| 0.30 | 0.9055 | [0.9,1.0] | 0.39500 | -0.00101 | True |
| 0.35 | 0.8340 | [0.8,0.9] | 0.39629 | -0.00547 | True |
| 0.40 | none | n/a | 0.39729 | 0.00011 | True |
## Caveats
- Finite-size numerical model; crossings are finite-sampling crossovers, not strict phase transitions.
- One-grid-step uncertainty given by the crossing interval (u_lo,u_hi).
- rel_diff vs raw table_008.dat is 0.0: TSV values are identical to the raw table.

# Change Boundaries of Multiple Observables under a Control Parameter

**Task ID:** `Physics_02_multiorbital_transition` · **T0:** 2023-04-21 · **Domain:** physics
**Model:** Anonymous two-component interacting model (finite-size numerical data, not an experimental material).
**Control parameter:** interaction ratio `U/W` (denoted `control_ratio` in the views, `U/W` in the raw tables).
**Secondary parameter:** Hund exchange ratio `Jh/U` (denoted `secondary_interaction_ratio` in `control_vs_correlation.tsv`).
**Data boundary:** only `data/normalized_views/*.tsv` and `data/source_data/all_source_data_deidentified.zip` (tables `table_001.dat`…`table_022.dat`, plus rendered PDFs). No external data, search, or precomputed results were used.

## Data inventory (verified by code)

| File | Role | Shape / structure |
|---|---|---|
| `data/normalized_views/control_vs_correlation.tsv` | inverse correlation length 1/ξ | 680 rows = 40×17 grid of (U/W, Jh/U) |
| `data/normalized_views/control_vs_gap_case_A.tsv` | magnon gap ΔSz=1 (Jh/U branches) | 291 rows, 8 contiguous branches |
| `data/normalized_views/control_vs_gap_case_B.tsv` | magnon gap ΔSz=2 (Jh/U branches) | 291 rows, 8 contiguous branches |
| `data/normalized_views/control_vs_local_observables.tsv` | total magnetic moment, charge fluctuation | 49 rows (U/W 0.0→8.0) |
| `data/normalized_views/control_vs_topological_proxy.tsv` | entanglement spectrum (λ, degeneracy) | 2400 rows = 48 controls × 50 states |
| `data/source_data/all_source_data_deidentified.zip` | raw archives | 22 `.dat` tables + 5 PDFs |

**Analysis unit (per `data_manifest.json`):** one complete control-parameter branch or independent gap scan; spectral entries at one control point are nested observations. A new magnon-gap branch is detected where `control_ratio` decreases; branch file order is `Jh/U = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40`. **Missingness:** none (no control point or branch removed).

## Methods and reproduction

The analysis was executed by delegated general-purpose sub-agents (Workflow `physics02-observable-boundaries`, run `wf_c9e09771-a92`, 7 concurrent agents) plus the main agent's own completed scripts for the two branches that stalled. Every script below is self-contained and re-runnable with `python3`; it reads only the workspace data and writes machine-readable results to `output/results/`.

- **Gap sign-change analysis:** `output/analysis/gap_A_analysis.py` (ΔSz=1) and `output/analysis/gap_B_analysis.py` (ΔSz=2). For each branch: sign-change detection, linear-interpolated crossing `U/W`, min/max gap, and a cross-check against the raw `table_009.dat`/`table_008.dat` blocks.
- **Topological proxy:** `output/analysis/topological_proxy_analysis.py`. Per-control ground-state degeneracy and gap to the second distinct level.
- **Local observables:** `output/analysis/local_observables_analysis.py`. Monotonicity + linear fits + raw cross-check (`table_007.dat`).
- **Correlation 2D + extrapolation:** `output/analysis/correlation_and_extrap.py` (main agent, completed the stalled branches). Critical cells (`|1/ξ|<1e-6`), raw cross-check (`table_012.dat`), and finite-size gap extrapolation (`table_010.dat`/`table_011.dat` vs `table_008.dat`/`table_009.dat`).
- **Figures:** `output/analysis/plot_figures.py` → 4 PNGs in `output/figures/`.

**How to reproduce (end-to-end):**
```
python3 output/analysis/gap_A_analysis.py
python3 output/analysis/gap_B_analysis.py
python3 output/analysis/topological_proxy_analysis.py
python3 output/analysis/local_observables_analysis.py
python3 output/analysis/correlation_and_extrap.py
python3 output/analysis/plot_figures.py
```
All scripts exit 0 and regenerate the CSV/JSON in `output/results/` and the PNGs in `output/figures/`.

**Reproducible method / executed trace:** the scripts above are the executed analysis source; their outputs are the result evidence. The sub-agent execution record is the Workflow run `wf_c9e09771-a92` (journal and per-agent transcripts under the session's `state/claude/projects/…/subagents/workflows/wf_c9e09771-a92/`).

## Findings

### Finding 1 — The magnon gap (both ΔSz=1 and ΔSz=2) shows a sign-change crossover in 6 of 8 `Jh/U` branches, with a `Jh/U`-dependent boundary that is *not* a common onset

- **Object and scope:** magnon gap `ΔSz=1` (`control_vs_gap_case_A.tsv` / `table_009.dat`) and `ΔSz=2` (`control_vs_gap_case_B.tsv` / `table_008.dat`), extrapolated to `L→∞`, across 8 `Jh/U` branches (0.05–0.40) and `U/W` from ~0.8–2.6 up to 8.0.
- **Relationship / boundary:** the gap changes sign (negative → positive) across a finite-sampling crossover interval in 6 of 8 branches. Linear-interpolated crossing `U/W`: ΔSz=1 and ΔSz=2 are nearly coincident (per-branch difference < 0.1%):

| Jh/U | ΔSz=1 cross U/W | ΔSz=2 cross U/W | sign change? |
|---|---|---|---|
| 0.05 | 2.6411 | 2.6402 | yes |
| 0.10 | 1.6623 | 1.6609 | yes |
| 0.15 | 1.3035 | 1.3033 | yes |
| 0.20 | 1.1118 | 1.1109 | yes |
| 0.25 | — | — | **no** (gap ≥ 0) |
| 0.30 | 0.9060 | 0.9055 | yes |
| 0.35 | 0.8367 | 0.8340 | yes |
| 0.40 | — | — | **no** (gap ≥ 0) |

The crossing `U/W` decreases as `Jh/U` increases, but the pattern is **not monotonic in `Jh/U`**: it is absent at `Jh/U = 0.25` and `0.40`. This rules out a "common onset" / "aligned" explanation: the boundary is `Jh/U`-specific and conditionally present.
- **Core quantitative result (analysis unit = one branch):** 6/8 branches with a sign change; crossing `U/W` spans 0.834–2.641.
- **Supporting evidence:**
  - analysis: `output/analysis/gap_A_analysis.py`, `output/analysis/gap_B_analysis.py`
  - results: `output/results/gap_A_branch_summary.csv`, `output/results/gap_A_summary.json`, `output/results/gap_B_branch_summary.csv`, `output/results/gap_B_summary.json`
  - figure: `output/figures/gap_vs_u.png`
- **Robustness:** crossing `U/W` computed independently from the normalized view and from the raw `table_009.dat`/`table_008.dat` blocks agree exactly (`rel_diff = 0.0` in every branch); the two channels (ΔSz=1 vs ΔSz=2) agree to < 0.1% on every crossing. The sign-change interval (one grid step in `U/W`) is reported explicitly, so the one-step finite-sampling uncertainty is bounded.
- **Counterevidence / limitation / alternatives:** a "zero crossing" is a finite-sampling crossover, not a strict phase transition (per the data guide warning). The absence of a crossing at `Jh/U = 0.25` and `0.40` means the boundary is not a single `U/W` value; alternatives (offset, continuously-varying, or grid-indeterminate onset) are all consistent with the data, and the data do not select among them.
- **Executable falsification/replication:** recompute the sign change on a finer `U/W` grid or a different finite-size extrapolation; if the crossover `U/W` shifts by more than one grid step or the sign-change vanishes for the `Jh/U = 0.25`/`0.40` branches, the boundary location is grid-dependent rather than physical.

### Finding 2 — The entanglement-spectrum topological proxy flips its ground-state degeneracy at `U/W ≈ 1.4`, a boundary *distinct* from the magnon-gap crossover

- **Object and scope:** entanglement spectrum (50 levels per control) in `control_vs_topological_proxy.tsv`, `U/W = 0.1–8.0`.
- **Relationship / boundary:** ground-state degeneracy (sum of degeneracy over levels within 1e-9 of the lowest level) is **1** for `U/W ≤ 1.3` and **2** for `U/W ≥ 1.4`. The gap to the second distinct level collapses below 1e-2 at `U/W ≥ 1.4` and below 1e-3 for `U/W ≥ 2.1`. The lowest level value decreases smoothly from 1.245 (`U/W=0.1`) to 0.738 (`U/W=8.0`).
- **Core quantitative result (analysis unit = one control point):** degeneracy flip at `U/W = 1.4` (first deg-2 control); 34 of 48 controls are deg-2.
- **Supporting evidence:**
  - analysis: `output/analysis/topological_proxy_analysis.py`
  - results: `output/results/topological_proxy_per_control.csv`, `output/results/topological_proxy_summary.json`
  - figure: `output/figures/topological_proxy.png`
- **Robustness:** degeneracy is computed with a 1e-9 tolerance around the minimum level and cross-checked against the explicit `degeneracy` column; the flip is a clean 1→2 step with no intermediate values.
- **Counterevidence / limitation / alternatives:** the flip is a finite-sampling crossover of the *discrete* spectrum, not a proven thermodynamic topological change; the 50-level truncation could in principle hide a higher degeneracy.
- **Executable falsification/replication:** recompute degeneracy on the full untruncated spectrum or with a tighter/looser tolerance; if the 1→2 flip moves or splits, it is a finite-size artifact rather than a stable boundary.

### Finding 3 — Local observables (total magnetic moment, charge fluctuation) show no boundary: smooth, monotonic variation with `U/W` (null result)

- **Object and scope:** `control_vs_local_observables.tsv`, `U/W = 0.0–8.0`, 49 points.
- **Relationship / boundary:** **none.** Total magnetic moment increases monotonically from 0.750 to 1.970; charge fluctuation decreases monotonically from 0.99998 to 0.0234. No sign change, no discontinuity, no critical point.
- **Core quantitative result (analysis unit = one point):** linear fits — magnetic moment: slope 0.1686, R² 0.718; charge fluctuation: slope −0.1215, R² 0.699.
- **Supporting evidence:**
  - analysis: `output/analysis/local_observables_analysis.py`
  - results: `output/results/local_observables_summary.csv`, `output/results/local_observables_summary.json`
  - figure: `output/figures/local_observables.png`
- **Robustness:** monotonicity verified with zero violations of successive differences; raw cross-check against `table_007.dat` gives max relative difference 0.0 for the magnetic moment.
- **Counterevidence / limitation / alternatives:** the R² ≈ 0.7 (not 1) indicates a weakly curved trend, but no non-monotonic feature or boundary is present; the null is the robust conclusion.
- **Executable falsification/replication:** scan a finer `U/W` grid or a different local observable; a sign change or a peak would refute the null.

### Finding 4 — Inverse-correlation-length scan has 14 scattered "critical" cells (`1/ξ ≈ 0`), with no common onset; perfect agreement with the raw archive

- **Object and scope:** 2D scan `control_vs_correlation.tsv` (40 `U/W` × 17 `Jh/U`), plus raw `table_012.dat` (`L=60`).
- **Relationship / boundary:** 14 of 680 cells have `|1/ξ| < 1e-6` (i.e. correlation length effectively infinite), clustered at (a) `U/W ∈ [0.6, 0.9]` with `Jh/U ∈ [0.2, 0.375]` and (b) `U/W ∈ [3.5, 3.8]` with `Jh/U ∈ [0.025, 0.05]`. These are **scattered in two isolated islands**, not a common onset or a single `U/W` line.
- **Core quantitative result (analysis unit = one (U/W, Jh/U) cell):** 14 critical cells across 8 distinct `U/W` values.
- **Supporting evidence:**
  - analysis: `output/analysis/correlation_and_extrap.py`
  - results: `output/results/correlation_per_control.csv`, `output/results/correlation_critical_cells.csv`, `output/results/correlation_summary.json`
  - figure: `output/figures/correlation_heatmap.png`
- **Robustness:** the 680-cell grid matches the raw `table_012.dat` exactly — 680/680 overlap, max relative difference 0.0 — so the critical cells are a property of the data, not a parsing artifact.
- **Counterevidence / limitation / alternatives:** the cells include small negative values (e.g. −3.9e-8), indicating finite-size sign noise around zero rather than a clean divergence; the "infinite correlation length" is thus a finite-sampling indicator, not a proven long-range-ordered phase.
- **Executable falsification/replication:** recompute on a larger `L` (raw `table_014.dat` has multiple `L`) or a finer grid; if the critical cells merge into a single line or vanish, the current pattern is a finite-size artifact.

### Finding 5 — The simple `1/L` linear extrapolation does *not* reproduce the source tables' "L→∞ extrapolated" gaps (applicability-boundary / unresolved-interpretation finding)

- **Object and scope:** finite-size gap data `table_010.dat` (ΔSz=2) and `table_011.dat` (ΔSz=1), one block per `U/W` at `Jh/U=0.25`, vs the source tables' own "extrapolated to L→∞" values in `table_008.dat`/`table_009.dat`.
- **Relationship / boundary:** fitting `gap = a + b·(1/L)` gives an intercept `a(U/W)` that is the *naive* `L→∞` value. For ΔSz=1 at `Jh/U=0.25`, `a(1.0) = 3.748` but the source's "extrapolated" gap is 0.000463 — a relative difference of ~8100 (max 8103 across `U/W`). The fit is poor (R² < 0), and the finite-size gap *decreases* with `L` (e.g. 0.338 at `1/L=0.1` → 0.00157 at `1/L=0.01`), so a linear model is inadequate.
- **Core quantitative result (analysis unit = one U/W block):** 61 matched pairs; max relative difference ΔSz=1 = 8103, ΔSz=2 = 6854.
- **Supporting evidence:**
  - analysis: `output/analysis/correlation_and_extrap.py`
  - results: `output/results/gap_extrapolation_fit.csv`, `output/results/gap_extrapolation_check.csv`, `output/results/gap_extrapolation_check.json`
- **Robustness:** the check is symmetric in the two channels and uses the source's own `Jh/U=0.25` branch as reference; the discrepancy is large in both, so it is not a single-channel artifact.
- **Counterevidence / limitation / alternatives:** this does *not* refute the existence of the magnon-gap sign change in Finding 1; it only shows that the "extrapolated to L→∞" numbers in `table_008/009` were produced by a non-linear extrapolation (or a different model), so the linear `1/L` intercept should not be taken as the thermodynamic gap. It is an applicability boundary on how to read the source tables.
- **Executable falsification/replication:** fit alternative extrapolation forms (e.g. `1/L + 1/L²`, or the functional form the source used) and compare to `table_008/009`; if a non-linear form recovers them, the linear model is simply the wrong one; if none do, the source extrapolation is an unresolved model assumption.

## Limitations and scope

1. **Finite-size numerical data, not experiment.** All "boundaries" are finite-sampling crossovers; a numerical zero crossing does not establish a strict phase transition (per the data guide).
2. **One `Jh/U` reference for the topological/local/correlation observables** (the normalized views collapse `Jh/U` into the grid for correlation only; local and topological views are single-branch).
3. **Extrapolation ambiguity.** The source tables' "L→∞ extrapolated" gaps are not reproducible by a linear `1/L` fit (Finding 5); interpret them with that caveat.
4. **No external literature used.** All conclusions rest solely on the workspace data.

## Summary of boundary structure

The data do **not** support a single common onset or ordering across observables:

- Magnon gap (ΔSz=1, ΔSz=2): sign-change crossover at `U/W ≈ 0.83–2.64`, `Jh/U`-specific, absent at `Jh/U = 0.25, 0.40`.
- Entanglement spectrum: degeneracy flip at `U/W ≈ 1.4`.
- Local observables: no boundary (smooth monotonic).
- Inverse correlation length: two isolated islands of `1/ξ ≈ 0` cells.
- These are **offset and conditionally-present**, consistent with the "offset / continuously-varying / grid-indeterminate" explanation class and inconsistent with a single aligned onset.

## Sources

No external sources were used (T0 boundary not needed). All evidence is from the workspace data files listed in the data inventory.

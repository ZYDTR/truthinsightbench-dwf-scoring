# Binary superlattice (C + H2) structural analysis — authoritative report

**Session code:** conv-d944711d-fa11-472b-99db-a1344a7c93b2
**Scope:** 14 low-energy binary nanoparticle configurations (types `C` and `H2`) plus a 6:6 (6 CH4 + 6 H2) accepted-local-minimum trajectory. All evidence is workspace data under `output/`; no external sources were used.

## Finding 1: Equal-size (R1==R2==3) interfacial preference Chi positively correlates with same-type nearest-neighbor clustering

- **Object and scope:** The four equal-size structures `structure_01`–`structure_04` (R1=R2=3.0, 9 C + 9 H2, n=18). Analysis unit: one global low-energy configuration per Chi value.
- **Relationship / direction / boundary:** Positive correlation. Larger Chi (stronger unlike-pair repulsion) yields a higher same-type nearest-neighbor (NN) pair fraction, i.e. more same-type clustering / demixing. The relationship is monotonically *increasing* in Chi (Spearman rho = +1.0 over 4 points — descriptive, not a statistical claim).
- **Core quantitative result:** At the first-peak cutoff c = 6.25 Å, the same-type NN pair fraction is: Chi 0.4 → 0.359, 0.6 → 0.4898 (~0.490), 0.8 → 0.6038 (~0.604), 1.0 → 0.875. Strictly increasing.
- **Analysis script(s):** `output/analysis/analyze_neighbors.py` (same-type vs cross-type pair counts at the first-peak cutoff and its ±20% and multiplier variants) and `output/analysis/analyze_structures.py`.
- **Supporting evidence:** `output/results/robustness_chi_clustering.csv`, `output/results/neighbor_analysis.csv`, `output/results/figure_data.csv`, `output/figures/chi_vs_samefrac.svg`.
- **Robustness result:** The trend is unchanged at the +20% cutoff (same_frac_+20 = 0.359 / 0.4898 / 0.6038 / 0.875 in `output/results/neighbor_analysis.csv`). At −20% (5.0 Å) all equal-size pair counts are zero (degenerate), because the minimum pair distance (~6.12 Å) exceeds 5.0 Å. Across the multiplier cutoffs in `output/results/robustness_chi_clustering.csv`, the increasing trend holds at c, 1.1c, 1.3c and 2.0c, but is *non-monotonic* at 1.6c and degenerate (no pairs) at 0.9c. Robustness is therefore partial, not universal.
- **Counterevidence / limitation / alternative explanation:** The mixed-size subset (R2 ≠ 3) shows no clean directional Chi→clustering trend (grouped means by own-cutoff: Chi 0.2→1.000, 0.4→0.600, 0.6→0.500, 0.8→0.830; non-monotonic, Spearman ≈ −0.4), confounded by varying R2, particle count and cutoff. With only 4 equal-size points, "monotonic" is descriptive. Alternative explanation: larger Chi simply demixes the mixture, so same-type clustering is a direct consequence of demixing rather than an independent structural property.
- **Falsifiable new observation / replication:** Generate an additional equal-size configuration (R1=R2=3.0, 9 C + 9 H2) with a Chi between existing values (e.g. Chi=0.5 or 0.9) using the same optimizer. If its same-type NN fraction at c=6.25 Å does not lie between the adjacent values (breaking monotonicity), the positive monotonic relationship is falsified. Also test Chi outside [0.4, 1.0].

## Finding 2: The 6:6 binary trajectory converges to a stable local minimum

- **Object and scope:** The accepted-local-minimum trajectory for the 6:6 system: 34 frames (0–33), 12 particles per frame, composition confirmed 6 CH4 + 6 H2 (= 6:6) in every frame.
- **Relationship / direction / boundary:** Energy descends from an initial positive value to a negative global minimum that is reached at the final frame. The settled state is a local minimum exhibiting dynamical oscillation in the tail, not a single frozen point.
- **Core quantitative result:** frame-0 energy = 6.95026; global minimum = −4.15068 at frame 33, which is the final frame (final == min). Only **2** frames (frames 28 and 33) lie within 0.05 of the minimum. From ~frame 14 onward the energy oscillates with amplitude ≈ 0.2107 about the minimum (band ≈ −4.1507 to −3.9007). Converged-frame packing: characteristic NN first peak = 5.95 Å, mean NN = 6.1108 Å, z-layering = 2 shells (CH4 ≈ z=−2.1, H2 ≈ z=+2.1).
- **Analysis script(s):** `output/analysis/analyze_structures.py` (parses the 6:6 trajectory energies/movie and writes `output/results/trajectory_summary.csv`); the trajectory note and per-frame CSV were written by `output/analysis/analyze_trajectory.py`.
- **Supporting evidence:** `output/results/trajectory_analysis.csv`, `output/results/trajectory_summary.json`, `output/results/trajectory_summary.csv`, `output/results/trajectory_note.txt`.
- **Robustness result:** The 36-block .xyz movie contains the last two frames as duplicates of frame 33 (identical energy −4.15068); they are counted as repeats, not new independent frames. The 6:6 composition is invariant across all 34 frames.
- **Counterevidence / limitation / alternative explanation:** Convergence is judged by accepted energy only; no per-frame force or step-size is stored. The endpoint is a minimum of a stochastic (basin-hopping / annealing-like) search, not a zero-temperature ground-state proof. The tail oscillation explicitly means it is not a frozen point.
- **Falsifiable new observation / replication:** Continue the trajectory beyond frame 33. If the energy drops below −4.15068 (a deeper minimum) or the 6:6 composition changes, the "stable local minimum at −4.15068" claim is falsified. Re-run the same optimization from a different initial condition; if it converges to a lower energy, then −4.15068 is not the basin's ground state.

## Finding 3: Energy is NOT a clean monotonic function of Chi alone (honest null / boundary)

- **Object and scope:** All 14 structures (n = 17–37 particles; Chi = 0.2–1.0; epsilon = 0.02–0.16; R2 = 1.0–3.0). Analysis unit: one low-energy configuration per parameter combination, with energy normalized per particle.
- **Relationship / direction / boundary:** No clean single-parameter monotonic Chi→energy relationship exists; energy is confounded by epsilon (well strength) and by particle number/composition.
- **Core quantitative result:** The most bound structure is `structure_13` (Chi=0.6, eps=0.16, R2=2.0, n=37) at −0.530 per particle. The least bound is `structure_12` (Chi=0.8, eps=0.02, R2=2.0, n=32) at −0.077 per particle, with `structure_01` (Chi=1.0, eps=0.03, R2=3.0, n=18) at −0.080 per particle essentially tied. Thus a high-Chi structure (`structure_01`, Chi=1.0) is among the *least* bound, while a mid-Chi structure (`structure_13`, Chi=0.6) is the *most* bound — the opposite of any monotonic Chi→energy expectation. (Note: the evidence files place the least-bound structure at `structure_12` = −0.077, not `structure_01`; `structure_01` is the next-weakest at −0.080.)
- **Analysis script(s):** `output/analysis/analyze_structures.py` (per-structure energy, energy_per_particle, composition, and geometry).
- **Supporting evidence:** `output/results/structure_features.csv`, `output/results/structure_features.json`.
- **Robustness result:** The null holds across the full 14-structure set and is not an artifact of one cutoff or subset; the confounders (epsilon, n) vary independently of Chi.
- **Counterevidence / limitation / alternative explanation:** Even within the equal-size subset (01–04, fixed R2=3, n=18), energy_per_particle varies with Chi (0.4→−0.447, 0.6→−0.366, 0.8→−0.157, 1.0→−0.080), but epsilon also varies (0.04, 0.08, 0.04, 0.03), so Chi remains confounded with epsilon. A single-parameter Chi→energy relationship is therefore not supported; only the Chi→clustering (Finding 1) and trajectory-convergence (Finding 2) relationships are clean.
- **Falsifiable new observation / replication:** Generate structures with Chi varied while holding epsilon and particle number fixed (e.g. R1=R2=3.0, 9 C + 9 H2, eps=0.04, Chi ∈ {0.4,0.5,0.6,0.7,0.8,0.9,1.0}). If energy_per_particle becomes a clean monotonic function of Chi, this null (Finding 3) is falsified.

## Finding 4: Particle-size ratio (R2/R1) changes the characteristic bond/packing distance

- **Object and scope:** All 14 structures; R2/R1 ranges from 0.3333 (R2=1.0) to 1.0 (R2=3.0). Analysis unit: characteristic NN distance = first peak of the 1D all-pair distance histogram (0.5 Å bins).
- **Relationship / direction / boundary:** Characteristic NN (bond/packing) distance varies with particle size ratio; size sets the packing length scale.
- **Core quantitative result:** First-peak (characteristic NN) distances range from 5.25 Å (`structure_06`, R2=2.0; `structure_13`, R2=2.0) to 13.75 Å (`structure_05`, R2=2.5), with the equal-size R2=3 structures clustering at 6.25 Å.
- **Analysis script(s):** `output/analysis/analyze_neighbors.py` (first_peak column) and `output/analysis/analyze_structures.py` (nn_dist_mean).
- **Supporting evidence:** `output/results/neighbor_analysis.csv` (first_peak column), `output/results/structure_features.csv` (nn_dist_mean).
- **Robustness result:** The first-peak distance is derived per-structure from the distance histogram and is internally consistent: the equal-size 6.25 Å value is reproduced in `output/results/neighbor_analysis.csv` and is close to the converged trajectory final-frame first peak of 5.95 Å (`output/results/trajectory_analysis.csv`).
- **Counterevidence / limitation / alternative explanation:** This is a descriptive correlation across only 14 structures; Chi and epsilon also vary and could contribute. It is not a controlled single-variable study.
- **Falsifiable new observation / replication:** Vary R2/R1 while holding Chi, epsilon and composition fixed (e.g. Chi=0.6, eps=0.02, 9 C + 9 H2, R2 ∈ {1.0,1.5,2.0,2.5,3.0}). If the characteristic NN distance does not shift with size ratio, the claim is falsified.

## Methods / Analysis workflow

Executed scripts (analysis unit: one global low-energy configuration per parameter combination; one trajectory per 6:6 system):

- `output/analysis/analyze_structures.py` — parses each `.xyz`, computes periodic pairwise distances (box from the header `(p)` value), z-layering by gap clustering, in-plane NN metrics, and writes `output/results/structure_features.csv/json`; also writes `output/results/trajectory_summary.csv`.
- `output/analysis/analyze_neighbors.py` — computes the first distance peak (characteristic NN distance) and same-type vs cross-type pair counts at ±20% of that peak; writes `output/results/neighbor_analysis.csv`.
- `output/analysis/test_parse.py` — parsing sanity check.
- `output/analysis/analyze_trajectory.py` — parses the 6:6 trajectory energies/movie and writes `output/results/trajectory_analysis.csv`, `output/results/trajectory_summary.json`, and `output/results/trajectory_note.txt`.
- `output/analysis/analyze_robustness.py` — computes same-type NN pair fractions across multiplier cutoffs {c, 0.9c, 1.1c, 1.3c, 1.6c, 2.0c} (periodic (p)-box x,y, z free; first-peak c) and writes `output/results/robustness_chi_clustering.csv` and `output/results/robustness_note.txt`.
- `output/analysis/build_chi_figure.py` — hand-built SVG (matplotlib unavailable) of equal-size same-type fraction vs Chi; writes `output/figures/chi_vs_samefrac.svg` and `output/results/figure_data.csv`.

## Reproducibility

Run from the project root (working directory `/data/user/conversations/conv-d944711d-fa11-472b-99db-a1344a7c93b2/project`):

```bash
python3 output/analysis/analyze_structures.py
python3 output/analysis/analyze_neighbors.py
python3 output/analysis/analyze_trajectory.py
python3 output/analysis/analyze_robustness.py
python3 output/analysis/build_chi_figure.py
python3 validate_output.py output
```

## Limitations

- Only 14 static low-energy configurations; "monotonic" and trend statements over 4 equal-size points are descriptive, not statistical.
- Convergence is judged by accepted energy only; no forces or step sizes are stored, and the trajectory endpoint is a local minimum of a stochastic search, not a proven ground state.
- Chi→energy (Finding 3) is confounded by epsilon and particle count; no controlled single-variable study was possible.
- The −20% cutoff is degenerate (empty) for the equal-size structures because the minimum pair distance (~6.12 Å) exceeds 5.0 Å; the 1.6c multiplier cutoff is non-monotonic.
- Simple global descriptors (first-peak distance, z-layering) are used; no force field or per-frame error bars.

## References / Sources

No external sources were used. All evidence is workspace data (T0 = 2022-01-01 boundary): the 14 `.xyz` structures, `structure_manifest.json`, the 6:6 trajectory files, and the derived artifacts under `output/` listed above. This satisfies the external-source policy.

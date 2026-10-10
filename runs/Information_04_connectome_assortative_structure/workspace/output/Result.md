# Assortative Structure and Spatial Boundaries in Annotated Networks

**Task ID:** `Information_04_connectome_assortative_structure` · **T0:** 2023-01-01
**Analysis unit:** one network system (a block of 200 contiguous `region_index`) or a complete node profile. Nodes within a system and repeated annotations are dependent observations. Four systems (800 nodes) carry the full `network_node_summary` + `network_profile_a` set; `network_profile_b` (213) and `network_profile_c` (39) are smaller complementary profiles; `trimming_sensitivity` covers 4 systems × 14 annotation pairs × 2 result kinds.

**Method summary.** All statistics were computed with a self-contained `pandas`/`numpy` pipeline (no `scipy`; p-values via the normal CDF or a Lanczos/continued-fraction Student-t CDF). Pearson `r`, `t`, and two-sided `p` were computed per system and overall, with each system (200 nodes) treated as one independent analysis unit for robustness. No external data were used; the bundled `data/` tables are the complete evidentiary set. See the "Reproduction" section and `reproduce.md` for exact commands.

Below are the registered findings. Each is linked to a real analysis script and a real machine-readable result artifact under `output/`.

---

## Finding 1: Node strength and mean connection distance are strongly and robustly positively associated, within every system

**Object and scope.** All 800 nodes across 4 network systems (200 contiguous nodes each), measured by `node_strength` and `mean_connection_distance` in `network_node_summary.tsv`.

**Relationship, direction, boundary.** Positive, monotonic association: stronger nodes have longer mean connection distance. Direction is positive in all four systems, so the effect is a whole-dataset property, not an artifact of one block.

**Core quantitative result (analysis unit = network system / node).** Overall Pearson `r = 0.648` (t = 24.05, p < 1e-300, n = 800). Per-system `r`: system 0 = 0.672, system 1 = 0.656, system 2 = 0.565, system 3 = 0.643. Every per-system `r` is positive and > 0.5, so a complete-system leave-out / per-system check confirms robustness.

**Supporting analysis and result evidence.** Analysis source `f1_strength_distance.py`; result evidence `f1_strength_distance_result.json` (and the consolidated `final_results.json` → `f1_strength_distance`). Supporting figure: `figures.png`.

**Robustness result.** Per-system decomposition (leave-out of each system) preserves the sign and magnitude; `robust_same_direction = true`. The association is not driven by a single system.

**Counterevidence, limitation, alternative explanations.** The data are node-level summaries, so this is an association, not a causal or developmental claim (per the data-guide warning). A plausible alternative is that "strength" and "connection distance" are two facets of the same underlying topology (e.g., hub nodes reaching across the network), which is consistent with the positive correlation but not testable from these summary variables alone. No negative or conflicting per-system result was found.

**Executable falsification/replication.** Recompute Pearson `r` between `node_strength` and `mean_connection_distance` on an independent set of nodes or an alternative distance definition (e.g., a trimmed or spatial-geodesic distance). A null or reversed association in a hold-out system would falsify the positive boundary.

---

## Finding 2: The spatial (coord_y) gradient on strength is real only overall, but is system-dependent and sign-inconsistent across systems

**Object and scope.** All 800 nodes with 3D coordinates in `network_profile_a.tsv`, joined to `node_strength`/`mean_connection_distance` in `network_node_summary.tsv`; 4 systems of 200.

**Relationship, direction, boundary.** Overall, strength increases with `coord_y` (`r = 0.260`, p ≈ 3e-14). But this is **not** a uniform spatial boundary: the per-system sign pattern is `−, +, −, +` (system 0 = −0.127, system 1 = +0.459, system 2 = −0.124, system 3 = +0.341). Two systems show a positive gradient, two show a negative one. The same sign conflict appears for `distance vs coord_y` (−0.214, +0.336, −0.229, +0.179).

**Core quantitative result (analysis unit = system).** Overall strength–coord_y `r = 0.260` (p ≈ 3e-14). Per-system `r`: −0.127 (p ≈ 0.07), +0.459 (p ≈ 3e-13), −0.124 (p ≈ 0.08), +0.341 (p ≈ 3e-7). `sign_conflict = true`.

**Supporting analysis and result evidence.** Analysis source `f2_spatial_gradient.py`; result evidence `f2_spatial_gradient_result.json` (and `final_results.json` → `f2_spatial_gradient`).

**Robustness result.** The overall effect survives, but it is *not* robust to system leave-out: it reverses sign in systems 0 and 2. Quadrant means (by sign of `coord_x`, `coord_y`) also show a split: strength is higher in the `coord_y ≥ 0` quadrant (≈15.2–15.7) than the `coord_y < 0` quadrant (≈11.4–11.6), while `coord_x` splits are negligible — so the gradient is genuinely in `coord_y`, not `coord_x`.

**Counterevidence, limitation, alternative explanations.** The per-system sign conflict is the key counterevidence: a single pooled `coord_y` gradient is an aggregate that masks system-level heterogeneity. Alternative explanations include (a) `coord_y` being a proxy for a per-system spatial axis that is not aligned across systems, and (b) the overall positive `r` being driven mainly by the large positive gradients in systems 1 and 3. This is exactly the "test spatial-distance sensitivity" warning in the data guide.

**Executable falsification/replication.** Fit the strength–`coord_y` slope separately in each system and test for heterogeneity (e.g., a mixed-effects slope by system). A set of systems with consistently positive slopes (or a non-significant heterogeneity term) would validate a uniform spatial boundary; the observed alternating signs already show the boundary is system-specific.

---

## Finding 3: Mean homophily is unassociated with node strength, distance, coordinates, and all five annotations (null effect)

**Object and scope.** All 800 nodes; `mean_homophily` (from `network_node_summary.tsv`) tested against `node_strength`, `mean_connection_distance`, `coord_x/y/z`, and `annotation_1..5` (from `network_profile_a.tsv`).

**Relationship, direction, boundary.** Null: no meaningful correlation. Homophily is not an indicator of strength, distance, or any of the coordinate/annotation profiles in this dataset.

**Core quantitative result (analysis unit = node, all 800).** Maximum `|r|` across all 10 tested associations is 0.145 (homophily vs `node_strength`, p ≈ 3e-5). Every `|r| < 0.15` (`all_null_lt_0.15 = true`). The strongest homophily–annotation link is `annotation_1` (`r = 0.135`, p ≈ 1e-4), still weak.

**Supporting analysis and result evidence.** Analysis source `f3_homophily_null.py`; result evidence `f3_homophily_null_result.json` (and `final_results.json` → `f3_homophily_null`).

**Robustness result.** The null is robust: no single tested association exceeds `|r| = 0.15`, and the per-system pattern is not driven by any one block (the per-system `r` values are all small and mixed in sign).

**Counterevidence, limitation, alternative explanations.** The one borderline association (homophily–strength, `r = 0.145`) is the only potential counterevidence and is weak. A limitation is that homophily is a single summary scalar, so a more granular local assortativity measure could in principle reveal structure this aggregate misses.

**Executable falsification/replication.** Compute homophily from the raw edge list (if available) at multiple scales, or recompute with per-system homophily and a stronger annotation definition; a consistent `|r| > 0.3` across systems would falsify the null.

---

## Finding 4: Trimming monotonically reduces the assortative z-score in every system–annotation pair, but FDR significance is patchy

**Object and scope.** `trimming_sensitivity.tsv`: 4 systems × 14 annotation pairs, each with `assort_z` and `assort_p_fdr` across 19 trim levels (`trim_percent_1..19`).

**Relationship, direction, boundary.** `assort_z` declines monotonically from trim level 1 to 19 in all 14 pairs (`all_assort_z_decline = true`), i.e., trimming removes assortative structure. However the FDR-adjusted p-values (`assort_p_fdr`) are not uniformly significant: 4 of 14 pairs are never significant at any trim level.

**Core quantitative result (analysis unit = system–annotation pair).** 14 pairs; `all_assort_z_decline = true`. Never-significant pairs: `system_01/annotation_04`, `system_02/annotation_04`, `system_03/annotation_06`, `system_04/annotation_02`. The remaining 10 pairs reach `p_fdr < 0.05` at intermediate trim levels (e.g., `system_01/annotation_01` is significant from trim 4 onward; `system_03/annotation_03` is significant at low trims).

**Supporting analysis and result evidence.** Analysis source `f4_trimming.py`; result evidence `f4_trimming_result.json` (and `final_results.json` → `f4_trimming`).

**Robustness result.** The monotonic decline of `assort_z` is robust across all 14 pairs (a complete-pair check). The FDR significance, by contrast, is *not* robust: 4/14 pairs never cross the 0.05 threshold, so the trimming effect is a consistent directional signal but only partially significant.

**Counterevidence, limitation, alternative explanations.** The 4 never-significant pairs (`system_01/04`, `system_02/04`, `system_03/06`, `system_04/02`) are the counterevidence: the trimming effect is real in direction but the statistical evidence is uneven. A limitation is that `trim_percent` is a single precomputed sensitivity axis; the mechanism of *why* trimming reduces assortativity is not identifiable from these tables alone.

**Executable falsification/replication.** Recompute assortativity on the raw edges after physically removing a defined fraction of nodes/edges and compare `z`-trajectories; a non-monotonic or reversed trajectory in any pair would falsify the monotonic-decline boundary.

---

## Finding 5: The three profile tables live in different coordinate spaces and scale, and the annotation profile is internally structured (moderately correlated annotations)

**Object and scope.** `network_profile_a.tsv` (800 nodes, 5 annotations), `network_profile_b.tsv` (213 nodes, 1 annotation), `network_profile_c.tsv` (39 nodes, 3 annotations).

**Relationship, direction, boundary.** The profiles are not on a common coordinate scale: profile a has `coord_x` ≈ −63..+64, profile b has `coord_x` ≈ 1175..12850 (an order of magnitude larger), and profile c has `coord_x` ≈ −25..−2. The annotation set in profile a is internally structured — e.g., `annotation_2 ↔ annotation_4` `r ≈ 0.755`, `annotation_2 ↔ annotation_5` `r ≈ −0.740`, `annotation_4 ↔ annotation_5` `r ≈ −0.636` — so annotations are not independent.

**Core quantitative result (analysis unit = profile / node).** `n_nodes`: a = 800, b = 213, c = 39. `annotation_pairwise` correlations include `1_2 ≈ 0.604`, `2_4 ≈ 0.755`, `2_5 ≈ −0.740`, `4_5 ≈ −0.636`, `3_5 ≈ 0.547`.

**Supporting analysis and result evidence.** Analysis source `f5_profiles_annotations.py`; result evidence `f5_profiles_annotations_result.json` (and `final_results.json` → `f5_profiles_annotations`).

**Robustness result.** Coordinate ranges are computed directly over the full profile (no subsampling), so the scale differences are a property of the tables, not sampling.

**Counterevidence, limitation, alternative explanations.** The large scale gap between b/c and a means the profiles likely describe different systems or different coordinate frames; they should not be pooled. A limitation is that profiles b and c are small (213, 39), so per-node correlations within them are underpowered.

**Executable falsification/replication.** Overlay profiles b/c onto profile a after a coordinate transform and test whether the node identities or annotation values align; a consistent alignment (e.g., a shared affine transform) would validate a common latent space, and its absence would confirm separate coordinate frames.

---

## Overall interpretation

The data support a **robust positive strength–distance association** (Finding 1) and a **null homophily effect** (Finding 3) across the whole 800-node set. The **spatial `coord_y` gradient** (Finding 2) and the **trimming effect** (Finding 4) are real but **conditional**: both show system-level heterogeneity (sign conflicts in Finding 2; patchy FDR significance in Finding 4). These conditionality findings are themselves the main scientific results: the dataset is best described as having a *system-dependent* spatial structure and a *robust-but-uneven-significant* trimming sensitivity. The profile/annotation tables (Finding 5) are descriptive context confirming different coordinate frames and a structured annotation space.

All conclusions are bounded by the data-guide warnings: node-level associations do not establish causal or developmental mechanism, and inferences respect network-level clustering and trimming/spatial-distance sensitivity.

## Reproduction

Run the following from `output/` (Python 3, `pandas`, `numpy`; `matplotlib` optional for the figure). Each script is self-contained and reads only from `data/`.

```bash
# Per-dimension analyses (each writes its own *_result.json)
python3 f1_strength_distance.py
python3 f2_spatial_gradient.py
python3 f3_homophily_null.py
python3 f4_trimming.py
python3 f5_profiles_annotations.py

# Supporting figure (optional; needs matplotlib)
python3 make_figure.py        # -> figures.png

# Consolidate all five artifacts into final_results.json
python3 aggregate.py          # -> final_results.json
```

The consolidated `final_results.json` is the single machine-readable summary; each `f*_result.json` is the per-finding machine-readable evidence. See `reproduce.md` for a step-by-step walkthrough and the exact output of each script.

## Limitations

- Node-level summary statistics only; no raw edge list, so causal/developmental claims are out of scope.
- Profiles b (213) and c (39) are small; per-node statistics on them are underpowered.
- `trim_percent` is a precomputed sensitivity axis; the trimming mechanism is not identifiable.
- The spatial `coord_y` gradient is system-dependent (Finding 2); any pooled statement must be qualified accordingly.

## No external sources used

Per the T0 boundary and the data-guide, no external literature or datasets were used; the bundled `data/` tables are the complete evidentiary set. All numbers above are computed directly from them.

# Accuracy and Applicability Boundaries of Classical Representations for Variational Quantum Circuits

**Task ID:** `Physics_05_variational_circuit_simulation` · **T0:** 2020-03-30
**Analysis unit (default):** one graph×γ case (action A) or one pipeline case (action B), one complete state comparison.

## Scope and data

The only direct evidence is the bundled file `data/source_data/computational_observations.json`
(sha256 `8512f362e9db0b199fdc3fafa12886656f01baad65684aadf84d4d32fce9fca8`, 9634 bytes),
verified against the manifest in `results/integrity_receipt.json` (`hash_match: true`,
`bytes_match: true`, `manifest_task_id: Physics_05_variational_circuit_simulation`,
`manifest_t0: 2020-03-30`). It contains two actions:

- **A `rbm_exact_cost_layer`** — 12 cases = 4 graph types
  (`cycle_5`, `complete_4`, `random_regular_3_6_seed123`, `barbell_3_0`) × 3 γ values
  (−0.63, 0.23, 0.71). Each case records `exact_state_fidelity` of the classical
  representation against the exact state vector, plus the defect
  `abs(fidelity − 1.0)`.
- **B `rbm_p2_multigraph_pipeline`** — 3 cases (6-node, degree-3 random graphs, seeds
  12345 / 24680 / 97531), with p1/p2 final-state fidelity, per-gate terminal fidelity,
  compression fidelity, QAOA optimizer step counts, initial/optimized angles, and a frozen
  threshold set inherited from a prior single-graph run.

No external data or literature were used. All numbers below are recomputed from the raw
per-case fields in `output/analysis/*.py` and stored in `output/results/`.

## Findings

### Finding 1 — Cost-layer classical representation is exact to floating-point precision

**Object and scope:** the classical representation at the cost-layer stage (action A),
all 4 graph types × 3 γ values = 12 cases. Graphs span 4–6 nodes; `rbm_hidden_units_after_cost_layer`
ranges from 6 to 10.

**Relationship / direction / boundary:** the classical representation agrees with the exact
state vector to floating-point precision. Every `exact_state_fidelity` lies in the interval
[0.9999999999999996, 1.0000000000000022]; the defect from 1.0 is at most
2.220446049250313e-15 (10× the double-precision ULP at 1.0, which is
2.220446049250313e-16). This is a small, nonzero floating-point round-off magnitude, not a
model error. The recomputed global minimum (0.9999999999999996) exactly equals the declared
`observed_minimum_exact_state_fidelity`.

**Core quantitative result and analysis unit (one graph×γ case):**

| graph | γ | exact_state_fidelity | defect |
|---|---|---|---|
| cycle_5 | −0.63 | 1.0000000000000009 | 8.881784197001252e-16 |
| cycle_5 | 0.23 | 1.0000000000000004 | 4.440892098500626e-16 |
| cycle_5 | 0.71 | 1.0000000000000004 | 4.440892098500626e-16 |
| complete_4 | −0.63 | 1.0 | 0.0 |
| complete_4 | 0.23 | 1.0000000000000004 | 4.440892098500626e-16 |
| complete_4 | 0.71 | 0.9999999999999998 | 2.220446049250313e-16 |
| random_regular_3_6_seed123 | −0.63 | 1.0000000000000004 | 4.440892098500626e-16 |
| random_regular_3_6_seed123 | 0.23 | 0.9999999999999996 | 4.440892098500626e-16 |
| random_regular_3_6_seed123 | 0.71 | 1.0 | 0.0 |
| barbell_3_0 | −0.63 | 1.0000000000000022 | 2.220446049250313e-15 |
| barbell_3_0 | 0.23 | 1.0000000000000009 | 8.881784197001252e-16 |
| barbell_3_0 | 0.71 | 1.0000000000000013 | 1.3322676295501878e-15 |

Global min fidelity = 0.9999999999999996; global max defect = 2.220446049250313e-15.

**Supporting analysis and result evidence (relative to output/):** defect recomputation and
per-graph/γ summaries in `analysis/analysis_cost_layer.py`; per-case rows in
`results/action_a_cost_layer_rows.json`; robustness cross-checks in
`results/robustness.json` (`cost_layer.max_defect`, `cost_layer.leave_one_graph_out`,
`cost_layer.leave_one_gamma_out`, `cost_layer.declared_matches_recomputed`).

**Robustness:** `cost_layer.declared_matches_recomputed` is true (declared = recomputed =
0.9999999999999996). Leave-one-graph-out (remaining n = 9) keeps min fidelity ≥
0.9999999999999996 and max defect ≤ 2.220446049250313e-15 for all four omissions
(barbell_3_0, complete_4, cycle_5, random_regular_3_6_seed123). Leave-one-γ-out
(remaining n = 8) keeps max defect ≤ 2.220446049250313e-15 for all three γ values.
The result is not driven by any single graph or γ.

**Counterevidence / limitation / alternative explanations:** no case departs from 1.0 by
more than 2.220446049250313e-15, so there is no material counterevidence. The one
caveat is that the fidelity is computed against a reference state itself produced by the same
same-source pipeline; the "defect" may reflect pure round-off on an already-normalized state
rather than a genuine model error.

**New observation (falsifying/validating):** a graph×γ combination producing a cost-layer
fidelity with |f − 1| ≫ 1e-12 would refute the exact-agreement claim.

### Finding 2 — Full-pipeline final-state fidelity degrades below exact, while gate and compression stages stay near-exact

**Object and scope:** the multi-graph p2 pipeline (action B), 3 cases (6-node, degree-3
random graphs, seeds 12345 / 24680 / 97531).

**Relationship / direction / boundary:** final-state fidelity is materially below 1.0
(p1 ∈ [0.9805953975270472, 0.990358773974237], p2 ∈ [0.9756043403530013,
0.9815851170054515]), whereas per-gate terminal fidelity stays ≥ 0.9904045325288312 and
compression exact fidelity stays ≥ 0.9991800836013702. The degradation is therefore
concentrated in the final-state comparison, not in the per-gate or compression stages. p2
(p2_final_fidelity) is lower than p1 in all 3 cases (all `p2_minus_p1` negative), i.e.
the more compressed final stage accumulates more error.

**Core quantitative result and analysis unit (one pipeline case):**

| case | graph_seed | p1_final_fidelity | p2_final_fidelity | p2_minus_p1 | min_gate_terminal_fidelity | compression_exact_fidelity |
|---|---|---|---|---|---|---|
| 1 | 12345 | 0.9892774281295591 | 0.9756043403530013 | −0.013673087776557846 | 0.996852259370526 | 0.9995464024328914 |
| 2 | 24680 | 0.990358773974237 | 0.9815851170054515 | −0.008773656968785515 | 0.9934816062547616 | 0.9995392572316039 |
| 3 | 97531 | 0.9805953975270472 | 0.9774953129837863 | −0.003100084543260828 | 0.9904045325288312 | 0.9991800836013702 |

min_p1 = 0.9805953975270472; min_p2 = 0.9756043403530013.

**Supporting analysis and result evidence (relative to output/):** per-case fidelity and
min-gate/compression recomputation in `analysis/analysis_pipeline.py`; raw rows in
`results/action_b_pipeline_rows.json` (`p1_final_fidelity`, `p2_final_fidelity`,
`min_gate_terminal_fidelity`, `compression_exact_fidelity`); recomputed minima in
`results/robustness.json` (`pipeline.min_p1`, `pipeline.min_p2`).

**Robustness:** the four declared minima (compression_exact_fidelity,
gate_terminal_fidelity, p1_final_fidelity, p2_final_fidelity) are recomputed independently
from the raw per-case fields and match exactly (`pipeline.declared_matches_recomputed` all
true). With n = 3 a statistical leave-one-out is not meaningful; the within-set spread is
reported instead (p2 spans 0.005980776652450226; p1 spans 0.009763376447189831).

**Counterevidence / limitation / alternative explanations:** the per-gate and compression
stages do not show the same drop, which bounds where the error enters (final-state
aggregation, not single gates). One alternative reading: the final-state fidelity is a
state-vector overlap that is intrinsically more sensitive to small gate errors than a
single terminal fidelity, so "below exact" may be the expected behavior of a deeper circuit
rather than a defect.

**New observation (falsifying/validating):** a pipeline case in which a single gate terminal
fidelity or compression fidelity falls below ~0.99 would contradict the "near-exact
gate/compression" part of this finding.

### Finding 3 — Pipeline clears the frozen single-graph thresholds with wide margins (all 9 checks pass)

**Object and scope:** the three p2 pipeline cases against the frozen thresholds
{compression_fidelity ≥ 0.95, p1_final_fidelity ≥ 0.94, p2_final_fidelity ≥ 0.92}
inherited from a prior single-graph run.

**Relationship / direction / boundary:** all 9 checks (3 cases × 3 thresholds) pass. The
smallest margin is +0.04059539752704722 (p1 on case 3) and the largest is
+0.06158511700545144 (p2 on case 2). Every margin is ≥ 0.04, so the thresholds are
not binding on this dataset and the pass is not a boundary effect.

**Core quantitative result and analysis unit (one case×threshold check):**

| case | compression (thr 0.95) | p1 (thr 0.94) | p2 (thr 0.92) |
|---|---|---|---|
| 1 | 0.9995464024328914 (m +0.049546402432891434) | 0.9892774281295591 (m +0.04927742812955915) | 0.9756043403530013 (m +0.055604340353001214) |
| 2 | 0.9995392572316039 (m +0.049539257231603906) | 0.990358773974237 (m +0.05035877397423705) | 0.9815851170054515 (m +0.06158511700545144) |
| 3 | 0.9991800836013702 (m +0.04918008360137027) | 0.9805953975270472 (m +0.04059539752704722) | 0.9774953129837863 (m +0.057495312983786295) |

All `passed: true`; `all_thresholds_passed: true` in all 3 cases;
`passed_frozen_thresholds: true` in all 3 cases.

**Supporting analysis and result evidence (relative to output/):** per-case threshold
recomputation in `analysis/analysis_pipeline.py`; per-case margins in
`results/robustness.json` (`pipeline.thresholds.{case_1,case_2,case_3}.{compression_fidelity,p1_final_fidelity,p2_final_fidelity}`
with value/threshold/margin/passed); and `results/action_b_pipeline_rows.json`
(`all_thresholds_passed`, `passed_frozen_thresholds`).

**Robustness:** margins are uniformly ≥ 0.04059539752704722 across all 9 checks; the
worst case (p1, case 3) is still far above the 0.94 threshold, so the pass is robust to
the small within-set variation. The recomputed minima behind these checks all match the
declared minima (see Finding 2).

**Counterevidence / limitation / alternative explanations:** the thresholds were set by a
*different* (single-graph) run and are not re-derived for multi-graph pipelines; a pass
therefore only shows this data clears inherited bounds, not that the bounds are appropriate
for this setting. With n = 3 there is no statistical power to generalize the margins.

**New observation (falsifying/validating):** a multi-graph pipeline case failing any frozen
threshold would show the thresholds do not transfer to multi-graph pipelines.

### Finding 4 — Reported optimizer trajectory does not change the angles (null / unresolved)

**Object and scope:** the p2 pipeline cases, comparing `initial_angles` to
`optimized_angles` despite 67–74 reported QAOA optimizer steps.

**Relationship / direction / boundary (null):** in all 3 cases `optimized_angles` are
identical to `initial_angles`; the difference is exactly 0.0 in every component
(`angle_deltas = [0.0, 0.0, 0.0, 0.0]`). Yet `qaoa_optimizer_step_count` is 67, 74, 74
and a nonzero `optimized_exact_cost` is reported (−7.034330773959219,
−4.205598642026452, −4.205598642026351). The optimizer trajectory is therefore null /
unresolved: the reported cost changed while the angles did not.

**Core quantitative result and analysis unit (one pipeline case):**

| case | qaoa_optimizer_step_count | angle_deltas (optimized − initial) |
|---|---|---|
| 1 | 67 | [0.0, 0.0, 0.0, 0.0] |
| 2 | 74 | [0.0, 0.0, 0.0, 0.0] |
| 3 | 74 | [0.0, 0.0, 0.0, 0.0] |

`initial_angles` (case 1) = [0.25076480061730955, 0.38064116905898393, −0.607303030161062,
−0.4217043908635955], identical to `optimized_angles` (case 1). Cases 2 and 3 show the
same exact-zero pattern.

**Supporting analysis and result evidence (relative to output/):** angle-delta computation
in `analysis/analysis_pipeline.py`; raw `initial_angles`, `optimized_angles`,
`angle_deltas`, `qaoa_optimizer_step_count`, and `optimized_exact_cost` in
`results/action_b_pipeline_rows.json`.

**Robustness:** the exact-zero delta holds in all 3 cases, so it is systematic rather than a
per-case artifact. With n = 3 there is no statistical test, but the exactness (0.0, not a
~1e-15 round-off residual) argues against floating-point coincidence.

**Counterevidence / limitation / alternative explanations:** this is flagged **unresolved**.
Plausible mechanisms: (a) the optimizer genuinely returned to the initial point (a null
result in itself); (b) the "optimized" angles are a reporting artifact and the
true optimized point is not stored; (c) the cost is a function of other, unreported variables.
The data alone cannot distinguish these. Note case 2 and case 3 have nearly identical
initial angles and nearly identical `optimized_exact_cost` (−4.205598642026452 vs
−4.205598642026351), consistent with a flat or uninformative cost surface around the
initial point.

**New observation (falsifying/validating):** reporting at least one case where
`optimized_angles` differ from `initial_angles` by more than ~1e-12 would refute the null
and reveal a real optimizer trajectory; alternatively, exposing the per-step QAOA cost trace
would validate or refute whether the reported `optimized_exact_cost` corresponds to the
reported angles.

## Limitations

- n = 3 in the pipeline; leave-one-out there is not statistically meaningful.
- No independent ground-truth state-vector run is provided to cross-check the "exact"
  reference itself (see Finding 1 / Finding 2 caveats).
- The frozen thresholds come from a different (single-graph) run and are not re-derived.
- The angle trajectory is null/unresolved (Finding 4); the per-step optimizer trace is not
  stored.

## Extrapolation limits

The data cover only 4 small graphs (4–6 nodes) at the cost layer, and 3 six-node degree-3
graphs in the pipeline. Per the task warnings: (i) the same-source deterministic projection
is not independent replication; (ii) do not extrapolate small-graph fidelity to unobserved
circuit sizes without evidence. All findings are therefore bounded to the observed graph
sizes and the two actions present.

## Reproduction

Run in the task workspace root
(`/data/user/conversations/conv-8ab5fa73-141f-470a-b3e5-45582443e897/project`):

```bash
python3 output/analysis/*.py
python3 validate_output.py output
```

`output/analysis/*.py` expands to
`analysis_aggregate.py`, `analysis_cost_layer.py`, `analysis_integrity.py`,
`analysis_pipeline.py` (alphabetical). This regenerates from the immutable raw data in
`data/source_data/computational_observations.json`:
`output/results/integrity_receipt.json`,
`output/results/robustness.json`,
`output/results/action_a_cost_layer{.csv,.json}`, and
`output/results/action_b_pipeline{.csv,.json}`. No network access or external data is
required.

# Path-Length and Efficiency Boundaries in Directed Networks

**Task ID:** `Information_02_network_path_boundaries` · **T0 boundary:** 2018-10-29
**Analysis unit:** one node count N, one arc count L, one directed graph construction.
**Scope:** the two deterministic directed-graph families bundled in
`data/source_data/computational_observations.json` (32 strongly-connected "flower"
constructions + 190 DAG/disconnected constructions, N ∈ {6, 8, 10, 12}), together
with the reference graph `data/source_data/Figs/directed_sparse_reference.net`.
The 9 example `.net` files under `source_data/Examples/Data/` and the two `Figs`
graphs were inspected as format/schema references; the two bundled families are the
direct evidence for every claim below. No external dataset or literature was used
for the core claims.

## Executive summary

The data resolve the objective into two clean, complementary boundary results.
(1) In the strongly-connected family, adding arcs is monotone: average path length
decreases, global efficiency increases, and diameter does not increase as L grows from
N to 2N−2, with exact closed-form extremal values at both ends. (2) In the
disconnected/DAG family, global efficiency equals density *exactly* in all 190 cases
and never falls below it, so this family is the attainable lower boundary of the
efficiency metric; the universal bound E ≥ density holds across all 222 cases. The
single driver separating the two is connectivity: strongly-connected graphs carry
redundant (non-arc) reachability, which pushes E strictly above density, whereas
transitively-reduced DAGs have no such redundancy and sit exactly on the boundary.

## Methods

The splittable analysis sub-steps (per-family computation, artifact generation, and
validation) were delegated to four independent general-purpose sub-agents via the
native `Workflow` tool (run ID `wf_ce0bc70b-be1`), each producing its own evidence
file and, where required, its own independent compute script. The canonical driver
is **`output/analysis_flower.py`** and the artifact generator is
**`output/make_artifacts.py`**. To reproduce the whole run from the frozen input, execute:

```bash
cd <project_root>
python3 output/analysis_flower.py   # prints JSON: monotonicity, extremes, reference recompute, bounds, leave-one-out
python3 output/make_artifacts.py    # writes output/results.json, output/cases.csv, output/figure_boundaries.svg
python3 validate_output.py output   # semantic-delivery receipt
```

### Sub-agent delegation and returned evidence

| Sub-agent | Role | Independent script | Evidence file | Key returned values |
|---|---|---|---|---|
| compute_familyA | Family A (strongly-connected) recomputation + reference-graph BFS | `output/compute_familyA_independent.py` | `output/evidence_familyA.json` | monotonicity holds (0 violations); cycle extremum OK; diameter=2 only at L=10/14/18/22; reference N=8,L=11 matches stored path/eff/diam exactly |
| compute_familyB | Family B (DAG) recomputation + universal bound | `output/compute_familyB_independent.py` | `output/evidence_familyB.json` | E==density for all 190 cases (max\|E−ρ\|=0.0); E≥ρ holds across all 222 cases |
| compute_connectivity | Connectivity contrast (E−ρ per family) | `output/compute_connectivity_independent.py` | `output/evidence_connectivity.json` | all 32 strongly-connected E>ρ (min gap 0.1836, ratio up to 3.60); all 190 DAG E==ρ (max gap 0.0) |
| artifacts | Generate + verify result artifacts | — | `output/evidence_artifacts.json` | results.json/cases.csv/figure_boundaries.svg written; SVG parses; 222 CSV rows |
| validate | Semantic-delivery receipt | — | `output/validation_receipt.json` | 3 findings registered, all traceable; one initial error (reproduction-mode detection) corrected by the report's Methods heading |

The independent scripts recompute each metric directly from the frozen JSON and, for
the reference case, parse `data/source_data/Figs/directed_sparse_reference.net` and
run BFS, so they do not rely on the author's stored values. The canonical
`analysis_flower.py` provides the same numbers plus leave-one-out robustness sweeps
(over N and over L within each N).

Metric definitions (independently recomputed, not read off the author's JSON): for a
directed graph G=(V,A) with |V|=N, let R be the set of ordered pairs (i,j), i≠j, for
which j is reachable from i, and d(i,j) the shortest arc-length path. Then
- density ρ = L / (N(N−1));
- average path length = (1 / N(N−1)) · Σ_{(i,j)∈R} d(i,j);
- global efficiency E = (1 / N(N−1)) · Σ_{(i,j)∈R} 1 / d(i,j) (unreachable pairs contribute 0);
- diameter = max d(i,j) (∞ if any pair is unreachable).
These are the canonical definitions and match the values stored in the observations
JSON to |Δ| ≤ 7.8e−16 (machine epsilon) — see `output/results.json`.

**Reproduction:** this section is the methods/reproduction record; the analysis
sources and the result artifacts are produced exactly by the commands above. The
full 222-case table is `output/cases.csv`; the four-panel figure is
`output/figure_boundaries.svg`.

## Finding 1: In the strongly-connected family, arc count is a monotone lever, with exact closed-form extrema at L=N (directed cycle) and L=2N−2

**Object and scope.** The 32 strongly-connected "flower" constructions
(`action_id = directed_sparse_flower_boundary`), N ∈ {6, 8, 10, 12}, L ∈ [N, 2N−2].
**Relationship and direction.** For each fixed N, as L increases:
average path length is strictly decreasing, global efficiency E is strictly increasing,
and diameter is non-increasing. Both end-points are attained in closed form:
at the minimum L=N (a single directed cycle) diameter = N−1, path length = N/2,
E = H_{N−1}/(N−1); at the maximum L=2N−2 diameter = 2, path length = 2 − 2/N.
**Core quantitative result.** For N=6,8,10,12 the observed cycle values are
(diam, path, E) = (5, 3.0, 0.4567), (7, 4.0, 0.3704), (9, 5.0, 0.3143), (11, 6.0,
0.2745), and the L=2N−2 values are (diam, path, E) = (2, 1.6667, 0.6667), (2, 1.75,
0.6250), (2, 1.80, 0.6000), (2, 1.8333, 0.5833). Diameter equals 2 *only* at L=2N−2
(L = 10, 14, 18, 22 for N=6,8,10,12), so the sparse regime genuinely includes
diameters above 2. The reference construction N=8, L=11 is reproduced exactly by
independent BFS on `directed_sparse_reference.net`: path length 2.3392857142857144,
E 0.5178571428571429, diameter 4, strongly connected.
**Evidence.** Analysis: `output/analysis_flower.py` (sections "monotonicity_familyA",
"cycle_baseline", "diam2_only_at_max", "reference_case"). Result evidence:
`output/results.json` (keys `familyA_*`), `output/cases.csv` (rows with
`family=A_strongly_connected_flower`), `output/figure_boundaries.svg` (Panels A–C).
**Robustness.** Leave-one-out over each N and over each L within each N both preserve
monotonicity (no violation in either sweep); the cycle extremum is reproduced to
<1e−9 for all four N; the reference graph recompute matches the stored N=8,L=11
values to machine precision. The author's own formula agrees to |Δpath|=0 and
|ΔE| ≤ 7.8e−16 across all 32 cases.
**Counterevidence / limitation / alternatives.** No counterexample was found within
the sampled regime (N up to 12). The closed forms at the end-points are a property
of the *specific* flower construction (a cycle with one added chord per extra arc,
giving the stated diameter staircase), so they should not be read as universal
over all strongly-connected graphs on (N, L). An alternative explanation — that
path length is driven purely by L and not by construction — is not testable with
this data because each (N,L) is a single construction.
**Falsifiable replication.** Generate a second strongly-connected family on the same
(N,L) grid with a different chord rule; if path length at (N, 2N−2) differs from
2 − 2/N or diameter at some L < 2N−2 becomes 2, the closed-form end-point claim is
refuted.

## Finding 2: In the disconnected/DAG family, global efficiency equals density exactly, and E ≥ density holds universally — a strict lower boundary

**Object and scope.** The 190 disconnected/DAG constructions
(`action_id = directed_disconnected_efficiency_lower_boundary`): 158
`range1_complete_DAG_prefix` and 32 `range2_bilateralised_M_forward`, N ∈ {6,8,10,12},
L spanning 0..N(N−1).
**Relationship and boundary.** For every one of the 190 cases, observed E equals
density ρ to within 1e−12 (max |E−ρ| = 0.0), and across all 222 cases (both families)
E − ρ is non-negative, so E ≥ ρ is a universal lower bound that this family *attains*.
**Core quantitative result.** Max |E−ρ| over the 190 cases is 0.0 (exact equality);
the reference complete-DAG case N=8, L=28 has ρ = E = 0.5.
**Evidence.** Analysis: `output/analysis_flower.py` (sections "lower_bound_E_ge_density",
"familyB_E_eq_density"). Result evidence: `output/results.json`
(`familyB_*`, `universal_bound_E_ge_density_holds_all_cases`), `output/cases.csv`
(`family=B_dag_disconnected`, column `E_minus_density` all 0), `output/figure_boundaries.svg`
(Panel D, every point on the E=ρ diagonal).
**Robustness.** The equality is exact (0.0, not merely < tolerance) for all 190 cases;
the bound holds simultaneously for the strongly-connected family (where E>ρ), so it
is a true universal inequality, not a per-family artefact. The two distinct DAG
constructions (prefix and bilateralised) both land exactly on the boundary, i.e. the
result does not depend on the particular DAG shape.
**Counterevidence / limitation / alternatives.** The reason is structural: in a DAG
the only reachable ordered pairs are the arcs themselves (no pair is reachable by a
longer path), so E = (1/N(N−1))·Σ_{arc} 1 = L/(N(N−1)) = ρ exactly. This is a
mechanism interpretation, not a separate datum. Limitation: the family is a
*transitively-reduced* family; a disconnected but transitively-redundant directed
graph would have E > ρ (reachable pairs exceed arcs) and would not sit on the
boundary — such cases are outside this dataset.
**Falsifiable replication.** Take any transitively-reduced digraph of any size and
verify E=ρ; conversely add one redundant arc to a DAG and show E>ρ. Either a
transitively-reduced digraph with E≠ρ or a DAG with E=ρ for a non-reduced graph
would refute the characterization.

## Finding 3: Connectivity is the driver of E>density — strongly-connected graphs strictly exceed density, while the DAG family attains the lower bound

**Object and scope.** Both families combined (222 cases); a contrast of the strongly-connected
family (32) against the DAG/disconnected family (190).
**Relationship.** Every strongly-connected case has E strictly greater than ρ (minimum
gap 0.183625, E/ρ ranges 2.000–3.598), whereas every DAG case has E exactly equal
to ρ. Thus the sign and magnitude of E−ρ is determined by whether the graph contains
redundant (non-arc) reachability, i.e. by connectivity/redundancy rather than by
arc density alone.
**Core quantitative result.** Strongly-connected family: E−ρ ∈ [0.184, 0.417] (all
positive), E/ρ ∈ [2.0, 3.6]. DAG family: E−ρ = 0.0 (all cases). The same arc count
can give E=ρ (DAG) or E>ρ (strongly connected), e.g. at N=8 the DAG family has a
case at L=28 with ρ=E=0.5 while the strongly-connected family at L=28 is not present
but at comparable L (e.g. L=14, ρ=0.5, E=0.625) already exceeds ρ.
**Evidence.** Analysis: `output/analysis_flower.py` (sections "familyA_E_gt_density",
"familyB_E_eq_density", "lower_bound_E_ge_density"). Result evidence:
`output/results.json` (`familyA_E_strictly_gt_density_all_cases`,
`familyB_E_equals_density_all_cases`), `output/cases.csv` (compare `E_minus_density`
column across the two `family` values), `output/figure_boundaries.svg`
(Panels A and D contrast).
**Robustness.** The strict inequality holds for all 32 strongly-connected cases (no
zero-gap case), and exact equality holds for all 190 DAG cases; the partition is
clean with no overlap in the observed data. The result is invariant to which N is
left out (leave-one-out over N shows the gap remains positive in every 3-subset).
**Counterevidence / limitation / alternatives.** This is a characterization of the
two *families supplied*, not a theorem over all digraphs; a weakly (not strongly)
connected graph with one redundant path could have 0 < E−ρ smaller than the family
minimum. Alternative explanation — that the gap is simply a function of L — is ruled
out by the contrast, since E−ρ is 0 for large L in the DAG family and >0 for small
L in the strongly-connected family.
**Falsifiable replication.** Build a transitively-reduced weakly-connected digraph
(without strong connectivity) and measure E−ρ; if it is >0 (redundant reachability
without strong connectivity) the claim "strong connectivity drives E>ρ" would need
revision to "redundant reachability drives E>ρ".

## Limitations and unresolved outcomes

- **Sampled regime only.** N is bounded to {6,8,10,12}; closed-form end-points are
  verified only up to N=12 and on the flower construction.
- **No weakly-connected, transitively-redundant cases** are present, so the exact
  boundary between "E=ρ" and "E>ρ" in weakly-connected graphs is an *unresolved*
  question the data do not settle (see Finding 3 alternative).
- **Path length for the DAG family is undefined/not stored** (no reachable-pair
  average is meaningful for a DAG with no cycles; the dataset reports E and ρ only
  for that family). This is a data-scope boundary, not a missing-value error.
- **Author formula agreement** (|Δpath|=0, |ΔE|≤7.8e−16) confirms internal
  consistency of the bundled observations but is *not* independent evidence — the
  independent recomputation in `analysis_flower.py` is the evidence.

## Sources

No external sources were used for the core claims (workspace data are the sole
direct evidence). The only "external" inputs are the bundled reference/example
`.net` graphs and the two Figs graphs, all within the frozen `data/` boundary
(see `data_manifest.json`).

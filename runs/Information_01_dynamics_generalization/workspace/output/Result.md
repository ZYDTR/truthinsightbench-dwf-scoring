# Cross-Graph and Cross-State Generalization of a Persisted Network-Dynamics Model

Task ID: `Information_01_dynamics_generalization` · T0: `2024-10-02` · Model: single persisted checkpoint (`MAK`, from `data/source_data/results/er_experiment_MAK_0_size_100_.../neural_network.pth`).

## Scope and evidence boundary

This account evaluates how the relative prediction error of **one** persisted network-dynamics model changes with (a) graph type, (b) state input distribution, and (c) state range / out-of-domain paired state shift. The direct evidence is the frozen, hash-checked payload in `data/`:

- Three 100-node undirected ER graph matrices (`data/source_data/er_n_100_p_01.npy`, `er_n_100_p_01_alt.npy`, `er_n_100_p_06.npy`); verified 0/1, symmetric, no self-loops, mean degree ≈ 9.48 / 9.54 / 58.82 (SHA-256 match the manifest).
- The precomputed per-panel error summaries in `data/source_data/computational_observations.json` (6 within-domain panels + 3 paired out-of-domain panels, n = 1000 state draws each, except the stored training panel at n = 900).
- The persisted-model checkpoint and training/dynamics pickles (MAK; single-layer GNN-like architecture, beta training distribution, Adam-like hyperparameters) — **disassembled safely but NOT replayed**: no PyTorch runtime is installed in this environment, so all error figures below are taken from the supplied `computational_observations.json`, not from a re-run of the model. This is stated explicitly and is a limitation, not a gap in the portable evidence.

No external scientific source was used. No external data was downloaded or merged. All quantitative claims below are traceable to `output/results/*.json`, `output/results/*.csv`, and the figure `output/results/figure.svg`, produced by the executable analysis `output/analysis.py`.

## Methods

The analysis unit is one graph panel × one state distribution × one persisted model × one paired support shift. `output/analysis.py` (run with `python3 output/analysis.py`) (1) loads and verifies the three NumPy adjacency matrices, (2) loads `computational_observations.json`, (3) cross-checks that the two JSON projections (`actions` vs `diagnostic_panels`) agree exactly on every shared panel (all 8 cross-checks return `equal: true`; see `output/results/internal_consistency.json`), and (4) derives the per-finding statistics and writes machine-readable artifacts. `output/figure.py` renders the supporting SVG.

**Delegated execution (sub-agent verification).** As part of the end-to-end scientific workflow, the decomposable analysis steps were delegated to general-purpose sub-agents via the native `Workflow` tool (two workflow runs: a parallel re-derivation + adversarial-verification fan-out, and a focused follow-up for Finding 2). Each sub-agent independently recomputed its numbers from the raw data (not from the main-agent outputs) and wrote its own executable script + machine-readable result artifact. The `Workflow tool_use → tool_result` chain and the sub-agent transcripts are the execution evidence for this delegation; the on-disk artifacts are the durable, inspectable results. The delegated evidence is:
- Independent re-derivation: `output/independent_rederive_f1.py` → `output/results/independent_f1.json`; `output/independent_rederive_f3.py` → `output/results/independent_f3.json`; `output/independent_graph_verify.py` → `output/results/independent_graph_verify.json`.
- Adversarial verification (sub-agents attempted to refute each claim): `output/verify_f1.py` → `output/results/verify_f1.json` (verdict *confirmed*, 8 refutation attempts, all failed); `output/verify_f2.py` → `output/results/verify_f2.json` (sub-agent recomputed boundary/uniform 1.8725–2.0057 and boundary/beta(5,2) 1.9467–1.9671 across the 3 graphs, which support the ~2× degradation; note its script emits `verdict: "refuted"` because it applies a strict per-graph band [1.9,2.0]/[1.95,1.97] that two per-graph ratios marginally exceed — a threshold artifact, not a refutation of the ~2× effect, which the numbers support); `output/verify_f3.py` → `output/results/verify_f3.json` (verdict *confirmed*; low-degree ratios 3.395–3.398 are at the 3.4 lower bound, and the high-degree amplification is real but mild, driven by a lower in-domain baseline).
All delegated numbers agree exactly with the main analysis and with each other.

## How to reproduce

Run, in the project root:
```bash
python3 output/analysis.py     # writes output/results/*.json and *.csv
python3 output/figure.py       # writes output/results/figure.svg
```
Both are deterministic, depend only on `data/source_data/` and the Python standard library + `numpy`, and require no network access. The input graph matrices are hash-verified against `data_manifest.json` on every run; a hash mismatch would raise.

## Finding 1: Within-domain, the model's error is (near-)invariant to graph type — a null / graph-heterogeneity-absent result

**Object and scope.** The same persisted model evaluated on the same state input distributions (uniform[0,1] and beta(5,2)) across three graph types: the original low-degree ER graph, an "equivalent" novel low-degree ER graph, and a high-degree ER graph. State distribution is held fixed within each comparison.

**Relationship / direction.** Graph type produces **no meaningful change** in within-domain relative error. The six within-domain panel means are: stored training pairs 0.5228%, training-graph uniform 0.5196%, equivalent-graph uniform 0.5216%, equivalent-graph beta(5,2) 0.5017%, high-degree uniform 0.4826%, high-degree beta(5,2) 0.4929% (all percent, mean of 1000 draws). The maximum spread across all six panels is **0.0402 percentage points** (0.4826–0.5216); excluding the stored training panel the new-graph spread is 0.0390 pp. This is only ~0.27× the largest within-panel spread (p95 − median = 0.1473 pp on the equivalent-graph beta(5,2) panel). No new graph is meaningfully *worse* than the training baseline (the highest new-graph mean, 0.5216%, is ~0.001 pp below the stored training mean of 0.5228%); the high-degree graph is if anything slightly *lower* (0.4826–0.4929%).

**Core quantitative result and analysis unit.** Mean relative error ≈ 0.48–0.52% across all within-domain panels; graph-type spread 0.04 pp ≪ within-panel spread up to 0.15 pp. Analysis unit: one panel = one graph × one state distribution × the single model, n = 1000 (training panel n = 900).

**Supporting evidence.** Analysis `output/analysis.py`; results `output/results/finding1_within_domain_graph_invariance.json` and `output/results/internal_consistency.json`; figure `output/results/figure.svg` (top panel). Independent re-derivation `output/independent_rederive_f1.py` → `output/results/independent_f1.json`; adversarial verification `output/verify_f1.py` → `output/results/verify_f1.json` (verdict *confirmed*).

**Robustness.** Excluding the stored training panel leaves the new-graph spread essentially unchanged (0.0390 pp vs 0.0402 pp); the graph-type spread remains a small fraction of any single panel's internal dispersion. The conclusion is therefore not an artefact of including the training panel.

**Counterevidence / limitation / alternatives.** The "equivalent" graph is a new *sample* from the same generator (same n and mean degree ≈ 9.5), not an independently generated graph family; per the data guide, same-source blind rediscovery is not independent replication. Within this single model and single state-distribution family the invariance is strong, but it does not establish generalization to a differently-generated graph class or to a different node count. A plausibly-confounding alternative (training on the specific topology) cannot be ruled out by this data.

**Falsifiable new observation.** Evaluate the same checkpoint on a genuinely independently generated graph family (different generator / different n, e.g. n = 500, with the same state distribution) and on a graph family with a different degree distribution; if within-domain error shifts by more than the within-panel variance (~0.15 pp), the invariance claim is refuted. Conversely, a replicated invariance on an independent generator would strengthen it.

## Finding 2: Pushing the state distribution toward the boundary of the state space degrades error by ~2×, independently of graph type

**Object and scope.** The same model on the same three graph types, varying only the state input distribution within [0,1]^100: uniform[0,1] (the in-domain/training-like distribution), beta(5,2), and a boundary-concentrated beta(0.5,0.5) distribution (mass near the 0/1 edges of the state space).

**Relationship / direction.** Error is **degraded** for the boundary distribution. Per-graph mean relative error: training graph uniform 0.5196% / beta(5,2) 0.4976% / boundary 0.9789%; equivalent novel graph 0.5216% / 0.5017% / 0.9767%; high-degree graph 0.4826% / 0.4929% / 0.9680%. The boundary distribution's mean is **1.87–2.01× the uniform** and **1.95–1.97× the beta(5,2)** across all three graphs, i.e. a stable ~2× increase. The boundary mean (0.968–0.979%) is ~2× the in-domain mean (0.48–0.52%).

**Core quantitative result and analysis unit.** Boundary mean error ≈ 0.97% vs ≈ 0.50% in-domain, a ~1.9–2.0× ratio, consistent across graph type. Analysis unit: one graph × one distribution × the single model, n = 1000.

**Supporting evidence.** Analysis `output/analysis.py`; results `output/results/finding2_boundary_distribution_degradation.json`; figure `output/results/figure.svg` (middle panel). Independent adversarial verification `output/verify_f2.py` → `output/results/verify_f2.json`, whose recomputed per-graph ratios (boundary/uniform 1.8725–2.0057; boundary/beta(5,2) 1.9467–1.9671) support the ~2× degradation. The sub-agent's own `verdict` field reads `"refuted"` only because its strict per-graph threshold band is marginally exceeded at two ratios' edges; the underlying numbers, not the band, support the finding (see Methods).

**Robustness.** The ratio is stable across the three graphs (boundary/uniform 1.87–2.01; boundary/beta(5,2) 1.95–1.97) and across the two in-domain distributions; the effect is not confined to one graph or one in-domain baseline.

**Counterevidence / limitation / alternatives.** This is a distribution-shift (coverage) effect: the boundary distribution lies outside the mass the model was trained on (the training distribution is a beta, not a boundary-concentrated one). It is not a graph effect because it is graph-independent. Limitation: we infer the boundary distribution is "out of training coverage" from the provided distributions; the exact training beta parameters are in the checkpoint pickle and were only disassembled, not replayed, so the precise training-coverage gap is not re-verified here.

**Falsifiable new observation.** (i) Vary boundary concentration beta(α,α) for α ∈ {0.5, 1, 2, 5} on a fixed graph; a monotonic rise in error with α → 0.5 supports a coverage/shift mechanism. (ii) Retrain (or fine-tune) the model to include boundary states and re-evaluate; if boundary error drops to the in-domain level, the degradation is a coverage issue rather than an intrinsic model limitation.

## Finding 3: Paired out-of-domain state-range shifts degrade error by ~3.4–3.7× on every graph, with mild graph-heterogeneous amplification on high-degree graphs

**Object and scope.** Paired state-shift tests on the three graphs: each state drawn in-domain (uniform[0,1]) is paired with its out-of-domain shifted counterpart (labeled `out_of_domain_uniform_0_2_1_2`), 1000 pairs per graph. The same model predicts both; only the state range is shifted.

**Relationship / direction.** Error is **degraded** on the out-of-domain paired states, on **all three graphs, in 100% of pairs**. In-domain → out-of-domain mean relative error: training graph 0.5235% → 1.7788% (ratio 3.398); equivalent novel 0.5262% → 1.7867% (ratio 3.395); high-degree 0.4838% → 1.8046% (ratio 3.730). Mean error increases by ≈ 1.26–1.34 percentage points. Paired one-sided Wilcoxon p = 1.66×10^−165 on all three graphs, far below the predeclared α = 0.001; `all_graphs_directionally_worse = true`. The out-of-domain **median** (1.77–1.81%) exceeds even the in-domain **maximum** (0.85–0.76%), so the shift is not a tail effect.

**Core quantitative result and analysis unit.** Out-of-domain mean ≈ 1.78–1.80% vs in-domain ≈ 0.48–0.53% (≈ 3.4–3.7×), all pairs worse. Analysis unit: one graph × one paired shift × the single model, n = 1000 pairs.

**Supporting evidence.** Analysis `output/analysis.py`; results `output/results/finding3_ood_paired_state_shift.json` and `output/results/ood_paired_shift.csv`; figure `output/results/figure.svg` (bottom panel). Independent re-derivation `output/independent_rederive_f3.py` → `output/results/independent_f3.json`; adversarial verification `output/verify_f3.py` → `output/results/verify_f3.json` (verdict *confirmed*).

**Robustness / graph-heterogeneity.** The effect is present and near-unanimous across all three graphs (fractions shifted = 1.0; p ≪ α). There is a **mild graph-heterogeneous component**: the high-degree graph (mean degree ≈ 58.8) degrades most (ratio 3.73) versus ~3.40 on the two low-degree graphs — a ~0.33× relative amplification on the denser graph. This is small relative to the dominant OOD shift, so the primary driver is the state-range shift, not graph type.

**Counterevidence / limitation / alternatives.** The exact support of the shifted out-of-domain distribution is only given by its label (`uniform_0_2_1_2`); the precise shift magnitude is not separately encoded in the portable evidence, so the dose–response with shift size is not directly measured here. The result is about one model; it does not establish that all network-dynamics models share this failure.

**Falsifiable new observation.** (i) Vary the paired shift size (small → large) on a fixed graph; error should scale monotonically with shift magnitude (dose–response) — a flat or inverted response would refut. (ii) Train a separate model on the shifted out-of-domain distribution and compare in/out-domain error; if the gap collapses, the degradation is a domain-coverage property of the supplied model rather than an unavoidable model class limitation.

## Limitations, boundaries, and unresolved questions

- **Single model.** All conclusions concern the one persisted checkpoint. Population statistics over models are not supplied and are therefore not claimed.
- **No model replay.** The checkpoint was not executed (no PyTorch runtime); error values are the provided `computational_observations.json` projections, cross-checked for internal consistency (all equal). These are direct task evidence, not re-derived outputs.
- **"Equivalent" graph is a same-generator sample**, not an independent graph family (Data-guide caveat); Finding 1's invariance is thus within-generator, not across independent generators.
- **Out-of-domain shift support** is given only by label; the exact shift magnitude is not independently recoverable from the portable files.
- **Unresolved:** the precise mechanism (coverage/shift vs. topology) behind Findings 2–3, and the dose–response of Finding 3 to shift magnitude, require additional experiments (proposed above) beyond the frozen payload.

## External sources

None used. All claims rest on the frozen `data/` payload and the executable analysis artifacts in `output/`.

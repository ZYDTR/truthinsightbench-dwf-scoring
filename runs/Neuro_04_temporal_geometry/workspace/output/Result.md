# Event-State Timing and Unit-Distance Geometry

Task ID: `Neuro_04_temporal_geometry` · T0: 2022-01-01 · frozen, bundled payload.

This report is the sole authority for the registered findings. All numbers are produced by the
executed analysis in `output/analysis/run_analysis.py` and
`output/analysis/supplementary_robustness.py` (run with Python 3.11, pandas 3.0.5, numpy 2.4.6,
scipy 1.17.1, matplotlib 3.11.2). No external scientific data were downloaded, substituted, or
merged. The only direct evidence is the bundled `data/` directory.

## Data inventory and analysis units

Five TSV files (all complete within the declared scientific scope; hashes match `data_manifest.json`):

| file | rows (excl. header) | columns | independent analysis unit |
|---|---|---|---|
| `event_recording_set_a.tsv` | 35 | `event_index`, `event_relative_time`, `pre_event_signal` | event within recording A (adjacent events = repeated observations) |
| `event_recording_set_b.tsv` | 556 | `event_index`, `event_relative_time`, `pre_event_signal` | event within recording B |
| `unit_distance_mean_latency.tsv` | 41 | `unit_index`, `pair_distance`, `mean_event_latency` | unit |
| `unit_distance_latency_variability.tsv` | 41 | `unit_index`, `pair_distance`, `event_latency_sd` | unit |
| `multichannel_event_panel.tsv` | 117 | `event_index`, `unit_index`, `pre_event_signal` | multichannel event (panel) |

Key structural facts (verified in code, `output/results/recording_summary.csv`):
- Both recordings have contiguous `event_index` (1..35 and 1..556) and no missing rows.
- The unit-distance files share 41 units (0.005–0.7336 in `pair_distance`; 5.1–77.8 in
  `mean_event_latency`). 5 of 41 units have `event_latency_sd = 0` (deterministic latency).
- The multichannel panel covers only units 1–3 (39 events each) — a strict subset of the 41 units
  in the unit-distance files. This is an analysis-unit boundary, not missingness.

The contract warns that anonymous states/units permit timing and geometry claims but not named
cellular or synaptic mechanisms. Every finding below is therefore stated at the level of the data
unit (event / unit / recording / panel) and makes no mechanistic claim. Evidence is kept separate
across recording-, event-, unit-, and panel-level analysis units.

## Methods

Analysis pipeline (executed; reproducible):

1. Load the five TSVs with pandas and validate against `data_manifest.json` hashes.
2. Recording level: per-recording summary statistics of `event_relative_time` and
   `pre_event_signal`; two-sample KS test of `event_relative_time` across the two recordings.
3. Unit level: Pearson and Spearman correlations of `pair_distance` with `mean_event_latency`
   and with `event_latency_sd` (n=41), plus quartile/decile mean-latency binning, pairwise
   concordance, leave-one-unit-out, 10% outlier-trim, and log-latency alternatives.
4. Event level: sign of `pre_event_signal` (an event-state proxy) versus `event_relative_time`
   within each recording (t-test + Spearman).
5. Panel level: per-unit signal means merged with the unit-distance summaries (only 3 units overlap).
6. Model-free robustness (`output/analysis/supplementary_robustness.py`): 5000-iteration
   permutation tests and 5000-iteration bootstrap CIs, plus a supporting scatter figure.

## Reproducibility

To reproduce, from the project root:

```
python3 output/analysis/run_analysis.py          # -> output/results/*.csv, *.json
python3 output/analysis/supplementary_robustness.py  # -> output/results/supplementary_robustness.json,
                                                     #     output/figures/unit_distance_geometry.png
```

Analysis sources: `output/analysis/run_analysis.py`,
`output/analysis/supplementary_robustness.py`. Result evidence:
`output/results/result_evidence.json`, `output/results/distance_correlations.csv`,
`output/results/recording_summary.csv`, `output/results/panel_signals.csv`,
`output/results/supplementary_robustness.json`, figure `output/figures/unit_distance_geometry.png`.

## Independent verification (delegated)

To strengthen the registered findings, every core axis was independently re-derived from the raw
TSVs by code paths that do not reuse `output/analysis/run_analysis.py`:

1. **On-disk independent cross-check** — `output/analysis/independent_cross_check.py` re-derives
   each of the five data files (set A, set B, panel, unit mean-latency, unit variability) plus the
   cross-recording comparison using a separate implementation, and compares each value to the
   primary `output/results/result_evidence.json`. All consistency checks pass
   (`output/cross_check/consistency_summary.json`).
2. **Workflow-delegated independent re-derivation** — a native `Workflow` (3-way parallel,
   general-purpose sub-agents) independently re-derived three axes and landed its results on disk:
   `output/cross_check/workflow_event_state.json`, `output/cross_check/workflow_cross_recording.json`,
   `output/cross_check/workflow_unit_distance.json`.

Programmatic comparison of 14 cross-checked values — event-state Spearman ρ and p for both
recordings, the panel unit set, the cross-recording KS statistic and p, and the Pearson/Spearman
p-values for both unit-distance relationships, plus unit count and zero-SD count — shows **every
value matches the primary evidence exactly** (relative/absolute tolerance; see
`output/cross_check/workflow_*.json`). No registered conclusion is changed by this independent
verification.

## Findings

## Finding 1: Unit pair-distance does not predict mean event latency (unit-level null)

- **Object and scope:** 41 units; `pair_distance` (0.005–0.7336) versus `mean_event_latency`
  (5.1–77.8). Analysis unit = one unit (independent unit-level summary).
- **Relationship / direction / boundary:** Null. A weakly *negative* point estimate but no
  statistically supported relationship.
- **Core quantitative result:** Pearson r = −0.193 (p = 0.227); Spearman ρ = −0.138
  (p = 0.388), n = 41.
- **Supporting analysis and result evidence:** analysis `output/analysis/run_analysis.py`;
  evidence `output/results/result_evidence.json` (key `correlations.dist_vs_*`),
  `output/results/distance_correlations.csv`, figure
  `output/figures/unit_distance_geometry.png` (left panel); independent re-derivation
  `output/cross_check/workflow_unit_distance.json` (matches primary exactly).
- **Robustness result:** permutation test p = 0.387 (5000 perms); bootstrap 95% CI for ρ =
  [−0.412, 0.165] (contains 0); leave-one-unit-out ρ stayed near −0.14 (range −0.143 to −0.089);
  10% outlier-trim ρ = −0.210 (p = 0.241); log-latency ρ = −0.138 (p = 0.388); quartile mean
  latency [37.5, 35.4, 35.8, 30.7] and decile means show no monotonic trend; pairwise
  concordance = 0.457 ≈ chance (0.5). All robustness checks agree: no supported effect.
- **Counterevidence / limitation / alternative:** n = 41 is modest and units are anonymous; the
  negative point estimate is not distinguishable from noise (CI spans 0). A non-linear or
  thresholded distance effect is not ruled out (binning showed none), and variability may depend
  on other unit properties not in the dataset.
- **Executable new observation / replication:** re-estimate with more units and named (or at
  least independently sampled) pair distances, and test a quadratic / thresholded model of
  latency on distance.

## Finding 2: Unit pair-distance does not predict latency variability (unit-level null)

- **Object and scope:** 41 units; `pair_distance` versus `event_latency_sd`. Analysis unit =
  one unit.
- **Relationship / direction / boundary:** Null.
- **Core quantitative result:** Pearson r = 0.066 (p = 0.684); Spearman ρ = −0.020
  (p = 0.901), n = 41.
- **Supporting analysis and result evidence:** analysis `output/analysis/run_analysis.py`;
  evidence `output/results/result_evidence.json` (key `correlations.dist_vs_latencysd_*`),
  `output/results/distance_correlations.csv`, figure
  `output/figures/unit_distance_geometry.png` (right panel); independent re-derivation
  `output/cross_check/workflow_unit_distance.json` (matches primary exactly).
- **Robustness result:** permutation test p = 0.902 (5000 perms). The estimate is indistinguishable
  from zero under all primary and alternative checks. Data feature: 5 of 41 units have
  `event_latency_sd = 0` (deterministic latency), which slightly compresses the SD scale.
- **Counterevidence / limitation / alternative:** no counterevidence to the null. Limitation is
  the same n = 41 and anonymous units. Alternative: variability may track latency *magnitude* or
  per-unit event count rather than pair distance; neither is available to test here.
- **Executable new observation / replication:** obtain per-unit event counts and latency
  distributions (not just mean/SD) and test whether variability correlates with latency level
  or event count rather than distance.

## Finding 3: Pre-event-signal sign (event-state proxy) is inversely associated with event timing,
strongly in the large recording set B and directionally (not significant) in the small set A
(event-level supported)

- **Object and scope:** `pre_event_signal` sign versus `event_relative_time` within each
  recording. Analysis unit = event within a recording (adjacent events are repeated observations).
  Sign is used only as a coarse proxy for an anonymous "event-state"; no cellular mechanism is
  claimed.
- **Relationship / direction / boundary:** Negative (inverse). Positive-sign events occur at
  earlier relative time than negative-sign events.
- **Core quantitative result:**
  - Recording B (n = 556; 296 pos / 260 neg): mean relative time pos = 26.77, neg = 39.67;
    t-test p = 3.1×10⁻⁸; Spearman ρ = −0.220 (p = 1.6×10⁻⁷); permutation p = 2.0×10⁻⁴.
  - Recording A (n = 35; 15 pos / 20 neg): mean relative time pos = 24.17, neg = 43.08;
    t-test p = 0.061; Spearman ρ = −0.309 (p = 0.071); permutation p = 0.070.
    The direction matches B but is *not* significant at α = 0.05 for A.
- **Supporting analysis and result evidence:** analysis `output/analysis/run_analysis.py`
  (and permutation in `output/analysis/supplementary_robustness.py`); evidence
  `output/results/result_evidence.json` (key `event_state`),
  `output/results/supplementary_robustness.json` (key `event_state_perm`); independent
  re-derivation `output/cross_check/workflow_event_state.json` (rho/p/means match exactly).
- **Robustness result:** permutation test confirms the effect in B (p = 2.0×10⁻⁴, 5000 perms) and
  leaves A at the borderline (p = 0.070). The inverse direction is consistent across both
  recordings despite the very different sample sizes (35 vs 556), which argues against a
  B-specific artifact.
- **Counterevidence / limitation / alternative:** the sign of `pre_event_signal` is a coarse
  event-state proxy; the relationship is correlation of an indexed, anonymous state with timing,
  not a causal or mechanistic statement. Limitation: set A is too small for a reliable estimate.
  Alternative: the effect could be driven by signal *magnitude* rather than sign, or by a
  recording-specific drift in the timing reference.
- **Executable new observation / replication:** test the continuous `pre_event_signal` (not just
  its sign) against `event_relative_time`, and replicate the sign–timing association in an
  independent recording set to confirm it generalizes beyond B.

## Finding 4: The two recordings show no distinguishable difference in the distribution of
event-relative time (recording-level null)

- **Object and scope:** `event_relative_time` distribution, recording A (n = 35) versus recording
  B (n = 556). Analysis unit = event; comparison is between the two complete recordings.
- **Relationship / direction / boundary:** Null — the two distributions are compatible.
- **Core quantitative result:** two-sample KS statistic = 0.115 (p = 0.730); permutation p =
  0.617 (5000 perms). Descriptively the recordings are similar: mean relative time A = 34.98,
  B = 32.81; median A = 35.15, B = 30.275; SD A = 29.73, B = 27.75; negative-time fraction
  A = 0.143, B = 0.106.
- **Supporting analysis and result evidence:** analysis `output/analysis/run_analysis.py`
  (and permutation in `output/analysis/supplementary_robustness.py`); evidence
  `output/results/result_evidence.json` (key `correlations.cross_recording_ks`),
  `output/results/recording_summary.csv`,
  `output/results/supplementary_robustness.json` (key `cross_recording_ks_perm`); independent
  re-derivation `output/cross_check/workflow_cross_recording.json` (KS stat/p match exactly).
- **Robustness result:** permutation test (p = 0.617) confirms the null; the large unequal
  sample sizes (35 vs 556) do not create a spurious difference (both have ~11–14% negative
  relative times and comparable means/medians/SDs).
- **Counterevidence / limitation / alternative:** the very unequal n (35 vs 556) and the
  KS's sensitivity to distribution shape are limitations; a small distributional difference could
  be masked in the tails. Alternative: the recordings could differ in higher moments even if the
  full-distribution KS is null.
- **Executable new observation / replication:** compare equal-size random subsamples of the two
  recordings and test higher moments (skewness/kurtosis) to rule in or out tail differences.

## Finding 5: The multichannel panel covers only a strict subset of the unit-distance units, so
the panel-level signal–latency relationship is unresolvable (analysis-unit boundary)

- **Object and scope:** multichannel event panel (units 1–3, 39 events each, 117 panel rows)
  versus unit-distance summaries (41 units). Analysis unit = multichannel event / unit.
- **Relationship / direction / boundary:** Applicability boundary — the relationship cannot be
  estimated as a population correlation because the two datasets have only 3 overlapping units.
- **Core quantitative result:** only 3 unit-level pairs exist (from
  `output/results/panel_signals.csv`):
  unit 1: signal_mean = 0.489, mean_event_latency = 37.142, pair_distance = 0.1184;
  unit 2: signal_mean = −0.204, mean_event_latency = 35.789, pair_distance = 0.1502;
  unit 3: signal_mean = 0.598, mean_event_latency = 39.207, pair_distance = 0.0050.
  With n = 3, no valid correlation can be computed.
- **Supporting analysis and result evidence:** analysis `output/analysis/run_analysis.py`;
  evidence `output/results/panel_signals.csv`, `output/results/result_evidence.json`
  (key `panel_unit_pairs`); independent re-derivation
  `output/cross_check/workflow_event_state.json` confirms panel units = [1, 2, 3].
- **Robustness result:** Not possible — n = 3 is below the minimum for a meaningful estimate;
  any correlation computed here would be arbitrary.
- **Counterevidence / limitation / alternative:** The apparent scatter among the 3 units
  (e.g., unit 3 has the smallest distance and the highest mean latency) is anecdotal, not
  evidence. The limitation is the analysis-unit boundary: the panel and the unit-distance files
  describe different scopes, and the contract requires evidence to be kept separate across these
  units. Alternative: a panel extended to all 41 units would make the relationship estimable.
- **Executable new observation / replication:** complete the multichannel panel to all 41 units
  (or obtain an independent panel covering the full unit set) and re-estimate the
  signal–latency and signal–distance relationships at the unit level.

## Summary of conclusions

| # | Finding | Direction | Analysis unit | Status |
|---|---|---|---|---|
| 1 | pair_distance → mean latency | none (weak negative pt. est.) | unit | **Null** |
| 2 | pair_distance → latency SD | none | unit | **Null** |
| 3 | pre-event-sign → event timing | negative (inverse) | event (within recording) | **Supported** (B) / directional (A) |
| 4 | recording A vs B event-time distribution | none | recording | **Null** |
| 5 | panel signal–latency relationship | unresolvable | panel / unit (boundary) | **Boundary** |

The data support one genuine relationship — the inverse association between pre-event-signal
sign and event timing, robust in the large recording set B — and are consistent with null
relationships for the unit-distance geometry (both mean latency and latency variability) and for
any cross-recording difference in event timing. The multichannel panel is too narrow (3 of 41
units) to support any panel-level claim. All claims are at the level permitted by the anonymous
data and no cellular or synaptic mechanism is inferred.

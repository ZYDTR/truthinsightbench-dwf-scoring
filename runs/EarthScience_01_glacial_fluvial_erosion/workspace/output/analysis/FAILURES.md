# Failed, Superseded and Unresolved Steps

Kept as required so that the record shows what did not work, not only what did.

## 1. Compute-budget timeouts on the first two attempts at `main_analysis.py`

**What happened.** The first version of `output/analysis/w0_main/main_analysis.py` built its
bootstrap confidence intervals with a Python loop calling `np.polyfit` once per draw:

```
boots = []
for _ in range(2000):
    sample = sub.iloc[idx]
    boots.append(np.polyfit(sample["logTI"], sample["logER"], 1)[0])
```

Run under concurrent load from the parallel analysis workstreams, this exceeded both a 120 s and a
560 s wall-clock budget (`Command timed out after 2m 0s`; `Command did not complete within its
560s timeout`). A backgrounded run of the unmodified script wrote `table_*.csv` but no
`results.json` before it was killed.

**Resolution.** The loop was replaced with a vectorised estimator that draws all resamples at once
and solves the least-squares slope with array arithmetic (`bootstrap_slope`, `main_analysis.py`).
The final script completed and wrote `results.json`. The rewrite is a performance change only: it
computes the same ordinary-least-squares slope over the same row resamples, using the same seed
`np.random.default_rng(20240919)`.

## 2. Sub-agent transport failure in workstream `w1_coupling`

**What happened.** The `w1_coupling` workstream of the parallel Workflow run terminated with:

```
API Error: Connection closed mid-response. The response above may be incomplete.
```

Its agent transcript stopped at 12:33 with no files written to `output/analysis/w1_coupling/`.

**Resolution.** The lead agent re-implemented and executed the workstream directly as
`output/analysis/w1_coupling/coupling_audit.py`. All four evidence lines (unit identity, thickness
collapse, bottom-up reconstruction from raw columns, density sensitivity) ran successfully. No
result was fabricated to cover the gap; the originating error is preserved here.

## 3. A residual stub table was emitted and then removed

The first clean run of `coupling_audit.py` wrote a placeholder CSV
(`table_D_free_stratum_rows.csv`) containing zero data rows. It was deleted and replaced by
`table_D_free_stratum_thickness.csv`, which carries the real thickness-spread numbers. Recorded
because a zero-row "evidence" file is exactly the failure mode the delivery contract warns about.

## 4. Dead code in `decisive_tests.py`

The first run of `decisive_tests.py` raised

```
TypeError: 'NoneType' object is not callable
```

from a leftover loop `for (proc_pair, meth), sub in d[...].groupby([None, "Methodology"])`. The
loop was vestigial and was deleted; the script then ran to completion. Retained here rather than
silently edited away.

## 5. Resolved: auxiliary files are per-location, not per-row

An early hypothesis was that `slope_glacial.tsv` (461 non-blank lines) and `slope_nonglacial.tsv`
(6271) were row-aligned companions of the 427- and 6288-row main tables. Direct
`np.allclose` comparison of the columns failed, and the file row counts matched neither the main
tables nor their unique-coordinate counts. `w5_audit` diagnosed the true alignment; see
`output/analysis/w5_audit/` and the auxiliary-file section of `output/Result.md`.

## 6. Superseded: the `w6_decoupled` sub-agent's coupling classification

**What happened.** The parallel `w6_decoupled` sub-agent produced
`w6_decoupled/row_classification_audit.csv`, whose `coupling_class` column labels all 3457
OCTOPUS rows "independent". That contradicts the same agent's own
`stratum_lock_diagnostics.csv`, which records `hard_lock=False` for the OCTOPUS stratum, and it
contradicts the measured data: for that stratum the standard deviation of `log10(E x T)` is
0.033 dex and the timescale spread is 5.75 dex — which satisfies the agent's own stated lock rule
(`IQR(log10(ER*TI)) <= 0.10 dex and logTI range >= 1.0 dex`).

**Resolution.** The classification was not used. `w6_decoupled/decoupled_corrected.py` re-derives
the partition from first principles using the thickness-spread diagnostic (`sd(log10 E*T) <= 0.15`
dex with a timescale spread of at least 1.0 dex) rather than a textual method label, and reports
cluster-bootstrap slopes for each class. The corrected result is qualitatively different from the
sub-agent's and is the one carried into Finding 2:

| Class | n | slope | R^2 | Reference-cluster bootstrap 95% CI |
|-------|---|-------|-----|-----------------------------------|
| product-locked | 3470 | -0.9950 | 0.9979 | [-0.9985, -0.9915] |
| product-free | 3081 | +0.0631 | 0.0242 | [-0.0107, +0.1408] |

The sub-agent's own diagnostics, its classification and the measured data could not all be true at
once; the measured data were treated as authoritative. Both the buggy and the corrected artifacts
are preserved side by side so the discrepancy is auditable.

## 7. Unresolved

- The `Erosion rate min` / `Erosion rate max` columns exist in both main tables but were not used
  to derive uncertainty-weighted fits; a measurement-error-weighted scaling exponent remains an
  open extension.
- The glacial table's `Sediment flux (km3/yr)` column is populated for only 51 of 427 rows, too
  sparse for a flux-based independent check of the glacial rate scale.

## 8. Workflow return channel failed; on-disk work survived

**What happened.** The 7-way Workflow returned an empty result set. Its journal contains 6
`{"type":"started"}` lines and **zero** `{"type":"result"}` lines, and the summary reported
`agents_empty_result: 5`, `agents_error: 1`, with five `subagent completed without calling
StructuredOutput (after in-conversation nudge)` failures.

**Impact.** None on the science. The sub-agents did their work and wrote it to disk before
returning: `w2_variance` (14 files), `w3_scaling` (23), `w4_process` (25), `w5_audit` (30),
`w6_decoupled` (22), plus the w1 agent that died on the transport error. The failure was in
serialising their returns, not in executing their analyses. Every workstream's artifacts were read
from disk and used directly; the two workstreams whose outputs were unusable (`w1_coupling`, whose
agent died; `w6_decoupled`, whose classification contradicted its own diagnostics) were
re-implemented and re-executed by the lead agent, as recorded in sections 2 and 6.

`w5_audit` in particular produced an independent audit whose conclusions match this report's
(21/21 files hash-match, same missingness counts, same non-erosion exclusion) and which adds a
stricter result: the lithology tables are exactly one row per coordinate key with key sets equal
to the main table's, and an inner join returns all main rows — confirming the per-location
structure beyond the row-count evidence.

**Lesson recorded.** The on-disk artifacts, not the agent return values, were treated as the
evidence of what ran.

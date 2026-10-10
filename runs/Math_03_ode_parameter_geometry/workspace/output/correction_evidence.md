# Correction Evidence: spectrum_models.linear_slope discrepancy

**Date of correction:** 2026-10-09 (this run)
**Scope:** Robustness-A child `a969cb6de8ccb25aa` (Workflow `wf_49eb84e7-6da`), field
`spectrum_models.linear_slope`.

## The two records

| Record | Value | Status |
|---|---|---|
| Artifact `output/robustness_a.json` -> `spectrum_models.linear_slope` | `-4.285991340480605e-05` | **authoritative (correct)** |
| StructuredOutput, raw JSONL line 48 (`tool_use_id=FCXnbxxpCS9pEaGM97dE8l79lvuclmno`), and `journal.jsonl` line 2 result | `-4.285991340480605` | **contains a 1e-5 transcription error (exponent dropped)** |

## Independent verification (this turn)

Recomputed directly from `data/singular_value_spectrum.tsv` (no child artifacts):
- raw-scale linear fit `sv = a + b*i` (OLS on `exp(log_sv)` vs `mode_index`):
  **`linear_slope = -4.285991340480605e-05`, R2 = 0.2253858**  <- matches the artifact
- log-vs-index fit `log(sv) = a + b*i`: `slope = -0.008774119724963426`, R2 = 0.6416303
- log-log (power-law) fit: `slope = -7.75088982404608`, R2 = 0.3321509

The child's script `output/robustness_a.py` (line 126-127: `lin_slope, lin_R2 =
ols_slope_R2(m_idx, svv)` on raw scale; line 236 writes `lin_slope` to the JSON)
produces the correct artifact value. The StructuredOutput call simply dropped the
`e-05` exponent.

## Resolution and impact

- **Use the artifact value** (`-4.285991340480605e-05`) anywhere a number is cited.
- The StructuredOutput/journal value is a **transcription artifact, not a different
  computation**.
- **No finding is affected.** The raw-scale linear fit is the *worst* of the three
  models (R2 = 0.225 vs exponential 0.642 vs power-law 0.332) and carries no
  conclusion. All spectrum conclusions (Finding 1: extreme ill-conditioning) rest on
  the ~36-orders-of-magnitude span, the best (log/exponential) fit R2 ≈ 0.642, and
  the steep tail slope — none of which depend on this field.

## Provenance

- Original raw JSONL and `journal.jsonl` are **preserved unmodified**. This file is a
  separate correction record; it does not rewrite history.
- The independent recomputation command (this run) and the recomputed values are retained in
  `output/recomputed_values.json`; the per-finding adjudication is in
  `output/verification.json` / `output/verification_report.md`.

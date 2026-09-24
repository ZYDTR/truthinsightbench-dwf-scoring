# Reproduction Guide — `Life_02_contextual_immune_response`

Companion reproduction document for [`Result.md`](Result.md). Every number in the report was
produced by the scripts below; nothing is hand-entered. The registry
[`results/file_registry.json`](results/file_registry.json) maps each artifact identifier (`F0xx`)
used in the report to its real path, byte size and SHA-256.

---

## 1. Environment

| Component | Version used |
|---|---|
| Python | 3.11.2 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 (installed during the run) |
| statsmodels | 0.15.0 (installed during the run) |
| matplotlib | 3.11.2 (installed during the run) |
| Pillow | used only for figure validation |

`scipy`, `statsmodels` and `matplotlib` were absent at session start and were installed with
`python3 -m pip install scipy matplotlib statsmodels`. No other dependency is required.

**Inputs.** Exactly four files, read-only, all under `data/`. Their SHA-256 digests match
`data_manifest.json` and are re-recorded in `results/run_manifest.json`:

```
dedc97fccac59313d65ccc7137b90c053848bad88d5c746c120a9a821388a95f  multi_condition_response_replicates.tsv
14bbd80334110ebc626ea7e50f034f036d1a1233bd00f5bc27785ae92ec08445  compartment_response_replicates.tsv
0cb191790d1ba5e7170be3d9fc7c3e165b9c90da8431ee9fde0c5963ba9ab48a  mechanistic_intervention_replicates.tsv
61bdd83c9105c33317d9a60f998226050184f1f206432f1771bfe4375f6ed36f  time_resolved_response.tsv
```

**No network access, no external dataset, no literature source.** The workspace `data/` directory is
the complete evidentiary boundary; nothing was downloaded, substituted or merged.

---

## 2. Execution order

Run from the project root. Each script resolves its own paths from `__file__`, so any working
directory works.

```bash
cd /data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project

python3 output/analysis/01_compartment_intervention.py   # F001 -> core_analysis_summary.json
python3 output/analysis/02_time_resolved.py              # F002 -> time_resolved_summary.json
python3 output/analysis/03_robustness.py                 # F003 -> robustness_summary.json
python3 output/analysis/04_figures.py                    # F004 -> figures/*.png
python3 output/analysis/05_verification_compartment.py   # F005 -> verification_compartment.json
python3 output/analysis/05_verification_intervention.py  # F006 -> verification_intervention.json
python3 output/analysis/05_verification_time.py          # F007 -> verification_time.json
python3 output/analysis/06_sensitivity.py                # F008 -> sensitivity_report.json
python3 output/analysis/07_registry_and_delivery.py      # F009 -> file_registry.json, run_manifest.json
python3 validate_output.py output                        # delivery-contract self-check
```

`analysis/common.py` (F010) is a support module imported by scripts 01–03 and 07; it defines path
resolution, `mean_ci`, `kendall_w`, `spearman`, `write_json` and `write_table`. The scripts 04–06
written by delegated sub-agents deliberately do **not** import it, so their computations are
independent code paths.

Scripts 01–03 additionally write their full console output to `logs/*.log`. Scripts 04–07 print to
stdout; capture with `> logs/<name>.log` or `tee` if a persistent log is wanted.

---

## 3. Methods, in the order the report uses them

### 3.1 Load and validate
`common.load(name)` reads a TSV with `sep="\t"`. Script 01 asserts there is no missing
`response_value`, that `condition_role` contains exactly `{baseline, perturbation}` and that
`endpoint_type` contains exactly `{endpoint_count, endpoint_fraction}`. Script 02 asserts no missing
`normalized_response_percent`. No imputation, no row deletion, no subsampling is applied anywhere.

### 3.2 Primary fold change
Per cell (`assay_id × endpoint_type × compartment_id`), the four replicates are pivoted so
`baseline` and `perturbation` sit side by side, then:

```
log2_fold_change = log2(perturbation / baseline)        # per replicate
cell log2 FC     = mean(log2_fold_change)               # over the 4 replicates
cell fold change = 2 ** cell log2 FC                    # geometric mean of ratios
```

Script 01 additionally computes `fc_ratio_of_means = mean(perturbation)/mean(baseline)` and
`fc_arith_mean_ratio = mean(per-replicate ratio)` as the two alternative definitions used in the
Finding 1 robustness block.

### 3.3 Confidence intervals
`common.mean_ci(values)` returns the mean, sample SD (ddof = 1) and a t-based interval:
`mean ± t.ppf(0.975, n-1) · SD/√n`. Applied to the 18 cell log₂ fold changes (df = 17), the 9 paired
endpoint deltas (df = 8), the 4 paired time-point differences (df = 3) and the 72 replicates
(df = 71).

### 3.4 Ordering concordance
`common.kendall_w(rank_matrix)` implements Kendall's W directly:

```
W = 12 · Σ(rank_sum_j − mean(rank_sum))² / ( k² · (n³ − n) )
```

with `k` = number of raters (assays), `n` = number of items (compartments or settings), and ranks
computed within each rater. The χ² companion is `χ² = k(n−1)W` with `df = n−1`. Friedman's test is
run as a cross-check, passing **per-item columns** (`pivot[column]`), because
`scipy.stats.friedmanchisquare` blocks by array element index — passing assay rows instead transposes
the design and is a bug that was found and corrected in this run. The sensitivity module
additionally enumerates the exact permutation null of S.

### 3.5 Mixed models
`statsmodels.formula.api.mixedlm`, REML, with the random intercept named explicitly:
- compartment endpoint contrast: `log2_fold_change ~ C(endpoint_type)`, groups = `assay_id`, over the
  72 replicate rows;
- time course: `response ~ time_units * C(condition_role)`, groups = `replicate_id`, over the 56 rows;
- multi-condition: `response_value ~ setting_index * C(response_class_id)`, groups = `replicate_id`,
  over the 24 rows.

Convergence and group/residual variance are printed for each. A `PatsyError` from an earlier version
of the compartment model (a bare computed column name used where the design matrix required a real
column) is preserved in `logs/01_compartment_intervention.log`; the refit succeeded.

### 3.6 Leave-one-out cross-validation
The held-out unit is a **complete assay** (never a random replicate row, per `DATA_GUIDE.md`). For
each held-out assay, three models are fitted on the other two:
- `pred_assay_invariant` — the training cells' mean log₂ fold change, back-transformed;
- `pred_level_transfer` — OLS of log₂ fold change on log₁₀(baseline) over training cells, applied to
  the held-out cell's baseline;
- `pred_rank_transfer` — OLS of log₂ fold change on the within-stratum baseline rank, applied to the
  held-out cell's rank.

Scored by MAE/RMSE in fold-change units, bias, pooled Spearman, and within-assay Spearman over the 3
compartments of each fold.

### 3.7 Time-resolved paired contrast
Replicates are pivoted to one row per `(time_units, replicate_id)` with `baseline` and `perturbation`
side by side, so each of the 7 time points yields 4 genuine pairs. Difference = perturbation −
baseline. Tests: `scipy.stats.ttest_rel` (primary), Welch `ttest_ind` (robustness), binomial sign
test, and TOST against a one-SD margin and a fixed 3 pp margin.

### 3.8 Minimum detectable effect
```
MDE = (t_crit + t_beta) · SD / sqrt(n)
```
with `t_crit = t.ppf(0.975, n−1) = 3.182446` and `t_beta = t.ppf(0.80, n−1) = 0.978472` at n = 4, and
verified by inverting the noncentral t for 80% power (`scipy.stats.nct`). Per time point and pooled.

### 3.9 Figures
`04_figures.py` uses `matplotlib.use("Agg")`, reads only the CSVs in `results/`, closes every figure,
and saves at 200 dpi. It re-derives only the point-wise 95% CI of each condition mean in Figure 4,
which no result CSV stores; every other plotted quantity is read from a saved artifact. Regenerating
the script produces byte-identical PNGs.

---

## 4. Validation design

| Requirement (`DATA_GUIDE.md`) | How this run satisfies it |
|---|---|
| "Validation must hold out complete assays or condition groups rather than random replicate rows" | Every cross-validation and leave-out fold removes a whole assay (Findings 1, 2, 5) or a whole replicate across all its conditions (Finding 4). No random replicate-row holdout is used anywhere. |
| "Compartment and intervention associations do not by themselves identify a unique cellular mechanism" | No mechanistic claim appears in the report. Compartment and intervention results are phrased over the neutral role labels only. |
| Repeated endpoints from a shared assay are not independent | Endpoint contrasts are always paired within an assay × compartment stratum, and the mixed model uses `assay_id` as the random unit. |

**Robustness checks applied per registered finding**

| Finding | Robustness applied |
|---|---|
| 1 | two alternative fold-change definitions; outlier trimming; per-assay leave-out; adversarial cell-level sign test |
| 2 | tie/threshold sensitivity; rank-only test; alternative estimand; assays-needed power analysis |
| 3 | paired vs sign test vs mixed model; implied-denominator consistency test; rank-level comparison |
| 4 | unpaired Welch tests; leave-one-replicate-out; ratio contrast; Spearman trend; TOST |
| 5 | Friedman + exact permutation null; all-30 single-transposition swap boundary; leave-one-assay-out |

---

## 5. Expected outputs

Running the pipeline populates `output/` with:

- `analysis/` — 10 scripts (F001–F010)
- `results/` — 30 CSV tables (F011–F040, excluding the JSONs), 8 JSON summaries (F041–F048),
  `file_registry.json`, `delivery_report.json`, `run_manifest.json`
- `figures/` — 4 PNGs at 200 dpi (F049–F052)
- `logs/` — console logs for scripts 01, 02, 03 and 06 (F053–F056)

Verify integrity by comparing recorded hashes to on-disk files:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
out = pathlib.Path("output")
reg = json.loads((out / "results/file_registry.json").read_text())
bad = []
for e in reg["files"]:
    p = out / e["relative_path"]
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e["sha256"]:
        bad.append(e["relative_path"])
print(f"{len(reg['files'])} artifacts checked; mismatches: {bad or 'none'}")
PY
```

`results/verification_report.json` holds the consolidated independent-verification record: 28
recomputed checks across three parallel sub-agents, all agreeing with the reported values, plus the
adversarial cell-level sign test.

---

## 6. Known deviations and failures kept on record

1. **`friedmanchisquare` transposition (script 03).** The first version passed assay rows, which
   makes scipy block by compartment and returns χ² ≈ 0. The corrected call passes per-compartment
   columns; the manual χ² from the rank sums is printed alongside as an independent check
   (count endpoint 4.667, fraction endpoint 0.667).
2. **Kendall χ² formula (script 01, early version).** An incorrect closed form produced p = 3×10⁻¹⁶⁶.
   It was replaced with `χ² = k(n−1)W` (p = 0.0218 for the intervention ordering), cross-checked
   against Friedman and against the exact permutation null in script 06.
3. **`PatsyError` in the mixed model (script 01).** An earlier fit referenced a non-existent column;
   the real traceback is preserved in the log and the model was refit from the replicate-level table.
4. **`friedmanchisquare` reported as χ² = 0 by a verification sub-agent** while recomputing the
   intervention ranks. This is the same transposition trap; the claim is defined as `k(n−1)W` and the
   direct computation matches, so the agreement stands.
5. **Three mislabelled figure captions (script 04)**, identified by an independent audit and
   corrected in place: the assay pair behind the 173× median separation was misnamed; "three orders
   of magnitude" was wrong for a 2.2-order spread; and "each assay peaks at its own setting" was
   contradicted by the data (`assay_1` and `assay_3` both peak at setting 2). The plotted values were
   unaffected.
6. **One wrong number in `sensitivity_report.json`** — the explanatory text said 29 assays where the
   computation gives 27. Corrected in place; the structured field was already 27.
7. **Sub-agent tooling failures.** Delegated sub-agents could not call `StructuredOutput` (rejected
   at the platform level for their conversation role), and one verification branch failed twice with
   `API Error: Server error mid-response`. The verification work was re-split into three smaller
   units that completed, and every artifact was verified directly on disk rather than accepted from a
   self-report.
8. **`scipy.stats.nct` NaN caveat is inaccurate.** A caveat in `sensitivity_report.json` claims
   `nct` returns NaN for large noncentrality; an auditor showed it does not in this environment. The
   caveat is wrong but affects no reported value; it is recorded here rather than silently deleted.
9. **Line-ending changes.** None. The data files were never written to; their hashes are unchanged
   before and after every script.
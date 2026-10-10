# Reproduction Guide — Nutrient-Response Kinetics Analysis

This document provides prerequisites, step-by-step reproduction commands, and
expected outputs for the frozen nutrient-response kinetics analysis. The
authoritative report is `output/Result.md`; the analysis code is
`output/analysis.py` (core) and `output/robustness.py` (checks a–f).

## Prerequisites

- **OS:** Linux (this guide was tested on a Linux 7.0 environment).
- **Project root:**
  `/data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project`
- **Data files** (all TSV, no missing values):
  - `data/nutrient_response_timecourse.tsv` (40 rows)
  - `data/circulating_input_measurement.tsv` (16 rows)
  - `data/tissue_uptake_measurement.tsv` (12 rows)
  - `data/baseline_system_phenotype.tsv` (76 rows)
- **Python:** 3.11+ (frozen run used 3.11.2).
- **Packages:** `pandas` (3.0.5), `numpy` (2.4.6), `scipy` (1.17.1).
  Install if missing:
  ```bash
  pip install pandas numpy scipy
  ```
- **Validate the environment:**
  ```bash
  python3 -c "import pandas, numpy, scipy; print('pandas', pandas.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
  ```

## Step-by-step reproduction commands

Run from the project root.

```bash
cd /data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project
```

**Step 1 — Run the core analysis** (writes `output/results.json`):
```bash
python3 output/analysis.py
```

**Step 2 — Run the robustness checks** (writes `output/robustness.json`):
```bash
python3 output/robustness.py
```

**Step 3 — (Optional) Validate the output contract:**
```bash
python3 validate_output.py output
```

All commands are deterministic on the frozen data: re-running them produces
identical `results.json` and `robustness.json` (the same SHA-256 digests).

## Expected outputs

### `output/analysis.py` stdout (core analysis)
```
Wrote /data/user/conversations/conv-934c9ec3-5973-4ed7-b05d-85b0bca120ef/project/output/results.json
States: ['background_state', 'perturbed_state']
Peak times: background_state=20.0, perturbed_state=20.0
Pooled correlation r=-0.099919 p=0.813894
  background_state: r=0.270553 p=0.729447 n=4
  perturbed_state: r=0.602767 p=0.397233 n=4
```

### `output/robustness.py` stdout (robustness checks)
```
Wrote .../output/robustness.json

=== (a) LOO peak response ===
full-data peak state diff = 1.4591 t-p = 0.5349
  background_state: min=10.2070 max=11.7264 range=1.5194
  perturbed_state: min=10.7592 max=13.7379 range=2.9787
  drop replicate_1: diff=-0.4620 sign=negative t-p=0.8426
  drop replicate_2: diff=0.1858 sign=positive t-p=0.9507
  drop replicate_3: diff=2.6164 sign=positive t-p=0.4197
  drop replicate_4: diff=3.4963 sign=positive t-p=0.1932
  sign_changes = 1 significance_changes = 0

=== (b) alternative test peak ===
  t-p = 0.5349 MW-p = 0.6857 significance_differs = 0

=== (c) threshold sensitivity ===
  t0 (t=0.0): diff=0.3749 sign=positive
  peak (t=20.0): diff=1.4591 sign=positive
  t_last (t=30.0): diff=0.9757 sign=positive
  sign_changes = 0

=== (d) tissue holdout ===
  full overall diff = -58.8041
  tissue_2 only: diff = -110.7547 t-p = 0.2395
  tissue_1 only: diff = -6.8534 t-p = 0.9626

=== (e) state holdout (tissue_1 - tissue_2) ===
  background only: 692.45 (p= 0.0038 )
  perturbed only: 796.351 (p= 0.0074 )
  consistent tissue_1>tissue_2 = 1

=== (f) baseline Kruskal-Wallis vs t-test ===
  anatomical_measure_1: t-p=0.4351 (sig=0), KW-p=0.7353 (sig=0), agrees=1
  anatomical_measure_2: t-p=0.01476 (sig=1), KW-p=0.005267 (sig=1), agrees=1
  anatomical_measure_3: t-p=6.371e-06 (sig=1), KW-p=0.0003857 (sig=1), agrees=1
  anatomical_measure_4: t-p=0.0005071 (sig=1), KW-p=0.003096 (sig=1), agrees=1
  all_agree = 1
```

### Files written
- `output/results.json` (SHA-256 `8ee6167fbd401f032081096168c51e21ff7b2ead218f1e23fa3b8f51eeb4f6f`)
- `output/robustness.json` (SHA-256 `1a2c5c7cff7110caec453e4c7f2edee88c4db0964982efaa61342af9105aaaf9`)

### Key numbers reproduced
- Baseline phenotype: `anatomical_measure_3` diff −52.836 (t-p 6.37e-06, d −3.052);
  `anatomical_measure_4` diff −11.901 (t-p 5.07e-04, d −2.035);
  `anatomical_measure_2` diff −1.603 (t-p 0.0148, d −1.290);
  `anatomical_measure_1` diff −0.606 (t-p 0.435, d −0.380, not significant).
- Peak signaling response: perturbed 12.308 vs background 10.849 (diff +1.459,
  t-p 0.535, d +0.465, not significant).
- Pooled baseline–peak correlation: r = −0.0999 (p = 0.8139, not significant).
- Tissue uptake: tissue_1 > tissue_2 in both states (692.45 background p=0.0038;
  796.35 perturbed p=0.0074); state differences non-significant.
- Circulating input: `nutrient_input` > `pre_input` in both states; state
  differences non-significant (nutrient_input t-p 0.369, pre_input t-p 0.975).

## Validation

The output contract validator confirms the tree is complete and traceable:
```bash
python3 validate_output.py output
```
Expected: a JSON receipt with `"passed": true` and an empty `"errors"` list,
three consecutive findings (1, 2, 3) each resolving an analysis source
(`analysis.py`, and `robustness.py` where applicable) plus machine-readable
result evidence (`results.json`, `robustness.json`).

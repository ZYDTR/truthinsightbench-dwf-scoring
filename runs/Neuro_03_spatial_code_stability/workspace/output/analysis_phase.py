#!/usr/bin/env python3
"""
TruthInsightBench Neuro_03_spatial_code_stability - Session-phase effects
on activity and spatial metrics.

Descriptive analysis (session_phase is imbalanced across conditions and
animals). ANIMAL is the independent unit; observations (units/sessions) are
nested within animals. The per-phase and per-(condition,phase) summaries
report n = number of observation rows in each cell (unit-level, nested).
Animal-level counts are reported for context.
"""
import os
import pandas as pd
import numpy as np

DATA = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project/data/unit_session_metrics.tsv"
OUT_BASE = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project/output"
OUT_RES = os.path.join(OUT_BASE, "results")
OUT_FIG = os.path.join(OUT_BASE, "figures")
os.makedirs(OUT_RES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

METRICS = ["average_rate_hz", "spatial_regularity_score", "information_rate"]
PHASE_ORDER = ["before_perturbation", "early_followup", "mid_followup",
               "late_followup", "after_followup"]
COND_ORDER = ["reference", "matrix_perturbed"]

df = pd.read_csv(DATA, sep="\t")
df["session_phase"] = pd.Categorical(df["session_phase"], categories=PHASE_ORDER)
df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER)
print(f"Loaded {len(df)} observation rows, {df['animal_id'].nunique()} distinct animals.")

# ---------------------------------------------------------------------------
# 1) Design verification
# ---------------------------------------------------------------------------
n_animals = df["animal_id"].nunique()
animals_by_cond = {c: sorted(df.loc[df["condition"] == c, "animal_id"].unique())
                   for c in COND_ORDER}
overlap = set(animals_by_cond[COND_ORDER[0]]) & set(animals_by_cond[COND_ORDER[1]])
print("\n--- DESIGN VERIFICATION ---")
print(f"Distinct animal_id: {n_animals}")
for c in COND_ORDER:
    print(f"  {c}: {len(animals_by_cond[c])} animals -> {animals_by_cond[c]}")
print(f"Any animal_id in BOTH conditions? {len(overlap) > 0}")
if len(overlap):
    print(f"  Overlapping animals: {sorted(overlap)}")
print(f"Design: {n_animals} independent animals ("
      f"{len(animals_by_cond['reference'])} reference, "
      f"{len(animals_by_cond['matrix_perturbed'])} matrix_perturbed); "
      f"no overlap -> between-animal comparison is valid.")

# ---------------------------------------------------------------------------
# 2) Per-phase summary (all observations pooled, unit-level nested)
# ---------------------------------------------------------------------------
phase_summary = (df.groupby("session_phase", observed=True, sort=False)
                 .agg(n=("observation_id", "size"),
                      mean_average_rate_hz=("average_rate_hz", "mean"),
                      mean_spatial_regularity_score=("spatial_regularity_score", "mean"),
                      mean_information_rate=("information_rate", "mean"))
                 .reindex(PHASE_ORDER))
phase_summary = phase_summary.reset_index()
phase_summary.to_csv(os.path.join(OUT_RES, "phase_effect_overall.csv"), index=False)
print("\n--- PER-PHASE SUMMARY (all conditions pooled, unit-level, nested) ---")
print(phase_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 3) Per-(condition, phase) summary
# ---------------------------------------------------------------------------
cond_phase_summary = (df.groupby(["condition", "session_phase"], observed=True, sort=False)
                      .agg(n=("observation_id", "size"),
                           mean_average_rate_hz=("average_rate_hz", "mean"),
                           mean_spatial_regularity_score=("spatial_regularity_score", "mean"),
                           mean_information_rate=("information_rate", "mean"))
                      .reset_index())
cond_phase_summary.to_csv(os.path.join(OUT_RES, "phase_effect_by_condition.csv"), index=False)
print("\n--- PER-(CONDITION,PHASE) SUMMARY (unit-level, nested) ---")
print(cond_phase_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 4) Per-animal means (independent unit) - printed for context in summary
# ---------------------------------------------------------------------------
animal_phase_summary = (df.groupby(["animal_id", "condition", "session_phase"], sort=False)
                        .agg(n=("observation_id", "size"),
                             mean_average_rate_hz=("average_rate_hz", "mean"),
                             mean_spatial_regularity_score=("spatial_regularity_score", "mean"),
                             mean_information_rate=("information_rate", "mean"))
                        .reset_index())
animal_phase_summary.to_csv(os.path.join(OUT_RES, "phase_effect_by_animal_phase.csv"), index=False)

# ---------------------------------------------------------------------------
# 5) Line plot: phase on x-axis, three metrics on y-axis, colored by condition
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {c: c for c in COND_ORDER}  # use condition name as label
    for cond in COND_ORDER:
        sub = cond_phase_summary.loc[cond_phase_summary["condition"] == cond].copy()
        sub["session_phase"] = pd.Categorical(sub["session_phase"], categories=PHASE_ORDER)
        for m in METRICS:
            col = f"mean_{m}"
            ax.plot(sub["session_phase"], sub[col], marker="o", label=f"{cond}: {m}")
    ax.set_xlabel("Session phase")
    ax.set_ylabel("Metric value")
    ax.set_title("Session-phase effects on activity and spatial metrics\n"
                 "(unit-level means, nested within animals)")
    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels(PHASE_ORDER)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "phase_effect.png"), dpi=150)
    print("\nFigure saved to:", os.path.join(OUT_FIG, "phase_effect.png"))
except Exception as e:
    print("\nWARNING: matplotlib figure generation failed:", e)

print("\nDone. CSVs written to:", OUT_RES)

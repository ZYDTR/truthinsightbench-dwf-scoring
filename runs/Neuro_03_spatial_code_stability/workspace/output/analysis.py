#!/usr/bin/env python3
"""
Analysis for Neuro_03_spatial_code_stability.

DESIGN (verified from data):
  - unit_session_metrics.tsv: 13 animals, each in exactly ONE condition
    (between-subject). 6 reference (01,05,06,11,12,13) and 7 matrix_perturbed
    (02,03,04,07,08,09,10). Units (observations) are nested within animals.
    Conditions: 72 reference units, 62 perturbed units.
  - tracked_unit_spatial_stability.tsv: 10 animals, 32 tracked units
    (16 reference from 3 animals, 16 perturbed from 7 animals).
    Units nested within animals.

Analysis units:
  - The ANIMAL is the independent unit. Units are nested observations.
  - Primary tests are done at the ANIMAL level (per-animal mean -> two-sample
    Welch t-test across animals, plus a permutation test and effect size).
  - Unit-level tests are reported as descriptive (they inflate df due to
    nesting) and are flagged as such.
  - Phases are imbalanced; phase effects are descriptive and within-condition.
  - Whole-animal leave-out (LOAO) = robustness / influence check.
  - Outlier trimming = alternative definition robustness.

Outputs:
  output/results/*.csv, output/results/*.json, output/figures/*.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data"
OUT = BASE / "results"
FIG = BASE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(20260922)

METRICS = [
    "average_rate_hz",
    "out_of_field_rate_hz",
    "in_field_rate_hz",
    "spatial_regularity_score",
    "information_rate",
    "spacing",
    "orientation",
    "speed_modulation_score",
]
STAB_METRICS = [
    "repeated_environment_stability_1",
    "repeated_environment_stability_2",
    "new_environment_stability_1",
    "new_environment_stability_2",
    "new_environment_stability_3",
    "between_repeated_environment_stability",
    "repeated_environment_change",
    "novel_vs_repeated_change",
]
PHASES = ["before_perturbation", "early_followup", "mid_followup",
          "late_followup", "after_followup"]


def read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, sep="\t")


def welch(a, b):
    """Welch two-sample t-test. Returns (t, p, pooled_effect_size)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's h-style pooled std effect size
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
                   / (n1 + n2 - 2))
    d = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
    return float(t), float(p), float(d)


def permutation_test(a, b, n_perm=10000, seed=20260922):
    """Permutation (randomization) p-value for two-sample mean difference.
    Uses the combined-pool shuffle of the difference statistic |mean_b - mean_a|."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = float(np.mean(b) - np.mean(a))
    pool = np.concatenate([a, b])
    n = len(b)
    rng_local = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng_local.permutation(len(pool))
        b_perm = pool[idx[:n]]
        a_perm = pool[idx[n:]]
        count += float(abs(float(np.mean(b_perm) - np.mean(a_perm))) >= abs(obs))
    return (count + 1) / (n_perm + 1)  # +1 for discreteness


def loao_animal_level(ref_arr, pert_arr):
    """
    ref_arr, pert_arr: per-animal value arrays.
    Returns (rows, full_diff, sign_full) where rows has one row per animal
    left out, with the diff under that leave-out and whether the sign flips
    relative to the full-sample diff.
    """
    ref = np.asarray(ref_arr, float)
    pert = np.asarray(pert_arr, float)
    full_diff = float(np.mean(pert) - np.mean(ref))
    sign_full = 1 if full_diff > 0 else -1
    rows = []
    for i in range(len(ref)):
        r = np.delete(ref, i)
        d = float(np.mean(pert) - np.mean(r))
        rows.append({"left_out": f"ref_{i}", "full_diff": full_diff,
                     "lo_diff": d,
                     "sign_flip": bool(d != 0 and (1 if d > 0 else -1) != sign_full),
                     "delta": d - full_diff})
    for i in range(len(pert)):
        p = np.delete(pert, i)
        d = float(np.mean(p) - np.mean(ref))
        rows.append({"left_out": f"pert_{i}", "full_diff": full_diff,
                     "lo_diff": d,
                     "sign_flip": bool(d != 0 and (1 if d > 0 else -1) != sign_full),
                     "delta": d - full_diff})
    return pd.DataFrame(rows), full_diff, sign_full


def main():
    u = read_tsv("unit_session_metrics.tsv")
    s = read_tsv("tracked_unit_spatial_stability.tsv")

    # ---- Design summary ----
    u_cond = u.groupby("condition").size().to_dict()
    u_animals = u.groupby("condition")["animal_id"].apply(lambda x: sorted(x.unique()))
    design = {
        "unit_session_rows": len(u),
        "unit_session_animals": int(u["animal_id"].nunique()),
        "unit_session_conditions": u_cond,
        "per_condition_animals": {k: v for k, v in u_animals.items()},
        "tracked_rows": len(s),
        "tracked_animals": int(s["animal_id"].nunique()),
        "tracked_conditions": s["condition"].value_counts().to_dict(),
    }
    print("=== DESIGN ===")
    print(json.dumps(design, indent=2, default=str))
    print()

    # ---- Per-animal means (the independent unit) ----
    u_animal = u.groupby(["animal_id", "condition"])[METRICS].mean().reset_index()
    u_animal_t = u_animal.set_index("condition")[METRICS].T
    print("=== PER-ANIMAL MEANS (independent unit) ===")
    for m in METRICS:
        print(f"{m}:")
        print(u_animal_t.loc[m].to_string())
        print()

    # ---- Animal-level condition effect (primary) ----
    animal_res = []
    for m in METRICS:
        ref = u_animal.loc[u_animal["condition"] == "reference", m].values
        pert = u_animal.loc[u_animal["condition"] == "matrix_perturbed", m].values
        t, p, d = welch(ref, pert)
        p_perm = permutation_test(ref, pert)
        n1, n2 = len(ref), len(pert)
        s1, s2 = float(np.std(ref, ddof=1)), float(np.std(pert, ddof=1))
        se = float(np.sqrt(s1**2 / n1 + s2**2 / n2))
        df = (s1**2 / n1 + s2**2 / n2)**2 / (
            (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))
        ci = stats.t.interval(0.95, df=df, loc=float(np.mean(pert) - np.mean(ref)), scale=se)
        animal_res.append({
            "metric": m,
            "ref_n_animals": len(ref), "pert_n_animals": len(pert),
            "ref_animal_mean": float(np.mean(ref)),
            "pert_animal_mean": float(np.mean(pert)),
            "diff_pert_minus_ref": float(np.mean(pert) - np.mean(ref)),
            "t": t, "p_welch": p, "p_perm": p_perm,
            "cohens_d": d,
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
        })
    animal_df = pd.DataFrame(animal_res)
    animal_df.to_csv(OUT / "animal_level_condition_effect.csv", index=False)
    print("=== ANIMAL-LEVEL CONDITION EFFECT (primary) ===")
    print(animal_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Unit-level condition effect (descriptive; nesting inflates df) ----
    unit_res = []
    for m in METRICS:
        ref = u.loc[u["condition"] == "reference", m].values
        pert = u.loc[u["condition"] == "matrix_perturbed", m].values
        t, p, d = welch(ref, pert)
        unit_res.append({
            "metric": m,
            "ref_n_units": len(ref), "pert_n_units": len(pert),
            "ref_unit_mean": float(np.mean(ref)),
            "pert_unit_mean": float(np.mean(pert)),
            "diff_pert_minus_ref": float(np.mean(pert) - np.mean(ref)),
            "t": t, "p_welch_unit": p, "cohens_d": d,
        })
    unit_df = pd.DataFrame(unit_res)
    unit_df.to_csv(OUT / "unit_level_condition_effect.csv", index=False)
    print("=== UNIT-LEVEL CONDITION EFFECT (descriptive; nested) ===")
    print(unit_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Phase effects ----
    pe = []
    for phase, sub in u.groupby("session_phase"):
        pe.append({"phase": phase, "n": len(sub),
                   "mean_rate": float(sub["average_rate_hz"].mean()),
                   "std_rate": float(sub["average_rate_hz"].std(ddof=1)) if len(sub) > 1 else np.nan,
                   "mean_reg": float(sub["spatial_regularity_score"].mean()),
                   "mean_info": float(sub["information_rate"].mean()),
                   "mean_speed_mod": float(sub["speed_modulation_score"].mean())})
    pe_df = pd.DataFrame(pe)
    pe_df.to_csv(OUT / "phase_effect_overall.csv", index=False)
    print("=== PHASE EFFECT (overall, descriptive) ===")
    print(pe_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    pe_c = []
    for (cond, phase), sub in u.groupby(["condition", "session_phase"]):
        pe_c.append({"condition": cond, "phase": phase, "n": len(sub),
                     "mean_rate": float(sub["average_rate_hz"].mean()),
                     "mean_reg": float(sub["spatial_regularity_score"].mean()),
                     "mean_info": float(sub["information_rate"].mean())})
    pe_c_df = pd.DataFrame(pe_c)
    pe_c_df.to_csv(OUT / "phase_effect_by_condition.csv", index=False)
    print("=== PHASE EFFECT BY CONDITION (descriptive) ===")
    print(pe_c_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Tracked-unit stability: unit-level + per-animal ----
    stab_unit = []
    for m in STAB_METRICS:
        ref = s.loc[s["condition"] == "reference", m].values
        pert = s.loc[s["condition"] == "matrix_perturbed", m].values
        t, p, d = welch(ref, pert)
        p_perm = permutation_test(ref, pert)
        stab_unit.append({
            "metric": m,
            "ref_n_units": len(ref), "pert_n_units": len(pert),
            "ref_unit_mean": float(np.mean(ref)),
            "pert_unit_mean": float(np.mean(pert)),
            "diff_pert_minus_ref": float(np.mean(pert) - np.mean(ref)),
            "t": t, "p_welch_unit": p, "p_perm": p_perm, "cohens_d": d,
        })
    stab_unit_df = pd.DataFrame(stab_unit)
    stab_unit_df.to_csv(OUT / "stability_unit_level_effect.csv", index=False)
    print("=== STABILITY UNIT-LEVEL EFFECT (tracked units; nested) ===")
    print(stab_unit_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # per-animal mean of stability metrics
    s_animal = s.groupby(["animal_id", "condition"])[STAB_METRICS].mean().reset_index()
    stab_animal_res = []
    for m in STAB_METRICS:
        ref = s_animal.loc[s_animal["condition"] == "reference", m].values
        pert = s_animal.loc[s_animal["condition"] == "matrix_perturbed", m].values
        if len(ref) < 2 or len(pert) < 2:
            stab_animal_res.append({"metric": m, "ref_n": len(ref), "pert_n": len(pert),
                                    "ref_mean": float(np.mean(ref)) if len(ref) else np.nan,
                                    "pert_mean": float(np.mean(pert)) if len(pert) else np.nan,
                                    "diff": np.nan, "p_welch": np.nan})
            continue
        t, p, d = welch(ref, pert)
        stab_animal_res.append({"metric": m, "ref_n": len(ref), "pert_n": len(pert),
                                "ref_mean": float(np.mean(ref)),
                                "pert_mean": float(np.mean(pert)),
                                "diff": float(np.mean(pert) - np.mean(ref)),
                                "p_welch": p})
    stab_animal_df = pd.DataFrame(stab_animal_res)
    stab_animal_df.to_csv(OUT / "stability_animal_level_effect.csv", index=False)
    print("=== STABILITY ANIMAL-LEVEL EFFECT (primary for stability) ===")
    print(stab_animal_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Whole-animal leave-out robustness (unit-session key metrics) ----
    loao_rows = []
    for m in METRICS:
        ref = u_animal.loc[u_animal["condition"] == "reference", m].values
        pert = u_animal.loc[u_animal["condition"] == "matrix_perturbed", m].values
        rows, full_diff, sign = loao_animal_level(ref, pert)
        for r in rows.to_dict("records"):
            r["metric"] = m
            loao_rows.append(r)
    loao_df = pd.DataFrame(loao_rows)
    loao_df.to_csv(OUT / "whole_animal_leave_out.csv", index=False)
    print("=== WHOLE-ANIMAL LEAVE-OUT (influence) — sign flips summary ===")
    loao_summary = loao_df.groupby("metric").agg(
        full_diff=("full_diff", "first"),
        n_animals=("left_out", "count"),
        n_sign_flips=("sign_flip", "sum"),
        max_abs_delta=("delta", lambda x: float(np.abs(x).max())),
    )
    print(loao_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Outlier trimming (10% within each condition) ----
    trim = []
    for m in METRICS:
        ref = u.loc[u["condition"] == "reference", m]
        pert = u.loc[u["condition"] == "matrix_perturbed", m]
        ref_lo, ref_hi = ref.quantile([0.10, 0.90])
        pert_lo, pert_hi = pert.quantile([0.10, 0.90])
        ref_t = ref[(ref >= ref_lo) & (ref <= ref_hi)]
        pert_t = pert[(pert >= pert_lo) & (pert <= pert_hi)]
        t, p, d = welch(ref_t, pert_t)
        trim.append({"metric": m,
                     "ref_mean": float(np.mean(ref)), "pert_mean": float(np.mean(pert)),
                     "diff": float(np.mean(pert) - np.mean(ref)),
                     "ref_trim_mean": float(np.mean(ref_t)), "pert_trim_mean": float(np.mean(pert_t)),
                     "diff_trim": float(np.mean(pert_t) - np.mean(ref_t)),
                     "n_ref_trim": len(ref_t), "n_pert_trim": len(pert_t),
                     "p_welch": p})
    trim_df = pd.DataFrame(trim)
    trim_df.to_csv(OUT / "outlier_trim_robustness.csv", index=False)
    print("=== OUTLIER TRIMMING (10% within condition) ROBUSTNESS ===")
    print(trim_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Correlations within condition ----
    corr = []
    for cond, sub in u.groupby("condition"):
        pairs = [
            ("spatial_regularity_score", "speed_modulation_score"),
            ("information_rate", "spatial_regularity_score"),
            ("average_rate_hz", "in_field_rate_hz"),
            ("in_field_rate_hz", "out_of_field_rate_hz"),
            ("information_rate", "average_rate_hz"),
        ]
        for a, b in pairs:
            r, p = stats.pearsonr(sub[a], sub[b])
            corr.append({"condition": cond, "var_a": a, "var_b": b,
                         "corr": float(r), "p": float(p), "n": len(sub)})
    corr_df = pd.DataFrame(corr)
    corr_df.to_csv(OUT / "within_condition_correlations.csv", index=False)
    print("=== WITHIN-CONDITION CORRELATIONS ===")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ---- Master results JSON ----
    master = {
        "design": design,
        "animal_level_key": {
            "information_rate": animal_df.loc[animal_df.metric == "information_rate"].iloc[0].to_dict(),
            "spatial_regularity_score": animal_df.loc[animal_df.metric == "spatial_regularity_score"].iloc[0].to_dict(),
            "average_rate_hz": animal_df.loc[animal_df.metric == "average_rate_hz"].iloc[0].to_dict(),
            "orientation": animal_df.loc[animal_df.metric == "orientation"].iloc[0].to_dict(),
        },
    }
    with open(OUT / "results_summary.json", "w") as f:
        json.dump(master, f, indent=2, default=float)
    print("=== MASTER SUMMARY (key metrics) ===")
    print(json.dumps(master["animal_level_key"], indent=2, default=float))
    print()

    make_figures(u, s, animal_df, stab_unit_df, pe_df, pe_c_df)

    print("Saved files:")
    for p in sorted(OUT.iterdir()):
        print("  results/", p.name)
    for p in sorted(FIG.iterdir()):
        print("  figures/", p.name)
    print("Analysis complete.")


def make_figures(u, s, animal_df, stab_unit_df, pe_df, pe_c_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures.")
        return

    # Figure 1: condition boxplots of key metrics (unit level)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["average_rate_hz", "spatial_regularity_score", "information_rate"]):
        data = [u.loc[u.condition == c, m].values for c in ["reference", "matrix_perturbed"]]
        ax.boxplot(data, tick_labels=["reference", "matrix_perturbed"])
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "condition_boxplots.png", dpi=150)
    plt.close()

    # Figure 2: phase effect lines (by condition)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["average_rate_hz", "spatial_regularity_score"]):
        for cond in ["reference", "matrix_perturbed"]:
            sub = pe_c_df[(pe_c_df.condition == cond) & (pe_c_df.phase.isin(PHASES))].set_index("phase").reindex(PHASES)
            ax.plot(sub.index, sub["mean_rate" if m == "average_rate_hz" else "mean_reg"],
                    marker="o", label=cond)
        ax.set_title(m)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG / "phase_effect.png", dpi=150)
    plt.close()

    # Figure 3: stability by condition (tracked units)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["between_repeated_environment_stability",
                             "repeated_environment_change",
                             "novel_vs_repeated_change"]):
        data = [s.loc[s.condition == c, m].values for c in ["reference", "matrix_perturbed"]]
        ax.boxplot(data, tick_labels=["reference", "matrix_perturbed"])
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "stability_by_condition.png", dpi=150)
    plt.close()

    # Figure 4: animal-level bar of key metric differences with CI
    fig, ax = plt.subplots(figsize=(10, 4))
    key = ["information_rate", "spatial_regularity_score", "average_rate_hz", "orientation"]
    x = np.arange(len(key))
    diffs, err = [], []
    for i, m in enumerate(key):
        row = animal_df[animal_df.metric == m].iloc[0]
        diffs.append(row["diff_pert_minus_ref"])
        err.append(abs(row["ci_hi"] - row["ci_lo"]) / 2)
    ax.bar(x, diffs, yerr=err, capsize=5, color=["#1f77b4"] * len(key))
    ax.set_xticks(x)
    ax.set_xticklabels(key, rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("perturbed - reference")
    ax.set_title("Animal-level condition differences (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "animal_level_diffs.png", dpi=150)
    plt.close()

    print("Figures saved to", FIG)


if __name__ == "__main__":
    main()

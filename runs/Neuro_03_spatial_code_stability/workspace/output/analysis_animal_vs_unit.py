#!/usr/bin/env python3
"""
Neuro_03_spatial_code_stability — STEP: animal-level (independent unit) vs
unit-level (nested) condition effects for the 8 unit-session metrics, plus a
whole-animal leave-out (LOAO) robustness check.

Design (verified from data):
  * animal_id is the INDEPENDENT analysis unit.
  * unit/observation rows (unit_session_*) are nested within animals.
  * Animals are split between conditions (no animal appears in both).

Primary (animal-level) tests per metric:
  * per-animal mean within each condition
  * Welch two-sample t-test across animals
  * Welch-Satterthwaite 95% CI on (pert_mean - ref_mean)
  * Cohen's d (pooled SD)
  * combined-pool permutation test: seed 20260922, 10000 perms, two-sided on
    |diff|, p = (count + 1) / (n_perm + 1)

Descriptive (unit-level, nested) tests per metric:
  * Welch t-test over all observation rows, Welch p, Cohen's d

Robustness (whole-animal leave-out):
  * for each animal removed, recompute the animal-level mean difference and
    flag whether its sign flips relative to the full-sample diff.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = "/data/user/conversations/conv-7eebfb61-13ed-4509-8100-c75a3d34d930/project"
DATA = os.path.join(BASE, "data/unit_session_metrics.tsv")
OUT = os.path.join(BASE, "output")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")

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

SEED = 20260922
N_PERM = 10000


def welch_two_sample(x, y):
    """Welch two-sample t-test; return (t, p)."""
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return float(t), float(p)


def welch_ci(x, y, alpha=0.05):
    """Welch-Satterthwaite 95% CI on (mean(y) - mean(x))."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    se = np.sqrt(sx**2 / nx + sy**2 / ny)
    df = (sx**2 / nx + sy**2 / ny) ** 2 / (
        (sx**2 / nx) ** 2 / (nx - 1) + (sy**2 / ny) ** 2 / (ny - 1)
    )
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    diff = my - mx
    return float(diff), float(diff - tcrit * se), float(diff + tcrit * se)


def cohens_d(x, y):
    """Cohen's d with pooled SD: (mean(y)-mean(x)) / SP."""
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    return float((np.mean(y) - np.mean(x)) / sp)


def permutation_p(x, y, seed=SEED, n_perm=N_PERM):
    """
    Combined-pool permutation test on |diff|.
    p = (count + 1) / (n_perm + 1), where count = number of permutations with
    |perm_diff| >= |obs_diff|.
    """
    rng = np.random.RandomState(seed)
    obs = np.mean(y) - np.mean(x)
    obs_abs = abs(obs)
    pool = np.concatenate([x, y])
    n = len(y)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        p_y = pool[perm[:n]]
        count += int(abs(np.mean(p_y) - np.mean(pool[n:])) >= obs_abs)
    return float((count + 1) / (n_perm + 1))


def main():
    df = pd.read_csv(DATA, sep="\t")

    # ---------- 1. Design verification ----------
    animals = sorted(df["animal_id"].unique())
    cond_of = {a: set(df.loc[df.animal_id == a, "condition"]) for a in animals}
    overlap = [a for a in animals if len(cond_of[a]) > 1]
    ref_animals = sorted(a for a in animals if "reference" in cond_of[a])
    pert_animals = sorted(a for a in animals if "matrix_perturbed" in cond_of[a])
    print("Design check:")
    print(f"  distinct animals: {len(animals)}")
    print(f"  reference animals: {ref_animals} ({len(ref_animals)})")
    print(f"  matrix_perturbed animals: {pert_animals} ({len(pert_animals)})")
    print(f"  animals in BOTH conditions: {overlap if overlap else 'NONE'}")

    # ---------- 2. Per-metric computations ----------
    animal_rows, unit_rows, loao_rows = [], [], []

    for m in METRICS:
        x = df.loc[df.condition == "reference", m].values
        y = df.loc[df.condition == "matrix_perturbed", m].values

        # (a) ANIMAL LEVEL (primary): per-animal means
        per_animal = df.groupby(["animal_id", "condition"])[m].transform("mean")
        ax = df.groupby(["animal_id", "condition"])[m].mean().unstack(level=1)
        ax = ax[["reference", "matrix_perturbed"]].dropna(how="all")
        ax_ref = ax.loc[ax.index.isin(ref_animals), "reference"].values
        ax_pert = ax.loc[ax.index.isin(pert_animals), "matrix_perturbed"].values

        t_a, p_a = welch_two_sample(ax_ref, ax_pert)
        diff_a, ci_lo, ci_hi = welch_ci(ax_ref, ax_pert)
        d_a = cohens_d(ax_ref, ax_pert)
        p_perm = permutation_p(ax_ref, ax_pert)

        animal_rows.append(
            {
                "metric": m,
                "ref_n_animals": len(ax_ref),
                "pert_n_animals": len(ax_pert),
                "ref_animal_mean": float(np.mean(ax_ref)),
                "pert_animal_mean": float(np.mean(ax_pert)),
                "diff_pert_minus_ref": diff_a,
                "t": t_a,
                "p_welch": p_a,
                "p_perm": p_perm,
                "cohens_d": d_a,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )

        # (b) UNIT LEVEL (descriptive, nested): all observation rows
        t_u, p_u = welch_two_sample(x, y)
        d_u = cohens_d(x, y)
        unit_rows.append(
            {
                "metric": m,
                "ref_n_units": len(x),
                "pert_n_units": len(y),
                "ref_unit_mean": float(np.mean(x)),
                "pert_unit_mean": float(np.mean(y)),
                "diff_pert_minus_ref": float(np.mean(y) - np.mean(x)),
                "t": t_u,
                "p_welch": p_u,
                "cohens_d": d_u,
            }
        )

        # (c) WHOLE-ANIMAL LEAVE-OUT
        full_diff = diff_a
        for a in animals:
            mask = df.animal_id != a
            ax_lo = df[mask].groupby(["animal_id", "condition"])[m].mean().unstack(level=1)
            ax_lo = ax_lo[["reference", "matrix_perturbed"]].dropna(how="all")
            lo_ref = ax_lo.loc[ax_lo.index.isin(ref_animals), "reference"]
            lo_pert = ax_lo.loc[ax_lo.index.isin(pert_animals), "matrix_perturbed"]
            lo_diff = float(np.mean(lo_pert) - np.mean(lo_ref))
            sign_flip = bool((np.sign(lo_diff) * np.sign(full_diff) < 0) if full_diff != 0 else False)
            loao_rows.append(
                {
                    "metric": m,
                    "left_out": a,
                    "full_diff": full_diff,
                    "lo_diff": lo_diff,
                    "sign_flip": sign_flip,
                    "delta": float(lo_diff - full_diff),
                }
            )

    # ---------- 3. Write CSVs ----------
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(animal_rows).to_csv(
        os.path.join(RES, "animal_level_condition_effect.csv"), index=False
    )
    pd.DataFrame(unit_rows).to_csv(
        os.path.join(RES, "unit_level_condition_effect.csv"), index=False
    )
    pd.DataFrame(loao_rows).to_csv(
        os.path.join(RES, "whole_animal_leave_out.csv"), index=False
    )
    print("CSVs written to", RES)

    # ---------- 4. Markdown summary ----------
    lines = [
        "# Animal-level vs unit-level condition effects (Neuro_03_spatial_code_stability)",
        "",
        "## Design verification",
        f"- Distinct animals: **{len(animals)}**",
        f"- Reference animals: **{len(ref_animals)}** ({', '.join(ref_animals)})",
        f"- Matrix-perturbed animals: **{len(pert_animals)}** ({', '.join(pert_animals)})",
        f"- Animals appearing in BOTH conditions: **{len(overlap)}** (none)",
        "No animal is in both conditions; animals are the independent analysis unit and "
        "unit-session rows are nested observations. Primary tests use per-animal means "
        "followed by two-sample tests across animals. Unit-level tests are descriptive "
        "only (nested) and must not be treated as independent samples.",
        "",
        "## Method",
        "- Animal-level (primary): per-animal mean within each condition, then Welch "
        "two-sample t-test (unequal variance), Welch-Satterthwaite 95% CI, Cohen's d "
        "(pooled SD), and a combined-pool permutation test (seed 20260922, 10000 "
        "permutations, two-sided on |diff|, p = (count+1)/(n_perm+1)).",
        "- Unit-level (descriptive, nested): Welch t-test over all observation rows.",
        "- Robustness: whole-animal leave-out (LOAO); for each animal left out, the "
        "animal-level mean difference is recomputed and flagged if its sign flips "
        "relative to the full-sample diff.",
        "",
        "## Results",
        "",
    ]

    # Load the computed frames (same as the CSVs) to build the summary
    adf = pd.DataFrame(animal_rows)
    udf = pd.DataFrame(unit_rows)
    loao_df = pd.DataFrame(loao_rows)

    for m in METRICS:
        ra = adf[adf.metric == m].iloc[0]
        ru = udf[udf.metric == m].iloc[0]
        sign_match = np.sign(ru.diff_pert_minus_ref) == np.sign(ra.diff_pert_minus_ref)
        survives = bool(ra.p_welch < 0.05 and sign_match)
        loao_flips = int(loao_df[(loao_df.metric == m) & (loao_df.sign_flip == True)].shape[0])
        lines.append(f"### {m}")
        lines.append(
            f"- Ref animal mean: {ra.ref_animal_mean:.4f} "
            f"(n={ra.ref_n_animals} animals); perturbed animal mean: "
            f"{ra.pert_animal_mean:.4f} (n={ra.pert_n_animals} animals)."
        )
        lines.append(
            f"- Animal-level diff (pert - ref): {ra.diff_pert_minus_ref:.4f}; "
            f"Welch p = {ra.p_welch:.4g}; permutation p = {ra.p_perm:.4g}; "
            f"Cohen's d = {ra.cohens_d:.3f}; 95% CI [{ra.ci_lo:.4f}, {ra.ci_hi:.4f}]."
        )
        lines.append(
            f"- Unit-level (nested, descriptive) diff: {ru.diff_pert_minus_ref:.4f}; "
            f"Welch p = {ru.p_welch:.4g}; Cohen's d = {ru.cohens_d:.3f}."
        )
        lines.append(
            f"- Unit-level effect SURVIVES at the animal level: **{str(survives)}** "
            f"(animal-level p < 0.05: {ra.p_welch < 0.05}; sign matches unit-level: "
            f"{sign_match})."
        )
        lines.append(
            f"- Whole-animal leave-out: {loao_flips}/{len(animals)} animals flipped the "
            f"sign of the animal-level diff when removed (full diff = {ra.diff_pert_minus_ref:.4f})."
        )
        lines.append("")

    with open(os.path.join(RES, "animal_vs_unit_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("Markdown summary written.")

    # ---------- 5. Figure ----------
    os.makedirs(FIG, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ys = np.arange(len(METRICS))
        bars = ax.barh(
            ys,
            [r["diff_pert_minus_ref"] for r in animal_rows],
            color="steelblue",
            align="center",
        )
        ax.axhline(0, color="k", lw=0.8, label="zero")
        for r, y in zip(animal_rows, ys):
            ax.plot(
                [r["diff_pert_minus_ref"], r["diff_pert_minus_ref"]],
                [y - 0.35, y + 0.35],
                color="k",
                lw=0.8,
            )
            ax.plot(
                [r["ci_lo"], r["ci_hi"]],
                [r["diff_pert_minus_ref"], r["diff_pert_minus_ref"]],
                color="k",
                lw=0.8,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(METRICS, rotation=0)
        ax.set_xlabel("Animal-level mean difference (perturbed - reference) + 95% CI")
        ax.set_title("Animal-level condition effect with Welch-Satterthwaite 95% CI")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "animal_level_diffs.png"), dpi=150)
        print("Figure saved to", os.path.join(FIG, "animal_level_diffs.png"))
    except Exception as e:
        print(f"MATPLOTLIB FAILED: {type(e).__name__}: {e}")

    # ---------- 6. Console digest ----------
    print("\n=== ANIMAL LEVEL (primary) ===")
    for r in animal_rows:
        print(
            f"{r['metric']:<28} ref={r['ref_animal_mean']:.4f} "
            f"pert={r['pert_animal_mean']:.4f} diff={r['diff_pert_minus_ref']:.4f} "
            f"p_welch={r['p_welch']:.4g} p_perm={r['p_perm']:.4g} "
            f"d={r['cohens_d']:.3f}"
        )
    print("\n=== UNIT LEVEL (descriptive, nested) ===")
    for r in unit_rows:
        print(
            f"{r['metric']:<28} ref={r['ref_unit_mean']:.4f} "
            f"pert={r['pert_unit_mean']:.4f} diff={r['diff_pert_minus_ref']:.4g} "
            f"p_welch={r['p_welch']:.4g} d={r['cohens_d']:.3f}"
        )
    print("\n=== LOAO sign-flip counts (by metric) ===")
    for m in METRICS:
        n = int(loao_df[(loao_df.metric == m) & (loao_df.sign_flip == True)].shape[0])
        print(f"{m:<28} {n}/{len(animals)} animals flipped sign")


if __name__ == "__main__":
    main()

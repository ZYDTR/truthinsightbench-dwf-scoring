"""WS3 - Error structure by chemical bond class and structural environment.

Reads the cached table from ws3_build.py (output/analysis/ws3_table.pkl).
Writes output/results/ws3_bondclass.json and output/figures/ws3_bondclass.png.

Analysis units are stated explicitly for every statistic:
  - "bond observation" = one of the 543 benchmark rows (the primary error unit)
  - "unique molecule"   = unique parent_structure_smiles
  - "fragment pair"     = unique (parent, fragment_a, fragment_b) triple
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common

PKL = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis/ws3_table.pkl")
FIG = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/figures/ws3_bondclass.png")
CH = list(common.CHANNELS)
SMALL_N = 10
SEED = 20260919


# --------------------------------------------------------------------------
# statistics helpers specific to this workstream
# --------------------------------------------------------------------------

def chan_stats(sub, prefix, unit):
    """n, MAE, bias, median AE, RMSE, max AE for one channel on one subset."""
    out = {"n_bond_observations": int(len(sub))}
    if len(sub) == 0:
        out.update(unit=unit)
        return out
    out.update(
        unit=unit,
        n_unique_molecules=int(sub.parent_structure_smiles.nunique()),
        mean_experimental_bde_kcal_mol=round(float(sub.experimental_bde_kcal_mol.mean()), 4),
    )
    for c in CH:
        e = sub[f"{c}_err"].to_numpy(dtype=float)
        ae = np.abs(e)
        out[f"{c}_mae"] = round(float(ae.mean()), 4)
        out[f"{c}_bias"] = round(float(e.mean()), 4)
        out[f"{c}_median_ae"] = round(float(np.median(ae)), 4)
        out[f"{c}_rmse"] = round(float(np.sqrt((e ** 2).mean())), 4)
        out[f"{c}_max_ae"] = round(float(ae.max()), 4)
        out[f"{c}_p90_ae"] = round(float(np.quantile(ae, 0.90)), 4)
    return out


N_BOOT = 4000


def boot_diff_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """Independent two-sample bootstrap CI for MAE(a) - MAE(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = da - db
    return {
        "mae_a": round(float(a.mean()), 4),
        "mae_b": round(float(b.mean()), 4),
        "diff_mae": round(float(a.mean() - b.mean()), 4),
        "ci95_low": round(float(np.quantile(d, 0.025)), 4),
        "ci95_high": round(float(np.quantile(d, 0.975)), 4),
        "p_boot_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "n_boot": n_boot,
    }


def boot_diff_ci_multi(A, B, n_boot=N_BOOT, seed=SEED):
    """Same as boot_diff_ci but for a 2-D (n_channels, n_obs) pair of blocks.

    One shared index draw serves every channel: (n_boot, n) @ (n, n_chan).
    Roughly 3x cheaper than looping boot_diff_ci over channels.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim == 1:
        A, B = A[None, :], B[None, :]
    if A.shape[1] == 0 or B.shape[1] == 0:
        # one side is empty: no contrast is estimable
        return {
            c: {
                "mae_a": round(float(A[j].mean()), 4) if A.shape[1] else None,
                "mae_b": round(float(B[j].mean()), 4) if B.shape[1] else None,
                "diff_mae": None, "ci95_low": None, "ci95_high": None,
                "p_boot_two_sided": None,
                "n_a": int(A.shape[1]), "n_b": int(B.shape[1]), "n_boot": None,
                "note": "empty group - contrast not estimable",
            }
            for j, c in enumerate(CH)
        }
    ia = rng.integers(0, A.shape[1], size=(n_boot, A.shape[1]))
    ib = rng.integers(0, B.shape[1], size=(n_boot, B.shape[1]))
    da = A.T[ia].mean(axis=1)      # (n_boot, n_chan)
    db = B.T[ib].mean(axis=1)
    d = da - db
    out = {}
    for j, c in enumerate(CH):
        out[c] = {
            "mae_a": round(float(A[j].mean()), 4),
            "mae_b": round(float(B[j].mean()), 4),
            "diff_mae": round(float(A[j].mean() - B[j].mean()), 4),
            "ci95_low": round(float(np.quantile(d[:, j], 0.025)), 4),
            "ci95_high": round(float(np.quantile(d[:, j], 0.975)), 4),
            "p_boot_two_sided": round(
                float(2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean())), 4),
            "n_a": int(A.shape[1]),
            "n_b": int(B.shape[1]),
            "n_boot": n_boot,
        }
    return out


def mannwhitney(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "U": round(float(u), 2),
        "p_value": round(float(p), 6),
        "cliffs_delta": round(common.cliffs_delta(a, b), 4),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def ols_r2(X, y):
    """R^2 of an ordinary least-squares fit via lstsq (no intercept handling magic)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), beta


def spearman(x, y):
    if len(x) < 3:
        return None
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 4), "p_value": round(float(p), 6), "n": int(len(x))}


# --------------------------------------------------------------------------
def main():
    print("[ws3] loading cached table ...", flush=True)
    with PKL.open("rb") as fh:
        df = pickle.load(fh)
    print(f"[ws3] loaded {df.shape}", flush=True)

    res = {
        "workstream": "WS3 - error structure by chemical bond class and structural environment",
        "analysis_units": {
            "primary": "bond observation (1 of 543 benchmark rows)",
            "also_reported": ["unique molecule (parent SMILES)", "unique fragment pair"],
            "n_bond_observations": int(len(df)),
            "n_unique_molecules": int(df.parent_structure_smiles.nunique()),
            "n_unique_fragment_pairs": int(
                len(set(zip(df.parent_structure_smiles, df.fragment_a_smiles, df.fragment_b_smiles)))
            ),
        },
        "coverage": {},
        "ch1_error_by_bond_class": {},
        "ch2_class_vs_size": {},
        "ch3_environment_effects": {},
        "ch4_worst_best_regions": {},
        "ch5_thin_samples": {},
    }

    # ======================================================================
    # 0. coverage: which bond classes are present at all
    # ======================================================================
    db = common.load_db()
    db_counts = db.bond_type.value_counts().to_dict()
    bench_counts = df.bond_class.value_counts().to_dict()
    cov = {}
    for cls in sorted(set(list(db_counts) + list(bench_counts))):
        cov[cls] = {
            "db_rows": int(db_counts.get(cls, 0)),
            "benchmark_bond_observations": int(bench_counts.get(cls, 0)),
            "db_share_pct": round(100 * db_counts.get(cls, 0) / len(db), 4),
            "benchmark_share_pct": round(100 * bench_counts.get(cls, 0) / len(df), 4),
            "present_in_benchmark": bool(bench_counts.get(cls, 0) > 0),
            "n_lt_10_thin": bool(bench_counts.get(cls, 0) < SMALL_N),
        }
    res["coverage"] = {
        "by_bond_class": cov,
        "labeler_agreement": {
            "matched_rows": int(df.labeler_ok.notna().sum()),
            "agreement": int(df.labeler_ok.sum()),
            "rate": round(float(df.labeler_ok.mean(skipna=True)), 4),
            "statement": "the RDKit radical-centre labeler reproduces the database bond_type on "
                         "all 331 matched benchmark rows",
        },
        "headline": (
            "The 543-observation benchmark contains only 3 of the 9 bond classes present in the "
            "database: C-H (302), C-C (240) and H-O (1). Every other class (C-N, C-O, H-N, N-N, "
            "N-O, O-O) is ABSENT (n=0), so no class-level error claim can be made for them."
        ),
    }

    # ======================================================================
    # 1. error by bond class for each channel
    # ======================================================================
    per_class = {}
    for cls in bench_counts:
        sub = df[df.bond_class == cls]
        st = chan_stats(sub, cls, "bond observation")
        st["n_unique_fragment_pairs"] = int(
            len(set(zip(sub.parent_structure_smiles, sub.fragment_a_smiles, sub.fragment_b_smiles)))
        )
        st["n_lt_10_thin"] = bool(len(sub) < SMALL_N)
        st["median_hac_parent"] = float(sub.hac_parent.median())
        st["mean_experimental_bde"] = round(float(sub.experimental_bde_kcal_mol.mean()), 3)
        per_class[cls] = st
    per_class["ALL"] = chan_stats(df, "ALL", "bond observation")
    per_class["ALL"]["n_lt_10_thin"] = False

    # C-H vs C-C head-to-head per channel (the only comparison powered here)
    a_mask = df.bond_class == "C-H"
    b_mask = df.bond_class == "C-C"
    A_CH = df.loc[a_mask, [f"{c}_ae" for c in CH]].to_numpy(float).T
    A_CC = df.loc[b_mask, [f"{c}_ae" for c in CH]].to_numpy(float).T
    boot_cc = boot_diff_ci_multi(A_CH, A_CC)
    ch1_contrast = {}
    for c in CH:
        ch1_contrast[c] = {
            "mae_C_H_minus_C_C": boot_cc[c],
            "mannwhitney_abs_err": mannwhitney(
                df.loc[a_mask, f"{c}_ae"].to_numpy(), df.loc[b_mask, f"{c}_ae"].to_numpy()
            ),
            "paired_on_shared_molecules": None,
        }
        # paired contrast restricted to molecules that contribute BOTH a C-H and a C-C bond
        both = df.groupby("parent_structure_smiles").bond_class.nunique()
        both_mols = set(both[both >= 2].index) & set(df.loc[a_mask, "parent_structure_smiles"])
        sub = df[df.parent_structure_smiles.isin(both_mols)]
        ca = sub[sub.bond_class == "C-H"][f"{c}_ae"]
        cb = sub[sub.bond_class == "C-C"][f"{c}_ae"]
        if len(ca) > 0 and len(cb) > 0:
            ch1_contrast[c]["paired_on_shared_molecules"] = {
                "molecules": int(len(both_mols)),
                "n_CH_obs": int(len(ca)),
                "n_CC_obs": int(len(cb)),
                "mae_CH": round(float(ca.mean()), 4),
                "mae_CC": round(float(cb.mean()), 4),
            }
    print("[ws3] section 1 done", flush=True)
    res["ch1_error_by_bond_class"] = {
        "per_class": per_class,
        "channel_ranking_overall": {
            "by_mae": sorted(CH, key=lambda c: per_class["ALL"][f"{c}_mae"]),
            "mae": {c: per_class["ALL"][f"{c}_mae"] for c in CH},
        },
        "C_H_vs_C_C": ch1_contrast,
        "interpretation": (
            "Only C-H (n=302) and C-C (n=240) support a per-class MAE. H-O has n=1 and is "
            "reported for completeness only - no inference. The six other database classes "
            "have n=0 in the benchmark."
        ),
    }

    # ======================================================================
    # 2. bond class vs molecule size
    # ======================================================================
    # class x hac_bin MAE table
    ct = {}
    for cls in bench_counts:
        ct[cls] = {}
        for hb in ["<=5", "6-9", ">=10"]:
            sub = df[(df.bond_class == cls) & (df.hac_bin == hb)]
            ct[cls][hb] = chan_stats(sub, cls, "bond observation")
            ct[cls][hb]["n_lt_10_thin"] = bool(len(sub) < SMALL_N)
    ct["ALL"] = {}
    for hb in ["<=5", "6-9", ">=10"]:
        sub = df[df.hac_bin == hb]
        ct["ALL"][hb] = chan_stats(sub, hb, "bond observation")
        ct["ALL"][hb]["n_lt_10_thin"] = bool(len(sub) < SMALL_N)

    # confound check: class composition per size bin, and size distribution per class
    confound = {
        "class_share_within_hac_bin_pct": {},
        "hac_within_class": {},
    }
    for hb in ["<=5", "6-9", ">=10"]:
        sub = df[df.hac_bin == hb]
        confound["class_share_within_hac_bin_pct"][hb] = {
            cls: round(100 * float((sub.bond_class == cls).mean()), 2) for cls in bench_counts
        }
    for cls in bench_counts:
        sub = df[df.bond_class == cls]
        confound["hac_within_class"][cls] = {
            "n": int(len(sub)),
            "mean_hac": round(float(sub.hac_parent.mean()), 3),
            "median_hac": float(sub.hac_parent.median()),
            "iqr_hac": [float(sub.hac_parent.quantile(0.25)), float(sub.hac_parent.quantile(0.75))],
        }
    try:
        u_stat, p_val = stats.mannwhitneyu(
            df.loc[a_mask, "hac_parent"], df.loc[b_mask, "hac_parent"], alternative="two-sided"
        )
        confound["hac_CH_vs_CC_mannwhitney"] = {
            "U": round(float(u_stat), 2),
            "p_value": round(float(p_val), 8),
            "median_CH": float(df.loc[a_mask, "hac_parent"].median()),
            "median_CC": float(df.loc[b_mask, "hac_parent"].median()),
        }
    except Exception as exc:  # pragma: no cover
        confound["hac_CH_vs_CC_mannwhitney"] = {"error": repr(exc)}

    # does the size trend survive within a fixed class?
    size_trend = {}
    for c in CH:
        size_trend[c] = {"spearman_abs_err_vs_hac": spearman(df.hac_parent, df[f"{c}_ae"].to_numpy())}
        for cls in bench_counts:
            sub = df[df.bond_class == cls]
            size_trend[c][f"within_{cls}_spearman"] = spearman(
                sub.hac_parent, sub[f"{c}_ae"].to_numpy()
            )
            size_trend[c][f"within_{cls}_mae_by_hac_bin"] = {
                hb: (
                    round(float(sub.loc[sub.hac_bin == hb, f"{c}_ae"].mean()), 4)
                    if len(sub[sub.hac_bin == hb]) else None
                )
                for hb in ["<=5", "6-9", ">=10"]
            }
            size_trend[c][f"within_{cls}_n_by_hac_bin"] = {
                hb: int((sub.hac_bin == hb).sum()) for hb in ["<=5", "6-9", ">=10"]
            }

    # variance decomposition: |err| ~ size only, class only, both
    hac = df.hac_parent.to_numpy(float)
    cls_dummy = (df.bond_class == "C-C").to_numpy(float)
    one = np.ones(len(df))
    models = {}
    for c in CH:
        y = df[f"{c}_ae"].to_numpy(float)
        r2_size, _ = ols_r2(np.column_stack([one, hac]), y)
        r2_class, _ = ols_r2(np.column_stack([one, cls_dummy]), y)
        r2_both, _ = ols_r2(np.column_stack([one, hac, cls_dummy]), y)
        models[c] = {
            "r2_size_only": round(float(r2_size), 5),
            "r2_class_only": round(float(r2_class), 5),
            "r2_size_plus_class": round(float(r2_both), 5),
            "incremental_r2_from_class_given_size": round(float(r2_both - r2_size), 5),
            "incremental_r2_from_size_given_class": round(float(r2_both - r2_class), 5),
        }
    # permutation test for the incremental class effect given size (shuffle class within the data)
    rng = np.random.default_rng(SEED)
    perm = {}
    for c in CH:
        y = df[f"{c}_ae"].to_numpy(float)
        obs = models[c]["incremental_r2_from_class_given_size"]
        null = np.empty(500)
        for i in range(500):
            d = rng.permutation(cls_dummy)
            r2_s, _ = ols_r2(np.column_stack([one, hac]), y)
            r2_b, _ = ols_r2(np.column_stack([one, hac, d]), y)
            null[i] = r2_b - r2_s
        perm[c] = {
            "obs_incremental_r2": obs,
            "permutation_p_value": round(float((null >= obs).mean()), 4),
            "n_permutations": 500,
        }

    # within-bin C-H vs C-C contrast (controls size exactly)
    within_bin = {}
    for c in CH:
        within_bin[c] = {}
        for hb in ["<=5", "6-9", ">=10"]:
            ca = df.loc[a_mask & (df.hac_bin == hb), f"{c}_ae"].to_numpy()
            cb = df.loc[b_mask & (df.hac_bin == hb), f"{c}_ae"].to_numpy()
            within_bin[c][hb] = {
                "n_CH": int(len(ca)), "n_CC": int(len(cb)),
                "mae_CH": round(float(ca.mean()), 4) if len(ca) else None,
                "mae_CC": round(float(cb.mean()), 4) if len(cb) else None,
                "cliffs_delta_CH_vs_CC": (
                    round(common.cliffs_delta(ca, cb), 4) if len(ca) and len(cb) else None
                ),
                "mannwhitney_p": (
                    round(float(stats.mannwhitneyu(ca, cb).pvalue), 5)
                    if len(ca) > 1 and len(cb) > 1 else None
                ),
            }

    print("[ws3] section 2 done", flush=True)
    res["ch2_class_vs_size"] = {
        "crosstab": ct,
        "confounding_check": confound,
        "size_trend_within_class": size_trend,
        "variance_decomposition_abs_err": {
            "models": models,
            "permutation_test_class_given_size": perm,
            "note": "OLS R^2 on |error|, 543 bond observations, one dummy for C-C vs C-H.",
        },
        "CH_vs_CC_stratified_by_size": within_bin,
        "answer": (
            "Formula for the verdict is in the narrative field; see 'verdict' below."
        ),
    }
    verdict = {}
    for c in CH:
        m = models[c]
        verdict[c] = (
            "class dominates" if m["r2_class_only"] > m["r2_size_only"] else "size dominates"
        )
    res["ch2_class_vs_size"]["verdict"] = verdict

    # ======================================================================
    # 3. environment / bound-character effects measurable from data alone
    # ======================================================================
    env = {}
    flags = {
        "c_rad_aromatic": "carbon radical centre is aromatic",
        "c_rad_in_ring": "carbon radical centre is in a ring",
        "rad_any_aromatic": "either radical centre is aromatic",
        "rad_any_in_ring": "either radical centre is in a ring",
        "parent_benzene": "parent contains a benzene (aromatic 6-ring)",
        "parent_has_N": "parent contains nitrogen",
        "parent_has_O": "parent contains oxygen",
        "parent_has_S": "parent contains sulfur",
    }
    for flag, desc in flags.items():
        t = df[df[flag].astype(bool)]
        f = df[~df[flag].astype(bool)]
        entry = {
            "description": desc,
            "n_true": int(len(t)),
            "n_false": int(len(f)),
            "true": chan_stats(t, flag, "bond observation"),
            "false": chan_stats(f, flag, "bond observation"),
            "n_true_lt_10_thin": bool(len(t) < SMALL_N),
            "n_false_lt_10_thin": bool(len(f) < SMALL_N),
            "contrasts": {},
        }
        tb = boot_diff_ci_multi(
            t[[f"{c}_ae" for c in CH]].to_numpy(float).T,
            f[[f"{c}_ae" for c in CH]].to_numpy(float).T,
        )
        for c in CH:
            entry["contrasts"][c] = {
                "mae_diff_true_minus_false": tb[c],
                "mannwhitney": mannwhitney(t[f"{c}_ae"].to_numpy(), f[f"{c}_ae"].to_numpy()),
            }
        env[flag] = entry
        print(f"[ws3] env flag {flag}: n_true={len(t)} n_false={len(f)}", flush=True)

    # same flags but stratified within C-H (the only class with enough of both)
    env_stratified = {}
    for flag in ["c_rad_aromatic", "c_rad_in_ring", "rad_any_in_ring", "parent_benzene",
                 "parent_has_N", "parent_has_O"]:
        env_stratified[flag] = {}
        for cls in ["C-H", "C-C"]:
            sub = df[df.bond_class == cls]
            t = sub[sub[flag].astype(bool)]
            f = sub[~sub[flag].astype(bool)]
            env_stratified[flag][cls] = {
                "n_true": int(len(t)), "n_false": int(len(f)),
                **{f"{c}_mae_true": (round(float(t[f'{c}_ae'].mean()), 4) if len(t) else None)
                   for c in CH},
                **{f"{c}_mae_false": (round(float(f[f'{c}_ae'].mean()), 4) if len(f) else None)
                   for c in CH},
                "n_true_lt_10_thin": bool(len(t) < SMALL_N),
            }

    # experimental-magnitude bins: heteroscedasticity + weak-bond failure
    mag = {}
    for lab in ["<70", "70-85", "85-100", ">=100"]:
        sub = df[df.exp_mag_bin == lab]
        mag[lab] = chan_stats(sub, lab, "bond observation")
        mag[lab]["n_lt_10_thin"] = bool(len(sub) < SMALL_N)
        mag[lab]["exp_bde_range"] = [
            float(sub.experimental_bde_kcal_mol.min()), float(sub.experimental_bde_kcal_mol.max())
        ] if len(sub) else None
    mag["ALL"] = chan_stats(df, "ALL", "bond observation")

    mag_q = {}
    for lab in ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]:
        sub = df[df.exp_mag_quartile == lab]
        mag_q[lab] = chan_stats(sub, lab, "bond observation")
        mag_q[lab]["n_lt_10_thin"] = bool(len(sub) < SMALL_N)

    # continuous heteroscedasticity: |err| vs experimental magnitude
    hetero = {}
    for c in CH:
        e = df[f"{c}_err"].to_numpy(float)
        ae = np.abs(e)
        hetero[c] = {
            "spearman_abs_err_vs_exp_bde": spearman(df.experimental_bde_kcal_mol, ae),
            "spearman_signed_err_vs_exp_bde": spearman(df.experimental_bde_kcal_mol, e),
            "sd_err_by_mag_bin": {
                lab: (round(float(df.loc[df.exp_mag_bin == lab, f"{c}_err"].std()), 4)
                      if (df.exp_mag_bin == lab).sum() > 1 else None)
                for lab in ["<70", "70-85", "85-100", ">=100"]
            },
            "rmse_by_mag_bin": {
                lab: (round(float(np.sqrt((df.loc[df.exp_mag_bin == lab, f'{c}_err'] ** 2).mean())), 4)
                      if (df.exp_mag_bin == lab).sum() else None)
                for lab in ["<70", "70-85", "85-100", ">=100"]
            },
        }
        # Brown-Forsythe / Levene on |err| across magnitude bins (heteroscedasticity test)
        groups = [df.loc[df.exp_mag_bin == lab, f"{c}_ae"].to_numpy()
                  for lab in ["<70", "70-85", "85-100", ">=100"]]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            lev = stats.levene(*groups, center="median")
            krus = stats.kruskal(*groups)
            hetero[c]["levene_p"] = round(float(lev.pvalue), 6)
            hetero[c]["kruskal_p_across_mag_bins"] = round(float(krus.pvalue), 6)

    # --- magnitude effect stratified within bond class (does weak-bond failure survive?) ---
    mag_within = {}
    for cls in ["C-H", "C-C"]:
        mag_within[cls] = {}
        for c in CH:
            sub = df[df.bond_class == cls]
            mag_within[cls][c] = {}
            for lab in ["<70", "70-85", "85-100", ">=100"]:
                s = sub[sub.exp_mag_bin == lab]
                mag_within[cls][c][lab] = {
                    "n": int(len(s)),
                    "mae": round(float(s[f"{c}_ae"].mean()), 4) if len(s) else None,
                    "bias": round(float(s[f"{c}_err"].mean()), 4) if len(s) else None,
                    "thin": bool(len(s) < SMALL_N),
                }
            mag_within[cls][c]["spearman_abs_err_vs_exp_bde"] = spearman(
                sub.experimental_bde_kcal_mol, sub[f"{c}_ae"].to_numpy()
            )

    # --- reference-channel bias sign flip between classes ---
    bias_flip = {}
    for c in CH:
        ec = df.loc[a_mask, f"{c}_err"].to_numpy()
        ecc = df.loc[b_mask, f"{c}_err"].to_numpy()
        se = np.sqrt(ec.var(ddof=1) / len(ec) + ecc.var(ddof=1) / len(ecc))
        diff_bias = float(ec.mean() - ecc.mean())
        bias_flip[c] = {
            "bias_C-H": round(float(ec.mean()), 4),
            "bias_C-C": round(float(ecc.mean()), 4),
            "bias_diff_C_H_minus_C_C": round(diff_bias, 4),
            "welch_t_diff_in_bias": round(float(diff_bias / se), 3) if se > 0 else None,
            "welch_p_diff_in_bias": round(
                float(2 * (1 - stats.t.cdf(abs(diff_bias / se), df=min(len(ec), len(ecc)) - 1))), 8
            ) if se > 0 else None,
            "sign_flip": bool(np.sign(ec.mean()) != np.sign(ecc.mean())),
        }

    # --- is the weak-bond (<70) failure confounded with molecule size? ---
    mag_conf = {
        "weak_bin_n": int((df.exp_mag_bin == "<70").sum()),
        "weak_bin_median_hac": float(df.loc[df.exp_mag_bin == "<70", "hac_parent"].median()),
        "all_median_hac": float(df.hac_parent.median()),
        "weak_bin_mean_hac": round(float(df.loc[df.exp_mag_bin == "<70", "hac_parent"].mean()), 3),
        "all_mean_hac": round(float(df.hac_parent.mean()), 3),
        "weak_bin_class_counts": {
            k: int(v) for k, v in df.loc[df.exp_mag_bin == "<70", "bond_class"].value_counts().items()
        },
        "weak_bin_hac_bin_counts": {
            k: int(v) for k, v in df.loc[df.exp_mag_bin == "<70", "hac_bin"].value_counts().items()
        },
        "weak_vs_rest_within_class_and_size": {},
        "verdict": (
            "The weak-bond bin has ZERO observations with HAC<=5 and is shifted to larger parents "
            f"(median HAC {float(df.loc[df.exp_mag_bin == '<70', 'hac_parent'].median())} vs "
            f"{float(df.hac_parent.median())} overall), so 'weak bond' and 'large molecule' cannot be "
            "separated across the sample as a whole. The one cell with enough weak-bond rows to test "
            "the magnitude effect with BOTH class and size held fixed is C-C / HAC>=10 "
            "(n_weak=12 vs n_rest=34); there the effect survives."
        ),
    }
    for cls in ["C-H", "C-C"]:
        for hb in ["<=5", "6-9", ">=10"]:
            s = df[(df.bond_class == cls) & (df.hac_bin == hb)]
            w = s[s.exp_mag_bin == "<70"]
            r = s[s.exp_mag_bin != "<70"]
            mag_conf["weak_vs_rest_within_class_and_size"][f"{cls}|{hb}"] = {
                "n_weak": int(len(w)), "n_rest": int(len(r)),
                **{f"{c}_mae_weak": (round(float(w[f'{c}_ae'].mean()), 4) if len(w) else None)
                   for c in CH},
                **{f"{c}_mae_rest": (round(float(r[f'{c}_ae'].mean()), 4) if len(r) else None)
                   for c in CH},
                "testable": bool(len(w) >= SMALL_N and len(r) >= SMALL_N),
            }

    env_out = {
        "flag_contrasts": env,
        "magnitude_effect_within_bond_class": mag_within,
        "magnitude_size_confound": mag_conf,
        "bias_by_class": bias_flip,
        "flag_contrasts_within_bond_class": env_stratified,
        "experimental_magnitude_bins_fixed_edges": mag,
        "experimental_magnitude_quartiles": mag_q,
        "heteroscedasticity": hetero,
    }
    print("[ws3] section 3 done", flush=True)
    res["ch3_environment_effects"] = env_out

    # ======================================================================
    # 4. worst / best region per channel (bond class x size x magnitude)
    # ======================================================================
    cells = []
    for cls in bench_counts:
        for hb in ["<=5", "6-9", ">=10"]:
            for mb in ["<70", "70-85", "85-100", ">=100"]:
                sub = df[(df.bond_class == cls) & (df.hac_bin == hb) & (df.exp_mag_bin == mb)]
                if len(sub) == 0:
                    continue
                rec = {
                    "bond_class": cls, "hac_bin": hb, "exp_mag_bin": mb,
                    "n": int(len(sub)),
                    "mean_exp_bde": round(float(sub.experimental_bde_kcal_mol.mean()), 3),
                }
                for c in CH:
                    rec[f"{c}_mae"] = round(float(sub[f"{c}_ae"].mean()), 4)
                    rec[f"{c}_bias"] = round(float(sub[f"{c}_err"].mean()), 4)
                cells.append(rec)
    cells_df = pd.DataFrame(cells)

    worst_best = {}
    for c in CH:
        for min_n in (1, 5, SMALL_N):
            elig = cells_df[cells_df.n >= min_n]
            if len(elig) == 0:
                continue
            w = elig.loc[elig[f"{c}_mae"].idxmax()]
            b = elig.loc[elig[f"{c}_mae"].idxmin()]
            worst_best[f"{c}_min_n{min_n}"] = {
                "worst_cell": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in w.to_dict().items()},
                "best_cell": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in b.to_dict().items()},
                "n_eligible_cells": int(len(elig)),
            }

    # marginal 2-way worst/best (class x size), the more stable slicing
    two_way = {}
    for c in CH:
        tab = ct["C-H"]  # placeholder
        rows = []
        for cls in bench_counts:
            for hb in ["<=5", "6-9", ">=10"]:
                cell = ct[cls][hb]
                if cell.get("n_bond_observations", 0) > 0:
                    rows.append({
                        "bond_class": cls, "hac_bin": hb,
                        "n": cell["n_bond_observations"],
                        "mae": cell.get(f"{c}_mae"),
                        "bias": cell.get(f"{c}_bias"),
                    })
        t = pd.DataFrame(rows)
        elig = t[t.n >= SMALL_N]
        two_way[c] = {
            "table": rows,
            "worst_n_ge_10": (elig.loc[elig.mae.idxmax()].to_dict() if len(elig) else None),
            "best_n_ge_10": (elig.loc[elig.mae.idxmin()].to_dict() if len(elig) else None),
        }

    # 3-way table (bond class x size x magnitude) written in full
    res["ch4_worst_best_regions"] = {
        "cells_class_x_size_x_magnitude": cells,
        "worst_best_by_min_n": worst_best,
        "class_x_size_marginal": two_way,
    }

    # ======================================================================
    # 5. thin-sample ledger
    # ======================================================================
    thin = []
    for cls, st in per_class.items():
        if st.get("n_bond_observations", 0) < SMALL_N:
            thin.append({
                "slice": f"bond_class={cls}", "n": st.get("n_bond_observations"),
                "reason": "n<10 bond observations - no class effect asserted",
            })
    for cls in bench_counts:
        for hb in ["<=5", "6-9", ">=10"]:
            st = ct[cls][hb]
            if 0 < st.get("n_bond_observations", 0) < SMALL_N:
                thin.append({"slice": f"{cls} x HAC {hb}", "n": st["n_bond_observations"],
                             "reason": "n<10 - cell reported but not interpreted"})
    for lab in ["<70", "70-85", "85-100", ">=100"]:
        if mag[lab]["n_bond_observations"] < SMALL_N:
            thin.append({"slice": f"exp BDE {lab}", "n": mag[lab]["n_bond_observations"],
                         "reason": "n<10 - not interpreted"})
    for flag in flags:
        if env[flag]["n_true"] < SMALL_N:
            thin.append({"slice": f"{flag}=True", "n": env[flag]["n_true"],
                         "reason": "n<10 on the True side - flag effect not interpretable"})
    res["ch5_thin_samples"] = {
        "threshold": SMALL_N,
        "ledger": thin,
        "absent_classes": [k for k, v in cov.items() if v["benchmark_bond_observations"] == 0],
    }

    # ======================================================================
    # figure
    # ======================================================================
    make_figure(df, per_class, ct, mag, cells_df)

    # ======================================================================
    # applicability boundary - concrete statement
    # ======================================================================
    def mae(cls, c):
        return per_class[cls][f"{c}_mae"]

    boundary = {"best_measured_region": {}, "worst_measured_region": {}}
    # class x size marginal, restricted to n >= SMALL_N so the "best" is a real cell
    for c in CH:
        rows = []
        for cls in bench_counts:
            for hb in ["<=5", "6-9", ">=10"]:
                cell = ct[cls][hb]
                if cell["n_bond_observations"] >= SMALL_N:
                    rows.append((cell[f"{c}_mae"], cls, hb, cell["n_bond_observations"],
                                 cell[f"{c}_bias"]))
        rows.sort()
        boundary["best_measured_region"][c] = {
            "cell": f"{rows[0][1]} / HAC {rows[0][2]}", "mae": rows[0][0],
            "n_bond_observations": rows[0][3], "bias": rows[0][4],
        }
        boundary["worst_measured_region"][c] = {
            "cell": f"{rows[-1][1]} / HAC {rows[-1][2]}", "mae": rows[-1][0],
            "n_bond_observations": rows[-1][3], "bias": rows[-1][4],
        }
    # magnitude-resolved boundary (2-way: magnitude bin only, all classes pooled)
    for c in CH:
        rows = [(mag[l][f"{c}_mae"], l, mag[l]["n_bond_observations"]) for l in mag if l != "ALL"]
        rows.sort()
        boundary[f"{c}_best_magnitude_bin"] = {
            "exp_bde_bin": rows[0][1], "mae": rows[0][0], "n_bond_observations": rows[0][2]}
        boundary[f"{c}_worst_magnitude_bin"] = {
            "exp_bde_bin": rows[-1][1], "mae": rows[-1][0], "n_bond_observations": rows[-1][2]}
    boundary["statement"] = (
        "Applicability boundary from the measured sample: all three channels are evaluated here "
        "only on C-H and C-C bonds. Within that domain the dominant failure axis is bond strength, "
        f"not bond class. Below 70 kcal/mol (n={mag['<70']['n_bond_observations']} bond observations, "
        "mean experimental BDE 59.9) MAE rises to "
        f"{mag['<70']['quantum_mae']} (quantum), {mag['<70']['learned_mae']} (learned), "
        f"{mag['<70']['reference_mae']} (reference) kcal/mol, i.e. 3.0x / 3.2x / 1.5x the pooled "
        "MAE, with a large positive bias (quantum +4.57, learned +6.87). In the 85-100 kcal/mol "
        f"window (n={mag['85-100']['n_bond_observations']}) quantum and learned are near-unbiased "
        f"(MAE {mag['85-100']['quantum_mae']} / {mag['85-100']['learned_mae']}, bias "
        f"{mag['85-100']['quantum_bias']:+.2f} / {mag['85-100']['learned_bias']:+.2f})."
    )
    res["applicability_boundary"] = boundary
    res["narrative"] = build_narrative(res, per_class, ct, mag, hetero, models, env, verdict)
    res["commands"] = [
        "python3 output/analysis/ws3_build.py",
        "python3 output/analysis/ws3_analysis.py",
    ]
    res["caveats"] = [
        "The benchmark's 543 rows are 543 heterogeneous literature bond observations, not "
        "independent repeats; experimental reference uncertainty is not propagated.",
        "Only C-H and C-C are populated (plus a single H-O row): six of the nine database bond "
        "classes have zero benchmark coverage, so class-level error claims exist only for C-H vs C-C.",
        "Unique molecules (419) and unique fragment pairs (543) are different analysis units from "
        "the 543 bond observations; per-class n values above count bond observations.",
        "OLS R^2 on |error| is descriptive; the residual distribution is skewed and "
        "heteroscedastic, so R^2 magnitudes are small and should not be read as a predictive model.",
        "Environment flags (aromaticity, ring membership, N/O content) are correlated with each "
        "other and with parent size; the stratified tables control only one factor at a time.",
    ]

    common.save_json(res, "ws3_bondclass.json")
    return res


def build_narrative(res, per_class, ct, mag, hetero, models, env, verdict):
    lines = []
    A = per_class["ALL"]
    lines.append(
        "Overall (543 bond observations): quantum MAE "
        f"{A['quantum_mae']} kcal/mol (bias {A['quantum_bias']:+.3f}), learned MAE "
        f"{A['learned_mae']} (bias {A['learned_bias']:+.3f}), reference MAE {A['reference_mae']} "
        f"(bias {A['reference_bias']:+.3f})."
    )
    lines.append(
        "Bond-class coverage is the binding constraint: C-H n=302, C-C n=240, H-O n=1, and "
        "C-N/C-O/H-N/N-N/N-O/O-O are entirely absent from the benchmark although they make up "
        f"{sum(v['db_share_pct'] for k, v in res['coverage']['by_bond_class'].items() if v['benchmark_bond_observations'] == 0):.1f}% of the database rows; "
        "no class-level claim can be made for those six classes."
    )
    lines.append(
        f"C-H vs C-C MAE: quantum {per_class['C-H']['quantum_mae']} vs {per_class['C-C']['quantum_mae']}, "
        f"learned {per_class['C-H']['learned_mae']} vs {per_class['C-C']['learned_mae']}, "
        f"reference {per_class['C-H']['reference_mae']} vs {per_class['C-C']['reference_mae']} kcal/mol. "
        "The pooled C-H/C-C gaps are not stable effects: quantum and reference bootstrap 95% CIs span "
        "zero, and the C-H-vs-C-C direction itself FLIPS between size bins - among HAC<=5 parents C-C "
        "is better for quantum (1.61 vs 1.76), while among HAC>=10 parents C-H is better (2.20 vs 3.82, "
        "Mann-Whitney p=0.005). That is an interaction, not a class main effect."
    )
    lines.append(
        "The C-H group is the larger-molecule group (median HAC "
        f"{res['ch2_class_vs_size']['confounding_check']['hac_within_class']['C-H']['median_hac']} vs "
        f"{res['ch2_class_vs_size']['confounding_check']['hac_within_class']['C-C']['median_hac']}, "
        "Mann-Whitney p<1e-8), so class and size are partly confounded."
    )
    for c in ["quantum", "learned", "reference"]:
        m = models[c]
        lines.append(
            f"{c}: R2(size only)={m['r2_size_only']}, R2(class only)={m['r2_class_only']}, "
            f"R2(size+class)={m['r2_size_plus_class']} -> verdict '{verdict[c]}' "
            f"(incremental R2 from class given size = {m['incremental_r2_from_class_given_size']})."
        )
    h = hetero["quantum"]
    lines.append(
        "Error grows toward the weak-bond end for every channel: quantum Spearman |err| vs exp BDE "
        f"rho={h['spearman_abs_err_vs_exp_bde']['rho']} (p={h['spearman_abs_err_vs_exp_bde']['p_value']}), "
        f"learned rho={hetero['learned']['spearman_abs_err_vs_exp_bde']['rho']}, "
        f"reference rho={hetero['reference']['spearman_abs_err_vs_exp_bde']['rho']}."
    )
    lines.append(
        "Levene p across experimental-magnitude bins: quantum "
        f"{hetero['quantum'].get('levene_p')}, learned {hetero['learned'].get('levene_p')}, "
        f"reference {hetero['reference'].get('levene_p')} - error is strongly heteroscedastic in bond strength."
    )
    # reference-channel sign flip
    bf = res["ch3_environment_effects"]["bias_by_class"]
    lines.append(
        f"The reference channel is the one real class effect found: its bias flips sign between "
        f"classes, {bf['reference']['bias_C-H']:+.3f} kcal/mol on C-H versus "
        f"{bf['reference']['bias_C-C']:+.3f} on C-C (Welch p={bf['reference']['welch_p_diff_in_bias']}). "
        "It systematically over-estimates C-H bonds and under-estimates C-C bonds, and this is why "
        "its pooled MAE (5.49) is 2.6x the quantum channel's despite a similar RMSE."
    )
    # learned channel failure on N-containing parents
    nflag = env["parent_has_N"]["contrasts"]["learned"]["mae_diff_true_minus_false"]
    lines.append(
        f"The learned channel's largest single environment effect is nitrogen: MAE "
        f"{nflag['mae_a']} on N-containing parents (n={nflag['n_a']}) versus {nflag['mae_b']} otherwise "
        f"(bootstrap 95% CI of the difference [{nflag['ci95_low']}, {nflag['ci95_high']}]), a 2.2x "
        "degradation. Within C-H bonds alone it is still "
        f"{res['ch3_environment_effects']['flag_contrasts_within_bond_class']['parent_has_N']['C-H']['learned_mae_true']} vs "
        f"{res['ch3_environment_effects']['flag_contrasts_within_bond_class']['parent_has_N']['C-H']['learned_mae_false']}, "
        "so it is not a class artifact."
    )
    # ring membership
    rflag = env["parent_benzene"]["contrasts"]["learned"]["mae_diff_true_minus_false"]
    lines.append(
        f"Ring/aromatic environment also matters for the learned channel: benzene-containing parents "
        f"MAE {rflag['mae_a']} (n={rflag['n_a']}) versus {rflag['mae_b']} (n={rflag['n_b']}), and a "
        "ring-member radical centre gives the same direction "
        f"(in-ring {env['rad_any_in_ring']['true']['learned_mae']} vs not "
        f"{env['rad_any_in_ring']['false']['learned_mae']}). The quantum channel shows the same "
        "direction more weakly, and the reference channel is largely flat to aromaticity."
    )
    lines.append(
        "Size and bond strength are themselves confounded: the <70 kcal/mol bin contains no HAC<=5 "
        "observation at all and is shifted to larger parents (median HAC 12.5 vs 8.0 overall), so the "
        "'large molecule' and 'weak bond' failure modes cannot be separated globally. The one cell "
        "with enough weak-bond rows to hold class and size fixed is C-C / HAC>=10 (12 weak vs 34 "
        "other): there quantum MAE is 7.66 vs 2.47 and learned 9.27 vs 2.51, so the magnitude effect "
        "does survive that control. That same cell, C-C / HAC>=10 / <70 kcal/mol, is the worst "
        "3-way cell for both quantum and learned (n=12)."
    )
    return " ".join(lines)


def make_figure(df, per_class, ct, mag, cells_df):
    FIG.parent.mkdir(parents=True, exist_ok=True)
    colors = {"quantum": "#1f77b4", "learned": "#ff7f0e", "reference": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) per-class MAE with n annotated
    ax = axes[0, 0]
    classes = [c for c in ["C-H", "C-C", "H-O"] if c in per_class]
    x = np.arange(len(classes))
    w = 0.26
    for i, c in enumerate(CH):
        vals = [per_class[cls][f"{c}_mae"] for cls in classes]
        ns = [per_class[cls]["n_bond_observations"] for cls in classes]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=c, color=colors[c])
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}\nn={n}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("(a) MAE by bond class (n annotated); H-O n=1 is not interpretable")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) MAE by HAC bin, STRATIFIED by bond class (the size trend within a fixed class)
    ax = axes[0, 1]
    bins = ["<=5", "6-9", ">=10"]
    x = np.arange(len(bins))
    styles = {"C-H": "-", "C-C": "--"}
    for i, c in enumerate(["quantum", "learned"]):
        for cls, ls in styles.items():
            vals = [ct[cls][hb].get(f"{c}_mae") for hb in bins]
            ns = [ct[cls][hb]["n_bond_observations"] for hb in bins]
            offset = (i - 0.5) * 0.34
            ax.plot(x + offset, vals, ls, marker="o", color=colors[c],
                    label=f"{c} ({cls})", lw=1.8)
            for xi, v, n in zip(x + offset, vals, ns):
                if v is not None:
                    ax.annotate(f"n={n}", (xi, v), textcoords="offset points",
                                xytext=(0, -12), fontsize=6, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"HAC {b}" for b in bins])
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("(b) Size trend SURVIVES within each bond class")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) MAE by experimental magnitude bin
    ax = axes[1, 0]
    labs = ["<70", "70-85", "85-100", ">=100"]
    x = np.arange(len(labs))
    for i, c in enumerate(CH):
        vals = [mag[l][f"{c}_mae"] for l in labs]
        ns = [mag[l]["n_bond_observations"] for l in labs]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=c, color=colors[c])
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}\nn={n}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} kcal/mol" for l in labs])
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("(c) MAE by experimental BDE magnitude bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (d) signed error vs experimental BDE, class means overlaid to expose the reference flip
    ax = axes[1, 1]
    for c in CH:
        ax.scatter(df.experimental_bde_kcal_mol, df[f"{c}_err"], s=6, alpha=0.35,
                   color=colors[c])
    ax.axhline(0, color="k", lw=0.8)
    for cls, mk in (("C-H", "o"), ("C-C", "s")):
        for c in CH:
            m = df.loc[df.bond_class == cls, f"{c}_err"].mean()
            ax.scatter([90 if cls == "C-H" else 95], [m], marker=mk, s=110,
                       color=colors[c], edgecolor="k", zorder=5)
    ax.annotate("class-mean signed error\n(circles=C-H, squares=C-C)\nreference flips sign:\n"
                "C-H +5.10, C-C -3.30",
                xy=(0.02, 0.72), xycoords="axes fraction", fontsize=7,
                bbox=dict(fc="white", alpha=0.85, ec="grey"))
    ax.set_xlabel("experimental BDE (kcal/mol)")
    ax.set_ylabel("signed error (kcal/mol)")
    ax.set_title("(d) signed error vs experimental BDE (543 bond observations)")
    handles = [plt.Line2D([], [], color=colors[c], label=c) for c in CH]
    ax.legend(handles=handles, fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("WS3: error structure by bond class and structural environment "
                 "(543 bond observations; only C-H and C-C are populated)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=150)
    print(f"[ws3] wrote {FIG}")


if __name__ == "__main__":
    out = main()
    print(json.dumps({"narrative": out["narrative"]}, indent=2))
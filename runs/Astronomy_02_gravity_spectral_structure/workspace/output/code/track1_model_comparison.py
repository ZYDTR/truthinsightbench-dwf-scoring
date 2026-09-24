"""TRACK 1 - Three-way residual-model comparison on the 22 time-resolved passes.

Data: data/pass_residual_traces.tsv (pass_id, relative_time_min,
      residual_model_A_mm_s, residual_model_C_mm_s, residual_model_B_mm_s).

Analysis unit: one complete pass (n = 22).  Time samples inside a pass are NEVER
treated as independent replicates for any null test; sample-level numbers are
reported only as labelled descriptive quantities.

Output: output/results/model_comparison.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code")
sys.path.insert(0, str(CODE))
from common import MODELS, load_traces, write_json  # noqa: E402

PROJECT = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
OUT_JSON = PROJECT / "output" / "results" / "model_comparison.json"

SEED = 20220101
N_BOOT = 20000
PAIRS = [("B", "A"), ("C", "A"), ("B", "C")]

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- load
df = load_traces()
df = df.sort_values(["pass_id", "relative_time_min"], kind="stable")
passes = sorted(df["pass_id"].unique().tolist())
npass = len(passes)

TR = {}
for pid, sub in df.groupby("pass_id", sort=True):
    TR[pid] = {
        "t": sub["relative_time_min"].to_numpy(float),
        "A": sub[MODELS["A"]].to_numpy(float),
        "B": sub[MODELS["B"]].to_numpy(float),
        "C": sub[MODELS["C"]].to_numpy(float),
    }
log(f"loaded {len(df)} rows, {npass} passes")

# gap diagnostics
gapinfo = {}
for pid in passes:
    dt = np.diff(TR[pid]["t"])
    gapinfo[pid] = {
        "n_samples": int(len(TR[pid]["t"])),
        "t_start_min": float(TR[pid]["t"][0]),
        "t_end_min": float(TR[pid]["t"][-1]),
        "median_dt_min": float(np.median(dt)) if dt.size else float("nan"),
        "max_dt_min": float(dt.max()) if dt.size else float("nan"),
        "n_gaps_gt_1p5min": int((dt > 1.5).sum()),
        "has_gap_gt_1p5min": bool((dt > 1.5).any()),
    }
GAP_PASSES = [p for p in passes if gapinfo[p]["has_gap_gt_1p5min"]]
log(f"gap passes (>1.5 min spacing): {GAP_PASSES}")


# ----------------------------------------------------------------------------- helpers
def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v ** 2)))


def detrend(v):
    v = np.asarray(v, float)
    x = np.arange(v.size, dtype=float)
    p = np.polyfit(x, v, 1)
    return v - np.polyval(p, x)


def smooth(v, k):
    v = np.asarray(v, float)
    if k <= 1:
        return v.copy()
    ker = np.ones(k, float) / k
    return np.convolve(v, ker, mode="same")


def per_pass_frame(transform=None):
    """Whole-pass RMS per model (one row per pass). transform: callable(v)->v."""
    rows = []
    for pid in passes:
        d = TR[pid]
        rows.append({m: rms(transform(d[m])) if transform else rms(d[m]) for m in "ABC"})
    return rows  # list aligned with `passes`


def ratios_from_rows(rows):
    out = {}
    for x, y in PAIRS:
        out[f"{x}/{y}"] = np.array([r[x] / r[y] for r in rows], float)
    return out


def boot_median_ratio(rat, rng, nb=N_BOOT):
    idx = rng.integers(0, rat.size, size=(nb, rat.size))
    med = np.median(rat[idx], axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    return {
        "point_median": float(np.median(rat)),
        "boot_median_mean": float(med.mean()),
        "ci95_percentile": [float(lo), float(hi)],
        "prob_ratio_lt_1": float((med < 1).mean()),
        "n_boot": int(nb),
    }


def loo_median_range(rat):
    n = rat.size
    vals = np.array([np.median(np.delete(rat, i)) for i in range(n)], float)
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(vals.max() - vals.min()),
        "loo_values": vals.tolist(),
        "always_below_1": bool((vals < 1).all()),
        "always_above_1": bool((vals > 1).all()),
    }


def paired_tests(x, y, name):
    """x, y are per-pass matched vectors (RMS of model X and Y). x<y favours X."""
    d = x - y
    w = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
    t = stats.ttest_rel(x, y)
    se = d.std(ddof=1) / np.sqrt(d.size)
    tc = stats.t.ppf(0.975, d.size - 1)
    return {
        "pair": name,
        "n_passes": int(d.size),
        "x_median": float(np.median(x)),
        "y_median": float(np.median(y)),
        "wins_x_lower_rms": int((x < y).sum()),
        "wins_y_lower_rms": int((y < x).sum()),
        "ties": int((x == y).sum()),
        "median_difference_x_minus_y": float(np.median(d)),
        "mean_difference_x_minus_y": float(d.mean()),
        "mean_difference_ci95": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
        "wilcoxon_statistic": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "wilcoxon_method": str(getattr(w, "method", "normal approximation (scipy default; no .method attr)")),
        "paired_t_statistic": float(t.statistic),
        "paired_t_p": float(t.pvalue),
        "paired_t_df": int(d.size - 1),
    }


rng = np.random.default_rng(SEED)

# ============================================================================= ITEM 1
rows_raw = per_pass_frame()
rat_raw = ratios_from_rows(rows_raw)

# orthogonalised ("augmented") comparison: residualise each model on the other two
# per pass, so that each number is that model's *additional* content beyond the
# others.  A = aB*B + aC*C + c + rA, etc.
aug = {}
for pid in passes:
    d = TR[pid]
    rec = {}
    for target, o1, o2 in [("A", "B", "C"), ("B", "A", "C"), ("C", "A", "B")]:
        X = np.column_stack([d[o1], d[o2], np.ones_like(d[o1])])
        beta, *_ = np.linalg.lstsq(X, d[target], rcond=None)
        pred = X @ beta
        rec[target] = {
            "coef_other1": float(beta[0]),
            "coef_other2": float(beta[1]),
            "intercept": float(beta[2]),
            "rms_residual": rms(d[target] - pred),
            "rms_raw": rms(d[target]),
            "r2": float(1 - np.sum((d[target] - pred) ** 2) / np.sum((d[target] - d[target].mean()) ** 2)),
        }
    aug[pid] = rec

rows_aug = [{"A": aug[p]["A"]["rms_residual"], "B": aug[p]["B"]["rms_residual"],
             "C": aug[p]["C"]["rms_residual"]} for p in passes]
rat_aug = ratios_from_rows(rows_aug)


def dist_summary(v):
    v = np.asarray(v, float)
    return {
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "values_by_pass": {p: float(val) for p, val in zip(passes, v)},
    }


item1 = {
    "definition": ("whole-pass RMS sqrt(mean(residual^2)) per model, one value per pass; "
                   "ratio X/Y computed pass-by-pass; median and full distribution over the 22 passes"),
    "per_pass_rms_by_model": {m: {p: float(r[m]) for p, r in zip(passes, rows_raw)} for m in "ABC"},
    "rms_distribution_by_model": {m: dist_summary([r[m] for r in rows_raw]) for m in "ABC"},
    "ratio_distributions": {k: dist_summary(v) for k, v in rat_raw.items()},
    "median_ratio_ordering_lowest_to_highest": sorted(
        ["B", "C", "A"], key=lambda m: float(np.median([r[m] for r in rows_raw]))),
    "augmented_note": ("orthogonalised variant: each model is first linearly residualised per pass on the "
                       "other two models (e.g. A = a*B + b*C + c + r_A) so that the compared number is the "
                       "model's additional content beyond the other two"),
    "augmented_rms_distribution_by_model": {m: dist_summary([r[m] for r in rows_aug]) for m in "ABC"},
    "augmented_ratio_distributions": {k: dist_summary(v) for k, v in rat_aug.items()},
}
log("item1 done: median raw ratios " + ", ".join(f"{k}={np.median(v):.4f}" for k, v in rat_raw.items()))

# ============================================================================= ITEM 2
item2 = {"definition": "paired tests across the 22 passes (the independent unit); lower RMS = winner",
         "tests": []}
for x, y in PAIRS:
    vx = np.array([r[x] for r in rows_raw])
    vy = np.array([r[y] for r in rows_raw])
    item2["tests"].append(paired_tests(vx, vy, f"{x} vs {y} (negative difference -> {x} wins)"))
for x, y in PAIRS:
    vx = np.array([r[x] for r in rows_aug])
    vy = np.array([r[y] for r in rows_aug])
    item2["tests"].append(paired_tests(vx, vy, f"[orthogonalised] {x} vs {y} (negative difference -> {x} wins)"))
log("item2 done")

# ============================================================================= ITEM 3
item3 = {"definition": "whole-pass bootstrap: resample the 22 passes with replacement, recompute the "
                        "median ratio each time; percentile CI; fraction of resamples with ratio < 1",
         "n_boot": N_BOOT, "seed": SEED, "raw": {}, "augmented": {}}
for k, v in rat_raw.items():
    item3["raw"][k] = boot_median_ratio(v, rng)
for k, v in rat_aug.items():
    item3["augmented"][k] = boot_median_ratio(v, rng)
log("item3 done: " + ", ".join(f"P({k}<1)={item3['raw'][k]['prob_ratio_lt_1']:.4f}" for k in rat_raw))

# ============================================================================= ITEM 4
item4 = {"definition": "leave-one-pass-out sensitivity of the median ratio (22 deletions)",
         "raw": {k: loo_median_range(v) for k, v in rat_raw.items()},
         "augmented": {k: loo_median_range(v) for k, v in rat_aug.items()}}
log("item4 done")

# ============================================================================= ITEM 5
def ordering_verdict(rat_dict):
    """B best, then C, then A, judged on median ratios."""
    mb = float(np.median(rat_dict["B/A"]))
    mc = float(np.median(rat_dict["C/A"]))
    mbc = float(np.median(rat_dict["B/C"]))
    strict = (mb < 1) and (mc < 1) and (mbc < 1)
    return {"median_B_over_A": mb, "median_C_over_A": mc, "median_B_over_C": mbc,
            "strict_ordering_B_lt_C_lt_A": bool(strict),
            "verdict": "B better than C better than A (both ratios < 1)" if strict else
                       ("mixed / ordering does not survive" )}


perturbations = {}

# (a) linear detrending each pass (per model, within pass)
rows_dt = per_pass_frame(transform=detrend)
perturbations["a_linear_detrend_per_pass"] = {
    "raw_ratios": {k: dist_summary(v) for k, v in ratios_from_rows(rows_dt).items()},
    "ordering": ordering_verdict(ratios_from_rows(rows_dt)),
}

# (b) boxcar smoothing widths 1..15 min
item5b = {}
for k in [1, 3, 5, 7, 9, 15]:
    rr = per_pass_frame(transform=lambda v, k=k: smooth(v, k))
    rk = ratios_from_rows(rr)
    item5b[f"k={k}"] = {
        "median_ratios": {kk: float(np.median(vv)) for kk, vv in rk.items()},
        "ordering": ordering_verdict(rk),
    }
perturbations["b_boxcar_smoothing"] = item5b

# (c) drop passes with time gaps > 1.5 min
keep = [i for i, p in enumerate(passes) if p not in GAP_PASSES]
rows_g = [rows_raw[i] for i in keep]
rat_g = ratios_from_rows(rows_g)
rng_g = np.random.default_rng(SEED + 1)
perturbations["c_drop_gap_passes"] = {
    "dropped_passes": GAP_PASSES,
    "n_passes_kept": len(keep),
    "median_ratios": {k: float(np.median(v)) for k, v in rat_g.items()},
    "ordering": ordering_verdict(rat_g),
    "bootstrap": {k: boot_median_ratio(v, rng_g, nb=10000) for k, v in rat_g.items()},
}

# (d) leave-one-pass-out from the pooled sample, two pooled definitions
item5d = {}
for k in rat_raw:
    x, y = k.split("/")
    med_loo, pooled_loo = [], []
    for i, p in enumerate(passes):
        idx = [j for j in range(npass) if j != i]
        med_loo.append(float(np.median(rat_raw[k][idx])))
        rx = np.concatenate([TR[passes[j]][x] for j in idx])
        ry = np.concatenate([TR[passes[j]][y] for j in idx])
        pooled_loo.append(float(rms(rx) / rms(ry)))
    item5d[k] = {
        "median_of_per_pass_ratios_loo_min": float(np.min(med_loo)),
        "median_of_per_pass_ratios_loo_max": float(np.max(med_loo)),
        "median_of_per_pass_ratios_loo_range": float(np.max(med_loo) - np.min(med_loo)),
        "median_of_per_pass_ratios_always_lt1": bool((np.array(med_loo) < 1).all()),
        "pooled_rms_ratio_full": float(
            rms(np.concatenate([TR[p][x] for p in passes])) / rms(np.concatenate([TR[p][y] for p in passes]))),
        "pooled_rms_ratio_loo_min": float(np.min(pooled_loo)),
        "pooled_rms_ratio_loo_max": float(np.max(pooled_loo)),
        "pooled_rms_ratio_loo_range": float(np.max(pooled_loo) - np.min(pooled_loo)),
        "pooled_rms_ratio_always_lt1": bool((np.array(pooled_loo) < 1).all()),
    }
perturbations["d_leave_one_pass_out_pooled"] = item5d

# (d2) exclusion influence on the raw median ratio (same as item 4 but reported here)
perturbations["d2_exclusion_of_each_pass_median_ratio"] = {
    k: {"loo_min": item4["raw"][k]["min"], "loo_max": item4["raw"][k]["max"],
        "range": item4["raw"][k]["range"], "always_lt1": item4["raw"][k]["always_below_1"]}
    for k in rat_raw
}

_surv = {
    "a_linear_detrend_per_pass": perturbations["a_linear_detrend_per_pass"]["ordering"]["strict_ordering_B_lt_C_lt_A"],
    "b_boxcar_smoothing": {k: v["ordering"]["strict_ordering_B_lt_C_lt_A"] for k, v in item5b.items()},
    "c_drop_gap_passes": perturbations["c_drop_gap_passes"]["ordering"]["strict_ordering_B_lt_C_lt_A"],
}
_all_ok = _surv["a_linear_detrend_per_pass"] and _surv["c_drop_gap_passes"] and all(_surv["b_boxcar_smoothing"].values())
perturbations["summary_ordering_survives_all_perturbations"] = bool(_all_ok)
perturbations["summary_ordering_survival_by_perturbation"] = _surv

item5 = {
    "definition": "robustness of the B < C < A ordering by median whole-pass ratio under alternative definitions",
    "perturbations": perturbations,
}
log("item5 done")

# ============================================================================= ITEM 6
def pooled_ols(y_name, x_names, label):
    y = np.concatenate([TR[p][y_name] for p in passes])
    X = np.column_stack([np.concatenate([TR[p][xn] for p in passes]) for xn in x_names])
    Xd = np.column_stack([X, np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ beta
    resid = y - pred
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    # whole-pass bootstrap CI for the coefficients (resample passes, refit)
    bs = np.empty((10000, beta.size))
    rngb = np.random.default_rng(SEED + 7)
    for b in range(bs.shape[0]):
        idx = rngb.integers(0, npass, npass)
        yy = np.concatenate([TR[passes[j]][y_name] for j in idx])
        XX = np.column_stack([np.concatenate([TR[passes[j]][xn] for j in idx]) for xn in x_names])
        XXd = np.column_stack([XX, np.ones_like(yy)])
        bs[b], *_ = np.linalg.lstsq(XXd, yy, rcond=None)
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    return {
        "label": label,
        "target": y_name,
        "regressors": x_names,
        "n_samples": int(y.size),
        "n_passes": npass,
        "coefficients": {x: float(beta[i]) for i, x in enumerate(x_names)},
        "intercept": float(beta[-1]),
        "coefficient_ci95_whole_pass_bootstrap": {
            x: [float(ci[0, i]), float(ci[1, i])] for i, x in enumerate(x_names)},
        "intercept_ci95": [float(ci[0, -1]), float(ci[1, -1])],
        "residual_rms": rms(resid),
        "target_rms": rms(y),
        "residual_rms_over_target_rms": float(rms(resid) / rms(y)),
        "r2": float(r2),
        "n_boot": 10000,
    }


item6 = {
    "definition": "pooled OLS across all 1698 samples (pooling is for coefficient estimation only; "
                  "uncertainty is bootstrapped over the 22 passes, never over samples)",
    "fits": [
        pooled_ols("C", ["A", "B"], "C ~ a*A + b*B + c"),
        pooled_ols("B", ["A", "C"], "B ~ a*A + b*C + d"),
        pooled_ols("A", ["B", "C"], "A ~ a*B + b*C + e"),
    ],
}
# extra structural checks
Aall = np.concatenate([TR[p]["A"] for p in passes])
Ball = np.concatenate([TR[p]["B"] for p in passes])
Call = np.concatenate([TR[p]["C"] for p in passes])


def ols_simple(y, X, names):
    Xd = np.column_stack([X, np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return beta, resid


b_sum, r_sum = ols_simple(Aall, np.column_stack([Ball + Call]), ["B+C"])
item6["structural_checks"] = {
    "fit_A_on_B_plus_C": {"coef_B_plus_C": float(b_sum[0]), "intercept": float(b_sum[1]),
                          "residual_rms": rms(r_sum), "residual_rms_over_A_rms": float(rms(r_sum) / rms(Aall)),
                          "r2": float(1 - np.sum(r_sum ** 2) / np.sum((Aall - Aall.mean()) ** 2))},
    "fit_A_minus_B_minus_C": {"rms": rms(Aall - Ball - Call),
                              "rms_over_A_rms": float(rms(Aall - Ball - Call) / rms(Aall))},
    "corr_of_A_minus_B_minus_C_with_B": float(np.corrcoef(Aall - Ball - Call, Ball)[0, 1]),
    "corr_A_B": float(np.corrcoef(Aall, Ball)[0, 1]),
    "corr_A_C": float(np.corrcoef(Aall, Call)[0, 1]),
    "corr_B_C": float(np.corrcoef(Ball, Call)[0, 1]),
}
log("item6 done")

# ============================================================================= ITEM 7
def pcorr(M, labels):
    Cm = np.corrcoef(M, rowvar=False)
    return {a: {b: float(Cm[i, j]) for j, b in enumerate(labels)} for i, a in enumerate(labels)}


pass_rms_mat = np.column_stack([[r[m] for r in rows_raw] for m in "ABC"])
pass_aug_mat = np.column_stack([[r[m] for r in rows_aug] for m in "ABC"])
samp_mat = np.column_stack([Aall, Ball, Call])
item7 = {
    "note": ("sample-level correlations use all 1698 samples and are descriptive only; those samples are "
             "NOT independent replicates (adjacent samples are strongly autocorrelated within a pass)"),
    "whole_pass_level_raw_rms_n22": pcorr(pass_rms_mat, ["A", "B", "C"]),
    "whole_pass_level_orthogonalised_rms_n22": pcorr(pass_aug_mat, ["A", "B", "C"]),
    "sample_level_n1698_descriptive_only": pcorr(samp_mat, ["A", "B", "C"]),
}
log("item7 done")

# ============================================================================= ITEM 8
item8 = {"definition": ("leave-one-pass-out cross-validation at the pass level: coefficients fitted on the "
                        "pooled samples of the OTHER 21 passes, prediction error evaluated on the held-out "
                        "pass only; ratio = out-of-sample RMS / held-out model's own RMS"),
         "targets": {}}
for target, x1, x2 in [("B", "A", "C"), ("C", "A", "B"), ("A", "B", "C")]:
    per_pass = {}
    for i, p in enumerate(passes):
        idx = [j for j in range(npass) if j != i]
        yy = np.concatenate([TR[passes[j]][target] for j in idx])
        XX = np.column_stack([np.concatenate([TR[passes[j]][x] for j in idx]) for x in (x1, x2)])
        beta, *_ = np.linalg.lstsq(np.column_stack([XX, np.ones_like(yy)]), yy, rcond=None)
        pred = beta[0] * TR[p][x1] + beta[1] * TR[p][x2] + beta[2]
        oos = rms(TR[p][target] - pred)
        own = rms(TR[p][target])
        per_pass[p] = {"coef_" + x1: float(beta[0]), "coef_" + x2: float(beta[1]),
                       "intercept": float(beta[2]), "oos_rms": oos, "own_rms": own,
                       "ratio_oos_over_own": float(oos / own)}
    rr = np.array([v["ratio_oos_over_own"] for v in per_pass.values()])
    item8["targets"][f"predict_{target}_from_{x1}_{x2}"] = {
        "per_pass": per_pass,
        "median_ratio": float(np.median(rr)),
        "min_ratio": float(rr.min()),
        "max_ratio": float(rr.max()),
        "n_passes_ratio_lt_1": int((rr < 1).sum()),
        "n_passes": npass,
    }
log("item8 done")

# ============================================================================= write
payload = {
    "track": "TRACK1_three_way_residual_model_comparison",
    "unit": "one complete time-resolved pass (n = 22 passes); adjacent time samples and grid rows are not independent",
    "data": "data/pass_residual_traces.tsv",
    "n_passes": npass,
    "n_samples_total": int(len(df)),
    "seed": SEED,
    "pass_gap_diagnostics": gapinfo,
    "item1_per_pass_whole_pass_rms_and_ratios": item1,
    "item2_blocked_paired_tests_over_22_passes": item2,
    "item3_whole_pass_bootstrap_median_ratio": item3,
    "item4_leave_one_pass_out_median_ratio": item4,
    "item5_robustness_alternative_definitions": item5,
    "item6_pooled_linear_relations_between_models": item6,
    "item7_pearson_correlation_matrices": item7,
    "item8_pass_level_cross_validation": item8,
    "software": {"numpy": np.__version__, "scipy": stats.__name__},
}
write_json(OUT_JSON, payload)
log(f"wrote {OUT_JSON}")
print("JSON_OK", OUT_JSON.stat().st_size)
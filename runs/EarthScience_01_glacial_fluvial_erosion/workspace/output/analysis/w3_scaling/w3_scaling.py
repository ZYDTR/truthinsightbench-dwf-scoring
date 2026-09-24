#!/usr/bin/env python3
"""
w3_scaling: method-stratified erosion-rate / timescale scaling with full robustness.

Workstream: method-stratified scale dependence.
All data from the frozen workspace TSVs only. No network, no external data.

Outputs (written next to this script):
  results.json
  tables/*.csv
  figures/*.png
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = "/data/user/conversations/conv-ab8a0c4c-4c0f-46e9-96bb-16e130938d6a/project"
OUT = os.path.join(ROOT, "output", "analysis", "w3_scaling")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RNG_SEED = 20260919
B_BOOT = 2000
LARGE = [
    ("Nonglacial", "Cosmogenic detrital"),
    ("Nonglacial", "Volumetric"),
    ("Glacial", "Volumetric"),
    ("Glacial", "Cosmogenic surface"),
    ("Glacial", "Thermochronometric"),
]

# ----------------------------------------------------------------------------
# 0. Load + parse
# ----------------------------------------------------------------------------
def as_num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan,
                                                           "None": np.nan}),
                        errors="coerce")


def load(path, process):
    df = pd.read_csv(path, sep="\t", dtype=str)
    for c in ["Erosion rate (mm/yr)", "Time interval (yr)", "Area (km2)", "Slope (m/km)",
              "Precipitation (mm/yr)", "Latitude", "Exhumation (m)"]:
        if c in df.columns:
            df[c] = as_num(df[c])
    df["Process"] = process
    df["ER"] = df["Erosion rate (mm/yr)"]
    df["TI"] = df["Time interval (yr)"]
    return df


gl = load(os.path.join(ROOT, "data/raw/glacial_erosion_Earth.tsv"), "Glacial")
ng = load(os.path.join(ROOT, "data/raw/nonglacial_erosion_Earth.tsv"), "Nonglacial")
dat = pd.concat([gl, ng], ignore_index=True)

# analysis filter: strictly positive ER and TI (log-space model)
dat["valid"] = dat["ER"].notna() & dat["TI"].notna() & (dat["ER"] > 0) & (dat["TI"] > 0)
dat["logER"] = np.where(dat["valid"], np.log10(dat["ER"].where(dat["ER"] > 0)), np.nan)
dat["logTI"] = np.where(dat["valid"], np.log10(dat["TI"].where(dat["TI"] > 0)), np.nan)

AN = dat[dat["valid"]].copy()

FILTER_RULES = {
    "row_filter": "ER > 0 AND TI > 0 (non-null); both files concatenated",
    "logER": "log10(Erosion rate (mm/yr)), rate > 0",
    "logTI": "log10(Time interval (yr)), interval > 0",
    "stratum_with_n_ge_5": "every (Process, Methodology) after row filter with n >= 5",
    "ols": "OLS logER ~ logTI (ordinary least squares, classical and HC3 SE)",
    "theil_sen": "scipy.stats.theilslopes(logER, logTI), method='hip' (median of pairwise slopes)",
    "bootstrap_rows": f"{B_BOOT} resamples of rows with replacement, percentile 95% CI on OLS slope",
    "bootstrap_clusters": f"{B_BOOT} resamples of Reference clusters with replacement, percentile 95% CI",
    "leave_one_compilation_out": "drop each Compilation level in turn, refit OLS slope",
    "threshold_sensitivity": "two large nonglacial strata only: ER 1%/5% two-sided trim; TI >= 1 yr; TI >= 10 yr; Area >= p10 (on non-null Area subset)",
    "outlier_treatment": "all rows; winsorized logER at 1/99 and 5/95; winsorized raw ER at 1/99 and 5/95; trimmed logER at 1%/5% two-sided",
}


# ----------------------------------------------------------------------------
# 1. OLS with classical + HC3 SE
# ----------------------------------------------------------------------------
def ols_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - 2
    ss_res = float(resid @ resid)
    s2 = ss_res / dof
    cs = s2 * XtX_inv  # classical covariance
    se = np.sqrt(np.diag(cs))
    tcrit = stats.t.ppf(0.975, dof)
    # HC3
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # leverages
    u = resid / (1.0 - h)
    meat = (X * (u ** 2)[:, None]).T @ X
    hc3 = XtX_inv @ meat @ XtX_inv
    se_hc3 = np.sqrt(np.diag(hc3))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope, inter = float(beta[1]), float(beta[0])
    t = slope / se[1] if se[1] > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else np.nan
    t_h = slope / se_hc3[1] if se_hc3[1] > 0 else np.nan
    p_h = float(2 * stats.t.sf(abs(t_h), dof)) if np.isfinite(t_h) else np.nan
    return {
        "n": int(n),
        "slope": slope,
        "intercept": inter,
        "se": float(se[1]),
        "se_hc3": float(se_hc3[1]),
        "ci95_lo": float(slope - tcrit * se[1]),
        "ci95_hi": float(slope + tcrit * se[1]),
        "ci95_hc3_lo": float(slope - tcrit * se_hc3[1]),
        "ci95_hc3_hi": float(slope + tcrit * se_hc3[1]),
        "t": float(t),
        "p": p,
        "p_hc3": p_h,
        "r2": float(r2),
        "resid_sd": float(np.sqrt(ss_res / dof)),
    }


def ols_slope_only(x, y):
    """Fast slope for bootstrap loops (returns nan if degenerate)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm = x - x.mean()
    sxx = float(xm @ xm)
    if not np.isfinite(sxx) or sxx <= 0 or len(x) < 3:
        return np.nan
    return float((xm @ (y - y.mean())) / sxx)


# ----------------------------------------------------------------------------
# 2. Bootstrap helpers
# ----------------------------------------------------------------------------
def boot_rows(x, y, B=B_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = ols_slope_only(x[idx], y[idx])
    return out[np.isfinite(out)]


def boot_clusters(x, y, groups, B=B_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        out[b] = ols_slope_only(x[idx], y[idx])
    return out[np.isfinite(out)], len(uniq)


def pct_ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# ----------------------------------------------------------------------------
# 3. Main strata fits (task item 1 & 2)
# ----------------------------------------------------------------------------
def stratum_df(proc, meth, df=AN):
    return df[(df["Process"] == proc) & (df["Methodology"] == meth)]


strata = (
    AN.groupby(["Process", "Methodology"]).size().reset_index(name="n")
    .query("n >= 5").sort_values(["Process", "n"], ascending=[True, False])
)

main_rows = []
for _, s in strata.iterrows():
    d = stratum_df(s["Process"], s["Methodology"])
    fit = ols_fit(d["logTI"].values, d["logER"].values)
    fit["process"] = s["Process"]
    fit["methodology"] = s["Methodology"]
    fit["compilation_levels"] = int(d["Compilation"].nunique())
    fit["reference_levels"] = int(d["Reference"].nunique())
    fit["n_unique_TI"] = int(d["TI"].nunique())
    fit["top_TI_value_share"] = float(d["TI"].value_counts(normalize=True).iloc[0])
    # conclusion flip flag classical vs HC3 at alpha=0.05
    sig_c = fit["p"] < 0.05
    sig_h = fit["p_hc3"] < 0.05
    fit["hc3_changes_significance"] = bool(sig_c != sig_h)
    fit["hc3_se_ratio"] = fit["se_hc3"] / fit["se"] if fit["se"] > 0 else np.nan
    main_rows.append(fit)

main_df = pd.DataFrame(main_rows)


# ----------------------------------------------------------------------------
# 4. Large-stratum robustness: Theil-Sen + row bootstrap + cluster bootstrap
# ----------------------------------------------------------------------------
robust_rows = []
boot_stores = {}
for proc, meth in LARGE:
    d = stratum_df(proc, meth)
    x, y = d["logTI"].values, d["logER"].values
    fit = ols_fit(x, y)
    # Theil-Sen. scipy >= 1.15 renamed the method kwarg: 'joint' (default) replaces 'hip'.
    try:
        ts = stats.theilslopes(y, x, method="joint", alpha=0.05)
        ts_method_used = "joint"
    except (ValueError, TypeError) as e_ts:
        ts = stats.theilslopes(y, x, alpha=0.05)
        ts_method_used = f"default (joint kwarg unavailable: {e_ts})"
    ts_slope = float(ts[0])
    ts_lo, ts_hi = float(ts[2]), float(ts[3])
    # row bootstrap
    br = boot_rows(x, y)
    br_lo, br_hi = pct_ci(br)
    # cluster bootstrap by Reference
    bc, nclust = boot_clusters(x, y, d["Reference"].values)
    bc_lo, bc_hi = pct_ci(bc)
    # cluster bootstrap by Compilation (extra, coarser)
    bcc, ncomp = boot_clusters(x, y, d["Compilation"].values, seed=RNG_SEED + 1)
    bcc_lo, bcc_hi = pct_ci(bcc)
    boot_stores[(proc, meth)] = {"rows": br, "clusters_ref": bc, "clusters_comp": bcc}
    robust_rows.append({
        "process": proc, "methodology": meth,
        "n": int(len(x)),
        "n_references": int(d["Reference"].nunique()),
        "n_compilations": int(d["Compilation"].nunique()),
        "ols_slope": fit["slope"], "ols_se": fit["se"], "ols_ci95_lo": fit["ci95_lo"],
        "ols_ci95_hi": fit["ci95_hi"], "ols_r2": fit["r2"], "ols_p": fit["p"],
        "ols_se_hc3": fit["se_hc3"], "ols_ci95_hc3_lo": fit["ci95_hc3_lo"],
        "ols_ci95_hc3_hi": fit["ci95_hc3_hi"], "ols_p_hc3": fit["p_hc3"],
        "theil_sen_slope": ts_slope,
        "theil_sen_method": ts_method_used,
        "theil_sen_ci_lo": ts_lo, "theil_sen_ci_hi": ts_hi,
        "boot_rows_slope_mean": float(np.mean(br)),
        "boot_rows_slope_sd": float(np.std(br, ddof=1)),
        "boot_rows_ci_lo": br_lo, "boot_rows_ci_hi": br_hi,
        "n_clusters_reference": int(nclust),
        "boot_clusters_ref_slope_sd": float(np.std(bc, ddof=1)),
        "boot_clusters_ref_ci_lo": bc_lo, "boot_clusters_ref_ci_hi": bc_hi,
        "n_clusters_compilation": int(ncomp),
        "boot_clusters_comp_ci_lo": bcc_lo, "boot_clusters_comp_ci_hi": bcc_hi,
        "theil_sen_vs_ols_ratio": ts_slope / fit["slope"] if fit["slope"] != 0 else np.nan,
    })

robust_df = pd.DataFrame(robust_rows)


# ----------------------------------------------------------------------------
# 4b. Within-Reference fixed-effects slope + variance decomposition
#
#     Motivation discovered while running: in nonglacial/Cosmogenic detrital the
#     per-Reference ER*TI product is nearly constant, so the pooled -1 slope may
#     be a definitional artefact (ER = Exhumation / TI). A within-Reference
#     (fixed-effects) slope removes all between-Reference level differences.
# ----------------------------------------------------------------------------
def within_fe(x, y, groups, min_within=2):
    """Fixed-effects (within) slope after demeaning y and x by group."""
    df = pd.DataFrame({"x": x, "y": y, "g": groups}).dropna()
    keep = df.groupby("g")["x"].transform("size") >= min_within
    df = df[keep]
    gm = df.groupby("g")[["x", "y"]].transform("mean")
    xw = (df["x"] - gm["x"]).values
    yw = (df["y"] - gm["y"]).values
    sxx = float(xw @ xw)
    if sxx <= 0:
        return {"fe_slope": None, "fe_se": None, "n_within": int(len(df)),
                "n_groups_used": int(df["g"].nunique()), "note": "no within-group x variation"}
    b = float(xw @ yw / sxx)
    resid = yw - b * xw
    ng = df["g"].nunique()
    dof = len(df) - ng - 1
    if dof > 0:
        s2 = float(resid @ resid) / dof
        se = float(np.sqrt(s2 / sxx))
        t = b / se if se > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else None
    else:
        se = p = None
    ss_tot = float(yw @ yw)
    r2w = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else None
    # identifiability: only groups with within-group x variation contribute to sxx
    xvar = df.assign(xw=xw).groupby("g")["xw"].apply(lambda v: float((v ** 2).sum()))
    n_gx = int((xvar > 0).sum())
    gsizes = df.groupby("g").size()
    n_rows_gx = int(gsizes.loc[xvar[xvar > 0].index].sum())
    return {"fe_slope": b, "fe_se": se, "fe_p": p, "fe_r2_within": r2w,
            "n_within": int(len(df)), "n_groups_used": int(ng),
            "n_groups_with_x_variation": n_gx,
            "n_rows_in_groups_with_x_variation": n_rows_gx,
            "effective_identifying_rows_share": n_rows_gx / len(df) if len(df) else None,
            "fe_dof": int(dof), "note": ""}


def var_decomp(v, groups):
    """Share of variance of v that is between groups (ETA^2 style)."""
    df = pd.DataFrame({"v": v, "g": groups}).dropna()
    if len(df) < 3:
        return {"n": int(len(df)), "between_share": None, "n_unique_values": None}
    grand = df["v"].mean()
    gm = df.groupby("g")["v"].transform("mean")
    ssb = float(((gm - grand) ** 2).sum())
    sst = float(((df["v"] - grand) ** 2).sum())
    return {"n": int(len(df)),
            "between_share": ssb / sst if sst > 0 else None,
            "n_unique_values": int(df["v"].nunique()),
            "n_groups": int(df["g"].nunique())}


fe_rows = []
for proc, meth in LARGE:
    d = stratum_df(proc, meth)
    x, y = d["logTI"].values, d["logER"].values
    base = ols_fit(x, y)
    for grp_col in ["Reference", "Compilation"]:
        fe = within_fe(x, y, d[grp_col].values)
        vd_ti = var_decomp(x, d[grp_col].values)
        vd_er = var_decomp(y, d[grp_col].values)
        fe_rows.append({
            "process": proc, "methodology": meth, "grouping": grp_col,
            "n": int(len(d)), "pooled_ols_slope": base["slope"],
            "fe_slope": fe["fe_slope"], "fe_se": fe["fe_se"], "fe_p": fe.get("fe_p"),
            "fe_r2_within": fe["fe_r2_within"],
            "n_within": fe["n_within"], "n_groups_used": fe["n_groups_used"],
            "n_groups_with_x_variation": fe["n_groups_with_x_variation"],
            "n_rows_in_groups_with_x_variation": fe["n_rows_in_groups_with_x_variation"],
            "effective_identifying_rows_share": fe["effective_identifying_rows_share"],
            "logTI_between_group_share": vd_ti["between_share"],
            "logTI_n_unique_values": vd_ti["n_unique_values"],
            "logER_between_group_share": vd_er["between_share"],
            "fe_note": fe["note"],
        })

fe_df = pd.DataFrame(fe_rows)

# ER*TI constancy diagnostic: per Reference coefficient of variation of ER*TI
et_rows = []
for proc, meth in LARGE:
    d = stratum_df(proc, meth).copy()
    d["ET"] = d["ER"] * d["TI"]
    g = d.groupby("Reference")["ET"].agg(["size", "median", "min", "max", "std"])
    g = g[g["size"] >= 3]
    if len(g) == 0:
        et_rows.append({"process": proc, "methodology": meth, "n_refs_ge3": 0})
        continue
    ratio = (g["max"] / g["min"]).values
    cv = (g["std"] / g["median"]).values
    et_rows.append({
        "process": proc, "methodology": meth,
        "n_refs_ge3": int(len(g)),
        "n_refs_ET_max_over_min_lt_1p01": int((ratio < 1.01).sum()),
        "n_refs_ET_max_over_min_lt_1p1": int((ratio < 1.10).sum()),
        "median_ET_max_over_min": float(np.median(ratio)),
        "median_within_ref_CV_of_ET": float(np.nanmedian(cv)),
        "interpretation": ("ER*TI nearly constant within Reference -> pooled slope is "
                           "near-definitional (ER = Exhumation/TI), not independent scaling"
                           if np.median(ratio) < 1.5 else
                           "ER*TI varies substantially within Reference; pooled slope carries "
                           "genuine between-row information"),
    })

et_df = pd.DataFrame(et_rows)

# ----------------------------------------------------------------------------
# 5. Leave-one-compilation-out (task item 4)
# ----------------------------------------------------------------------------
loco_records = []
loco_summary = []
for _, s in strata.iterrows():
    d = stratum_df(s["Process"], s["Methodology"])
    base = ols_fit(d["logTI"].values, d["logER"].values)
    comps = d["Compilation"].value_counts()
    recs = []
    for comp in comps.index:
        dd = d[d["Compilation"] != comp]
        if len(dd) < 3 or dd["logTI"].nunique() < 3:
            continue
        f = ols_fit(dd["logTI"].values, dd["logER"].values)
        recs.append({"dropped_compilation": comp, "n_dropped": int(len(d) - len(dd)),
                     "n_left": int(len(dd)), "slope": f["slope"], "se": f["se"],
                     "p": f["p"], "r2": f["r2"], "ci_lo": f["ci95_lo"], "ci_hi": f["ci95_hi"]})
    if not recs:
        continue
    rdf = pd.DataFrame(recs)
    slopes = rdf["slope"].values
    sign_flip = bool(np.any(np.sign(slopes) != np.sign(base["slope"])))
    sig_flip = bool(np.any((rdf["p"].values < 0.05) != (base["p"] < 0.05)))
    minr = rdf.loc[rdf["slope"].idxmin()]
    maxr = rdf.loc[rdf["slope"].idxmax()]
    loco_summary.append({
        "process": s["Process"], "methodology": s["Methodology"],
        "baseline_slope": base["slope"], "baseline_p": base["p"],
        "n_drop_levels": int(len(rdf)),
        "min_slope": float(slopes.min()), "max_slope": float(slopes.max()),
        "median_slope": float(np.median(slopes)),
        "max_abs_dev_from_base": float(np.max(np.abs(slopes - base["slope"]))),
        "sign_flips": sign_flip, "significance_flips": sig_flip,
        "extremes": {"min_from": minr["dropped_compilation"], "max_from": maxr["dropped_compilation"]},
        "pct_change_at_extreme": float(100 * maxr["slope"] / base["slope"] - 100) if base["slope"] else np.nan,
    })
    for _, r in rdf.iterrows():
        rec = r.to_dict(); rec["process"] = s["Process"]; rec["methodology"] = s["Methodology"]
        rec["baseline_slope"] = base["slope"]
        loco_records.append(rec)

loco_df = pd.DataFrame(loco_records)
loco_sum_df = pd.DataFrame(loco_summary)


# ----------------------------------------------------------------------------
# 6. Per-Reference slopes, 10 largest References (task item 5)
# ----------------------------------------------------------------------------
per_ref_rows = []
per_ref_dist = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric")]:
    d = stratum_df(proc, meth)
    top = d["Reference"].value_counts().head(10)
    slopes = []
    for ref, cnt in top.items():
        dd = d[d["Reference"] == ref]
        nuniq = int(dd["TI"].nunique())
        ti_vals = sorted(dd["TI"].unique().tolist())[:4]
        if len(dd) < 3 or nuniq < 3:
            per_ref_rows.append({"process": proc, "methodology": meth, "reference": ref,
                                 "n": int(len(dd)), "n_unique_TI": nuniq,
                                 "TI_values": str(ti_vals), "slope": None, "se": None, "p": None,
                                 "r2": None,
                                 "note": f"degenerate x: only {nuniq} unique Time interval value(s); "
                                         f"within-Reference slope not identified"})
            continue
        f = ols_fit(dd["logTI"].values, dd["logER"].values)
        slopes.append(f["slope"])
        per_ref_rows.append({"process": proc, "methodology": meth, "reference": ref,
                             "n": int(len(dd)), "n_unique_TI": nuniq, "TI_values": str(ti_vals),
                             "slope": f["slope"], "se": f["se"],
                             "p": f["p"], "r2": f["r2"], "ci_lo": f["ci95_lo"],
                             "ci_hi": f["ci95_hi"], "note": ""})
    arr = np.array([s for s in slopes if s is not None], float)
    if len(arr) == 0:
        per_ref_dist.append({
            "process": proc, "methodology": meth,
            "n_references_ranked": int(len(top)), "n_refs_fit": 0,
            "note": "all top-10 References had <3 rows or <3 unique logTI; no per-Reference OLS fittable",
        })
        continue
    per_ref_dist.append({
        "process": proc, "methodology": meth,
        "n_references_ranked": int(len(top)),
        "n_refs_fit": int(len(arr)),
        "slope_min": float(np.min(arr)), "slope_max": float(np.max(arr)),
        "slope_median": float(np.median(arr)), "slope_mean": float(np.mean(arr)),
        "slope_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
        "slope_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))] if len(arr) > 1 else None,
        "n_negative": int((arr < 0).sum()), "n_positive": int((arr > 0).sum()),
        "n_ci_excludes_0": int(sum(1 for r in per_ref_rows if r.get("p") is not None
                                   and r["process"] == proc and r["methodology"] == meth
                                   and r["p"] < 0.05)),
        "note": "",
    })

# Per-compilation slopes (every Compilation level with a fittable design), which is the
# coarser version of the per-Reference question and is fittable in Volumetric too.
per_comp_rows = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric"),
                   ("Glacial", "Volumetric"), ("Glacial", "Cosmogenic surface"),
                   ("Glacial", "Thermochronometric")]:
    d = stratum_df(proc, meth)
    for comp, sub in d.groupby("Compilation"):
        if len(sub) < 3 or sub["TI"].nunique() < 3:
            per_comp_rows.append({"process": proc, "methodology": meth, "compilation": comp,
                                  "n": int(len(sub)), "n_unique_TI": int(sub["TI"].nunique()),
                                  "slope": None, "note": "degenerate x"})
            continue
        f = ols_fit(sub["logTI"].values, sub["logER"].values)
        per_comp_rows.append({"process": proc, "methodology": meth, "compilation": comp,
                              "n": int(len(sub)), "n_unique_TI": int(sub["TI"].nunique()),
                              "slope": f["slope"], "se": f["se"], "p": f["p"], "r2": f["r2"],
                              "ci_lo": f["ci95_lo"], "ci_hi": f["ci95_hi"], "note": ""})
per_comp_df = pd.DataFrame(per_comp_rows)

per_ref_df = pd.DataFrame(per_ref_rows)
per_ref_dist_df = pd.DataFrame(per_ref_dist)

# Count how many References in each of the two large nonglacial strata can be fit at
# all (>= 3 rows AND >= 3 unique Time interval values). Directly shows whether
# "per-Reference slopes" is even a defined quantity in that stratum.
ref_fittable_rows = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric"),
                   ("Glacial", "Volumetric"), ("Glacial", "Cosmogenic surface"),
                   ("Glacial", "Thermochronometric")]:
    d = stratum_df(proc, meth)
    g = d.groupby("Reference").agg(n=("TI", "size"), nuniqTI=("TI", "nunique"))
    ref_fittable_rows.append({
        "process": proc, "methodology": meth,
        "n_references": int(len(g)),
        "n_refs_n_ge3": int((g["n"] >= 3).sum()),
        "n_refs_n_ge3_and_uniqueTI_ge3": int(((g["n"] >= 3) & (g["nuniqTI"] >= 3)).sum()),
        "share_refs_fittable": float(((g["n"] >= 3) & (g["nuniqTI"] >= 3)).mean()),
    })
ref_fittable_df = pd.DataFrame(ref_fittable_rows)

# TI composition by Compilation in the two large nonglacial strata (shows that the
# strata are built from different-epoch compilations, so Compilation is a proxy for
# the time slice the data were assembled for).
ti_comp_rows = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric"),
                   ("Glacial", "Volumetric")]:
    d = stratum_df(proc, meth)
    top_ti = d["TI"].value_counts(normalize=True).head(3)
    for comp, sub in d.groupby("Compilation"):
        ti_comp_rows.append({
            "process": proc, "methodology": meth, "compilation": comp, "n": int(len(sub)),
            "median_TI": float(sub["TI"].median()),
            "min_TI": float(sub["TI"].min()), "max_TI": float(sub["TI"].max()),
            "median_ER": float(sub["ER"].median()),
            "top_TI_values": "; ".join(f"{v:g} ({f:.0%})" for v, f in top_ti.items()),
        })
ti_comp_df = pd.DataFrame(ti_comp_rows)


# ----------------------------------------------------------------------------
# 7. Threshold sensitivity (task item 6) - two large nonglacial strata
# ----------------------------------------------------------------------------
thresh_records = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric")]:
    d = stratum_df(proc, meth)
    base = ols_fit(d["logTI"].values, d["logER"].values)
    variants = {}
    for q in (1, 5):
        lo, hi = np.percentile(d["ER"].values, [q, 100 - q])
        variants[f"drop_ER_below_p{q}_above_p{100-q}"] = d[(d["ER"] >= lo) & (d["ER"] <= hi)]
    variants["TI_ge_1yr"] = d[d["TI"] >= 1]
    variants["TI_ge_10yr"] = d[d["TI"] >= 10]
    variants["TI_ge_0.5yr(fallback)"] = d[d["TI"] >= 0.5]
    sub_area = d[d["Area (km2)"].notna()]
    if len(sub_area) >= 10:
        ap10 = np.percentile(sub_area["Area (km2)"].values, 10)
        variants["Area_ge_p10(fit on non-null Area subset)"] = sub_area[sub_area["Area (km2)"] >= ap10]
        variants["Area_nonnull_only(comparability baseline)"] = sub_area
    for name, v in variants.items():
        if len(v) < 5 or v["logTI"].nunique() < 3:
            thresh_records.append({"process": proc, "methodology": meth, "variant": name,
                                   "n": int(len(v)), "slope": None, "note": "degenerate"})
            continue
        f = ols_fit(v["logTI"].values, v["logER"].values)
        in_ci = base["ci95_lo"] <= f["slope"] <= base["ci95_hi"]
        thresh_records.append({
            "process": proc, "methodology": meth, "variant": name,
            "n": int(len(v)), "n_dropped": int(len(d) - len(v)),
            "n_unique_TI": int(v["TI"].nunique()),
            "slope": f["slope"], "se": f["se"], "p": f["p"], "r2": f["r2"],
            "ci_lo": f["ci95_lo"], "ci_hi": f["ci95_hi"],
            "delta_vs_base": f["slope"] - base["slope"],
            "pct_change_vs_base": 100 * (f["slope"] / base["slope"] - 1) if base["slope"] else None,
            "base_slope_in_variant_ci": bool(in_ci),
        })
    thresh_records.append({"process": proc, "methodology": meth, "variant": "BASELINE",
                           "n": base["n"], "n_dropped": 0, "slope": base["slope"],
                           "se": base["se"], "p": base["p"], "r2": base["r2"],
                           "ci_lo": base["ci95_lo"], "ci_hi": base["ci95_hi"],
                           "delta_vs_base": 0.0, "pct_change_vs_base": 0.0,
                           "base_slope_in_variant_ci": True})

thresh_df = pd.DataFrame(thresh_records)


# ----------------------------------------------------------------------------
# 8. Outlier treatment comparison (task item 7)
# ----------------------------------------------------------------------------
outlier_records = []
for proc, meth in [("Nonglacial", "Cosmogenic detrital"), ("Nonglacial", "Volumetric"),
                   ("Glacial", "Volumetric"), ("Glacial", "Cosmogenic surface"),
                   ("Glacial", "Thermochronometric")]:
    d = stratum_df(proc, meth)
    x = d["logTI"].values
    y = d["logER"].values
    treatments = {"all_rows": (x, y)}
    for q in (1, 5):
        lo, hi = np.percentile(y, [q, 100 - q])
        treatments[f"winsor_logER_{q}_{100-q}"] = (x, np.clip(y, lo, hi))
        lo_e, hi_e = np.percentile(d["ER"].values, [q, 100 - q])
        y_w = np.log10(np.clip(d["ER"].values, lo_e, hi_e))
        treatments[f"winsor_rawER_{q}_{100-q}"] = (x, y_w)
        lo_l, hi_l = np.percentile(y, [q, 100 - q])
        m = (y >= lo_l) & (y <= hi_l)
        treatments[f"trim_logER_{q}_{100-q}"] = (x[m], y[m])
    base = ols_fit(x, y)
    for name, (xv, yv) in treatments.items():
        if len(xv) < 3 or len(np.unique(xv)) < 3:
            continue
        f = ols_fit(xv, yv)
        outlier_records.append({
            "process": proc, "methodology": meth, "treatment": name,
            "n": int(len(xv)), "slope": f["slope"], "se": f["se"], "p": f["p"],
            "r2": f["r2"], "ci_lo": f["ci95_lo"], "ci_hi": f["ci95_hi"],
            "delta_vs_all_rows": f["slope"] - base["slope"],
            "pct_change_vs_all_rows": 100 * (f["slope"] / base["slope"] - 1) if base["slope"] else None,
            "all_rows_slope_in_ci": bool(base["ci95_lo"] <= f["slope"] <= base["ci95_hi"]),
        })

outlier_df = pd.DataFrame(outlier_records)


# ----------------------------------------------------------------------------
# 9. Consistency with -1 / 0 / intermediate (task item 8)
# ----------------------------------------------------------------------------
def classify(ci_lo, ci_hi, slope):
    has_m1 = ci_lo <= -1.0 <= ci_hi
    has_0 = ci_lo <= 0.0 <= ci_hi
    if has_m1 and has_0:
        lab = "indeterminate (CI covers both -1 and 0)"
    elif has_m1:
        lab = "consistent with -1 (mechanical coupling)"
    elif has_0:
        lab = "consistent with 0 (no relation)"
    else:
        side = "negative" if slope < 0 else "positive"
        lab = f"intermediate / other ({side}, CI excludes both -1 and 0)"
    return {"ci_covers_minus1": bool(has_m1), "ci_covers_0": bool(has_0), "classification": lab}


classify_rows = []
for _, s in strata.iterrows():
    d = stratum_df(s["Process"], s["Methodology"])
    f = ols_fit(d["logTI"].values, d["logER"].values)
    c_cls = classify(f["ci95_lo"], f["ci95_hi"], f["slope"])
    c_hc3 = classify(f["ci95_hc3_lo"], f["ci95_hc3_hi"], f["slope"])
    row = {"process": s["Process"], "methodology": s["Methodology"], "n": f["n"],
           "slope": f["slope"], "ci95_lo": f["ci95_lo"], "ci95_hi": f["ci95_hi"],
           "classical": c_cls["classification"],
           "hc3": c_hc3["classification"],
           "d_from_minus1_se": (f["slope"] + 1) / f["se"], "d_from_0_se": f["slope"] / f["se"]}
    classify_rows.append(row)
classify_df = pd.DataFrame(classify_rows)

# same for Theil-Sen / bootstrap on large strata
robust_classify = []
for _, r in robust_df.iterrows():
    c_ols = classify(r["ols_ci95_lo"], r["ols_ci95_hi"], r["ols_slope"])
    c_ts = classify(r["theil_sen_ci_lo"], r["theil_sen_ci_hi"], r["theil_sen_slope"])
    c_br = classify(r["boot_rows_ci_lo"], r["boot_rows_ci_hi"], r["boot_rows_slope_mean"])
    c_bc = classify(r["boot_clusters_ref_ci_lo"], r["boot_clusters_ref_ci_hi"], r["ols_slope"])
    robust_classify.append({
        "process": r["process"], "methodology": r["methodology"], "n": r["n"],
        "ols": c_ols["classification"], "theil_sen": c_ts["classification"],
        "boot_rows": c_br["classification"], "boot_clusters_ref": c_bc["classification"],
    })
robust_classify_df = pd.DataFrame(robust_classify)


# ----------------------------------------------------------------------------
# 10. Exhumation identity sanity check (verify known-fact claim)
# ----------------------------------------------------------------------------
exh_records = []
for proc, sub in [("Glacial", gl), ("Nonglacial", ng)]:
    d = sub[sub["Exhumation (m)"].notna()]
    d = d[d["ER"].notna() & d["TI"].notna() & (d["Exhumation (m)"] != 0)]
    if len(d) == 0:
        exh_records.append({"process": proc, "n_exhumation_populated": 0,
                            "n_with_ER_and_TI": 0})
        continue
    lhs = d["ER"].values * d["TI"].values          # mm if ER in mm/yr, TI in yr
    rhs = d["Exhumation (m)"].values * 1000.0      # m -> mm
    rel = np.abs(lhs - rhs) / np.abs(rhs)
    ratio = lhs / rhs
    rec = {
        "process": proc,
        "n_exhumation_populated": int((sub["Exhumation (m)"].notna()).sum()),
        "n_with_ER_and_TI_nonzero_exh": int(len(d)),
        "n_rel_lt_1e-12_exact": int((rel < 1e-12).sum()),
        "n_rel_lt_1e-6": int((rel < 1e-6).sum()),
        "n_rel_lt_1e-3": int((rel < 1e-3).sum()),
        "n_rel_lt_1e-2": int((rel < 1e-2).sum()),
        "median_rel_diff": float(np.median(rel)),
        "ratio_ET_over_Exh_times1000_quantiles": {
            "p05": float(np.percentile(ratio, 5)), "p25": float(np.percentile(ratio, 25)),
            "p50": float(np.percentile(ratio, 50)), "p75": float(np.percentile(ratio, 75)),
            "p95": float(np.percentile(ratio, 95)),
        },
        "by_compilation_rel_lt_1e-2": d.assign(rel=rel).groupby("Compilation")
            .agg(n=("rel", "size"), n_ident=("rel", lambda v: int((v < 1e-2).sum())))
            .reset_index().to_dict("records"),
        "example_rows": d.loc[np.argsort(-rel)[:3], ["Reference", "ER", "TI",
                                                     "Exhumation (m)"]].to_dict("records"),
    }
    exh_records.append(rec)


# ----------------------------------------------------------------------------
# 11. Figures
# ----------------------------------------------------------------------------
# 11a. scatter + fits, nonglacial and glacial panels
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, proc in zip(axes, ["Nonglacial", "Glacial"]):
    sub = AN[AN["Process"] == proc]
    ax.scatter(sub["logTI"], sub["logER"], s=6, c="0.75", alpha=0.5, linewidths=0)
    for _, s in strata[lambda d: d["Process"] == proc].iterrows():
        d = stratum_df(proc, s["Methodology"])
        f = ols_fit(d["logTI"].values, d["logER"].values)
        xs = np.linspace(d["logTI"].min(), d["logTI"].max(), 50)
        ax.plot(xs, f["intercept"] + f["slope"] * xs, lw=2,
                label=f"{s['Methodology']} (n={f['n']}, b={f['slope']:.2f})")
    ax.set_xlabel("log10 Time interval (yr)")
    ax.set_ylabel("log10 Erosion rate (mm/yr)")
    ax.set_title(f"{proc}: methodology strata")
    ax.legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_scatter_strata.png"), dpi=150)
plt.close(fig)

# 11b. forest plot: OLS classical CI, HC3 CI, row boot, cluster boot for large strata
labels, ys, ols_pt, ols_lo, ols_hi, hc_lo, hc_hi, br_lo, br_hi, bc_lo, bc_hi = ([],
    [], [], [], [], [], [], [], [], [], [])
for _, r in robust_df.iterrows():
    labels.append(f"{r['process'][:3]}/{r['methodology']}\n(n={r['n']})")
    ys.append(len(labels))
    ols_pt.append(r["ols_slope"]); ols_lo.append(r["ols_ci95_lo"]); ols_hi.append(r["ols_ci95_hi"])
    hc_lo.append(r["ols_ci95_hc3_lo"]); hc_hi.append(r["ols_ci95_hc3_hi"])
    br_lo.append(r["boot_rows_ci_lo"]); br_hi.append(r["boot_rows_ci_hi"])
    bc_lo.append(r["boot_clusters_ref_ci_lo"]); bc_hi.append(r["boot_clusters_ref_ci_hi"])
ys = np.array(ys)
fig, ax = plt.subplots(figsize=(9, 0.9 * len(labels) + 2))
ax.axvline(0, color="k", lw=1, ls=":")
ax.axvline(-1, color="r", lw=1, ls="--", label="slope = -1")
for i, y in enumerate(ys):
    ax.plot([ols_lo[i], ols_hi[i]], [y + 0.18, y + 0.18], color="C0", lw=2,
            label="OLS classical 95% CI" if i == 0 else None)
    ax.plot([hc_lo[i], hc_hi[i]], [y + 0.06, y + 0.06], color="C1", lw=2,
            label="OLS HC3 95% CI" if i == 0 else None)
    ax.plot([br_lo[i], br_hi[i]], [y - 0.06, y - 0.06], color="C2", lw=2,
            label="row bootstrap 95% CI" if i == 0 else None)
    ax.plot([bc_lo[i], bc_hi[i]], [y - 0.18, y - 0.18], color="C3", lw=2,
            label="cluster(Reference) boot 95% CI" if i == 0 else None)
    ax.plot(ols_pt[i], y + 0.18, "o", color="C0", ms=4)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("OLS slope of logER ~ logTI (dex per dex)")
ax.set_title("Large-stratum slope estimates with four uncertainty treatments")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_slope_forest.png"), dpi=150)
plt.close(fig)

# 11b2. Pooled vs within-Reference fixed-effects slopes for large strata
fig, ax = plt.subplots(figsize=(8.5, 4.5))
w = 0.35
idx = np.arange(len(fe_df[fe_df["grouping"] == "Reference"]))
sub = fe_df[fe_df["grouping"] == "Reference"].reset_index(drop=True)
subc = fe_df[fe_df["grouping"] == "Compilation"].reset_index(drop=True)
lbl = [f"{r['process'][:3]}/{r['methodology']}" for _, r in sub.iterrows()]
ax.barh(idx - w / 2, sub["pooled_ols_slope"], height=w, color="C0", label="pooled OLS")
ax.barh(idx + w / 2, sub["fe_slope"].astype(float), height=w, color="C3",
        label="within-Reference fixed effects")
for i, r in sub.iterrows():
    ax.text(r["pooled_ols_slope"], i - w / 2, f" {r['pooled_ols_slope']:.2f}", va="center", fontsize=7)
    if r["fe_slope"] is not None and np.isfinite(r["fe_slope"]):
        ax.text(r["fe_slope"], i + w / 2, f" {r['fe_slope']:.2f}", va="center", fontsize=7)
ax.axvline(0, color="k", lw=1, ls=":")
ax.axvline(-1, color="r", lw=1, ls="--", label="slope = -1")
ax.set_yticks(idx); ax.set_yticklabels(lbl, fontsize=8)
ax.set_xlabel("slope of logER ~ logTI")
ax.set_title("Pooled OLS vs within-Reference fixed-effects slope")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_pooled_vs_within.png"), dpi=150)
plt.close(fig)

# 11c. LOCO slope ranges for strata with >=2 compilations
plot_l = loco_sum_df[loco_sum_df["n_drop_levels"] >= 2]
if len(plot_l):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(plot_l) + 2))
    for i, (_, r) in enumerate(plot_l.iterrows()):
        ax.plot([r["min_slope"], r["max_slope"]], [i, i], color="0.4", lw=3, solid_capstyle="butt")
        ax.plot(r["baseline_slope"], i, "o", color="C0", ms=7, label="baseline" if i == 0 else None)
        ax.plot(r["min_slope"], i, "|", color="C3", ms=10)
        ax.plot(r["max_slope"], i, "|", color="C3", ms=10)
    ax.axvline(-1, color="r", ls="--", lw=1, label="slope = -1")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(plot_l)))
    ax.set_yticklabels([f"{r['process'][:3]}/{r['methodology']}" for _, r in plot_l.iterrows()], fontsize=8)
    ax.set_xlabel("OLS slope range when dropping one Compilation at a time")
    ax.set_title("Leave-one-compilation-out slope ranges")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_loco_ranges.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 12. Write tables + results.json
# ----------------------------------------------------------------------------
main_df.to_csv(os.path.join(TAB, "strata_main_ols.csv"), index=False)
robust_df.to_csv(os.path.join(TAB, "large_strata_robustness.csv"), index=False)
fe_df.to_csv(os.path.join(TAB, "within_reference_fixed_effects.csv"), index=False)
et_df.to_csv(os.path.join(TAB, "ERxTI_constancy_diagnostic.csv"), index=False)
loco_df.to_csv(os.path.join(TAB, "leave_one_compilation_out_all.csv"), index=False)
loco_sum_df.to_csv(os.path.join(TAB, "leave_one_compilation_out_summary.csv"), index=False)
per_ref_df.to_csv(os.path.join(TAB, "per_reference_slopes.csv"), index=False)
per_ref_dist_df.to_csv(os.path.join(TAB, "per_reference_distribution.csv"), index=False)
ref_fittable_df.to_csv(os.path.join(TAB, "per_reference_fittability.csv"), index=False)
per_comp_df.to_csv(os.path.join(TAB, "per_compilation_slopes.csv"), index=False)
ti_comp_df.to_csv(os.path.join(TAB, "TI_composition_by_compilation.csv"), index=False)
thresh_df.to_csv(os.path.join(TAB, "threshold_sensitivity.csv"), index=False)
outlier_df.to_csv(os.path.join(TAB, "outlier_treatments.csv"), index=False)
classify_df.to_csv(os.path.join(TAB, "consistency_classification.csv"), index=False)
robust_classify_df.to_csv(os.path.join(TAB, "consistency_classification_robust.csv"), index=False)
pd.DataFrame(exh_records).to_csv(os.path.join(TAB, "exhumation_identity_check.csv"), index=False)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


results = {
    "workstream": "w3_scaling",
    "topic": "method-stratified erosion-rate/timescale scaling with robustness",
    "filter_rules": FILTER_RULES,
    "input_shapes": {"glacial_rows": int(len(gl)), "nonglacial_rows": int(len(ng)),
                     "rows_passing_ER>0_and_TI>0": {"Glacial": int((gl["ER"].notna() & gl["TI"].notna() & (gl["ER"] > 0) & (gl["TI"] > 0)).sum()),
                                                    "Nonglacial": int((ng["ER"].notna() & ng["TI"].notna() & (ng["ER"] > 0) & (ng["TI"] > 0)).sum())}},
    "strata_main_ols_hc3": clean(main_df.to_dict("records")),
    "large_strata_robustness": clean(robust_df.to_dict("records")),
    "within_reference_fixed_effects": clean(fe_df.to_dict("records")),
    "ERxTI_constancy_diagnostic": clean(et_df.to_dict("records")),
    "leave_one_compilation_out_summary": clean(loco_sum_df.to_dict("records")),
    "leave_one_compilation_out_records": clean(loco_df.to_dict("records")),
    "per_reference_slopes": clean(per_ref_df.to_dict("records")),
    "per_reference_distribution": clean(per_ref_dist_df.to_dict("records")),
    "per_reference_fittability": clean(ref_fittable_df.to_dict("records")),
    "per_compilation_slopes": clean(per_comp_df.to_dict("records")),
    "TI_composition_by_compilation": clean(ti_comp_df.to_dict("records")),
    "threshold_sensitivity": clean(thresh_df.to_dict("records")),
    "outlier_treatments": clean(outlier_df.to_dict("records")),
    "consistency_vs_minus1_and_0": clean(classify_df.to_dict("records")),
    "consistency_robust_estimators": clean(robust_classify_df.to_dict("records")),
    "exhumation_identity_check": clean(exh_records),
}
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)


# ----------------------------------------------------------------------------
# 13. Console report
# ----------------------------------------------------------------------------
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print("=" * 100)
print("1+2. MAIN STRATA: OLS logER ~ logTI, classical and HC3 SE")
print("=" * 100)
print(main_df[["process", "methodology", "n", "slope", "se", "se_hc3", "ci95_lo", "ci95_hi",
               "p", "p_hc3", "r2", "hc3_changes_significance"]].to_string(index=False))
print()
print("=" * 100)
print("3. LARGE-STRATUM ROBUSTNESS (Theil-Sen, row boot, cluster boot)")
print("=" * 100)
print(robust_df[["process", "methodology", "n", "ols_slope", "theil_sen_slope",
                 "boot_rows_ci_lo", "boot_rows_ci_hi", "boot_clusters_ref_ci_lo",
                 "boot_clusters_ref_ci_hi", "ols_r2"]].to_string(index=False))
print()
print("=" * 100)
print("4b. WITHIN-REFERENCE FIXED-EFFECTS SLOPE + BETWEEN-GROUP VARIANCE SHARE")
print("=" * 100)
print(fe_df.to_string(index=False))
print()
print(et_df.to_string(index=False))
print()
print("=" * 100)
print("4. LEAVE-ONE-COMPILATION-OUT")
print("=" * 100)
print(loco_sum_df.to_string(index=False))
print()
print("=" * 100)
print("5. PER-REFERENCE SLOPES (10 largest refs)")
print("=" * 100)
print(per_ref_df.to_string(index=False))
print()
print(per_ref_dist_df.to_string(index=False))
print()
print(ref_fittable_df.to_string(index=False))
print()
print(ti_comp_df.to_string(index=False))
print()
print("PER-COMPILATION SLOPES:")
print(per_comp_df.to_string(index=False))
print()
print("=" * 100)
print("6. THRESHOLD SENSITIVITY")
print("=" * 100)
print(thresh_df.to_string(index=False))
print()
print("=" * 100)
print("7. OUTLIER TREATMENTS")
print("=" * 100)
print(outlier_df[["process", "methodology", "treatment", "n", "slope", "se", "p", "r2",
                  "pct_change_vs_all_rows"]].to_string(index=False))
print()
print("=" * 100)
print("8. CONSISTENCY WITH -1 / 0")
print("=" * 100)
print(classify_df.to_string(index=False))
print()
print(robust_classify_df.to_string(index=False))
print()
print("=" * 100)
print("EXHUMATION IDENTITY CHECK")
print("=" * 100)
for r in exh_records:
    print(r)
print()
print("DONE. results.json written to", os.path.join(OUT, "results.json"))
import sys
sys.stdout.flush()
os._exit(0)  # avoid interpreter-teardown hang observed in this sandbox after plotting
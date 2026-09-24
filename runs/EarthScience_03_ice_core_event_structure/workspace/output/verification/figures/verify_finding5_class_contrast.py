#!/usr/bin/env python3
"""Independent verification of Draft Finding 5 (v2: fully vectorised).

Claim under test:
  "The two curator label classes behave the same and are distinguishable only by
   position.  The 5 event_NN labels and the 9 unidentified_event labels do not
   differ significantly on sulfate amplitude (medians 507.6 vs 300.8 ug/kg,
   Mann-Whitney p=0.257), on excess over a rolling-median background (411.3 vs
   203.8, p=0.205), or on local prominence (91.4 vs 65.7, p=0.162)."

Written from scratch: no import of output/analysis/common.py; rolling medians,
prominence, run widths, ranks, LOOCV and power are all re-implemented here.
Mann-Whitney U p-values come from scipy.stats.mannwhitneyu (exact where
possible).  Monte-Carlo steps use vectorised scipy calls (axis=) for speed.

Artifacts (output/verification/figures/):
  f5_metric_table.csv, f5_positions.csv, f5_power_curve.csv, f5_power_table.csv,
  f5_interleave.csv, f5_summary.json, f5_power.png, f5_log.txt
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

T0 = time.time()
PROJ = Path("/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project")
OUT = PROJ / "output" / "verification" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
_log: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _log.append(str(msg))


NUM = ["c01_depth_top_m", "c02_depth_ice_snow_m", "c03_samp_thick_cm",
       "c04_sulfate_ug_kg", "c05_density_ice_g_cm_3", "c06_depth_w_e_m",
       "c08_age_a_ad_ce", "c09_age_ka_bp"]

raw = pd.read_csv(PROJ / "data" / "observations.tsv", sep="\t", dtype=str, keep_default_na=False)
df = raw.copy()
for c in NUM:
    df[c] = pd.to_numeric(raw[c].str.strip(), errors="coerce")
df["signal"] = raw["c07_signal"].str.strip()

n_rows = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
TH = df["c03_samp_thick_cm"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
RHO = df["c05_density_ice_g_cm_3"].to_numpy(float)
W = df["c06_depth_w_e_m"].to_numpy(float)
AGE = df["c08_age_a_ad_ce"].to_numpy(float)
labels = df["signal"].to_numpy()

lab_rows = np.where(labels != "")[0]
named = np.array([i for i in lab_rows if labels[i].startswith("event_")])
unid = np.array([i for i in lab_rows if labels[i] == "unidentified_event"])
ref = np.array([i for i in lab_rows if labels[i] != "" and not labels[i].startswith("event_")
                and labels[i] != "unidentified_event"])
log("=" * 78)
log("FINDING 5 INDEPENDENT VERIFICATION (v2, vectorised)")
log("=" * 78)
log(f"rows={n_rows}; reference={list(labels[ref])}; named={len(named)}; unidentified={len(unid)}")

# ---------------------------------------------------------------------------
# 0. hand-rolled summary statistics (independent of the original code)
# ---------------------------------------------------------------------------
def centered_median(x: np.ndarray, half: int) -> np.ndarray:
    """Centered rolling median; NaN-aware (matches pandas' NaN-skipping)."""
    n = len(x)
    return np.array([np.nanmedian(x[max(0, i - half): min(n, i + half + 1)]) for i in range(n)])


def window_min(x: np.ndarray, half: int) -> np.ndarray:
    n = len(x)
    return np.array([x[max(0, i - half): min(n, i + half + 1)].min() for i in range(n)])


def run_width(x: np.ndarray, i: int, thr: float) -> int:
    if x[i] <= thr:
        return 0
    j = k = i
    while j - 1 >= 0 and x[j - 1] > thr:
        j -= 1
    while k + 1 < len(x) and x[k + 1] > thr:
        k += 1
    return k - j + 1


def rank_desc(x: np.ndarray, i: int) -> int:
    return int((x > x[i]).sum() + 1)


BG101 = centered_median(SO, 50)
BG201 = centered_median(SO, 100)
BG51 = centered_median(SO, 25)
BG_pd = pd.Series(SO).rolling(101, center=True, min_periods=25).median().to_numpy()
log(f"[sanity] hand-rolled 101-pt centered median == pandas rolling: "
    f"{np.allclose(BG101, BG_pd, equal_nan=True)}")

EXC101 = SO - BG101
EXC201 = SO - BG201
EXC51 = SO - BG51

# local linear detrend over +/-25 samples, residual at the centre point (NaN-safe:
# c04 has one missing value at row 61 which must not poison whole windows)
def local_linear_resid(x: np.ndarray, half: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        t = np.arange(lo, hi, dtype=float)
        seg = x[lo:hi]
        m = np.isfinite(seg)
        if not np.isfinite(x[i]) or m.sum() < 3:
            continue
        b, a = np.polyfit(t[m], seg[m], 1)
        out[i] = x[i] - (a + b * i)
    return out


EXC_LIN = local_linear_resid(SO, 25)
PROM1 = SO - np.minimum(np.concatenate([[SO[1]], SO[:-1]]),
                        np.concatenate([SO[1:], [SO[-2]]]))
PROM5 = SO - window_min(SO, 5)

metrics: dict[str, np.ndarray] = {
    "c04_sulfate_ug_kg": SO,
    "excess_over_BG101": EXC101,
    "excess_over_BG201": EXC201,
    "excess_over_BG51": EXC51,
    "excess_local_linear25": EXC_LIN,
    "local_prominence_pm1": PROM1,
    "local_prominence_pm5": PROM5,
    "c03_samp_thick_cm": TH,
    "c05_density_g_cm_3": RHO,
    "c01_depth_top_m": D,
    "c06_depth_w_e_m": W,
    "c08_age_a_ad_ce": AGE,
}
for t in (150.0, 200.0, 250.0, 300.0):
    metrics[f"width_abs_gt{int(t)}"] = np.array([run_width(SO, i, t) for i in range(n_rows)], float)
for t in (100.0, 150.0, 200.0):
    metrics[f"width_excess101_gt{int(t)}"] = np.array(
        [run_width(EXC101, i, t) for i in range(n_rows)], float)
metrics["fwhm_over_bg101"] = np.array(
    [run_width(SO, i, (SO[i] + BG101[i]) / 2.0) for i in range(n_rows)], float)
metrics["rank_c04_desc"] = np.array([rank_desc(SO, i) for i in range(n_rows)], float)
metrics["rank_excess101_desc"] = np.array([rank_desc(EXC101, i) for i in range(n_rows)], float)
metrics["peak_to_bg_ratio"] = SO / BG101

mass = np.zeros(n_rows)
for i in range(n_rows):
    if SO[i] > BG101[i] + 50.0:
        j = k = i
        while j - 1 >= 0 and SO[j - 1] > BG101[j - 1] + 50.0:
            j -= 1
        while k + 1 < n_rows and SO[k + 1] > BG101[k + 1] + 50.0:
            k += 1
        mass[i] = float(np.sum(SO[j:k + 1] - BG101[j:k + 1]))
metrics["excess_mass_run_bg+50"] = mass

METRIC_NAMES = list(metrics.keys())
ALL = np.concatenate([named, unid])
IS_NAMED = np.concatenate([np.ones(5, bool), np.zeros(9, bool)])
n1, n2 = 5, 9


def mwu_full(a: np.ndarray, b: np.ndarray) -> dict:
    U = float(stats.mannwhitneyu(a, b, alternative="greater").statistic)
    try:
        p_exact = float(stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
        method = "exact"
    except Exception:
        p_exact = float("nan")
        method = "unavailable"
    p_asym = float(stats.mannwhitneyu(a, b, alternative="two-sided",
                                      method="asymptotic", use_continuity=False).pvalue)
    p_asym_cc = float(stats.mannwhitneyu(a, b, alternative="two-sided",
                                         method="asymptotic", use_continuity=True).pvalue)
    return {"U": U, "auc": U / (n1 * n2), "p_exact": p_exact, "exact_method": method,
            "p_asymptotic": p_asym, "p_asymptotic_cc": p_asym_cc}


def vector_aucs(vals2d: np.ndarray) -> np.ndarray:
    """AUC (P(named>unid)) for shape (nsim, 14) arrays, first 5 named."""
    a = vals2d[:, :5]
    b = vals2d[:, 5:]
    gt = (a[:, :, None] > b[:, None, :]).sum(axis=(1, 2))
    eq = (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))
    return (gt + 0.5 * eq) / (5 * 9)


rng = np.random.default_rng(20190101)
rows = []
tbl_start = time.time()
for name in METRIC_NAMES:
    v = metrics[name]
    r = {"metric": name}
    r.update(mwu_full(v[named], v[unid]))
    r["named_median"] = float(np.median(v[named]))
    r["unid_median"] = float(np.median(v[unid]))
    r["named_mean"] = float(np.mean(v[named]))
    r["unid_mean"] = float(np.mean(v[unid]))
    r["named_min"] = float(np.min(v[named]))
    r["named_max"] = float(np.max(v[named]))
    r["unid_min"] = float(np.min(v[unid]))
    r["unid_max"] = float(np.max(v[unid]))
    sd_a = np.std(v[named], ddof=1)
    sd_b = np.std(v[unid], ddof=1)
    r["sd_within"] = float(np.sqrt(((n1 - 1) * sd_a ** 2 + (n2 - 1) * sd_b ** 2) / (n1 + n2 - 2)))
    r["median_diff"] = float(np.median(v[named]) - np.median(v[unid]))
    # label-preserving bootstrap of the two class medians and their difference
    ia = rng.integers(0, n1, size=(6000, n1))
    ib = rng.integers(0, n2, size=(6000, n2))
    ma = np.median(v[named][ia], axis=1)
    mb = np.median(v[unid][ib], axis=1)
    r["named_median_ci95"] = [float(np.percentile(ma, 2.5)), float(np.percentile(ma, 97.5))]
    r["unid_median_ci95"] = [float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))]
    r["median_diff_ci95"] = [float(np.percentile(ma - mb, 2.5)),
                             float(np.percentile(ma - mb, 97.5))]
    # bootstrap CI for the observed AUC (resample within classes)
    vn = v[named][rng.integers(0, n1, size=(6000, n1))]
    vu = v[unid][rng.integers(0, n2, size=(6000, n2))]
    gt = (vn[:, :, None] > vu[:, None, :]).sum(axis=(1, 2))
    eq = (vn[:, :, None] == vu[:, None, :]).sum(axis=(1, 2))
    aucs = (gt + 0.5 * eq) / (n1 * n2)
    r["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "f5_metric_table.csv", index=False)
log(f"[timing] metric table (incl. 24x6000 vectorised bootstraps) took "
    f"{time.time() - tbl_start:.1f}s")

log("")
log("=" * 78)
log("1. RECOMPUTED MANN-WHITNEY COMPARISONS (scipy.stats.mannwhitneyu)")
log("=" * 78)
for name in ["c04_sulfate_ug_kg", "excess_over_BG101", "local_prominence_pm1"]:
    r = tab[tab.metric == name].iloc[0]
    log(f"{name:26s} medians {r['named_median']:10.3f} vs {r['unid_median']:10.3f} ug/kg | "
        f"U={r['U']:4.0f} AUC={r['auc']:.3f} | p_exact={r['p_exact']:.4f} "
        f"({r['exact_method']}) | p_asym={r['p_asymptotic']:.4f} | "
        f"p_asym+cc={r['p_asymptotic_cc']:.4f}")
log("")
log("ALL discriminators, sorted by p_exact (Bonferroni factor = %d):" % len(tab))
for _, r in tab.sort_values("p_exact").iterrows():
    log(f"  {r['metric']:26s} med {r['named_median']:10.2f} vs {r['unid_median']:10.2f} | "
        f"U={r['U']:4.0f} AUC={r['auc']:.3f} | p_exact={r['p_exact']:.4f} | "
        f"p_bonf={min(1.0, r['p_exact'] * len(tab)):.3f} | "
        f"med_diff 95%CI [{r['median_diff_ci95'][0]:.2f}, {r['median_diff_ci95'][1]:.2f}]")

log("")
log("headline metrics, label-preserving bootstrap CIs:")
for name in ["c04_sulfate_ug_kg", "excess_over_BG101", "local_prominence_pm1",
             "c01_depth_top_m", "c05_density_g_cm_3", "fwhm_over_bg101",
             "c03_samp_thick_cm", "excess_mass_run_bg+50"]:
    r = tab[tab.metric == name].iloc[0]
    log(f"  {name:24s} medians [{r['named_median_ci95'][0]:9.2f}, {r['named_median_ci95'][1]:9.2f}]"
        f" vs [{r['unid_median_ci95'][0]:9.2f}, {r['unid_median_ci95'][1]:9.2f}] | "
        f"diff {r['median_diff']:9.2f} CI [{r['median_diff_ci95'][0]:9.2f}, "
        f"{r['median_diff_ci95'][1]:9.2f}]")

# Welch t-test on density (parametric cross-check; Mann-Whitney may miss a mean shift)
tt = stats.ttest_ind(RHO[named], RHO[unid], equal_var=False)
log("")
log(f"[cross-check] Welch t on c05 density: t={tt.statistic:.3f}, p={tt.pvalue:.4f} "
    f"(medians {np.median(RHO[named]):.4f} vs {np.median(RHO[unid]):.4f}, "
    f"means {RHO[named].mean():.4f} vs {RHO[unid].mean():.4f})")
tt_so = stats.ttest_ind(SO[named], SO[unid], equal_var=False)
log(f"[cross-check] Welch t on c04 sulfate: t={tt_so.statistic:.3f}, p={tt_so.pvalue:.4f}")

# ---------------------------------------------------------------------------
# 1b. permutation null for every discriminator (exact enumeration, tied data safe)
#     The scipy 'exact' p assumes no ties; permutation handles ties properly.
# ---------------------------------------------------------------------------
log("")
log("=" * 78)
log("1b. PERMETRIC PERMUTATION p (all C(14,5)=2002 class splits) + FAMILY-WISE")
log("=" * 78)
M = np.array([metrics[m][ALL] for m in METRIC_NAMES])  # (K,14)
obs_auc = np.array([tab[tab.metric == m].iloc[0]["auc"] for m in METRIC_NAMES])
K = len(METRIC_NAMES)
null_auc = np.full((comb(14, 5), K), np.nan)
for k, combo in enumerate(combinations(range(14), 5)):
    sel = np.zeros(14, bool)
    sel[list(combo)] = True
    a, b = M[:, sel], M[:, ~sel]
    gt = (a[:, :, None] > b[:, None, :]).sum(axis=(1, 2))
    eq = (a[:, :, None] == b[:, None, :]).sum(axis=(1, 2))
    null_auc[k] = (gt + 0.5 * eq) / 45.0
perm_p = np.nanmean(np.abs(null_auc - 0.5) >= np.abs(obs_auc - 0.5)[None, :] - 1e-12, axis=0)
tab["p_permutation"] = perm_p
tab.to_csv(OUT / "f5_metric_table.csv", index=False)
obs_off = np.where(np.isfinite(obs_auc), np.abs(obs_auc - 0.5), np.nan)
max_off = np.nanmax(obs_off)
best_metric = METRIC_NAMES[int(np.nanargmax(obs_off))]
null_max = np.nanmax(np.abs(null_auc - 0.5), axis=1)
fw_p = float(np.nanmean(null_max >= max_off - 1e-12))
lo_p = float(np.nanmin(perm_p))
log(f"observed max |AUC-0.5| over {K} metrics = {max_off:.3f} "
    f"(driver: {best_metric}, AUC={obs_auc[METRIC_NAMES.index(best_metric)]:.3f}, "
    f"permutation p={perm_p[METRIC_NAMES.index(best_metric)]:.4f})")
log(f"smallest per-metric permutation p = {lo_p:.4f}; "
    f"family-wise permutation p = {fw_p:.4f}")
log("")
log("per-metric permutation p (two-sided) vs scipy exact p:")
for _, r in tab.sort_values("p_permutation").iterrows():
    log(f"  {r['metric']:26s} AUC={r['auc']:.3f} p_perm={r['p_permutation']:.4f} "
        f"p_exact={r['p_exact']:.4f}")

# ---------------------------------------------------------------------------
# 2. positions, interleaving, runs test, LOOCV
# ---------------------------------------------------------------------------
log("")
log("=" * 78)
log("2. POSITIONS AND INTERLEAVING")
log("=" * 78)
pos = pd.DataFrame({
    "row": ALL, "label": [labels[i] for i in ALL],
    "class": np.where(IS_NAMED, "named", "unidentified"),
    "depth_top_m": D[ALL], "depth_w_e_m": W[ALL], "age_ce": AGE[ALL],
}).sort_values("depth_top_m").reset_index(drop=True)
pos["order"] = np.arange(1, 15)
pos.to_csv(OUT / "f5_positions.csv", index=False)
log(pos.to_string(index=False))

order = pos["class"].to_numpy()
runs = 1 + int((order[1:] != order[:-1]).sum())


def runs_pvalue(obs_runs: int, a: int = 5, b: int = 9) -> float:
    tot_ = comb(a + b, a)
    p = 0.0
    for r in range(2, a + b + 1):
        if r % 2 == 0:
            ways = comb(a - 1, r // 2 - 1) * comb(b - 1, r // 2 - 1)
        else:
            ways = (comb(a - 1, (r - 1) // 2) * comb(b - 1, (r - 3) // 2) +
                    comb(b - 1, (r - 1) // 2) * comb(a - 1, (r - 3) // 2))
        if r <= obs_runs:
            p += 2 * ways / tot_
    return min(p, 1.0)


log("")
log(f"named depths      : {sorted(np.round(D[named], 2).tolist())}")
log(f"unidentified depth: {sorted(np.round(D[unid], 2).tolist())}")
log(f"named depth range [{D[named].min():.2f}, {D[named].max():.2f}] m; "
    f"unid [{D[unid].min():.2f}, {D[unid].max():.2f}] m; "
    f"overlap = [{max(D[named].min(), D[unid].min()):.2f}, "
    f"{min(D[named].max(), D[unid].max()):.2f}] m "
    f"(overlap length {min(D[named].max(), D[unid].max()) - max(D[named].min(), D[unid].min()):.2f} m "
    f"of a total span {D[ALL].max() - D[ALL].min():.2f} m)")
log(f"class runs in depth order = {runs} (2 = perfectly separable, 14 = maximally mixed); "
    f"Wald-Wolfowitz one-sided p = {runs_pvalue(runs):.4f}")
log(f"fraction of named inside unid range = "
    f"{np.mean((D[named] >= D[unid].min()) & (D[named] <= D[unid].max())):.2f}; "
    f"fraction of unid inside named range = "
    f"{np.mean((D[unid] >= D[named].min()) & (D[unid] <= D[named].max())):.2f}")

# interleaving detail: nearest unlike-class neighbour gap for every horizon
rows_i = []
for i in ALL:
    cls = "named" if IS_NAMED[list(ALL).index(i)] else "unidentified"
    other = unid if cls == "named" else named
    j = other[int(np.argmin(np.abs(D[other] - D[i])))]
    rows_i.append({"row": int(i), "label": labels[i], "class": cls,
                   "depth_m": D[i], "nearest_other_class_depth_m": D[j],
                   "gap_m": abs(D[i] - D[j])})
inter = pd.DataFrame(rows_i)
inter.to_csv(OUT / "f5_interleave.csv", index=False)
log(f"nearest unlike-class neighbour gap: median {inter.gap_m.median():.2f} m, "
    f"max {inter.gap_m.max():.2f} m; "
    f"{int((inter.gap_m < 3).sum())}/14 horizons have an unlike-class neighbour within 3 m")


def loocv_threshold(vals: np.ndarray, is_named: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(vals)
    preds = np.zeros(n, int)
    for i in range(n):
        tr = np.ones(n, bool)
        tr[i] = False
        vt, yt = vals[tr], is_named[tr]
        best_acc, best_dir, best_t = max(yt.mean(), 1 - yt.mean()), 1.0, float(np.median(vt))
        cand = np.unique(vt)
        cand = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else np.array([cand[0]])
        for t in cand:
            for d in (1.0, -1.0):
                pred = (vt > t) if d > 0 else (vt < t)
                acc = float((pred == yt).mean())
                if acc > best_acc + 1e-12:
                    best_acc, best_dir, best_t = acc, d, float(t)
        pi = (vals[i] > best_t) if best_dir > 0 else (vals[i] < best_t)
        preds[i] = int(pi)
    return float((preds == is_named).mean()), preds


log("")
log("single-feature threshold classifier, leave-one-out (baseline = 9/14 = 0.643):")
loocv_rows = []
for m in ["c01_depth_top_m", "c06_depth_w_e_m", "c04_sulfate_ug_kg", "excess_over_BG101",
          "local_prominence_pm1", "c05_density_g_cm_3", "c03_samp_thick_cm",
          "fwhm_over_bg101", "excess_mass_run_bg+50", "c08_age_a_ad_ce",
          "excess_over_BG201", "local_prominence_pm5"]:
    acc, preds = loocv_threshold(metrics[m][ALL], IS_NAMED)
    r = tab[tab.metric == m].iloc[0]
    loocv_rows.append({"metric": m, "loocv_accuracy": acc, "baseline": 9 / 14,
                       "auc": float(r["auc"]), "p_exact": float(r["p_exact"])})
    log(f"  {m:24s} LOOCV acc = {acc:.3f}  AUC = {r['auc']:.3f}  p_exact = {r['p_exact']:.4f}  "
        f"(depth uses the fixed direction named=shallower)")

# the depth rule: named shallower / deeper only in the 13-37 m window
dep = D[ALL]
log("")
log(f"outlier-rule accuracy (named iff depth < 13 m or depth > 37 m): "
    f"{float((((dep < 13) | (dep > 37)) == IS_NAMED).mean()):.3f}")
# exact McNemar: depth LOOCV vs sulfate LOOCV
acc_d, pr_d = loocv_threshold(D[ALL], IS_NAMED)
acc_s, pr_s = loocv_threshold(SO[ALL], IS_NAMED)
b01 = int(((pr_d != IS_NAMED) & (pr_s == IS_NAMED)).sum())
b10 = int(((pr_d == IS_NAMED) & (pr_s != IS_NAMED)).sum())
if b01 + b10 > 0:
    p_mcn = float(min(1.0, 2 * sum(comb(b01 + b10, k) for k in range(0, min(b01, b10) + 1))
                      / 2 ** (b01 + b10)))
else:
    p_mcn = float("nan")
log(f"depth LOOCV acc={acc_d:.3f} vs sulfate LOOCV acc={acc_s:.3f}; discordant "
    f"b01={b01}, b10={b10}; exact McNemar p={p_mcn:.3f}" if b01 + b10 else
    f"depth LOOCV acc={acc_d:.3f} vs sulfate LOOCV acc={acc_s:.3f}; zero discordant pairs")

# --------------------------------------------------------------------------
# 2b. HEAD-TO-HEAD: position (depth) vs amplitude (sulfate) as classifiers
#     Best single-cut rule, in-sample accuracy + exact permutation p over all
#     C(14,5) label assignments, plus LOOCV accuracy of the same rule family.
# --------------------------------------------------------------------------
log("")
log("=" * 78)
log("2b. POSITION vs AMPLITUDE, SINGLE-THRESHOLD RULES")
log("=" * 78)


def best_cut(xs: np.ndarray, y: np.ndarray) -> dict:
    """Best single threshold, BOTH directions (named if x>cut or x<cut)."""
    srt = np.sort(xs)
    cands = (srt[:-1] + srt[1:]) / 2
    A = np.zeros((2 * len(cands), len(xs)), bool)
    for i, c in enumerate(cands):
        A[2 * i] = xs > c
        A[2 * i + 1] = xs < c
    accs = (A == y[None, :]).mean(axis=1)
    bi = int(np.argmax(accs))
    best_acc = float(accs[bi])
    cut = float(cands[bi // 2])
    direction = ">" if bi % 2 == 0 else "<"
    pred = A[bi]
    cnt = 0
    tot = 0
    for combo in combinations(range(len(xs)), int(y.sum())):
        yy = np.zeros(len(xs), bool)
        yy[list(combo)] = True
        if (A == yy[None, :]).mean(axis=1).max() >= best_acc - 1e-12:
            cnt += 1
        tot += 1
    return {"cut": cut, "direction": direction, "accuracy": best_acc, "perm_p": cnt / tot,
            "misclassified": [{"value": float(xs[i]), "true": "named" if y[i] else "unidentified"}
                              for i in range(len(xs)) if pred[i] != y[i]]}


head_rows = []
best_rule = {}
for nm, v in [("c01_depth_top_m", D[ALL]), ("c06_depth_w_e_m", W[ALL]),
              ("c04_sulfate_ug_kg", SO[ALL]), ("excess_over_BG101", EXC101[ALL]),
              ("peak_to_bg_ratio", (SO / BG101)[ALL])]:
    r = best_cut(v, IS_NAMED)
    acc_l, _ = loocv_threshold(v, IS_NAMED)
    r.update({"metric": nm, "loocv_accuracy": acc_l})
    head_rows.append(r)
    log(f"  {nm:22s} best cut = {r['cut']:9.3f} ({r['direction']})  in-sample acc = "
        f"{r['accuracy']:.3f}  exact perm p = {r['perm_p']:.4f}  LOOCV acc = {acc_l:.3f}")
    log(f"      misclassified: {[(round(m['value'], 2), m['true']) for m in r['misclassified']]}")
pd.DataFrame(head_rows).to_csv(OUT / "f5_position_vs_amplitude.csv", index=False)

# family-wise permutation over the max single-cut accuracy across all metrics
Y = np.zeros((comb(14, 5), 14), bool)
for k, combo in enumerate(combinations(range(14), 5)):
    Y[k, list(combo)] = True
max_acc = np.zeros(len(Y))
for nm in ["c01_depth_top_m", "c06_depth_w_e_m", "c04_sulfate_ug_kg",
           "excess_over_BG101", "peak_to_bg_ratio", "c03_samp_thick_cm",
           "local_prominence_pm1", "c05_density_g_cm_3"]:
    xs = metrics[nm][ALL]
    srt = np.sort(xs)
    cands = (srt[:-1] + srt[1:]) / 2
    B = np.zeros((2 * len(cands), 14), bool)
    for i, c in enumerate(cands):
        B[2 * i] = xs > c
        B[2 * i + 1] = xs < c
    acc = np.zeros(len(Y))
    for b in B:
        acc = np.maximum(acc, (Y == b[None, :]).mean(axis=1))
    max_acc = np.maximum(max_acc, acc)
obs_max_acc = max(r["accuracy"] for r in head_rows)
fw_cut_p = float((max_acc >= obs_max_acc - 1e-12).mean())
log("")
log(f"family-wise permutation over best-cut accuracy across 8 metrics: observed max = "
    f"{obs_max_acc:.3f}, p = {fw_cut_p:.4f}")

# does depth add anything over sulfate?  L2-regularised logistic via damped Newton
def logit_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 60) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xa.shape[1])
    P = np.eye(Xa.shape[1])
    P[0, 0] = 0.0
    for _ in range(iters):
        zz = np.clip(Xa @ b, -30, 30)
        p = 1 / (1 + np.exp(-zz))
        W = p * (1 - p) + 1e-9
        g = Xa.T @ (p - y) + lam * (P @ b)
        H = Xa.T @ (Xa * W[:, None]) + lam * P
        step = np.linalg.solve(H, g)
        b = b - step
        if np.abs(step).max() < 1e-9:
            break
    return b


def logit_loocv(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> float:
    correct = 0
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        beta = logit_fit(X[tr], y[tr], lam)
        z = np.concatenate([[1.0], X[i]]) @ beta
        correct += int((z > 0) == bool(y[i]))
    return correct / len(y)


zmean = lambda v: (v - v.mean()) / v.std()
Xsulf = zmean(SO[ALL])[:, None]
Xdepth = zmean(D[ALL])[:, None]
Xboth = np.column_stack([zmean(SO[ALL]), zmean(D[ALL])])
log("")
log("L2-regularised logistic LOOCV (lambda=1, z-scored features; descriptive only):")
for nm, X in [("sulfate only", Xsulf), ("depth only", Xdepth),
              ("sulfate + depth", Xboth)]:
    log(f"  {nm:18s} LOOCV accuracy = {logit_loocv(X, IS_NAMED):.3f}")

# ---------------------------------------------------------------------------
# 3. power, n1=5 vs n2=9
# ---------------------------------------------------------------------------
log("")
log("=" * 78)
log("3. POWER (two-sided Mann-Whitney, alpha=0.05, normal shift model)")
log("=" * 78)
NSIM = 4000
deltas = np.linspace(0.0, 4.0, 21)


def power_at_deltas(ds: np.ndarray) -> np.ndarray:
    """Power via the exact two-sided Mann-Whitney null CDF (no scipy per-sim calls)."""
    x = rng.normal(0.0, 1.0, (NSIM, 5))
    out = np.empty(len(ds))
    for k, d in enumerate(ds):
        y = rng.normal(d, 1.0, (NSIM, 9))
        gt = (x[:, :, None] > y[:, None, :]).sum(axis=(1, 2))
        eq = (x[:, :, None] == y[:, None, :]).sum(axis=(1, 2))
        U = gt + 0.5 * eq
        pv = np.ones(NSIM)
        m = U >= 22.5
        # exact two-sided p = min(1, 2 * one-sided tail of the exact null)
        pv[m] = np.minimum(1.0, 2 * np.mean(null_sums[None, :] >= U[m, None] - 1e-9, axis=1))
        pv[~m] = np.minimum(1.0, 2 * np.mean(null_sums[None, :] <= U[~m, None] + 1e-9, axis=1))
        out[k] = (pv < 0.05).mean()
    return out


from itertools import combinations as _comb
null_sums = np.array([sum(c) - 15 for c in _comb(range(1, 15), 5)], float)
assert len(null_sums) == 2002


pwr_start = time.time()
powers = power_at_deltas(deltas)
log(f"[timing] {len(deltas)}x{NSIM} vectorised power sims took {time.time() - pwr_start:.1f}s")
auc_of_delta = stats.norm.cdf(deltas / sqrt(2.0))
k80 = int(np.argmax(powers >= 0.80))
if powers[k80] >= 0.80 and k80 > 0:
    fine = np.linspace(deltas[k80 - 1], deltas[k80], 21)
    pfine = power_at_deltas(fine)
    kf = int(np.argmax(pfine >= 0.80))
    d80 = float(fine[kf]) if pfine[kf] >= 0.80 else float(deltas[k80])
else:
    d80 = float(deltas[k80]) if powers[k80] >= 0.80 else float("nan")
auc_of_delta = stats.norm.cdf(deltas / sqrt(2.0))
auc80 = float(stats.norm.cdf(d80 / sqrt(2.0)))
log(f"minimum detectable standardised shift at 80% power: delta = {d80:.3f} SD -> "
    f"AUC = {auc80:.3f}")
# binomial (Wilson) floor check: with 4000 sims the smallest AUC whose power CI
# excludes 0.80 is slightly above auc80; report the conservative value too.
def wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - r) / d


lo80 = None
for a, p in zip(auc_of_delta, powers):
    if wilson_lower(float(p), NSIM) >= 0.80:
        lo80 = float(a)
        break
min_detectable_auc_wilson = lo80 if lo80 is not None else float("nan")
log(f"Wilson-95%-lower-bound >= 0.80 at AUC = {min_detectable_auc_wilson:.3f}")
pw_at = lambda auc: float(np.interp(auc, auc_of_delta, powers))
for a in (0.60, 0.70, 0.75, 0.80, 0.90):
    log(f"  power at AUC {a:.2f}: {pw_at(a):.3f}")
pd.DataFrame({"delta_sd": deltas, "auc": auc_of_delta, "power": powers}).to_csv(
    OUT / "f5_power_curve.csv", index=False)

best = tab.sort_values("p_exact").iloc[0]
log("")
log(f"best raw discriminator: {best['metric']} (AUC={best['auc']:.3f}, "
    f"p_exact={best['p_exact']:.4f})")
for m in ["c04_sulfate_ug_kg", "excess_over_BG101", "local_prominence_pm1",
          "c01_depth_top_m", "c05_density_g_cm_3", "fwhm_over_bg101"]:
    r = tab[tab.metric == m].iloc[0]
    log(f"  {m:24s} AUC={r['auc']:.3f} -> achieved power {pw_at(r['auc']):.3f}")

power_rows = []
for _, r in tab.iterrows():
    raw80 = d80 * r["sd_within"]
    power_rows.append({"metric": r["metric"], "auc": r["auc"],
                       "achieved_power_at_observed_auc": float(pw_at(r["auc"])) if r["auc"] > 0.5 else 0.0,
                       "min_detectable_raw_shift_at_80pct": float(raw80),
                       "observed_median_diff": float(r["median_diff"]),
                       "pooled_sd": float(r["sd_within"])})
pd.DataFrame(power_rows).to_csv(OUT / "f5_power_table.csv", index=False)
log("")
log("minimum detectable raw shift at 80% power vs observed median difference:")
for row in sorted(power_rows, key=lambda z: -z["auc"]):
    log(f"  {row['metric']:26s} AUC={row['auc']:.3f} power={row['achieved_power_at_observed_auc']:.3f} "
        f"| min raw shift @80% = {row['min_detectable_raw_shift_at_80pct']:9.1f} "
        f"vs observed diff {row['observed_median_diff']:9.1f} (pooled SD {row['pooled_sd']:.1f})")

# ---------------------------------------------------------------------------
# 4. figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
axes[0].plot(auc_of_delta, powers, "-", color="#1f77b4")
axes[0].axhline(0.8, color="grey", ls="--", lw=1)
axes[0].axvline(auc80, color="grey", ls=":", lw=1)
for m in ["c04_sulfate_ug_kg", "excess_over_BG101", "local_prominence_pm1",
          "c01_depth_top_m", "c05_density_g_cm_3", "fwhm_over_bg101"]:
    r = tab[tab.metric == m].iloc[0]
    axes[0].plot([r["auc"]], [pw_at(r["auc"])], "o", ms=5)
    axes[0].annotate(m, (r["auc"], pw_at(r["auc"])), fontsize=6, xytext=(3, 3),
                     textcoords="offset points")
axes[0].set_xlabel("true AUC")
axes[0].set_ylabel("power")
axes[0].set_title(f"n=5 vs 9: 80% power needs AUC >= {auc80:.2f}")
axes[0].set_xlim(0.4, 1.0)
axes[0].set_ylim(0, 1.02)
bp = axes[1].boxplot([SO[named], SO[unid], D[named], D[unid]],
                     tick_labels=["SO named", "SO unid", "depth named", "depth unid"])
axes[1].set_title("amplitude overlap vs depth separation")
fig.tight_layout()
fig.savefig(OUT / "f5_power.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. summary
# ---------------------------------------------------------------------------
def rec(r):
    d = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
         for k, v in r.items()}
    return d


summary = {
    "claim": {"medians_sulfate": [507.6, 300.8], "p_sulfate": 0.257,
              "medians_excess": [411.3, 203.8], "p_excess": 0.205,
              "medians_prominence": [91.4, 65.7], "p_prominence": 0.162},
    "recomputed": {m: rec(tab[tab.metric == m].iloc[0].to_dict())
                   for m in ["c04_sulfate_ug_kg", "excess_over_BG101", "local_prominence_pm1"]},
    "logit_loocv": {"sulfate_only": logit_loocv(Xsulf, IS_NAMED),
                    "depth_only": logit_loocv(Xdepth, IS_NAMED),
                    "sulfate_plus_depth": logit_loocv(Xboth, IS_NAMED)},
    "family_wise_best_cut_p": fw_cut_p,
    "n_named": 5, "n_unidentified": 9,
    "metric_table": [rec(r) for r in tab.to_dict(orient="records")],
    "family_wise_permutation_p": fw_p,
    "family_wise_driver_metric": best_metric,
    "smallest_per_metric_permutation_p": lo_p,
    "observed_max_abs_auc_offset": max_off,
    "min_detectable_auc_80pct_wilson": min_detectable_auc_wilson,
    "positions": pos.to_dict(orient="records"),
    "class_runs_depth_order": runs,
    "runs_test_p": runs_pvalue(runs),
    "depth_ranges": {"named": [float(D[named].min()), float(D[named].max())],
                     "unidentified": [float(D[unid].min()), float(D[unid].max())]},
    "interleave": inter.to_dict(orient="records"),
    "loocv": loocv_rows,
    "position_vs_amplitude": head_rows,
    "loocv_depth_vs_sulfate": {"acc_depth": acc_d, "acc_sulfate": acc_s,
                               "mcnemar_b01": b01, "mcnemar_b10": b10, "p_mcnemar": p_mcn},
    "welch_density": {"t": float(tt.statistic), "p": float(tt.pvalue)},
    "welch_sulfate": {"t": float(tt_so.statistic), "p": float(tt_so.pvalue)},
    "power": {"n1": 5, "n2": 9, "alpha": 0.05, "nsim": NSIM,
              "min_detectable_delta_sd_80pct": d80, "min_detectable_auc_80pct": auc80,
              "power_curve": [{"auc": float(a), "power": float(p)}
                              for a, p in zip(auc_of_delta, powers)],
              "power_table": power_rows},
    "elapsed_s": time.time() - T0,
}
(OUT / "f5_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
(OUT / "f5_log.txt").write_text("\n".join(_log) + "\n")
log(f"[artifacts written to {OUT}] elapsed {time.time() - T0:.1f}s")
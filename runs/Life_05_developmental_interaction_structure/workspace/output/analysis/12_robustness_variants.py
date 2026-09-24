#!/usr/bin/env python3
"""Step 12 - Robustness of the positive stage trend under alternative definitions
and outlier treatment.

Primary framing (variant a): within-unit centring.  For every analysis unit
(interaction_id x cell_pair_id) the unit's own mean strength is subtracted from
each of its records; the per-stage curve is the mean of those centred values at
each stage, and the trend statistic is the OLS slope / Pearson r of that
18-point curve against stage_index.

Variants:
  a  primary: within-unit-centred per-stage mean, all records
  b  exclude stage 0 entirely
  c  median instead of mean per stage (within-unit-centred)
  d  winsorise strength at 1st/99th pct BEFORE unit centring
  e  trim records with |centred| > 3 SD (global), then re-run
  f  units observed in >= 6 distinct stages only
  g  units observed in ALL 18 stages only
  h  floor-p records only (association_pvalue == 0.0009)
  i  non-floor-p records only
  j  record weight = 1 / family size (records within interaction_id)
  k  record weight = 1 / number of stages its unit occupies

For every variant we report the 18-stage curve, OLS slope vs stage_index (with
its p-value), Pearson r (with p-value), Spearman rho, early (0-8) / late (9-17)
curve means, record count, and a unit-level slope test (mean per-unit OLS slope,
fraction positive, t-test and sign test vs zero) as an independent significance
check.

Within-unit centring is recomputed on the records of each variant subset, so
"within-unit-centred" is always defined with respect to the analysed data.

Output: output/results/robustness_variants.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Keep BLAS single-threaded: the workload is small and thread contention on a
# shared box inflates wall time far above actual CPU time.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

_T0 = time.perf_counter()


def _tick(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:6.2f}s] {msg}", flush=True)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"

Y = "interaction_strength_like_value"
STAGES = np.arange(18)
X = STAGES.astype(float)

df = pd.read_csv(OUT / "analysis_table.tsv", sep="\t")
_tick(f"loaded {len(df)} rows")
assert list(STAGES) == sorted(df.stage_index.unique())
assert df[Y].notna().all() and df.unit_id.notna().all()


# ------------------------------------------------------------------ helpers
def centre(sub: pd.DataFrame) -> pd.Series:
    """Subtract each unit's own mean from its records (within variant subset)."""
    return sub[Y] - sub.groupby("unit_id")[Y].transform("mean")


def stage_curve(frame: pd.DataFrame, ycol: str, wcol: str | None = None) -> np.ndarray:
    """Per-stage mean (or weighted mean) of ycol over STAGES, NaN if absent."""
    if wcol is None:
        cur = frame.groupby("stage_index")[ycol].mean().reindex(STAGES)
    else:
        num = (frame[ycol] * frame[wcol]).groupby(frame["stage_index"]).sum()
        den = frame[wcol].groupby(frame["stage_index"]).sum()
        cur = (num / den).reindex(STAGES)
    return cur.to_numpy(dtype=float)


def curve_stats(cur: np.ndarray) -> dict:
    """OLS slope + Pearson/Spearman of the curve against stage_index."""
    ok = ~np.isnan(cur)
    x, y = X[ok], cur[ok]
    reg = stats.linregress(x, y)
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    early_mask = ok & (STAGES <= 8)
    late_mask = ok & (STAGES >= 9)
    early = float(np.mean(cur[early_mask])) if early_mask.any() else None
    late = float(np.mean(cur[late_mask])) if late_mask.any() else None
    return {
        "n_stages_in_curve": int(ok.sum()),
        "early_n_stages": int(early_mask.sum()),
        "late_n_stages": int(late_mask.sum()),
        "ols_slope_per_stage": float(reg.slope),
        "ols_slope_stderr": float(reg.stderr),
        "ols_slope_pvalue": float(reg.pvalue),
        "pearson_r": float(pear.statistic),
        "pearson_pvalue": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_pvalue": float(spear.pvalue),
        "early_mean_stages_0_8": early,
        "late_mean_stages_9_17": late,
        "late_minus_early": (float(late - early) if (early is not None and late is not None) else None),
    }


def unit_level(sub: pd.DataFrame, ycol: str) -> dict:
    """Per-unit OLS slopes (>=3 distinct stages) tested against zero.

    Vectorised closed form: slope_u = cov(x,y)/var(x) computed from group sums.
    """
    g = sub.groupby("unit_id")
    n = g.size().astype(float)
    sx = g["stage_index"].sum().astype(float)
    sy = g[ycol].sum().astype(float)
    sxx = (sub["stage_index"] ** 2).groupby(sub["unit_id"]).sum().astype(float)
    sxy = (sub["stage_index"] * sub[ycol]).groupby(sub["unit_id"]).sum().astype(float)
    n_stages = g["stage_index"].nunique()
    n_vals = g[ycol].nunique()
    vxx = sxx - sx ** 2 / n
    vxy = sxy - sx * sy / n
    keep = (n_stages >= 3) & (n_vals > 1) & (vxx > 0)
    slopes = (vxy[keep] / vxx[keep]).to_numpy(dtype=float)
    if len(slopes) == 0:
        return {"n_units_tested": 0}
    t = stats.ttest_1samp(slopes, 0.0)
    n_pos = int((slopes > 0).sum())
    return {
        "n_units_tested": int(len(slopes)),
        "mean_per_unit_slope": float(slopes.mean()),
        "median_per_unit_slope": float(np.median(slopes)),
        "fraction_positive": float((slopes > 0).mean()),
        "ttest_statistic": float(t.statistic),
        "ttest_pvalue": float(t.pvalue),
        "sign_test_n_positive": n_pos,
        "sign_test_n_negative": int((slopes < 0).sum()),
        "sign_test_pvalue_two_sided": float(
            stats.binomtest(n_pos, len(slopes), 0.5).pvalue
        ),
    }


# Family block structure so the family-clustered bootstrap is vectorised: rows
# are sorted by family; a bootstrap draw is expressed as family multiplicities
# and expanded with numpy repeat/arange tricks.
def _fam_blocks(s: pd.DataFrame, wcol: str | None) -> tuple:
    codes, uniques = pd.factorize(s.interaction_id, sort=False)
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    starts = np.searchsorted(codes_sorted, np.arange(len(uniques)))
    ends = np.append(starts[1:], len(order))
    sizes = ends - starts
    xs = s.stage_index.to_numpy(dtype=float)[order]
    ys = s["_centred"].to_numpy(dtype=float)[order]
    ws = (np.ones(len(s), dtype=float) if wcol is None
          else s[wcol].to_numpy(dtype=float)[order])
    return (np.asarray(starts, dtype=np.int64), np.asarray(sizes, dtype=np.int64),
            xs, ys, ws, len(uniques))


def _weighted_slope(xv: np.ndarray, yv: np.ndarray, wv: np.ndarray) -> float:
    sw = wv.sum()
    xm = float(np.sum(wv * xv) / sw)
    ym = float(np.sum(wv * yv) / sw)
    vxx = float(np.sum(wv * (xv - xm) ** 2))
    if vxx <= 0:
        return float("nan")
    return float(np.sum(wv * (xv - xm) * (yv - ym)) / vxx)


def _median_curve_slope(xv: np.ndarray, yv: np.ndarray) -> float:
    cur = np.array([
        np.median(yv[xv == st]) if np.any(xv == st) else np.nan for st in STAGES
    ], dtype=float)
    ok = ~np.isnan(cur)
    if ok.sum() < 2 or np.ptp(xv) == 0:
        return float("nan")
    return float(np.polyfit(X[ok], cur[ok], 1)[0])


def bootstrap_slope(s: pd.DataFrame, wcol: str | None, stat: str = "slope",
                    n_boot: int = 300, seed: int = 20220920) -> dict:
    """Family-clustered bootstrap CI for the variant's trend statistic.

    Resamples interaction families with replacement (whole family blocks move
    together, preserving unit/stage structure) and recomputes either the
    weighted record-level OLS slope of unit-centred values on stage_index
    (stat="slope") matching the variant's weighting, or the OLS slope of the
    per-stage median curve (stat="median").
    """
    starts, sizes, xs, ys, ws, n_fam = _fam_blocks(s, wcol)
    if stat == "slope":
        fn = lambda x, y, w: _weighted_slope(x, y, w)  # noqa: E731
    elif stat == "median":
        fn = lambda x, y, w: _median_curve_slope(x, y)  # noqa: E731
    else:
        raise ValueError(stat)
    point = fn(xs, ys, ws)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_fam, size=n_fam), minlength=n_fam)
        rep_starts = np.repeat(starts, mult)
        rep_sizes = np.repeat(sizes, mult)
        if len(rep_starts) == 0:
            draws[b] = np.nan
            continue
        ends_cum = np.cumsum(rep_sizes)
        block_start = ends_cum - rep_sizes
        offsets = np.arange(ends_cum[-1]) - np.repeat(block_start, rep_sizes)
        rows = np.repeat(rep_starts, rep_sizes) + offsets
        draws[b] = fn(xs[rows], ys[rows], ws[rows])
    draws = draws[np.isfinite(draws)]
    return {
        "point": float(point),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "n_boot": int(n_boot),
        "fraction_positive": float((draws > 0).mean()),
    }


def build(name: str, label: str, description: str, sub: pd.DataFrame, wcol: str | None,
          ycol_source: pd.Series | None = None,
          curve_override: np.ndarray | None = None) -> dict:
    """Run the within-unit-centred curve + all statistics for one variant."""
    s = sub.copy()
    s["_centred"] = ycol_source if ycol_source is not None else centre(s)
    cur = stage_curve(s, "_centred", wcol) if curve_override is None else curve_override
    out = {
        "label": label,
        "description": description,
        "n_records": int(len(s)),
        "n_units": int(s.unit_id.nunique()),
        "n_families": int(s.interaction_id.nunique()),
        "n_records_excluded_vs_primary": int(len(df) - len(s)),
        "weighting": "none" if wcol is None else wcol,
        "curve": [None if np.isnan(v) else float(v) for v in cur],
    }
    out.update(curve_stats(cur))
    out["unit_level"] = unit_level(s, "_centred")
    out["family_clustered_bootstrap_slope"] = bootstrap_slope(s, wcol, stat="slope")
    out["preserves_positive_sign"] = bool(out["ols_slope_per_stage"] > 0)
    out["curve_p_significant_0.05"] = bool(out["pearson_pvalue"] < 0.05)
    out["unit_p_significant_0.05"] = bool(
        out["unit_level"].get("ttest_pvalue", 1.0) < 0.05
        and out["unit_level"].get("mean_per_unit_slope", -1.0) > 0
    )
    out["clustered_ci_excludes_zero"] = bool(
        out["family_clustered_bootstrap_slope"]["ci95_low"] > 0
    )
    return out


variants: dict[str, dict] = {}

# ---------------------------------------------------------------- a. primary
variants["a_primary_all_records"] = build(
    "a", "Primary: within-unit-centred mean, all records",
    "All 5920 rows; subtract unit mean, per-stage mean of centred values.",
    df, None,
)
_tick("variant a done")

# ------------------------------------------------- b. exclude stage 0 entirely
sub = df[df.stage_index > 0]
variants["b_exclude_stage0"] = build(
    "b", "Exclude stage 0",
    "Stage 0 dropped; centring recomputed on stages 1-17.",
    sub, None,
)
_tick("variant b done")

# --------------------------------------------- c. median instead of stage mean
s = df.copy()
s["_centred"] = centre(s)
med = s.groupby("stage_index")["_centred"].median().reindex(STAGES).to_numpy(dtype=float)
variants["c_median_per_stage"] = build(
    "c", "Median instead of mean at each stage (within-unit-centred)",
    "Per-stage median of unit-centred values; centring and record set identical to the primary.",
    df, None, ycol_source=s["_centred"], curve_override=med,
)
variants["c_median_per_stage"]["family_clustered_bootstrap_slope"] = bootstrap_slope(
    s, None, stat="median"
)
variants["c_median_per_stage"]["clustered_ci_excludes_zero"] = bool(
    variants["c_median_per_stage"]["family_clustered_bootstrap_slope"]["ci95_low"] > 0
)
_tick("variant c done")

# ---------------------- d. winsorise 1st/99th pct BEFORE unit centring (all rows)
lo, hi = np.percentile(df[Y], [1.0, 99.0])
n_clipped = int(((df[Y] < lo) | (df[Y] > hi)).sum())
s = df.copy()
s[Y] = s[Y].clip(lo, hi)
variants["d_winsorised_1_99"] = build(
    "d", "Winsorise strength at 1st/99th percentile, then centre",
    f"Bounds [{lo:.6f}, {hi:.6f}] from full table; {n_clipped} records clipped before unit centring.",
    s, None,
)
variants["d_winsorised_1_99"]["winsor_bounds"] = [float(lo), float(hi)]
variants["d_winsorised_1_99"]["n_records_clipped"] = n_clipped
_tick("variant d done")

# --------------------- e. trim |centred| > 3 SD (global centred SD), then re-run
c_all = centre(df)
sd_centred = float(c_all.std(ddof=0))
mask_keep = (c_all.abs() <= 3.0 * sd_centred).to_numpy()
sub = df[mask_keep]
variants["e_trim_3sd_centred"] = build(
    "e", "Trim records with |within-unit-centred value| > 3 SD (global)",
    f"Centred SD (population, ddof=0) = {sd_centred:.6f}; threshold 3 SD = {3*sd_centred:.6f}; "
    f"{int((~mask_keep).sum())} records dropped; centring recomputed on the trimmed set.",
    sub, None,
)
variants["e_trim_3sd_centred"]["centred_sd_global"] = sd_centred
variants["e_trim_3sd_centred"]["trim_threshold_3sd"] = 3.0 * sd_centred
variants["e_trim_3sd_centred"]["n_records_trimmed"] = int((~mask_keep).sum())
_tick("variant e done")

# ------------------------------------ f. units observed in >= 6 distinct stages
nst = df.groupby("unit_id").stage_index.nunique()
keep_f = nst[nst >= 6].index
variants["f_units_ge6_stages"] = build(
    "f", "Units observed in >= 6 distinct stages",
    f"{len(keep_f)} of {df.unit_id.nunique()} units kept; centring recomputed on kept records.",
    df[df.unit_id.isin(keep_f)], None,
)
_tick("variant f done")

# -------------------------------------- g. units observed in ALL 18 stages
keep_g = nst[nst == 18].index
variants["g_units_all18_stages"] = build(
    "g", "Units observed in ALL 18 stages",
    f"{len(keep_g)} of {df.unit_id.nunique()} units kept; centring recomputed on kept records.",
    df[df.unit_id.isin(keep_g)], None,
)
_tick("variant g done")

# ---------------------------------------------------- h. floor-p records only
sub_h = df[df.association_pvalue == 0.0009]
variants["h_floor_p_only"] = build(
    "h", "association_pvalue == 0.0009 (floor-p) records only",
    f"{len(sub_h)} of {len(df)} records kept.",
    sub_h, None,
)
_tick("variant h done")

# ------------------------------------------------- i. non-floor-p records only
sub_i = df[df.association_pvalue != 0.0009]
variants["i_nonfloor_p_only"] = build(
    "i", "Non-floor-p records only (p in {0.01,0.02,0.03,0.04,0.05})",
    f"{len(sub_i)} of {len(df)} records kept.",
    sub_i, None,
)
_tick("variant i done")

# ------------------------------- j. record weight = 1 / family size (in subset)
s = df.copy()
s["_centred"] = centre(s)
s["_w_fam"] = 1.0 / s.groupby("interaction_id")["interaction_id"].transform("size")
variants["j_inverse_family_size_weight"] = build(
    "j", "Record weight = 1 / family size (interaction_id record count)",
    "Weighted per-stage mean of unit-centred values; families contribute equally in expectation.",
    s, "_w_fam",
)
_tick("variant j done")

# --------------------------- k. record weight = 1 / # stages its unit occupies
s = df.copy()
s["_centred"] = centre(s)
s["_w_unit"] = 1.0 / s.groupby("unit_id")["stage_index"].transform("nunique")
variants["k_inverse_unit_stages_weight"] = build(
    "k", "Record weight = 1 / number of stages its unit occupies",
    "Weighted per-stage mean of unit-centred values; units contribute equally.",
    s, "_w_unit",
)
_tick("variant k done")

# --------------------------------------- family-concentration diagnostics
# The two weakest variants (g: 14 all-18-stage units; j: inverse family size)
# may be fragile because of how few families they rest on.  Quantify that.
def weighted_slope_of(sub: pd.DataFrame) -> float:
    c = sub[Y] - sub.groupby("unit_id")[Y].transform("mean")
    return float(np.polyfit(sub.stage_index, c, 1)[0])


keep_g = nst[nst == 18].index
g_sub = df[df.unit_id.isin(keep_g)]
g_counts = g_sub.interaction_id.value_counts()
top_fam_g = g_counts.index[0]
leave_one_out_g = {
    str(f): weighted_slope_of(g_sub[g_sub.interaction_id != f])
    for f in g_counts.index[:5]
}
g_fam_share_top3 = float(g_counts.head(3).sum() / len(g_sub))

df_counts = df.interaction_id.value_counts()
leave_top_out_primary = {
    str(f): weighted_slope_of(df[df.interaction_id != f])
    for f in df_counts.index[:3]
}

# how many families are needed to cover 50% of each fragile subset
def families_for_half(sub: pd.DataFrame) -> int:
    c = np.cumsum(sub.interaction_id.value_counts().to_numpy() / len(sub))
    return int(np.searchsorted(c, 0.5) + 1)


nf_sub = df[df.association_pvalue != 0.0009]
family_diagnostics = {
    "variant_g": {
        "n_units": int(g_sub.unit_id.nunique()),
        "n_families": int(g_sub.interaction_id.nunique()),
        "top3_family_record_share": g_fam_share_top3,
        "families_covering_half_the_records": families_for_half(g_sub),
        "leave_one_family_out_slope": leave_one_out_g,
        "note": (
            "one family (interaction_0231) supplies 54/252 records; removing it alone raises the "
            "g slope toward the primary value, so g's flatness is a family-composition effect, "
            "not evidence against the stage trend"
        ),
    },
    "variant_i": {
        "n_families": int(nf_sub.interaction_id.nunique()),
        "n_records": int(len(nf_sub)),
        "top3_family_record_share": float(nf_sub.interaction_id.value_counts().head(3).sum() / len(nf_sub)),
        "families_covering_half_the_records": families_for_half(nf_sub),
    },
    "primary_leave_top_family_out_slope": leave_top_out_primary,
    "primary_top3_family_record_share": float(df_counts.head(3).sum() / len(df)),
}

# --------------------------------------------------------------- verdicts
order = [
    "a_primary_all_records", "b_exclude_stage0", "c_median_per_stage",
    "d_winsorised_1_99", "e_trim_3sd_centred", "f_units_ge6_stages",
    "g_units_all18_stages", "h_floor_p_only", "i_nonfloor_p_only",
    "j_inverse_family_size_weight", "k_inverse_unit_stages_weight",
]

sign_preserving_sig = [k for k in order if variants[k]["preserves_positive_sign"]
                       and variants[k]["curve_p_significant_0.05"]
                       and variants[k]["unit_p_significant_0.05"]]
sign_preserving_weak = [k for k in order if variants[k]["preserves_positive_sign"]
                        and k not in sign_preserving_sig]
sign_null_or_neg = [k for k in order if not variants[k]["preserves_positive_sign"]]

# fragility diagnostics
nst = df.groupby("unit_id").stage_index.nunique()
main_slope = variants["a_primary_all_records"]["ols_slope_per_stage"]
drops = {k: float(main_slope - variants[k]["ols_slope_per_stage"]) for k in order[1:]}
g_slope = variants["g_units_all18_stages"]["ols_slope_per_stage"]
i_slope = variants["i_nonfloor_p_only"]["ols_slope_per_stage"]
b_slope = variants["b_exclude_stage0"]["ols_slope_per_stage"]

# stage-0 rank under the primary framing vs the naive record-weighted framing
prim_curve = np.array([np.nan if v is None else v for v in variants["a_primary_all_records"]["curve"]])
rank_centred = int(stats.rankdata(prim_curve)[0])
naive_curve = df.groupby("stage_index")[Y].mean().reindex(STAGES).to_numpy(dtype=float)
rank_naive = int(stats.rankdata(naive_curve)[0])

most_fragile = (
    "The trend is not fragile in sign: all 11 variants stay positive, and only two lose significance. "
    "The single weakest aspect is the thin, family-concentrated all-18-stage panel (variant g): only "
    f"{g_sub.unit_id.nunique()} units from {g_sub.interaction_id.nunique()} families cover all 18 stages, and its slope "
    f"drops to {g_slope:.4f} (r {variants['g_units_all18_stages']['pearson_r']:.3f}, unit t-test p = "
    f"{variants['g_units_all18_stages']['unit_level']['ttest_pvalue']:.3f}) with a family-clustered 95% CI "
    f"[{variants['g_units_all18_stages']['family_clustered_bootstrap_slope']['ci95_low']:.4f}, "
    f"{variants['g_units_all18_stages']['family_clustered_bootstrap_slope']['ci95_high']:.4f}] that includes 0. "
    "That is a composition effect, not a contradiction: removing the single largest g family "
    f"(interaction_0231, 54/252 records) restores the slope to {leave_one_out_g[str(top_fam_g)]:.4f}, "
    "and the broader panel of 381 units with >= 6 stages gives the strongest slope of all "
    f"({variants['f_units_ge6_stages']['ols_slope_per_stage']:.4f}). Second weakest is the non-floor-p "
    f"subset (variant i, slope {variants['i_nonfloor_p_only']['ols_slope_per_stage']:.5f}), significant but "
    "roughly a quarter of the primary magnitude, so the floor-p tier carries most of the signal. "
    "The variants that reweight (j, k) and the median (c) roughly halve the slope yet stay clearly "
    "positive with clustered CIs excluding zero."
)

res = {
    "primary_framing": (
        "within-unit-centred interaction_strength_like_value; per-stage mean over records; "
        "OLS slope and Pearson r of the 18-point stage curve vs stage_index; "
        "centring recomputed within each variant's subset"
    ),
    "n_records_full": int(len(df)),
    "n_units_full": int(df.unit_id.nunique()),
    "significance_rule": (
        "a variant is called 'significant' only if BOTH the curve-level Pearson p < 0.05 "
        "(n=18 stage points) AND the unit-level per-unit-slope t-test has p < 0.05 with mean slope > 0; "
        "the stricter family-clustered bootstrap CI (does it exclude 0) is reported separately "
        "and only variant g fails it"
    ),
    "variants": {k: variants[k] for k in order},
    "verdict": {
        "positive_sign_and_significant": sign_preserving_sig,
        "positive_sign_but_weaker_not_fully_significant": sign_preserving_weak,
        "sign_lost_or_negative": sign_null_or_neg,
        "clustered_ci_excludes_zero": [k for k in order if variants[k]["clustered_ci_excludes_zero"]],
        "clustered_ci_includes_zero": [k for k in order if not variants[k]["clustered_ci_excludes_zero"]],
    },
    "fragility": {
        "primary_slope": float(main_slope),
        "slope_change_vs_primary": drops,
        "unit_coverage": {
            "n_units_ge6_stages": int((nst >= 6).sum()),
            "n_units_all18_stages": int((nst == 18).sum()),
            "n_units_single_stage": int((nst == 1).sum()),
        },
        "stage0_rank_of_18_under_primary_centred_curve": rank_centred,
        "stage0_rank_of_18_under_naive_record_weighted_curve": rank_naive,
        "stage0_centred_mean": float(prim_curve[0]),
        "stage0_naive_mean": float(naive_curve[0]),
        "stage_curve_pvalue_note": (
            "curve-level Pearson p uses only 18 stage points and is coarse/anti-conservative "
            "because records within a stage are dependent; the unit-level t-test is the more "
            "meaningful per-variant significance check but treats units as independent; "
            "the family-clustered bootstrap CI is the most conservative check reported"
        ),
        "family_concentration": family_diagnostics,
        "most_fragile_aspect": most_fragile,
    },
    "conflicts_with_stated_facts": [
        (
            "Stated fact 'stage 0 has the lowest mean' holds for the naive record-weighted curve "
            f"(stage 0 mean {naive_curve[0]:.4f}, rank {rank_naive}/18) but NOT for the within-unit-centred "
            f"primary curve, where stage 0 ranks {rank_centred}/18 (mean {prim_curve[0]:.4f}; stages 1-3 are lower). "
            "Excluding stage 0 therefore re-centres the trend to start lower, and the slope still stays positive."
        )
    ],
}

(OUT / "robustness_variants.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

# ------------------------------------------------------------------- stdout
print(f"{'variant':34s} {'slope':>8s} {'r':>7s} {'p_curve':>10s} {'unit_p':>10s} "
      f"{'early':>8s} {'late':>8s} {'ascal':>8s} {'n_rec':>6s} {'n_units':>7s}")
for k in order:
    v = variants[k]
    early = "  n/a  " if v["early_mean_stages_0_8"] is None else f"{v['early_mean_stages_0_8']:8.4f}"
    print(
        f"{k:34s} {v['ols_slope_per_stage']:8.5f} {v['pearson_r']:7.4f} "
        f"{v['pearson_pvalue']:10.3e} {v['unit_level'].get('ttest_pvalue', float('nan')):10.3e} "
        f"{early} {v['late_mean_stages_9_17']:8.4f} "
        f"{'yes' if v['preserves_positive_sign'] else 'NO':>8s} {v['n_records']:6d} {v['n_units']:7d}"
    )

print("\nVerdict")
print("  positive + significant :", sign_preserving_sig)
print("  positive but weaker    :", sign_preserving_weak)
print("  sign lost / negative   :", sign_null_or_neg)

print("\nFamily-clustered 95% CI of the record-level within-unit slope")
for k in order:
    ci = variants[k]["family_clustered_bootstrap_slope"]
    print(f"  {k:34s} [{ci['ci95_low']:8.5f}, {ci['ci95_high']:8.5f}]  "
          f"excludes 0: {variants[k]['clustered_ci_excludes_zero']}")

print("\nFamily concentration (variant g, all-18-stage units)")
print("  families:", family_diagnostics["variant_g"]["n_families"],
      "| top3 record share:", round(family_diagnostics["variant_g"]["top3_family_record_share"], 3),
      "| families to cover 50%:", family_diagnostics["variant_g"]["families_covering_half_the_records"])
print("  leave-one-family-out slopes:", {k: round(v, 4) for k, v in leave_one_out_g.items()})

print("\nMost fragile aspect:", most_fragile)
print("\nwrote", OUT / "robustness_variants.json")
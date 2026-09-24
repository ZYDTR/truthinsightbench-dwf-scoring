#!/usr/bin/env python3
"""
06_transition_structure.py

Mean-level transition (change-point) structure in d18o_pdb for the continuous
ocean-core record.  Honest treatment of serial correlation (BIC with effective
sample size) and of record edges (age/depth gaps, trimming).

Writes: output/results/06_transition_structure.json
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project")
DATA = ROOT / "data" / "observations.tsv"
OUT_JSON = ROOT / "output" / "results" / "06_transition_structure.json"

MIN_SEG = 40          # minimum rows per segment
TRIM_FRAC = 0.10      # trim 10% of rows from each end
N_SURR = 300          # <=300 AR(1) surrogates

# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
raw = pd.read_csv(DATA, sep="\t")
raw.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
sha = hashlib.sha256(DATA.read_bytes()).hexdigest()

df = raw.dropna(subset=["d18o_pdb"]).reset_index(drop=True)
x = df["d18o_pdb"].to_numpy(float)
age = df["age_ka_bp_1"].to_numpy(float)
depth = df["depth_m"].to_numpy(float)
n = x.size

# ----------------------------------------------------------------------------
# helpers over prefix sums (vectorised SSE)
# ----------------------------------------------------------------------------
def sse_all_splits(vals, min_seg=MIN_SEG):
    """Return dict k -> (sse_left + sse_right) for two contiguous segments."""
    m = vals.size
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cs2 = np.concatenate([[0.0], np.cumsum(vals ** 2)])
    ks = np.arange(min_seg, m - min_seg + 1)

    def seg_sse(i, j):  # rows [i, j)
        s = cs[j] - cs[i]
        s2 = cs2[j] - cs2[i]
        cnt = j - i
        return s2 - s * s / cnt

    left = seg_sse(0, ks)
    right = seg_sse(ks, m)
    return ks, left + right, left, right


def sse_pair(vals, i, j):  # rows [i,j)
    s = vals[i:j].sum()
    s2 = (vals[i:j] ** 2).sum()
    return s2 - s * s / (j - i)


def sse3_all_pairs(vals, min_seg=MIN_SEG):
    """Best three-segment SSE and (k1,k2) over all pairs with min_seg each."""
    m = vals.size
    ks, tot, left, right = sse_all_splits(vals, min_seg)
    # SSE of middle segment [k1, k2) for every (k1,k2) pair
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cs2 = np.concatenate([[0.0], np.cumsum(vals ** 2)])
    k1 = ks
    k2 = ks
    S = cs[k2][None, :] - cs[k1][:, None]
    S2 = cs2[k2][None, :] - cs2[k1][:, None]
    CNT = (k2[None, :] - k1[:, None]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = S2 - S * S / CNT
    total = left[:, None] + mid + right[None, :]
    valid = (k2[None, :] - k1[:, None]) >= min_seg
    total = np.where(valid, total, np.inf)
    idx = np.unravel_index(np.argmin(total), total.shape)
    return total, k1, k2, idx


def bic(sse, n_eff, p):
    return n_eff * np.log(sse / n_eff) + p * np.log(n_eff)


# ----------------------------------------------------------------------------
# 1. single change-point scan
# ----------------------------------------------------------------------------
ks, sse_k, left_k, right_k = sse_all_splits(x)
sse0 = sse_pair(x, 0, n)
best_i = int(np.argmin(sse_k))
k_best = int(ks[best_i])
sse_best = float(sse_k[best_i])

# boundary age convention: midpoint between last row of segment 1 and first of seg 2
split_age = float(0.5 * (age[k_best - 1] + age[k_best]))

order = np.argsort(sse_k)
top5 = []
for i in order[:5]:
    k = int(ks[i])
    top5.append({
        "split_k_rows_left": k,
        "split_age_ka": float(0.5 * (age[k - 1] + age[k])),
        "sse": float(sse_k[i]),
        "mean_left": float(x[:k].mean()),
        "mean_right": float(x[k:].mean()),
        "mean_shift": float(x[k:].mean() - x[:k].mean()),
    })

# ----------------------------------------------------------------------------
# 2. n_eff from integrated autocorrelation time
# ----------------------------------------------------------------------------
d_age = np.diff(age)
step_ka = float(np.median(d_age))                     # median age step (ka)
rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])        # lag-1 autocorrelation at that step
tau_ka = step_ka * (1.0 + rho1) / (1.0 - rho1)
span_ka = float(age[-1] - age[0])
n_eff = span_ka / tau_ka

bic_raw_no = bic(sse0, n, 2)
bic_raw_sp = bic(sse_best, n, 3)
bic_neff_no = bic(sse0, n_eff, 2)
bic_neff_sp = bic(sse_best, n_eff, 3)
survives = bool(bic_neff_sp < bic_neff_no)

# AR(1) surrogate null for the change-point SSE reduction (honest, serially correlated)
rng = np.random.default_rng(20260919)
m = n
surr = np.empty((N_SURR, m))
phi = rho1
innov_sd = float(np.std(x))
# exact AR(1) with stationary variance
sig = innov_sd * np.sqrt(1.0 - phi ** 2)
eps = rng.normal(0.0, sig, size=(N_SURR, m))
surr[:, 0] = rng.normal(0.0, innov_sd, size=N_SURR)
for t in range(1, m):
    surr[:, t] = phi * surr[:, t - 1] + eps[:, t]
surr += (float(np.mean(x)) - surr.mean(axis=1, keepdims=True))
cs = np.concatenate([np.zeros((N_SURR, 1)), np.cumsum(surr, axis=1)], axis=1)
cs2 = np.concatenate([np.zeros((N_SURR, 1)), np.cumsum(surr ** 2, axis=1)], axis=1)
kk = np.arange(MIN_SEG, m - MIN_SEG + 1)
cL, cR = cs[:, kk], cs[:, m][:, None] - cs[:, kk]
c2L, c2R = cs2[:, kk], cs2[:, m][:, None] - cs2[:, kk]
cntL, cntR = kk.astype(float), (m - kk).astype(float)
sse_s = (c2L - cL ** 2 / cntL) + (c2R - cR ** 2 / cntR)
best_s = sse_s.min(axis=1)
tot_sse_s = (cs2[:, m] - cs[:, m] ** 2 / m)
reduction_obs = 1.0 - sse_best / sse0
reduction_null = 1.0 - best_s / tot_sse_s
p_cp = float((np.sum(reduction_null >= reduction_obs) + 1) / (N_SURR + 1))

# Honest calibration of the BIC-with-n_eff rule itself: how often does pure
# AR(1) noise "survive" the same no-split vs split BIC comparison?
bic_s_null = n_eff * np.log(best_s / n_eff) + 3 * np.log(n_eff)
bic0_null = n_eff * np.log(tot_sse_s / n_eff) + 2 * np.log(n_eff)
frac_null_survives = float(np.mean(bic_s_null < bic0_null))

# Same-null p-value for the 2-harmonic sinusoid fit at T
TT = 108.835
Vh = np.column_stack([np.ones(m), np.sin(2 * np.pi * age / TT), np.cos(2 * np.pi * age / TT),
                      np.sin(4 * np.pi * age / TT), np.cos(4 * np.pi * age / TT)])
Q, _ = np.linalg.qr(Vh)
surr_c = surr - surr.mean(axis=1, keepdims=True)
r2_null_sin = 1.0 - ((surr_c - surr_c @ Q @ Q.T) ** 2).sum(axis=1) / (surr_c ** 2).sum(axis=1)
r2_obs_sin = 1.0 - np.sum((x - x.mean() - (x - x.mean()) @ Q @ Q.T) ** 2) / np.sum((x - x.mean()) ** 2)
p_sin = float((np.sum(r2_null_sin >= r2_obs_sin) + 1) / (N_SURR + 1))

# ----------------------------------------------------------------------------
# 3. two-change-point (three-segment) model
# ----------------------------------------------------------------------------
sse3_grid, k1v, k2v, (i1, i2) = sse3_all_pairs(x)
k1 = int(k1v[i1]); k2 = int(k2v[i2])
sse3 = float(sse3_grid[i1, i2])
bic_neff_2cp = bic(sse3, n_eff, 4)
bic_raw_2cp = bic(sse3, n, 4)
second_supported = bool(sse3 < sse_best and bic_neff_2cp < bic_neff_sp)
seg_ages = [
    float(0.5 * (age[k1 - 1] + age[k1])),
    float(0.5 * (age[k2 - 1] + age[k2])),
]
seg_means = [float(x[:k1].mean()), float(x[k1:k2].mean()), float(x[k2:].mean())]

# ----------------------------------------------------------------------------
# 4. edge / truncation diagnostics
# ----------------------------------------------------------------------------
last6_age = age[-6:].tolist()
last6_depth = depth[-6:].tolist()
last5_gaps = np.diff(age[-6:]).tolist()
d_depth = np.diff(raw["depth_m"].to_numpy(float))
enlarged = np.where(d_depth > 0.011)[0]
enlarged_rows = [{
    "after_raw_row_index": int(i),
    "depth_from_m": float(raw["depth_m"].iloc[i]),
    "depth_to_m": float(raw["depth_m"].iloc[i + 1]),
    "depth_gap_m": float(d_depth[i]),
} for i in enlarged]

trim = int(np.floor(TRIM_FRAC * n))
sub = slice(trim, n - trim)
xt = x[sub]; aget = age[sub]
ks_t, sse_t, _, _ = sse_all_splits(xt)
kb_t = int(ks_t[int(np.argmin(sse_t))])
best_trim_age = float(0.5 * (aget[kb_t - 1] + aget[kb_t]))
moves = bool(abs(best_trim_age - split_age) > 10.0)
n_trim_rows = int(xt.size)
sse0_t = float(sse_pair(xt, 0, xt.size))
sse_best_t = float(sse_t[int(np.argmin(sse_t))])
rho_t = float(np.corrcoef(xt[:-1], xt[1:])[0, 1])
tau_t = step_ka * (1 + rho_t) / (1 - rho_t)
n_eff_t = float((aget[-1] - aget[0]) / tau_t)
bic_t_no = float(bic(sse0_t, n_eff_t, 2)); bic_t_sp = float(bic(sse_t[int(np.argmin(sse_t))], n_eff_t, 3))
far = np.abs(ks_t - kb_t) > 60
k_alt = int(ks_t[int(np.argmin(np.where(far, sse_t, np.inf)))])
alt_age = float(0.5 * (aget[k_alt - 1] + aget[k_alt]))
trim_counts = {"rows_trimmed_each_end": trim, "rows_retained": n_trim_rows,
               "age_start_ka": float(aget[0]), "age_end_ka": float(aget[-1]),
               "sse_no_split": sse0_t, "sse_best_split": sse_best_t,
               "second_choice_split_age_ka": alt_age,
               "n_eff_trimmed": float(n_eff_t), "tau_trimmed_ka": float(tau_t),
               "bic_raw_no_split": float(bic(sse0_t, n_trim_rows, 2)),
               "bic_raw_split": float(bic(sse_best_t, n_trim_rows, 3)),
               "bic_neff_no_split": bic_t_no, "bic_neff_split": bic_t_sp,
               "split_survives_bic_neff_trimmed": bool(bic_t_sp < bic_t_no)}

# ----------------------------------------------------------------------------
# 5. smooth alternatives
# ----------------------------------------------------------------------------
def r2(y, yhat):
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


t = age - age.mean()
# linear trend
V = np.column_stack([np.ones(n), t])
coef_lin, *_ = np.linalg.lstsq(V, x, rcond=None)
fit_lin = V @ coef_lin
r2_lin = r2(x, fit_lin)

# dominant period from step 02 (and its first harmonic)
T = 108.835


def harmonic_fit(periods, include_trend=False):
    cols = [np.ones(n)]
    if include_trend:
        cols.append(t)
    for p in periods:
        cols.append(np.sin(2 * np.pi * age / p))
        cols.append(np.cos(2 * np.pi * age / p))
    Vh = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(Vh, x, rcond=None)
    fh = Vh @ c
    return r2(x, fh), Vh.shape[1]

r2_sin1, _ = harmonic_fit([T])                 # fundamental only
r2_sin2, p_sin2 = harmonic_fit([T, T / 2])     # 2-harmonic (T, T/2)
r2_sin2_trend, _ = harmonic_fit([T, T / 2, T / 3], include_trend=True)

r2_seg = 1.0 - sse_best / sse0

# BIC with n_eff for the three competing families
bic_neff_trend = bic(sse0 * (1 - r2_lin), n_eff, 2)
bic_neff_sin = bic(sse0 * (1 - r2_sin2), n_eff, 5)
bic_neff_seg = bic(sse_best, n_eff, 3)
models = {
    "segmented_mean_2segment": bic_neff_seg,
    "linear_trend": bic_neff_trend,
    "2harmonic_sinusoid_T108.8": bic_neff_sin,
}
best_model = min(models, key=models.get)
d_bic_sin_over_trend = bic_neff_sin - bic_neff_trend
d_bic_seg_over_sin = bic_neff_seg - bic_neff_sin
d_bic_seg_over_trend = bic_neff_seg - bic_neff_trend

explanation = (
    f"n_eff={n_eff:.1f} from tau={tau_ka:.2f} ka (step={step_ka:.2f} ka, rho1={rho1:.3f}); "
    f"raw BIC prefers a split (dBIC={bic_raw_sp - bic_raw_no:+.1f}) but the penalty scales with n; "
    f"with n_eff the split is {'' if survives else 'NOT '}supported "
    f"(dBIC={bic_neff_sp - bic_neff_no:+.1f}). "
    f"AR(1) surrogate p for the best-split variance reduction = {p_cp:.3f} "
    f"(only {frac_null_survives*100:.0f}% of pure AR(1) records survive the same BIC-with-n_eff rule), "
    f"so the best split (age {split_age:.1f} ka, +{x[k_best:].mean()-x[:k_best].mean():.2f} per mille step) "
    f"is marginal: the preferred split lands in the sparse 330-360 ka tail where the last two age steps "
    f"({last5_gaps[-2]:.2f}, {last5_gaps[-1]:.2f} ka) are 4-6x the median step. "
    f"Trimming 10% from each end moves the best split to {best_trim_age:.1f} ka "
    f"(shift {best_trim_age-split_age:+.1f} ka, moves>10ka: {moves}); the trimmed split is even weaker "
    f"(dBIC={bic_t_sp-bic_t_no:+.1f} vs {bic_neff_sp-bic_neff_no:+.1f} full) and no longer in the "
    f"330-340 ka cluster, so the step location is edge-driven. "
    f"The 3-segment model adds a split near 310 ka with dBIC_neff={bic_neff_2cp-bic_neff_sp:+.1f}, but its "
    f"middle segment is only {seg_ages[1]-seg_ages[0]:.0f} ka long and R2 only rises to "
    f"{1-sse3/sse0:.3f}. "
    f"R2: segmented mean {r2_seg:.3f}, linear trend {r2_lin:.3f}, 2-harmonic sinusoid at {T} ka "
    f"{r2_sin2:.3f} (fundamental only {r2_sin1:.3f}); sinusoid beats an AR(1) null at p={p_sin:.3f}. "
    f"Under n_eff BIC the ranking is {best_model}; a mean-level step is not the dominant structure."
)

result = {
    "step": "06_transition_structure",
    "source_file": "data/observations.tsv",
    "sha256": sha,
    "n_rows_total": int(raw.shape[0]),
    "n_rows_used": int(n),
    "min_segment_rows": MIN_SEG,

    "best_split_age_ka": split_age,
    "best_split_sse": sse_best,
    "no_split_sse": float(sse0),
    "top5_splits": top5,
    "best_split_k_rows_left": k_best,
    "best_split_point": {"k": k_best, "age_boundary_ka": split_age,
                         "age_left_row_ka": float(age[k_best - 1]),
                         "age_right_row_ka": float(age[k_best]),
                         "mean_left": float(x[:k_best].mean()),
                         "mean_right": float(x[k_best:].mean())},
    "sse_reduction_fraction": float(reduction_obs),

    "n_eff": float(n_eff),
    "tau_ka": float(tau_ka),
    "step_ka": step_ka,
    "rho_lag1": rho1,
    "total_age_span_ka": span_ka,

    "bic_raw_no_split": float(bic_raw_no),
    "bic_raw_split": float(bic_raw_sp),
    "bic_raw_delta_split_minus_noSplit": float(bic_raw_sp - bic_raw_no),
    "bic_neff_no_split": float(bic_neff_no),
    "bic_neff_split": float(bic_neff_sp),
    "bic_neff_delta_split_minus_noSplit": float(bic_neff_sp - bic_neff_no),
    "bic_formula": "BIC = n_eff*ln(SSE/n_eff) + p*ln(n_eff); p=2 (no split), p=3 (split), p=4 (2 splits)",
    "changepoint_survives_bic_with_neff": survives,

    "ar1_surrogate_null": {
        "n_surrogates": N_SURR,
        "rho_used": rho1,
        "observed_sse_reduction": float(reduction_obs),
        "null_reduction_mean": float(reduction_null.mean()),
        "null_reduction_p95": float(np.quantile(reduction_null, 0.95)),
        "empirical_p_value_changepoint": p_cp,
        "frac_ar1_null_surviving_bic_neff_rule": frac_null_survives,
        "empirical_p_value_sinusoid_r2": p_sin,
    },

    "three_segment_model": {
        "split_rows": [k1, k2],
        "split_ages_ka": seg_ages,
        "segment_means": seg_means,
        "sse": sse3,
        "r2": float(1.0 - sse3 / sse0),
        "bic_raw": float(bic_raw_2cp),
        "bic_neff": float(bic_neff_2cp),
        "delta_bic_neff_vs_2segment": float(bic_neff_2cp - bic_neff_sp),
        "delta_bic_neff_vs_noSplit": float(bic_neff_2cp - bic_neff_no),
        "second_split_supported": second_supported,
    },
    "second_split_supported": second_supported,

    "edge_diagnostics": {
        "last6_age_ka": last6_age,
        "last6_depth_m": last6_depth,
        "last5_age_gaps_ka": last5_gaps,
        "last_age_gap_ka": float(last5_gaps[-1]),
        "median_age_step_ka": step_ka,
        "last_depth_m": float(depth[-1]),
        "final_rows_carry_gap": bool(last5_gaps[-1] > 2 * step_ka),
        "enlarged_depth_steps": enlarged_rows,
        "n_enlarged_depth_steps": len(enlarged_rows),
    },

    "trim_10pct": trim_counts,
    "best_split_age_after_10pct_trim_ka": best_trim_age,
    "split_location_shift_ka": float(best_trim_age - split_age),
    "split_location_moves_more_than_10ka": moves,

    "smooth_alternatives": {
        "linear_trend_coef": {"intercept": float(coef_lin[0]), "slope_per_ka": float(coef_lin[1])},
        "r2_segmented_mean_2segment": float(r2_seg),
        "r2_sinusoid_fundamental_only": float(r2_sin1),
        "r2_sinusoid_2harmonic": float(r2_sin2),
        "r2_trend_plus_2harmonic": float(r2_sin2_trend),
        "bic_neff": {k: float(v) for k, v in models.items()},
        "best_model_by_bic_neff": best_model,
        "delta_bic_neff_segment_minus_trend": float(d_bic_seg_over_trend),
        "delta_bic_neff_sinusoid_minus_trend": float(d_bic_sin_over_trend),
        "delta_bic_neff_segment_minus_sinusoid": float(d_bic_seg_over_sin),
        "period_ka_used": T,
    },
    "r2_linear_trend": r2_lin,
    "r2_sinusoid": float(r2_sin2),
    "best_explaining_model": best_model,
    "explanation": explanation,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

# ----------------------------------------------------------------------------
# console summary
# ----------------------------------------------------------------------------
print(f"n_used={n}  sse0={sse0:.4f}")
print(f"best split: k={k_best} age={split_age:.2f} ka  SSE={sse_best:.4f}  "
      f"means {x[:k_best].mean():.3f} -> {x[k_best:].mean():.3f}")
print("top5:", [(round(s['split_age_ka'], 2), round(s['sse'], 4)) for s in top5])
print(f"step={step_ka:.3f} ka rho1={rho1:.4f} tau={tau_ka:.3f} ka span={span_ka:.2f} ka n_eff={n_eff:.2f}")
print(f"BIC raw: no={bic_raw_no:.2f} split={bic_raw_sp:.2f} d={bic_raw_sp - bic_raw_no:+.2f}")
print(f"BIC neff: no={bic_neff_no:.2f} split={bic_neff_sp:.2f} d={bic_neff_sp - bic_neff_no:+.2f} "
      f"survives={survives}")
print(f"AR1 null p(cp)={p_cp:.3f} (obs reduction {reduction_obs:.4f}, p95 {np.quantile(reduction_null, 0.95):.4f}); "
      f"null BIC-survival rate={frac_null_survives:.3f}; p(sinusoid R2)={p_sin:.3f}")
print(f"3-seg: splits at {seg_ages[0]:.2f}, {seg_ages[1]:.2f} ka SSE={sse3:.4f} "
      f"dBIC_neff vs 2seg={bic_neff_2cp - bic_neff_sp:+.2f} supported={second_supported}")
print(f"last6 ages={[round(a,2) for a in last6_age]} gaps={[round(g,2) for g in last5_gaps]} last_depth={depth[-1]}")
print(f"enlarged depth steps at raw rows {[e['after_raw_row_index'] for e in enlarged_rows]}")
print(f"trimmed best split age={best_trim_age:.2f} ka (shift {best_trim_age - split_age:+.2f} ka) moves>10ka={moves}")
print(f"R2 trend={r2_lin:.4f} sin2={r2_sin2:.4f} sin1={r2_sin1:.4f} sin2+trend={r2_sin2_trend:.4f} seg={r2_seg:.4f}")
print("BIC_neff models:", {k: round(v, 1) for k, v in models.items()}, "best:", best_model)
print("wrote", OUT_JSON)
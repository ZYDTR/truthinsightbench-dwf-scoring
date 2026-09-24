#!/usr/bin/env python3
"""
Adversarial verification of CANDIDATE CLAIM C3:

  "The sedimentation-rate maximum forms a single contiguous episode at ~240.9-241.9 ka
   (6 consecutive rows above the 99th percentile of LSR), where the deepest and fastest
   accumulation of the core occurs, and d18o there is locally low (values 2.65-3.42
   against a record median of 3.37)."

Independent re-derivation, then two attacks:
  ATTACK A - sampling-grid artefact: LSR = d(depth)/d(age).  Depth steps are 0.01 m for
             615 of 620 intervals, so the LSR ranking is essentially the ranking of the
             RECIPROCAL AGE STEP.  Do the extreme rows sit on enlarged depth steps, and
             what part of the LSR top end is fixed by the age-model step atoms?
  ATTACK B - differentiator sensitivity: row-paired secant, hand central difference,
             3-pt and 5-pt Savitzky-Golay derivatives (both on the raw uneven grid with
             the mean spacing and on a uniform resampled age grid), plus an
             age-grid-independent fixed-width window ("span") slope.
Plus: block-permutation null (circular block bootstrap, 300 surrogates, vectorised) for
      the "d18o locally low" sub-claim, and the depth-position ("deepest") sub-claim.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJ = "/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project"
DATA = os.path.join(PROJ, "data", "observations.tsv")
OUTJ = os.path.join(PROJ, "output", "results", "verify_C3.json")
OUTC = os.path.join(PROJ, "output", "results", "verify_C3_lsr.csv")

RNG = np.random.default_rng(20260919)
R4 = lambda x: round(float(x), 4)
R6 = lambda x: round(float(x), 6)

# ------------------------------------------------------------------------------- load
df = pd.read_csv(DATA, sep="\t")
df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
d = df["depth_m"].to_numpy(float)
a = df["age_ka_bp_1"].to_numpy(float)
y = df["d18o_pdb"].to_numpy(float)
n = len(df)
Dd = np.diff(d)
Da = np.diff(a)

# ------------------------------------------------------- differentiators (row level)
secant = Dd / Da                                  # interval i = [a_i, a_{i+1}], assigned to row i
cent = np.empty(n)
cent[0] = (d[1] - d[0]) / (a[1] - a[0])
cent[-1] = (d[-1] - d[-2]) / (a[-1] - a[-2])
cent[1:-1] = (d[2:] - d[:-2]) / (a[2:] - a[:-2])
grad = np.gradient(d, a)
da_mean = float(np.mean(Da))
sg3_raw = savgol_filter(d, 3, 2, deriv=1, delta=da_mean)
sg5_raw = savgol_filter(d, 5, 2, deriv=1, delta=da_mean)


def episodes(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    return [[int(s), int(e)] for s, e in zip(np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]])]


def top_struct(v):
    thr = float(np.quantile(v, 0.99))
    m = v > thr
    eps = episodes(m)
    return dict(p99=thr, n_above=int(m.sum()), n_episodes=len(eps), episodes=eps,
                row_indices=np.flatnonzero(m).tolist(), max=float(np.max(v)),
                argmax_row=int(np.argmax(v)), median=float(np.median(v)))


diff = {
    "row_paired_secant": secant,
    "np_gradient": grad,
    "hand_central_diff": cent,
    "savgol_w3_raw_grid": sg3_raw,
    "savgol_w5_raw_grid": sg5_raw,
}
central_vs_gradient_interior = float(np.max(np.abs(cent[1:-1] - grad[1:-1])))

# --------------------------------------- uniform resampling of depth(age), then diff
g_step = 0.25
g_age = np.arange(a[0], a[-1] + 1e-9, g_step)
g_depth = np.interp(g_age, a, d)
g_lsr_cent = np.gradient(g_depth, g_age)                      # central difference, uniform grid
g_lsr_sg3 = savgol_filter(g_depth, 3, 2, deriv=1, delta=g_step)
g_lsr_sg5 = savgol_filter(g_depth, 5, 2, deriv=1, delta=g_step)
g_lsr_sg9 = savgol_filter(g_depth, 9, 2, deriv=1, delta=g_step)
uni = {
    "uniform_grid_central": g_lsr_cent,
    "uniform_grid_savgol_w3": g_lsr_sg3,
    "uniform_grid_savgol_w5": g_lsr_sg5,
    "uniform_grid_savgol_w9": g_lsr_sg9,
}
uni_struct = {}
for k, v in uni.items():
    s = top_struct(v)
    s["argmax_age_ka"] = float(g_age[s["argmax_row"]])
    s["pct_rank_at_241ka"] = float((v <= v[int(np.argmin(np.abs(g_age - 241.0)))]).mean() * 100)
    uni_struct[k] = s

# --------------------- fixed-width window span slope on the uniform age grid (W ka)
def span_scan(W):
    k = int(round(W / g_step))
    if k >= len(g_age):
        return None
    sl = (g_depth[k:] - g_depth[:-k]) / (g_age[k:] - g_age[:-k])
    mid = g_age[:-k] + (g_age[k:] - g_age[:-k]) / 2.0
    j = int(np.argmax(sl))
    near = np.abs(mid - 241.0) < 1.0
    return dict(width_ka=W, best_slope_m_per_ka=R6(sl[j]), best_mid_age_ka=round(float(mid[j]), 2),
                slope_at_241ka=R6(sl[np.argmin(np.abs(mid - 241.0))]),
                pct_rank_of_241ka_window=float((sl <= sl[np.argmin(np.abs(mid - 241.0))]).mean() * 100),
                best_window_overlaps_241ka=bool(mid[j] - W / 2 < 242.0 and mid[j] + W / 2 > 240.0),
                best_window_is_very_close=bool(abs(mid[j] - 241.0) <= 1.0),
                n_local_maxima_above_241ka_max=int(np.sum(sl > sl[np.argmin(np.abs(mid - 241.0))])))


span = {f"W{w}ka": span_scan(w) for w in (1, 2, 4, 8, 16, 32)}

# ------------------------------------------------------------------- ATTACK A facts
dd_counts = {str(round(float(k), 2)): int(v) for k, v in zip(*np.unique(np.round(Dd, 4), return_counts=True))}
enlarged_rows = np.flatnonzero(Dd > 0.0101)
Da_rank_pct = np.array([float((Da <= Da[i]).mean() * 100) for i in range(len(Da))])
p99_secant = float(np.quantile(secant, 0.99))
p99_grad = float(np.quantile(grad, 0.99))
# does the secant p99 rule reduce to "depth step 0.01 and age step <= 0.23"?
p99_atom_rows = np.flatnonzero(secant > p99_secant)
corr_lsr_invDa = float(np.corrcoef(secant, 1.0 / Da)[0, 1])
n_atom_023 = int(np.sum((np.abs(secant - 0.01 / 0.23) < 1e-9)))
n_Da_le_023 = int(np.sum(Da <= 0.2301))
# what fraction of the 620 interval slopes are exactly a function of the age step alone?
lsr_unique_from_Da = len(np.unique(np.round(secant, 9)))

# -------------------------------------------------------------- claim-row accounting
CLAIM_SETS = {}
for name, v in (("np_gradient", grad), ("row_paired_secant", secant)):
    s = top_struct(v)
    CLAIM_SETS[name] = s

ep_set_grad = np.array(CLAIM_SETS["np_gradient"]["row_indices"])      # 411..417
ep_vals_all = y[ep_set_grad]
ep_nonnan = ep_vals_all[np.isfinite(ep_vals_all)]
ep_mean = float(np.mean(ep_nonnan))
rec_median = float(np.nanmedian(y))
rec_p = {k: float(v) for k, v in zip(["p1", "p5", "p10", "p25", "p50", "min"],
                                     np.nanpercentile(y, [1, 5, 10, 25, 50, 0]))}
ep_depth = d[ep_set_grad]
ep_age = a[ep_set_grad]

# ------------------------------------------------------------------- d18o block null
# circular block bootstrap replicas, block = 25 rows (~4x integral time scale), B = 300
nan_mask = np.isnan(y)
y_fill = y.copy()
y_fill[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), y[~nan_mask])
bl, B = 25, 300
n_blocks = int(np.ceil(n / bl))
st = RNG.integers(0, n, size=(B, n_blocks))
idx = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n]
y_aug = np.concatenate([y_fill, y_fill[:bl]])
surr = y_aug[idx]                                    # (B, n)


def block_null(ep_idx, nb=50):
    ep = surr[:, ep_idx].mean(axis=1)
    lo, hi = max(0, ep_idx[0] - nb), min(n, ep_idx[-1] + 1 + nb)
    ring = np.r_[np.arange(lo, ep_idx[0]), np.arange(ep_idx[-1] + 1, hi)]
    obs_ep = float(np.mean(y_fill[ep_idx]))
    obs_ring = float(np.mean(y_fill[ring]))
    obs_contrast = obs_ep - obs_ring
    sur_contrast = ep - surr[:, ring].mean(axis=1)
    return dict(ep_idx=[int(ep_idx[0]), int(ep_idx[-1])], n_ep_rows=int(ep_idx.size),
                obs_ep_mean=R4(obs_ep), obs_ring_mean=R4(obs_ring),
                obs_contrast=R4(obs_contrast),
                p_ep_mean_le_null=float((ep <= obs_ep).mean()),
                p_contrast_le_null=float((sur_contrast <= obs_contrast).mean()),
                null_p025=R4(np.quantile(ep, 0.025)), null_p975=R4(np.quantile(ep, 0.975)),
                null_median=R4(np.median(ep)))


d18o_tests = {
    "claim_6_rows_412_417": block_null(np.arange(412, 418)),
    "seven_rows_411_417": block_null(np.arange(411, 418)),
    "row410_only": block_null(np.array([410])),
}

# ---------------------------------------------------- depth-position ("deepest") fact
depth_pct_of_episode = float((d <= ep_depth.mean()).mean() * 100)
age_pct_of_episode = float((a <= ep_age.mean()).mean() * 100)
deeper_rows = int(np.sum(d > ep_depth.max()))

# ------------------------------------------------------- 6-row windows at row level
def best_windows(v, w):
    means = pd.Series(v).rolling(w).mean().to_numpy()
    j = int(np.nanargmax(means))
    return dict(w=w, best_start=j - w + 1, best_end=j,
                best_mean_lsr=R6(means[j]), p99_of_windows=R6(np.nanquantile(means, 0.99)),
                age_range_ka=[round(float(a[j - w + 1]), 2), round(float(a[j]), 2)],
                overlaps_claim=bool(j - w + 1 <= 412 <= j))


row_windows = {f"w{w}": best_windows(grad, w) for w in (5, 6, 7, 8)}

# ------------------------------------------------------------------------------ out
res = dict(
    claim_id="C3",
    n_rows=int(n), n_d18o=int(np.isfinite(y).sum()),
    hand_central_equals_np_gradient_interior_max_abs_diff=central_vs_gradient_interior,
    record=dict(depth_median_m=float(np.median(d)), depth_max_m=float(d.max()),
                age_max_ka=float(a.max()), d18o_median=rec_median,
                d18o_quantiles=rec_p, d18o_min_ka_at=str(round(float(a[np.nanargmin(y)]), 2))),
    # --- replicated claim facts
    claim_replication=dict(
        np_gradient_top=CLAIM_SETS["np_gradient"],
        row_paired_secant_top=CLAIM_SETS["row_paired_secant"],
        episode_rows_rows_411_417=dict(
            ages_ka=[round(float(x), 2) for x in ep_age],
            depths_m=[round(float(x), 3) for x in ep_depth],
            d18o=[None if not np.isfinite(v) else R4(v) for v in ep_vals_all],
            d18o_non_nan_n=int(ep_nonnan.size),
            d18o_min=R4(np.min(ep_nonnan)), d18o_max=R4(np.max(ep_nonnan)),
            d18o_mean=R4(ep_mean), d18o_mean_minus_record_median=R4(ep_mean - rec_median),
            n_below_record_median=int(np.sum(ep_nonnan < rec_median)),
            interval_age_steps_ka=[round(float(x), 3) for x in Da[ep_set_grad]],
            interval_depth_steps_m=[round(float(x), 3) for x in Dd[ep_set_grad]]),
    ),
    # --- ATTACK A
    attackA_sampling_grid=dict(
        depth_step_counts=dd_counts,
        enlarged_depth_step_rows=enlarged_rows.tolist(),
        overlap_p99_gradient_and_enlarged_depth=sorted(set(enlarged_rows.tolist()) &
                                                       set(CLAIM_SETS["np_gradient"]["row_indices"])),
        overlap_p99_secant_and_enlarged_depth=sorted(set(enlarged_rows.tolist()) &
                                                     set(CLAIM_SETS["row_paired_secant"]["row_indices"])),
        p99_secant=R6(p99_secant), p99_np_gradient=R6(p99_grad),
        corr_lsr_vs_inverse_age_step=corr_lsr_invDa,
        n_interval_slopes_at_0p01_over_0p23_atom=n_atom_023,
        n_interval_age_steps_le_0p23_ka=n_Da_le_023,
        n_intervals_at_age_step_0p24_ka=int(np.sum(np.abs(Da - 0.24) < 1e-9)),
        n_distinct_interval_slopes=lsr_unique_from_Da,
        episode_age_step_percentile_ranks=[round(float(Da_rank_pct[i]), 2) for i in ep_set_grad],
        max_lsr_possible_at_age_step_0p24=0.01 / 0.24,
        note="depth step is 0.01 m for 615/620 intervals, so LSR ranking ~ 1/age-step ranking",
    ),
    # --- ATTACK B
    attackB_differentiators=dict(
        row_level={k: top_struct(v) for k, v in diff.items()},
        row_level_pct_rank_of_episode_rows={
            k: [round(float((v <= v[i]).mean() * 100), 2) for i in ep_set_grad] for k, v in diff.items()},
        uniform_grid=uni_struct,
        span_slopes=span,
        row_windows=row_windows,
        differentiator_best_row_ages_ka={k: round(float(a[top_struct(v)["argmax_row"]]), 2)
                                         for k, v in diff.items()},
        uniform_grid_best_ages_ka={k: v["argmax_age_ka"] for k, v in uni_struct.items()},
    ),
    # --- d18o and depth sub-claims
    d18o_block_null=d18o_tests,
    depth_position=dict(episode_mean_depth_m=R4(float(ep_depth.mean())),
                        episode_depth_percentile_of_record=round(depth_pct_of_episode, 1),
                        n_rows_deeper_than_episode=deeper_rows,
                        deepest_row_depth_m=float(d.max()),
                        episode_age_percentile_of_record=round(age_pct_of_episode, 1)),
)
with open(OUTJ, "w") as f:
    json.dump(res, f, indent=2, default=str)

# ------------------------------------------------- extra decisive C3 sub-claim probes
# (i) identity: the secant p99 set vs the set of intervals with the smallest age-step atom
sec_rows = set(CLAIM_SETS["row_paired_secant"]["row_indices"])
atom_rows = set(np.flatnonzero(Da <= 0.2301).tolist())
# (ii) deepest part of the core vs the episode
deep_mask = d > 5.0
deep_rows_idx = np.flatnonzero(deep_mask)
res["extra_probes"] = dict(
    secant_p99_rows=sorted(sec_rows),
    intervals_with_age_step_le_0p23=sorted(atom_rows),
    secant_p99_set_equals_smallest_age_step_set=bool(sec_rows == atom_rows),
    n_rows_deeper_than_5m=int(deep_rows_idx.size),
    max_depth_m=float(d.max()),
    mean_lsr_deeper_than_5m=R6(float(np.mean(secant[np.clip(deep_rows_idx, 0, len(secant) - 1)]))),
    mean_lsr_record_median=R6(float(np.median(secant))),
    lsr_percentile_at_5_to_6p3m=round(float(np.mean(secant <= np.mean(secant[np.clip(deep_rows_idx, 0, len(secant) - 1)])) * 100), 1),
    last_deformed_rows=int((Da > 1.0).sum()),
    n_age_steps_above_1ka=int((Da > 1.0).sum()),
    fraction_of_intervals_at_modal_0p24_ka=round(float(np.mean(np.abs(Da - 0.24) < 1e-9)), 3),
    mean_age_step_in_window_230_250ka=round(float(np.mean(Da[(a[:-1] >= 230) & (a[:-1] <= 250)])), 3),
    mean_age_step_record=round(float(np.mean(Da)), 3),
)
with open(OUTJ, "w") as f:
    json.dump(res, f, indent=2, default=str)

pd.DataFrame({
    "row": np.arange(n), "depth_m": d, "age_ka": a, "d18o": y,
    "lsr_row_paired_secant": np.r_[secant, np.nan], "lsr_np_gradient": grad,
    "lsr_hand_central": cent, "lsr_savgol3_raw": sg3_raw, "lsr_savgol5_raw": sg5_raw,
}).to_csv(OUTC, index=False)

# ------------------------------------------------------------------------- console
print(json.dumps(res, indent=2, default=str))
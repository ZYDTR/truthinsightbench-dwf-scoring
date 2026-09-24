#!/usr/bin/env python3
"""
Adversarial verification of CANDIDATE CLAIM C4:

  "The two age coordinates age_ka_bp_1 and age_ka_bp_2 are identical to within
   0.005 ka for all rows deeper than 0.55 m and differ only for 52 rows in the
   depth interval 0.03-0.55 m, with a maximum absolute difference of 1.63 ka;
   therefore the choice of age coordinate cannot change any record-wide result."

(A) Re-derive the descriptive facts with independently written code.
(B) Attack the INFERENCE -- refit the dominant sinusoid under age_1 alone,
    age_2 alone, and on the 569 rows where the coordinates agree; then push
    adversarial alternative coordinates and perturbations far past what the
    claim itself involves, and find where the record-wide period estimate
    actually moves by more than the period resolution P^2/T.

BLAS threading hangs in this sandbox -> pin threads to 1 before numpy import.
All scans are chunked matrix multiplications (no Python loop over grid points);
surrogate counts <= 300.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import time

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project"
DATA = os.path.join(ROOT, "data", "observations.tsv")
OUTJ = os.path.join(ROOT, "output", "results", "verify_C4.json")
RNG = np.random.default_rng(20260919)

df = pd.read_csv(DATA, sep="\t")
df.columns = ["depth_m", "age1", "age2", "d18o"]
z = df.depth_m.to_numpy(float)
a1 = df.age1.to_numpy(float)
a2 = df.age2.to_numpy(float)
y = df.d18o.to_numpy(float)
assert len(df) == 621
CO = ~np.isnan(y)

CHUNK = 32


# ----------------------------------------------------------------------------
# core: vectorised scan
# ----------------------------------------------------------------------------
def scan(t, yy, grid, two_harm=True):
    tc = t - t.mean()
    n_ = tc.size
    yty = float(yy @ yy)
    sst = float(((yy - yy.mean()) ** 2).sum())
    cols = 6 if two_harm else 4
    r2 = np.empty(grid.size)
    amp = np.empty(grid.size)
    ph = np.empty(grid.size)
    coef = np.empty((grid.size, cols))
    for s in range(0, grid.size, CHUNK):
        e = min(s + CHUNK, grid.size)
        g = grid[s:e]
        wt = (2.0 * np.pi / g)[:, None] * tc[None, :]
        X = np.empty((g.size, n_, cols))
        X[:, :, 0] = 1.0
        X[:, :, 1] = tc
        X[:, :, 2] = np.cos(wt)
        X[:, :, 3] = np.sin(wt)
        if two_harm:
            X[:, :, 4] = np.cos(2.0 * wt)
            X[:, :, 5] = np.sin(2.0 * wt)
        N = np.matmul(X.transpose(0, 2, 1), X)
        b = np.tensordot(X, yy, axes=([1], [0]))
        c = np.linalg.solve(N, b[..., None])[..., 0]
        rss = (yty - 2.0 * np.einsum("gm,gm->g", c, b)
               + np.einsum("gm,gm->g", c, np.matmul(N, c[..., None])[..., 0]))
        r2[s:e] = 1.0 - rss / sst
        amp[s:e] = np.hypot(c[:, 2], c[:, 3])
        ph[s:e] = np.arctan2(-c[:, 3], c[:, 2])
        coef[s:e] = c
    return r2, amp, ph, coef


def fit_exact(t, yy, coarse_step=0.25, fstep=0.0005, half=0.4):
    """Two-stage fit; validated against a brute-force 0.0005-ka scan."""
    g = np.arange(40.0, 300.0 + 1e-12, coarse_step)
    r2, _, _, _ = scan(t, yy, g)
    P0 = float(g[int(np.nanargmax(r2))])
    gf = np.arange(P0 - half, P0 + half + 1e-12, fstep)
    rf, af, pf, cf = scan(t, yy, gf)
    j = int(np.nanargmax(rf))
    c = cf[j]
    return dict(period_ka=float(gf[j]), r2_full=float(rf[j]), amp=float(af[j]),
                phase_rad=float(pf[j]), b_lin=float(c[1]),
                intercept=float(c[0]))


def fit_fast(t, yy, step=0.5):
    """Coarse scan + parabolic vertex refinement (surrogates only)."""
    g = np.arange(40.0, 300.0 + 1e-12, step)
    r2, af, pf, _ = scan(t, yy, g)
    j = int(np.nanargmax(r2))
    if 0 < j < g.size - 1:
        y0, y1, y2 = r2[j - 1], r2[j], r2[j + 1]
        den = y0 - 2.0 * y1 + y2
        d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0
    else:
        d = 0.0
    return dict(period_ka=float(g[j] + d * step), r2_full=float(r2[j]),
                amp=float(af[j]), phase_rad=float(pf[j]))


def max_r2_over(t, yy, grid):
    r2, _, _, _ = scan(t, yy, grid)
    j = int(np.nanargmax(r2))
    return float(r2[j]), float(grid[j])


# ----------------------------------------------------------------------------
# PART A -- descriptive facts, independently re-derived
# ----------------------------------------------------------------------------
absdiff = np.abs(a1 - a2)
mask_top = z <= 0.55
mask_deep = z > 0.55
mask_diff = absdiff > 0.0
agree = absdiff < 0.005

A = dict(
    n_rows=int(len(df)),
    n_rows_absdiff_gt_0=int(mask_diff.sum()),
    n_rows_absdiff_gt_1e_9=int((absdiff > 1e-9).sum()),
    n_rows_absdiff_gt_0p005=int((absdiff > 0.005).sum()),
    n_rows_absdiff_exactly_0=int((absdiff == 0.0).sum()),
    n_rows_absdiff_in_open_0_0p005=int(((absdiff > 0.0) & (absdiff < 0.005)).sum()),
    differing_rows_equal_the_set_depth_le_0p55m=bool(
        np.array_equal(np.where(mask_diff)[0], np.where(mask_top)[0])),
    diff_rows_depth_min_m=float(z[mask_diff].min()),
    diff_rows_depth_max_m=float(z[mask_diff].max()),
    max_abs_diff_ka=float(absdiff.max()),
    max_abs_diff_depth_m=float(z[absdiff.argmax()]),
    max_abs_diff_signed_ka=float((a1 - a2)[absdiff.argmax()]),
    max_abs_diff_deeper_than_0p55m_ka=float(absdiff[mask_deep].max()),
    n_rows_deeper_than_0p55m=int(mask_deep.sum()),
    n_agree_rows=int(agree.sum()),
    frac_rows_differing=float(mask_diff.mean()),
    frac_depth_span_top_0p55m=float(0.55 / (z.max() - z.min())),
    frac_age_span_top_interval_age1=float(
        (a1[mask_top].max() - a1[mask_top].min()) / (a1.max() - a1.min())),
    age1_span_ka=float(a1.max() - a1.min()),
    age2_span_ka=float(a2.max() - a2.min()),
    age_span_rel_diff=float(abs((a2.max() - a2.min()) - (a1.max() - a1.min()))
                            / (a1.max() - a1.min())),
    top_of_core_age1_ka=float(a1[0]),
    top_of_core_age2_ka=float(a2[0]),
    top_of_core_offset_ka=float(a1[0] - a2[0]),
    top_interval_age_span_age1_ka=float(a1[mask_top].max() - a1[mask_top].min()),
    top_interval_age_span_age2_ka=float(a2[mask_top].max() - a2[mask_top].min()),
    max_abs_diff_as_pct_of_record_span=float(100 * absdiff.max()
                                             / (a1.max() - a1.min())),
    max_abs_diff_in_median_age_steps=float(absdiff.max() / 0.46),
)

# ----------------------------------------------------------------------------
# PART B -- record-wide fits under each coordinate
# ----------------------------------------------------------------------------
t_start = time.time()
sub_ag = agree & CO
assert sub_ag.sum() == 541
top_ok = mask_top & CO

fit_a1 = fit_exact(a1[CO], y[CO])
fit_a2 = fit_exact(a2[CO], y[CO])
fit_ag = fit_exact(a1[sub_ag], y[sub_ag])
fit_t1 = fit_exact(a1[top_ok], y[top_ok])
fit_t2 = fit_exact(a2[top_ok], y[top_ok])
fit_fast_err = abs(fit_fast(a1[CO], y[CO])["period_ka"] - fit_a1["period_ka"])

T_span = float(a1[CO].max() - a1[CO].min())
P_REF = fit_a1["period_ka"]
RAY = P_REF ** 2 / T_span      # period-space resolution at P (Rayleigh)

B = {}
for f, k in ((fit_a1, "age1"), (fit_a2, "age2"),
             (fit_ag, "rows_where_coords_agree"),
             (fit_t1, "top52_age1"), (fit_t2, "top52_age2")):
    B[f"{k}_period_ka"] = f["period_ka"]
    B[f"{k}_r2_full"] = f["r2_full"]
    B[f"{k}_amp"] = f["amp"]
    B[f"{k}_phase_deg"] = float(np.degrees(f["phase_rad"]))

B.update(dict(
    n_complete_rows=int(CO.sum()),
    n_rows_where_coords_agree=int(agree.sum()),
    n_agree_and_complete=int(sub_ag.sum()),
    t_span_complete_rows_ka=T_span,
    period_resolution_rayleigh_P2_over_T_ka=float(RAY),
    one_over_T_per_ka=float(1.0 / T_span),
    period_shift_age1_vs_age2_ka=abs(fit_a1["period_ka"] - fit_a2["period_ka"]),
    period_shift_age1_vs_age2_in_rayleigh=abs(
        fit_a1["period_ka"] - fit_a2["period_ka"]) / RAY,
    period_shift_age1_vs_age2_pct=float(100 * abs(
        fit_a1["period_ka"] - fit_a2["period_ka"]) / fit_a1["period_ka"]),
    period_shift_age1_vs_agree_rows_ka=abs(fit_a1["period_ka"] - fit_ag["period_ka"]),
    period_shift_age1_vs_agree_rows_in_rayleigh=abs(
        fit_a1["period_ka"] - fit_ag["period_ka"]) / RAY,
    r2_shift_age1_vs_age2=abs(fit_a1["r2_full"] - fit_a2["r2_full"]),
    amp_shift_age1_vs_age2=abs(fit_a1["amp"] - fit_a2["amp"]),
    amp_rel_shift_age1_vs_age2=abs(fit_a1["amp"] - fit_a2["amp"]) / fit_a1["amp"],
    phase_shift_age1_vs_age2_deg=abs(np.degrees(
        fit_a1["phase_rad"] - fit_a2["phase_rad"])),
    full_record_corr_age1=float(np.corrcoef(a1[CO], y[CO])[0, 1]),
    full_record_corr_age2=float(np.corrcoef(a2[CO], y[CO])[0, 1]),
    top52_corr_age1=float(np.corrcoef(a1[top_ok], y[top_ok])[0, 1]),
    top52_corr_age2=float(np.corrcoef(a2[top_ok], y[top_ok])[0, 1]),
    fit_fast_vs_exact_period_error_ka=float(fit_fast_err),
))
B["full_record_corr_abs_shift"] = abs(B["full_record_corr_age1"]
                                      - B["full_record_corr_age2"])
B["top52_corr_abs_shift"] = abs(B["top52_corr_age1"] - B["top52_corr_age2"])

# ----------------------------------------------------------------------------
# PART C -- broadband R^2 / amplitude / phase comparison, restricted to bands
#           the data actually resolve
# ----------------------------------------------------------------------------
GRID_ALL = np.arange(5.0, 400.0 + 1e-12, 0.5)
r2_1, amp_1, ph_1, _ = scan(a1[CO], y[CO], GRID_ALL)
r2_2, amp_2, ph_2, _ = scan(a2[CO], y[CO], GRID_ALL)
d_r2 = np.abs(r2_1 - r2_2)
d_amp_rel = np.abs(amp_1 - amp_2) / amp_1
d_ph = np.degrees(np.abs(np.angle(np.exp(1j * (ph_1 - ph_2)))))

band = {}
for thr in (0.0, 0.05, 0.10, 0.20, 0.30):
    m = r2_1 >= thr
    band[f"r2_ge_{thr:.2f}"] = dict(
        n_grid_pts=int(m.sum()),
        period_lo_ka=float(GRID_ALL[m].min()), period_hi_ka=float(GRID_ALL[m].max()),
        max_abs_dR2=float(d_r2[m].max()),
        max_rel_amp_change=float(d_amp_rel[m].max()),
        max_rel_amp_change_period_ka=float(GRID_ALL[m][int(d_amp_rel[m].argmax())]),
        r2_at_that_period_age1=float(r2_1[m][int(d_amp_rel[m].argmax())]),
        max_abs_dphase_deg=float(d_ph[m].max()),
        max_dphase_period_ka=float(GRID_ALL[m][int(d_ph[m].argmax())]))
i = int(d_r2.argmax())
B["broadband_worst_case"] = dict(
    grid_ka=[5.0, 400.0], step_ka=0.5,
    global_max_abs_dR2=float(d_r2[i]), global_max_abs_dR2_period_ka=float(GRID_ALL[i]),
    r2_at_that_period_age1=float(r2_1[i]),
    global_max_rel_amp_change=float(d_amp_rel.max()),
    global_max_rel_amp_period_ka=float(GRID_ALL[int(d_amp_rel.argmax())]),
    r2_at_max_rel_amp_period_age1=float(r2_1[int(d_amp_rel.argmax())]),
    global_max_abs_dphase_deg=float(d_ph.max()),
    global_max_dphase_period_ka=float(GRID_ALL[int(d_ph.argmax())]),
    r2_at_max_dphase_period_age1=float(r2_1[int(d_ph.argmax())]),
    bands=band,
)

# depth-domain spectrum as a contrast (how much of the "period" is age-model).
# restricted to P <= 40 m to keep the design well conditioned; the normal
# equations become singular when the scan period approaches the record span.
GRID_DEPTH = np.arange(0.5, 40.0 + 1e-12, 0.05)
r2_d, _, _, _ = scan(z[CO], y[CO], GRID_DEPTH)
B["depth_domain_best_r2_same_grid"] = dict(
    grid_m=[0.5, 40.0],
    max_r2=float(r2_d.max()),
    at_period_m=float(GRID_DEPTH[int(r2_d.argmax())]),
    note="depth-domain scan is a conditioning contrast only; it is not the "
         "task's age-domain target")

# ----------------------------------------------------------------------------
# PART D -- adversarial alternative coordinates
# ----------------------------------------------------------------------------
span_z = z.max() - z.min()
a_uni_own = (z - z.min()) / (span_z / (a2.max() - a2.min()))
a_uni_match = (z - z.min()) / (span_z / (a1.max() - a1.min()))
a2_translated = a2 + (a1[0] - a2[0])

alt = {}
for name, talt in [("age1_shipped", a1), ("age2_shipped", a2),
                   ("uniform_rate_age2_span", a_uni_own),
                   ("uniform_rate_age1_span", a_uni_match),
                   ("age2_translated_to_age1_top", a2_translated)]:
    f = fit_exact(talt[CO], y[CO])
    alt[name] = dict(period_ka=f["period_ka"], r2_full=f["r2_full"],
                     amp=f["amp"], phase_deg=float(np.degrees(f["phase_rad"])),
                     top_age_ka=float(talt[0]), bottom_age_ka=float(talt[-1]),
                     period_shift_vs_age1_ka=abs(f["period_ka"] - P_REF),
                     period_shift_vs_age1_in_rayleigh=abs(f["period_ka"] - P_REF) / RAY,
                     amp_rel_shift_vs_age1=abs(f["amp"]
                                               - fit_a1["amp"]) / fit_a1["amp"])

# does the age-resampled series itself change?  (whole-record product)
tt = np.linspace(a1.min(), a1.max(), 400)
o1 = np.argsort(a1[CO]); o2 = np.argsort(a2[CO])
s1 = np.interp(tt, a1[CO][o1], y[CO][o1])
s2 = np.interp(tt, a2[CO][o2], y[CO][o2])
r2s1, _, _, _ = scan(tt, s1, GRID_ALL)
r2s2, _, _, _ = scan(tt, s2, GRID_ALL)
B["age_resampled_uniform_400pt_series"] = dict(
    corr_between_the_two_resampled_series=float(np.corrcoef(s1, s2)[0, 1]),
    max_abs_dR2=float(np.abs(r2s1 - r2s2).max()),
    best_period_age1_resampled_ka=float(GRID_ALL[int(r2s1.argmax())]),
    best_period_age2_resampled_ka=float(GRID_ALL[int(r2s2.argmax())]),
)

# ----------------------------------------------------------------------------
# PART E -- lambda scaling of the ACTUAL discrepancy shape
# ----------------------------------------------------------------------------
LAMS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 11.0, 12.0, 20.0, 40.0, 80.0])
lam_out = {}
for lam in LAMS:
    talt = a2 + lam * (a1 - a2)
    f = fit_exact(talt[CO], y[CO])
    lam_out[f"{lam:g}"] = dict(
        lam=float(lam), period_ka=f["period_ka"], r2_full=f["r2_full"],
        amp=f["amp"], phase_deg=float(np.degrees(f["phase_rad"])),
        max_top_discrepancy_ka=float(lam * absdiff[mask_top].max()),
        period_shift_vs_age2_ka=abs(f["period_ka"] - fit_a2["period_ka"]),
        period_shift_vs_age1_ka=abs(f["period_ka"] - P_REF),
        period_shift_vs_age1_in_rayleigh=abs(f["period_ka"] - P_REF) / RAY)
B["lambda_scan"] = dict(
    definition="age(lam)=age2+lam*(age1-age2); lam=0 shipped age2, lam=1 shipped age1",
    grid=lam_out,
    smallest_lam_with_period_shift_gt_rayleigh=next(
        (float(k) for k in sorted(lam_out, key=float)
         if lam_out[k]["period_shift_vs_age1_in_rayleigh"] > 1.0), None),
    shipped_max_top52_discrepancy_ka=float(absdiff[mask_top].max()),
    period_resolution_rayleigh_ka=float(RAY))

# ----------------------------------------------------------------------------
# PART F -- random monotone warps of the top 52 rows
#           (plus radii/evidence-preserving tests of the fast-scan flags)
# ----------------------------------------------------------------------------
top_idx = np.where(mask_top)[0]
shape = (0.55 - z[top_idx]) / 0.55
AMP_GRID = np.array([0.0, 0.5, 1.0, 1.63, 3.0, 6.0, 12.0, 25.0, 50.0, 90.0,
                     150.0, 250.0, 500.0])
N_ITER = 100

det_shift, det_shift_fine = {}, {}
for A_amp in AMP_GRID:
    talt = a2.copy()
    talt[top_idx] = a2[top_idx] + A_amp * shape
    det_shift[float(A_amp)] = abs(fit_fast(talt[CO], y[CO])["period_ka"] - P_REF)
    det_shift_fine[float(A_amp)] = abs(fit_exact(talt[CO], y[CO])["period_ka"] - P_REF)

rnd = {}
for A_amp in AMP_GRID:
    sh = np.empty(N_ITER)
    for it in range(N_ITER):
        sign = 1.0 if RNG.random() < 0.5 else -1.0
        talt = a2.copy()
        talt[top_idx] = a2[top_idx] + sign * A_amp * shape
        sh[it] = abs(fit_fast(talt[CO], y[CO])["period_ka"] - P_REF)
    rnd[float(A_amp)] = dict(
        amp_ka=float(A_amp), median_abs_period_shift_ka=float(np.median(sh)),
        q90_abs_period_shift_ka=float(np.quantile(sh, 0.90)),
        max_abs_period_shift_ka=float(sh.max()),
        median_in_rayleigh=float(np.median(sh) / RAY),
        frac_gt_rayleigh=float((sh > RAY).mean()))
B["top52_random_warp"] = dict(
    n_iter=N_ITER,
    shape="monotone ramp 1 at 0.03 m -> 0 at 0.55 m added to age_2, random sign",
    deterministic_shift_ka=det_shift,
    deterministic_shift_ka_exact_fit=det_shift_fine,
    random_signed=rnd,
    smallest_grid_amp_deterministic_gt_rayleigh=next(
        (float(k) for k in AMP_GRID if det_shift_fine[float(k)] > RAY), None),
    smallest_grid_amp_median_random_gt_rayleigh=next(
        (float(k) for k in AMP_GRID if rnd[float(k)]["median_abs_period_shift_ka"] > RAY),
        None),
    shipped_max_discrepancy_ka=float(absdiff[mask_top].max()))

# ----------------------------------------------------------------------------
# PART G -- serial-correlation-aware null for the dominant peak, per coordinate
# ----------------------------------------------------------------------------
def block_bootstrap_peaks(n_iter=200, block=60):
    m = int(CO.sum())
    nb = int(np.ceil(m / block))
    starts = RNG.integers(0, m - block, size=(n_iter, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_iter, -1)[:, :m]
    obs1, _ = max_r2_over(a1[CO], y[CO], np.arange(40.0, 300.0 + 1e-12, 0.5))
    obs2, _ = max_r2_over(a2[CO], y[CO], np.arange(40.0, 300.0 + 1e-12, 0.5))
    null1 = np.empty(n_iter)
    null2 = np.empty(n_iter)
    diffs = np.empty(n_iter)
    grid = np.arange(40.0, 300.0 + 1e-12, 0.5)
    for it in range(n_iter):
        ii = idx[it]
        null1[it], _ = max_r2_over(a1[CO][ii], y[CO][ii], grid)
        p1 = fit_fast(a1[CO][ii], y[CO][ii])["period_ka"]
        null2[it], _ = max_r2_over(a2[CO][ii], y[CO][ii], grid)
        p2 = fit_fast(a2[CO][ii], y[CO][ii])["period_ka"]
        diffs[it] = abs(p1 - p2)
    return obs1, obs2, null1, null2, diffs


obs1, obs2, null1, null2, diffs = block_bootstrap_peaks(200, 60)
B["block_null_dominant_peak"] = dict(
    n_iter=200, block_rows=60,
    observed_max_r2_age1=float(obs1), observed_max_r2_age2=float(obs2),
    null_max_r2_age1_q95=float(np.quantile(null1, 0.95)),
    null_max_r2_age2_q95=float(np.quantile(null2, 0.95)),
    p_value_age1=float((null1 >= obs1).mean()),
    p_value_age2=float((null2 >= obs2).mean()),
    p_value_abs_difference=float(abs((null1 >= obs1).mean() - (null2 >= obs2).mean())),
    age1_vs_age2_period_diff_median_ka=float(np.median(diffs)),
    age1_vs_age2_period_diff_q95_ka=float(np.quantile(diffs, 0.95)),
    observed_period_diff_ka=B["period_shift_age1_vs_age2_ka"],
    observed_period_diff_percentile=float((diffs < B["period_shift_age1_vs_age2_ka"]).mean()),
)

# ----------------------------------------------------------------------------
# PART H -- other record-wide quantities
# ----------------------------------------------------------------------------
g1 = np.gradient(a1, z)
g2 = np.gradient(a2, z)
lsr1, lsr2 = 1.0 / g1, 1.0 / g2
mx1, mx2 = lsr1.max(), lsr2.max()
w1 = np.where(lsr1 >= mx1 - 1e-12)[0]
w2 = np.where(lsr2 >= mx2 - 1e-12)[0]
dz = np.diff(z)
G = dict(
    max_lsr_age1_m_per_ka=float(mx1), max_lsr_age2_m_per_ka=float(mx2),
    argmax_lsr_idx_age1=int(lsr1.argmax()), argmax_lsr_idx_age2=int(lsr2.argmax()),
    argmax_lsr_row_identical=bool(lsr1.argmax() == lsr2.argmax()),
    max_lsr_rows_identical=bool(np.array_equal(w1, w2)),
    argmax_lsr_depth_m=float(z[lsr1.argmax()]),
    argmax_lsr_age_ka=float(a1[lsr1.argmax()]),
    min_lsr_row_identical=bool(lsr1.argmin() == lsr2.argmin()),
    max_abs_lsr_diff_m_per_ka=float(np.abs(lsr1 - lsr2).max()),
    max_abs_lsr_diff_below_0p55m_m_per_ka=float(np.abs(lsr1 - lsr2)[mask_deep].max()),
    mean_lsr_age1_m_per_ka=float(span_z / (a1.max() - a1.min())),
    mean_lsr_age2_m_per_ka=float(span_z / (a2.max() - a2.min())),
    median_age_step_age1_ka=float(np.median(np.diff(a1[CO]))),
    median_age_step_age2_ka=float(np.median(np.diff(a2[CO]))),
    max_age_step_age1_ka=float(np.diff(a1[CO]).max()),
    max_age_step_age2_ka=float(np.diff(a2[CO]).max()),
    median_depth_step_m=float(np.median(dz)),
)
G["mean_lsr_rel_diff"] = abs(G["mean_lsr_age2_m_per_ka"]
                             - G["mean_lsr_age1_m_per_ka"]) / G["mean_lsr_age1_m_per_ka"]
G["median_age_step_rel_diff"] = abs(G["median_age_step_age2_ka"]
                                    - G["median_age_step_age1_ka"]) / G["median_age_step_age1_ka"]

# ----------------------------------------------------------------------------
# verdict bookkeeping
# ----------------------------------------------------------------------------
tol = dict(period_rayleigh=float(RAY), r2=0.01, amp_rel=0.05, phase_deg=5.0)
moved = {
    "period_gt_rayleigh": bool(B["period_shift_age1_vs_age2_in_rayleigh"] > 1.0),
    "r2_gt_0p01": bool(B["r2_shift_age1_vs_age2"] > tol["r2"]),
    "amp_rel_gt_5pct": bool(B["amp_rel_shift_age1_vs_age2"] > tol["amp_rel"]),
    "phase_gt_5deg": bool(B["phase_shift_age1_vs_age2_deg"] > tol["phase_deg"]),
    "full_corr_gt_0p01": bool(B["full_record_corr_abs_shift"] > 0.01),
    "uniform_coord_period_gt_rayleigh": bool(
        alt["uniform_rate_age2_span"]["period_shift_vs_age1_in_rayleigh"] > 1.0),
    "translated_age2_period_gt_rayleigh": bool(
        alt["age2_translated_to_age1_top"]["period_shift_vs_age1_in_rayleigh"] > 1.0),
    "resampled_series_best_period_moves": bool(
        abs(B["age_resampled_uniform_400pt_series"]["best_period_age1_resampled_ka"]
            - B["age_resampled_uniform_400pt_series"]["best_period_age2_resampled_ka"]) > 1.0),
    "block_null_pvalue_diff_gt_0p05": bool(B["block_null_dominant_peak"]["p_value_abs_difference"] > 0.05),
}

out = dict(
    claim_id="C4",
    part_A_descriptive=A,
    part_B_record_wide_fits=B,
    part_C_adversarial_coordinates=alt,
    part_H_sedimentation_rate=G,
    tolerances=tol,
    per_metric_moves_under_shipped_pair=moved,
    verdict_inputs=dict(
        descriptive_facts_reproduced=bool(
            A["n_rows_absdiff_gt_0"] == 52
            and A["n_rows_absdiff_gt_0p005"] == 52
            and abs(A["diff_rows_depth_min_m"] - 0.03) < 1e-9
            and abs(A["diff_rows_depth_max_m"] - 0.55) < 1e-9
            and abs(A["max_abs_diff_ka"] - 1.63) < 1e-3
            and A["max_abs_diff_deeper_than_0p55m_ka"] < 0.005
            and A["differing_rows_equal_the_set_depth_le_0p55m"]),
        any_record_wide_scientific_metric_moves=bool(any(moved.values())),
        lam_needed_for_rayleigh_period_shift=B["lambda_scan"][
            "smallest_lam_with_period_shift_gt_rayleigh"],
        top52_amp_needed_for_rayleigh_shift_exact=B["top52_random_warp"][
            "smallest_grid_amp_deterministic_gt_rayleigh"],
        top52_amp_needed_for_rayleigh_shift_random_median=B["top52_random_warp"][
            "smallest_grid_amp_median_random_gt_rayleigh"],
        shipped_max_top52_discrepancy_ka=float(absdiff[mask_top].max()),
        period_resolution_rayleigh_ka=float(RAY),
    ),
)
os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
with open(OUTJ, "w") as fh:
    json.dump(out, fh, indent=2, default=float)

# ----------------------------------------------------------------------------
def show(title, d, skip=()):
    print("=" * 78)
    print(title)
    print("=" * 78)
    for k, v in d.items():
        if k in skip:
            continue
        print(f"  {k:52s} = {v}")


show("PART A  descriptive facts", A)
print()
show("PART B  record-wide fits", B,
     skip=("lambda_scan", "top52_random_warp", "block_null_dominant_peak",
           "broadband_worst_case"))
print()
print("  broadband worst case:", json.dumps(B["broadband_worst_case"], indent=4))
print()
print("  lambda scan (age = age2 + lam*(age1-age2)):")
print(f"  {'lam':>6s} {'max top disc ka':>16s} {'period ka':>11s} {'r2':>9s} "
      f"{'dP vs age1 ka':>14s} {'/Rayleigh':>10s}")
for k in sorted(lam_out, key=float):
    v = lam_out[k]
    print(f"  {v['lam']:6.2f} {v['max_top_discrepancy_ka']:16.3f} {v['period_ka']:11.4f} "
          f"{v['r2_full']:9.4f} {v['period_shift_vs_age1_ka']:14.4f} "
          f"{v['period_shift_vs_age1_in_rayleigh']:10.4f}")
print(f"  smallest lam with period shift > 1 Rayleigh: "
      f"{B['lambda_scan']['smallest_lam_with_period_shift_gt_rayleigh']}")
print()
print("  top-52 monotone warp:")
print(f"  {'amp ka':>8s} {'det(exact)':>11s} {'med rand':>10s} {'q90':>10s} "
      f"{'max':>10s} {'med/Ray':>9s} {'frac>Ray':>9s}")
for k in AMP_GRID:
    r = rnd[float(k)]
    print(f"  {float(k):8.2f} {det_shift_fine[float(k)]:11.4f} "
          f"{r['median_abs_period_shift_ka']:10.4f} {r['q90_abs_period_shift_ka']:10.4f} "
          f"{r['max_abs_period_shift_ka']:10.4f} {r['median_in_rayleigh']:9.4f} "
          f"{r['frac_gt_rayleigh']:9.3f}")
print(f"  smallest amp with deterministic (exact-fit) shift > Rayleigh: "
      f"{B['top52_random_warp']['smallest_grid_amp_deterministic_gt_rayleigh']}")
print(f"  smallest amp with median random shift > Rayleigh: "
      f"{B['top52_random_warp']['smallest_grid_amp_median_random_gt_rayleigh']}")
print(f"  shipped max top-52 discrepancy: {absdiff[mask_top].max():.3f} ka")
print()
print("  block-bootstrap null for the dominant peak:", json.dumps(
    B["block_null_dominant_peak"], indent=4))
print()
print("  adversarial coordinates:")
for k, v in alt.items():
    print(f"    {k:34s} P={v['period_ka']:9.4f} R2={v['r2_full']:.4f} "
          f"amp={v['amp']:.4f} dP={v['period_shift_vs_age1_ka']:9.4f} ka "
          f"({v['period_shift_vs_age1_in_rayleigh']:6.3f} Rayleigh)")
print()
show("PART H  sedimentation rate / axis spacing", G)
print()
show("PER-METRIC MOVES UNDER SHIPPED PAIR", moved)
print()
show("VERDICT INPUTS", out["verdict_inputs"])
print()
print(f"wrote {OUTJ}")
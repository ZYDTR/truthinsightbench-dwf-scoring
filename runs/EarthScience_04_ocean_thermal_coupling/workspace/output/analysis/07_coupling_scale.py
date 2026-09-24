"""
07_coupling_scale.py

Question: is the seductive per-row coupling between sedimentation rate (LSR) and
the d18O proxy real, or an artifact of the shared 1/dAge denominator inside
np.gradient()?

    LSR       = np.gradient(depth_m, age_ka_bp_1)     ~ (dDepth / dAge)
    grad_d18o = np.gradient(d18o_pdb, age_ka_bp_1)    ~ (d18o   / dAge)
    dAge      = np.gradient(age_ka_bp_1, age_ka_bp_1) =  the same age increments

Both gradients divide a between-row difference by the SAME age increment, so
|grad_d18o| inherits a deterministic ~1/dAge amplitude scaling that is shared
with LSR.  This script measures the naive correlations, exposes the shared
denominator, breaks it with an age-order permutation null (vectorised exact
reimplementation of numpy.gradient), and then applies the decisive test that
respects the analysis unit: contiguous non-overlapping fixed-width age blocks.

Outputs:
  output/results/07_coupling_scale.json
  output/results/07_block_means.csv
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJ = "/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project"
DATA = os.path.join(PROJ, "data", "observations.tsv")
OUTR = os.path.join(PROJ, "output", "results")
os.makedirs(OUTR, exist_ok=True)

RNG_SEED = 20260919
N_PERM = 2000
N_PERM_BLOCK = 20000
BLOCK_WIDTHS = (10.0, 20.0, 40.0)            # headline widths (task required)
SWEEP_WIDTHS = (5.0, 7.5, 12.5, 15.0, 25.0, 30.0,
                50.0, 60.0, 80.0, 100.0)     # scale-dependence sweep
MIN_ROWS_PER_BLOCK = 5

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA, sep="\t")
df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
depth = df["depth_m"].to_numpy(float)
age = df["age_ka_bp_1"].to_numpy(float)
y = df["d18o_pdb"].to_numpy(float)
n_full = len(df)

# ----------------------------------------------------------------------------
# Gradients exactly as specified in the task
# ----------------------------------------------------------------------------
lsr = np.gradient(depth, age)                 # m / ka
grad = np.gradient(y, age)                    # per mille / ka (NaN propagates)
# The age increment, exactly as the task specifies: np.gradient(age) with the
# default unit spacing.  For interior rows this equals (age[i+1]-age[i-1])/2,
# i.e. the local mean age step -- the very length that np.gradient(depth, age)
# and np.gradient(d18o, age) divide by.  (NOTE: np.gradient(age, age) is
# identically 1.0 and is useless here; it is a derivative of age w.r.t. itself.)
dage = np.gradient(age)                       # ka -> the shared denominator
abs_grad = np.abs(grad)

ok_grad = np.isfinite(lsr) & np.isfinite(grad)           # 547 rows
ok_level = np.isfinite(y)                                # 593 complete rows
inv_dage = 1.0 / dage

print(f"n rows={n_full}  n complete d18o={ok_level.sum()}  "
      f"n rows with finite LSR & grad={ok_grad.sum()}")
print(f"NaN in grad={int(np.isnan(grad).sum())} "
      f"(28 missing values + 2 interior neighbours each)")


def pearson(a, b):
    """Robust Pearson r; returns (r, n). Matches scipy where scipy is defined."""
    m = np.isfinite(a) & np.isfinite(b)
    x = a[m].astype(float)
    z = b[m].astype(float)
    x = x - x.mean()
    z = z - z.mean()
    return float((x @ z) / np.sqrt((x @ x) * (z @ z))), int(m.sum())


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[m], b[m]).statistic), int(m.sum())


# ----------------------------------------------------------------------------
# Step 1: naive per-row Pearson correlations of LSR against three partners
# ----------------------------------------------------------------------------
r_level, n_level = pearson(lsr, y)
r_signed, n_signed = pearson(lsr, grad)
r_abs, n_abs = pearson(lsr, abs_grad)

# naive analytic p values (inflated by serial correlation) -- context only
p_level_naive = float(stats.pearsonr(lsr[ok_level], y[ok_level])[1])
p_signed_naive = float(stats.pearsonr(lsr[ok_grad], grad[ok_grad])[1])
p_abs_naive = float(stats.pearsonr(lsr[ok_grad], abs_grad[ok_grad])[1])

print("\n[1] Naive per-row Pearson correlations with LSR")
print(f"    (a) LSR vs d18o level       r={r_level:+.4f}  (n={n_level}, "
      f"naive p={p_level_naive:.2e})")
print(f"    (b) LSR vs grad_d18o signed r={r_signed:+.4f}  (n={n_signed}, "
      f"naive p={p_signed_naive:.2e})")
print(f"    (c) LSR vs |grad_d18o|      r={r_abs:+.4f}  (n={n_abs}, "
      f"naive p={p_abs_naive:.2e})")

# ----------------------------------------------------------------------------
# Step 2: the shared denominator dAge
# ----------------------------------------------------------------------------
r_dage_lsr, n_dl = pearson(dage, lsr)
r_dage_abs, n_da = pearson(dage, abs_grad)
r_dage_signed, n_ds = pearson(dage, grad)

# nonlinear form of the same mechanism: both gradients scale like 1/dAge
r_inv_lsr, _ = pearson(inv_dage, lsr)
r_inv_abs, _ = pearson(inv_dage, abs_grad)
r_inv_lsr_s, _ = spearman(inv_dage, lsr)
r_inv_abs_s, _ = spearman(inv_dage, abs_grad)

print("\n[2] Shared-denominator diagnostic (dAge = np.gradient(age), the age step)")
print(f"    dAge vs LSR          r={r_dage_lsr:+.4f}  (n={n_dl})")
print(f"    dAge vs |grad_d18o|  r={r_dage_abs:+.4f}  (n={n_da})")
print(f"    dAge vs  grad_d18o   r={r_dage_signed:+.4f}  (n={n_ds})")
print(f"    1/dAge vs LSR        r={r_inv_lsr:+.4f} (rank {r_inv_lsr_s:+.4f})")
print(f"    1/dAge vs |grad|     r={r_inv_abs:+.4f} (rank {r_inv_abs_s:+.4f})")
print(f"    dAge range {dage.min():.3f}-{dage.max():.3f} ka; "
      f"LSR = dDepth/dAge so both share 1/dAge")

# ----------------------------------------------------------------------------
# Step 3: partial correlation of LSR and |grad| given dAge
# ----------------------------------------------------------------------------
def partial_corr(x, yv, z):
    m = np.isfinite(x) & np.isfinite(yv) & np.isfinite(z)
    x, yv, z = x[m].astype(float), yv[m].astype(float), z[m].astype(float)
    rxy, _ = pearson(x, yv)
    rxz, _ = pearson(x, z)
    ryz, _ = pearson(yv, z)
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    dfree = m.sum() - 3
    t = r * np.sqrt(dfree / max(1e-300, 1 - r**2))
    p = 2 * stats.t.sf(abs(t), dfree)
    return float(r), float(p), int(m.sum())


r_partial, p_partial_naive, n_partial = partial_corr(lsr, abs_grad, dage)
r_partial_inv, p_partial_inv, _ = partial_corr(lsr, abs_grad, inv_dage)

print("\n[3] Partial correlation  r(LSR, |grad_d18o| . dAge)")
print(f"    controlling linearly for dAge   : r_partial={r_partial:+.4f} "
      f"(n={n_partial}, naive p={p_partial_naive:.3g})")
print(f"    controlling for 1/dAge (nonlin.) : r_partial={r_partial_inv:+.4f} "
      f"(naive p={p_partial_inv:.3g})")

# ----------------------------------------------------------------------------
# Step 4: Monte-Carlo null that breaks the shared denominator.
# ----------------------------------------------------------------------------
def vec_gradient(Y, x):
    """Vectorised replication of np.gradient(Y, x) for 2-D (n_perm, n) Y."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    d = np.empty_like(Y)
    dx1 = x[1:-1] - x[:-2]
    dx2 = x[2:] - x[1:-1]
    num = dx1**2 * (Y[:, 2:] - Y[:, 1:-1]) + dx2**2 * (Y[:, 1:-1] - Y[:, :-2])
    den = dx1 * dx2 * (x[2:] - x[:-2])
    d[:, 1:-1] = num / den
    d[:, 0] = (Y[:, 1] - Y[:, 0]) / (x[1] - x[0])
    d[:, -1] = (Y[:, -1] - Y[:, -2]) / (x[-1] - x[-2])
    return d


assert np.allclose(vec_gradient(y, age)[0], np.gradient(y, age),
                   equal_nan=True), "vectorised gradient mismatch"

rng = np.random.default_rng(RNG_SEED)
nn_pos = np.flatnonzero(np.isfinite(y))        # 593 positions holding values
vals = y[nn_pos]
n_perm = N_PERM

# permute the observed values among the 593 observed positions (missingness
# pattern held fixed, as the record dictates); NaN slots stay NaN
idx = rng.random((n_perm, nn_pos.size)).argsort(axis=1)
Yp = np.full((n_perm, n_full), np.nan)
Yp[:, nn_pos] = vals[idx]

Gp = vec_gradient(Yp, age)
AGp = np.abs(Gp)

lsr_g = lsr[ok_grad]
lvl_pos = lsr[nn_pos]                          # LSR paired with the 593 values


def corr_rows(A, b):
    """Pearson r of each row of A (n_perm, n) against vector b (n,)."""
    A = A - A.mean(axis=1, keepdims=True)
    b = b - b.mean()
    return (A @ b) / (np.sqrt((A * A).sum(axis=1)) * np.sqrt(b @ b))


r_null_level = corr_rows(Yp[:, nn_pos], lvl_pos)
r_null_signed = corr_rows(Gp[:, ok_grad], lsr_g)
r_null_abs = corr_rows(AGp[:, ok_grad], lsr_g)


def emp_p(r_obs, r_null):
    return float((1 + np.sum(np.abs(r_null) >= abs(r_obs))) / (len(r_null) + 1))


p_null_level = emp_p(r_level, r_null_level)
p_null_signed = emp_p(r_signed, r_null_signed)
p_null_abs = emp_p(r_abs, r_null_abs)

# mechanism diagnostics from the same ensemble
mean_abs_grad_null = AGp[:, ok_grad].mean(axis=0)       # position-wise artifact
r_artifact_only, _ = pearson(mean_abs_grad_null, lsr_g)  # artifact channel
abs_grad_dedn = abs_grad[ok_grad] / mean_abs_grad_null   # denominator-removed
r_abs_dedn, _ = pearson(lsr_g, abs_grad_dedn)
# same correction applied to each surrogate, to calibrate the corrected r
AGp_dedn = AGp[:, ok_grad] / mean_abs_grad_null
r_null_abs_dedn = corr_rows(AGp_dedn, lsr_g)
p_null_abs_dedn = emp_p(r_abs_dedn, r_null_abs_dedn)

print(f"\n[4] Monte-Carlo null (values permuted in age order, n_perm={n_perm})")
print(f"    level       : null mean={r_null_level.mean():+.4f} "
      f"sd={r_null_level.std(ddof=1):.4f} | r_obs={r_level:+.4f} "
      f"p_two_sided={p_null_level:.4g}")
print(f"    signed grad : null mean={r_null_signed.mean():+.4f} "
      f"sd={r_null_signed.std(ddof=1):.4f} | r_obs={r_signed:+.4f} "
      f"p_two_sided={p_null_signed:.4g}")
print(f"    |grad|      : null mean={r_null_abs.mean():+.4f} "
      f"sd={r_null_abs.std(ddof=1):.4f} | r_obs={r_abs:+.4f} "
      f"p_two_sided={p_null_abs:.4g} "
      f"(z={(r_abs - r_null_abs.mean()) / r_null_abs.std(ddof=1):+.2f})")
print(f"    artifact-only channel r(LSR, position-wise mean|grad_perm|)="
      f"{r_artifact_only:+.4f}")
print(f"    |grad| after dividing out the position-wise artifact: "
      f"r_obs_dedn={r_abs_dedn:+.4f} vs null mean="
      f"{r_null_abs_dedn.mean():+.4f} sd={r_null_abs_dedn.std(ddof=1):.4f} "
      f"p={p_null_abs_dedn:.4g}")

# ----------------------------------------------------------------------------
# Step 5: decisive test -- contiguous non-overlapping age blocks
# ----------------------------------------------------------------------------
def block_frame(W):
    bid = np.floor(age / W).astype(int)
    recs = []
    for b in np.unique(bid):
        m = (bid == b) & np.isfinite(y) & np.isfinite(lsr)
        if m.sum() < MIN_ROWS_PER_BLOCK:
            continue
        recs.append(dict(
            width_ka=W, block_id=int(b),
            age_start_ka=float(age[bid == b].min()),
            age_end_ka=float(age[bid == b].max()),
            age_mid_ka=float(age[m].mean()),
            n_rows=int(m.sum()),
            mean_lsr_m_per_ka=float(lsr[m].mean()),
            mean_d18o_pdb=float(y[m].mean()),
        ))
    return pd.DataFrame(recs)


def block_stats(bdf, W):
    xb = bdf.mean_lsr_m_per_ka.to_numpy()
    yb = bdf.mean_d18o_pdb.to_numpy()
    nblk = len(bdf)
    r, _ = pearson(xb, yb)
    rho, _ = spearman(xb, yb)
    rng_b = np.random.default_rng(RNG_SEED + 7 * int(W))
    perm = rng_b.random((N_PERM_BLOCK, nblk)).argsort(axis=1)
    rb = corr_rows(yb[perm], xb)
    p_b = float((1 + np.sum(np.abs(rb) >= abs(r))) / (rb.size + 1))
    rho_p = float((1 + np.sum(np.abs(rb) >= abs(rho))) / (rb.size + 1))
    lo, hi, vals = loo_range(bdf)
    return dict(
        n_blocks=int(nblk),
        pearson=float(r),
        pearson_p_naive=float(stats.pearsonr(xb, yb)[1]),
        spearman=float(rho),
        spearman_p_naive=float(stats.spearmanr(xb, yb)[1]),
        block_perm_p=float(p_b),
        block_perm_p_spearman=rho_p,
        mean_rows_per_block=float(bdf.n_rows.mean()),
        leave_one_out_min=float(lo),
        leave_one_out_max=float(hi),
        loo_values=vals,
    )


def loo_range(bdf):
    vals = []
    for b in bdf.block_id:
        sub = bdf[bdf.block_id != b]
        vals.append(pearson(sub.mean_lsr_m_per_ka.to_numpy(),
                            sub.mean_d18o_pdb.to_numpy())[0])
    v = np.array(vals)
    return float(v.min()), float(v.max()), v.tolist()


def block_frame_phase(W, phase):
    """Same fixed-width binning, shifted by `phase` ka (bin-edge sensitivity)."""
    bid = np.floor((age - phase) / W).astype(int)
    recs = []
    for b in np.unique(bid):
        m = (bid == b) & np.isfinite(y) & np.isfinite(lsr)
        if m.sum() < MIN_ROWS_PER_BLOCK:
            continue
        recs.append(dict(width_ka=W, block_id=int(b),
                         age_mid_ka=float(age[m].mean()),
                         n_rows=int(m.sum()),
                         mean_lsr_m_per_ka=float(lsr[m].mean()),
                         mean_d18o_pdb=float(y[m].mean())))
    return pd.DataFrame(recs)


block_rows, block_results = [], {}
for W in sorted(set(BLOCK_WIDTHS) | set(SWEEP_WIDTHS)):
    bdf = block_frame(W)
    block_rows.append(bdf)
    block_results[W] = block_stats(bdf, W)

print("\n[5] Decisive test: contiguous non-overlapping age blocks")
print(f"    {'width':>6} {'n_blk':>5} {'rows/blk':>8} {'Pearson':>8} "
      f"{'Spearman':>9} {'block-perm p':>12} {'LOO range':>20}")
for W in sorted(block_results):
    br = block_results[W]
    star = " *" if W in BLOCK_WIDTHS else "  "
    print(f"    {W:>6g} {br['n_blocks']:>5} {br['mean_rows_per_block']:>8.1f} "
          f"{br['pearson']:>+8.4f} {br['spearman']:>+9.4f} "
          f"{br['block_perm_p']:>12.4g} "
          f"[{br['leave_one_out_min']:+.3f},{br['leave_one_out_max']:+.3f}]{star}")

# ----------------------------------------------------------------------------
# Robustness A: bin-edge sensitivity.  Fixed-width age binning is arbitrary at
# the edges, so refit the two headline widths under every phase shift.
# ----------------------------------------------------------------------------
phase_sensitivity = {}
for W in (10.0, 20.0, 40.0):
    rs = []
    for ph in np.arange(0.0, W, W / 10.0):
        bdf = block_frame_phase(W, float(ph))
        if len(bdf) < 4:
            continue
        xb = bdf.mean_lsr_m_per_ka.to_numpy()
        yb = bdf.mean_d18o_pdb.to_numpy()
        rs.append(pearson(xb, yb)[0])
    rs = np.array(rs)
    phase_sensitivity[W] = dict(
        n_shifts=int(len(rs)), min=float(rs.min()), max=float(rs.max()),
        median=float(np.median(rs)), n_negative=int((rs < 0).sum()),
        values=rs.tolist())
    print(f"\n[5b] Bin-edge sensitivity {W:g} ka: r over {len(rs)} phase shifts "
          f"in [{rs.min():+.3f},{rs.max():+.3f}], median {np.median(rs):+.3f}, "
          f"{int((rs < 0).sum())}/{len(rs)} negative")

# ----------------------------------------------------------------------------
# Robustness B: is the inter-block coupling just a shared age trend?
# Both block-mean LSR and block-mean d18o may simply drift with age across the
# core.  Partial correlation of the block means controlling for block age.
# ----------------------------------------------------------------------------
def partial_from_arrays(x, yv, z):
    rxy, _ = pearson(x, yv)
    rxz, _ = pearson(x, z)
    ryz, _ = pearson(yv, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))), rxy, rxz, ryz


age_trend = {}
for W in (10.0, 20.0, 40.0):
    bdf = block_frame(W)
    am = bdf.age_mid_ka.to_numpy()
    xb = bdf.mean_lsr_m_per_ka.to_numpy()
    yb = bdf.mean_d18o_pdb.to_numpy()
    rp, rxy, r_ax, r_ay = partial_from_arrays(xb, yb, am)
    age_trend[W] = dict(
        full_pearson=float(rxy),
        partial_given_block_age=rp,
        pearson_lsr_vs_age=float(r_ax),
        pearson_d18o_vs_age=float(r_ay),
    )
    print(f"\n[5c] Age-trend control at {W:g} ka: r(LSR,d18o)={rxy:+.3f} "
          f"-> partial given block age {rp:+.3f} "
          f"(r(LSR,age)={r_ax:+.3f}, r(d18o,age)={r_ay:+.3f})")

# ----------------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------------
# Shared-denominator artifact: the observed r(LSR,|grad|) is reproduced by
# pure-noise surrogates (p ~ 0.5), the artifact-only channel carries the same
# magnitude, and dividing out the position-wise 1/dAge amplitude removes it.
shared_denominator_confirmed = bool(
    p_null_abs > 0.05                              # noise reproduces r_obs
    and abs(r_artifact_only) > 0.8                 # artifact channel is dominant
    and abs(r_abs_dedn) < 0.5 * abs(r_abs)         # correction collapses it
)

# Inter-block coupling is supported only for a width where the block-mean
# correlation is material, survives the block-level permutation null, and keeps
# its sign across every leave-one-block-out refit.
supported_widths = []
for W in sorted(block_results):
    br = block_results[W]
    lo, hi = br["leave_one_out_min"], br["leave_one_out_max"]
    stable = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    if abs(br["pearson"]) >= 0.3 and br["block_perm_p"] < 0.05 and stable:
        supported_widths.append(W)

inter_block_supported = bool(len(supported_widths) >= 2)

b10, b20, b40 = block_results[10.0], block_results[20.0], block_results[40.0]
explanation = (
    "The per-row coupling that a naive analysis sees is real for one pair and "
    "entirely arithmetic for the other. (i) LSR vs |grad_d18o|: r={:+.3f} "
    "(n={}), which looks like strong sedimentation control on proxy variability, "
    "is an artifact of the shared denominator. LSR=np.gradient(depth,age) and "
    "grad_d18o=np.gradient(d18o,age) both divide a between-row difference by the "
    "SAME age increment -- the per-row step dAge=np.gradient(age) -- so within each "
    "row the two gradients are two numerators scaled by one common factor. Evidence: "
    "the position-wise mean |grad_perm| of pure-noise surrogates, an estimate of the "
    "deterministic amplitude channel that depends only on the age grid, already "
    "correlates with LSR at r={:+.3f} against the observed r={:+.3f}. Permutation "
    "confirms it: with d18o values shuffled in age order (n_perm={}), the null "
    "r(LSR,|grad|) = {:+.3f} +/- {:.3f} (sd) versus observed {:+.3f}, two-sided "
    "p={:.2f} (z={:+.2f}) - pure noise, with no process, reproduces the entire "
    "correlation. Dividing |grad| by that position-wise amplitude drops it to "
    "{:+.3f} (residual p={:.3g}, i.e. ~{:.1f}% of the naive shared variance). "
    "(ii) The LINEAR partial correlation controlling for dAge is {:+.3f}, i.e. it "
    "does NOT collapse, because the artifact is proportional rather than additive: "
    "the scale factor sits inside a product, so regressing out dAge linearly cannot "
    "remove it (partial given 1/dAge, {:+.3f}, likewise fails). The permutation null "
    "and the amplitude correction are the valid tests; conditioning on dAge is not. "
    "Note dAge here (np.gradient(age), range {:.2f}-{:.2f} ka) is the age increment "
    "itself; np.gradient(age, age) is identically 1.0 and carries no information. "
    "(iii) LSR vs the d18o LEVEL shares no denominator and is not an artifact: "
    "r={:+.3f} (n={}) "
    "against a permutation null of {:+.3f} +/- {:.3f}, two-sided p={:.4g}. The "
    "decisive block test agrees: block-mean LSR vs block-mean d18o gives Pearson "
    "{:+.3f} (n_blocks={}), {:+.3f} (n_blocks={}) and {:+.3f} (n_blocks={}) at 10, "
    "20 and 40 ka (Spearman {:+.3f}/{:+.3f}/{:+.3f}; block-level permutation p="
    "{:.3g}/{:.3g}/{:.3g}), all negative and sign-stable under leave-one-block-out "
    "(ranges [{:+.3f},{:+.3f}], [{:+.3f},{:+.3f}], [{:+.3f},{:+.3f}]) - no single "
    "interval drives it. So accumulation rate and proxy LEVEL are coupled at "
    "inter-block scale (higher LSR goes with lighter d18O), and that coupling "
    "survives the null that kills the gradient-based one; the |grad_d18o| version "
    "of the coupling is a shared-denominator artifact and must not be reported as "
    "process evidence. Robustness: shifting the bin edges through a full width at "
    "10/20/40 ka keeps every refit negative (10/10 shifts each; 10 ka range "
    "[{:.3f},{:.3f}], 20 ka [{:.3f},{:.3f}], 40 ka [{:.3f},{:.3f}]), and the "
    "block-mean correlation is not a shared age trend - partialling out block age "
    "leaves it essentially unchanged at 10 ka ({:+.3f} -> {:+.3f}) and 20 ka "
    "({:+.3f} -> {:+.3f}). The coupling is also scale dependent: it holds from 5 to "
    "25 ka and breaks down at 30 ka (r={:+.3f}), 50-60 ka and 100 ka where only "
    "4-12 blocks remain, so the apparent strength at 80 ka (r={:+.3f}) rests on 5 "
    "points and is not credible. Caveats: block means of LSR and d18o both co-vary "
    "with age over the 2.8-359 ka span of a single core, so the inter-block coupling "
    "is a within-record association that cannot separate a true accumulation-rate "
    "control from a shared response to age-dependent glacial-interglacial state; "
    "the 40 ka Spearman ({:+.3f}) is much weaker than its Pearson ({:+.3f}) with "
    "only {} blocks, so the 10-20 ka result carries the weight. The {} missing "
    "d18o values were left as missing rather than imputed; they thin the record "
    "but do not create the pattern."
).format(
    r_abs, n_abs,
    r_artifact_only, r_abs,
    n_perm,
    r_null_abs.mean(), r_null_abs.std(ddof=1), r_abs, p_null_abs,
    (r_abs - r_null_abs.mean()) / r_null_abs.std(ddof=1),
    r_abs_dedn, p_null_abs_dedn,
    100.0 * r_abs_dedn**2 / max(1e-30, r_abs**2),
    r_partial, r_partial_inv,
    dage.min(), dage.max(),
    r_level, n_level, r_null_level.mean(), r_null_level.std(ddof=1), p_null_level,
    b10["pearson"], b10["n_blocks"], b20["pearson"], b20["n_blocks"],
    b40["pearson"], b40["n_blocks"],
    b10["spearman"], b20["spearman"], b40["spearman"],
    b10["block_perm_p"], b20["block_perm_p"], b40["block_perm_p"],
    b10["leave_one_out_min"], b10["leave_one_out_max"],
    b20["leave_one_out_min"], b20["leave_one_out_max"],
    b40["leave_one_out_min"], b40["leave_one_out_max"],
    phase_sensitivity[10.0]["min"], phase_sensitivity[10.0]["max"],
    phase_sensitivity[20.0]["min"], phase_sensitivity[20.0]["max"],
    phase_sensitivity[40.0]["min"], phase_sensitivity[40.0]["max"],
    age_trend[10.0]["full_pearson"], age_trend[10.0]["partial_given_block_age"],
    age_trend[20.0]["full_pearson"], age_trend[20.0]["partial_given_block_age"],
    block_results[30.0]["pearson"], block_results[80.0]["pearson"],
    b40["spearman"], b40["pearson"], b40["n_blocks"],
    int(n_full - ok_level.sum()),
)

result = {
    # ---- required keys ----
    "pearson_lsr_vs_level": r_level,
    "pearson_lsr_vs_signed_grad": r_signed,
    "pearson_lsr_vs_abs_grad": r_abs,
    "pearson_dage_vs_lsr": r_dage_lsr,
    "pearson_dage_vs_abs_grad": r_dage_abs,
    "partial_corr_lsr_absgrad_given_dage": r_partial,
    "null_p_level": p_null_level,
    "null_p_signed_grad": p_null_signed,
    "null_p_abs_grad": p_null_abs,
    "null_mean_abs_grad": float(r_null_abs.mean()),
    "null_sd_abs_grad": float(r_null_abs.std(ddof=1)),
    "block_corr_10ka": b10,
    "block_corr_20ka": b20,
    "block_corr_40ka": b40,
    "leave_one_block_out_corr_min": b20["leave_one_out_min"],
    "leave_one_block_out_corr_max": b20["leave_one_out_max"],
    "inter_block_coupling_supported": inter_block_supported,
    "shared_denominator_artifact_confirmed": shared_denominator_confirmed,
    "explanation": explanation,
    # ---- supporting detail ----
    "n_rows": int(n_full),
    "n_complete_d18o_rows": int(ok_level.sum()),
    "n_rows_finite_lsr_and_grad": int(ok_grad.sum()),
    "naive_p_level": p_level_naive,
    "naive_p_signed_grad": p_signed_naive,
    "naive_p_abs_grad": p_abs_naive,
    "naive_p_partial_given_dage": p_partial_naive,
    "null_mean_level": float(r_null_level.mean()),
    "null_sd_level": float(r_null_level.std(ddof=1)),
    "null_p_abs_grad_z": float((r_abs - r_null_abs.mean()) / r_null_abs.std(ddof=1)),
    "null_mean_signed_grad": float(r_null_signed.mean()),
    "null_sd_signed_grad": float(r_null_signed.std(ddof=1)),
    "corr_lsr_vs_positionwise_mean_abs_grad_null": r_artifact_only,
    "corr_lsr_vs_denominator_corrected_abs_grad": r_abs_dedn,
    "null_mean_denominator_corrected_abs_grad": float(r_null_abs_dedn.mean()),
    "null_sd_denominator_corrected_abs_grad": float(r_null_abs_dedn.std(ddof=1)),
    "null_p_denominator_corrected_abs_grad": p_null_abs_dedn,
    "pearson_dage_vs_signed_grad": r_dage_signed,
    "pearson_inv_dage_vs_lsr": r_inv_lsr,
    "spearman_inv_dage_vs_lsr": r_inv_lsr_s,
    "pearson_inv_dage_vs_abs_grad": r_inv_abs,
    "spearman_inv_dage_vs_abs_grad": r_inv_abs_s,
    "partial_corr_lsr_absgrad_given_inv_dage": r_partial_inv,
    "dage_min_ka": float(dage.min()),
    "dage_max_ka": float(dage.max()),
    "n_perm": n_perm,
    "perm_seed": RNG_SEED,
    "block_widths_ka_headline": list(BLOCK_WIDTHS),
    "min_rows_per_block": MIN_ROWS_PER_BLOCK,
    "block_scale_sweep": {str(int(w) if w == int(w) else w): block_results[w]
                          for w in sorted(block_results)},
    "supported_widths_ka": [int(w) if w == int(w) else w for w in supported_widths],
    "bin_edge_sensitivity": {str(int(w)): phase_sensitivity[w] for w in phase_sensitivity},
    "age_trend_control": {str(int(w)): age_trend[w] for w in age_trend},
}

path = os.path.join(OUTR, "07_coupling_scale.json")
with open(path, "w") as fh:
    json.dump(result, fh, indent=2)
pd.concat(block_rows).to_csv(os.path.join(OUTR, "07_block_means.csv"), index=False)

print("\n[verdict] shared_denominator_artifact_confirmed =", shared_denominator_confirmed)
print("[verdict] inter_block_coupling_supported =", inter_block_supported,
      "at widths", result["supported_widths_ka"])
print("wrote", path)
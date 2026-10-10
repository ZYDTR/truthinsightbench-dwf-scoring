#!/usr/bin/env python3
"""Phase 3: per-trajectory scaling, regime patterns, and model comparison.
Pure numpy (scipy/sklearn not available)."""
import os
import json
import numpy as np

BASE = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project"
DATA = os.path.join(BASE, "data", "anonymous_trajectories.npz")
OUT = os.path.join(BASE, "output")
RES = os.path.join(BASE, "output", "results")

os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d = np.load(DATA)
X = d["trajectories"]            # (240,128)
T = X.shape[1]                  # 128
Ntraj = X.shape[0]              # 240
Ninc = T - 1                    # 127

# change point labels
cp_path = os.path.join(RES, "change_points.csv")
rows = open(cp_path).read().splitlines()
traj_ids = []
cp_steps = []
labels = []
for r in rows[1:]:
    p = r.split(",")
    traj_ids.append(int(p[0]))
    cp_steps.append(float(p[1]))
    labels.append(p[6])
traj_ids = np.array(traj_ids)
cp_steps = np.array(cp_steps)
labels = np.array(labels)
nonstat_ids = np.array([t for t, l in zip(traj_ids, labels) if l == "nonstationary"])
nonstat_set = set(nonstat_ids.tolist())
n_nonstat = len(nonstat_ids)
n_stat = Ntraj - n_nonstat
print("n_nonstationary =", n_nonstat, " n_stationary =", n_stat)

# ---------------------------------------------------------------------------
# TASK 1: per-trajectory scaling
# ---------------------------------------------------------------------------
hs = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
M = np.zeros((Ntraj, len(hs)))
for j, h in enumerate(hs):
    hi = int(h)
    inc = X[:, hi:] - X[:, :T - hi]          # shape (240, 128-h)
    M[:, j] = np.mean(inc * inc, axis=1)
M1 = M[:, 0]
M8 = M[:, 3]
M16 = M[:, 4]
M32 = M[:, 5]
M64 = M[:, 6]

# log-log slope per trajectory
logh = np.log(hs)
logM = np.log(M + 1e-12)
slopes = np.zeros(Ntraj)
for i in range(Ntraj):
    # polyfit degree 1
    coeffs = np.polyfit(logh, logM[i], 1)
    slopes[i] = coeffs[0]

active_05 = (M1 >= 0.05).astype(int)
active_01 = (M1 >= 0.01).astype(int)
active_010 = (M1 >= 0.1).astype(int)
count_05 = int(active_05.sum())
count_01 = int(active_01.sum())
count_010 = int(active_010.sum())

scaling = np.column_stack([
    traj_ids, M1, M8, M16, M32, M64, slopes, active_05
])
np.savetxt(os.path.join(RES, "scaling.csv"), scaling,
           delimiter=",", header="traj_id,M1,M8,M16,M32,M64,slope,active_flag",
           comments="", fmt="%.10g")
print("scaling.csv written")

scaling_stats = {
    "slope_mean": float(np.mean(slopes)),
    "slope_median": float(np.median(slopes)),
    "slope_min": float(np.min(slopes)),
    "slope_max": float(np.max(slopes)),
    "count_0.05": count_05,
    "count_0.01": count_01,
    "count_0.1": count_010,
}
print("scaling stats:", scaling_stats)

# ---------------------------------------------------------------------------
# TASK 2a: recurring regime patterns (binary H/L)
# ---------------------------------------------------------------------------
pv = np.genfromtxt(os.path.join(RES, "profile_blockvar.csv"), delimiter=",", skip_header=1)
# pv shape (240, 9): first col is traj_id, cols 1: are the 8 block variances
B = pv[:, 1:]                              # (240, 8) block variances
maxv = np.max(B, axis=1, keepdims=True)
# avoid divide-by-zero for constant-zero rows
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = B / np.where(maxv > 0, maxv, 1.0)
binary = (ratio >= 0.5)  # boolean (240,8)
patterns = []
for i in range(Ntraj):
    patterns.append("".join("H" if b else "L" for b in binary[i]))
from collections import Counter
pat_counts = Counter(patterns)
# sort by count desc, then pattern asc for determinism
sorted_pats = sorted(pat_counts.items(), key=lambda kv: (-kv[1], kv[0]))
top10 = [{"pattern": p, "count": c} for p, c in sorted_pats[:10]]
print("distinct patterns:", len(pat_counts))
print("top10:", top10)

# ---------------------------------------------------------------------------
# TASK 2b: k-means on row-standardized 240x8 block variance matrix
# ---------------------------------------------------------------------------
row_mean = B.mean(axis=1, keepdims=True)
row_std = B.std(axis=1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    Vstd = (B - row_mean) / np.where(row_std > 0, row_std, 1.0)

def kmeans(Xm, k, seed=42, n_restarts=10, max_iter=300):
    rng = np.random.default_rng(seed)
    n = Xm.shape[0]
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    for _ in range(n_restarts):
        idx = rng.choice(n, size=k, replace=False)
        centers = Xm[idx].copy()
        labels = None
        for _it in range(max_iter):
            dist = np.linalg.norm(Xm[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dist, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(k):
                m = labels == c
                if m.sum() > 0:
                    new_centers[c] = Xm[m].mean(axis=0)
                else:
                    new_centers[c] = Xm[rng.integers(0, n)].copy()
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        dist = np.linalg.norm(Xm[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        inertia = float(np.sum(dist[labels, labels] ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers
    return best_labels, best_centers, best_inertia

def cluster_report(Vstd, k):
    labels, centers, inertia = kmeans(Vstd, k)
    sizes = np.bincount(labels, minlength=k)
    return {
        "K": k,
        "cluster_sizes": sizes.tolist(),
        "inertia": inertia,
        "mean_profiles": [np.mean(B[labels == c], axis=0).tolist() if np.any(labels == c) else [0.0]*8
                          for c in range(k)],
    }

k2 = cluster_report(Vstd, 2)
k3 = cluster_report(Vstd, 3)
print("k2 sizes:", k2["cluster_sizes"])
print("k3 sizes:", k3["cluster_sizes"])

regime = {
    "top_patterns": top10,
    "k2": k2,
    "k3": k3,
}
with open(os.path.join(RES, "regime_comparison.json"), "w") as f:
    json.dump(regime, f, indent=2)
print("regime_comparison.json written")

# ---------------------------------------------------------------------------
# TASK 3: model comparison (pooled Gaussian log-likelihood over increments)
# ---------------------------------------------------------------------------
inc = X[:, 1:] - X[:, :-1]          # (240,127)
N = Ntraj * Ninc                    # 30480
logN = np.log(N)

def gll(n, s2):
    """Gaussian log-likelihood contribution for n points with MLE variance s2:
    LL = -0.5*n*log(2*pi*s2) - 0.5*n  (since sum(res^2)/s2 = n at MLE)
    """
    return -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * n

EPS = 1e-12  # numerical floor on variance (consistent with +1e-12 convention in task)
def gll_raw(res, s2):
    s2 = max(float(s2), EPS)
    return -0.5 * len(res) * np.log(2 * np.pi * s2) - 0.5 * np.sum(res * res) / s2

# ---- a. stationary-global ----
gmean = inc.mean()
ginc = inc - gmean
s2_global = float(np.mean(ginc * ginc))
LL_a = gll_raw(ginc, s2_global)
k_a = 1
BIC_a = -2 * LL_a + k_a * logN

# ---- b. stationary-per-traj ----
LL_b = 0.0
s2_per = np.zeros(Ntraj)
for i in range(Ntraj):
    m = inc[i].mean()
    r = inc[i] - m
    s2 = float(np.mean(r * r))
    s2_per[i] = s2
    LL_b += gll_raw(r, s2)
k_b = Ntraj
BIC_b = -2 * LL_b + k_b * logN

# ---- c. piecewise ----
LL_c = 0.0
per_traj_ll_c = np.zeros(Ntraj)
per_traj_ll_b = np.zeros(Ntraj)
for i in range(Ntraj):
    tid = traj_ids[i]
    if tid in nonstat_set:
        cpos = int(cp_steps[traj_ids == tid][0])
        # segment1: increments t=0..cpos-1 ; segment2: t=cpos..126
        seg1 = inc[i, :cpos]
        seg2 = inc[i, cpos:]
        m1 = seg1.mean(); r1 = seg1 - m1; s2_1 = float(np.mean(r1*r1))
        m2 = seg2.mean(); r2 = seg2 - m2; s2_2 = float(np.mean(r2*r2))
        ll_i = gll_raw(r1, s2_1) + gll_raw(r2, s2_2)
    else:
        m = inc[i].mean(); r = inc[i] - m; s2 = float(np.mean(r*r))
        ll_i = gll_raw(r, s2)
    LL_c += ll_i
    per_traj_ll_c[i] = ll_i
    per_traj_ll_b[i] = gll_raw(inc[i] - inc[i].mean(), s2_per[i])
k_c = Ntraj + 4 * n_nonstat
BIC_c = -2 * LL_c + k_c * logN

# ---- d. smooth: per-traj linear drift + global sigma^2 ----
# fit levels: X_i[t] ~ a + b*t ; residual on increments = inc - b_i
slopes_i = np.zeros(Ntraj)
inc_res = np.zeros_like(inc)
for i in range(Ntraj):
    t = np.arange(T, dtype=float)
    # linear fit: X ~ a + b*t
    A = np.column_stack([np.ones(T), t])
    beta = np.linalg.lstsq(A, X[i], rcond=None)[0]
    b = beta[1]
    slopes_i[i] = b
    inc_res[i] = inc[i] - b
s2_smooth = float(np.mean(inc_res * inc_res))
LL_d = gll_raw(inc_res, s2_smooth)
k_d = 2 * Ntraj + 1   # 481
BIC_d = -2 * LL_d + k_d * logN

# fraction where piecewise per-traj LL exceeds stationary-per-traj by > 10 units
frac_beats = float(np.mean(per_traj_ll_c - per_traj_ll_b > 10.0))

models = {
    "stationary-global": (LL_a, BIC_a, k_a),
    "stationary-per-traj": (LL_b, BIC_b, k_b),
    "piecewise": (LL_c, BIC_c, k_c),
    "smooth": (LL_d, BIC_d, k_d),
}
best = min(models, key=lambda m: models[m][1])
print("Models LL/BIC:")
for m, (ll, bic, k) in models.items():
    print("  %s: LL=%.4f BIC=%.4f k=%d" % (m, ll, bic, k))
print("best model:", best)
print("frac piecewise beats stationary by >10:", frac_beats)

# save model comparison
with open(os.path.join(RES, "model_comparison.json"), "w") as f:
    json.dump({
        "N": N,
        "n_nonstationary": n_nonstat,
        "models": {m: {"LL": float(ll), "BIC": float(bic), "k": int(k)} for m, (ll, bic, k) in models.items()},
        "best_model": best,
        "fraction_piecewise_beats_stationary_gt10": frac_beats,
    }, f, indent=2)
print("model_comparison.json written")

# ---------------------------------------------------------------------------
# Summary dict for parent
# ---------------------------------------------------------------------------
summary = {
    "scaling_slope_mean": scaling_stats["slope_mean"],
    "scaling_slope_median": scaling_stats["slope_median"],
    "scaling_slope_min": scaling_stats["slope_min"],
    "scaling_slope_max": scaling_stats["slope_max"],
    "active_count_0_01": count_01,
    "active_count_0_05": count_05,
    "active_count_0_1": count_010,
    "top_patterns": top10,
    "k2_cluster_sizes": k2["cluster_sizes"],
    "k3_cluster_sizes": k3["cluster_sizes"],
    "best_model": best,
    "model_bic": {m: float(models[m][1]) for m in models},
    "model_ll": {m: float(models[m][0]) for m in models},
    "fraction_piecewise_beats_stationary": frac_beats,
}
print("SUMMARY:")
print(json.dumps(summary, indent=2))

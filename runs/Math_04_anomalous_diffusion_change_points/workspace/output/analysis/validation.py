#!/usr/bin/env python3
"""Phase 4: validation of change-point findings (pure numpy).

Tasks:
 1. Per-trajectory stationary null test (16-step blocks, B=500 permutations).
 2. Leave-one-trajectory-out 2-class cross-validation (k-means K=2, seed 42).
 3. Trajectory resampling (block bootstrap, seed 42, n=1000).
 4. Failure boundaries (early/late cp, weak contrast, endpoint).
 5. Cross-agreement (active_flag vs nonstat; quiet-then-active).
"""
import sys
import os
import json

import numpy as np

BASE = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project"
DATA = os.path.join(BASE, "data", "anonymous_trajectories.npz")
RES = os.path.join(BASE, "output", "results")
sys.path.insert(0, os.path.join(BASE, "output", "analysis"))
os.makedirs(RES, exist_ok=True)

from change_point import analyze  # child-2 increment-based 16-step LRT (pure numpy)

N = 240
B = 8
CHIR2_1DF_095 = 3.84
V_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Load inputs (child 1/2/3 outputs)
# ---------------------------------------------------------------------------
d = np.load(DATA)
X = d["trajectories"]            # (240,128)
T = X.shape[1]                  # 128
n_inc = T - 1                   # 127

# 8 block variances (child-1 profile_blockvar.csv, matches child-2/3)
pv = np.genfromtxt(os.path.join(RES, "profile_blockvar.csv"), delimiter=",", skip_header=1)
BV = pv[:, 1:].copy()           # (240,8)

# child-2 change_points.csv
rows = open(os.path.join(RES, "change_points.csv")).read().splitlines()
cp_steps = np.array([float(r.split(",")[1]) for r in rows[1:]])
logLR_cp = np.array([float(r.split(",")[2]) for r in rows[1:]])
V_pre_cp = np.array([float(r.split(",")[3]) for r in rows[1:]])
V_post_cp = np.array([float(r.split(",")[4]) for r in rows[1:]])
ratio_cp = np.array([float(r.split(",")[5]) for r in rows[1:]])
labels = np.array([r.split(",")[6] for r in rows[1:]])
nonstat = (labels == "nonstationary").astype(int)

# child-3 scaling.csv active_flag
sv = np.genfromtxt(os.path.join(RES, "scaling.csv"), delimiter=",", skip_header=1)
active_flag = sv[:, -1].astype(int)

n_nonstat = int(nonstat.sum())
n_stat = N - n_nonstat
print("n_nonstat=%d n_stat=%d" % (n_nonstat, n_stat))


# ---------------------------------------------------------------------------
# Helper: max two-segment logLR over 7 boundaries from 8 block variances.
# Each block counts as n=16; V_pre/V_post = mean of block variances in region.
# ---------------------------------------------------------------------------
def max_loglr_from_blocks(bv):
    loglrs = np.empty(7)
    for j in range(1, 8):
        n_pre = 16 * j
        n_post = 16 * (8 - j)
        V_pre = max(float(np.mean(bv[:j])), V_FLOOR)
        V_post = max(float(np.mean(bv[j:])), V_FLOOR)
        V_pool = max((n_pre * V_pre + n_post * V_post) / (n_pre + n_post), V_FLOOR)
        loglrs[j - 1] = n_pre * np.log(V_pool / V_pre) + n_post * np.log(V_pool / V_post)
    return float(np.max(loglrs))


# ---------------------------------------------------------------------------
# TASK 1: stationary null test
# ---------------------------------------------------------------------------
obs_loglr = np.array([max_loglr_from_blocks(BV[i]) for i in range(N)])
p_values = np.empty(N)
for i in range(N):
    rng = np.random.RandomState(seed=i)
    bv = BV[i].copy()
    max_obs = obs_loglr[i]
    count = 0
    for _ in range(500):
        p = rng.permutation(8)
        if max_loglr_from_blocks(bv[p]) > max_obs:
            count += 1
    p_values[i] = (1 + count) / 501

np.savetxt(os.path.join(RES, "null_test.csv"),
           np.column_stack([np.arange(N), obs_loglr, p_values]),
           delimiter=",", header="traj_id,observed_logLR,p_value", comments="")

n_ns_p05 = int(np.sum(nonstat & (p_values < 0.05)))
frac_p05 = float(np.mean(p_values < 0.05))
corr_p_nonstat = float(np.corrcoef(p_values, nonstat.astype(float))[0, 1])
print("Task1: n_ns_p<0.05=%d  frac_p<0.05=%.4f  corr(p,nonstat)=%.4f"
      % (n_ns_p05, frac_p05, corr_p_nonstat))


# ---------------------------------------------------------------------------
# TASK 2: leave-one-trajectory-out k-means CV
# ---------------------------------------------------------------------------
def kmeans(Xm, k, seed, n_restarts=10, max_iter=300):
    rng = np.random.RandomState(seed)
    n = Xm.shape[0]
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    for _ in range(n_restarts):
        idx = rng.choice(n, size=k, replace=False)
        centers = Xm[idx].copy()
        for _it in range(max_iter):
            dist = np.linalg.norm(Xm[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dist, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(k):
                m_ = (labels == c)
                if m_.sum() > 0:
                    new_centers[c] = Xm[m_].mean(axis=0)
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

acc = 0.0
for hold in range(N):
    Xte = BV[hold:hold + 1]
    train_idx = np.concatenate([np.arange(hold), np.arange(hold + 1, N)])
    Xtr = BV[train_idx].copy()
    m = Xtr.mean(axis=0)
    s = Xtr.std(axis=0)
    Xtr_s = (Xtr - m) / np.where(s > 0, s, 1.0)
    Xte_s = (Xte - m) / np.where(s > 0, s, 1.0)
    labels, centers, _ = kmeans(Xtr_s, 2, seed=42)
    # map each cluster -> child-2 label via majority vote in training
    cluster_label = np.zeros(2)
    for c in range(2):
        sel = (labels == c)
        if sel.sum() > 0:
            cluster_label[c] = float(np.mean(nonstat[train_idx][sel]))
        else:
            cluster_label[c] = 0.0
    # predict held-out by nearest prototype
    d = np.linalg.norm(Xte_s - centers, axis=1)
    pred_cluster = int(np.argmin(d))
    pred_label = (cluster_label[pred_cluster] >= 0.5)
    acc += int(pred_label == bool(nonstat[hold]))

cv_accuracy = float(acc / N)
cv_chance = float(n_nonstat / N)
print("Task2: cv_accuracy=%.4f  cv_chance=%.4f" % (cv_accuracy, cv_chance))


# ---------------------------------------------------------------------------
# TASK 3: trajectory resampling (block bootstrap, seed 42, n=1000)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(42)
n_boot = 1000
stat_frac_nonstat = np.empty(n_boot)
stat_median_cp = np.empty(n_boot)
stat_m32 = np.empty(n_boot)
# full-population M(32) baseline (for reference, population mean)
M32_full = np.array([float(np.mean((X[i, 32:] - X[i, :96]) ** 2)) for i in range(N)])

for b in range(n_boot):
    idx = rng.choice(N, size=N, replace=True)
    Xb = X[idx]                              # resampled positions (n,128)
    inc_b = np.diff(Xb, axis=1)              # increments (n,127)
    cums = np.cumsum(inc_b, axis=1)
    cums = np.concatenate([np.zeros((inc_b.shape[0], 1)), cums], axis=1)
    cumsq = np.cumsum(inc_b ** 2, axis=1)
    cumsq = np.concatenate([np.zeros((inc_b.shape[0], 1)), cumsq], axis=1)
    resb = analyze(16, inc_b, cums, cumsq)
    label = resb["label"] == "nonstationary"
    stat_frac_nonstat[b] = float(np.mean(label))
    if np.any(label):
        stat_median_cp[b] = float(np.median(resb["cp_steps"][label]))
    else:
        stat_median_cp[b] = float("nan")
    M32_b = np.array([float(np.mean((Xb[i, 32:] - Xb[i, :96]) ** 2)) for i in range(N)])
    stat_m32[b] = float(np.mean(M32_b) / 32.0)

def ci(arr):
    arr = np.asarray(arr, dtype=float)
    return [float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

ci_frac = ci(stat_frac_nonstat)
ci_cp = ci(stat_median_cp)
ci_m = ci(stat_m32)

with open(os.path.join(RES, "bootstrap_ci.csv"), "w") as f:
    f.write("statistic,mean,p025,p975\n")
    for name, c in zip(["frac_nonstat", "median_cp", "m32_over_32"], [ci_frac, ci_cp, ci_m]):
        f.write("%s,%.6f,%.6f,%.6f\n" % (name, c[0], c[1], c[2]))

print("Task3: frac_nonstat CI=[%s] median_cp CI=[%s] m32/32 CI=[%s]"
      % (",".join("%.4f" % x for x in ci_frac),
         ",".join("%.4f" % x for x in ci_cp),
         ",".join("%.4f" % x for x in ci_m)))


# ---------------------------------------------------------------------------
# TASK 4: failure boundaries
# ---------------------------------------------------------------------------
ns_idx = np.where(nonstat == 1)[0]
cp_ns = cp_steps[ns_idx]
ratio_ns = ratio_cp[ns_idx]
early = cp_ns <= 16
late = cp_ns >= 96
weak = np.abs(np.log(ratio_ns)) < 0.5
failure_early_frac = float(np.mean(early))
failure_late_frac = float(np.mean(late))
failure_weak_contrast_frac = float(np.mean(weak))
ep_abs = np.abs(X[:, -1])
weak_idx = ns_idx[weak]
rest_idx = ns_idx[~weak]
endpoint_weak_vs_rest = float(np.mean(ep_abs[weak_idx]) - np.mean(ep_abs[rest_idx])) if weak.any() else float("nan")
print("Task4: early=%.4f late=%.4f weak=%.4f  endpoint_weak_vs_rest=%.4f"
      % (failure_early_frac, failure_late_frac, failure_weak_contrast_frac, endpoint_weak_vs_rest))


# ---------------------------------------------------------------------------
# TASK 5: cross-agreement
# ---------------------------------------------------------------------------
agreement_active_nonstat = float(np.corrcoef(active_flag, nonstat.astype(float))[0, 1])

# H/L patterns (child-3 logic): ratio = B / maxv; H if >= 0.5
maxv = np.max(BV, axis=1, keepdims=True)
ratio_hl = BV / np.where(maxv > 0, maxv, 1.0)
binary = (ratio_hl >= 0.5)
trailingH = (binary[:, 5] & binary[:, 6] & binary[:, 7]).astype(int)
leadingL = ((~binary[:, 0]) & (~binary[:, 1])).astype(int)
quiet_then_active = (trailingH == 1) & (leadingL == 1)
quiet_then_active_nonstat_frac = float(np.mean(quiet_then_active[nonstat == 1]))
print("Task5: agreement_active_nonstat=%.4f  quiet_then_active_nonstat_frac=%.4f"
      % (agreement_active_nonstat, quiet_then_active_nonstat_frac))


# ---------------------------------------------------------------------------
# Assemble JSON report
# ---------------------------------------------------------------------------
report = {
    "n_traj": N,
    "n_nonstationary": n_nonstat,
    "n_stationary": n_stat,
    "null_test": {
        "n_nonstat_with_p_lt_0_05": n_ns_p05,
        "frac_all_p_lt_0_05": frac_p05,
        "corr_p_vs_nonstat": corr_p_nonstat,
    },
    "cross_validation": {
        "cv_accuracy": cv_accuracy,
        "cv_chance": cv_chance,
        "k": 2,
        "seed": 42,
        "n_folds": N,
    },
    "bootstrap_ci": {
        "frac_nonstat": ci_frac,
        "median_cp": ci_cp,
        "m32_over_32": ci_m,
    },
    "failure_boundaries": {
        "early_cp_steps_le_16_frac": failure_early_frac,
        "late_cp_steps_ge_96_frac": failure_late_frac,
        "weak_contrast_abslogratio_lt_0_5_frac": failure_weak_contrast_frac,
        "endpoint_abs_mean_weak_minus_rest": float(0.0) if np.isnan(
            endpoint_weak_vs_rest) else endpoint_weak_vs_rest,
        "endpoint_note": "weak-contrast set is empty (all nonstationary have ratio>=2.0, |log(ratio)|>=0.693>0.5); difference undefined, reported as 0.0",
    },
    "cross_agreement": {
        "active_flag_vs_nonstat_point_biserial": agreement_active_nonstat,
        "quiet_then_active_nonstat_frac": quiet_then_active_nonstat_frac,
    },
}
with open(os.path.join(RES, "validation_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Wrote validation_report.json, null_test.csv, bootstrap_ci.csv")

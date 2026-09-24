#!/usr/bin/env python3
"""Supplementary model-free robustness: permutation + bootstrap CIs, plus a figure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent            # output/analysis
OUT = HERE.parent                                  # output/
DATA = OUT.parent / "data"                         # project/data
RES = OUT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
np.random.seed(20260922)

m = pd.read_csv(DATA / "unit_distance_mean_latency.tsv", sep="\t")
v = pd.read_csv(DATA / "unit_distance_latency_variability.tsv", sep="\t")
set_a = pd.read_csv(DATA / "event_recording_set_a.tsv", sep="\t")
set_b = pd.read_csv(DATA / "event_recording_set_b.tsv", sep="\t")

m["pair_distance"] = m["pair_distance"].to_numpy(float)
m["mean_event_latency"] = m["mean_event_latency"].to_numpy(float)
v["event_latency_sd"] = v["event_latency_sd"].to_numpy(float)

out = {
    "permutation_test": {},
    "bootstrap_95ci": {},
    "event_state_perm": {},
    "cross_recording_ks_perm": {},
}

# ---- permutation test: distance vs mean latency ----
def perm_spearman_p(x, y, n_perm=5000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = stats.spearmanr(x, y).pvalue
    rng = np.random.default_rng(42)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        p = stats.spearmanr(x, yp).pvalue
        if p <= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)

obs_sp, perm_p_sp = perm_spearman_p(m["pair_distance"], m["mean_event_latency"])
out["permutation_test"]["dist_vs_meanlat_spearman"] = {
    "obs_p": float(obs_sp), "perm_p": float(perm_p_sp), "n_perm": 5000}
obs_ps, perm_p_ps = perm_spearman_p(m["pair_distance"], v["event_latency_sd"])
out["permutation_test"]["dist_vs_latencysd_spearman"] = {
    "obs_p": float(obs_ps), "perm_p": float(perm_p_ps), "n_perm": 5000}

# ---- bootstrap 95% CI for Spearman rho (distance vs mean latency) ----
def boot_spearman(x, y, n_boot=5000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(7)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = stats.spearmanr(x[idx], y[idx]).statistic
    return np.percentile(vals, [2.5, 97.5]), stats.spearmanr(x, y).statistic

ci_sp, obs_rho = boot_spearman(m["pair_distance"], m["mean_event_latency"])
out["bootstrap_95ci"] = {
    "dist_vs_meanlat_spearman_rho_ci95": [float(ci_sp[0]), float(ci_sp[1])],
    "point_estimate": float(obs_rho), "n_boot": 5000}

# ---- event-state sign vs time: permutation, both recordings ----
def perm_eventstate(df, n_perm=5000):
    t = df["event_relative_time"].to_numpy(float)
    s = df["pre_event_signal"].to_numpy(float)
    obs = stats.spearmanr(np.sign(s), t).pvalue
    rng = np.random.default_rng(1)
    cnt = 0
    for _ in range(n_perm):
        sg = rng.integers(0, 2, len(s)) * 2 - 1  # random +/-1 sign
        p = stats.spearmanr(sg, t).pvalue
        if p <= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)

for nm, df in [("set_a", set_a), ("set_b", set_b)]:
    obs, perm_p = perm_eventstate(df)
    out["event_state_perm"][nm] = {
        "spearman_p": float(obs), "perm_p": float(perm_p), "n_perm": 5000}
    print(f"eventstate {nm}: spearman_p={obs:.4f} perm_p={perm_p:.4f}")

# ---- cross-recording KS permutation (nonparametric) ----
ta = set_a["event_relative_time"].to_numpy(float)
tb = set_b["event_relative_time"].to_numpy(float)
obs_ks = stats.ks_2samp(ta, tb).pvalue
rng = np.random.default_rng(3)
cnt = 0
for _ in range(5000):
    perm = tb[rng.integers(0, len(tb), len(tb))]
    p = stats.ks_2samp(ta, perm).pvalue
    if p <= obs_ks:
        cnt += 1
perm_ks = (cnt + 1) / 5001
print(f"cross_recording KS: obs_p={obs_ks:.4f} perm_p={perm_ks:.4f}")
out["cross_recording_ks_perm"] = {
    "obs_p": float(obs_ks), "perm_p": float(perm_ks), "n_perm": 5000}

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(m["pair_distance"], m["mean_event_latency"], s=28, alpha=0.8, edgecolors="none")
for i, r in m.iterrows():
    ax.annotate(str(int(r["unit_index"])), (r["pair_distance"], r["mean_event_latency"]),
                fontsize=6, ha="center", va="bottom")
ax.set_xlabel("pair_distance")
ax.set_ylabel("mean_event_latency")
ax.set_title("Unit pair-distance vs mean event latency\n(n=41; Spearman rho=-0.138, p=0.388)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(m["pair_distance"], v["event_latency_sd"], s=28, alpha=0.8, edgecolors="none")
ax.set_xlabel("pair_distance")
ax.set_ylabel("event_latency_sd")
ax.set_title("Unit pair-distance vs latency SD\n(n=41; Spearman rho=-0.020, p=0.901)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "unit_distance_geometry.png", dpi=140, bbox_inches="tight")
print("wrote", FIG / "unit_distance_geometry.png")

with open(RES / "supplementary_robustness.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("wrote", RES / "supplementary_robustness.json")
print("supplementary done")

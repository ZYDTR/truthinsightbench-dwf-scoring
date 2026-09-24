#!/usr/bin/env python3
"""Independent cross-check: re-derive every core axis directly from the raw TSVs
using a *separate* code path (independent from run_analysis.py) and compare against
the primary evidence JSON. Writes per-axis JSON + a consolidated consistency summary
to output/cross_check/. Each block is self-contained so it can be re-run alone."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]            # project root (..../project)
DATA = ROOT / "data"
OUT = ROOT / "output" / "cross_check"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = (ROOT / "output" / "results" / "result_evidence.json").read_text()

def load(name):
    return pd.read_csv(DATA / name, sep="\t")

def spearman_rho_p(x, y):
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}

def pearson_r_p(x, y):
    r = stats.pearsonr(x, y)
    return {"r": float(r.statistic), "p": float(r.pvalue)}

def perm_spearman_p(x, y, n_perm=2000, seed=1234):
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = stats.spearmanr(x, y).pvalue
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = stats.spearmanr(x, rng.permutation(y)).pvalue
        if p <= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))

# ---- Axis 1: event_recording_set_a ----
a = load("event_recording_set_a.tsv")
t = a["event_relative_time"].to_numpy(float); s = a["pre_event_signal"].to_numpy(float)
axis1 = {
    "file": "event_recording_set_a.tsv",
    "n": int(len(a)),
    "contiguous_index": bool((a["event_index"].max() - a["event_index"].min() + 1) == len(a)
                             and a["event_index"].is_unique),
    "mean_time": float(t.mean()), "median_time": float(np.median(t)), "sd_time": float(t.std(ddof=1)),
    "neg_time_frac": float((t < 0).mean()),
    "signal_mean": float(s.mean()), "signal_median": float(np.median(s)),
    "event_state_spearman": spearman_rho_p(np.sign(s), t),
    "state_pos_mean_time": float(t[s > 0].mean()), "state_neg_mean_time": float(t[s < 0].mean()),
    "state_ttest_p": float(stats.ttest_ind(t[s > 0], t[s < 0]).pvalue),
}
(OUT / "independent_event_set_a.json").write_text(json.dumps(axis1, indent=2))

# ---- Axis 2: event_recording_set_b ----
b = load("event_recording_set_b.tsv")
t = b["event_relative_time"].to_numpy(float); s = b["pre_event_signal"].to_numpy(float)
axis2 = {
    "file": "event_recording_set_b.tsv",
    "n": int(len(b)),
    "contiguous_index": bool((b["event_index"].max() - b["event_index"].min() + 1) == len(b)
                             and b["event_index"].is_unique),
    "mean_time": float(t.mean()), "median_time": float(np.median(t)), "sd_time": float(t.std(ddof=1)),
    "neg_time_frac": float((t < 0).mean()),
    "signal_mean": float(s.mean()), "signal_median": float(np.median(s)),
    "event_state_spearman": spearman_rho_p(np.sign(s), t),
    "state_pos_mean_time": float(t[s > 0].mean()), "state_neg_mean_time": float(t[s < 0].mean()),
    "state_ttest_p": float(stats.ttest_ind(t[s > 0], t[s < 0]).pvalue),
}
(OUT / "independent_event_set_b.json").write_text(json.dumps(axis2, indent=2))

# ---- Axis 3: multichannel_event_panel ----
p = load("multichannel_event_panel.tsv")
units = sorted(p["unit_index"].unique().tolist())
axis3 = {
    "file": "multichannel_event_panel.tsv",
    "n_rows": int(len(p)),
    "units_present": units,
    "events_per_unit": {int(u): int((p["unit_index"] == u).sum()) for u in units},
    "per_unit_signal_mean": {int(u): float(p.loc[p["unit_index"] == u, "pre_event_signal"].mean()) for u in units},
}
(OUT / "independent_panel.json").write_text(json.dumps(axis3, indent=2, default=str))

# ---- Axis 4: unit_distance_mean_latency ----
ml = load("unit_distance_mean_latency.tsv")
d = ml["pair_distance"].to_numpy(float); y = ml["mean_event_latency"].to_numpy(float)
# quartile bin means (independent ordering)
mlc = ml.copy()
mlc["bin"] = pd.qcut(mlc["pair_distance"], q=4, duplicates="drop")
axis4 = {
    "file": "unit_distance_mean_latency.tsv",
    "n_units": int(len(ml)),
    "dist_range": [float(d.min()), float(d.max())],
    "meanlat_range": [float(y.min()), float(y.max())],
    "pearson": pearson_r_p(d, y),
    "spearman": spearman_rho_p(d, y),
    "perm_p": perm_spearman_p(d, y, n_perm=2000),
    "quartile_meanlat": {str(k): float(v) for k, v in
                         mlc.groupby("bin", observed=True)["mean_event_latency"].mean().items()},
}
(OUT / "independent_unit_meanlat.json").write_text(json.dumps(axis4, indent=2, default=str))

# ---- Axis 5: unit_distance_latency_variability ----
vl = load("unit_distance_latency_variability.tsv")
d = vl["pair_distance"].to_numpy(float); y = vl["event_latency_sd"].to_numpy(float)
axis5 = {
    "file": "unit_distance_latency_variability.tsv",
    "n_units": int(len(vl)),
    "pearson": pearson_r_p(d, y),
    "spearman": spearman_rho_p(d, y),
    "zero_sd_count": int((y == 0).sum()),
    "perm_p": perm_spearman_p(d, y, n_perm=2000),
}
(OUT / "independent_unit_variability.json").write_text(json.dumps(axis5, indent=2))

# ---- Axis 6: cross-recording KS (set_a vs set_b) ----
ta = load("event_recording_set_a.tsv")["event_relative_time"].to_numpy(float)
tb = load("event_recording_set_b.tsv")["event_relative_time"].to_numpy(float)
obs_ks = stats.ks_2samp(ta, tb).pvalue
rng = np.random.default_rng(9)
cnt = 0
for _ in range(2000):
    if stats.ks_2samp(ta, tb[rng.permutation(len(tb))]).pvalue <= obs_ks:
        cnt += 1
axis6 = {
    "files": ["event_recording_set_a.tsv", "event_recording_set_b.tsv"],
    "ks_stat": float(stats.ks_2samp(ta, tb).statistic),
    "ks_p": float(obs_ks),
    "perm_p": float((cnt + 1) / 2001),
}
(OUT / "independent_cross_recording.json").write_text(json.dumps(axis6, indent=2))

# ---- Consistency: compare independent re-derivations to primary evidence ----
primary = json.loads(PRIMARY)
def close(x, y, rel=True, tol=0.01):
    if x is None or y is None:
        return None
    if x == y:
        return True
    if rel:
        return abs(x - y) / (abs(x) + abs(y)) < tol
    return abs(x - y) < tol

report = {}
# event state b spearman
rep_b = {
    "spearman_p": close(primary["event_state"]["set_b"]["spearman_p"],
                        axis2["event_state_spearman"]["p"], rel=True, tol=1e-3),
    "mean_pos": close(primary["event_state"]["set_b"]["mean_time_pos"],
                      axis2["state_pos_mean_time"], rel=True, tol=0.01),
}
# event state a
rep_a = {
    "spearman_p": close(primary["event_state"]["set_a"]["spearman_p"],
                        axis1["event_state_spearman"]["p"], rel=True, tol=1e-3),
}
# unit meanlat correlations
rep_ml = {
    "pearson_p": close(primary["correlations"]["dist_vs_meanlat_pearson"]["p"],
                       axis4["pearson"]["p"], rel=True, tol=1e-3),
    "spearman_p": close(primary["correlations"]["dist_vs_meanlat_spearman"]["p"],
                        axis4["spearman"]["p"], rel=True, tol=1e-3),
    "n_units": primary["unit_distance_n"] == axis4["n_units"],
}
# unit variability
rep_vl = {
    "pearson_p": close(primary["correlations"]["dist_vs_latencysd_pearson"]["p"],
                       axis5["pearson"]["p"], rel=True, tol=1e-3),
    "spearman_p": close(primary["correlations"]["dist_vs_latencysd_spearman"]["p"],
                        axis5["spearman"]["p"], rel=True, tol=1e-3),
    "zero_sd_count": primary["zero_sd_units"] == axis5["zero_sd_count"],
}
# cross recording
rep_ks = {
    "ks_p": close(primary["correlations"]["cross_recording_ks"]["p"],
                  axis6["ks_p"], rel=True, tol=0.02),
}
report["axis1_event_set_a"] = axis1
report["axis2_event_set_b"] = axis2
report["axis3_panel"] = axis3
report["axis4_unit_meanlat"] = axis4
report["axis5_unit_variability"] = axis5
report["axis6_cross_recording"] = axis6
report["consistency_vs_primary"] = {
    "event_state_b": rep_b,
    "event_state_a": rep_a,
    "unit_meanlat": rep_ml,
    "unit_variability": rep_vl,
    "cross_recording": rep_ks,
}
(OUT / "consistency_summary.json").write_text(json.dumps(report, indent=2, default=str))
print("WROTE", len(list(OUT.glob("*.json"))), "files to", OUT)
for f in sorted(OUT.glob("*.json")):
    print("  ", f.name, f"{f.stat().st_size} bytes")
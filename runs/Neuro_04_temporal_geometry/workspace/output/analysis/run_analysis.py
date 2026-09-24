#!/usr/bin/env python3
"""Executable analysis for Neuro_04_temporal_geometry.

Inputs (workspace data/, read-only):
  event_recording_set_a.tsv      event_index, event_relative_time, pre_event_signal
  event_recording_set_b.tsv      event_index, event_relative_time, pre_event_signal
  unit_distance_mean_latency.tsv unit_index, pair_distance, mean_event_latency
  unit_distance_latency_variability.tsv  unit_index, pair_distance, event_latency_sd
  multichannel_event_panel.tsv   event_index, unit_index, pre_event_signal

Outputs (output/results, output/figures):
  recording_summary.csv          per-recording timing statistics
  distance_correlations.csv      distance-vs-latency statistics (primary + robustness)
  panel_signals.csv              per-unit signal statistics in the multichannel panel
  cross_recording.csv            cross-recording distribution comparison
  contingency_event_state.csv    event-state (sign of pre-event signal) x timing summary
  result_evidence.json           machine-readable evidence package for Result.md
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, spatial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # project root
DATA = ROOT / "data"
OUT = HERE.parent                  # output/
RES = OUT / "results"
FIG = OUT / "figures"
np.random.seed(20260922)

# ----------------------------------------------------------------- load ----
def load(name):
    p = DATA / name
    df = pd.read_csv(p, sep="\t", dtype=str)
    return df

set_a = load("event_recording_set_a.tsv")
set_b = load("event_recording_set_b.tsv")
mean_lat = load("unit_distance_mean_latency.tsv")
var_lat = load("unit_distance_latency_variability.tsv")
panel = load("multichannel_event_panel.tsv")

set_a["event_index"] = pd.to_numeric(set_a["event_index"])
set_a["event_relative_time"] = pd.to_numeric(set_a["event_relative_time"])
set_a["pre_event_signal"] = pd.to_numeric(set_a["pre_event_signal"])
set_b["event_index"] = pd.to_numeric(set_b["event_index"])
set_b["event_relative_time"] = pd.to_numeric(set_b["event_relative_time"])
set_b["pre_event_signal"] = pd.to_numeric(set_b["pre_event_signal"])
mean_lat["unit_index"] = pd.to_numeric(mean_lat["unit_index"])
mean_lat["pair_distance"] = pd.to_numeric(mean_lat["pair_distance"])
mean_lat["mean_event_latency"] = pd.to_numeric(mean_lat["mean_event_latency"])
var_lat["unit_index"] = pd.to_numeric(var_lat["unit_index"])
var_lat["pair_distance"] = pd.to_numeric(var_lat["pair_distance"])
var_lat["event_latency_sd"] = pd.to_numeric(var_lat["event_latency_sd"])
panel["event_index"] = pd.to_numeric(panel["event_index"])
panel["unit_index"] = pd.to_numeric(panel["unit_index"])
panel["pre_event_signal"] = pd.to_numeric(panel["pre_event_signal"])

# ----------------------------------------------------------------- 1 -------
# Recording-level timing summaries (independent analysis unit = one recording)
def rec_summary(df, name):
    t = df["event_relative_time"].to_numpy()
    out = {
        "recording": name,
        "n_events": int(len(df)),
        "event_index_min": int(df["event_index"].min()),
        "event_index_max": int(df["event_index"].max()),
        "index_contiguous": bool((df["event_index"].max() - df["event_index"].min() + 1) == len(df) and df["event_index"].is_unique),
        "relative_time_mean": float(t.mean()),
        "relative_time_median": float(np.median(t)),
        "relative_time_sd": float(t.std(ddof=1)),
        "relative_time_iqr": float(np.subtract(*np.percentile(t, [75, 25]))),
        "relative_time_range": float(t.max() - t.min()),
        "negative_time_frac": float((t < 0).mean()),
        "signal_mean": float(df["pre_event_signal"].mean()),
        "signal_median": float(df["pre_event_signal"].median()),
        "signal_sd": float(df["pre_event_signal"].std(ddof=1)),
    }
    return out

sa = rec_summary(set_a, "set_a")
sb = rec_summary(set_b, "set_b")
rec_summary_df = pd.DataFrame([sa, sb])
rec_summary_df.to_csv(RES / "recording_summary.csv", index=False)

# cross-recording distribution comparison (event-relative-time, independent unit)
ks_stat, ks_p = stats.ks_2samp(set_a["event_relative_time"], set_b["event_relative_time"])
print("KS two-recording event_relative_time: stat=%.4f p=%.6f" % (ks_stat, ks_p))

# pre-event signal sign (event-state proxy): positive vs negative event-state
def state_stats(df):
    s = df["pre_event_signal"].copy()
    pos = s[s > 0]
    neg = s[s < 0]
    return {
        "n_pos": int(len(pos)), "pos_mean": float(pos.mean()), "pos_median": float(pos.median()),
        "n_neg": int(len(neg)), "neg_mean": float(neg.mean()), "neg_median": float(neg.median()),
        "n_zero": int((s == 0).sum()),
    }
state_a = state_stats(set_a)
state_b = state_stats(set_b)
print("state a:", state_a)
print("state b:", state_b)

# event-state (sign of pre_event_signal) vs event_relative_time within each recording
event_state = {}
for name, df in [("set_a", set_a), ("set_b", set_b)]:
    t = df["event_relative_time"].to_numpy()
    sg = df["pre_event_signal"].to_numpy()
    pos, neg = t[sg > 0], t[sg < 0]
    from scipy import stats as _s
    tt_res = _s.ttest_ind(pos, neg)
    sp = _s.spearmanr(np.sign(sg), t)
    event_state[name] = {
        "n_events": int(len(df)),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "mean_time_pos": float(pos.mean()), "mean_time_neg": float(neg.mean()),
        "ttest_p": float(tt_res.pvalue),
        "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
    }
print("event_state:", event_state)

# event-state timing: does sign of pre_event_signal relate to event_relative_time within each recording?
for name, df in [("set_a", set_a), ("set_b", set_b)]:
    t = df["event_relative_time"].to_numpy()
    s = df["pre_event_signal"].to_numpy()
    pos, neg = t[s > 0], t[s < 0]
    print(f"{name}: mean_time pos={np.mean(pos):.2f} neg={np.mean(neg):.2f} "
          f"t-test p={stats.ttest_ind(pos, neg).pvalue:.4f} "
          f"spearman rho({np.sign(s)}, t)={stats.spearmanr(np.sign(s), t).statistic:.3f} p={stats.spearmanr(np.sign(s), t).pvalue:.3f}")

# ----------------------------------------------------------------- 2 -------
# Unit-distance geometry: correlate pair_distance with mean latency and latency SD
m = mean_lat.copy()
v = var_lat.copy()
m = m.sort_values("unit_index")
v = v.sort_values("unit_index")
merged = m.merge(v, on=["unit_index", "pair_distance"], how="outer", indicator=True)
print("units in mean:", len(m), "units in var:", len(v), "pairwise merge rows:", len(merged))
zero_sd = merged[merged["event_latency_sd"] == 0]
print("zero-SD units:", zero_sd[["unit_index", "pair_distance", "mean_event_latency"]].to_dict("records"))

r_p, p_p = stats.pearsonr(m["pair_distance"], m["mean_event_latency"])
rs_p, ps_p = stats.spearmanr(m["pair_distance"], m["mean_event_latency"])
r_s, p_s = stats.pearsonr(m["pair_distance"], v["event_latency_sd"])
rs_s, ps_s = stats.spearmanr(m["pair_distance"], v["event_latency_sd"])
print("distance vs mean latency:  pearson r=%.4f p=%.4f | spearman rho=%.4f p=%.4f" % (r_p, p_p, rs_p, ps_p))
print("distance vs latency SD:    pearson r=%.4f p=%.4f | spearman rho=%.4f p=%.4f" % (r_s, p_s, rs_s, ps_s))

# ----------------------------------------------------------------- 3 -------
# Multichannel panel: per-unit signal stats; compare with unit-distance latency
pu = panel.groupby("unit_index")["pre_event_signal"].agg(
    ["mean", "median", "std", "count"]).rename(columns={"mean": "signal_mean"}).reset_index()
pu = pu.merge(m[["unit_index", "pair_distance", "mean_event_latency"]], on="unit_index", how="left")
print("panel units:", pu["unit_index"].tolist(),
      "covered in unit-distance (pair_distance notna):", int(pu["pair_distance"].notna().sum()))
print(pu)
pu.to_csv(RES / "panel_signals.csv", index=False)

# Only 3 units -> population correlation is underpowered (n=3). Report the 3 pairs
# explicitly and flag as an analysis-unit boundary, not a general correlation.
print("panel unit-by-unit (unit, signal_mean, mean_event_latency, pair_distance):")
for _, row in pu.iterrows():
    print(f"  unit {int(row['unit_index'])}: signal_mean={row['signal_mean']:.3f} "
          f"mean_event_latency={row['mean_event_latency']:.3f} pair_distance={row['pair_distance']:.4f}")

# ----------------------------------------------------------------- 4 -------
# Cross-recording vs unit-distance: are the two recordings' event timings
# compatible with the unit-distance summaries (different analysis units)?
# Recording-level mean of event_relative_time (mean of set_a, set_b) vs mean of unit latency
all_units = m["mean_event_latency"].to_numpy()
rec_means = [sa["relative_time_mean"], sb["relative_time_mean"]]
print("recording mean event_relative_time: a=%.2f b=%.2f | unit mean_event_latency: mean=%.2f sd=%.2f"
      % (rec_means[0], rec_means[1], all_units.mean(), all_units.std(ddof=1)))

# ----------------------------------------------------------------- 5 -------
# Robustness checks (per contract: alternative definition, leave-one-out, threshold sensitivity)
robust = []

# 5a leave-one-out Spearman for distance vs mean latency
for i in range(len(m)):
    d = m["pair_distance"].drop(index=m.index[i])
    y = m["mean_event_latency"].drop(index=m.index[i])
    rho = stats.spearmanr(d, y).statistic
    robust.append({"check": "leave_one_unit_out", "unit": int(m["unit_index"].iloc[i]), "metric": "spearman_dist_vs_meanlat",
                   "rho": float(rho)})

# 5b alternative definition: use median latency (mean of event_latency? we only have mean_event_latency)
#    alternative: log(mean_event_latency) vs distance
m2 = m.copy(); m2["log_meanlat"] = np.log(m2["mean_event_latency"].clip(lower=1e-9))
r_log, p_log = stats.pearsonr(m2["pair_distance"], m2["log_meanlat"])
rs_log, ps_log = stats.spearmanr(m2["pair_distance"], m2["log_meanlat"])
robust.append({"check": "alternative_scale_loglatency", "metric": "pearson_dist_vs_log_meanlat", "r": float(r_log), "p": float(p_log)})
robust.append({"check": "alternative_scale_loglatency", "metric": "spearman_dist_vs_log_meanlat", "rho": float(rs_log), "p": float(ps_log)})

# 5c threshold sensitivity: bin distance into quartiles, compare mean latency across bins
m3 = m.copy()
m3["bin"] = pd.qcut(m3["pair_distance"], q=4, duplicates="drop")
binlat = m3.groupby("bin", observed=True)["mean_event_latency"].agg(["mean", "median", "std", "count"]).reset_index()
print(binlat)
# also bin by decile for the full range
m4 = m.copy()
m4["bin10"] = pd.qcut(m4["pair_distance"], q=10, duplicates="drop")
bin10 = m4.groupby("bin10", observed=True)["mean_event_latency"].agg(["mean", "count"]).reset_index()
print(bin10)

# 5d outlier treatment: trim top/bottom 10% of distance
lo, hi = np.percentile(m["pair_distance"], [10, 90])
m_trim = m[(m["pair_distance"] >= lo) & (m["pair_distance"] <= hi)]
r_t, p_t = stats.pearsonr(m_trim["pair_distance"], m_trim["mean_event_latency"])
rs_t, ps_t = stats.spearmanr(m_trim["pair_distance"], m_trim["mean_event_latency"])
robust.append({"check": "outlier_trim_10pct", "metric": "pearson_dist_vs_meanlat", "r": float(r_t), "p": float(p_t)})
robust.append({"check": "outlier_trim_10pct", "metric": "spearman_dist_vs_meanlat", "rho": float(rs_t), "p": float(ps_t)})

# 5e monotonicity: fraction of pairs where larger distance => larger mean latency
n_up = 0; n_down = 0
for i in range(len(m)):
    for j in range(i+1, len(m)):
        d1, d2 = m["pair_distance"].iloc[i], m["pair_distance"].iloc[j]
        y1, y2 = m["mean_event_latency"].iloc[i], m["mean_event_latency"].iloc[j]
        if d1 == d2: continue
        if (d1 < d2) == (y1 < y2): n_up += 1
        else: n_down += 1
concordance = n_up / (n_up + n_down)
robust.append({"check": "pairwise_concordance", "metric": "fraction_concordant", "value": float(concordance)})
print("pairwise concordance fraction:", concordance)

# 5f distance vs latency SD robustness
robust.append({"check": "primary", "metric": "spearman_dist_vs_meanlat", "rho": float(rs_p), "p": float(ps_p)})
robust.append({"check": "primary", "metric": "pearson_dist_vs_meanlat", "r": float(r_p), "p": float(p_p)})
robust.append({"check": "primary", "metric": "spearman_dist_vs_latency_sd", "rho": float(rs_s), "p": float(ps_s)})
robust.append({"check": "primary", "metric": "pearson_dist_vs_latency_sd", "r": float(r_s), "p": float(p_s)})

pd.DataFrame(robust).to_csv(RES / "distance_correlations.csv", index=False)

# ----------------------------------------------------------------- 6 -------
# Contiguous-event validation
for name, df in [("set_a", set_a), ("set_b", set_b)]:
    idx = df["event_index"].to_numpy()
    contiguous = bool((idx[1:] - idx[:-1] == 1).all() and idx[0] == 1)
    t = df["event_relative_time"].to_numpy()
    # how many adjacent events are temporally ordered (t[i+1] > t[i])?
    ordered = float((t[1:] > t[:-1]).mean())
    print(f"{name}: index_contiguous={contiguous} adjacent_temporal_order_frac={ordered:.3f}")

# ----------------------------------------------------------------- 7 -------
# Multichannel panel vs recording b consistency (same event_index, unit=1..3)
panel123 = panel[panel["unit_index"] <= 3].sort_values(["unit_index", "event_index"])
print("panel events per unit:", panel.groupby("unit_index").size().to_dict())
# overlap with recording b events 1..118 for unit 1 (if the panel is a slice of recording b)
print("panel unit1 event118 index max:", panel[panel["unit_index"]==1]["event_index"].max())
print("panel unit1 first 5:", panel[panel["unit_index"]==1][["event_index","pre_event_signal"]].head().to_dict("records"))
print("recording b first 5:", set_b[["event_index","pre_event_signal"]].head().to_dict("records"))

# ----------------------------------------------------------------- 8 -------
# Summary JSON for the report
summary = {
    "set_a": {k: v for k, v in sa.items()},
    "set_b": {k: v for k, v in sb.items()},
    "unit_distance_n": int(len(m)),
    "unit_dist_range": [float(m["pair_distance"].min()), float(m["pair_distance"].max())],
    "mean_latency_range": [float(m["mean_event_latency"].min()), float(m["mean_event_latency"].max())],
    "zero_sd_units": int((v["event_latency_sd"] == 0).sum()),
    "event_state": event_state,
    "correlations": {
        "dist_vs_meanlat_pearson": {"r": float(r_p), "p": float(p_p)},
        "dist_vs_meanlat_spearman": {"rho": float(rs_p), "p": float(ps_p)},
        "dist_vs_latencysd_pearson": {"r": float(r_s), "p": float(p_s)},
        "dist_vs_latencysd_spearman": {"rho": float(rs_s), "p": float(ps_s)},
        "cross_recording_ks": {"stat": float(ks_stat), "p": float(ks_p)},
    },
    "panel_unit_pairs": pu[["unit_index", "signal_mean", "mean_event_latency",
                            "pair_distance"]].to_dict("records"),
    "robustness": pd.DataFrame(robust).to_dict("records"),
    "pairwise_concordance": float(concordance),
    "contiguous_events": {
        "set_a": bool(sa["index_contiguous"]),
        "set_b": bool(sb["index_contiguous"]),
    },
}
with open(RES / "result_evidence.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("DONE")

#!/usr/bin/env python3
"""GROUP D: parse and characterize time-series / fluorescence / clamp tables.

All mean/SD recomputed from the run columns (pandas, ddof=1). The workbook
Mean/Normalized/S.D. columns are NOT used as evidence. The independent unit
is one run.
"""
import json
import os
import csv
import pandas as pd
import numpy as np

BASE = "/data/user/conversations/conv-4bd502e1-a0d9-48ae-83ce-c85e4cbc094b/project"
DATA = os.path.join(BASE, "data", "source_tables")
OUT = os.path.join(BASE, "output", "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- helpers
def load(f):
    with open(os.path.join(DATA, f), newline="") as fh:
        return [row for row in csv.reader(fh)]

def to_num(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def rnum(row, col):
    return to_num(row[col]) if col < len(row) else None

def describe_rise_plateau(first, last, mx):
    if mx and mx > 0 and last >= 0.90 * mx:
        return f"rises then plateaus (first {first:.3f}, last {last:.3f}, peak {mx:.3f})"
    if last > first * 1.15:
        return f"increases (first {first:.3f} -> last {last:.3f})"
    return f"roughly constant (first {first:.3f}, last {last:.3f})"

def describe_trend(first, last, mn, mx):
    if last > first * 1.15:
        return "rises" if last < 0.95 * mx else "rises then plateaus"
    if last < first * 0.85:
        return "decays"
    return "roughly constant"

errors = []
results = {}

# ----------------------------------------------------------- TABLE 10 -----
try:
    rows = load("source_table_10.csv")
    data = []
    for i in range(4, len(rows)):  # 0-based data rows (task rows 4-54)
        r = rows[i]
        b = rnum(r, 1)
        if b is None:
            continue
        ca = [rnum(r, j) for j in (2, 3, 4)]          # Ca Green run 1-3
        alexa = [rnum(r, j) for j in (10, 11, 12)]    # Alexa 647N run 1-3
        trans = [rnum(r, j) for j in (18, 19, 20)]    # Transmittance run 1-3
        if any(v is None for v in ca + alexa + trans):
            continue
        data.append({"bins": b,
                     "ca_r1": ca[0], "ca_r2": ca[1], "ca_r3": ca[2],
                     "alexa_r1": alexa[0], "alexa_r2": alexa[1], "alexa_r3": alexa[2],
                     "trans_r1": trans[0], "trans_r2": trans[1], "trans_r3": trans[2]})
    df = pd.DataFrame(data)

    def block_summary(cols):
        s = df[cols]
        mean = s.mean(axis=1)
        sd = s.std(axis=1, ddof=1)
        first = float(mean.iloc[0]); last = float(mean.iloc[-1])
        mx = float(mean.max()); mn = float(mean.min())
        return {
            "n_bins": int(len(df)),
            "first_mean": round(first, 4),
            "last_mean": round(last, 4),
            "max_mean": round(mx, 4),
            "min_mean": round(mn, 4),
            "mean_of_means": round(float(mean.mean()), 4),
            "sd_of_means": round(float(sd.mean()), 4),
            "trend": describe_rise_plateau(first, last, mx),
        }

    results["table10"] = {
        "Ca_green": block_summary(["ca_r1", "ca_r2", "ca_r3"]),
        "Alexa_647N": block_summary(["alexa_r1", "alexa_r2", "alexa_r3"]),
        "Transmittance": block_summary(["trans_r1", "trans_r2", "trans_r3"]),
    }
except Exception as e:
    errors.append(f"table10: {type(e).__name__}: {e}")

# ----------------------------------------------------------- TABLE 12 -----
try:
    rows = load("source_table_12.csv")
    # Block B: time=1, SensorA=2, SensorB=3
    # Block C: time=5, Calcium=6, SensorA=7, SensorB=8

    def block_stats(time_col, sensor_cols, calcium_col=None):
        pts = []
        for i in range(4, len(rows)):  # 0-based data rows (task rows 4-1004)
            r = rows[i]
            t = rnum(r, time_col)
            if t is None:
                continue
            vals = {name: rnum(r, col) for name, col in sensor_cols.items()}
            if any(v is None for v in vals.values()):
                continue
            if calcium_col is not None:
                cval = rnum(r, calcium_col)
                if cval is None:
                    continue
                vals["calcium"] = cval
            pts.append((t, vals))
        if not pts:
            return None
        times = [p[0] for p in pts]
        last_t, last_vals = pts[-1]
        t99 = {}
        for name in sensor_cols:
            tv = None
            for t, vals in pts:
                if vals[name] >= 99.0:
                    tv = t
                    break
            t99[name] = round(tv, 4) if tv is not None else None
        return {
            "n_rows": len(pts),
            "time_range": [round(times[0], 4), round(times[-1], 4)],
            "final": {k: round(v, 4) for k, v in last_vals.items()},
            "t99": t99,
        }

    blockB = block_stats(1, {"SensorA": 2, "SensorB": 3})
    blockC = block_stats(5, {"SensorA": 7, "SensorB": 8}, calcium_col=6)
    results["table12"] = {"block_B": blockB, "block_C": blockC}
except Exception as e:
    errors.append(f"table12: {type(e).__name__}: {e}")

# ----------------------------------------------------------- TABLE 13 -----
try:
    rows = load("source_table_13.csv")
    configs = {
        "6FC_single":  {"time": 1,  "SA": 3,  "SB1": 4, "SB2": 5},
        "12FC_single": {"time": 13, "SA": 15, "SB1": 16, "SB2": 17},
        "6FC_dual":    {"time": 25, "SA": 27, "SB1": 28, "SB2": 29},
    }
    table13 = {}
    finals = {}
    for name, cols in configs.items():
        pts = []
        for i in range(7, len(rows)):  # 0-based data rows (task rows 7-274)
            r = rows[i]
            t = rnum(r, cols["time"])
            if t is None:
                continue
            sa = rnum(r, cols["SA"]); sb1 = rnum(r, cols["SB1"]); sb2 = rnum(r, cols["SB2"])
            if sa is None or sb1 is None or sb2 is None:
                continue
            pts.append((t, sa, sb1, sb2))
        if not pts:
            table13[name] = {"final": None, "contrast": "no data"}
            continue
        last_t, sa, sb1, sb2 = pts[-1]
        final = {"SensorA": round(sa, 4),
                 "SensorB_Scheme1": round(sb1, 4),
                 "SensorB_Scheme2": round(sb2, 4)}
        table13[name] = {"final": final, "n_rows": len(pts), "t_final": round(last_t, 4)}
        finals[name] = final

    table13["6FC_single"]["contrast"] = (
        f"6FC single: SensorA and SensorB(Scheme1) hit 100, but SensorB(Scheme2) lags "
        f"at {finals['6FC_single']['SensorB_Scheme2']:.2f} at t={table13['6FC_single']['t_final']}")
    table13["12FC_single"]["contrast"] = (
        f"12FC single: all three signals (SensorA, SensorB Scheme1/2) reach 100 by "
        f"t={table13['12FC_single']['t_final']}")
    table13["6FC_dual"]["contrast"] = (
        f"6FC dual: all three signals (SensorA, SensorB Scheme1/2) reach 100 by "
        f"t={table13['6FC_dual']['t_final']}")
    results["table13"] = table13
except Exception as e:
    errors.append(f"table13: {type(e).__name__}: {e}")

# ----------------------------------------------------------- TABLE 15 -----
try:
    rows = load("source_table_15.csv")
    # 0-based: col1=bins, col2=lipid None, col3=lipid SensorB,
    #          col5=target None, col6=target SensorB
    signals = {"lipid_None": 2, "lipid_SensorB": 3,
               "target_None": 5, "target_SensorB": 6}
    table15 = {}
    for name, col in signals.items():
        vals = []
        for i in range(4, len(rows)):  # 0-based data rows (task rows 4-303)
            r = rows[i]
            v = rnum(r, col)
            if v is None:
                continue
            vals.append(v)
        if not vals:
            table15[name] = {"min": None, "max": None, "trend": "no data"}
            continue
        first, last = vals[0], vals[-1]
        mn, mx = min(vals), max(vals)
        table15[name] = {
            "n": len(vals),
            "min": round(mn, 4),
            "max": round(mx, 4),
            "first": round(first, 4),
            "last": round(last, 4),
            "trend": describe_trend(first, last, mn, mx),
        }
    results["table15"] = table15
except Exception as e:
    errors.append(f"table15: {type(e).__name__}: {e}")

# ------------------------------------------------- write + print ----------
out_path = os.path.join(OUT, "timeseries_stats.json")
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)

print("=" * 72)
print("GROUP D  --  TIME SERIES / FLUORESCENCE / CLAMP  (parse_D.py)")
print("=" * 72)
for k, v in results.items():
    print(f"\n[{k}]")
    for kk, vv in v.items():
        print(f"  {kk}: {json.dumps(vv)}")
if errors:
    print("\nERRORS:")
    for e in errors:
        print("  -", e)
else:
    print("\nNo errors.")
print(f"\nJSON written to: {out_path}")

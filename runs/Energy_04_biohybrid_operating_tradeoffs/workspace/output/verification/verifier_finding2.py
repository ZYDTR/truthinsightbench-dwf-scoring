#!/usr/bin/env python3
"""
Independent verifier for Finding 2 (sweep hysteresis) in output/Result.md.

Independently recomputes, from data/time_response_curves.tsv ONLY:
  1. matched-time branch structure in rows 259-293 (1-based, = df index 258-292)
  2. raw values at t=2 and desc/asc ratios (+ jackknife over desc replicates)
  3. which times have >1 row in the sweep, and on which branches
  4. loop areas on log-time grid [50..9000] with own log10 interpolation + trapezoid
  5. loop height / full channel max
  6. descending/ascending peak ratios (reversal check for condition_b)
  7. apex locations and time difference
  8. alternative model: OLS of channel on log10(time) on ascending branch only,
     mean residual on descending branch (normal equations + grid search cross-check)

Methods deliberately differ from the obvious ones where possible.
No import of output/analysis/analysis.py.
"""
import json
import math
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
TSV = f"{BASE}/data/time_response_curves.tsv"

df = pd.read_csv(TSV, sep="\t")
assert df.shape == (1367, 7), df.shape

# Report uses 0-BASED positional row numbers (verified: "row 260" = df.iloc[260]
# = ascending t=2 row with prom=3, cond_a=4, cond_b=43). The sweep window
# 259..293 (0-based, inclusive) = 35 rows, df.iloc[259:294].
SW = df.iloc[259:294, :].reset_index(drop=True)
assert len(SW) == 35

channels = ["promoter_like_signal", "condition_a", "condition_b"]

# ---- branch definition as stated in the report ------------------------------
asc = SW.iloc[0:9]    # rows 259-267  (df idx 259-267), time rising 0 -> 10083
desc = SW.iloc[9:25]  # rows 268-283  (df idx 268-283), time falling 9643 -> 6
tail = SW.iloc[25:35]  # rows 284-293

out = {}
print("=" * 78)
print("BRANCHES (1-based data rows)")
print("=" * 78)
for name, br in [("asc", asc), ("desc", desc), ("tail", tail)]:
    print(f"{name}: n={len(br)} rows, time {br.time.iloc[0]}..{br.time.iloc[-1]}, "
          f"monotone_incr={bool((br.time.diff().dropna() > 0).all())}, "
          f"monotone_decr={bool((br.time.diff().dropna() < 0).all())}")

# ---------------------------------------------------------------------------
# (1)+(2) matched t=2 raw values and ratios, and full multiplicity audit
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(1) MULTIPLICITY: times with >1 row in the sweep, split by branch")
print("=" * 78)
mult = {}
for name, br in [("ascending(259-267)", asc), ("descending(268-283)", desc),
                 ("tail(284-293)", tail)]:
    vc = br.time.value_counts()
    multi = {int(t): int(c) for t, c in vc.items() if c > 1}
    mult[name] = multi
    print(f"{name}: times with >1 row -> {multi}")
allmulti = SW.time.value_counts()
allmulti = {int(t): int(c) for t, c in allmulti.items() if c > 1}
print(f"whole sweep: times with >1 row -> {allmulti}")
out["multiplicity_by_branch"] = mult
out["multiplicity_whole_sweep"] = allmulti

# times appearing on BOTH ascending branch and any later row
asc_times = set(asc.time.tolist())
desc_times = set(desc.time.tolist())
tail_times = set(tail.time.tolist())
both_ad = sorted(asc_times & desc_times)
both_at = sorted(asc_times & tail_times)
print(f"times on both asc & desc branches: {both_ad}")
print(f"times on both asc branch & tail : {both_at}")
out["matched_times_asc_desc"] = both_ad
out["matched_times_asc_tail"] = both_at

# descending t=2 replicates = rows 284, 286, 288 (1-based) -> SW index 25,27,29
desc_t2_rows_1based = [284, 286, 288]
asc_t2_row_1based = [260]
idx = lambda r: r - 259  # 1-based data row -> SW positional index
asc_t2 = SW.iloc[[idx(r) for r in asc_t2_row_1based]]
desc_t2 = SW.iloc[[idx(r) for r in desc_t2_rows_1based]]
print("\nascending t=2 row(s):")
print(asc_t2.to_string(index=False))
print("descending t=2 replicates:")
print(desc_t2.to_string(index=False))

print("\n" + "=" * 78)
print("(2) MATCHED t=2 RAW VALUES AND DESC/ASC RATIOS")
print("=" * 78)
res_t2 = {}
for ch in channels:
    a = float(asc_t2[ch].iloc[0])
    d = [float(x) for x in desc_t2[ch].tolist()]
    ratios = [x / a for x in d]
    print(f"{ch:22s} asc={a:g}  desc={d}  ratios={[round(r,2) for r in ratios]}  "
          f"range={min(ratios):.2f}..{max(ratios):.2f}")
    res_t2[ch] = {"asc": a, "desc": d, "ratios": ratios,
                  "ratio_min": min(ratios), "ratio_max": max(ratios)}
out["matched_t2"] = res_t2

# ---------------------------------------------------------------------------
# (3) loop area on log-time grid, OWN interpolation + OWN trapezoid
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(3) LOOP AREA on log10 grid [50,100,200,400,800,1600,3200,6400,9000]")
print("=" * 78)
grid = [50, 100, 200, 400, 800, 1600, 3200, 6400, 9000]
lg = np.log10(grid)

def interp_loglog(br, ch):
    """Linear interpolation of value vs log10(time). Sorted by log-time, then
    np.interp (piecewise linear). Time==0 rows cannot be log-interpolated and
    are irrelevant (both branches are strictly positive on 2..10083)."""
    x = np.log10(br.time.to_numpy(dtype=float))
    y = br[ch].to_numpy(dtype=float)
    o = np.argsort(x)
    return np.interp(lg, x[o], y[o])

def trapz_own(x, y):
    """Own trapezoid rule (also cross-checked against np.trapezoid)."""
    s = 0.0
    for i in range(len(x) - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s

loop = {}
for ch in channels:
    up = interp_loglog(asc, ch)
    dn = interp_loglog(desc, ch)
    diff = dn - up
    area = trapz_own(lg, diff)
    area_np = float(np.trapezoid(diff, lg))
    cmax = float(df[ch].max())
    frac = area / cmax
    print(f"{ch:22s} area={area:.4f}  area_np.trapezoid={area_np:.4f}  "
          f"global_max={cmax:g}  area/max={frac:.6f}")
    print(f"    up  ={np.round(up,1).tolist()}")
    print(f"    dn  ={np.round(dn,1).tolist()}")
    print(f"    dn-up={np.round(diff,1).tolist()}  all_positive={bool((diff>0).all())}")
    loop[ch] = {"area": area, "area_np": area_np, "global_max": cmax,
                "area_over_max": frac, "up": up.tolist(), "dn": dn.tolist(),
                "diff": diff.tolist()}
out["loop_logtime_grid"] = loop

# ---------------------------------------------------------------------------
# (4) peak ratios desc/asc (raw, no interpolation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(4) PEAK RATIOS (raw branch maxima)")
print("=" * 78)
peaks = {}
for ch in channels:
    pa, pd_ = float(asc[ch].max()), float(desc[ch].max())
    print(f"{ch:22s} asc_peak={pa:g} desc_peak={pd_:g} ratio={pd_/pa:.4f}")
    peaks[ch] = {"asc_peak": pa, "desc_peak": pd_, "ratio": pd_ / pa}
out["peak_ratios"] = peaks

# ---------------------------------------------------------------------------
# (5) apex locations
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(5) APEX LOCATIONS within sweep (argmax over all 35 rows)")
print("=" * 78)
apex = {}
for ch in channels:
    i = int(np.argmax(SW[ch].to_numpy()))
    row1 = 259 + i
    t = int(SW.time.iloc[i])
    v = float(SW[ch].iloc[i])
    apex[ch] = {"row_1based": row1, "time": t, "value": v}
    print(f"{ch:22s} apex row={row1} (SW idx {i}, df idx {258+i})  time={t}  value={v:g}")
print(f"time difference prom/cond_a apex vs condition_b apex = "
      f"{apex['condition_b']['time'] - apex['promoter_like_signal']['time']}")
out["apex"] = apex

# ---------------------------------------------------------------------------
# (6) jackknife over the three matched t=2 descending replicates
# (leaving one OUT: ratio computed on remaining two -> a range of 3 values)
# Also report the report's presumed reading (dropping one row from the FULL
# three-replicate ratio set, i.e. mean of the other two) for completeness.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(6) JACKKNIFE over t=2 descending replicates (rows 284/286/288)")
print("=" * 78)
jk = {}
for ch in channels:
    a = float(asc_t2[ch].iloc[0])
    d = np.array([float(x) for x in desc_t2[ch].tolist()])
    loo = [float(np.delete(d, i).mean() / a) for i in range(3)]
    indent = [float(d[i] / a) for i in range(3)]
    print(f"{ch:22s} leave-one-out ratio range = {min(loo):.3f}..{max(loo):.3f} "
          f"(values {[round(x,3) for x in loo]})")
    print(f"{'':22s} individual ratios        = {[round(x,3) for x in indent]}")
    jk[ch] = {"loo_range": [min(loo), max(loo)], "loo": loo, "individual": indent}
out["jackknife_t2"] = jk

# ---------------------------------------------------------------------------
# (7) alternative model: OLS on ascending branch, predict descending branch
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(7) ALTERNATIVE MODEL: channel ~ a + b*log10(time), fit on ASC only")
print("=" * 78)
alt = {}
for ch in channels:
    x = np.log10(asc.time.to_numpy(dtype=float))
    y = asc[ch].to_numpy(dtype=float)
    # normal equations by hand
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a0 = (sy - b * sx) / n
    # grid-search cross-check of slope
    bs = np.linspace(b - 2, b + 2, 200001)
    a_grid = (sy - bs * sx) / n
    sse = ((y[None, :] - (a_grid[:, None] + bs[:, None] * x[None, :])) ** 2).sum(axis=1)
    b_grid = float(bs[int(np.argmin(sse))])
    # residuals on descending branch (log-time is well-defined for desc times)
    xd = np.log10(desc.time.to_numpy(dtype=float))
    pred = a0 + b * xd
    resid = desc[ch].to_numpy(dtype=float) - pred
    mean_resid = float(resid.mean())
    print(f"{ch:22s} slope={b:.6f} (grid={b_grid:.6f})  intercept={a0:.4f}  "
          f"mean_desc_resid={mean_resid:.4f}")
    alt[ch] = {"slope": float(b), "slope_grid": b_grid, "intercept": float(a0),
               "mean_desc_resid": mean_resid, "resid": resid.tolist()}
out["alt_model"] = alt

# ---------------------------------------------------------------------------
# (8) extra adversarial checks not explicitly requested
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(8) EXTRA ADVERSARIAL CHECKS")
print("=" * 78)
# (a) if the grid is taken as 100..10000 instead of 50..9000
grid2 = [100, 200, 400, 800, 1600, 3200, 6400, 9000]
lg2 = np.log10(grid2)
def interp_loglog2(br, ch):
    x = np.log10(br.time.to_numpy(dtype=float))
    y = br[ch].to_numpy(dtype=float)
    o = np.argsort(x)
    return np.interp(lg2, x[o], y[o])
print("grid-upper-edge sensitivity (area with grid 100..9000):")
for ch in channels:
    up = interp_loglog2(asc, ch); dn = interp_loglog2(desc, ch)
    print(f"    {ch:22s} area={trapz_own(lg2, dn-up):.2f}  vs {loop[ch]['area']:.2f}")
# (b) areas with grid 50..10000
grid3 = grid + [10000]
lg3 = np.log10(grid3)
for ch in channels:
    x = np.log10(asc.time.to_numpy(float)); y = asc[ch].to_numpy(float)
    o = np.argsort(x); up = np.interp(lg3, x[o], y[o])
    x = np.log10(desc.time.to_numpy(float)); y = desc[ch].to_numpy(float)
    o = np.argsort(x); dn = np.interp(lg3, x[o], y[o])
    print(f"    {ch:22s} area(grid 50..10000)={trapz_own(lg3, dn-up):.2f}")
# (c) count of >1-row times in whole file and in tail to check sweep uniqueness
print("times with >1 row in whole file (top 10 by count):",
      df.time.value_counts().head(10).to_dict())
# (d) how many rows would change if 1-based vs 0-based row numbering -> direct check
print("row 260 (1-based) time =", df.time.iloc[259],
      "| row 270 (1-based) time =", df.time.iloc[269],
      "| row 267 (1-based) time =", df.time.iloc[266],
      "| row 284 (1-based) time =", df.time.iloc[283])
# (e) fraction of asc branch strictly above desc on raw rows at matched times
for t in sorted(both_ad):
    av = asc[asc.time == t]
    dv = desc[desc.time == t]
    print(f"t={t}: asc vals {av[channels].values.tolist()} vs desc vals "
          f"{dv[channels].values.tolist()}")
# (f) loop area normalised by channel range (max-min) as an alternative
print("loop height fraction using (max-min) instead of max:")
for ch in channels:
    rng = float(df[ch].max() - df[ch].min())
    print(f"    {ch:22s} area/range = {loop[ch]['area']/rng:.6f}")

with open(f"{BASE}/output/verification/verifier_finding2_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote", f"{BASE}/output/verification/verifier_finding2_results.json")
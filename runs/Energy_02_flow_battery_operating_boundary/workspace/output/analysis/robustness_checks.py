#!/usr/bin/env python3
"""Leave-out / alternative-definition robustness checks for each registered finding.

Produces output/results/robustness_*.csv and output/results/robustness_summary.json.

Run:  python3 output/analysis/robustness_checks.py
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RES = ROOT / "output" / "results"
RES.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(31337)
SET = list(range(1, 8))
S = {}


def jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def wjson(name, payload):
    (RES / name).write_text(json.dumps(payload, indent=2, default=jd), encoding="utf-8")
    print(f"[write] output/results/{name}")


def wcsv(name, header, rows):
    with (RES / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print(f"[write] output/results/{name}")


# ---- load ---------------------------------------------------------------
g = collections.defaultdict(list)
with (DATA / "rate_dependent_voltage_curves.tsv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        g[(r["scan_role"], r["anonymous_rate_setting"])].append(
            (float(r["state_of_charge_percent"]), float(r["cell_voltage_V"])))
C = {}
for k, v in g.items():
    v.sort(); C[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
eff, pw = {}, {}
with (DATA / "rate_tradeoff_summaries.tsv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        (eff if r["summary_role"] == "efficiency" else pw)[float(r["anonymous_rate_value"])] = float(r["response_value"])
cyc = np.array([float(r["capacity_like_value"]) for r in csv.DictReader(
    (DATA / "long_cycle_capacity.tsv").open(encoding="utf-8"), delimiter="\t")])
CYC_IDX = np.array([int(r["cycle_index"]) for r in csv.DictReader(
    (DATA / "long_cycle_capacity.tsv").open(encoding="utf-8"), delimiter="\t")])
assert len(CYC_IDX) == len(cyc), "cycle_index/capacity length mismatch"
RE, RP = sorted(eff), sorted(pw)

# =========================================================================
print("=" * 78); print("ROBUSTNESS A -- leave-one-setting-out for the ordering claim"); print("=" * 78)
rows = []
for drop in SET:
    keep = [i for i in SET if i != drop]
    for soc in (30, 50, 70):
        c = np.array([float(np.interp(soc, *C[("charge_like", f"rate_setting_{i}")])) for i in keep])
        d = np.array([float(np.interp(soc, *C[("discharge_like", f"rate_setting_{i}")])) for i in keep])
        gp = c - d
        rows.append([drop, soc, bool(np.all(np.diff(c) > 0)), bool(np.all(np.diff(d) < 0)),
                     bool(np.all(np.diff(gp) > 0)), float(gp[0]), float(gp[-1]), float(gp[-1] / gp[0])])
wcsv("robustness_leave_one_setting_out.csv",
     ["dropped_setting", "soc_probe", "charge_strict_inc", "discharge_strict_dec",
      "gap_strict_inc", "gap_first_V", "gap_last_V", "gap_growth_factor"], rows)
ok = sum(1 for r in rows if r[2] and r[3] and r[4])
print(f"  {ok}/{len(rows)} leave-one-setting-out configurations preserve all three orderings")
for r in rows:
    if not (r[2] and r[3] and r[4]):
        print(f"    FAIL when dropping setting {r[0]} at SoC={r[1]}")
S["A_leave_one_setting_out"] = dict(configs=len(rows), all_orderings_hold=ok)

# alternative definition of the gap: area between branches on the shared window
print("  alternative gap definition (SoC-integrated branch separation, 5-95%):")
alt = []
grid = np.linspace(5, 95, 400)
for i in SET:
    vc = np.interp(grid, *C[("charge_like", f"rate_setting_{i}")])
    vd = np.interp(grid, *C[("discharge_like", f"rate_setting_{i}")])
    alt.append(float(np.mean(vc - vd)))
print(f"    integrated gaps = {np.round(alt,4).tolist()}")
print(f"    strictly increasing: {bool(np.all(np.diff(alt) > 0))}  "
      f"growth factor {alt[-1]/alt[0]:.3f}")
S["A_alternative_gap_definition"] = dict(values=alt, strictly_increasing=bool(np.all(np.diff(alt) > 0)),
                                         growth_factor=float(alt[-1] / alt[0]))

# =========================================================================
print(); print("=" * 78)
print("ROBUSTNESS B -- linkage: how sharp is the SoC=50 optimum, exactly?"); print("=" * 78)
fine = []
for soc in np.arange(40.0, 60.001, 0.05):
    e = np.array([100 * float(np.interp(soc, *C[("discharge_like", f"rate_setting_{i}")])) /
                  float(np.interp(soc, *C[("charge_like", f"rate_setting_{i}")])) - eff[r]
                  for r, i in zip(RE, SET)])
    v = np.array([pw[r] / r - float(np.interp(soc, *C[("discharge_like", f"rate_setting_{i}")]))
                  for r, i in zip(RP[:7], SET)])
    fine.append([float(soc), float(np.sqrt((e ** 2).mean())), float(np.sqrt((v ** 2).mean())),
                 float(np.abs(e).max())])
wcsv("robustness_soc_optimum_fine.csv", ["soc", "rmse_eff_pp", "rmse_V_V", "max_abs_eff_res_pp"], fine)
arr = np.array(fine)
ib = int(np.argmin(arr[:, 1] + arr[:, 2]))
print(f"  fine scan minimum at SoC={arr[ib,0]:.2f} (RMSE_eff={arr[ib,1]:.4f} pp)")
i50 = int(np.argmin(np.abs(arr[:, 0] - 50.0)))
print(f"  at SoC=50.00 exactly: RMSE_eff={arr[i50,1]:.4f} pp, RMSE_V={arr[i50,2]:.5f} V")
# where does the fit degrade to the energy-weighted alternative?
target = next(r for r in fine if abs(r[0] - 50.0) < 1e-9)
band = [r[0] for r in fine if r[1] <= 1.0]
print(f"  SoC values with RMSE_eff <= 1.0 pp: {min(band):.2f} to {max(band):.2f}")
S["B_soc_optimum"] = dict(fine_min_soc=arr[ib, 0], fine_min_rmse_eff_pp=arr[ib, 1],
                          rmse_eff_at_50_pp=target[1], rmse_V_at_50=target[2],
                          band_rmse_le_1pp=[min(band), max(band)])
# nearest measured abscissa to 50 in each file
for role in ("charge_like", "discharge_like"):
    for i in (1, 4, 7):
        x = C[(role, f"rate_setting_{i}")][0]
        j = int(np.argmin(np.abs(x - 50)))
        print(f"  nearest measured x to 50 in {role} s{i}: x={x[j]:.5f} (offset {x[j]-50:+.5f})")
S["B_nearest_measured_to_50"] = {
    f"{role}_s{i}": float(C[(role, f"rate_setting_{i}")][0][int(np.argmin(np.abs(C[(role, f'rate_setting_{i}')][0] - 50)))] - 50)
    for role in ("charge_like", "discharge_like") for i in (1, 4, 7)}

# leave-one-rate-out, both series
loro = []
for k in range(7):
    m = np.ones(7, bool); m[k] = False
    e = np.array([100 * float(np.interp(50, *C[("discharge_like", f"rate_setting_{i}")])) /
                  float(np.interp(50, *C[("charge_like", f"rate_setting_{i}")])) - eff[r]
                  for r, i in zip(np.array(RE)[m], np.array(SET)[m])])
    v = np.array([pw[r] / r - float(np.interp(50, *C[("discharge_like", f"rate_setting_{i}")]))
                  for r, i in zip(np.array(RP[:7])[m], np.array(SET)[m])])
    loro.append([RE[k], float(np.sqrt((e ** 2).mean())), float(np.abs(e).max()),
                 float(np.sqrt((v ** 2).mean())), float(np.abs(v).max())])
wcsv("robustness_linkage_leave_one_rate_out.csv",
     ["held_out_rate", "rmse_eff_pp", "max_abs_eff_pp", "rmse_V", "max_abs_V"], loro)
print(f"  leave-one-rate-out efficiency RMSE range: "
      f"{min(r[1] for r in loro):.4f}-{max(r[1] for r in loro):.4f} pp")
S["B_leave_one_rate_out"] = dict(rmse_eff_range_pp=[min(r[1] for r in loro), max(r[1] for r in loro)],
                                 rmse_V_range=[min(r[3] for r in loro), max(r[3] for r in loro)])

# alternative definition: use the measured point nearest to 50 instead of interpolation
alt_eff, alt_v = [], []
for r, i in zip(RE, SET):
    xd, yd = C[("discharge_like", f"rate_setting_{i}")]
    xc, yc = C[("charge_like", f"rate_setting_{i}")]
    vd = float(yd[int(np.argmin(np.abs(xd - 50)))]); vc = float(yc[int(np.argmin(np.abs(xc - 50)))])
    alt_eff.append(100 * vd / vc - eff[r])
for r, i in zip(RP[:7], SET):
    xd, yd = C[("discharge_like", f"rate_setting_{i}")]
    alt_v.append(pw[r] / r - float(yd[int(np.argmin(np.abs(xd - 50)))]))
alt_eff = np.array(alt_eff); alt_v = np.array(alt_v)
print(f"  nearest-measured-point definition: RMSE_eff={np.sqrt((alt_eff**2).mean()):.4f} pp, "
      f"RMSE_V={np.sqrt((alt_v**2).mean()):.5f} V")
S["B_nearest_point_definition"] = dict(rmse_eff_pp=float(np.sqrt((alt_eff ** 2).mean())),
                                       rmse_V=float(np.sqrt((alt_v ** 2).mean())))

# =========================================================================
print(); print("=" * 78)
print("ROBUSTNESS C -- cycling: leave-out and alternative-outcome checks"); print("=" * 78)
x = np.arange(1, 601, dtype=float)


def slope(y, block=25, reps=1500):
    a, b, r = (lambda c: (c[0], c[1], y - np.vstack([x, np.ones(600)]).T @ c))(
        np.linalg.lstsq(np.vstack([x, np.ones(600)]).T, y, rcond=None)[0])
    starts = np.arange(0, 600 - block + 1)
    out = np.empty(reps)
    for k in range(reps):
        ss = RNG.choice(starts, int(np.ceil(600 / block)))
        idx = np.concatenate([np.arange(t, t + block) for t in ss])[:600]
        out[k] = np.linalg.lstsq(np.vstack([x[idx], np.ones(600)]).T, y[idx], rcond=None)[0][0]
    return a, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


a, lo, hi = slope(cyc)
print(f"  all 600 cycles          : slope={a:+.5f} CI=[{lo:+.5f},{hi:+.5f}] excludes0={not(lo<=0<=hi)}")
S["C_full"] = dict(slope=a, ci=[lo, hi], excludes_zero=bool(not (lo <= 0 <= hi)))
# Outlier treatment must preserve time order; trimming the sorted values would
# scramble the trend and produce a meaningless positive slope.
_cl, *_ = np.linalg.lstsq(np.vstack([x, np.ones(600)]).T, cyc, rcond=None)
_rl = cyc - np.vstack([x, np.ones(600)]).T @ _cl
_sd = _rl.std()
_keep = np.abs(_rl) < 2 * _sd
# NOTE (post-verification correction): the previous version re-used
# np.arange(1, len(y)+1) as the abscissa after subsampling.  For the odd/even
# variants that halves the index spacing, so the fitted slope came out exactly
# 2x the true per-cycle slope and the reported "per cycle" range was inflated.
# Each variant now carries its true cycle_index as its abscissa.
_odd_i, _even_i = CYC_IDX[0::2], CYC_IDX[1::2]
variants = {
    "drop odd cycles": (cyc[1::2], _even_i),
    "drop even cycles": (cyc[0::2], _odd_i),
    "drop first 50": (cyc[50:], CYC_IDX[50:]),
    "drop last 50": (cyc[:550], CYC_IDX[:550]),
    "first half only": (cyc[:300], CYC_IDX[:300]),
    "second half only": (cyc[300:], CYC_IDX[300:]),
    "drop |residual|>2sd (order kept)": (cyc[_keep], CYC_IDX[_keep]),
}
rows = []
for name, (y, xreal) in variants.items():
    xx = xreal.astype(float)          # true cycle index, not a re-indexed 1..n
    aa = np.linalg.lstsq(np.vstack([xx, np.ones(len(y))]).T, y, rcond=None)[0][0]
    blk = 25
    starts = np.arange(0, len(y) - blk + 1)
    out = np.empty(1200)
    for k in range(1200):
        ss = RNG.choice(starts, int(np.ceil(len(y) / blk)))
        idx = np.concatenate([np.arange(t, t + blk) for t in ss])[:len(y)]
        out[k] = np.linalg.lstsq(np.vstack([xx[idx], np.ones(len(y))]).T, y[idx], rcond=None)[0][0]
    l2, h2 = np.percentile(out, [2.5, 97.5])
    rows.append([name, len(y), aa, l2, h2, bool(not (l2 <= 0 <= h2)),
                 float(y[-1] / y[0] * 100)])
    print(f"  {name:34s}: n={len(y):3d} slope={aa:+.5f} CI=[{l2:+.5f},{h2:+.5f}] "
          f"excludes0={str(not (l2 <= 0 <= h2)):5s} ret={y[-1]/y[0]*100:.2f}%")
wcsv("robustness_cycling_variants.csv",
     ["variant", "n_cycles", "slope_per_cycle", "ci_lo", "ci_hi", "ci_excludes_zero",
      "endpoint_retention_pct"], rows)
S["C_variants"] = [dict(zip(["variant", "n", "slope", "lo", "hi", "excl0", "ret"], r)) for r in rows]
n_excl = sum(1 for r in rows if r[5])
print(f"  {n_excl}/{len(rows)} cyclic-section variants resolve a non-zero slope")

# alternative-outcome check: is any sub-window consistent with a stable plateau?
seg = list(csv.DictReader((RES / "cycling_segments.csv").open(), delimiter=","))
plateau = [r["cycle_from"] for r in seg if r["ci_contains_zero"] == "True"]
print(f"  windows whose 95% CI includes zero slope: {len(plateau)}/12 -> "
      f"{'no unique stable window identifiable' if len(plateau) > 1 else 'one stable window'}")
S["C_plateau_windows"] = plateau

# =========================================================================
print(); print("=" * 78)
print("ROBUSTNESS D -- efficiency model: bootstrap the convexity"); print("=" * 78)
EE = np.array([eff[r] for r in RE]); LE = np.log10(np.array(RE, float))
resid = EE - np.vstack([LE, np.ones(7)]).T @ np.linalg.lstsq(np.vstack([LE, np.ones(7)]).T, EE, rcond=None)[0]
boot = np.empty(4000)
for b in range(4000):
    idx = RNG.integers(0, 7, 7)
    yy = EE[idx]
    if len(set(idx.tolist())) < 3:
        boot[b] = np.nan; continue
    A = np.vstack([LE[idx] ** 2, LE[idx], np.ones(7)]).T
    boot[b] = np.linalg.lstsq(A, yy, rcond=None)[0][0]
boot = boot[~np.isnan(boot)]
print(f"  quadratic coefficient: point estimate "
      f"{np.linalg.lstsq(np.vstack([LE**2,LE,np.ones(7)]).T, EE, rcond=None)[0][0]:.4f}, "
      f"bootstrap 95% CI [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
print(f"  fraction of bootstrap replicates with negative curvature: {(boot<0).mean():.4f}")
S["D_quadratic_bootstrap"] = dict(point=float(np.linalg.lstsq(np.vstack([LE ** 2, LE, np.ones(7)]).T, EE, rcond=None)[0][0]),
                                  ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                  frac_negative=float((boot < 0).mean()))
# leave-one-rate-out on monotone steepening
steep_ok = 0
for k in range(7):
    m = np.ones(7, bool); m[k] = False
    e = EE[m]; l = LE[m]
    sl = [(e[j + 1] - e[j]) / (l[j + 1] - l[j]) for j in range(len(e) - 1)]
    steep_ok += int(np.all(np.diff(sl) < 0))
print(f"  monotone steepening holds in {steep_ok}/7 leave-one-rate-out subsets")
S["D_steepening_leave_one_out"] = dict(holds=steep_ok, of=7)

# =========================================================================
print(); print("=" * 78)
print("ROBUSTNESS E -- power turnover under alternative peak definitions"); print("=" * 78)
RPf = np.array(RP, float); PP = np.array([pw[r] for r in RP], float); VV = PP / RPf
alts = {
    "raw argmax (all 9 points)": int(np.argmax(PP)),
    "exclude rate 300": int(np.argmax(np.where(RPf == 300, -np.inf, PP))),
    "exclude pure log-line extrapolation": None,
    "implied-voltage intersection with 0.99 V floor": int(np.argmax(RPf[VV >= 0.99])),
}
rows = []
for name, idx in alts.items():
    if idx is None:
        continue
    rows.append([name, float(RPf[idx]), float(PP[idx])])
    print(f"  {name:46s} -> peak rate {RPf[idx]:.0f} (P={PP[idx]:.2f})")
wcsv("robustness_power_peak_definitions.csv", ["definition", "peak_rate", "peak_power_density"], rows)
S["E_peak_definitions"] = rows
S["E_requires_rate_300"] = dict(stated_peak=330.0, peak_if_300_removed=float(PP[RPf != 300].max()))
print(f"  the turnover's interior maximum depends on rate 300: without it the maximum is the "
      f"rate-200 endpoint value {PP[RPf != 300].max():.2f}")

wjson("robustness_summary.json", S)
print(); print("robustness_summary.json written")
#!/usr/bin/env python3
"""Independent adversarial verification of Draft Finding 1 (claim_id draft-finding-1).

Written from scratch; imports nothing from output/analysis/.
All results printed here come from data/observations.tsv at runtime.

Key idea used throughout: the columns live on exact decimal lattices
(c01 on 1 mm, c03 on 0.01 cm, c04 on 0.001 ug/kg, c05 on 0.001 g/cm3,
c06 on 0.01 m), so several relations can be tested in EXACT integer
arithmetic instead of floating point.
"""
import json
import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv"
OUT = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/output/verification/figures"

df = pd.read_csv(DATA, sep="\t")
n = len(df)
R = {}
out = []


def rec(key, s):
    print(("[%s] " % key) + str(s))
    out.append("[%s] %s" % (key, s))
    R[key] = s


# exact integer representations (verify the lattices first)
c1_mm = np.round(df.c01_depth_top_m.values * 1000).astype(np.int64)          # mm
assert np.abs(df.c01_depth_top_m.values * 1000 - c1_mm).max() < 1e-9
c2_mm = np.round(df.c02_depth_ice_snow_m.values * 1000).astype(np.int64)     # mm
assert np.abs(df.c02_depth_ice_snow_m.values * 1000 - c2_mm).max() < 1e-9
c3_u = np.round(df.c03_samp_thick_cm.values * 100).astype(np.int64)          # 1e-4 m units
assert np.abs(df.c03_samp_thick_cm.values * 100 - c3_u).max() < 1e-9
c5_mg = np.round(df.c05_density_ice_g_cm_3.values * 1000).astype(np.int64)   # 1e-3 g/cm3
assert np.abs(df.c05_density_ice_g_cm_3.values * 1000 - c5_mg).max() < 1e-9
c6_cm = np.round(df.c06_depth_w_e_m.values * 100).astype(np.int64)           # cm = 1e-2 m
assert np.abs(df.c06_depth_w_e_m.values * 100 - c6_cm).max() < 1e-9
rec("lattices", "c01 mm-grid ok, c02 mm-grid ok, c03 0.01cm-grid ok, c05 0.001 g/cm3 grid ok, c06 cm-grid ok")

# =============================================================== (a)
d21_mm = c2_mm - c1_mm
vals, cnts = np.unique(d21_mm, return_counts=True)
rec("a", "c02-c01 in mm units: values=%s counts=%s ; max=%d mm ; n>20mm=%d"
    % (vals.tolist(), cnts.tolist(), np.abs(d21_mm).max(), int((np.abs(d21_mm) > 20).sum())))
# exact decimal 0.02 m = 20 mm inclusive -> passes; float artefact made 0.0200000000000031
rec("a", "float (c02-c01) max = %.17g (0.02 in float = %.17g)" %
    ((df.c02_depth_ice_snow_m.values - df.c01_depth_top_m.values).max(), 0.02))
# is the 0/10/20 pattern a function of c01?
for delta in vals.tolist():
    m = d21_mm == delta
    rec("a", "delta=%dmm rows=%d c01 range=[%.2f, %.2f] first three c01=%s"
        % (delta, m.sum(), df.c01_depth_top_m.values[m].min(), df.c01_depth_top_m.values[m].max(),
           np.round(df.c01_depth_top_m.values[m][:3], 3).tolist()))

# =============================================================== (b)
m8 = df.c08_age_a_ad_ce.notna().values
m9 = df.c09_age_ka_bp.notna().values
rec("b", "non-null: c08=%d c09=%d identical mask=%s" % (m8.sum(), m9.sum(), bool(np.all(m8 == m9))))
# exact integer test: c09 on 0.001 ka lattice, c08 integer years
c8_i = np.round(df.c08_age_a_ad_ce.values[m8]).astype(np.int64)
c9_mka = np.round(df.c09_age_ka_bp.values[m9] * 1000).astype(np.int64)
rec("b", "c08 integer years? max dev=%.1e ; c09 0.001-ka lattice? max dev=%.1e"
    % (np.abs(df.c08_age_a_ad_ce.values[m8] - c8_i).max(),
       np.abs(df.c09_age_ka_bp.values[m9] * 1000 - c9_mka).max()))
rec("b", "exact integer identity c8_i + c9_mka == 1950 : %s ; unique c08+c09*1000 = %s"
    % (bool(np.all(c8_i + c9_mka == 1950)), np.unique(c8_i + c9_mka).tolist()))
rec("b", "residual (1950-c08)/1000 - c09: max|.|=%.3e over %d rows"
    % (np.abs((1950 - df.c08_age_a_ad_ce.values[m8]) / 1000 - df.c09_age_ka_bp.values[m9]).max(), m8.sum()))

# =============================================================== (c) exact integer
Rint = c1_mm.astype(np.int64) * 10 - np.cumsum(c3_u)     # units of 1e-4 m (0.1 mm); integers
rec("c", "exact integer R_i = c01(0.1mm) - cumsum(c03)(0.1mm): min=%d max=%d distinct=%d"
    % (Rint.min(), Rint.max(), len(np.unique(Rint))))
dR = np.diff(Rint)
jumps = np.where(dR != 0)[0]
big_jump = int(np.argmax(np.abs(dR)))
rec("c", "dR=dc01-c03 in 0.1mm units: values=%s (n rows each)"
    % {int(v): int((dR == v).sum()) for v in np.unique(dR)})
rec("c", "largest dR at row->row %d->%d : value=%d units = %.4f m ; c01[%d]=%.3f c01[%d]=%.3f c03[%d]=%.4f"
    % (big_jump, big_jump + 1, dR[big_jump], dR[big_jump] / 10000.0,
       big_jump, df.c01_depth_top_m.values[big_jump], big_jump + 1, df.c01_depth_top_m.values[big_jump + 1],
       big_jump + 1, df.c03_samp_thick_cm.values[big_jump + 1]))
# what fraction of rows have dR == typical thickness mismatch?
rec("c", "c03 distinct (0.01cm ints) = %s" % np.unique(c3_u).tolist())
rec("c", "dc01 in mm: distinct=%s counts_top=%s"
    % (np.unique(np.diff(c1_mm)).tolist(),
       {int(v): int((np.diff(c1_mm) == v).sum()) for v in np.unique(np.diff(c1_mm))}))
# plateau statistics in exact units
bp = big_jump + 1
rec("c", "before jump rows [0,%d]: R range=[%d,%d] units (%.4f m), sd=%.6f m"
    % (bp - 1, Rint[:bp].min(), Rint[:bp].max(),
       (Rint[:bp].max() - Rint[:bp].min()) / 10000.0, Rint[:bp].std(ddof=0) / 10000.0))
rec("c", "after jump rows [%d,%d]: R range=[%d,%d] units (%.4f m), sd=%.6f m"
    % (bp, n - 1, Rint[bp:].min(), Rint[bp:].max(),
       (Rint[bp:].max() - Rint[bp:].min()) / 10000.0, Rint[bp:].std(ddof=0) / 10000.0))
# claim: c01 = cumsum(c03)/100 + const + step.  best float fit with free slope/intercept/step
cum = np.cumsum(df.c03_samp_thick_cm.values) / 100.0
y = df.c01_depth_top_m.values
best = None
for bp_try in range(50, 160):
    X = np.vstack([cum, (np.arange(n) >= bp_try).astype(float), np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    m = np.abs(r).max()
    if best is None or m < best[0]:
        best = (m, bp_try, coef)
rec("c", "free breakpoint scan rows 50..159: best bp=%d max|res|=%.6f m coef(slope,step,intercept)=(%.9f, %.9f, %.9f)"
    % (best[1], best[0], best[2][0], best[2][1], best[2][2]))
X = np.vstack([cum, (np.arange(n) >= 106).astype(float), np.ones(n)]).T
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
r106 = y - X @ coef
rec("c", "bp=106 fixed, free slope/step/int: max|res|=%.6f sd=%.6f coef=(%.9f,%.9f,%.9f)"
    % (np.abs(r106).max(), r106.std(ddof=0), coef[0], coef[1], coef[2]))
# restricted model: slope fixed to 1 (claim's form), only step + intercept free
X2 = np.vstack([cum, (np.arange(n) >= 106).astype(float), np.ones(n)]).T
rhs = y - cum
X2b = np.vstack([(np.arange(n) >= 106).astype(float), np.ones(n)]).T
coef2, *_ = np.linalg.lstsq(X2b, rhs, rcond=None)
r2 = rhs - X2b @ coef2
rec("c", "slope fixed to 1: step=%.6f m intercept=%.6f m max|res|=%.6f m" % (coef2[0], coef2[1], np.abs(r2).max()))

# --- ADVERSARIAL c: (i) smooth drift instead of step; (ii) drift+step; (iii) reverse
g = y - cum
for name, Xd in [
    ("const only", np.ones((n, 1))),
    ("linear in c01", np.vstack([y, np.ones(n)]).T),
    ("quadratic in c01", np.vstack([y ** 2, y, np.ones(n)]).T),
    ("cubic in c01", np.vstack([y ** 3, y ** 2, y, np.ones(n)]).T),
    ("piecewise-linear continuous, knot at bp 2.99 m", np.vstack([y, np.maximum(0, y - 2.99), np.ones(n)]).T),
]:
    cc, *_ = np.linalg.lstsq(Xd, g, rcond=None)
    rr = g - Xd @ cc
    rec("c-alt", "gap explained by %-42s : max|res|=%.6f m R2=%.8f"
        % (name, np.abs(rr).max(), 1 - np.sum(rr ** 2) / np.sum((g - g.mean()) ** 2)))
# reverse direction: c03 from 100*diff(c01)
dc = np.diff(c1_mm)           # mm
rev = dc * 10                 # in 0.01cm units == c03 units? 1 mm = 10 * (0.01cm)
assert True
rec("c-rev", "100*diff(c01) in 0.01cm units: distinct=%s" % np.unique(rev).tolist())
ok = np.abs(rev - c3_u[1:]).max()
excl = np.arange(len(rev)) != big_jump
rec("c-rev", "100*diff(c01) vs c03[1:]: n exact=%d/%d (%.4f); incl. jump row max dev=%d units (%.4f cm); excl. jump row max dev=%d units"
    % (int((rev == c3_u[1:]).sum()), len(rev), (rev == c3_u[1:]).mean(),
       ok, ok / 100.0, np.abs(rev[excl] - c3_u[1:][excl]).max()))
d_rev = rev - c3_u[1:]
rec("c-rev", "100*diff(c01)-c03 in 0.01cm units: distinct values/counts = %s"
    % {int(v): int((d_rev == v).sum()) for v in np.unique(d_rev)})
# can c01 be exactly the 1-mm rounding of an underlying continuous depth built from c03?
# test: does there exist integer offset a (0.1mm units) with c01_mm*10 = 10*round((cumsum(c3_u)+a)/10)?
for a in range(-4000, 4001):
    cand = np.round((np.cumsum(c3_u) + a) / 10.0) * 10
    if np.array_equal(cand[:bp], c1_mm[:bp] * 10):
        rec("c-round", "exact mm-rounding of cumsum(c03)+a reproduces c01[:%d] with a=%d (%.4f m)" % (bp, a, a / 10000.0))
        break
else:
    # best match fraction
    bestfrac, besta = -1, None
    for a in range(-4000, 4001):
        cand = np.round((np.cumsum(c3_u) + a) / 10.0) * 10
        frac = (cand[:bp] == c1_mm[:bp] * 10).mean()
        if frac > bestfrac:
            bestfrac, besta = frac, a
    rec("c-round", "no exact mm-rounding offset for rows <106; best a=%d gives %.4f agreement" % (besta, bestfrac))

# =============================================================== (d)
prod = df.c01_depth_top_m.values * df.c05_density_ice_g_cm_3.values
r6 = df.c06_depth_w_e_m.values - prod
rec("d", "c06 - c01*c05: max|.|=%.6f  min=%.6f  mean=%+.6f  sd=%.6f  n_exact=%d"
    % (np.abs(r6).max(), r6.min(), r6.mean(), r6.std(ddof=0), int((r6 == 0).sum())))
rec("d", "span c01 = %.2f m ; max rel resid (c06>0) = %.4f" %
    (y[-1] - y[0], np.nanmax(np.abs(r6[y > 0] / df.c06_depth_w_e_m.values[y > 0]))))
# candidate exact generators for c06
for name, pred in [
    ("round(c01*c05,2)", np.round(prod, 2)),
    ("round(c01*c05,3)", np.round(prod, 3)),
    ("round(c02*c05,2)", np.round(df.c02_depth_ice_snow_m.values * df.c05_density_ice_g_cm_3.values, 2)),
    ("floor(c01*c05*100)/100", np.floor(prod * 100) / 100.0),
    ("round(c01*c05,2) (banker)", np.round(prod, 2)),
]:
    e = df.c06_depth_w_e_m.values - pred
    rec("d-gen", "%-26s exact=%d/%d (%.4f)  max|err|=%.6f" % (name, int((np.abs(e) < 5e-9).sum()), n, (np.abs(e) < 5e-9).mean(), np.abs(e).max()))
# residual correlates with c01?  (quantization story: r6 ~ c01*(rho_true - c05) + round err)
rec("d", "corr(r6, c01)=%+.4f ; corr(r6, c05)=%+.4f" % (np.corrcoef(r6, y)[0, 1], np.corrcoef(r6, c5_mg.astype(float))[0, 1]))
# r6 vs quantization bound 41.98*0.0005 + 0.005 = 0.026
rec("d", "quantization bound if c06=round(c01*rho_true,2) and c05=round(rho_true,3): 0.005+41.98*0.0005=%.4f ; observed max %.4f" % (0.005 + 41.98 * 0.0005, np.abs(r6).max()))
# implied rho from c06/c01 where c01 large -- compare with c05
big = y >= 5.0
imp = df.c06_depth_w_e_m.values[big] / y[big]
rec("d", "for c01>=5m (n=%d): |c06/c01 - c05| max=%.6f  frac<=0.0005: %.4f" %
    (big.sum(), np.abs(imp - df.c05_density_ice_g_cm_3.values[big]).max(),
     (np.abs(imp - df.c05_density_ice_g_cm_3.values[big]) <= 0.0005).mean()))

# =============================================================== (e)
u5, cnt5 = np.unique(c5_mg, return_counts=True)
rec("e", "c05 distinct=%d min=%.3f max=%.3f ; increments distinct=%s counts=%s"
    % (len(u5), u5.min() / 1000, u5.max() / 1000,
       (np.diff(u5)).tolist()[:3] + ["..."], {int(v): int((np.diff(u5) == v).sum()) for v in np.unique(np.diff(u5))}))
# are c05 values exactly the lattice 379..? with one 5-step?
lat = np.arange(u5.min(), u5.min() + 1000 * (u5.max() - u5.min()) // 1000 + 1)
# simpler: check representation u5 = 379 + k + jump
rec("e", "lattice check: u5 sorted, first/last=%d/%d ; missing lattice values between: %d of %d"
    % (u5.min(), u5.max(), (u5.max() - u5.min() + 1) - len(u5), u5.max() - u5.min() + 1))
# equal c01 implies equal c05?
g1 = {}
bad = 0
for a, b in zip(c1_mm, c5_mg):
    if a in g1 and g1[a] != b:
        bad += 1
    else:
        g1.setdefault(a, b)
rec("e", "distinct c01=%d ; equal-c01->equal-c05 violations=%d (test is VACUOUS: every c01 distinct)" % (len(g1), bad))
# monotone in c01?
rec("e", "c05 non-increasing events when sorted by c01: %d ; equal-c05 adjacent pairs: %d"
    % (int((np.diff(c5_mg[y.argsort(kind='stable')]) < 0).sum()),
       int((np.diff(c5_mg[y.argsort(kind='stable')]) == 0).sum())))
# equal c05 -> equal c01? (i.e. is the map injective)
gu = {}
for a, b in zip(c5_mg, c1_mm):
    gu.setdefault(a, []).append(b)
mult = [len(v) for v in gu.values()]
rec("e", "c05 levels replicated: max multiplicity=%d, median=%d; range of c01 within a level > 1mm for %d levels"
    % (max(mult), int(np.median(mult)), sum(1 for v in gu.values() if max(v) - min(v) > 1)))
# quadratic fit, and rounding consistency of c05 = round(f(c01),3)
for deg in (1, 2, 3):
    cc = np.polyfit(y, df.c05_density_ice_g_cm_3.values, deg)
    rr = df.c05_density_ice_g_cm_3.values - np.polyval(cc, y)
    rec("e", "c05 ~ poly%d(c01): max|res|=%.9f ; rounding-consistent frac (|res|<=5e-4): %.4f"
        % (deg, np.abs(rr).max(), (np.abs(rr) <= 5e-4).mean()))
    if deg == 2:
        pred_round = np.round(np.polyval(cc, y), 3)
        rec("e", "quadratic->round(3dp) exactly equals c05: %d/%d (%.4f)" % (int((pred_round == df.c05_density_ice_g_cm_3.values).sum()), n, (pred_round == df.c05_density_ice_g_cm_3.values).mean()))
# step-function-of-depth test: is c05 constant within 1-mm depth bin? (it is 262-valued, so no)
# how many rows per c05 level, does c01 increase within each level? yes by construction

# ---- model family: how exactly is c06 generated from depth and density?
rec("d-gen2", "== generator grid for c06 (base depth x c05, rounding rule) ==")
bases = {"c01": y, "c02": df.c02_depth_ice_snow_m.values, "mean(c01,c02)": (y + df.c02_depth_ice_snow_m.values) / 2}
for bname, b in bases.items():
    for rname, fn in [("none", lambda v: v), ("floor.01", lambda v: np.floor(v * 100) / 100),
                      ("round.01", lambda v: np.round(v, 2)), ("round.02", lambda v: np.round(v, 2)),
                      ("round.03", lambda v: np.round(v, 3))]:
        if rname == "round.02":
            continue
        e = df.c06_depth_w_e_m.values - fn(b * df.c05_density_ice_g_cm_3.values)
        rec("d-gen2", "base=%-11s rule=%-8s exact=%4d/%d (%.4f) max|err|=%.4f rms=%.4f"
            % (bname, rname, int((np.abs(e) < 5e-9).sum()), n, (np.abs(e) < 5e-9).mean(), np.abs(e).max(),
               np.sqrt((e ** 2).mean())))
# compare with 0.01 m rounding of the *continuous* product of the true density
rec("d-gen2", "NOTE: c06 on 1-cm grid: max dev=%.2e" % np.abs(df.c06_depth_w_e_m.values * 100 - c6_cm).max())

# ---- sub-interval structure of the step (adversarial: is it really one step?)
rec("c-alt2", "== segment-wise fit of c01 = a*cum + b (no step), per 5 m depth band ==")
for lo in np.arange(0, 40, 5.0):
    m = (y >= lo) & (y < lo + 5.0)
    if m.sum() < 8:
        continue
    cc = np.polyfit(cum[m], y[m], 1)
    rr = y[m] - np.polyval(cc, cum[m])
    rec("c-alt2", "band [%.0f,%.0f) n=%4d slope=%.6f intercept=%+.6f max|res|=%.6f"
        % (lo, lo + 5, m.sum(), cc[0], cc[1], np.abs(rr).max()))
rec("c-alt2", "== same, per 5 m band, slope fixed to 1 ==")
for lo in np.arange(0, 40, 5.0):
    m = (y >= lo) & (y < lo + 5.0)
    rr = y[m] - cum[m] - np.median(y[m] - cum[m])
    rec("c-alt2", "band [%.0f,%.0f) n=%4d median(c01-cum)=%+.4f  half-range=%.4f"
        % (lo, lo + 5, m.sum(), np.median(y[m] - cum[m]), 0.5 * (y[m] - cum[m]).max() - 0.5 * (y[m] - cum[m]).min()))

# =============================================================== TASK 1 pair scan
rec("t1", "")
rec("t1", "== exhaustive pair scan (8 numeric cols; c07 handled separately) ==")
num = ["c01_depth_top_m", "c02_depth_ice_snow_m", "c03_samp_thick_cm", "c04_sulfate_ug_kg",
       "c05_density_ice_g_cm_3", "c06_depth_w_e_m", "c08_age_a_ad_ce", "c09_age_ka_bp"]
for i in range(len(num)):
    for j in range(i + 1, len(num)):
        A = df[num[i]].values.astype(float)
        B = df[num[j]].values.astype(float)
        ok = np.isfinite(A) & np.isfinite(B)
        if ok.sum() < 20:
            rec("t1", "%-22s %-22s n=%d skip" % (num[i], num[j], ok.sum()))
            continue
        M = np.vstack([A[ok] / np.ptp(A[ok]), B[ok] / np.ptp(B[ok]), np.ones(ok.sum())]).T
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        v = Vt[-1]
        val = M @ v
        val = val - val.mean()
        # express as normalized a*A+b*B+c=0 residual in combined units
        rec("t1", "%-22s %-22s sv_min/sv_max=%.2e  max|aA+bB+c|=%.4g" % (num[i], num[j], s[-1] / s[0], np.abs(M @ Vt[-1]).max()))
# targeted c04 checks
def group_variation(key, vals, nd=9):
    d = {}
    for k, v in zip(np.round(key, nd), vals):
        d.setdefault(k, []).append(v)
    multi = {k: v for k, v in d.items() if len(v) > 1}
    varying = sum(1 for v in multi.values() if len(set(np.round(v, 9))) > 1)
    return len(multi), varying
for keyname, key in [("c06", np.round(df.c06_depth_w_e_m.values, 9)), ("c01", np.round(df.c01_depth_top_m.values, 9)),
                     ("c05", np.round(df.c05_density_ice_g_cm_3.values, 9)), ("c03", np.round(df.c03_samp_thick_cm.values, 9)),
                     ("c02", np.round(df.c02_depth_ice_snow_m.values, 9))]:
    nm, nv = group_variation(key, df.c04_sulfate_ug_kg.values)
    rec("t1", "c04 within equal-%s groups: %d groups mult>1, %d of them have >1 distinct c04" % (keyname, nm, nv))
# and does any geometry column repeat at ALL?
for keyname, key in [("c01", c1_mm), ("c02", c2_mm), ("c06", c6_cm)]:
    u, c = np.unique(key, return_counts=True)
    rec("t1", "column %s: distinct=%d max multiplicity=%d" % (keyname, len(u), c.max()))
# c04 correlations
for c in num:
    if c == "c04_sulfate_ug_kg":
        continue
    A = df[c].values.astype(float)
    ok = np.isfinite(A) & df.c04_sulfate_ug_kg.notna().values
    rec("t1", "pearson c04 vs %-22s r=%+.4f (n=%d)" % (c, np.corrcoef(A[ok], df.c04_sulfate_ug_kg.values[ok])[0, 1], ok.sum()))
c4 = df.c04_sulfate_ug_kg.dropna().values
rec("t1", "c04 n=%d distinct=%d lattice0.001 maxdev=%.2e" % (len(c4), len(np.unique(c4)), np.abs(c4 - np.round(c4, 3)).max()))
# c07 checks
rec("t1", "-- c07 --")
sig = df.c07_signal.notna().values
rec("t1", "c07 non-empty rows=%d ; labels=%s ; same rows as c08/c09 non-null: %s"
    % (sig.sum(), df.loc[sig, "c07_signal"].value_counts().to_dict(), bool(np.all(sig == m8))))
idx = np.where(sig)[0]
rec("t1", "signal rows (idx, c01, c04, c06, c09): %s" % [(int(i), float(df.c01_depth_top_m.values[i]),
      float(df.c04_sulfate_ug_kg.values[i]), float(df.c06_depth_w_e_m.values[i]),
      float(df.c09_age_ka_bp.values[i])) for i in idx])
rec("t1", "c04 percentile of signal rows: %s" % np.round(pd.Series(df.c04_sulfate_ug_kg.values).rank(pct=True).values[idx], 3).tolist())
# do labels repeat at same geometry?
for lab, sub in df[sig].groupby("c07_signal"):
    rec("t1", "label %-20s n=%d c01=%s c04=%s" % (lab, len(sub), np.round(sub.c01_depth_top_m.values, 3).tolist(), np.round(sub.c04_sulfate_ug_kg.values, 3).tolist()))
# c08 vs geometry: any functional form?
ok8 = m8
for deg in (1, 2, 3, 4, 5):
    cc = np.polyfit(df.c01_depth_top_m.values[ok8], df.c08_age_a_ad_ce.values[ok8], deg)
    rr = df.c08_age_a_ad_ce.values[ok8] - np.polyval(cc, df.c01_depth_top_m.values[ok8])
    rec("t1", "c08 ~ poly%d(c01) on 15 dated rows: max|res|=%.2f a (range %d a)" % (deg, np.abs(rr).max(), int(np.ptp(df.c08_age_a_ad_ce.values[ok8]))))
rec("t1", "c08 monotonically non-increasing in c01 on dated rows: %s"
    % bool(np.all(np.diff(df.c08_age_a_ad_ce.values[ok8][np.argsort(df.c01_depth_top_m.values[ok8])]) <= 0)))

# =============================================================== TASK 3 direction
rec("t3", "")
rec("t3", "== direction test c05<->c06 ==")
ratio = df.c06_depth_w_e_m.values / y
rA = df.c05_density_ice_g_cm_3.values - np.round(ratio, 3)   # c05 rounded from c06/c01
rB = df.c06_depth_w_e_m.values - np.round(prod, 3)           # c06 rounded from c01*c05 (3dp)
rB2 = df.c06_depth_w_e_m.values - np.round(prod, 2)          # 2dp
for nm, rr in [("c05 - round(c06/c01,3)", rA), ("c06 - round(c01*c05,3)", rB), ("c06 - round(c01*c05,2)", rB2)]:
    fin = np.isfinite(rr)
    rec("t3", "%-26s frac|.|<=5e-4 = %.4f (%d/%d) ; frac exact = %.4f ; max|.|=%.4g"
        % (nm, (np.abs(rr[fin]) <= 5e-4).mean(), int((np.abs(rr[fin]) <= 5e-4).sum()), n,
           (np.abs(rr[fin]) < 5e-9).mean(), np.abs(rr[fin]).max()))
# fail locations of reverse model
fin = np.isfinite(rA)
bad = fin & (np.abs(rA) > 5e-4)
rec("t3", "reverse-model failures: %d rows ; their c01: min=%.3f max=%.3f median=%.3f"
    % (bad.sum(), y[bad].min(), y[bad].max(), np.median(y[bad])))
rec("t3", "reverse-model failures for c01>=1 m: %d/%d (%.4f) ; for c01<1 m: %d/%d (%.4f)"
    % (int((bad & (y >= 1)).sum()), int((y >= 1).sum()), (bad & (y >= 1)).sum() / (y >= 1).sum(),
       int((bad & (y < 1)).sum()), int((y < 1).sum()), (bad & (y < 1)).sum() / (y < 1).sum()))
# ratio monotonicity vs c05 monotonicity
rec("t3", "ratio c06/c01 non-increasing events when sorted by c01: %d (c05 itself has %d)"
    % (int((np.diff((df.c06_depth_w_e_m.values / y)[y.argsort(kind='stable')]) < 0).sum()),
       int((np.diff(c5_mg[y.argsort(kind='stable')]) < 0).sum())))
rec("t3", "mean overshoot r6 = c06 - c01*c05 is %+.6f m -> c05 < c06/c01 on average (consistent with c05=round down / c06 rounded up?)" % r6.mean())
# where is reverse forced by data? rows where c01*c05 > c06 + 0.005  (would break c06=round(product,2))
viol = prod > df.c06_depth_w_e_m.values + 0.005
rec("t3", "rows where c01*c05 > c06+0.005 (would falsify c06=round(c01*c05,2) unless rho>c05): %d ; max excess=%.4f"
    % (int(viol.sum()), (prod - df.c06_depth_w_e_m.values)[viol].max() if viol.any() else 0.0))
viol2 = prod < df.c06_depth_w_e_m.values - 0.005
rec("t3", "rows where c01*c05 < c06-0.005: %d ; max deficit=%.4f" % (int(viol2.sum()), (df.c06_depth_w_e_m.values - prod)[viol2].max() if viol2.any() else 0.0))

# =============================================================== figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(y, Rint / 10000.0, lw=0.8)
ax[0, 0].set_title("exact residual c01 - cumsum(c03)/100 (m)")
ax[0, 1].plot(y, r6, ".", ms=2)
ax[0, 1].set_title("c06 - c01*c05 (m)")
ax[1, 0].plot(df.c01_depth_top_m.values, df.c05_density_ice_g_cm_3.values, ".", ms=2)
ax[1, 0].set_title("c05 vs c01 (ramp)")
ax[1, 1].hist(np.diff(c1_mm), bins=np.arange(-1, 6) - 0.5)
ax[1, 1].set_title("depth increment (mm)")
for a in ax.ravel():
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "/verify_finding1.png", dpi=120)
rec("figure", "wrote verify_finding1.png")

with open(OUT + "/verify_finding1_output.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
with open(OUT + "/verify_finding1_summary.json", "w") as fh:
    json.dump(R, fh, indent=1)
print("\nwrote output files to", OUT)
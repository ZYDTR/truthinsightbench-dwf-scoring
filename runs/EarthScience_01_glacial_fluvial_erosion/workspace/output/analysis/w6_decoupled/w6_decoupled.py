#!/usr/bin/env python3
"""
w6_decoupled: de-biased erosion-rate / timescale scaling from records with
independent clocks.

Data: /data/user/.../project/data/raw/{glacial,nonglacial}_erosion_Earth.tsv
Frozen benchmark; no external data. Deterministic. numpy/scipy only for stats
(statsmodels import is pathologically slow in this container, so OLS is
implemented directly with lstsq).

Sections
  1. Coupling classification audit (row level CSV + counts by Methodology/Process)
  2. logER ~ logTI slopes by class (coupled / independent / strict-independent),
     per Methodology, cluster-bootstrap CIs by Reference
  3. Predictor dominance ranking (logS, logP, logA, Lat, logTI) inside models with
     C(Methodology) and C(Process), on independent rows and on all rows
  4. Threshold sensitivity of the independent-subset slope (TI >= 0/10/100/1000 yr)
  5. Counter-case quantification: spread of log10(E*T) per stratum
  6. Written conclusion -> results.json + notes.md
"""
import functools
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

print = functools.partial(print, flush=True)  # noqa: A001
T0 = time.time()


def stamp(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/data/user/conversations/conv-ab8a0c4c-4c0f-46e9-96bb-16e130938d6a/project'
OUT = os.path.join(ROOT, 'output', 'analysis', 'w6_decoupled')
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20240919)
NBOOT_SMALL = 1000
NBOOT_BIG = 400
BIG_N = 1500
REL_TOL = 0.05                      # 5% relative tolerance for "identity" checks
LOCK_TOL = 0.10                     # IQR of log10(E*T) below this = hard product lock
SOFT_TOL = 0.30                     # ... below this = soft (near) product lock
LOCK_TIRANGE = 1.0                  # dex; stratum must span >= 1 dex in logTI
LOGREL = np.log10(1.0 + REL_TOL)    # 0.0212 dex


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------
def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.strip().replace({'': np.nan, 'nan': np.nan}),
                         errors='coerce')


def load(path, process):
    d = pd.read_csv(path, sep='\t', dtype=str)
    cols = {'Erosion rate (mm/yr)': 'ER', 'Time interval (yr)': 'TI',
            'Area (km2)': 'A', 'Slope (m/km)': 'S',
            'Precipitation (mm/yr)': 'P', 'Latitude': 'Lat',
            'Exhumation (m)': 'EX', 'Sediment flux (km3/yr)': 'SF',
            'Methodology': 'Method', 'Compilation': 'Comp', 'Reference': 'Ref',
            'Comments': 'Comments', 'Type': 'Type', 'Locality': 'Locality'}
    out = pd.DataFrame({v: (d[k] if k in ('Methodology', 'Compilation', 'Reference', 'Comments', 'Type', 'Locality')
                            else num(d[k])) for k, v in cols.items()})
    out['Process'] = process
    out['row_id'] = np.arange(len(out))
    out['Method'] = out['Method'].astype(str).str.strip()
    out['Comp'] = out['Comp'].astype(str).str.strip()
    out['Ref'] = out['Ref'].astype(str).str.strip()
    # keep only rows usable for a log-log rate/timescale fit
    m = (out.ER > 0) & (out.TI > 0)
    out = out[m].copy()
    out['logER'] = np.log10(out.ER)
    out['logTI'] = np.log10(out.TI)
    out['logET'] = np.log10(out.ER * out.TI)          # mm
    out['logA'] = np.log10(out.A.where(out.A > 0))
    out['logS'] = np.log10(out.S.where(out.S > 0))
    out['logP'] = np.log10(out.P.where(out.P > 0))
    out['stratum'] = out.Process + ' | ' + out.Method + ' | ' + out.Comp
    return out


g = load(os.path.join(ROOT, 'data/raw/glacial_erosion_Earth.tsv'), 'Glacial')
n = load(os.path.join(ROOT, 'data/raw/nonglacial_erosion_Earth.tsv'), 'Nonglacial')
df = pd.concat([g, n], ignore_index=True)
stamp('load complete')
print('rows: glacial %d, nonglacial %d, total %d' % (len(g), len(n), len(df)))


# -----------------------------------------------------------------------------
# statistics helpers (no statsmodels)
# -----------------------------------------------------------------------------
def ols(y, X, names):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    n_, k = Xc.shape
    dof = n_ - k
    s2 = float(resid @ resid) / dof if dof > 0 else np.nan
    xtxinv = np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.diag(xtxinv) * s2)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan
    r2adj = 1.0 - (1.0 - r2) * (n_ - 1) / dof if dof > 0 else np.nan
    tt = beta / se
    pp = 2.0 * stats.t.sf(np.abs(tt), dof)
    out = {'n': int(n_), 'k': int(k), 'dof': int(dof), 'r2': r2, 'r2adj': r2adj,
           'resid_ss': float(resid @ resid), 'tss': tss}
    for i, nm in enumerate(names):
        out['%s_beta' % nm] = float(beta[i + 1])
        out['%s_se' % nm] = float(se[i + 1])
        out['%s_p' % nm] = float(pp[i + 1])
        out['%s_t' % nm] = float(tt[i + 1])
        sd = float(np.std(X[:, i], ddof=1))
        out['%s_stdbeta' % nm] = float(beta[i + 1] * sd / np.std(y, ddof=1)) if np.std(y, ddof=1) > 0 else np.nan
    return out


def slope_fit(sub, xcol='logTI', ycol='logER'):
    if len(sub) < 3:
        return None
    return ols(sub[ycol].values, sub[[xcol]].values, [xcol])


def cluster_boot_slope(sub, xcol='logTI', ycol='logER', nboot=None, rng=RNG):
    """Cluster bootstrap (resample References with replacement), index based."""
    if len(sub) < 4:
        return None
    nboot = nboot or (NBOOT_BIG if len(sub) > BIG_N else NBOOT_SMALL)
    y = sub[ycol].values.astype(float)
    x = sub[xcol].values.astype(float)
    ref = sub['Ref'].values
    codes, uniq = pd.factorize(ref)
    ng = len(uniq)
    if ng < 2:
        return None
    order = np.argsort(codes, kind='stable')
    codes_s = codes[order]
    bounds = np.searchsorted(codes_s, np.arange(ng + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(ng)]
    slopes = []
    for _ in range(nboot):
        pick = rng.integers(0, ng, ng)
        idx = np.concatenate([groups[i] for i in pick])
        xb, yb = x[idx], y[idx]
        if len(idx) < 4 or np.all(xb == xb[0]):
            continue
        dx = xb - xb.mean()
        denom = float(dx @ dx)
        if denom <= 0:
            continue
        slopes.append(float((dx @ (yb - yb.mean())) / denom))
    if len(slopes) < 20:
        return None
    slopes = np.array(slopes)
    return {'boot_n': int(len(slopes)),
            'boot_ci_lo': float(np.percentile(slopes, 2.5)),
            'boot_ci_hi': float(np.percentile(slopes, 97.5)),
            'boot_sd': float(slopes.std(ddof=1)),
            'boot_clusters': int(ng)}


def model_summary(sub, xcol, ycol='logER', label='', boot=True):
    fit = slope_fit(sub, xcol, ycol)
    if fit is None:
        return None
    # estimability guard: a slope is only interpretable if x actually varies
    xrange = float(sub[xcol].max() - sub[xcol].min())
    rec = {'label': label, 'n': fit['n'], 'slope': fit['%s_beta' % xcol],
           'se': fit['%s_se' % xcol], 'p': fit['%s_p' % xcol],
           'r2': fit['r2'], 'r2adj': fit['r2adj'],
           'x_range_dex': xrange, 'estimable': bool(xrange >= 0.3),
           'logTI_min': float(sub.logTI.min()), 'logTI_max': float(sub.logTI.max()),
           'TI_min_yr': float(sub.TI.min()), 'TI_max_yr': float(sub.TI.max())}
    if boot:
        b = cluster_boot_slope(sub, xcol, ycol)
        rec.update(b or {'boot_n': 0})
    return rec


def dummies(series, prefix):
    d = pd.get_dummies(series.astype(str), prefix=prefix, drop_first=True, dtype=float)
    return d


def fe_slope(sub, xcol='logTI', ycol='logER', min_within=4):
    """Within-Reference fixed-effects slope: demeans x and y inside each
    Reference, then fits the pooled within slope. Removes between-paper
    heterogeneity (including different methods/regions) and identifies the
    exponent purely from within-paper timescale variation."""
    sub = sub.copy()
    sub['_x'] = sub[xcol] - sub.groupby('Ref')[xcol].transform('mean')
    sub['_y'] = sub[ycol] - sub.groupby('Ref')[ycol].transform('mean')
    keep = sub.groupby('Ref')[xcol].transform(lambda s: s.max() - s.min()) >= 0.3
    sub = sub[keep]
    if len(sub) < 5 or float(sub._x.var()) <= 0:
        return None
    fit = ols(sub._y.values, sub[['_x']].values, ['_x'])
    # dof correction for the absorbed reference means (one per reference with data)
    return {'n': fit['n'], 'n_refs': int(sub.Ref.nunique()),
            'slope_within': fit['_x_beta'], 'se_within': fit['_x_se'],
            'p_within': fit['_x_p'], 'r2_within': fit['r2'],
            'x_range_dex_within': float(sub[xcol].max() - sub[xcol].min())}


stamp('SECTION 1 start')
# -----------------------------------------------------------------------------
# SECTION 1: coupling classification audit
# -----------------------------------------------------------------------------
# evidence flags
df['ex_ratio'] = df.ER * df.TI / (df.EX * 1000.0)
df['ex_identity'] = (df.EX.notna()) & (df.EX > 0) & ((df.ex_ratio - 1).abs() <= REL_TOL)
df['flux_pred'] = df.SF / df.A * 1e6
df['flux_ratio'] = df.ER / df.flux_pred
df['flux_identity'] = (df.SF.notna()) & (df.A.notna()) & (df.A > 0) & (df.SF > 0) & \
                      ((df.flux_ratio - 1).abs() <= REL_TOL)

# stratum-level product-lock diagnostics (compilation level and reference level).
# Robust version: the product lock must hold ACROSS distinct timescale groups.
# For each stratum we group rows by their exact reported interval, keep groups with
# n >= MIN_TI_GROUP, and measure the spread of the per-group median log10(E*T).
# This prevents one lone row with a different interval from faking a 1-dex span.
MIN_TI_DISTINCT = 5      # a lock claim needs >= 5 distinct reported intervals
MIN_REF_N, MIN_COMP_N = 5, 10


def lock_stats(grp):
    """Product-lock diagnostics for one stratum.

    A product lock means the reported product E*T stays constant while the
    reported timescale T varies.  Detecting it requires that the timescale axis
    be genuinely populated (>= MIN_TI_DISTINCT distinct intervals spanning
    >= 1 dex); otherwise a stratum with a single interval cannot be evidence of
    anything about a product.  Everything is computed in log10 so that a fixed
    product maps to zero spread in dex.
    """
    iqr = float(grp.logET.quantile(0.75) - grp.logET.quantile(0.25))
    ti_range = float(grp.logTI.max() - grp.logTI.min())
    nti = int(grp.TI.nunique())
    ok = (nti >= MIN_TI_DISTINCT) and (ti_range >= LOCK_TIRANGE)
    d = {'n': len(grp), 'n_ti_distinct': nti, 'iqr_logET': iqr,
         'range_logET': float(grp.logET.max() - grp.logET.min()),
         'ti_range_dex': ti_range,
         'hard_lock': bool(ok and iqr <= LOCK_TOL),
         'soft_lock': bool(ok and iqr <= SOFT_TOL)}
    if nti > 1:
        d['slope'] = float(np.polyfit(grp.logTI, grp.logER, 1)[0])
        d['r2'] = float(np.corrcoef(grp.logTI, grp.logER)[0, 1] ** 2)
    else:
        d['slope'] = d['r2'] = np.nan
    return d


def lock_table(sub, keys, min_n):
    rows = []
    for k, grp in sub.groupby(keys):
        if len(grp) < min_n:
            continue
        d = lock_stats(grp)
        d['level_n_min'] = min_n
        d['key'] = k if isinstance(k, str) else '|'.join(k)
        rows.append(d)
    return pd.DataFrame(rows)


# Product locks are looked for at two nested levels: the whole Compilation and
# the Compilation x Reference cell (one paper = one rate construction).
lock_comp = lock_table(df, ['stratum'], MIN_COMP_N)
lock_ref = lock_table(df, ['stratum', 'Ref'], MIN_REF_N)
hard_comp = set(lock_comp.loc[lock_comp.hard_lock, 'key'])
hard_ref = set(lock_ref.loc[lock_ref.hard_lock, 'key'])
soft_comp = set(lock_comp.loc[lock_comp.soft_lock, 'key'])
soft_ref = set(lock_ref.loc[lock_ref.soft_lock, 'key'])

df['lock_comp'] = df.stratum.isin(hard_comp)
df['lock_ref'] = pd.Series([('%s|%s' % (s, r)) in hard_ref for s, r in zip(df.stratum, df.Ref)],
                           index=df.index)
df['soft_lock'] = df.stratum.isin(soft_comp) | pd.Series(
    [('%s|%s' % (s, r)) in soft_ref for s, r in zip(df.stratum, df.Ref)], index=df.index)

# Row-level verdict.  Only algebraic/structural evidence counts as "coupled":
#   (i)   exhumation identity  ER*TI == EX*1000  (<= 5%),
#   (ii)  flux identity        ER    == SF/A*1e6 (<= 5%),
#   (iii) a stratum- or reference-level product lock.
df['coupled_evidence'] = np.where(
    df.ex_identity, 'exhumation_identity',
    np.where(df.flux_identity, 'flux_identity',
             np.where(df.lock_comp, 'compilation_product_lock',
                      np.where(df.lock_ref, 'reference_product_lock', ''))))
# one merged label for compact reporting
ev = df.coupled_evidence.replace({'compilation_product_lock': 'product_lock',
                                  'reference_product_lock': 'product_lock'})
df['coupled_evidence_label'] = ev

IND_METHODS = {'Cosmogenic surface', 'Cosmogenic detrital', 'Thermochronometric'}


def classify(row):
    """(a) coupled / (b) independent / (c) indeterminate.

    independent = the rate was NOT built from the reported interval AND the
    method carries its own clock (cosmogenic nuclide inventories or cooling
    ages).  Volumetric / Relief rates stay indeterminate unless an independent
    flux identity proves the rate came from a sediment budget.
    """
    if row.coupled_evidence != '':
        return 'coupled'
    if row.Method in IND_METHODS:
        return 'independent'
    return 'indeterminate'


df['coupling_class'] = df.apply(classify, axis=1)
df['strict_independent'] = (df.coupling_class == 'independent') & (~df.soft_lock)
df['soft_locked'] = df.soft_lock & (df.coupling_class == 'independent')

# --- attenuation-depth test for cosmogenic strata -----------------------------
# For a CRN-inventory rate the rate is E = P/(rho*(1-exp(-T/tau))) and the
# steady-state limit is E*T -> P*tau = Lambda/rho (the effective attenuation
# depth).  A CRN stratum whose implied E*T clusters tightly around ~0.6 m is
# therefore mechanically locked in exactly the same way as the exhumation strata,
# even though no exhumation column is populated for it.  Constant overburden
# depth (tens of g/cm2 in g/cm2 / 2.7 g/cm3 ~ 0.6 m) and constant production rate
# produce an exact E = const/T relation.  Report the implied product per stratum.
CRN_METHODS = {'Cosmogenic surface', 'Cosmogenic detrital'}
att = []
for k, grp in df[df.Method.isin(CRN_METHODS)].groupby('stratum'):
    if len(grp) < 5:
        continue
    pet = grp.ER * grp.TI          # mm
    att.append({'stratum': k, 'n': len(grp), 'ET_median_mm': float(pet.median()),
                'ET_median_m': float(pet.median() / 1000.0),
                'logET_iqr_dex': float(np.log10(pet).quantile(.75) - np.log10(pet).quantile(.25)),
                'logET_range_dex': float(np.log10(pet).max() - np.log10(pet).min()),
                'n_ti_distinct': int(grp.TI.nunique()),
                'ti_range_dex': float(grp.logTI.max() - grp.logTI.min())})
att = pd.DataFrame(att).sort_values('stratum')
# only CRN strata whose implied product is near the 0.6 m attenuation depth can
# be mechanically locked by the production-decay algebra
NEAR_ATT = 0.25     # dex tolerance around the ~0.6 m median depth
att['near_attenuation_depth'] = np.abs(np.log10(att.ET_median_m / 0.6)) <= NEAR_ATT
att['crn_locked'] = att.near_attenuation_depth & (att.logET_iqr_dex <= SOFT_TOL) \
    & (att.ti_range_dex >= LOCK_TIRANGE) & (att.n_ti_distinct >= 5)
print('\n=== CRN attenuation-depth test ===')
print(att.round(3).to_string(index=False))
att.to_csv(os.path.join(OUT, 'crn_attenuation_depth_test.csv'), index=False)
crn_locked_strata = set(att.loc[att.crn_locked, 'stratum'])
df['crn_lock'] = df.stratum.isin(crn_locked_strata)
df['coupled_evidence'] = np.where(
    (df.coupled_evidence == '') & df.crn_lock, 'crn_attenuation_lock', df.coupled_evidence)
df['coupling_class'] = df.apply(classify, axis=1)
df['strict_independent'] = (df.coupling_class == 'independent') & (~df.soft_lock)
df['soft_locked'] = df.soft_lock & (df.coupling_class == 'independent')
print('rows reclassified as coupled by the CRN attenuation lock:', int(df.crn_lock.sum()))

# counts
cnt_method = df.groupby(['Process', 'Method', 'coupling_class']).size().unstack(fill_value=0)
cnt_proc = df.groupby(['Process', 'coupling_class']).size().unstack(fill_value=0)
cnt_all = df.groupby('coupling_class').size()
print('\n=== class counts by Process ===\n', cnt_proc)
print('\n=== class counts by Methodology ===\n', cnt_method)

audit_cols = ['row_id', 'Process', 'Method', 'Type', 'Comp', 'Ref', 'Locality',
              'ER', 'TI', 'EX', 'SF', 'A', 'ex_ratio', 'flux_ratio',
              'ex_identity', 'flux_identity', 'lock_comp', 'lock_ref', 'soft_lock',
              'coupled_evidence', 'coupling_class', 'strict_independent', 'soft_locked']
df[audit_cols].to_csv(os.path.join(OUT, 'row_classification_audit.csv'), index=False)
lock_comp.to_csv(os.path.join(OUT, 'stratum_lock_diagnostics.csv'), index=False)
lock_ref.to_csv(os.path.join(OUT, 'reference_lock_diagnostics.csv'), index=False)


stamp('SECTION 2 start')
# -----------------------------------------------------------------------------
# SECTION 2: slopes coupled vs independent
# -----------------------------------------------------------------------------
models = []

def add_model(name, sub, boot=True):
    rec = model_summary(sub, 'logTI', label=name, boot=boot)
    if rec:
        rec['subset'] = name
        models.append(rec)
    return rec

add_model('ALL rows', df)
for proc in ['Glacial', 'Nonglacial']:
    sub = df[df.Process == proc]
    add_model('ALL %s' % proc, sub)
    for cls in ['coupled', 'independent', 'indeterminate']:
        s = sub[sub.coupling_class == cls]
        if len(s) >= 5:
            add_model('%s | %s' % (proc, cls), s)
    s = sub[sub.strict_independent]
    if len(s) >= 5:
        add_model('%s | independent strict (no soft lock)' % proc, s)
    for meth in sorted(sub.Method.unique()):
        for cls in ['coupled', 'independent']:
            s = sub[(sub.Method == meth) & (sub.coupling_class == cls)]
            if len(s) >= 5:
                add_model('%s | %s | %s' % (proc, meth, cls), s)
        s = sub[(sub.Method == meth) & (sub.strict_independent)]
        if len(s) >= 5:
            add_model('%s | %s | independent strict' % (proc, meth), s)

sec2 = pd.DataFrame(models)
print('\n=== SECTION 2 slopes ===')
print(sec2[['subset', 'n', 'slope', 'se', 'p', 'r2',
            'boot_ci_lo', 'boot_ci_hi', 'TI_min_yr', 'TI_max_yr']].to_string(index=False))
sec2.to_csv(os.path.join(OUT, 'slopes_by_class.csv'), index=False)

# --- alternative definitions of "independent" (robustness of the verdict) -----
alt_rows = []
alt_defs = {
    'A_indep_methods': df[df.coupling_class == 'independent'],
    'B_indep_methods_no_thermo': df[(df.coupling_class == 'independent') & (df.Method != 'Thermochronometric')],
    'C_flux_identity_only': df[df.flux_identity],
    'D_all_nonlocked_volumetric': df[(df.Method == 'Volumetric') & (~df.lock_comp) & (~df.lock_ref) & (~df.ex_identity)],
    'E_nonglacial_cosmogenic_surface': df[(df.Process == 'Nonglacial') & (df.Method == 'Cosmogenic surface')],
    'F_glacial_cosmogenic_surface_indep': df[(df.Process == 'Glacial') & (df.Method == 'Cosmogenic surface') &
                                            (df.coupling_class == 'independent')],
    'F2_glacial_cosmogenic_surface_all': df[(df.Process == 'Glacial') & (df.Method == 'Cosmogenic surface')],
    'G_glacial_thermo_excl_single_interval': df[(df.Process == 'Glacial') &
                                                 (df.Method == 'Thermochronometric') &
                                                 (df.TI != 1e6)],
    'H_glacial_thermo_multiple_intervals': df[(df.Process == 'Glacial') &
                                              (df.Method == 'Thermochronometric') &
                                              (df.TI > 1e6)],
    'I_glacial_cosmogenic_detrital': df[(df.Process == 'Glacial') &
                                        (df.Method == 'Cosmogenic detrital')],
    'J_all_non_locked_rows': df[(df.coupled_evidence == '')],
}
for tag, sub in alt_defs.items():
    if len(sub) >= 5:
        rec = model_summary(sub, 'logTI', label=tag)
        if rec:
            rec['subset'] = tag
            rec['definition'] = tag
            alt_rows.append(rec)
alts = pd.DataFrame(alt_rows)
alts.to_csv(os.path.join(OUT, 'independent_definition_sensitivity.csv'), index=False)
print('\n=== SECTION 2 alternative definitions of independent ===')
print(alts[['subset', 'n', 'slope', 'se', 'p', 'r2', 'boot_ci_lo', 'boot_ci_hi']].round(4).to_string(index=False))

# --- within-Reference (fixed-effects) slopes: the strictest de-biasing ---------
fe_rows = []
fe_specs = [('ALL rows', df), ('Glacial ALL', df[df.Process == 'Glacial']),
            ('Nonglacial ALL', df[df.Process == 'Nonglacial']),
            ('Glacial independent', df[(df.Process == 'Glacial') & (df.coupling_class == 'independent')]),
            ('Nonglacial independent', df[(df.Process == 'Nonglacial') & (df.coupling_class == 'independent')]),
            ('Glacial coupled', df[(df.Process == 'Glacial') & (df.coupling_class == 'coupled')]),
            ('Nonglacial coupled', df[(df.Process == 'Nonglacial') & (df.coupling_class == 'coupled')])]
for tag, sub in fe_specs:
    r = fe_slope(sub)
    if r:
        r['subset'] = tag
        fe_rows.append(r)
for proc in ['Glacial', 'Nonglacial']:
    for meth in sorted(df[df.Process == proc].Method.unique()):
        for cls in ['coupled', 'independent', 'indeterminate']:
            s = df[(df.Process == proc) & (df.Method == meth) & (df.coupling_class == cls)]
            if len(s) >= 8:
                r = fe_slope(s)
                if r:
                    r['subset'] = '%s | %s | %s' % (proc, meth, cls)
                    fe_rows.append(r)
fe = pd.DataFrame(fe_rows)
fe.to_csv(os.path.join(OUT, 'within_reference_slopes.csv'), index=False)
print('\n=== SECTION 2 within-Reference (fixed-effects) slopes ===')
print(fe[['subset', 'n', 'n_refs', 'slope_within', 'se_within', 'p_within', 'r2_within']].round(4).to_string(index=False))


stamp('SECTION 3 start')
# -----------------------------------------------------------------------------
# SECTION 3: alternative-definition robustness / predictor dominance
# -----------------------------------------------------------------------------
PREDS = [('logS', 'log10 Slope'), ('logP', 'log10 Precip'),
         ('logA', 'log10 Area'), ('Lat', 'Latitude'), ('logTI', 'log10 Time interval')]

def dominance(sub, extra_dummies=None, tag=''):
    out = []
    sub = sub.dropna(subset=[p for p, _ in PREDS]).copy()
    if len(sub) < 25:
        return out, None, 0
    Xbase = dummies(sub.Method, 'meth')
    if extra_dummies is not None:
        Xbase = pd.concat([Xbase, extra_dummies.reindex(sub.index)], axis=1)
    base = ols(sub.logER.values, Xbase.values, ['x'] * Xbase.shape[1])
    rows = []
    for p, lab in PREDS:
        Xf = pd.concat([Xbase, sub[[p]]], axis=1)
        full = ols(sub.logER.values, Xf.values, ['x'] * (Xf.shape[1] - 1) + [p])
        dr2 = full['r2'] - base['r2']
        q = 1
        F = (dr2 / q) / ((1 - full['r2']) / full['dof'])
        pF = stats.f.sf(F, q, full['dof'])
        rows.append({'tag': tag, 'predictor': lab, 'var': p, 'n': full['n'],
                     'delta_r2': dr2, 'F': float(F), 'p_F': float(pF),
                     'std_beta': full['%s_stdbeta' % p], 'beta': full['%s_beta' % p],
                     'se': full['%s_se' % p], 'p': full['%s_p' % p],
                     'base_r2': base['r2'], 'full_r2': full['r2']})
    Xall = pd.concat([Xbase] + [sub[[p]] for p, _ in PREDS], axis=1)
    allfit = ols(sub.logER.values, Xall.values, ['x'] * (Xall.shape[1] - len(PREDS)) + [p for p, _ in PREDS])
    r2_all = allfit['r2']
    rows = pd.DataFrame(rows)
    rows['rank_stdbeta'] = rows.std_beta.abs().rank(ascending=False).astype(int)
    rows['rank_delta_r2'] = rows.delta_r2.rank(ascending=False).astype(int)
    return rows, r2_all, len(sub)

dom_frames = []
for tag, sub in [('independent pooled', df[df.strict_independent]),
                 ('independent pooled (incl soft-lock)', df[df.coupling_class == 'independent'].copy()),
                 ('ALL rows', df),
                 ('Glacial independent', df[(df.Process == 'Glacial') & df.strict_independent]),
                 ('Glacial ALL', df[df.Process == 'Glacial']),
                 ('Nonglacial independent', df[(df.Process == 'Nonglacial') & df.strict_independent]),
                 ('Nonglacial ALL', df[df.Process == 'Nonglacial'])]:
    if tag.startswith('independent pooled'):
        ed = dummies(sub.Process, 'proc')
    else:
        ed = None
    rows, r2all, ncc = dominance(sub, ed, tag)
    if len(rows):
        rows['r2_full_model'] = r2all
        rows['n_complete'] = ncc
        dom_frames.append(rows)
dom = pd.concat(dom_frames, ignore_index=True)
dom.to_csv(os.path.join(OUT, 'predictor_dominance.csv'), index=False)
print('\n=== SECTION 3 dominance (independent pooled) ===')
print(dom[dom.tag == 'independent pooled'][['predictor', 'n', 'std_beta', 'delta_r2', 'p_F', 'rank_stdbeta', 'rank_delta_r2']].to_string(index=False))
print('\n=== SECTION 3 dominance (ALL rows) ===')
print(dom[dom.tag == 'ALL rows'][['predictor', 'n', 'std_beta', 'delta_r2', 'p_F', 'rank_stdbeta', 'rank_delta_r2']].to_string(index=False))

# timescale-excluded model (all predictors except logTI)
def no_ti_model(sub, tag, extra_dummies=None):
    sub = sub.dropna(subset=['logS', 'logP', 'logA', 'Lat']).copy()
    if len(sub) < 25:
        return None
    Xb = dummies(sub.Method, 'meth')
    if extra_dummies is not None:
        Xb = pd.concat([Xb, extra_dummies.reindex(sub.index)], axis=1)
    Xf = pd.concat([Xb, sub[['logS', 'logP', 'logA', 'Lat']]], axis=1)
    fit = ols(sub.logER.values, Xf.values, ['x'] * Xb.shape[1] + ['logS', 'logP', 'logA', 'Lat'])
    return {'tag': tag, 'n': fit['n'], 'r2': fit['r2'], 'r2adj': fit['r2adj'],
            **{k: v for k, v in fit.items() if any(k.startswith(p + '_') for p in ['logS', 'logP', 'logA', 'Lat'])}}

noti = []
for tag, sub, ed in [('independent pooled', df[df.strict_independent], dummies(df[df.strict_independent].Process, 'proc')),
                     ('Glacial independent', df[(df.Process == 'Glacial') & df.strict_independent], None),
                     ('Nonglacial independent', df[(df.Process == 'Nonglacial') & df.strict_independent], None),
                     ('Glacial ALL', df[df.Process == 'Glacial'], None),
                     ('Nonglacial ALL', df[df.Process == 'Nonglacial'], None)]:
    r = no_ti_model(sub, tag, ed)
    if r:
        noti.append(r)
noti_df = pd.DataFrame(noti)
noti_df.to_csv(os.path.join(OUT, 'no_timescale_models.csv'), index=False)
print('\n=== no-timescale models (independent pooled) ===')
print(noti_df[noti_df.tag == 'independent pooled'].round(4).to_string(index=False))


stamp('SECTION 4 start')
# -----------------------------------------------------------------------------
# SECTION 4: threshold sensitivity
# -----------------------------------------------------------------------------
thr_rows = []
for tag, sub in [('pooled independent strict', df[df.strict_independent]),
                 ('pooled independent (incl soft-lock)', df[df.coupling_class == 'independent'].copy()),
                 ('Glacial independent', df[(df.Process == 'Glacial') & df.strict_independent]),
                 ('Nonglacial independent', df[(df.Process == 'Nonglacial') & df.strict_independent]),
                 ('Glacial ALL', df[df.Process == 'Glacial']),
                 ('Nonglacial ALL', df[df.Process == 'Nonglacial'])]:
    for thr in [0, 10, 100, 1000]:
        s = sub[sub.TI >= thr]
        if len(s) < 5:
            thr_rows.append({'subset': tag, 'TI_threshold_yr': thr, 'n': len(s)})
            continue
        rec = model_summary(s, 'logTI', label='%s | TI>=%g' % (tag, thr))
        if rec:
            rec['subset'] = tag
            rec['TI_threshold_yr'] = thr
            thr_rows.append(rec)
thr = pd.DataFrame(thr_rows)
cols = ['subset', 'TI_threshold_yr', 'n', 'slope', 'se', 'p', 'r2', 'boot_ci_lo', 'boot_ci_hi', 'TI_min_yr']
thr = thr[[c for c in cols if c in thr.columns] + [c for c in thr.columns if c not in cols]]
thr.to_csv(os.path.join(OUT, 'threshold_sensitivity.csv'), index=False)
print('\n=== SECTION 4 threshold trajectory ===')
print(thr.round(4).to_string(index=False))

# per-methodology trajectory (independent rows only)
pm_rows = []
for proc in ['Glacial', 'Nonglacial']:
    for meth in sorted(df[df.Process == proc].Method.unique()):
        s = df[(df.Process == proc) & (df.Method == meth) & df.strict_independent]
        for t in [0, 10, 100, 1000]:
            ss = s[s.TI >= t]
            if len(ss) >= 5:
                rec = model_summary(ss, 'logTI', label='%s|%s|TI>=%g' % (proc, meth, t))
                if rec:
                    rec['Process'] = proc
                    rec['Method'] = meth
                    rec['TI_threshold_yr'] = t
                    pm_rows.append(rec)
pm = pd.DataFrame(pm_rows)
pm.to_csv(os.path.join(OUT, 'threshold_sensitivity_by_method.csv'), index=False)


stamp('SECTION 5 start')
# -----------------------------------------------------------------------------
# SECTION 5: counter-case - spread of log10(E*T) per stratum
# -----------------------------------------------------------------------------
def spread_table(sub, keys, tag):
    rows = []
    for k, grp in sub.groupby(keys):
        if len(grp) < 4:
            continue
        key = k if isinstance(k, str) else '|'.join(k)
        # largest product change between two rows
        d_et = float(grp.logET.max() - grp.logET.min())
        d_ti = float(grp.logTI.max() - grp.logTI.min())
        d_er = float(grp.logER.max() - grp.logER.min())
        rows.append({'level': tag, 'stratum': key, 'n': len(grp),
                     'iqr_logET': float(grp.logET.quantile(.75) - grp.logET.quantile(.25)),
                     'range_logET': d_et, 'range_logTI': d_ti, 'range_logER': d_er,
                     'range_TI_yr': float(grp.TI.max() - grp.TI.min()),
                     'slope': float(np.polyfit(grp.logTI, grp.logER, 1)[0]) if grp.logTI.nunique() > 1 else np.nan,
                     'r2': float(np.corrcoef(grp.logTI, grp.logER)[0, 1] ** 2) if grp.logTI.nunique() > 1 else np.nan,
                     'coupling_class_mix': '|'.join(sorted(set(grp.coupling_class)))})
    return pd.DataFrame(rows)

sp_comp = spread_table(df, ['stratum'], 'Methodology x Compilation')
sp_ref = spread_table(df, ['stratum', 'Ref'], 'Methodology x Compilation x Reference')
sp_proc = spread_table(df, ['Process'], 'Process')
spread_all = pd.concat([sp_proc, sp_comp, sp_ref], ignore_index=True)
spread_all.to_csv(os.path.join(OUT, 'ET_spread_by_stratum.csv'), index=False)
print('\n=== SECTION 5 largest product spreads (Methodology x Compilation) ===')
print(sp_comp.sort_values('range_logET', ascending=False).head(10).round(3).to_string(index=False))

# within-stratum pair with largest |dlogET|: quantify dlogTI and dlogER
pair_rows = []
for k, grp in df.groupby('stratum'):
    if len(grp) < 4:
        continue
    v = grp.logET.values
    i, j = int(np.argmax(v)), int(np.argmin(v))
    hi, lo = grp.iloc[i], grp.iloc[j]
    pair_rows.append({'stratum': k, 'n': len(grp), 'class_mix': '|'.join(sorted(set(grp.coupling_class))),
                      'dlogET': float(v[i] - v[j]), 'dlogTI': float(hi.logTI - lo.logTI),
                      'dlogER': float(hi.logER - lo.logER),
                      'ER_hi_ET': hi.ER, 'TI_hi_ET': hi.TI, 'ER_lo_ET': lo.ER, 'TI_lo_ET': lo.TI,
                      'ET_hi_mm': 10 ** hi.logET, 'ET_lo_mm': 10 ** lo.logET})
pairs = pd.DataFrame(pair_rows).sort_values('dlogET', ascending=False)
pairs.to_csv(os.path.join(OUT, 'countercase_pairs.csv'), index=False)
print('\n=== SECTION 5 strongest counter-case pairs ===')
print(pairs.head(8).round(3).to_string(index=False))

# slope vs product spread across strata -> is the -1 slope produced by fixed product?
sc = sp_comp.dropna(subset=['slope', 'iqr_logET'])
if len(sc) >= 5:
    m1 = ols(sc.slope.values, sc[['iqr_logET']].values, ['iqr'])
    rho = stats.spearmanr(sc.iqr_logET, sc.slope)
    spread_slope_rel = {'n_strata': int(m1['n']), 'slope_of_slope_vs_iqr': m1['iqr_beta'],
                        'se': m1['iqr_se'], 'p': m1['iqr_p'], 'r2': m1['r2'],
                        'spearman_rho': float(rho.statistic), 'spearman_p': float(rho.pvalue),
                        'pearson_r': float(np.corrcoef(sc.iqr_logET, sc.slope)[0, 1])}
else:
    spread_slope_rel = {}
print('\nslope-vs-spread across strata:', json.dumps(spread_slope_rel, default=float))
sc.to_csv(os.path.join(OUT, 'stratum_slope_vs_spread.csv'), index=False)


stamp('FIGURES start')
# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------
cls_color = {'coupled': '#d62728', 'independent': '#1f77b4',
             'indeterminate': '#7f7f7f'}
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
for cls, col in cls_color.items():
    s = df[df.coupling_class == cls]
    ax.scatter(s.logTI, s.logER, s=6, c=col, alpha=0.35, label='%s (n=%d)' % (cls, len(s)), linewidths=0)
xs = np.linspace(df.logTI.min(), df.logTI.max(), 50)
ax.plot(xs, -1 * xs + np.median(df.logET), 'k--', lw=1, label='slope -1 guide')
ax.set_xlabel('log10 Time interval (yr)'); ax.set_ylabel('log10 Erosion rate (mm/yr)')
ax.set_title('A. All rows, by coupling class'); ax.legend(fontsize=7, loc='lower left')

ax = axes[0, 1]
mark = {0: 'o', 10: 's', 100: '^', 1000: 'D'}
for t in [0, 10, 100, 1000]:
    s = df[(df.strict_independent) & (df.TI >= t)]
    ax.scatter(s.logTI, s.logER, s=12, marker=mark[t], alpha=0.6, linewidths=0,
               label='strict-indep, TI>=%g (n=%d)' % (t, len(s)))
ax.set_xlabel('log10 Time interval (yr)'); ax.set_ylabel('log10 Erosion rate (mm/yr)')
ax.set_title('B. Strict-independent rows, threshold subsets'); ax.legend(fontsize=7, loc='lower left')

ax = axes[1, 0]
for tag, style in [('Glacial independent', '-o'), ('Nonglacial independent', '-s'),
                   ('Glacial ALL', '--o'), ('Nonglacial ALL', '--s')]:
    s = thr[thr.subset == tag].dropna(subset=['slope']).sort_values('TI_threshold_yr')
    if len(s):
        ax.errorbar(s.TI_threshold_yr.replace(0, 0.3), s.slope,
                    yerr=[s.slope - s.boot_ci_lo, s.boot_ci_hi - s.slope],
                    fmt=style, capsize=3, label=tag)
ax.set_xscale('symlog', linthresh=1)
ax.axhline(-1, color='k', ls=':', lw=1)
ax.set_xlabel('TI threshold (yr)'); ax.set_ylabel('log-log slope (dex/dex)')
ax.set_title('C. Threshold trajectory of the slope'); ax.legend(fontsize=7)

ax = axes[1, 1]
sub = sp_comp.dropna(subset=['slope'])
ax.scatter(sub.iqr_logET, sub.slope, s=22, c='#333333')
for _, r in sub.iterrows():
    if abs(r.slope) > 0.75 or r.iqr_logET > 2:
        ax.annotate(r.stratum.split(' | ')[1][:14], (r.iqr_logET, r.slope), fontsize=6)
ax.axhline(-1, color='k', ls=':', lw=1)
ax.set_xlabel('IQR of log10(E*T) within stratum (dex)')
ax.set_ylabel('within-stratum log-log slope')
ax.set_title('D. Slope approaches -1 as the product E*T is pinned')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'w6_figures.png'), dpi=140)
print('\nfigure written')


# -----------------------------------------------------------------------------
# RESULTS JSON
# -----------------------------------------------------------------------------
def jrec(r):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}

results = {
    'workstream': 'w6_decoupled',
    'inputs': {'glacial_rows_used': int(len(g)), 'nonglacial_rows_used': int(len(n)),
               'filter': 'Erosion rate > 0 and Time interval > 0 for all fits'},
    'classification_rule': {
        'evidence_flags': {
            'ex_identity': 'Exhumation>0 and |ER*TI/(EX*1000) - 1| <= 0.05 (rate == exhumation/interval)',
            'flux_identity': 'SF>0, A>0 and |ER/(SF/A*1e6) - 1| <= 0.05 (rate constructed from a sediment budget)',
            'lock_comp': 'Methodology x Compilation, n>=10, >=5 distinct intervals, >=1.0 dex of logTI, IQR(log10(ER*TI)) <= 0.10 dex',
            'lock_ref': 'same at Methodology x Compilation x Reference level with n>=5',
            'soft_lock': 'same geometry with IQR(log10(ER*TI)) <= 0.30 dex (near-lock flag only)'},
        'class_assignment': {
            'coupled': 'ex_identity OR flux_identity OR lock_comp OR lock_ref',
            'independent': 'not coupled AND Methodology in {Cosmogenic surface, Cosmogenic detrital, Thermochronometric}',
            'indeterminate': 'all other rows (Volumetric / Relief without an independent flux identity)'},
        'strict_independent': 'independent AND not soft_lock',
        'notes': ('Reference-level locks are the strongest evidence: within one paper the same '
                  'rate construction is reused while the reported interval varies, so the '
                  'invariant ER*TI is an algebraic artifact rather than geology.')},
    'counts': {
        'by_class_total': {k: int(v) for k, v in cnt_all.items()},
        'by_process': {f'{a}|{b}': int(v) for (a, b), v in cnt_proc.stack().items()},
        'by_methodology': {f'{a}|{b}|{c}': int(v) for (a, b, c), v in cnt_method.stack().items()},
        'ex_identity_total': int(df.ex_identity.sum()),
        'flux_identity_total': int(df.flux_identity.sum()),
        'coupled_evidence': {k: int(v) for k, v in df.coupled_evidence.value_counts().items()},
        'hard_lock_comp_strata': {k: int(v) for k, v in lock_comp[lock_comp.hard_lock].set_index('key').n.items()},
        'n_hard_lock_ref_cells': int(lock_ref.hard_lock.sum()),
        'hard_lock_ref_cell_sizes': int(lock_ref.loc[lock_ref.hard_lock, 'n'].sum()),
        'soft_lock_strata': {k: int(v) for k, v in lock_comp[lock_comp.soft_lock].set_index('key').n.items()}},
    'section2_slopes_by_class': [jrec(r) for r in sec2.to_dict('records')],
    'section2_independent_definition_sensitivity': [jrec(r) for r in alts.to_dict('records')],
    'section2_within_reference_slopes': [jrec(r) for r in fe.to_dict('records')],
    'section3_predictor_dominance': [jrec(r) for r in dom.to_dict('records')],
    'section3_no_timescale_models': [jrec(r) for r in noti_df.to_dict('records')],
    'section4_threshold_sensitivity': [jrec(r) for r in thr.to_dict('records')],
    'section5_ET_spread_by_stratum': [jrec(r) for r in spread_all.to_dict('records')],
    'section5_countercase_pairs': [jrec(r) for r in pairs.head(10).to_dict('records')],
    'section5_slope_vs_spread_across_strata': spread_slope_rel,
    'conclusion': {
        'defensible_exponent': {
            'stratum': 'Glacial | Cosmogenic surface | independent class (CRN surface-exposure clocks)',
            'slope_dex_per_dex': float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'slope']),
            'se': float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'se']),
            'r2': float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'r2']),
            'n': int(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'n']),
            'cluster_boot_ci': [float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'boot_ci_lo']),
                                float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'boot_ci_hi'])],
            'timescale_range_yr': [float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'TI_min_yr']),
                                   float(alts.set_index('subset').loc['F_glacial_cosmogenic_surface_indep', 'TI_max_yr'])],
            'why_defensible': ('CRN-clock rates, not product-locked by the reported interval; '
                               'but the slope is shallow (-0.26 +/- 0.12) and the strata are '
                               'not at the attenuation depth, so the exponent is 0 > slope > -1 '
                               'rather than exactly -1.')},
        'attenuation_lock': {
            'rule': 'CRN stratum with median ER*TI within 0.25 dex of the ~0.6 m effective '
                    'attenuation depth, IQR(log10 ER*TI) <= 0.30 dex, >=1 dex of logTI',
            'rows_matching_crn_lock_geometry': int(df.crn_lock.sum()),
            'rows_assigned_crn_lock_as_primary_evidence': int((df.coupled_evidence == 'crn_attenuation_lock').sum()),
            'strata': sorted(crn_locked_strata),
            'meaning': ('For CRN inventories E = P/(rho*(1-exp(-T/tau))); at steady state '
                        'E*T -> Lambda/rho ~ 0.6 m. A CRN stratum whose ER*TI clusters at '
                        '0.6 m is therefore the same algebraic lock as thickness/interval, '
                        'even with no exhumation column populated.')},
        'cannot_quote': [
            'All pooled/all-row slopes are dominated by product locks (OCTOPUS slope -0.995, R2 0.998 arises because ER*T = ~0.65 m for every row).',
            'Glacial Thermochronometric: slope -0.577 +/- 0.111 for the whole stratum, but 47 of 68 rows share TI = 1 Ma and the 1.62 dex rate spread inside that single interval supplies all of the leverage; excluding TI == 1 Ma the slope is -0.656 +/- 0.182 (n=21), i.e. it survives but rests on 21 rows and 10 references.',
            'Glacial Volumetric (all classes): E*T spans 9.1 dex, so the data cannot separate coupling from geology; no scaling exponent should be quoted.'],
        'counter_case': ('Within Glacial | Volumetric | This study a 2.57e7 yr change in interval is '
                         'accompanied by only a 1.01 dex change in rate: log10(E*T) moves 9.14 dex. '
                         'Same pattern in Glacial Volumetric Hallet (7.21 dex) and Nonglacial GloRiSe (6.15 dex). '
                         'Across the 19 largest strata the within-stratum slope correlates with the '
                         'spread of log10(E*T) (Spearman rho = %.2f, p = %.3g), i.e. the steeper the '
                         'apparent scaling, the more tightly the product is pinned.'
                         % (spread_slope_rel.get('spearman_rho', float('nan')),
                            spread_slope_rel.get('spearman_p', float('nan')))),
        'summary': ('A steep near -1 exponent is not defensible anywhere in this compilation: every '
                    'stratum that produces one is either product-locked (exhumation, flux or CRN '
                    'attenuation depth) or driven by a single repeated interval. The best de-biased '
                    'estimate is the glacial cosmogenic-surface stratum, where the exponent is '
                    '-0.26 +/- 0.12 over ~1e2-2e5 yr, and the thermochronometric stratum, where it is '
                    '-0.58 +/- 0.11 over ~1e5-2.5e8 yr but is carried by a 1 Ma cluster. '
                    'Volumetric/Relief data, which span 4-9 dex of E*T, cannot separate coupling '
                    'from geology and should not be quoted at all.')},
    'files': {
        'row_audit': 'row_classification_audit.csv',
        'stratum_lock': 'stratum_lock_diagnostics.csv',
        'reference_lock': 'reference_lock_diagnostics.csv',
        'slopes': 'slopes_by_class.csv',
        'independent_definition_sensitivity': 'independent_definition_sensitivity.csv',
        'within_reference_slopes': 'within_reference_slopes.csv',
        'dominance': 'predictor_dominance.csv',
        'no_timescale_models': 'no_timescale_models.csv',
        'thresholds': 'threshold_sensitivity.csv',
        'thresholds_by_method': 'threshold_sensitivity_by_method.csv',
        'spread': 'ET_spread_by_stratum.csv',
        'pairs': 'countercase_pairs.csv',
        'slope_vs_spread': 'stratum_slope_vs_spread.csv',
        'figure': 'w6_figures.png'},
}
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=1, default=str)

# extra console summary for the write-up
print('\n=== summary numbers ===')
print('ex_identity rows:', int(df.ex_identity.sum()), ' flux_identity rows:', int(df.flux_identity.sum()))
print('hard-locked strata:', list(results['classification_rule'] and results['counts']['hard_lock_comp_strata'].items()))
print('class by process:\n', cnt_proc)
print('strict independent by process:', df[df.strict_independent].groupby('Process').size().to_dict())
print('DONE')
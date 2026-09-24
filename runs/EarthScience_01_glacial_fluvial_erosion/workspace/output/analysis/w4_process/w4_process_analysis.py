#!/usr/bin/env python3
"""
w4_process : process-category effect conditional on scale and method.
Workstream 4 of the glacial vs nonglacial erosion-rate benchmark task.

Question: do glacial and nonglacial erosion rates differ, and under what conditions?

Everything is computed from the two frozen TSVs under data/raw/.
No external data, no network. Deterministic (fixed RNG seed).

Run:  cd <project root> && python3 output/analysis/w4_process/w4_process_analysis.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[3]          # .../project
RAW = BASE / 'data' / 'raw'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20240919
NBOOT = 2000
KNN = 5

NUMCOLS = ['Area (km2)', 'Time interval (yr)', 'Erosion rate (mm/yr)',
           'Slope (m/km)', 'Precipitation (mm/yr)', 'Latitude',
           'Exhumation (m)', 'Elevation (m)', 'Sediment flux (km3/yr)']

rng = np.random.default_rng(RNG_SEED)


def num(s):
    """Canonical numeric parse: blank string -> NaN."""
    return pd.to_numeric(pd.Series(s, dtype='object').astype(str).str.strip()
                         .replace({'': np.nan, 'nan': np.nan, 'None': np.nan}),
                         errors='coerce')


def load(path, process):
    d = pd.read_csv(path, sep='\t', dtype=str)
    for c in NUMCOLS:
        d[c] = num(d[c]) if c in d.columns else np.nan
    d['Process'] = process
    return d


def jf(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ============================================================================
# load + derived variables
# ============================================================================
g = load(RAW / 'glacial_erosion_Earth.tsv', 'Glacial')
n = load(RAW / 'nonglacial_erosion_Earth.tsv', 'Nonglacial')
KEEP = ['Compilation', 'Reference', 'Locality', 'Region', 'Methodology', 'Type',
        'Area (km2)', 'Time interval (yr)', 'Erosion rate (mm/yr)',
        'Slope (m/km)', 'Precipitation (mm/yr)', 'Latitude', 'Exhumation (m)',
        'Process']
both = pd.concat([g[KEEP], n[KEEP]], ignore_index=True)

both['ER'] = both['Erosion rate (mm/yr)']
both['TI'] = both['Time interval (yr)']
both['logER'] = np.where(both['ER'] > 0, np.log10(both['ER'].where(both['ER'] > 0)), np.nan)
both['logTI'] = np.where(both['TI'] > 0, np.log10(both['TI'].where(both['TI'] > 0)), np.nan)
both['logA'] = np.where(both['Area (km2)'] > 0, np.log10(both['Area (km2)'].where(both['Area (km2)'] > 0)), np.nan)
both['logS'] = np.where(both['Slope (m/km)'] > 0, np.log10(both['Slope (m/km)'].where(both['Slope (m/km)'] > 0)), np.nan)
both['logP'] = np.where(both['Precipitation (mm/yr)'] > 0, np.log10(both['Precipitation (mm/yr)'].where(both['Precipitation (mm/yr)'] > 0)), np.nan)
both['absLat'] = both['Latitude'].abs()
both['ET'] = both['ER'] * both['TI']

glog = both.loc[both['Process'] == 'Glacial', 'logER'].dropna()
nlog = both.loc[both['Process'] == 'Nonglacial', 'logER'].dropna()
glogti = both.loc[both['Process'] == 'Glacial', 'logTI'].dropna()
nlogti = both.loc[both['Process'] == 'Nonglacial', 'logTI'].dropna()
MARGINAL = float(glog.mean() - nlog.mean())

res = {'meta': {
    'workstream': 'w4_process',
    'title': 'process-category effect conditional on scale and method',
    'glacial_rows': int(len(g)), 'nonglacial_rows': int(len(n)),
    'knn_k': KNN, 'nboot': NBOOT, 'seed': RNG_SEED,
    'process_label_contrast': 'positive value = Glacial rate HIGHER than Nonglacial (dex)',
    'filters': {
        'logER': 'rows with Erosion rate (mm/yr) > 0 (427/427 glacial, 6284/6288 nonglacial)',
        'logTI': 'rows with Time interval (yr) > 0',
        'matched': f'each glacial row with finite logER+logTI matched to k={KNN} nearest '
                   'nonglacial neighbours on |dlogTI| within the same Methodology '
                   '(k reduced to n available if fewer)',
        'regression': 'complete cases on logER, logTI, Process, Methodology'}},
    'verification': {}, 'part1_marginal': {}, 'part2_method_matched': {},
    'part3_scale_matched': {}, 'part4_stratified_regression': {},
    'part5_confounding': {}, 'part6_controls': {}, 'part7_reversals': {}}


def desc(x):
    x = np.asarray(x, dtype=float)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=int(len(x)), mean=jf(x.mean()), median=jf(q2), sd=jf(x.std(ddof=1)),
                q1=jf(q1), q3=jf(q3), iqr=jf(q3 - q1), min=jf(x.min()), max=jf(x.max()),
                p10=jf(np.percentile(x, 10)), p90=jf(np.percentile(x, 90)))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return dict(n_a=int(na), n_b=int(nb), mean_a=jf(a.mean()), mean_b=jf(b.mean()),
                mean_diff_dex=jf(a.mean() - b.mean()), t=jf(t), p=jf(p),
                cohens_d=jf((a.mean() - b.mean()) / sp if sp > 0 else np.nan),
                median_diff_dex=jf(np.median(a) - np.median(b)),
                ratio_10pow_mean=jf(10 ** (a.mean() - b.mean())))


def cluster_se_mean(d, groups):
    """CR1 cluster-robust SE of the mean, clusters=groups."""
    d = np.asarray(d, float)
    gs = pd.Series(groups)
    n_ = len(d)
    G = gs.nunique()
    if G < 2:
        return np.nan
    s = pd.Series(d - d.mean()).groupby(gs.values).sum()
    corr = (G / (G - 1)) * ((n_ - 1) / (n_ - 1))
    return float(np.sqrt(corr * (s ** 2).sum() / n_ ** 2))


# ============================================================================
# 0. verification of structural facts
# ============================================================================
v = {}
ex = both[both['Exhumation (m)'].notna() & both['ER'].notna() & both['TI'].notna()].copy()
ex['prod_mm'] = ex['ER'] * ex['TI']
ex['ratio_m_over_ex'] = ex['prod_mm'] / 1000.0 / ex['Exhumation (m)']
ex['ratio_mm_over_ex'] = ex['prod_mm'] / ex['Exhumation (m)']
v['exhumation_rows_with_all3'] = int(len(ex))
v['exhumation_rows_by_process'] = {k: int(x) for k, x in ex['Process'].value_counts().items()}
v['n_rows_where_ratio_m_over_ex_within_1pct'] = int(
    np.isclose(ex['ratio_m_over_ex'], 1.0, rtol=0.01).sum())
v['n_rows_where_ratio_m_over_ex_within_0.1pct'] = int(
    np.isclose(ex['ratio_m_over_ex'], 1.0, rtol=0.001).sum())
v['median_ratio_m_over_ex'] = jf(ex['ratio_m_over_ex'].median())
v['median_ratio_mm_over_ex'] = jf(ex['ratio_mm_over_ex'].median())
ex['ratio_rounded_1pct'] = np.isclose(ex['ratio_m_over_ex'], 1.0, rtol=0.01)
ex['ratio_as_mm'] = np.isclose(ex['ratio_m_over_ex'], 1000.0, rtol=0.1)
v['n_rows_ratio_in_0.99_1.01'] = int(((ex['ratio_m_over_ex'] >= 0.99) & (ex['ratio_m_over_ex'] <= 1.01)).sum())
v['n_rows_ratio_in_900_1100_stored_as_mm'] = int(((ex['ratio_m_over_ex'] >= 900) & (ex['ratio_m_over_ex'] <= 1100)).sum())
v['ratio_identity_by_process'] = json.loads(
    pd.crosstab(ex['Process'], ex['ratio_rounded_1pct']).to_json(orient='index'))
v['ratio_identity_by_method'] = json.loads(
    pd.crosstab([ex['Process'], ex['Methodology']], ex['ratio_rounded_1pct']).to_json(orient='index'))
v['exhumation_note'] = (
    'PARTIALLY CONFIRMED, NOT "exactly". ER(mm/yr)*TI(yr)/1000 (proper m conversion) equals the '
    f'column labelled "Exhumation (m)" within 1% for {v["n_rows_where_ratio_m_over_ex_within_1pct"]}/{len(ex)} '
    f'rows (within 0.1%: {v["n_rows_where_ratio_m_over_ex_within_0.1pct"]}); median ratio '
    f'{v["median_ratio_m_over_ex"]:.6f}. The un-converted product ER*TI equals the column directly '
    f'for 0/{len(ex)} rows. A MINORITY subset ({v["n_rows_ratio_in_900_1100_stored_as_mm"]}/{len(ex)}) has '
    'the column numerically equal to the RAW unconverted mm product, i.e. those rows carry a '
    'genuine mm-as-m unit error (all are nonglacial Rd 7/11 nonglacial rows). The remaining rows fail '
    'both identities because rate/interval were independently rounded before deposition. Net: the '
    'column is mostly (but not exactly) ER*TI in metres, with a small mm-contaminated subset.')

slope_check = []
for proc, sub in both.groupby('Process'):
    for meth, s2 in sub.groupby('Methodology'):
        s2 = s2.dropna(subset=['logER', 'logTI'])
        if len(s2) >= 10:
            m = smf.ols('logER ~ logTI', data=s2).fit()
            slope_check.append({'Process': proc, 'Methodology': meth, 'n': int(len(s2)),
                                'slope_dex_per_dex': jf(m.params['logTI']),
                                'se': jf(m.bse['logTI']), 'p': jf(m.pvalues['logTI']),
                                'r2': jf(m.rsquared)})
v['logER_logTI_slopes_by_stratum'] = slope_check
pd.DataFrame(slope_check).to_csv(OUT / 'verification_ER_TI_slopes.csv', index=False)
res['verification'] = v

# ============================================================================
# 1. marginal comparison of logER
# ============================================================================
marg = {'Glacial': desc(glog), 'Nonglacial': desc(nlog)}
wt = welch(glog, nlog)
u, pu = st.mannwhitneyu(glog, nlog, alternative='two-sided')
rb = 1 - 2 * u / (len(glog) * len(nlog))
marg['welch_t'] = dict(n_a=wt['n_a'], n_b=wt['n_b'], mean_diff_dex=wt['mean_diff_dex'],
                       t=wt['t'], p=wt['p'], cohens_d=wt['cohens_d'],
                       ratio=10 ** wt['mean_diff_dex'],
                       ci95_mean_diff_lo=jf(wt['mean_diff_dex'] - 1.96 * np.sqrt(
                           glog.var(ddof=1) / len(glog) + nlog.var(ddof=1) / len(nlog))),
                       ci95_mean_diff_hi=jf(wt['mean_diff_dex'] + 1.96 * np.sqrt(
                           glog.var(ddof=1) / len(glog) + nlog.var(ddof=1) / len(nlog))))
marg['mannwhitney'] = dict(U=jf(u), p=jf(pu), rank_biserial=jf(rb))
marg['median_diff_dex'] = jf(np.median(glog) - np.median(nlog))
marg['median_ratio'] = jf(10 ** (np.median(glog) - np.median(nlog)))
marg['cluster_robust_se_glacial_reference'] = jf(cluster_se_mean(
    glog.values, both.loc[glog.index, 'Reference'].values))
marg['cluster_robust_se_nonglacial_reference'] = jf(cluster_se_mean(
    nlog.values, both.loc[nlog.index, 'Reference'].values))
res['part1_marginal'] = marg

# ============================================================================
# 2. method-matched comparison + per-type breakdowns
# ============================================================================
mm = []
for meth in sorted(both['Methodology'].dropna().unique()):
    a = both.loc[(both['Methodology'] == meth) & (both['Process'] == 'Glacial'), 'logER'].dropna()
    b = both.loc[(both['Methodology'] == meth) & (both['Process'] == 'Nonglacial'), 'logER'].dropna()
    rec = {'Methodology': meth, 'n_glacial': int(len(a)), 'n_nonglacial': int(len(b))}
    if len(a) >= 2 and len(b) >= 2:
        rec.update(welch(a, b))
        rec['mannwhitney_p'] = jf(st.mannwhitneyu(a, b, alternative='two-sided')[1])
        rec['cluster_robust_se_diff'] = jf(np.sqrt(
            cluster_se_mean(a.values, both.loc[a.index, 'Reference'].values) ** 2 +
            cluster_se_mean(b.values, both.loc[b.index, 'Reference'].values) ** 2))
        # reference-clustered 95% CI
        rec['ci95_lo'] = jf(rec['mean_diff_dex'] - 1.96 * rec['cluster_robust_se_diff'])
        rec['ci95_hi'] = jf(rec['mean_diff_dex'] + 1.96 * rec['cluster_robust_se_diff'])
    else:
        rec['note'] = 'insufficient n in one arm'
    mm.append(rec)
res['part2_method_matched']['by_methodology'] = mm
pd.DataFrame(mm).to_csv(OUT / 'method_matched_logER.csv', index=False)

tb = []
for proc in ['Glacial', 'Nonglacial']:
    for typ, s in both[both['Process'] == proc].groupby('Type'):
        x = s['logER'].dropna()
        tb.append({'Process': proc, 'Type': typ, **desc(x)})
res['part2_method_matched']['by_type'] = tb
pd.DataFrame(tb).to_csv(OUT / 'type_logER_stats.csv', index=False)

# glacial type vs nonglacial Fluvial, method-matched
fluvial = both[(both['Process'] == 'Nonglacial') & (both['Type'] == 'Fluvial')]
tb2 = []
for typ, s in both[both['Process'] == 'Glacial'].groupby('Type'):
    for meth in sorted(both['Methodology'].dropna().unique()):
        a = s.loc[s['Methodology'] == meth, 'logER'].dropna()
        b = fluvial.loc[fluvial['Methodology'] == meth, 'logER'].dropna()
        if len(a) >= 2 and len(b) >= 2:
            r = {'Glacial Type': typ, 'Methodology': meth, 'n_glacial': int(len(a)),
                 'n_nonglacial_fluvial': int(len(b))}
            r.update(welch(a, b))
            tb2.append(r)
res['part2_method_matched']['glacial_type_vs_fluvial_by_method'] = tb2
pd.DataFrame(tb2).to_csv(OUT / 'glacial_type_vs_fluvial.csv', index=False)

tb3 = []
for typ, s in both[both['Process'] == 'Glacial'].groupby('Type'):
    a = s['logER'].dropna()
    tb3.append({'Arm': 'Glacial/' + typ, 'compared_to': 'all Nonglacial',
                'median_ratio': jf(10 ** (np.median(a) - np.median(nlog))), **welch(a, nlog)})
for typ, s in both[both['Process'] == 'Nonglacial'].groupby('Type'):
    a = s['logER'].dropna()
    tb3.append({'Arm': 'Nonglacial/' + typ, 'compared_to': 'all Glacial',
                'median_ratio': jf(10 ** (np.median(a) - np.median(glog))), **welch(a, glog)})
res['part2_method_matched']['type_vs_other_process'] = tb3
pd.DataFrame(tb3).to_csv(OUT / 'type_vs_other_process.csv', index=False)

# ============================================================================
# 3. scale-matched comparison (kNN on logTI within Methodology)
# ============================================================================
MATCHCOLS = ['logTI', 'logA', 'logS', 'logP', 'absLat']


def knn_pairs(meth, k=KNN, caliper=None, keys=('logTI',)):
    sc = both[(both['Methodology'] == meth) & both['logER'].notna() & both['logTI'].notna()]
    gs, ns = sc[sc['Process'] == 'Glacial'].copy(), sc[sc['Process'] == 'Nonglacial'].copy()
    if len(gs) == 0 or len(ns) == 0:
        return pd.DataFrame(), len(gs), len(ns)
    nm = ns[list(keys)].to_numpy(dtype=float)
    out = []
    for _, row in gs.iterrows():
        d = np.abs(nm - row[list(keys)].to_numpy(dtype=float)).sum(axis=1)
        o = np.argsort(d, kind='stable')
        if caliper is not None:
            o = o[d[o] <= caliper]
            if len(o) == 0:
                continue
        sel = o[:min(k, len(o))]
        nsel = ns.iloc[sel]
        rec = {c: row[c] for c in ['Reference', 'Compilation', 'Type', 'logER', 'logTI'] + MATCHCOLS}
        rec.update({f'nbr_{c}': nsel[c].mean() for c in ['logER'] + MATCHCOLS})
        rec['k_used'] = int(len(sel))
        rec['dlogTI'] = float(row['logTI'] - nsel['logTI'].mean())
        out.append(rec)
    return pd.DataFrame(out), len(gs), len(ns)


def matched_test(df, label, groups_col='Reference'):
    if len(df) == 0:
        return {'label': label, 'n': 0, 'note': 'no matched rows'}
    d = (df['logER'] - df['nbr_logER']).to_numpy(dtype=float)
    t, p = st.ttest_1samp(d, 0.0)
    idx = np.arange(len(d))
    # iid row bootstrap (vectorised)
    boot = d[rng.integers(0, len(idx), size=(NBOOT, len(idx)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # cluster bootstrap on glacial Reference (vectorised via cumulative sums)
    refs = df[groups_col].to_numpy()
    uref, inv = np.unique(refs, return_inverse=True)
    G = len(uref)
    ordr = np.argsort(inv, kind='stable')
    ds = d[ordr]
    cnt = np.bincount(inv, minlength=G)
    starts = np.concatenate([[0], np.cumsum(cnt)])
    gsum = np.array([ds[starts[i]:starts[i + 1]].sum() for i in range(G)])
    gcnt = cnt.astype(float)
    picks = rng.integers(0, G, size=(NBOOT, G))
    cb = gsum[picks].sum(axis=1) / gcnt[picks].sum(axis=1)
    cl_se = cluster_se_mean(d, refs)
    return {'label': label, 'n': int(len(d)),
            'mean_diff_dex': jf(d.mean()), 'median_diff_dex': jf(np.median(d)),
            'sd_diff': jf(d.std(ddof=1)), 'se': jf(d.std(ddof=1) / np.sqrt(len(d))),
            'cluster_robust_se': jf(cl_se),
            't': jf(t), 'p': jf(p),
            't_cluster': jf(d.mean() / cl_se) if cl_se and cl_se > 0 else None,
            'p_cluster_normal_approx': jf(2 * st.norm.sf(abs(d.mean() / cl_se)))
            if cl_se and cl_se > 0 else None,
            'n_reference_clusters_glacial': int(len(uref)),
            'boot_ci95_lo': jf(lo), 'boot_ci95_hi': jf(hi),
            'boot_cluster_by_reference_ci95': [jf(np.percentile(cb, 2.5)), jf(np.percentile(cb, 97.5))],
            'ratio_10pow_mean': jf(10 ** d.mean()), 'ratio_10pow_median': jf(10 ** np.median(d)),
            'frac_glacial_faster': jf((d > 0).mean()),
            'cohens_dz': jf(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
            'mean_abs_dlogTI': jf(np.abs(df['dlogTI']).mean()),
            'median_abs_dlogTI': jf(np.abs(df['dlogTI']).median()),
            'k_used_mean': jf(df['k_used'].mean()),
            'frac_pairs_tight_match_0.1dex': jf((np.abs(df['dlogTI']) <= 0.1).mean()),
            'n_glacial_supported': int(df['k_used'].sum()),
            'n_nonglacial_pool': None,
            'match_quality': 'DEGENERATE (k == entire nonglacial stratum; every glacial row '
                             'matched against the same handful of rows)'
            if df['k_used'].min() >= 4 and len(df['logER'].unique()) < len(df) * 0.5 else 'ok',
            'n_unique_nonglacial_matched': None}


scale_rows, all_pairs = [], {}
for meth in sorted(both['Methodology'].dropna().unique()):
    n_n = int(((both['Methodology'] == meth) & (both['Process'] == 'Nonglacial')).sum())
    k = int(min(KNN, n_n))
    df, ng, nn_ = knn_pairs(meth, k=k)
    all_pairs[meth] = df
    rec = matched_test(df, f'{meth} (k={k})') if len(df) else {'label': f'{meth} (k={k})', 'n': 0,
                                                              'note': 'no nonglacial rows'}
    rec.update({'Methodology': meth, 'n_glacial_eligible': int(ng),
                'n_nonglacial_eligible': int(nn_), 'k': k,
                'coverage_glacial': jf(len(df) / ng) if ng else None})
    scale_rows.append(rec)

matched_all = pd.concat([d for d in all_pairs.values() if len(d)], ignore_index=True)
pooled_rec = matched_test(matched_all, 'ALL methods pooled (k=5 within method)')
pooled_rec['Methodology'] = 'ALL'
scale_rows.append(pooled_rec)
res['part3_scale_matched']['per_methodology'] = scale_rows
pd.DataFrame(scale_rows).to_csv(OUT / 'scale_matched_knn5.csv', index=False)

# alternative matchers
alt = []
for meth in sorted(both['Methodology'].dropna().unique()):
    for lbl, kw in [('knn1', dict(k=1)), ('caliper0.25dex', dict(k=KNN, caliper=0.25)),
                    ('knn5_2D_logTI_logArea', dict(k=KNN, keys=('logTI', 'logA')))]:
        d_, ng_, nn_ = (knn_pairs(meth, **kw) if 'keys' not in kw else
                        knn_pairs(meth, k=kw['k'], keys=kw['keys']))
        alt.append({**matched_test(d_, f'{meth} {lbl}'), 'variant': lbl, 'Methodology': meth})
# pooled across methods, logTI only
sc = both[both['logER'].notna() & both['logTI'].notna()]
ns_all = sc[sc['Process'] == 'Nonglacial']
nm_all = ns_all[['logTI']].to_numpy(float)
pp = []
for _, row in sc[sc['Process'] == 'Glacial'].iterrows():
    d_ = np.abs(nm_all[:, 0] - float(row['logTI']))
    sel = np.argsort(d_, kind='stable')[:KNN]
    nsel = ns_all.iloc[sel]
    pp.append({'Reference': row['Reference'], 'Type': row['Type'], 'logER': row['logER'],
               'logTI': row['logTI'], 'nbr_logER': nsel['logER'].mean(),
               'nbr_logTI': nsel['logTI'].mean(), 'k_used': len(sel),
               'dlogTI': float(row['logTI'] - nsel['logTI'].mean())})
alt.append({**matched_test(pd.DataFrame(pp), 'ALL pooled across methods, logTI only'),
            'variant': 'pooled_knn5_ti', 'Methodology': 'ALL'})
# reverse direction: anchor on nonglacial rows (asymmetry check)
for meth in sorted(both['Methodology'].dropna().unique()):
    sc2 = both[(both['Methodology'] == meth) & both['logER'].notna() & both['logTI'].notna()]
    gs2, ns2 = sc2[sc2['Process'] == 'Glacial'], sc2[sc2['Process'] == 'Nonglacial']
    if len(gs2) == 0 or len(ns2) == 0:
        continue
    gm = gs2[['logTI']].to_numpy(float)
    dd = np.array([gs2.iloc[np.argsort(np.abs(gm[:, 0] - float(r['logTI'])), kind='stable')
                            [:min(KNN, len(gm))]]['logER'].mean() - r['logER']
                   for _, r in ns2.iterrows()])
    t_, p_ = st.ttest_1samp(dd, 0.0)
    alt.append({'variant': 'reverse_nonglacial_anchored', 'Methodology': meth, 'n': int(len(dd)),
                'mean_diff_dex': jf(dd.mean()), 'median_diff_dex': jf(np.median(dd)),
                't': jf(t_), 'p': jf(p_), 'ratio_10pow_mean': jf(10 ** dd.mean())})
res['part3_scale_matched']['alternative_matchers'] = alt
pd.DataFrame(alt).to_csv(OUT / 'scale_matched_alternatives.csv', index=False)

# TI decile bins (no nearest-neighbour weighting)
both['TIbin'] = pd.qcut(both['logTI'], 10, duplicates='drop')
binned = []
for b, s in both.dropna(subset=['TIbin']).groupby('TIbin', observed=True):
    a, c = s.loc[s['Process'] == 'Glacial', 'logER'].dropna(), s.loc[s['Process'] == 'Nonglacial', 'logER'].dropna()
    if len(a) >= 2 and len(c) >= 2:
        binned.append({'TI_bin': str(b), 'logTI_lo': jf(b.left), 'logTI_hi': jf(b.right),
                       'median_TI_yr': jf(10 ** s['logTI'].median()),
                       'n_glacial': int(len(a)), 'n_nonglacial': int(len(c)), **welch(a, c)})
res['part3_scale_matched']['within_TI_decile'] = binned
pd.DataFrame(binned).to_csv(OUT / 'within_TI_decile.csv', index=False)

# per-Type matched comparison inside Volumetric
vt = []
for typ, s in both[(both['Process'] == 'Glacial') & (both['Methodology'] == 'Volumetric')].groupby('Type'):
    sub = pd.concat([s, both[(both['Process'] == 'Nonglacial') & (both['Methodology'] == 'Volumetric')]])
    sub = sub[sub['logER'].notna() & sub['logTI'].notna()]
    gs_, ns_ = sub[sub['Process'] == 'Glacial'], sub[sub['Process'] == 'Nonglacial']
    nm = ns_[['logTI']].to_numpy(float)
    dd, tl = [], []
    for _, row in gs_.iterrows():
        sel = np.argsort(np.abs(nm[:, 0] - float(row['logTI'])), kind='stable')[:min(KNN, len(nm))]
        dd.append(row['logER'] - ns_.iloc[sel]['logER'].mean())
        tl.append(float(row['logTI'] - ns_.iloc[sel]['logTI'].mean()))
    dd = np.array(dd)
    if len(dd) >= 2:
        t_, p_ = st.ttest_1samp(dd, 0.0)
        vt.append({'Type': typ, 'Methodology': 'Volumetric', 'n': int(len(dd)),
                   'mean_diff_dex': jf(dd.mean()), 'median_diff_dex': jf(np.median(dd)),
                   't': jf(t_), 'p': jf(p_), 'ratio_10pow_mean': jf(10 ** dd.mean()),
                   'mean_abs_dlogTI': jf(np.mean(np.abs(tl)))})
res['part3_scale_matched']['volumetric_by_glacial_type'] = vt
pd.DataFrame(vt).to_csv(OUT / 'volumetric_by_glacial_type.csv', index=False)

# ---- common-support restriction (the honest overlap window) -----------------
# overlap window = intersection of the two logTI central ranges
comm_lo = max(np.percentile(glogti, 5), np.percentile(nlogti, 5))
comm_hi = min(np.percentile(glogti, 95), np.percentile(nlogti, 95))
cs = both[(both['logTI'] >= comm_lo) & (both['logTI'] <= comm_hi) &
          both['logER'].notna() & both['logTI'].notna()]
csa = cs.loc[cs['Process'] == 'Glacial', 'logER'].dropna()
csb = cs.loc[cs['Process'] == 'Nonglacial', 'logER'].dropna()
cs_pairs = pd.concat([knn_pairs(m, k=min(KNN, int(((cs['Methodology'] == m) &
                                                   (cs['Process'] == 'Nonglacial')).sum())))
                      [0] for m in sorted(cs['Methodology'].dropna().unique())
                      if ((cs['Methodology'] == m) & (cs['Process'] == 'Nonglacial')).sum() > 0
                      and ((cs['Methodology'] == m) & (cs['Process'] == 'Glacial')).sum() > 0],
                     ignore_index=True)
res['part3_scale_matched']['common_support'] = {
    'logTI_range': [jf(comm_lo), jf(comm_hi)],
    'TI_yr_range': [jf(10 ** comm_lo), jf(10 ** comm_hi)],
    'n_glacial': int(len(csa)), 'n_nonglacial': int(len(csb)),
    'marginal_diff_dex': jf(csa.mean() - csb.mean()),
    'welch': welch(csa, csb),
    'matched_diff_dex': jf((cs_pairs['logER'] - cs_pairs['nbr_logER']).mean()),
    'matched': matched_test(cs_pairs, 'common support, kNN within method') if len(cs_pairs) else None,
    'share_of_glacial_retained': jf(len(csa) / len(glog)),
    'share_of_nonglacial_retained': jf(len(csb) / len(nlog))}

# ---------------------------------------------------------------------------
# 3c. Volumetric sign-conflict diagnosis (matched says glacial-faster; the
#     linear-in-logTI regression says nonglacial-faster; why?)
# ---------------------------------------------------------------------------
vol = both[(both['Methodology'] == 'Volumetric') & both['logER'].notna()
           & both['logTI'].notna()].copy()
vg = vol[vol['Process'] == 'Glacial']
vn = vol[vol['Process'] == 'Nonglacial']
vol_diag = {
    'n_glacial': int(len(vg)), 'n_nonglacial': int(len(vn)),
    'glacial_logTI_q': {q: jf(np.percentile(vg['logTI'], p)) for q, p in
                        [('p05', 5), ('q25', 25), ('median', 50), ('q75', 75), ('p95', 95)]},
    'nonglacial_logTI_q': {q: jf(np.percentile(vn['logTI'], p)) for q, p in
                           [('p05', 5), ('q25', 25), ('median', 50), ('q75', 75), ('p95', 95)]},
    'glacial_logER_q': {q: jf(np.percentile(vg['logER'], p)) for q, p in
                        [('q25', 25), ('median', 50), ('q75', 75)]},
    'nonglacial_logER_q': {q: jf(np.percentile(vn['logER'], p)) for q, p in
                           [('q25', 25), ('median', 50), ('q75', 75)]},
    'note': 'The nonglacial Volumetric logTI distribution is strongly bimodal '
            '(median 0.70 dex with a wide IQR); the glacial one is unimodal at ~1.65 dex. '
            'A single linear logTI slope therefore cannot be a common adjustment, and the '
            'regression-adjusted contrast is not comparable to the matched contrast.'}
# TI-quantile-matched contrast inside Volumetric
qs = np.linspace(0, 1, 6)
vq = []
for i in range(5):
    lo_, hi_ = np.percentile(vn['logTI'], qs[i] * 100), np.percentile(vn['logTI'], qs[i + 1] * 100)
    band = vn[(vn['logTI'] >= lo_) & (vn['logTI'] <= hi_)]
    gg = vg[(vg['logTI'] >= lo_) & (vg['logTI'] <= hi_)]
    if len(gg) >= 2 and len(band) >= 2:
        vq.append({'nonglacial_logTI_quintile': i + 1, 'logTI_lo': jf(lo_), 'logTI_hi': jf(hi_),
                   'n_glacial': int(len(gg)), 'n_nonglacial': int(len(band)),
                   'mean_logER_glacial': jf(gg['logER'].mean()),
                   'mean_logER_nonglacial': jf(band['logER'].mean()),
                   'diff_dex': jf(gg['logER'].mean() - band['logER'].mean()),
                   'welch_p': jf(st.ttest_ind(gg['logER'], band['logER'], equal_var=False)[1])})
    else:
        vq.append({'nonglacial_logTI_quintile': i + 1, 'logTI_lo': jf(lo_), 'logTI_hi': jf(hi_),
                   'n_glacial': int(len(gg)), 'n_nonglacial': int(len(band)),
                   'note': 'insufficient rows in one arm'})
vol_diag['contrast_by_nonglacial_logTI_quintile'] = vq
vol_diag['glacial_rows_inside_nonglacial_TI_range'] = int(
    ((vg['logTI'] >= vn['logTI'].min()) & (vg['logTI'] <= vn['logTI'].max())).sum())
vol_diag['frac_glacial_inside_nonglacial_TI_range'] = jf(
    float(((vg['logTI'] >= vn['logTI'].min()) & (vg['logTI'] <= vn['logTI'].max())).mean()))
res['part3_scale_matched']['volumetric_sign_conflict_diagnosis'] = vol_diag
pd.DataFrame(vq).to_csv(OUT / 'volumetric_TI_quintile_contrast.csv', index=False)

# ============================================================================
# 4. stratified regression
# ============================================================================
regdf = both.dropna(subset=['logER', 'logTI', 'Process', 'Methodology']).copy()
regdf['Process'] = pd.Categorical(regdf['Process'], categories=['Glacial', 'Nonglacial'])
regdf['logTI_c'] = regdf['logTI'] - regdf['logTI'].mean()
P = 'C(Process, Treatment("Glacial"))[T.Nonglacial]'

m_main = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI + C(Methodology)', data=regdf).fit()
m_inter = smf.ols('logER ~ C(Process, Treatment("Glacial")) * logTI + C(Methodology)', data=regdf).fit()
m_inter_c = smf.ols('logER ~ C(Process, Treatment("Glacial")) * logTI_c + C(Methodology)', data=regdf).fit()
m_full = smf.ols('logER ~ C(Process, Treatment("Glacial")) * C(Methodology) * logTI', data=regdf).fit()
# cluster-robust (by Reference) for the interaction model
m_inter_cr = smf.ols('logER ~ C(Process, Treatment("Glacial")) * logTI_c + C(Methodology)',
                     data=regdf).fit(cov_type='cluster',
                                     cov_kwds={'groups': regdf['Reference']})


def ct(m, keys=None, label='', robust=False):
    o = []
    for k in m.params.index:
        if keys is None or any(r in k for r in keys):
            ci = m.conf_int().loc[k]
            o.append({'model': label, 'term': k, 'estimate': jf(m.params[k]), 'se': jf(m.bse[k]),
                      't': jf(m.tvalues[k]), 'p': jf(m.pvalues[k]),
                      'ci95_lo': jf(ci[0]), 'ci95_hi': jf(ci[1])})
    return o


reg_res = {'n': int(len(regdf))}
reg_res['model_main_coeffs'] = ct(m_main, ['Process', 'logTI'], 'logER ~ Process + logTI + Methodology')
reg_res['model_main_fit'] = {'n': int(m_main.nobs), 'r2': jf(m_main.rsquared),
                             'adj_r2': jf(m_main.rsquared_adj), 'f_p': jf(m_main.f_pvalue)}
reg_res['model_interaction_coeffs'] = ct(m_inter, ['Process', 'logTI'],
                                         'logER ~ Process*logTI + Methodology')
reg_res['model_interaction_fit'] = {'n': int(m_inter.nobs), 'r2': jf(m_inter.rsquared),
                                    'adj_r2': jf(m_inter.rsquared_adj), 'f_p': jf(m_inter.f_pvalue)}
reg_res['model_interaction_centered_coeffs'] = ct(
    m_inter_c, ['Process', 'logTI'], 'logER ~ Process*logTI_c + Methodology (logTI centered)')
reg_res['model_interaction_cluster_robust_coeffs'] = ct(
    m_inter_cr, ['Process', 'logTI'], 'same, SE clustered by Reference', robust=True)
reg_res['model_full_coeffs'] = ct(m_full, ['Process'], 'logER ~ Process*Methodology*logTI')
reg_res['model_full_fit'] = {'n': int(m_full.nobs), 'r2': jf(m_full.rsquared),
                             'adj_r2': jf(m_full.rsquared_adj)}
sl = []
for meth in sorted(regdf['Methodology'].unique()):
    sub = regdf[regdf['Methodology'] == meth]
    if sub['Process'].nunique() < 2 or len(sub) < 8:
        sl.append({'Methodology': meth, 'n': int(len(sub)),
                   'note': 'one process arm only or n<8'})
        continue
    mm2 = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI', data=sub).fit()
    ci = mm2.conf_int().loc[P]
    # cluster-robust
    cr = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['Reference']})
    sl.append({'Methodology': meth, 'n': int(len(sub)),
               'n_glacial': int((sub['Process'] == 'Glacial').sum()),
               'n_nonglacial': int((sub['Process'] == 'Nonglacial').sum()),
               'logTI_slope': jf(mm2.params['logTI']), 'logTI_slope_se': jf(mm2.bse['logTI']),
               'process_adj_diff_dex': jf(mm2.params[P]), 'se': jf(mm2.bse[P]),
               'p': jf(mm2.pvalues[P]), 'ci95_lo': jf(ci[0]), 'ci95_hi': jf(ci[1]),
               'se_cluster': jf(cr.bse[P]), 'p_cluster': jf(cr.pvalues[P]),
               'r2': jf(mm2.rsquared)})
reg_res['simple_slopes_by_methodology'] = sl
reg_res['incremental_r2'] = {
    'r2_method_only': jf(smf.ols('logER ~ C(Methodology)', data=regdf).fit().rsquared),
    'r2_logTI_only': jf(smf.ols('logER ~ logTI', data=regdf).fit().rsquared),
    'r2_process_only': jf(smf.ols('logER ~ C(Process, Treatment("Glacial"))', data=regdf).fit().rsquared),
    'r2_logTI_plus_method': jf(smf.ols('logER ~ logTI + C(Methodology)', data=regdf).fit().rsquared),
    'r2_TI_method_process': jf(m_main.rsquared),
    'delta_r2_from_process': jf(m_main.rsquared - smf.ols('logER ~ logTI + C(Methodology)', data=regdf).fit().rsquared),
    'r2_plus_process_x_logTI': jf(m_inter.rsquared),
    'delta_r2_from_interaction': jf(m_inter.rsquared - m_main.rsquared),
    'r2_full_3way': jf(m_full.rsquared)}
res['part4_stratified_regression'] = reg_res
pd.DataFrame(reg_res['model_interaction_coeffs']).to_csv(OUT / 'stratified_regression_coeffs.csv', index=False)
pd.DataFrame(sl).to_csv(OUT / 'simple_slopes_by_method.csv', index=False)

# ============================================================================
# 5. confounding: TI distributions, composition, Kitagawa-Oaxaca-Blinder
# ============================================================================
c = {}
gt, nt = both.loc[both['Process'] == 'Glacial', 'TI'].dropna(), both.loc[both['Process'] == 'Nonglacial', 'TI'].dropna()
lo = np.floor(min(glogti.min(), nlogti.min()) * 4) / 4
hi = np.ceil(max(glogti.max(), nlogti.max()) * 4) / 4
edges = np.arange(lo, hi + 0.25, 0.25)
pg, _ = np.histogram(glogti, bins=edges)
pn, _ = np.histogram(nlogti, bins=edges)
pg, pn = pg / pg.sum(), pn / pn.sum()
ovl = float(np.minimum(pg, pn).sum())
giqr, niqr = np.percentile(glogti, [25, 75]), np.percentile(nlogti, [25, 75])
c['TI'] = {'glacial': desc(gt), 'nonglacial': desc(nt),
           'glacial_logTI': desc(glogti), 'nonglacial_logTI': desc(nlogti),
           'logTI_overlap_coeff_bin0.25dex': ovl,
           'frac_nonglacial_within_glacial_logTI_IQR': jf(((nlogti >= giqr[0]) & (nlogti <= giqr[1])).mean()),
           'frac_glacial_within_nonglacial_logTI_IQR': jf(((glogti >= niqr[0]) & (glogti <= niqr[1])).mean()),
           'mannwhitney_logTI_p': jf(st.mannwhitneyu(glogti, nlogti, alternative='two-sided')[1]),
           'median_TI_ratio_g_over_n': jf(np.median(gt) / np.median(nt)),
           'glacial_logTI_IQR': jf(giqr[1] - giqr[0]),
           'nonglacial_logTI_IQR': jf(niqr[1] - niqr[0])}
comp = pd.crosstab(both['Methodology'], both['Process'])
comp_pct = pd.crosstab(both['Methodology'], both['Process'], normalize='columns') * 100
c['method_share_pct_within_process'] = json.loads(comp_pct.to_json(orient='index'))
c['method_share_count'] = json.loads(comp.to_json(orient='index'))
c['method_overlap'] = {m: bool(r['Glacial'] > 0 and r['Nonglacial'] > 0) for m, r in comp.iterrows()}
c['n_methods_shared'] = int(sum(c['method_overlap'].values()))
c['type_share_count'] = json.loads(pd.crosstab(both['Type'], both['Process']).to_json(orient='index'))
comp_pct.to_csv(OUT / 'method_composition_by_process.csv')

# --- Kitagawa-Oaxaca-Blinder decomposition of the marginal gap -------------
# gap = sum_s [w_g^s * (mg^s - mn^s)]  +  sum_s [(w_g^s - w_n^s) * mn^s]
#       \_______ within-stratum process effect _______/   \____ composition ____/
def kob(strata_col):
    cols = [strata_col] if isinstance(strata_col, str) else strata_col
    sub = both[both['logER'].notna() & both['logTI'].notna()]
    gp = sub[sub['Process'] == 'Glacial'].groupby(cols)['logER'].agg(['mean', 'size'])
    np_ = sub[sub['Process'] == 'Nonglacial'].groupby(cols)['logER'].agg(['mean', 'size'])
    N_g, N_n = gp['size'].sum(), np_['size'].sum()
    idx = gp.index.union(np_.index)
    gp = gp.reindex(idx); np_ = np_.reindex(idx)
    wg = (gp['size'] / N_g).fillna(0.0)
    wn = (np_['size'] / N_n).fillna(0.0)
    expl, comp_, rows = 0.0, 0.0, []
    for k in idx:
        mg, mn = gp.loc[k, 'mean'], np_.loc[k, 'mean']
        w_g, w_n = wg.loc[k], wn.loc[k]
        within = w_g * ((mg - mn) if pd.notna(mg) and pd.notna(mn) else 0.0)
        comp = (w_g - w_n) * (mn if pd.notna(mn) else 0.0)
        expl += within; comp_ += comp
        rows.append({'stratum': str(k), 'w_glacial': jf(w_g), 'w_nonglacial': jf(w_n),
                     'mean_logER_glacial': jf(mg), 'mean_logER_nonglacial': jf(mn),
                     'within_effect': jf(within), 'composition_effect': jf(comp)})
    return {'total_marginal_gap_dex': jf(sub.loc[sub['Process'] == 'Glacial', 'logER'].mean() -
                                         sub.loc[sub['Process'] == 'Nonglacial', 'logER'].mean()),
            'within_stratum_effect_dex': jf(expl), 'composition_effect_dex': jf(comp_),
            'sum_check_dex': jf(expl + comp_), 'n_strata': int(len(idx)), 'table': rows}


c['KOB_decomposition_by_method'] = kob('Methodology')
c['KOB_decomposition_by_method_logTIbin'] = kob(['Methodology', 'TIbin'])
# TI-only reweighting (within-process reweighting of the nonglacial TI distribution)
sub_all = both[both['logER'].notna() & both['logTI'].notna()]


def ti_reweight_contrast(sub):
    gs_, ns_ = sub[sub['Process'] == 'Glacial'], sub[sub['Process'] == 'Nonglacial']
    if len(gs_) < 3 or len(ns_) < 3:
        return None
    bins = np.histogram_bin_edges(np.concatenate([gs_['logTI'], ns_['logTI']]), bins=15)
    tgt, _ = np.histogram(gs_['logTI'], bins=bins); tgt = tgt / tgt.sum()
    wn, _ = np.histogram(ns_['logTI'], bins=bins)
    bi = np.clip(np.digitize(ns_['logTI'], bins) - 1, 0, len(tgt) - 1)
    w = np.ones(len(ns_))
    for b in range(len(tgt)):
        m = bi == b
        if m.sum() > 0 and wn[b] > 0:
            w[m] = tgt[b] / (wn[b] / wn.sum())
    w = w / w.sum()
    return float(gs_['logER'].mean() - np.sum(ns_['logER'].values * w))


c['reweighted_contrast_logTI_overall'] = jf(ti_reweight_contrast(sub_all))
c['reweighted_contrast_logTI_by_method'] = {
    m: jf(ti_reweight_contrast(sub_all[sub_all['Methodology'] == m]))
    for m in sorted(sub_all['Methodology'].unique())
    if ti_reweight_contrast(sub_all[sub_all['Methodology'] == m]) is not None}
c['cascade_of_estimates_dex'] = {
    '1_marginal_unadjusted': jf(MARGINAL),
    '2_TI_reweighted_within_process': c['reweighted_contrast_logTI_overall'],
    '3_method_matched_kNN5': jf((matched_all['logER'] - matched_all['nbr_logER']).mean()),
    '4_method_and_TI_matched_KOB_within_effect': c['KOB_decomposition_by_method']['within_stratum_effect_dex'],
    '5_common_support_matched': res['part3_scale_matched']['common_support'].get('matched_diff_dex'),
    '6_fully_adjusted_OLS_all_controls': None}
c['attribution_of_marginal_gap'] = {
    'marginal_gap_dex': jf(MARGINAL),
    'method_composition_component_dex': c['KOB_decomposition_by_method']['composition_effect_dex'],
    'within_method_process_component_dex': c['KOB_decomposition_by_method']['within_stratum_effect_dex'],
    'share_of_gap_from_method_composition': jf(c['KOB_decomposition_by_method']['composition_effect_dex'] / MARGINAL),
    'share_of_gap_from_within_method_process': jf(c['KOB_decomposition_by_method']['within_stratum_effect_dex'] / MARGINAL)}
res['part5_confounding'] = c

# ============================================================================
# 6. alternative-explanation controls
# ============================================================================
alts = []
for meth in sorted(both['Methodology'].dropna().unique()):
    df = all_pairs.get(meth, pd.DataFrame())
    if len(df) == 0:
        continue
    for ctrl in ['logA', 'logS', 'logP', 'absLat']:
        d = df.dropna(subset=[ctrl, 'nbr_' + ctrl]).copy()
        if len(d) < 5:
            continue
        dd = (d['logER'] - d['nbr_logER']).values
        dc = (d[ctrl] - d['nbr_' + ctrl]).values
        m = smf.ols('d ~ dc', data=pd.DataFrame({'d': dd, 'dc': dc})).fit()
        alts.append({'Methodology': meth, 'control': ctrl, 'n': int(len(d)),
                     'raw_mean_diff_dex': jf(dd.mean()),
                     'control_mean_imbalance_dex': jf(dc.mean()),
                     'adjusted_intercept_dex': jf(m.params['Intercept']),
                     'intercept_se': jf(m.bse['Intercept']), 'intercept_p': jf(m.pvalues['Intercept']),
                     'control_slope': jf(m.params['dc']), 'control_slope_p': jf(m.pvalues['dc']),
                     'raw_p': jf(st.ttest_1samp(dd, 0.0)[1])})
res['part6_controls']['matched_control_delta_regressions'] = alts
pd.DataFrame(alts).to_csv(OUT / 'matched_control_deltas.csv', index=False)

CC = ['logA', 'nbr_logA', 'logS', 'nbr_logS', 'logP', 'nbr_logP', 'absLat', 'nbr_absLat']
dfp = pd.concat([d for d in all_pairs.values() if len(d)], ignore_index=True).dropna(subset=CC)
if len(dfp) > 10:
    reg = pd.DataFrame({'d': (dfp['logER'] - dfp['nbr_logER']).values,
                        'dlogA': (dfp['logA'] - dfp['nbr_logA']).values,
                        'dlogS': (dfp['logS'] - dfp['nbr_logS']).values,
                        'dlogP': (dfp['logP'] - dfp['nbr_logP']).values,
                        'dabsLat': (dfp['absLat'] - dfp['nbr_absLat']).values})
    mj = smf.ols('d ~ dlogA + dlogS + dlogP + dabsLat', data=reg).fit()
    res['part6_controls']['pooled_joint_control_model'] = {
        'n': int(len(reg)), 'r2': jf(mj.rsquared),
        'raw_mean_diff_dex': jf(reg['d'].mean()),
        'raw_p': jf(st.ttest_1samp(reg['d'], 0.0)[1]),
        'adjusted_intercept_dex': jf(mj.params['Intercept']),
        'adjusted_intercept_se': jf(mj.bse['Intercept']),
        'adjusted_intercept_p': jf(mj.pvalues['Intercept']),
        'adjusted_intercept_ci95': [jf(mj.conf_int().loc['Intercept'][0]),
                                    jf(mj.conf_int().loc['Intercept'][1])],
        'control_slopes': {k: jf(mj.params[k]) for k in ['dlogA', 'dlogS', 'dlogP', 'dabsLat']},
        'control_slopes_p': {k: jf(mj.pvalues[k]) for k in ['dlogA', 'dlogS', 'dlogP', 'dabsLat']},
        'note': 'matched-difference model: intercept = process effect net of within-pair imbalance in controls'}

cc = both.dropna(subset=['logER', 'logTI', 'Methodology', 'logA', 'logS', 'logP', 'absLat']).copy()
cc['Process'] = pd.Categorical(cc['Process'], categories=['Glacial', 'Nonglacial'])
m_cc = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI + C(Methodology) + logA + logS + logP + absLat',
               data=cc).fit()
m_cc_cr = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI + C(Methodology) + logA + logS + logP + absLat',
                  data=cc).fit(cov_type='cluster', cov_kwds={'groups': cc['Reference']})
ci = m_cc.conf_int().loc[P]
res['part6_controls']['fully_adjusted_ols'] = {
    'n': int(m_cc.nobs), 'r2': jf(m_cc.rsquared),
    'process_coef_dex': jf(m_cc.params[P]), 'se': jf(m_cc.bse[P]), 'p': jf(m_cc.pvalues[P]),
    'ci95_lo': jf(ci[0]), 'ci95_hi': jf(ci[1]),
    'coef_dex_cluster_robust': jf(m_cc_cr.params[P]), 'se_cluster': jf(m_cc_cr.bse[P]),
    'p_cluster': jf(m_cc_cr.pvalues[P]),
    'n_glacial': int((cc['Process'] == 'Glacial').sum()),
    'n_nonglacial': int((cc['Process'] == 'Nonglacial').sum()),
    'note': 'complete cases on logA/logS/logP/absLat; coefficient = Nonglacial minus Glacial in dex. '
            'This is an out-of-support extrapolation: it forces a common control slope across the '
            'two very different samples.'}
# same but on the common-support subset
cc_cs = cc[(cc['logTI'] >= comm_lo) & (cc['logTI'] <= comm_hi)]
if cc_cs['Process'].nunique() == 2 and len(cc_cs) > 50:
    m_cs = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI + C(Methodology) + logA + logS + logP + absLat',
                   data=cc_cs).fit()
    ci2 = m_cs.conf_int().loc[P]
    res['part6_controls']['fully_adjusted_ols_common_support'] = {
        'n': int(m_cs.nobs), 'r2': jf(m_cs.rsquared), 'process_coef_dex': jf(m_cs.params[P]),
        'se': jf(m_cs.bse[P]), 'p': jf(m_cs.pvalues[P]),
        'ci95_lo': jf(ci2[0]), 'ci95_hi': jf(ci2[1]),
        'n_glacial': int((cc_cs['Process'] == 'Glacial').sum()),
        'n_nonglacial': int((cc_cs['Process'] == 'Nonglacial').sum())}
# control imbalance diagnostics (paired)
bal = []
for ctrl in ['logA', 'logS', 'logP', 'absLat']:
    d = dfp if len(dfp) else pd.DataFrame()
    dd = (dfp[ctrl] - dfp['nbr_' + ctrl]) if len(dfp) else pd.Series(dtype=float)
    bal.append({'control': ctrl, 'n': int(len(dd)), 'mean_pair_imbalance': jf(dd.mean()),
                'median_pair_imbalance': jf(dd.median()),
                'p_paired_t': jf(st.ttest_1samp(dd.dropna(), 0.0)[1]) if len(dd.dropna()) > 2 else None})
res['part6_controls']['pair_imbalance_diagnostics'] = bal

res['part6_controls']['attenuation_summary'] = {
    'marginal_dex': jf(MARGINAL),
    'method_plus_TI_matched_pooled_dex': c['cascade_of_estimates_dex']['3_method_matched_kNN5'],
    'common_support_matched_dex': res['part3_scale_matched']['common_support'].get('matched_diff_dex'),
    'fully_adjusted_ols_dex': res['part6_controls']['fully_adjusted_ols']['process_coef_dex'],
    'frac_of_marginal_removed_by_method_matching': jf(
        1 - (matched_all['logER'] - matched_all['nbr_logER']).mean() / MARGINAL),
    'frac_of_marginal_removed_by_common_support_matching': jf(
        1 - (res['part3_scale_matched']['common_support'].get('matched_diff_dex') or np.nan) / MARGINAL),
    'note': 'the fully adjusted OLS overshoots past zero (sign reversal) because it extrapolates a '
            'common control slope across samples with disjoint latitude/precipitation support.'}
c['cascade_of_estimates_dex']['6_fully_adjusted_OLS_all_controls'] = \
    res['part6_controls']['fully_adjusted_ols']['process_coef_dex']

# ============================================================================
# 7. reversals / nulls
# ============================================================================
rev = []
for meth in sorted(both['Methodology'].dropna().unique()):
    for typ in sorted(both.loc[both['Process'] == 'Glacial', 'Type'].dropna().unique()):
        sub = both[(both['Methodology'] == meth) & both['logER'].notna() & both['logTI'].notna()]
        gs_, ns_ = sub[(sub['Process'] == 'Glacial') & (sub['Type'] == typ)], sub[sub['Process'] == 'Nonglacial']
        if len(gs_) < 2 or len(ns_) < 2:
            continue
        nm = ns_[['logTI']].to_numpy(float)
        dd = []
        for _, row in gs_.iterrows():
            sel = np.argsort(np.abs(nm[:, 0] - float(row['logTI'])), kind='stable')[:min(KNN, len(nm))]
            dd.append(row['logER'] - ns_.iloc[sel]['logER'].mean())
        dd = np.array(dd)
        t_, p_ = st.ttest_1samp(dd, 0.0)
        cse = cluster_se_mean(dd, gs_['Reference'].values)
        rev.append({'stratum_type': 'methodology_x_glacial_type', 'Methodology': meth,
                    'Glacial Type': typ, 'n': int(len(dd)), 'n_reference_clusters': int(gs_['Reference'].nunique()),
                    'mean_diff_dex': jf(dd.mean()), 'median_diff_dex': jf(np.median(dd)),
                    't': jf(t_), 'p': jf(p_), 'cluster_robust_se': jf(cse),
                    'p_cluster': jf(2 * st.norm.sf(abs(dd.mean() / cse))) if cse and cse > 0 else None,
                    'ratio_10pow_mean': jf(10 ** dd.mean()),
                    'direction': 'glacial_faster' if dd.mean() > 0 else 'nonglacial_faster_or_equal',
                    'null_at_0.05': bool(p_ > 0.05)})
for comp_ in sorted(both['Compilation'].dropna().unique()):
    s = both[both['Compilation'] == comp_]
    a, b2 = s.loc[s['Process'] == 'Glacial', 'logER'].dropna(), s.loc[s['Process'] == 'Nonglacial', 'logER'].dropna()
    if len(a) >= 2 and len(b2) >= 2:
        r = welch(a, b2)
        r.update({'stratum_type': 'compilation', 'Compilation': comp_,
                  'direction': 'glacial_faster' if r['mean_diff_dex'] > 0 else 'nonglacial_faster_or_equal',
                  'null_at_0.05': bool(r['p'] > 0.05)})
        rev.append(r)
dfrev = pd.DataFrame(rev)
dfrev.to_csv(OUT / 'reversal_strata.csv', index=False)
res['part7_reversals'] = {
    'strata': json.loads(dfrev.to_json(orient='records')),
    'n_strata_tested': int(len(rev)),
    'n_reversed_nonglacial_faster': int((dfrev['mean_diff_dex'] < 0).sum()),
    'n_null_p_gt_0.05': int((dfrev['p'] > 0.05).sum()),
    'reversed_strata_significant': json.loads(
        dfrev[(dfrev['mean_diff_dex'] < 0) & (dfrev['p'] < 0.05)].to_json(orient='records')),
    'null_strata': json.loads(dfrev[dfrev['p'] > 0.05].to_json(orient='records')),
    'survive_cluster_adjustment': json.loads(
        dfrev[(dfrev.get('p_cluster').notna()) & (dfrev['mean_diff_dex'] < 0) &
              (dfrev['p_cluster'] < 0.05)].to_json(orient='records'))
    if 'p_cluster' in dfrev.columns else []}

# ============================================================================
# 3b. sensitivity: does the sign of the process contrast depend on
#     (a) the functional form of the TI adjustment, (b) within-method TI range?
#     This resolves the apparent conflict between part 3 (matched) and part 4.
# ============================================================================
sens = []
for meth in sorted(both['Methodology'].dropna().unique()):
    sub = both[(both['Methodology'] == meth) & both['logER'].notna() & both['logTI'].notna()]
    gs_, ns_ = sub[sub['Process'] == 'Glacial'], sub[sub['Process'] == 'Nonglacial']
    if len(gs_) < 3 or len(ns_) < 3:
        continue
    lin = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI', data=sub).fit()
    qua = smf.ols('logER ~ C(Process, Treatment("Glacial")) + logTI + I(logTI**2)', data=sub).fit()
    med_g, med_n = gs_['logTI'].median(), ns_['logTI'].median()
    sens.append({'Methodology': meth, 'n_glacial': int(len(gs_)), 'n_nonglacial': int(len(ns_)),
                 'marginal_diff_dex': jf(gs_['logER'].mean() - ns_['logER'].mean()),
                 'linear_TI_adjusted_dex': jf(lin.params[P]), 'linear_se': jf(lin.bse[P]),
                 'linear_p': jf(lin.pvalues[P]),
                 'quad_TI_adjusted_dex': jf(qua.params[P]), 'quad_se': jf(qua.bse[P]),
                 'quad_p': jf(qua.pvalues[P]),
                 'knn5_matched_dex': jf((all_pairs[meth]['logER'] - all_pairs[meth]['nbr_logER']).mean())
                 if meth in all_pairs and len(all_pairs[meth]) else None,
                 'glacial_median_logTI': jf(med_g), 'nonglacial_median_logTI': jf(med_n),
                 'median_logTI_gap': jf(med_g - med_n),
                 'glacial_logTI_IQR': jf(np.percentile(gs_['logTI'], 75) - np.percentile(gs_['logTI'], 25)),
                 'nonglacial_logTI_IQR': jf(np.percentile(ns_['logTI'], 75) - np.percentile(ns_['logTI'], 25)),
                 'TI_range_overlap_frac': jf(float(np.mean(
                     (ns_['logTI'] >= gs_['logTI'].min()) & (ns_['logTI'] <= gs_['logTI'].max()))))})
res['part3_scale_matched']['functional_form_sensitivity'] = sens
pd.DataFrame(sens).to_csv(OUT / 'functional_form_sensitivity.csv', index=False)

# within-methodology common-support matched contrasts
wcs = []
for meth in sorted(both['Methodology'].dropna().unique()):
    sub = both[(both['Methodology'] == meth) & both['logER'].notna() & both['logTI'].notna()]
    gs_, ns_ = sub[sub['Process'] == 'Glacial'], sub[sub['Process'] == 'Nonglacial']
    if len(gs_) < 3 or len(ns_) < 3:
        continue
    lo_ = max(np.percentile(gs_['logTI'], 5), np.percentile(ns_['logTI'], 5))
    hi_ = min(np.percentile(gs_['logTI'], 95), np.percentile(ns_['logTI'], 95))
    if hi_ <= lo_:
        wcs.append({'Methodology': meth, 'note': 'NO logTI common support at the 5-95% level',
                    'glacial_5_95': [jf(np.percentile(gs_['logTI'], 5)), jf(np.percentile(gs_['logTI'], 95))],
                    'nonglacial_5_95': [jf(np.percentile(ns_['logTI'], 5)), jf(np.percentile(ns_['logTI'], 95))]})
        continue
    gg = gs_[(gs_['logTI'] >= lo_) & (gs_['logTI'] <= hi_)]
    nn2 = ns_[(ns_['logTI'] >= lo_) & (ns_['logTI'] <= hi_)]
    wcs.append({'Methodology': meth, 'logTI_common_lo': jf(lo_), 'logTI_common_hi': jf(hi_),
                'n_glacial_in_support': int(len(gg)), 'n_nonglacial_in_support': int(len(nn2)),
                'frac_glacial_in_support': jf(len(gg) / len(gs_)),
                'marginal_diff_in_support_dex': jf(gg['logER'].mean() - nn2['logER'].mean())
                if len(gg) > 1 and len(nn2) > 1 else None,
                'welch_p_in_support': jf(st.ttest_ind(gg['logER'], nn2['logER'], equal_var=False)[1])
                if len(gg) > 1 and len(nn2) > 1 else None,
                'TI_adjusted_dex_in_support': jf(smf.ols(
                    'logER ~ C(Process, Treatment("Glacial")) + logTI',
                    data=pd.concat([gg, nn2])).fit().params[P]) if len(gg) > 2 and len(nn2) > 2 else None})
res['part3_scale_matched']['within_method_common_support'] = wcs
pd.DataFrame(wcs).to_csv(OUT / 'within_method_common_support.csv', index=False)

# ---------------------------------------------------------------------------
# 3d. fine-cell (Methodology x logTI decile) process contrast, restricted to
#     cells that actually contain enough rows of BOTH processes.
#     This is the most defensible "conditional on scale and method" estimate.
# ---------------------------------------------------------------------------
fine = []
for (m, t), s in both.dropna(subset=['logER', 'logTI']).groupby(['Methodology', 'TIbin'],
                                                                observed=True):
    gg, nn2 = s[s['Process'] == 'Glacial'], s[s['Process'] == 'Nonglacial']
    fine.append({'Methodology': m, 'TI_decile': str(t), 'n_glacial': int(len(gg)),
                 'n_nonglacial': int(len(nn2)),
                 'mean_logER_glacial': jf(gg['logER'].mean()) if len(gg) else None,
                 'mean_logER_nonglacial': jf(nn2['logER'].mean()) if len(nn2) else None,
                 'diff_dex': jf(gg['logER'].mean() - nn2['logER'].mean())
                 if len(gg) and len(nn2) else None,
                 'welch_p': jf(st.ttest_ind(gg['logER'], nn2['logER'], equal_var=False)[1])
                 if len(gg) >= 2 and len(nn2) >= 2 else None})
pd.DataFrame(fine).to_csv(OUT / 'fine_cell_contrasts.csv', index=False)
well = {}
for ming, minn in [(2, 5), (3, 20), (5, 50)]:
    sel = [f for f in fine if f['n_glacial'] >= ming and f['n_nonglacial'] >= minn
           and f['diff_dex'] is not None]
    tot = sum(f['n_glacial'] for f in sel)
    well[f'min_glacial_{ming}_min_nonglacial_{minn}'] = {
        'n_cells': len(sel), 'glacial_rows_covered': int(tot),
        'mean_glacial_weighted_diff_dex': jf(sum(f['n_glacial'] * f['diff_dex'] for f in sel) / tot)
        if tot else None,
        'cells': sel}
res['part3_scale_matched']['well_supported_fine_cells'] = well
# the two largest cells, quoted explicitly
res['part3_scale_matched']['well_supported_fine_cells']['interpretation'] = (
    'Only two Methodology x logTI-decile cells contain >=3 glacial and >=20 nonglacial rows: '
    'Volumetric / logTI in (-2.563, 0.699] (n_glacial=68, n_nonglacial=2579, diff=+1.079 dex) and '
    'Cosmogenic detrital / logTI in (0.699, 3.083] (n_glacial=4, n_nonglacial=487, diff=+0.168 dex). '
    'The widely quoted Volumetric contrast is therefore dominated by a single cell holding 2579 of the '
    '2583 nonglacial Volumetric rows. Broadening the cell set to >=2 glacial rows drops the '
    'glacial-weighted within-cell contrast to 0.610 dex over 6 cells (93 glacial rows); only the two '
    'large cells reach 1.03-1.08 dex. The scale-conditional process contrast is therefore not stable: '
    'it ranges from ~0.2 to ~2.1 dex across cells, and shrinks as more (smaller) cells are admitted.')

# ============================================================================
# figures
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
meths = sorted(both['Methodology'].dropna().unique())
pos, xt, xl = 0, [], []
for meth in meths:
    for proc in ['Glacial', 'Nonglacial']:
        d = both.loc[(both['Methodology'] == meth) & (both['Process'] == proc), 'logER'].dropna()
        if len(d) == 0:
            continue
        bp = axes[0].boxplot([d.values], positions=[pos], widths=0.7, patch_artist=True,
                             medianprops=dict(color='k'))
        bp['boxes'][0].set_facecolor('#d95f02' if proc == 'Glacial' else '#1b9e77')
        bp['boxes'][0].set_alpha(0.75)
        xt.append(pos); xl.append(f'{proc[:3]}\n{len(d)}')
        pos += 1
    pos += 1
axes[0].set_xticks(xt); axes[0].set_xticklabels(xl, fontsize=7)
axes[0].set_ylabel('log10 erosion rate (mm/yr)')
axes[0].set_title('logER by Methodology and Process')
axes[0].axhline(np.median(glog), color='#d95f02', ls='--', lw=1)
axes[0].axhline(np.median(nlog), color='#1b9e77', ls='--', lw=1)
for i, m in enumerate(meths):
    axes[0].text(pos + 0.5, axes[0].get_ylim()[0], m, fontsize=6, rotation=90, va='bottom')
rr = pd.DataFrame(scale_rows); rr = rr[rr['Methodology'] != 'ALL']
axes[1].errorbar(rr['mean_diff_dex'], np.arange(len(rr)),
                 xerr=1.96 * rr['cluster_robust_se'].fillna(rr['se']), fmt='o', color='k')
axes[1].axvline(0, color='r', ls='--')
axes[1].set_yticks(np.arange(len(rr))); axes[1].set_yticklabels(rr['Methodology'], fontsize=8)
axes[1].set_xlabel('mean paired dlogER (dex), Glacial - Nonglacial, kNN on logTI')
axes[1].set_title('Scale-matched process contrast (cluster-robust CI)')
plt.tight_layout(); plt.savefig(OUT / 'fig_w4_process_overview.png', dpi=140); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
axes[0].hist(glogti, bins=edges, alpha=0.55, label=f'Glacial (n={len(glogti)})', color='#d95f02', density=True)
axes[0].hist(nlogti, bins=edges, alpha=0.55, label=f'Nonglacial (n={len(nlogti)})', color='#1b9e77', density=True)
axes[0].axvspan(comm_lo, comm_hi, color='grey', alpha=0.15)
axes[0].set_xlabel('log10 Time interval (yr)'); axes[0].set_ylabel('density')
axes[0].legend(); axes[0].set_title(f'logTI distributions, overlap={ovl:.3f}, grey = common support')
d0 = (matched_all['logER'] - matched_all['nbr_logER']).dropna()
axes[1].hist(d0, bins=50, color='#7570b3')
axes[1].axvline(0, color='r', ls='--'); axes[1].axvline(d0.mean(), color='k')
axes[1].set_xlabel('paired dlogER (dex), kNN matched')
axes[1].set_title(f'pooled matched diffs (n={len(d0)}, mean={d0.mean():.3f} dex)')
plt.tight_layout(); plt.savefig(OUT / 'fig_w4_matched_and_TI.png', dpi=140); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
for proc, col in [('Glacial', '#d95f02'), ('Nonglacial', '#1b9e77')]:
    s = both[(both['Process'] == proc) & both['logER'].notna() & both['logTI'].notna()]
    axes[0].scatter(s['logTI'], s['logER'], s=6, alpha=0.35, color=col, label=proc)
    if len(s) > 5:
        m = smf.ols('logER ~ logTI', data=s).fit()
        xs = np.linspace(s['logTI'].min(), s['logTI'].max(), 50)
        axes[0].plot(xs, m.params['Intercept'] + m.params['logTI'] * xs, color=col, lw=2)
axes[0].set_xlabel('log10 Time interval (yr)'); axes[0].set_ylabel('log10 Erosion rate (mm/yr)')
axes[0].legend(); axes[0].set_title('ER vs TI by process (lines = OLS)')
bd = pd.DataFrame(binned)
# 95% CI of the difference from the Welch SE implied by t and mean_diff
bd['se_diff'] = (bd['mean_diff_dex'].abs() / bd['t'].abs()).replace([np.inf, -np.inf], np.nan)
axes[1].errorbar(bd['median_TI_yr'], bd['mean_diff_dex'],
                 yerr=1.96 * bd['se_diff'], fmt='o-', color='k', capsize=3)
axes[1].set_xscale('log'); axes[1].axhline(0, color='r', ls='--')
axes[1].set_xlabel('median Time interval of decile (yr)')
axes[1].set_ylabel('Glacial - Nonglacial mean dlogER (dex)')
axes[1].set_title('Process difference by TI decile (null in overlap)')
plt.tight_layout(); plt.savefig(OUT / 'fig_w4_ER_TI_by_process.png', dpi=140); plt.close()

with open(OUT / 'results.json', 'w') as f:
    json.dump(res, f, indent=2, default=str)
pd.DataFrame({k: v for k, v in marg.items() if k in ('Glacial', 'Nonglacial')}).T.to_csv(
    OUT / 'marginal_logER_stats.csv')

print(json.dumps({
    'verification_exhumation': {'within_1pct_m_conversion': v['n_rows_where_ratio_m_over_ex_within_1pct'],
                                'n': v['exhumation_rows_with_all3'],
                                'median_ratio_m_over_ex': v['median_ratio_m_over_ex'],
                                'within_0.1pct': v['n_rows_where_ratio_m_over_ex_within_0.1pct'],
                                'rows_with_raw_mm_identity': v['n_rows_ratio_in_900_1100_stored_as_mm']},
    'sensitivity_functional_form': sens,
    'within_method_common_support': wcs,
    'well_supported_fine_cells': {k: {kk: vv for kk, vv in val.items()
                                      if kk not in ('cells', 'interpretation')}
                                  for k, val in well.items() if isinstance(val, dict)},
    'part1': {k: marg['welch_t'][k] for k in ['mean_diff_dex', 'p', 'cohens_d']} | {
        'n_g': marg['Glacial']['n'], 'n_n': marg['Nonglacial']['n'],
        'median_diff_dex': marg['median_diff_dex']},
    'part3_pooled': pooled_rec,
    'part3_common_support': res['part3_scale_matched']['common_support'],
    'part4_interaction': reg_res['model_interaction_coeffs'],
    'part5': {'overlap': ovl, 'cascade': c['cascade_of_estimates_dex'],
              'attribution': c['attribution_of_marginal_gap']},
    'part6': res['part6_controls']['attenuation_summary'],
    'part7': {'n_strata': len(rev), 'n_reversed': res['part7_reversals']['n_reversed_nonglacial_faster'],
              'n_null': res['part7_reversals']['n_null_p_gt_0.05']}}, indent=2, default=str))
print('\nWROTE', OUT)
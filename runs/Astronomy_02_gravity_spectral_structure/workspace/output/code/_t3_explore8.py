import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
g = load_ranked_grid()
c = load_candidates()
m = load_modes()

# rank monotonicity
print('ranks 1..250 complete:', sorted(g['candidate_rank'].astype(int).tolist()) == list(range(1, 251)))
print('ic monotone in rank:', bool(np.all(np.diff(g.sort_values('candidate_rank')['ic'].values) >= -1e-9)))
print('ties in ic:', int(g['ic'].duplicated().sum()))

c2 = c.rename(columns={'max_velocity_cm_s': 'vmax', 'min_velocity_cm_s': 'vmin',
                       'peak_frequency_microhz': 'f', 'peak_width_microhz': 'w',
                       'information_criterion': 'ic'})
for df in (c2, g):
    df['k_vmax'] = df['vmax'].round(3)
    df['k_vmin'] = df['vmin'].round(3)
    df['k_f'] = df['f'].round(3)
    df['k_w'] = df['w'].round(3)
key = ['k_vmax', 'k_vmin', 'k_f', 'k_w']
VM = [1., 3., 10., 20., 30., 50., 100.]
Vmin = [0., 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
F = list(np.arange(860., 1311., 50.))
W = [250., 275., 300., 325.]
cel = c2[c2['k_vmax'].isin(VM) & c2['k_vmin'].isin(Vmin) & c2['k_f'].isin(F) & c2['k_w'].isin(W)].copy()
gkeys = set(map(tuple, g[key].values))
cel['in_table'] = [tuple(r) in gkeys for r in cel[key].values]
fn = cel[(~cel['in_table']) & (cel['ic'] < 137.2)]
print('FN rows (in candidate file, ic<137.2, absent from table):')
print(fn[['candidate_family', 'vmax', 'vmin', 'f', 'w', 'ic']].to_string(index=False))
print('min ic absent:', cel.loc[~cel['in_table'], 'ic'].min(),
      'max ic present:', cel.loc[cel['in_table'], 'ic'].max())
print('lattice cells in candidate file per family:')
print(cel.groupby('candidate_family')['in_table'].agg(['sum', 'count']).to_string())

# rigid-comb null
mf = np.sort(m['mode_frequency_microhz'].values)
cands = np.array([860., 910., 960., 1010., 1060., 1110., 1160., 1210., 1260., 1310.])
obs = np.median(np.abs(cands[:, None] - mf[None, :]).min(axis=1))
rng = np.random.default_rng(99)
offs = rng.uniform(0, 50, size=100000)
meds = []
for d0 in offs:
    cc = 860 + np.arange(10) * 50.0 + d0
    meds.append(np.median(np.abs(cc[:, None] - mf[None, :]).min(axis=1)))
meds = np.array(meds)
print('rigid-comb null: obs median', obs, 'null median', np.median(meds),
      '5-95', np.percentile(meds, [5, 95]), 'p(<=obs)', (meds <= obs).mean(),
      'min possible', meds.min())
# also rigid comb with f range fixed to the ranked-grid axis
cc0 = np.arange(860., 1311., 50.)
print('phase 0 median', np.median(np.abs(cc0[:, None] - mf[None, :]).min(axis=1)))

# fa restricted fits
fa = c[c['candidate_family'] == 'frequency_amplitude_scan']
rows = []
for v, s in fa.groupby('max_velocity_cm_s'):
    b = s.loc[s['information_criterion'].idxmin()]
    rows.append((v, b['peak_frequency_microhz'], b['information_criterion']))
r = pd.DataFrame(rows, columns=['vmax', 'fbest', 'ic'])
for lo in [0.1, 0.5, 1.0, 3.0]:
    s = r[r['vmax'] >= lo]
    A = np.column_stack([np.ones(len(s)), np.log10(s['vmax'])])
    cf, *_ = np.linalg.lstsq(A, np.log10(s['fbest']), rcond=None)
    pred = A @ cf
    rr = 1 - np.sum((np.log10(s['fbest']) - pred) ** 2) / np.sum((np.log10(s['fbest']) - np.log10(s['fbest']).mean()) ** 2)
    print(f'fa vmax>={lo}: n={len(s)} slope={cf[1]:.4f} A={10**cf[0]:.1f} R2={rr:.4f} maxres={np.abs(np.log10(s["fbest"])-pred).max():.4f} dex')
# boundary-pinned count
print('fa fbest == 610 (scan edge):', int((r['fbest'] == 610).sum()), 'at vmax', r.loc[r['fbest'] == 610, 'vmax'].values)
# full family ridge via weighted centroid of IC? report local curvature at optimum
for v in [10., 20., 30., 50.]:
    s = fa[fa['max_velocity_cm_s'] == v].sort_values('peak_frequency_microhz')
    k = int(np.argmin(s['information_criterion'].values))
    ic = s['information_criterion'].values
    print('vmax', v, 'fstar', s['peak_frequency_microhz'].values[k], 'ic*', ic[k],
          'penalty -50', (ic[k-1]-ic[k]) if k > 0 else None,
          'penalty +50', (ic[k+1]-ic[k]) if k < len(ic)-1 else None)

# mv family best vmin vs vmax
mv = c[c['candidate_family'] == 'minimum_velocity_scan']
rows = []
for v, s in mv.groupby('max_velocity_cm_s'):
    b = s.loc[s['information_criterion'].idxmin()]
    rows.append((v, b['min_velocity_cm_s'], b['information_criterion'], len(s)))
rm = pd.DataFrame(rows, columns=['vmax', 'vmin_best', 'ic', 'n_scanned'])
print(rm.to_string(index=False))
s = rm[rm['vmax'] >= 0.1]
A = np.column_stack([np.ones(len(s)), np.log10(s['vmax'])])
cf, *_ = np.linalg.lstsq(A, np.log10(s['vmin_best']), rcond=None)
pred = A @ cf
rr = 1 - np.sum((np.log10(s['vmin_best']) - pred) ** 2) / np.sum((np.log10(s['vmin_best']) - np.log10(s['vmin_best']).mean()) ** 2)
print(f'mv vmax>=0.1: n={len(s)} slope={cf[1]:.4f} A={10**cf[0]:.4f} R2={rr:.4f}')
print('vmin_best levels used:', np.unique(rm['vmin_best']))

# leave-one-level-out stability of the 4-parameter regression
sub = g[g['vmin'] > 0].copy()
y = sub['ic'].values


def r2f(yp, yy):
    return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)


X = np.column_stack([np.log10(sub['vmax']), np.log10(sub['vmin']), sub['f'], sub['w'], np.ones(len(sub))])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
print('full fit coef', np.round(b, 4), 'R2', r2f(X @ b, y))
for lv in sorted(sub['vmax'].unique()):
    s2 = sub[sub['vmax'] != lv]
    X2 = np.column_stack([np.log10(s2['vmax']), np.log10(s2['vmin']), s2['f'], s2['w'], np.ones(len(s2))])
    b2, *_ = np.linalg.lstsq(X2, s2['ic'].values, rcond=None)
    print(f'  drop vmax={lv:>5g} n={len(s2)} R2={r2f(X2@b2, s2["ic"].values):.4f} coef={np.round(b2,3)}')
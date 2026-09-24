import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
c = load_candidates()
g = load_ranked_grid()
mv = c[c['candidate_family'] == 'minimum_velocity_scan'].copy()
lg = np.log10(mv['information_criterion'].values)
lv = np.log10(mv['max_velocity_cm_s'].values)
lm = np.log10(mv['min_velocity_cm_s'].values)


def r2(yp, y):
    return 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)


def f(name, cols):
    X = np.column_stack(cols + [np.ones(len(mv))])
    b, *_ = np.linalg.lstsq(X, lg, rcond=None)
    print(f'{name}: R2={r2(X@b, lg):.4f} RMSE={np.sqrt(np.mean((lg-X@b)**2)):.4f} dex')
    return b


f('mv log10IC ~ log vmax only', [lv])
f('mv log10IC ~ log vmin only', [lm])
f('mv log10IC ~ log vmax + log vmin (additive)', [lv, lm])
f('mv log10IC ~ log vmax * log vmin (interaction)', [lv, lm, lv * lm])
f('mv log10IC ~ log ratio vmin/vmax only', [lm - lv])
f('mv log10IC ~ log vmax + log ratio', [lv, lm - lv])
f('mv log10IC ~ log vmax + log vmin + quad', [lv, lm, lv ** 2, lm ** 2])
f('mv log10IC ~ log vmax + log vmin + quad + interact', [lv, lm, lv ** 2, lm ** 2, lv * lm])

# curvatures
print('--- curvature in vmin at fixed vmax (mv family) ---')
for v in [10., 20., 30., 50., 100.]:
    s = mv[mv['max_velocity_cm_s'] == v].sort_values('min_velocity_cm_s')
    ic = s['information_criterion'].values
    k = int(np.argmin(ic))
    print('vmax', v, 'vmin*', s['min_velocity_cm_s'].values[k], 'ic*', ic[k],
          'ic@0.01', ic[0], 'ic@max', ic[-1], 'dIC(edge-best)', ic[0] - ic[k])

# ranked-grid: no-factorisation ledger on the truncated slice
sub = g[g['vmin'] > 0]
print('--- truncated ranked-grid slice (vmin>0, n=%d) ---' % len(sub))
for nm, cols in [('log vmax', [np.log10(sub['vmax'])]),
                 ('log vmin', [np.log10(sub['vmin'])]),
                 ('log vmax + log vmin', [np.log10(sub['vmax']), np.log10(sub['vmin'])]),
                 ('f', [sub['f']]),
                 ('w', [sub['w']]),
                 ('log vmax+log vmin+f+w', [np.log10(sub['vmax']), np.log10(sub['vmin']), sub['f'], sub['w']]),
                 ('log(vmax*vmin)+f+w', [np.log10(sub['vmax'] * sub['vmin']), sub['f'], sub['w']]),
                 ('log(vmax/vmin)+f+w', [np.log10(sub['vmax'] / sub['vmin']), sub['f'], sub['w']])]:
    X = np.column_stack(cols + [np.ones(len(sub))])
    b, *_ = np.linalg.lstsq(X, sub['ic'].values, rcond=None)
    print(f'{nm}: R2={r2(X@b, sub["ic"].values):.4f}')

# top-30 distinct cells
t30 = g.nsmallest(30, 'ic')
print('top-30 distinct (vmax,vmin) pairs:', t30.groupby(['vmax', 'vmin']).size().to_dict())
print('top-30 distinct (f,w) pairs:', t30.groupby(['f', 'w']).size().to_dict())
print('top-10 cells:')
print(g.nsmallest(10, 'ic')[['vmax', 'vmin', 'f', 'w', 'ic']].to_string(index=False))

# Threshold-census consistency ledger
c2 = c.rename(columns={'max_velocity_cm_s': 'vmax', 'min_velocity_cm_s': 'vmin',
                       'peak_frequency_microhz': 'f', 'peak_width_microhz': 'w',
                       'information_criterion': 'ic'})
VM = [1., 3., 10., 20., 30., 50., 100.]
Vmin = [0., 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
F = list(np.arange(860., 1311., 50.))
W = [250., 275., 300., 325.]
for df in (c2, g):
    for cc, kk in [('vmax', 'k_vmax'), ('vmin', 'k_vmin'), ('f', 'k_f'), ('w', 'k_w')]:
        df[kk] = df[cc].round(3)
key = ['k_vmax', 'k_vmin', 'k_f', 'k_w']
cel = c2[c2['k_vmax'].isin(VM) & c2['k_vmin'].isin(Vmin) & c2['k_f'].isin(F) & c2['k_w'].isin(W)].copy()
gkeys = set(map(tuple, g[key].values))
cel['in_table'] = [tuple(r) in gkeys for r in cel[key].values]
print('ledger: lattice cells known (candidate file):', len(cel),
      'in table:', int(cel['in_table'].sum()),
      'known & ic<137.2:', int((cel['ic'] < 137.2).sum()),
      'known & ic<137.2 but absent:', int(((cel['ic'] < 137.2) & ~cel['in_table']).sum()),
      'in table but not known from candidate file:', 250 - int(cel['in_table'].sum()))
print('table census: 250 cells = 29 whose ic is independently recoverable from the candidate file'
      ' + 221 whose ic is known only from the ranked table itself')
# how many ranked rows lie OUTSIDE the declared lattice
out = [tuple(r) not in set(map(tuple, cel[key].values)) for r in g[key].values]
print('ranked rows outside the declared 2240 lattice:', int(sum(out)))
print('outside rows ic range:', g.loc[out, 'ic'].min(), g.loc[out, 'ic'].max())
print(g.loc[out, ['vmax', 'vmin', 'f', 'w', 'ic']].head(12).to_string(index=False))
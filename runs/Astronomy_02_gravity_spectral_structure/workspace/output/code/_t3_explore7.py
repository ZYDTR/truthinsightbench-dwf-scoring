import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
g = load_ranked_grid()
c = load_candidates()
c2 = c.rename(columns={'max_velocity_cm_s': 'vmax', 'min_velocity_cm_s': 'vmin',
                       'peak_frequency_microhz': 'f', 'peak_width_microhz': 'w',
                       'information_criterion': 'ic'})

for fam, s in c2.groupby('candidate_family'):
    mn = s['ic'].min()
    print(fam, 'min', mn)
    for thr in [2, 5, 10, 20]:
        print('   within', thr, int((s['ic'] - mn < thr).sum()))
    top = s.nsmallest(8, 'ic')
    print(top[['vmax', 'vmin', 'f', 'w', 'ic']].to_string(index=False))

# ranked grid profile per axis
print('--- profiles ---')
for col in ['vmax', 'vmin', 'f', 'w']:
    p = g.groupby(col)['ic'].min().sort_values()
    print(col)
    print(p.to_string())
    print('   best level', p.index[0], 'gap to next', p.iloc[1] - p.iloc[0])

print('--- top-N margins ---')
g = g.sort_values('ic')
for n in [1, 2, 3, 5, 10, 20, 30, 50]:
    print('top', n, 'max ic', g['ic'].iloc[n - 1])
print('consecutive gaps in top 20:', np.diff(g['ic'].values[:20]))

# richer basis on non-zero vmin
sub = g[g['vmin'] > 0].copy().sort_values('ic')
y = sub['ic'].values


def r2(yp, yy):
    return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)


def fit(cols, name):
    X = np.column_stack(cols + [np.ones(len(sub))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ b
    print(f'{name}: R2={r2(p, y):.4f} RMSE={np.sqrt(np.mean((y-p)**2)):.3f} coef={np.round(b,4)}')
    return b, p


v = sub['vmax'].values
vm = sub['vmin'].values
f = sub['f'].values
w = sub['w'].values
fit([np.log10(v), np.log10(vm), f, w], 'A log v, log vmin, f, w')
fit([np.log10(v), np.log10(vm), f, w, f ** 2, w ** 2], 'B + f^2, w^2')
fit([np.log10(v), np.log10(vm), f, w, f ** 2, w ** 2, 1 / v, v, 1 / vm, vm], 'C richer (1/v, v, 1/vmin, vmin)')
fit([f, f ** 2], 'D f only quad')
fit([v, 1 / v, f, f ** 2], 'E v,1/v,f,f^2')

# factorisation test: interaction terms
Xa = np.column_stack([np.log10(v) - np.log10(v).mean(), f - f.mean(), w - w.mean(), np.ones(len(sub))])
ba, *_ = np.linalg.lstsq(Xa, y, rcond=None)
pa = Xa @ ba
print('additive(log v, f, w) R2', r2(pa, y))
Xi = np.column_stack([np.log10(v), f, w, np.log10(v) * f, np.log10(v) * w, f * w, np.ones(len(sub))])
bi, *_ = np.linalg.lstsq(Xi, y, rcond=None)
print('with interactions R2', r2(Xi @ bi, y))

# vmin/vmax ratio?
sub['ratio'] = sub['vmin'] / sub['vmax']
print('corr ic with ratio', np.corrcoef(sub['ratio'], y)[0, 1])
print('corr ic with log ratio', np.corrcoef(np.log10(sub['ratio']), y)[0, 1])
print('corr ic with vmin', np.corrcoef(vm, y)[0, 1], 'corr with log vmin', np.corrcoef(np.log10(vm), y)[0, 1])
print('corr with log vmax', np.corrcoef(np.log10(v), y)[0, 1], 'corr with f', np.corrcoef(f, y)[0, 1],
      'corr with w', np.corrcoef(w, y)[0, 1])

# whole-pass bootstrap on per-pass band peak
tr = load_traces()
f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
peaks = {}
for tag, col in MODELS.items():
    vals = []
    for pid, s in tr.groupby('pass_id'):
        p = ls_power(s['relative_time_min'].values, s[col].values, f_hz)
        mm = (f_uhz >= 860) & (f_uhz <= 1310)
        vals.append(f_uhz[mm][int(np.argmax(p[mm]))])
    peaks[tag] = np.array(vals)
rng = np.random.default_rng(7)
for tag, vals in peaks.items():
    bs = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(20000)])
    loo = np.array([np.median(np.delete(vals, i)) for i in range(len(vals))])
    print(tag, 'median peak', np.median(vals), 'bootstrap 5-95', np.percentile(bs, [5, 95]),
          'LOO range', loo.min(), loo.max())
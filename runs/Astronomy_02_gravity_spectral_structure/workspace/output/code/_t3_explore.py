import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)

g = load_ranked_grid()
print('rows', len(g))
print('--- counts per axis ---')
for c in ['vmax', 'vmin', 'f', 'w']:
    print(c, g[c].value_counts().sort_index().to_dict())
print('vmin==0 rows:', int((g['vmin'] == 0).sum()))
print('distinct io cells', g[['vmax', 'vmin', 'f', 'w']].drop_duplicates().shape[0])
print('full cartesian', 7 * 8 * 10 * 4)
print('unique ic', g['ic'].nunique())
print('head 30 ic:')
print(g.head(30).to_string())
for thr in [1, 2, 5, 10, 20]:
    print('ic<', thr, int((g['ic'] < thr).sum()))
print('--- best per axis (marginal min-IC value of each axis) ---')
for c in ['vmax', 'vmin', 'f', 'w']:
    print(c, g.loc[g['ic'].idxmin(), c])
print('--- top-1 row ---')
print(g.loc[g['ic'].idxmin()].to_dict())

m = load_modes()
print('--- modes ---')
print(m['mode_frequency_microhz'].sort_values().values)
inb = m[(m['mode_frequency_microhz'] >= 860) & (m['mode_frequency_microhz'] <= 1310)]
print('modes in cand band', len(inb), sorted(inb['mode_frequency_microhz'].values))
d = np.diff(np.sort(m['mode_frequency_microhz'].values))
print('mode spacing overall median', np.median(d), 'min', d.min(), 'max', d.max())
d2 = np.diff(np.sort(inb['mode_frequency_microhz'].values))
print('mode spacing in band', d2 if len(d2) else None)

c = load_candidates()
print('--- freq_amp_scan: best f per vmax ---')
fa = c[c['candidate_family'] == 'frequency_amplitude_scan']
for vmax, sub in fa.groupby('max_velocity_cm_s'):
    b = sub.loc[sub['information_criterion'].idxmin()]
    print(f'{vmax:10.2g} n={len(sub):3d} fbest={b["peak_frequency_microhz"]:7.1f} ic={b["information_criterion"]:9.2f}')
print('--- min_vel_scan: best vmin per vmax ---')
mv = c[c['candidate_family'] == 'minimum_velocity_scan']
for vmax, sub in mv.groupby('max_velocity_cm_s'):
    b = sub.loc[sub['information_criterion'].idxmin()]
    print(f'{vmax:10.2g} n={len(sub):3d} vminbest={b["min_velocity_cm_s"]:8.3g} ic={b["information_criterion"]:9.2f}')
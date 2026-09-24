import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

g = load_ranked_grid()
c = load_candidates()
VM = [1., 3., 10., 20., 30., 50., 100.]
Vmin = [0., 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
F = list(np.arange(860., 1311., 50.))
W = [250., 275., 300., 325.]
lattice = set()
for a in VM:
    for b in Vmin:
        for f in F:
            for w in W:
                lattice.add((round(a, 3), round(b, 3), round(f, 3), round(w, 3)))
print('lattice size', len(lattice))
keys = list(zip(g['vmax'].round(3), g['vmin'].round(3), g['f'].round(3), g['w'].round(3)))
inn = [k in lattice for k in keys]
print('ranked rows inside declared lattice:', sum(inn), 'outside:', len(inn) - sum(inn))
# per sub-lattice coverage: cartesian of vmax x vmin x f x w restricted
print('coverage of lattice: %.4f' % (sum(inn) / len(lattice)))
# marginal coverage: for each (vmax,vmin) pair, how many of 40 (f,w) cells?
cnt = g.groupby(['vmax', 'vmin']).size()
print('n distinct (vmax,vmin) pairs in table:', len(cnt), 'of', 7 * 8)
print('max cells per (vmax,vmin) pair:', cnt.max(), '(of 40 possible)')
# coverage per vmax level
for a in VM:
    s = g[g['vmax'] == a]
    print(f'vmax={a:6g}: {len(s):3d} cells = {len(s)/ (8*10*4):.3f} of its 320-cell sub-lattice')
print('median ic by vmax:', g.groupby('vmax')['ic'].median().to_dict())
print('min ic by vmax:', g.groupby('vmax')['ic'].min().to_dict())

# single-pass resolution simulation with real timestamps
tr = load_traces()
p = tr[tr['pass_id'] == 'pass_01']
t = p['relative_time_min'].values
f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
rng = np.random.default_rng(2024)
T = (t.max() - t.min()) * 60.0
print('span sec', T, 'invT uHz', 1e6 / T)
for df in [50., 100., 211., 250., 275., 300., 325., 450.]:
    nsep = []
    for k in range(20):
        ph = rng.uniform(0, 2 * np.pi, 2)
        ysim = np.cos(2 * np.pi * 1000e-6 * t * 60 + ph[0]) + np.cos(2 * np.pi * (1000 + df) * 1e-6 * t * 60 + ph[1])
        pw = ls_power(t, ysim, f_hz)
        m = (f_uhz >= 1000 + df / 2 - 3 * df) & (f_uhz <= 1000 + df / 2 + 3 * df)
        seg = pw[m]
        # count local maxima above 0.5*max
        loc = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.5 * seg.max()]
        nsep.append(len(loc))
    print(f'df={df:6.1f} uHz ({df/(1e6/T):.3f} x 1/T): median n_local_max={int(np.median(nsep))} '
          f'frac_with_2_peaks={np.mean(np.array(nsep) >= 2):.2f}')
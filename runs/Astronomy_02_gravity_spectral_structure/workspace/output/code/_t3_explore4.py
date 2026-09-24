import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
g = load_ranked_grid()
c = load_candidates()
tr = load_traces()

c2 = c.rename(columns={'max_velocity_cm_s': 'vmax', 'min_velocity_cm_s': 'vmin',
                       'peak_frequency_microhz': 'f', 'peak_width_microhz': 'w',
                       'information_criterion': 'ic_full'})
c2['vmax_k'] = c2['vmax'].round(3)
c2['vmin_k'] = c2['vmin'].round(3)
c2['f_k'] = c2['f'].round(3)
c2['w_k'] = c2['w'].round(3)
g2 = g.copy()
g2['vmax_k'] = g2['vmax'].astype(float).round(3)
g2['vmin_k'] = g2['vmin'].round(3)
g2['f_k'] = g2['f'].round(3)
g2['w_k'] = g2['w'].round(3)
key = ['vmax_k', 'vmin_k', 'f_k', 'w_k']
m = g2.merge(c2[key + ['ic_full', 'candidate_family']], on=key, how='left')
print('ranked rows total', len(g2), 'matched', int(m['ic_full'].notna().sum()))
mm = m[m['ic_full'].notna()].copy()
mm['offset'] = mm['ic_full'] - mm['ic']
print(mm[['candidate_rank', 'vmax', 'vmin', 'f', 'w', 'ic', 'ic_full', 'offset', 'candidate_family']].to_string())
print('matched family counts', mm['candidate_family'].value_counts().to_dict())
print('matched w values', mm['w'].value_counts().to_dict())

# Are the ranked-grid cells that belong to the two candidate families fully present?
in_fa = c2[(c2['candidate_family'] == 'frequency_amplitude_scan')]
print('fa rows in candidate file', len(in_fa))
print('how many fa rows have vmax in ranked axis set and f in ranked axis set and w in 250..325:',
      int(in_fa['vmax'].isin([1, 3, 10, 20, 30, 50, 100]).sum()))

# ranked grid: coverage of cartesian grid
print('grid coverage: 250 of 2240 =', 250 / 2240)
print('per-axis row counts (fraction of 250):')
for col in ['vmax', 'vmin', 'f', 'w']:
    vc = g[col].value_counts().sort_index()
    print(col, {float(k): int(v) for k, v in vc.items()})

# expected count under a uniform random 250-subset
print('expected per level (uniform):')
print('vmax 250/7 =', 250 / 7, ' vmin 250/8 =', 250 / 8, ' f 250/10 =', 250 / 10, ' w 250/4 =', 250 / 4)

# trace time structure per pass
print('--- per-pass time structure ---')
for pid, sub in tr.groupby('pass_id'):
    t = np.sort(sub['relative_time_min'].values)
    d = np.diff(t) * 60.0
    print(pid, 'n', len(t), 't0', round(t[0], 3), 't1', round(t[-1], 3),
          'span_min', round(t[-1] - t[0], 3), 'median_dt_s', round(float(np.median(d)), 2))
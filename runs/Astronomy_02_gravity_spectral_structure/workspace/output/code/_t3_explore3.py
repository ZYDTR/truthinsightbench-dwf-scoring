import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 240)
g = load_ranked_grid()
c = load_candidates()

# cross-file consistency: match on (vmax,vmin,f,w=width) where w==300
sub = g[g['w'] == 300].copy()
c2 = c.rename(columns={'peak_width_microhz': 'w'})
sub['vmin_r'] = sub['vmin'].round(6)
c2['vmin_r'] = c2['min_velocity_cm_s'].round(6)
c2 = c2.rename(columns={'max_velocity_cm_s': 'vmax', 'min_velocity_cm_s': 'vmin',
                        'peak_frequency_microhz': 'f', 'peak_width_microhz': 'w'})
c2['vmin_r'] = c2['vmin'].round(6)
key = ['vmax', 'vmin_r', 'f', 'w']
mrg = sub.merge(c2[key + ['information_criterion', 'candidate_family']].rename(
    columns={'information_criterion': 'ic_full'}), on=key, how='left')
print('ranked-grid rows with w=300:', len(sub), 'matched in candidates:',
      int(mrg['ic_full'].notna().sum()))
mrg['offset'] = mrg['ic_full'] - mrg['ic']
print(mrg['offset'].describe())
print(mrg[['vmax', 'vmin_r', 'f', 'w', 'ic', 'ic_full', 'offset', 'candidate_family']].head(20).to_string())
print('offset unique (rounded):', np.unique(mrg['offset'].round(3))[:20])

# full IC distributions of the two families vs the 137.2 cutoff
for fam, s in c.groupby('candidate_family'):
    print(fam, 'n', len(s), 'min IC', s['information_criterion'].min(),
          'n<=137.2+min:', int((s['information_criterion'] <= s['information_criterion'].min() + 137.2).sum()))

# fa curves at a few vmax
fa = c[c['candidate_family'] == 'frequency_amplitude_scan']
for vmax in [0.1, 1.0, 10.0, 20.0, 30.0, 50.0, 100.0, 500.0]:
    s = fa[fa['max_velocity_cm_s'] == vmax].sort_values('peak_frequency_microhz')
    print('vmax=', vmax)
    print(s[['peak_frequency_microhz', 'information_criterion']].to_string(index=False))

# log-log fit of fbest vs vmax
rows = []
for vmax, s in fa.groupby('max_velocity_cm_s'):
    b = s.loc[s['information_criterion'].idxmin()]
    rows.append((vmax, b['peak_frequency_microhz'], b['information_criterion']))
r = pd.DataFrame(rows, columns=['vmax', 'fbest', 'ic'])
A = np.column_stack([np.ones(len(r)), np.log10(r['vmax'])])
coef, *_ = np.linalg.lstsq(A, np.log10(r['fbest']), rcond=None)
print('log-log fit fbest = a * vmax^b : b =', coef[1], 'a =', 10 ** coef[0])
pred = coef[0] + coef[1] * np.log10(r['vmax'])
print('R2 =', 1 - np.sum((np.log10(r['fbest']) - pred) ** 2) / np.sum((np.log10(r['fbest']) - np.log10(r['fbest']).mean()) ** 2))
print('resid max (dex)', np.abs(np.log10(r['fbest']) - pred).max())

# use all fa rows (not only argmin): fit peak-frequency ridge over the whole family
A2 = np.column_stack([np.ones(len(fa)), np.log10(fa['max_velocity_cm_s'])])
y2 = np.log10(fa['peak_frequency_microhz'])
cf2, *_ = np.linalg.lstsq(A2, np.log10(fa['information_criterion']), rcond=None)
print('regression of log10 IC on log10 vmax (all fa rows):', cf2)

# ratio fbest / vmax**b
print('fbest / vmax^b:', (r['fbest'] / r['vmax'] ** coef[1]).values)

# combined statistic: is IC min constant along f ~ c*vmax^b?
r['f_over_vmxb'] = r['fbest'] / r['vmax'] ** coef[1]
print('combined quantity spread:', r['f_over_vmxb'].min(), r['f_over_vmxb'].max())

# min_velocity_scan
mv = c[c['candidate_family'] == 'minimum_velocity_scan']
print('mv rows per vmax (nested):')
print(mv.groupby('max_velocity_cm_s')['min_velocity_cm_s'].apply(lambda s: sorted(s)).to_string())
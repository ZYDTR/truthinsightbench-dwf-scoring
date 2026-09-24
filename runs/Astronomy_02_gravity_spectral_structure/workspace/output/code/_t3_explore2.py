import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)
g = load_ranked_grid()
c = load_candidates()
m = load_modes()
tr = load_traces()

# vmin<vmax structure in ranked grid
print('ranked grid: rows with vmin>=vmax:', int((g['vmin'] >= g['vmax']).sum()))
print('ranked grid min ratio vmin/vmax:', (g['vmin'] / g['vmax']).min())

# freq_amp_scan structure
fa = c[c['candidate_family'] == 'frequency_amplitude_scan']
print('fa n vmax:', fa.groupby('max_velocity_cm_s').size().to_dict())
print('fa: vmin unique', fa['min_velocity_cm_s'].unique(), 'w unique', fa['peak_width_microhz'].unique())
print('fa: f per vmax counts:', fa.groupby('max_velocity_cm_s')['peak_frequency_microhz'].nunique().to_dict())

mv = c[c['candidate_family'] == 'minimum_velocity_scan']
print('mv pairs:')
print(mv[['max_velocity_cm_s', 'min_velocity_cm_s']].sort_values(['max_velocity_cm_s', 'min_velocity_cm_s']).to_string())
print('mv constraint vmin<vmax violations:', int((mv['min_velocity_cm_s'] >= mv['max_velocity_cm_s']).sum()))
print('mv vmin/vmax max:', (mv['min_velocity_cm_s'] / mv['max_velocity_cm_s']).max())

# best f per vmax in fa: combined value vmax/f ratios
rows = []
for vmax, sub in fa.groupby('max_velocity_cm_s'):
    b = sub.loc[sub['information_criterion'].idxmin()]
    rows.append((vmax, b['peak_frequency_microhz'], b['information_criterion']))
r = pd.DataFrame(rows, columns=['vmax', 'fbest', 'ic'])
print('fa collapse table')
print(r.to_string())
print('fbest/vmax ratio:', (r['fbest'] / r['vmax']).values)
print('fbest <-> sqrt?', (r['fbest'] ** 0.5).values)

# mode grid: freq vs surface velocity
mm = m.sort_values('mode_frequency_microhz')
print(mm[['angular_degree', 'radial_order', 'mode_frequency_microhz', 'surface_velocity_cm_s']].to_string())
print('mode velocity range', mm['surface_velocity_cm_s'].min(), mm['surface_velocity_cm_s'].max())

# per-pass spans
gpass = tr.groupby('pass_id')['relative_time_min']
span = (gpass.max() - gpass.min())
print('per-pass span min', span.min(), 'max', span.max(), 'median', span.median())
print('n passes', len(span))
n_samp = tr.groupby('pass_id').size()
print('samples per pass:', n_samp.unique())
print('cadence median', np.median(np.diff(np.sort(tr['relative_time_min'].values))) * 60, 's')
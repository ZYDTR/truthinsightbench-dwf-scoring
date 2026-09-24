import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)

g = load_ranked_grid()
c = load_candidates()
tr = load_traces()

# ---- truncation hypothesis test ----
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

# cells of the declared lattice present in candidate file
cel = c2[c2['k_vmax'].isin(VM) & c2['k_vmin'].isin(Vmin) & c2['k_f'].isin(F) & c2['k_w'].isin(W)].copy()
print('candidate-file rows inside declared lattice:', len(cel))
print(cel['candidate_family'].value_counts().to_dict())
gkeys = set(map(tuple, g[key].values))
cel['in_table'] = [tuple(r) in gkeys for r in cel[key].values]
print('of those, present in ranked table:', int(cel['in_table'].sum()))
print(cel.groupby('in_table')['ic'].agg(['count', 'min', 'max']).to_string())
# recall / precision of the rule "in table <=> ic < 137.2"
thr = g['ic'].max()
cel['pred'] = cel['ic'] < thr
tp = int((cel['pred'] & cel['in_table']).sum())
fp = int((cel['pred'] & ~cel['in_table']).sum())
fn = int((~cel['pred'] & cel['in_table']).sum())
print(f'rule ic<{thr}: TP={tp} FP={fp} FN={fn}')
print('max ic among candidate-file lattice cells NOT in table:', cel.loc[~cel['in_table'], 'ic'].min())
print('cells with ic<137.2 in table but NOT from candidate file (uncovered):', 250 - tp)

# ---- ridge-function factorisation ----
sub = g[g['vmin'] > 0].copy()
y = sub['ic'].values


def r2(yp, y):
    return 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)


X4 = np.column_stack([np.log10(sub['vmax']), np.log10(sub['vmin']), sub['f'], sub['w'], np.ones(len(sub))])
b4, *_ = np.linalg.lstsq(X4, y, rcond=None)
p4 = X4 @ b4
print('4D linear R2 =', r2(p4, y), 'RMSE =', np.sqrt(np.mean((y - p4) ** 2)))

# ridge function: u = f - a*vmax^bb
fa = c2[c2['candidate_family'] == 'frequency_amplitude_scan']
rows = []
for v, s in fa.groupby('vmax'):
    b = s.loc[s['ic'].idxmin()]
    rows.append((v, b['f'], b['ic']))
r = pd.DataFrame(rows, columns=['vmax', 'fbest', 'ic'])
A = np.column_stack([np.ones(len(r)), np.log10(r['vmax'])])
cf, *_ = np.linalg.lstsq(A, np.log10(r['fbest']), rcond=None)
a = 10 ** cf[0]
bb = cf[1]
print('ridge a, b =', a, bb)
u = sub['f'].values - a * sub['vmax'].values ** bb
Xu = np.column_stack([u, u ** 2, np.ones(len(sub))])
bu, *_ = np.linalg.lstsq(Xu, y, rcond=None)
pu = Xu @ bu
print('ridge-function (u,u^2) R2 =', r2(pu, y), 'RMSE =', np.sqrt(np.mean((y - pu) ** 2)))
# compare with plain f
Xf = np.column_stack([sub['f'], sub['f'] ** 2, np.ones(len(sub))])
bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
print('f-only quadratic R2 =', r2(Xf @ bf, y))

# grid-search over (a,b) for the best single-combination model u = f - a*vmax^b, quadratic
best = None
for bb2 in np.arange(-0.2, 1.01, 0.05):
    for a2 in np.arange(200, 1600, 25):
        u2 = sub['f'].values - a2 * sub['vmax'].values ** bb2
        X = np.column_stack([u2, u2 ** 2, np.ones(len(sub))])
        bb3, *_ = np.linalg.lstsq(X, y, rcond=None)
        rr = r2(X @ bb3, y)
        if best is None or rr > best[0]:
            best = (rr, a2, bb2)
print('best single-combination quadratic model: R2=%.4f a=%.0f b=%.2f' % best)

# ---- empirical single-pass resolution: LS peak FWHM ----
f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
res = []
for pid, s in tr.groupby('pass_id'):
    t = s['relative_time_min'].values
    yv = s['residual_model_A_mm_s'].values
    p = ls_power(t, yv, f_hz)
    m = (f_uhz >= 860) & (f_uhz <= 1310)
    k = np.argmax(p[m]) + np.where(m)[0][0]
    # interpolate half-power crossing around the peak
    half = p[k] / 2.0
    lo = k
    while lo > 0 and p[lo] > half:
        lo -= 1
    hi = k
    while hi < len(p) - 1 and p[hi] > half:
        hi += 1
    fwhm = f_uhz[hi] - f_uhz[lo]
    T = (t.max() - t.min()) * 60.0
    res.append((pid, T, 1e6 / T, p[k], fwhm))
res = pd.DataFrame(res, columns=['pass', 'T_s', 'invT_uhz', 'peakpow', 'fwhm_uhz'])
print(res.to_string(index=False))
print('median invT', res['invT_uhz'].median(), 'median fwhm', res['fwhm_uhz'].median(),
      'IQR fwhm', res['fwhm_uhz'].quantile(0.25), res['fwhm_uhz'].quantile(0.75))
print('model B / C too:')
for tag, col in MODELS.items():
    rr = []
    for pid, s in tr.groupby('pass_id'):
        t = s['relative_time_min'].values
        p = ls_power(t, s[col].values, f_hz)
        m = (f_uhz >= 860) & (f_uhz <= 1310)
        k = np.argmax(p[m]) + np.where(m)[0][0]
        half = p[k] / 2.0
        lo = k
        while lo > 0 and p[lo] > half:
            lo -= 1
        hi = k
        while hi < len(p) - 1 and p[hi] > half:
            hi += 1
        rr.append(f_uhz[hi] - f_uhz[lo])
    print(tag, 'median fwhm', np.median(rr), 'min', np.min(rr), 'max', np.max(rr))
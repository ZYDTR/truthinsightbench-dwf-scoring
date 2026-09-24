import sys
sys.path.insert(0, '/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
from common import *
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
c = load_candidates()
tr = load_traces()
g = load_ranked_grid()
m = load_modes()

fa = c[c['candidate_family'] == 'frequency_amplitude_scan']
print('fa shape', fa.shape, 'vmax x f complete?', fa.groupby('max_velocity_cm_s').size().unique())
piv = fa.pivot_table(index='max_velocity_cm_s', columns='peak_frequency_microhz', values='information_criterion')
print(piv.round(1).to_string())
Y = np.log10(piv.values)
V = np.log10(piv.index.values)[:, None]
Fq = piv.columns.values[None, :]
Vf = np.broadcast_to(V, Y.shape)
Ff = np.broadcast_to(Fq, Y.shape)
# full model with interaction
Xf = np.column_stack([Vf.ravel(), Ff.ravel(), (Vf * Ff).ravel(), Vf.ravel() ** 2, Ff.ravel() ** 2, np.ones(Y.size)])
bf, *_ = np.linalg.lstsq(Xf, Y.ravel(), rcond=None)
r2f = 1 - np.sum((Y.ravel() - Xf @ bf) ** 2) / np.sum((Y.ravel() - Y.mean()) ** 2)
# additive
Xa = np.column_stack([Vf.ravel() - Vf.mean(), Ff.ravel() - Ff.mean(), np.ones(Y.size)])
ba, *_ = np.linalg.lstsq(Xa, Y.ravel(), rcond=None)
r2a = 1 - np.sum((Y.ravel() - Xa @ ba) ** 2) / np.sum((Y.ravel() - Y.mean()) ** 2)
res_a = Y.ravel() - Xa @ ba
print('fa factorial: full R2', r2f, 'additive R2', r2a, 'additive RMSE dex', np.sqrt(np.mean(res_a ** 2)))
# ridge collapse: is Y a function of u = f - a*vmax^b alone?
for a, b in [(709.3, 0.1214), (600, 0.35), (1000, 0.0)]:
    u = Ff - a * piv.index.values[:, None] ** b
    X1 = np.column_stack([u.ravel(), u.ravel() ** 2, np.ones(Y.size)])
    b1, *_ = np.linalg.lstsq(X1, Y.ravel(), rcond=None)
    r2_1 = 1 - np.sum((Y.ravel() - X1 @ b1) ** 2) / np.sum((Y.ravel() - Y.mean()) ** 2)
    print(f'ridge a={a} b={b}: R2={r2_1:.4f} RMSE={np.sqrt(np.mean((Y.ravel()-X1@b1)**2)):.4f} dex')

# ---- resolution: per-pass LS peak scatter in band ----
f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
pk = {}
for tag, col in MODELS.items():
    vals = []
    for pid, s in tr.groupby('pass_id'):
        p = ls_power(s['relative_time_min'].values, s[col].values, f_hz)
        mm = (f_uhz >= 860) & (f_uhz <= 1310)
        vals.append(f_uhz[mm][int(np.argmax(p[mm]))])
    pk[tag] = np.array(vals)
    print(tag, 'per-pass band peak: median', np.median(vals), 'IQR', np.percentile(vals, 25),
          np.percentile(vals, 75), 'min', min(vals), 'max', max(vals), 'sd', np.std(vals, ddof=1))
np.save('/tmp/t3_peaks.npy', np.array([pk[k] for k in 'ABC']))

# ---- mode alignment ----
mf = np.sort(m['mode_frequency_microhz'].values)
cands = np.array([860., 910., 960., 1010., 1060., 1110., 1160., 1210., 1260., 1310.])
d = np.abs(cands[:, None] - mf[None, :]).min(axis=1)
print('candidate nearest-mode distances', dict(zip(cands.astype(int), np.round(d, 3))))
print('observed median', np.median(d), 'mean', d.mean())
pc = np.array([x / f * 100 for x, f in zip(d, cands)])
print('pct offsets', np.round(pc, 4), 'median pct', np.median(pc))
rng = np.random.default_rng(12345)
lo, hi = mf.min(), mf.max()
n = 200000


def null_dists(lo, hi, ncand, n, chunk=20000):
    out = []
    for _ in range(int(np.ceil(n / chunk))):
        draws = rng.uniform(lo, hi, size=(chunk, ncand))
        dd = np.abs(draws[:, :, None] - mf[None, None, :]).min(axis=2)
        out.append(dd)
    return np.vstack(out)


dn = null_dists(lo, hi, len(cands), n)
medn = np.median(dn, axis=1)
print('null over full mode range: median of medians', np.median(medn),
      '5-95%', np.percentile(medn, [5, 95]), 'p(<=obs)=', (medn <= np.median(d)).mean())
dn2 = null_dists(860, 1310, len(cands), n)
medn2 = np.median(dn2, axis=1)
print('null over candidate band: median of medians', np.median(medn2),
      '5-95%', np.percentile(medn2, [5, 95]), 'p(<=obs)=', (medn2 <= np.median(d)).mean())
for i, f0 in enumerate(cands):
    print('  cand', f0, 'd', round(d[i], 3), 'p', (dn[:, i] <= d[i]).mean(),
          'p2', (dn2[:, i] <= d[i]).mean())
# mode spacing
dm = np.diff(mf)
inb = (mf >= 860) & (mf <= 1310)
mfb = mf[inb]
print('modes in band', len(mfb), 'spacing median', np.median(np.diff(mfb)), 'mean', np.diff(mfb).mean())
print('overall spacing median', np.median(dm), 'mean', dm.mean(), 'n modes', len(mf))
print('band span', mfb[-1] - mfb[0], 'mean rate', (len(mfb) - 1) / (mfb[-1] - mfb[0]))
# S42-ish: expected distance for uniform given same count over band
lam = (len(mfb) - 1) / (mfb[-1] - mfb[0])
print('expected mean nearest dist for uniform in band = 1/(2*lambda) =', 1 / (2 * lam))
#!/usr/bin/env python3
"""
Independent recomputation + adversarial attack on Draft Finding 1
(claim: of 9 columns only c04 and c07 carry independent information; a-e).

Written from scratch (no reuse of output/analysis or other verification scripts).
All numbers printed here are the real numbers observed on this run.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = '/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project'
DATA = PROJ + '/data/observations.tsv'
OUT = PROJ + '/output/verification/figures'

LOG = []
def log(msg=''):
    print(msg)
    LOG.append(str(msg))

res = {}

# ---------------------------------------------------------------- load
df = pd.read_csv(DATA, sep='\t', dtype={'c07_signal': str})
n = len(df)
log('rows=%d cols=%d' % (n, df.shape[1]))
log('columns: ' + ', '.join(df.columns))

names = list(df.columns)
c01 = df[names[0]].to_numpy(dtype=float)
c02 = df[names[1]].to_numpy(dtype=float)
c03 = df[names[2]].to_numpy(dtype=float)
c04 = df[names[3]].to_numpy(dtype=float)
c05 = df[names[4]].to_numpy(dtype=float)
c06 = df[names[5]].to_numpy(dtype=float)
c07 = df[names[6]].to_numpy(dtype=object)
c08 = df[names[7]].to_numpy(dtype=float)
c09 = df[names[8]].to_numpy(dtype=float)

lab = ~pd.isna(c08)
log('dated rows (c08 non-null) = %d ; c07 non-empty labels = %d'
    % (lab.sum(), (~pd.isna(c07)).sum()))
log('c01 min=%.4f max=%.4f span=%.4f' % (c01.min(), c01.max(), c01.max() - c01.min()))
res['n_rows'] = int(n)
res['c01_span'] = float(c01.max() - c01.min())

def stats(x, y):
    """max abs / rms / mean abs of x-y"""
    d = np.asarray(x, float) - np.asarray(y, float)
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).mean())

# ---------------------------------------------------------------- (a) c02 vs c01
log('\n===== (a) c02 vs c01 =====')
d21 = c02 - c01
log('max |c02-c01| = %.6f  (claim: <=0.02)' % np.abs(d21).max())
log('min c02-c01 = %.6f' % d21.min())
vals, cnts = np.unique(np.round(d21, 6), return_counts=True)
log('distinct (c02-c01) values: ' + ', '.join('%g(%d)' % (v, k) for v, k in zip(vals, cnts)))
res['a_max_abs'] = float(np.abs(d21).max())
res['a_n_over_0p02'] = int((np.abs(d21) > 0.02 + 1e-9).sum())
res['a_offset_counts'] = {str(v): int(k) for v, k in zip(vals, cnts)}
res['a_exact_plus_0p01_frac'] = float(np.mean(np.abs(d21 - 0.01) < 1e-9))

# ---------------------------------------------------------------- (b) c09 vs c08
log('\n===== (b) c09 vs (1950-c08)/1000 =====')
pred_b = (1950.0 - c08[lab]) / 1000.0
obs_b = c09[lab]
diff_b = obs_b - pred_b
log('dated n=%d  max |c09-(1950-c08)/1000| = %.3e  (exact float eq: %d/%d)'
    % (lab.sum(), np.abs(diff_b).max(), int((diff_b == 0).sum()), lab.sum()))
log('also exact vs round(...,3): %d/%d' %
    (int((obs_b == np.round(pred_b, 3)).sum()), lab.sum()))
log('c08 values: ' + ', '.join('%g' % v for v in sorted(np.unique(c08[lab]))))
log('c09 values: ' + ', '.join('%g' % v for v in sorted(np.unique(c09[lab]))))
res['b_max_abs'] = float(np.abs(diff_b).max())
res['b_exact_float'] = int((diff_b == 0).sum())
res['b_n_dated'] = int(lab.sum())

# ---------------------------------------------------------------- (c) c01 vs cumsum(c03)
log('\n===== (c) c01 vs cumsum(c03)/100 =====')
cs = np.cumsum(c03) / 100.0
r = c01 - cs                       # residual of c01 against cumulative thickness
res['c_r_min'] = float(r.min())
res['c_r_max'] = float(r.max())
log('raw residual r=c01-cumsum(c03)/100 : min=%.4f max=%.4f' % (r.min(), r.max()))

# single-breakpoint scan: two segment means, minimize max|resid| and rms
S = np.concatenate([[0.0], np.cumsum(r)])
S2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
mx = np.concatenate([[-np.inf], np.maximum.accumulate(r)])
mn = np.concatenate([[np.inf], np.minimum.accumulate(r)])
suf_mx = np.concatenate([np.maximum.accumulate(r[::-1])[::-1], [-np.inf]])
suf_mn = np.concatenate([np.minimum.accumulate(r[::-1])[::-1], [np.inf]])
ks = np.arange(1, n)
m1 = S[ks] / ks
m2 = (S[n] - S[ks]) / (n - ks)
maxdev = np.maximum.reduce([
    np.abs(mx[ks] - m1), np.abs(mn[ks] - m1),
    np.abs(suf_mx[ks] - m2), np.abs(suf_mn[ks] - m2)])
ss = (S2[ks] - 2 * m1 * S[ks] + ks * m1 ** 2) + \
     ((S2[n] - S2[ks]) - 2 * m2 * (S[n] - S[ks]) + (n - ks) * m2 ** 2)
rms = np.sqrt(ss / n)
k_best_max = int(ks[np.argmin(maxdev)])
k_best_rms = int(ks[np.argmin(rms)])
log('best single breakpoint by max|resid|: after row idx %d (0-based; row %d) -> max|res|=%.6f'
    % (k_best_max - 1, k_best_max, maxdev.min()))
log('best single breakpoint by rms       : after row idx %d (0-based; row %d) -> rms=%.6f'
    % (k_best_rms - 1, k_best_rms, rms.min()))
order = np.argsort(maxdev)[:8]
log('top-8 breakpoints (max|resid|): ' +
    ', '.join('k=%d:%.5f' % (ks[i], maxdev[i]) for i in order))
res['c_best_break_k'] = k_best_max
res['c_best_break_maxres'] = float(maxdev.min())
res['c_best_break_rms'] = float(rms[k_best_rms])

k = k_best_max
seg = np.zeros(n, int); seg[k:] = 1
resid_step = np.empty(n); step_est = np.empty(2)
for s in (0, 1):
    m = seg == s
    step_est[s] = r[m].mean()
    resid_step[m] = r[m] - step_est[s]
jump = step_est[1] - step_est[0]
log('segment means: before=%.5f  after=%.5f  implied JUMP=%.5f m' % (step_est[0], step_est[1], jump))
log('residual after step model: max|res|=%.5f rms=%.5f' % (np.abs(resid_step).max(), np.sqrt(np.mean(resid_step**2))))
log('band widths (max-min within segment): before=%.5f after=%.5f' %
    (r[:k].max() - r[:k].min(), r[k:].max() - r[k:].min()))
log('c01 around break: c01[%d]=%.2f c01[%d]=%.2f  raw diff=%.4f' % (k - 1, c01[k - 1], k, c01[k], c01[k] - c01[k - 1]))
res['c_jump'] = float(jump)
res['c_seg_means'] = [float(step_est[0]), float(step_est[1])]
res['c_resid_max_after_step'] = float(np.abs(resid_step).max())
res['c_resid_rms_after_step'] = float(np.sqrt(np.mean(resid_step ** 2)))
res['c_band_before'] = float(r[:k].max() - r[:k].min())
res['c_band_after'] = float(r[k:].max() - r[k:].min())

# OLS fit c01 = a + b*cumsum + step*I  (this is what yields a single fitted step/intercept)
X = np.column_stack([np.ones(n), cs, seg.astype(float)])
beta, *_ = np.linalg.lstsq(X, c01, rcond=None)
resid_ols = c01 - X @ beta
log('OLS c01 ~ 1 + cumsum + step*I: intercept=%.5f slope_cumsum=%.8f step=%.5f' %
    (beta[0], beta[1], beta[2]))
log('OLS residual: max|res|=%.5f rms=%.5f' % (np.abs(resid_ols).max(), np.sqrt(np.mean(resid_ols**2))))
res['c_ols'] = dict(intercept=float(beta[0]), slope=float(beta[1]), step=float(beta[2]),
                    max_abs=float(np.abs(resid_ols).max()),
                    rms=float(np.sqrt(np.mean(resid_ols ** 2))))

# --- adversarial alternatives at the same breakpoint
log('\n--- (c) alternative explanations at k=%d ---' % k)
i = np.arange(n, dtype=float)
def fit_report(name, cols):
    Xx = np.column_stack(cols)
    bb, *_ = np.linalg.lstsq(Xx, r, rcond=None)
    rr = r - Xx @ bb
    log('%-42s max|res|=%.5f rms=%.5f' % (name, np.abs(rr).max(), np.sqrt(np.mean(rr ** 2))))
    return float(np.abs(rr).max()), float(np.sqrt(np.mean(rr ** 2)))
fit_report('global constant (1 param)', [np.ones(n)])
fit_report('single step (2 params: segment means)', [np.ones(n), seg.astype(float)])
fit_report('global linear drift (2 params)', [np.ones(n), i])
fit_report('global quadratic (3 params)', [np.ones(n), i, i ** 2])
fit_report('CONTINUOUS piecewise-linear, knot k (3p)', [np.ones(n), i, np.maximum(i - k, 0.0)])
fit_report('DISCONTINUOUS piecewise-linear, knot k (4p)', [np.ones(n), i, seg.astype(float), np.maximum(i - k, 0.0)])
fit_report('single step + post-step linear drift (3p)', [np.ones(n), seg.astype(float), seg * (i - k)])
# smooth low-order polynomial alone (no step) -- can it absorb the jump?
fit_report('quartic poly in index, no step (5p)', [np.ones(n), i, i**2, i**3, i**4])
# smooth in depth
fit_report('cubic in cumsum, no step (4p)', [np.ones(n), cs, cs**2, cs**3])

# does a SECOND step help beyond the first?
r1 = resid_step.copy()
S1 = np.concatenate([[0.0], np.cumsum(r1)])
S12 = np.concatenate([[0.0], np.cumsum(r1 ** 2)])
mx1 = np.concatenate([[-np.inf], np.maximum.accumulate(r1)])
mn1 = np.concatenate([[np.inf], np.minimum.accumulate(r1)])
suf_mx1 = np.concatenate([np.maximum.accumulate(r1[::-1])[::-1], [-np.inf]])
suf_mn1 = np.concatenate([np.minimum.accumulate(r1[::-1])[::-1], [np.inf]])
m1b = S1[ks] / ks
m2b = (S1[n] - S1[ks]) / (n - ks)
maxdev1 = np.maximum.reduce([np.abs(mx1[ks] - m1b), np.abs(mn1[ks] - m1b),
                             np.abs(suf_mx1[ks] - m2b), np.abs(suf_mn1[ks] - m2b)])
k2 = int(ks[np.argmin(maxdev1)])
log('greedy SECOND step: after row idx %d -> max|res|=%.5f (was %.5f)'
    % (k2 - 1, maxdev1.min(), np.abs(resid_step).max()))
res['c_second_step_k'] = k2
res['c_second_step_maxres'] = float(maxdev1.min())

# --- (iii) reversed derivation: c03 from diff(c01)
log('\n--- (c)(iii) reversed derivation: c03 from diff(c01) ---')
d01 = np.diff(c01) * 100.0           # cm
d01r = np.round(d01, 6)
tgt = c03[:-1]
exact = float(np.mean(np.abs(d01r - tgt) < 1e-9))
close = float(np.mean(np.abs(d01r - tgt) < 0.005))
rs = np.round(d01r)
exact_int = float(np.mean(np.abs(rs - tgt) < 1e-9))
log('diff(c01)*100 distinct values: ' + ', '.join('%g(%d)' % (v, kk) for v, kk in zip(*np.unique(d01r, return_counts=True))))
log('frac |round(diff(c01)*100) - c03| == 0 : %.4f' % exact_int)
log('frac |diff(c01)*100 - c03| < 0.005    : %.4f' % close)
xt = pd.crosstab(pd.Series(np.round(d01r, 3), name='diff*100'), pd.Series(tgt, name='c03'))
log('crosstab diff(c01)*100 (rows) vs c03 (cols), cm:')
log(xt.to_string())
res['c_rev_exact_int_frac'] = exact_int
res['c_rev_close_frac'] = close

# --- forward direction quality: is c01 = round(cumsum + const, 2) per segment?
log('\n--- (c) forward: is c01 consistent with round(cumsum(c03)/100 + const, 2)? ---')
for s, name in ((0, 'pre-step'), (1, 'post-step')):
    m = seg == s
    w = r[m].max() - r[m].min()
    log('%s: band width of r = %.5f m -> consistent with cm-rounding (<=0.01)? %s'
        % (name, w, 'YES' if w <= 0.0100001 else 'NO'))
log('=> c03 carries 0.01-cm resolution; c01 is quantized to 0.01 m. '
    'Reverse derivation is information-destroying; forward is plausible only if band<=0.01.')

# exact-match test of forward rounded model per segment (scan around the segment mean)
for s, name in ((0, 'pre-step'), (1, 'post-step')):
    m = seg == s
    best = (0.0, 0.0)
    for off in np.round(np.arange(r[m].mean() - 0.03, r[m].mean() + 0.03, 0.0005), 6):
        frac = np.mean(np.abs(c01[m] - np.round(cs[m] + off, 2)) < 1e-9)
        if frac > best[0]:
            best = (frac, off)
    log('best per-segment const offset %s: off=%.4f -> exact c01 match frac=%.4f' % (name, best[1], best[0]))
    res['c_fwd_%d' % s] = dict(off=float(best[1]), frac=float(best[0]))

# --- step magnitude: spectroscopy of estimators (which one is ~1.008?) ---
log('\n--- (c) STEP MAGNITUDE under different estimators ---')
est = {}
est['raw c01 jump'] = float(c01[k] - c01[k - 1])
est['jump minus c03[k-1]/100'] = float(c01[k] - c01[k - 1] - c03[k - 1] / 100)
est['jump minus c03[k]/100'] = float(c01[k] - c01[k - 1] - c03[k] / 100)
est['jump minus mean c03[90:106]/100'] = float(c01[k] - c01[k - 1] - c03[90:k].mean() / 100)
est['mean(r_post)-mean(r_pre)'] = float(jump)
est['median(r_post)-median(r_pre)'] = float(np.median(r[k:]) - np.median(r[:k]))
est['midband(r_post)-midband(r_pre)'] = float((r[k:].max() + r[k:].min()) / 2 - (r[:k].max() + r[:k].min()) / 2)
est['OLS a+b*cs+step*I (step coef)'] = float(beta[2])
est['OLS a+b*cs+step*I (step - slope*dT)'] = float(beta[2] + (1 - beta[1]) * (cs[k] - cs[k - 1]))
for nm, v in est.items():
    log('  %-38s = %.5f' % (nm, v))
# per-segment affine fit, evaluate jump at breakpoint k
for s, name in ((0, 'pre'), (1, 'post')):
    m = seg == s
    ii = np.where(m)[0].astype(float)
    b2, b1 = np.polyfit(ii, r[m], 1)
    log('  affine fit %-4s: slope=%.3e per row, value at k=%.5f (rms=%.5f)' %
        (name, b1, b1 * k + b2, np.sqrt(np.mean((r[m] - (b1 * ii + b2)) ** 2))))
    res['c_affine_%s' % name] = dict(slope=float(b1), val_at_k=float(b1 * k + b2))
    res['c_affine_%s_max' % name] = float(np.abs(r[m] - (b1 * ii + b2)).max())
# alignment-shift test: does c03[i] describe interval [c01[i], c01[i+1]) or (c01[i-1], c01[i]]?
shift_cs = np.concatenate([[0.0], np.cumsum(c03[:-1])]) / 100.0
r_sh = c01 - shift_cs
log('  shifted alignment (c03[i] -> interval ending at c01[i]): pre-mean=%.5f post-mean=%.5f jump=%.5f max|res|=%.5f' %
    (r_sh[:k].mean(), r_sh[k:].mean(), r_sh[k:].mean() - r_sh[:k].mean(), np.abs(r_sh - np.where(seg == 0, r_sh[:k].mean(), r_sh[k:].mean())).max()))
res['c_shift_jump'] = float(r_sh[k:].mean() - r_sh[:k].mean())
res['c_shift_max'] = float(np.abs(r_sh - np.where(seg == 0, r_sh[:k].mean(), r_sh[k:].mean())).max())
# Chebyshev (min-max) vs least-squares segment constants
c_pre = (r[:k].max() + r[:k].min()) / 2
c_post = (r[k:].max() + r[k:].min()) / 2
log('  Chebyshev constants: pre=%.5f post=%.5f -> max|res|=%.5f' %
    (c_pre, c_post, max(r[:k].max() - c_pre, r[k:].max() - c_post)))
res['c_cheb_max'] = float(max(r[:k].max() - c_pre, r[k:].max() - c_post))
# residual band stats
log('  r pre : min=%.5f max=%.5f band=%.5f' % (r[:k].min(), r[:k].max(), r[:k].max() - r[:k].min()))
log('  r post: min=%.5f max=%.5f band=%.5f' % (r[k:].min(), r[k:].max(), r[k:].max() - r[k:].min()))
log('  c03 rows 95..112: ' + ' '.join('%g' % v for v in c03[95:113]))
log('  c01 rows 95..112: ' + ' '.join('%.2f' % v for v in c01[95:113]))
log('  diff(c01)*100 rows 95..112: ' + ' '.join('%g' % v for v in (np.diff(c01) * 100)[95:112]))
res['c_pre_band'] = float(r[:k].max() - r[:k].min())
res['c_post_band'] = float(r[k:].max() - r[k:].min())
# increment-level: is d01 cm an integer always?
d01cm = np.diff(c01) * 100
log('  d01*100: all integers? %s ; mean=%.4f vs mean(c03)=%.4f' %
    (bool(np.all(np.abs(d01cm - np.round(d01cm)) < 1e-9)), d01cm.mean(), c03.mean()))
log('  sum(d01*100)=%.1f  sum(c03)=%.1f  relative gap=%.6f' %
    (d01cm.sum(), c03.sum(), d01cm.sum() / c03.sum() - 1))

# ---------------------------------------------------------------- (d) c06 vs depth*density
log('\n===== (d) c06 vs c01*c05  (and alternatives) =====')
cands = {
    'c01*c05': c01 * c05,
    'c02*c05': c02 * c05,
    '(c01+c03/100)*c05': (c01 + c03 / 100.0) * c05,
    'c01*c05 + c03/100*c05': c01 * c05 + (c03 / 100.0) * c05,
}
for nm, v in cands.items():
    mx_, rms_, mabs_ = stats(c06, v)
    log('%-28s max|res|=%.5f rms=%.5f mean|res|=%.5f' % (nm, mx_, rms_, mabs_))
    res['d_' + nm] = dict(max_abs=mx_, rms=rms_)
# rounded variants
for nm, v in cands.items():
    rv = np.round(v, 2)
    mx_, rms_, mabs_ = stats(c06, rv)
    log('round(%-23s) max|res|=%.5f  frac exact=%d/%d' % (nm + ',2)', mx_, int((np.abs(c06 - rv) < 1e-9).sum()), n))
    res['d_round_' + nm] = dict(max_abs=mx_, exact=int((np.abs(c06 - rv) < 1e-9).sum()))
# cumulative density-weighted thickness
cum_w = np.cumsum(c03 / 100.0 * c05)
mx_, rms_, mabs_ = stats(c06, cum_w)
log('cumsum(c03/100*c05)          max|res|=%.5f rms=%.5f' % (mx_, rms_))
# product with c01/c02 quantized?
log('c06 distinct values=%d ; are all c06 multiples of 0.01? %s' %
    (len(np.unique(c06)), bool(np.all(np.abs(c06 * 100 - np.round(c06 * 100)) < 1e-9))))
res['d_c06_distinct'] = int(len(np.unique(c06)))

# where is the max residual for c01*c05 and c02*c05?
for nm in ('c01*c05', 'c02*c05'):
    dd = np.abs(c06 - cands[nm])
    idx = int(np.argmax(dd))
    log('max-res row for %s: idx=%d c01=%.2f c02=%.2f c05=%.3f c06=%.2f prod=%.4f' %
        (nm, idx, c01[idx], c02[idx], c05[idx], c06[idx], cands[nm][idx]))

# ---- task 3: BOTH directions, agreement fractions
log('\n--- (d)/(task3) direction test: c06=round(c01*c05,2) vs c05=round(c06/c01,3) ---')
num = c06 / c01
bad = ~np.isfinite(num)
A_exact = np.abs(c06 - np.round(c01 * c05, 2)) < 1e-9
B_exact = np.abs(c05 - np.round(np.where(bad, np.nan, num), 3)) < 1e-9
A_5e4 = np.abs(c06 - c01 * c05) <= 0.0005 + 1e-12
B_5e4 = np.abs(c05 - np.where(bad, np.nan, num)) <= 0.0005 + 1e-12
log('direction A  c06 == round(c01*c05,2) exact frac = %.4f (%d/%d)' % (A_exact.mean(), A_exact.sum(), n))
log('direction B  c05 == round(c06/c01,3) exact frac = %.4f (%d/%d)' % (B_exact.mean(), B_exact.sum(), n))
log('A |c06-c01*c05|<=0.0005 frac=%.4f   B |c05-c06/c01|<=0.0005 frac=%.4f' % (A_5e4.mean(), B_5e4.mean()))
log('A max|c06-c01*c05|=%.6f   B max|c05-c06/c01|=%.6f' %
    (np.abs(c06 - c01 * c05).max(), np.nanmax(np.abs(c05 - np.where(bad, np.nan, num)))))
# same for c02 direction
A2 = np.abs(c06 - np.round(c02 * c05, 2)) < 1e-9
B2 = np.abs(c05 - np.round(c06 / c02, 3)) < 1e-9
log('with c02: A2 exact frac=%.4f  B2 exact frac=%.4f' % (A2.mean(), B2.mean()))
res['task3'] = dict(A_exact=float(A_exact.mean()), B_exact=float(B_exact.mean()),
                    A2_exact=float(A2.mean()), B2_exact=float(B2.mean()),
                    A_5e4=float(A_5e4.mean()), B_5e4=float(B_5e4.mean()),
                    n_zero_c01=int((c01 == 0).sum()))

# ---------------------------------------------------------------- (e) c05 structure
log('\n===== (e) c05 structure =====')
u = np.unique(c05)
log('c05 distinct=%d (claim 261), min=%.4f max=%.4f' % (len(u), u.min(), u.max()))
log('c05 non-decreasing as a sequence? %s ; strictly increasing? %s' %
    (bool(np.all(np.diff(c05) >= -1e-12)), bool(np.all(np.diff(c05) > 0))))
log('c05 all multiples of 0.001? %s' % bool(np.all(np.abs(c05 * 1000 - np.round(c05 * 1000)) < 1e-9)))
runs = 1 + int((np.diff(c05) != 0).sum())
log('runs of constant c05 = %d ; max run length = %d' % (runs, int(pd.Series(c05).value_counts().max())))
res['e_distinct'] = int(len(u))
res['e_nondecreasing'] = bool(np.all(np.diff(c05) >= -1e-12))
res['e_runs'] = runs
# quadratic in c01
p1 = np.polyfit(c01, c05, 2)
rp1 = c05 - np.polyval(p1, c01)
p2 = np.polyfit(np.arange(n), c05, 2)
rp2 = c05 - np.polyval(p2, np.arange(n))
p3 = np.polyfit(c06, c05, 2)
rp3 = c05 - np.polyval(p3, c06)
log('quadratic in c01 : max|res|=%.6f rms=%.6f (claim max 0.00055)' % (np.abs(rp1).max(), np.sqrt(np.mean(rp1**2))))
log('quadratic in idx : max|res|=%.6f rms=%.6f' % (np.abs(rp2).max(), np.sqrt(np.mean(rp2**2))))
log('quadratic in c06 : max|res|=%.6f rms=%.6f' % (np.abs(rp3).max(), np.sqrt(np.mean(rp3**2))))
p4 = np.polyfit(c01, c05, 4)
rp4 = c05 - np.polyval(p4, c01)
log('quartic   in c01 : max|res|=%.6f rms=%.6f' % (np.abs(rp4).max(), np.sqrt(np.mean(rp4**2))))
log('c05 = round(quadratic(c01),3) exact frac=%.4f' % np.mean(np.abs(c05 - np.round(np.polyval(p1, c01), 3)) < 1e-9))
res['e_quad_c01_max'] = float(np.abs(rp1).max())
res['e_quad_c01_rms'] = float(np.sqrt(np.mean(rp1 ** 2)))
res['e_quad_idx_max'] = float(np.abs(rp2).max())
res['e_quad_c06_max'] = float(np.abs(rp3).max())
res['e_quartic_c01_max'] = float(np.abs(rp4).max())
# equal-c01 implies equal-c05 test (vacuous if c01 unique)
log('c01 distinct=%d -> test "equal c01 => equal c05" is vacuous' % len(np.unique(c01)))
# residual structure vs depth: is the quadratic residual smooth?
seg_r = rp1
log('quad-resid in first/last 10%%: max|.|=%.6f / %.6f' %
    (np.abs(seg_r[:n//10]).max(), np.abs(seg_r[-n//10:]).max()))

# ---------------------------------------------------------------- ATTACK: all pairs
log('\n===== ATTACK 1: systematic pair scan =====')
pairs = [('c01', c01), ('c02', c02), ('c03', c03), ('c04', c04), ('c05', c05), ('c06', c06), ('c08', c08), ('c09', c09)]
log('pairwise |corr| on complete rows (Pearson):')
for a in range(len(pairs)):
    for b in range(a + 1, len(pairs)):
        x, y = pairs[a][1], pairs[b][1]
        m = np.isfinite(x) & np.isfinite(y)
        cc = np.corrcoef(x[m], y[m])[0, 1]
        if abs(cc) > 0.9:
            log('  %s-%s r=%.4f' % (pairs[a][0], pairs[b][0], cc))
# specific "hidden function" probes
log('repeat-value probes:')
for nm, v in (('c04', c04), ('c06', c06), ('c05', c05), ('c02', c02)):
    vv = v[np.isfinite(v)]
    uu, cc = np.unique(vv, return_counts=True)
    dup = uu[cc > 1]
    log('  %s: %d distinct / %d rows -> %d repeated values' % (nm, len(uu), len(vv), len(dup)))
    if len(dup) and nm != 'c05':
        for dv in dup[:5]:
            idx = np.where(v == dv)[0]
            log('    %s=%g at rows %s ; c01=%s ; c05=%s ; c06=%s' %
                (nm, dv, idx.tolist(), np.round(c01[idx], 3).tolist(),
                 np.round(c05[idx], 3).tolist(), np.round(c06[idx], 2).tolist()))
# does equal c06 imply equal c01 (or vice versa)?
uu, cc = np.unique(c06, return_counts=True)
dups = uu[cc > 1]
spread = []
for dv in dups:
    idx = np.where(c06 == dv)[0]
    spread.append((np.ptp(c01[idx]), dv, idx.tolist(), np.ptp(c05[idx])))
spread.sort(reverse=True)
log('  repeated c06 -> max spread of c01 = %.4f m (n_dup=%d)' % (spread[0][0], len(dups)))
log('  worst: c06=%g rows %s c01 spread %.4f c05 spread %.4f' % (spread[0][1], spread[0][2], spread[0][0], spread[0][3]))
# c04 vs geometry
log('c04 vs geometry: R^2 of c04 on cubic in c01 = %.4f'
    % (1 - np.var(c04[np.isfinite(c04)] - np.polyval(np.polyfit(c01[np.isfinite(c04)], c04[np.isfinite(c04)], 3), c01[np.isfinite(c04)])) / np.var(c04[np.isfinite(c04)])))
# c07 vs c04: are the 15 labelled rows local maxima of c04?
log('\nATTACK 1b: is c07 derivable from c04 (local-max rule)?')
lab_idx = np.where(~pd.isna(c07))[0]
log('labelled rows: %s' % lab_idx.tolist())
for w in (1, 2, 3, 5, 10, 20):
    good = 0
    for ii in lab_idx:
        lo, hi = max(0, ii - w), min(n, ii + w + 1)
        if c04[ii] >= np.nanmax(c04[lo:hi]) - 1e-12:
            good += 1
    fp = 0
    for ii in range(n):
        lo, hi = max(0, ii - w), min(n, ii + w + 1)
        if c04[ii] >= np.nanmax(c04[lo:hi]) - 1e-12:
            fp += 1
    log('  window +-%d: labelled rows that are window max = %d/%d ; total rows that are window max = %d -> precision %.3f'
        % (w, good, len(lab_idx), fp, good / max(fp, 1)))
log('c04 at labelled rows: ' + ', '.join('%.1f' % c04[ii] for ii in lab_idx))
log('c04 quantile of labelled rows: ' + ', '.join('%.3f' % (np.mean(c04[~np.isnan(c04)] < c04[ii]))
    for ii in lab_idx))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(c01, r, '.', ms=1)
ax[0, 0].axvline(c01[k], color='r', lw=0.8)
ax[0, 0].set_title('c01 - cumsum(c03)/100 (break at c01=%.2f, jump=%.3f m)' % (c01[k], jump))
ax[0, 0].set_xlabel('c01 (m)'); ax[0, 0].set_ylabel('residual (m)')
ax[0, 1].plot(c01, resid_step, '.', ms=1)
ax[0, 1].set_title('residual after single step model (max=%.4f m)' % np.abs(resid_step).max())
ax[0, 1].set_xlabel('c01 (m)'); ax[0, 1].set_ylabel('residual (m)')
ax[1, 0].plot(c01, c05, '.', ms=1, label='c05')
ax[1, 0].plot(c01, np.polyval(p1, c01), '-', lw=0.8, label='quadratic fit')
ax[1, 0].set_title('c05 vs c01 (max quad resid %.5f)' % np.abs(rp1).max())
ax[1, 0].set_xlabel('c01 (m)'); ax[1, 0].set_ylabel('density'); ax[1, 0].legend()
dd1 = np.abs(c06 - c01 * c05); dd2 = np.abs(c06 - c02 * c05)
ax[1, 1].plot(c01, dd1, '.', ms=1, label='|c06-c01*c05| max %.4f' % dd1.max())
ax[1, 1].plot(c01, dd2, '.', ms=1, label='|c06-c02*c05| max %.4f' % dd2.max())
ax[1, 1].set_yscale('log'); ax[1, 1].legend(); ax[1, 1].set_xlabel('c01 (m)')
ax[1, 1].set_title('c06 residual vs depth')
fig.tight_layout()
fig.savefig(OUT + '/g7_indep_figure.png', dpi=110)
log('\nsaved figure: %s' % (OUT + '/g7_indep_figure.png'))

with open(OUT + '/g7_indep_results.json', 'w') as f:
    json.dump(res, f, indent=2, default=float)
with open(OUT + '/g7_indep_log.txt', 'w') as f:
    f.write('\n'.join(LOG) + '\n')
log('saved json/log')
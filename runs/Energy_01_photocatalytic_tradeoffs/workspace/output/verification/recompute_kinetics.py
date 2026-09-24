import numpy as np, json, pandas as pd
from scipy.optimize import least_squares

df = pd.read_csv('/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/conversion_timecourses.tsv', sep='\t')
res = {}
for fid, g in df.groupby('formulation_id', sort=True):
    g = g.sort_values('illumination_time_h')
    t = g['illumination_time_h'].to_numpy(float)
    y = g['gaseous_product_umol'].to_numpy(float)
    n = len(t)
    final = float(y[t == 4.0][0]) if (t == 4.0).any() else float(y[-1])
    mean_rate = final / 4.0
    seg = np.diff(y) / np.diff(t)
    ratio = float(seg[-1] / seg[0])

    def rss_of(pred):
        return float(np.sum((y - pred)**2))

    def stats(pred, kp):
        r = rss_of(pred)
        r2 = 1 - r / np.sum((y - y.mean())**2)
        rmse = np.sqrt(r / n)
        denom = n - kp - 1
        aicc = (n*np.log(r/n) + 2*kp + 2*kp*(kp+1)/denom) if denom > 0 else float('inf')
        return dict(R2=float(r2), RMSE=float(rmse), RSS=float(r), AICc=float(aicc))

    models = {}
    # power law y = a t^b (all 5 points; t=0 contributes residual 0 when b>0)
    def pl_res(p):
        a, b = p
        tb = np.where(t > 0, np.power(np.where(t > 0, t, 1.0), b), 0.0)
        return a * tb - y
    best = None
    for a0 in [0.5, 1, 2, 5, 10]:
        for b0 in [0.3, 0.6, 1.0, 1.5, 2.0]:
            s = least_squares(pl_res, [a0, b0], method='lm', max_nfev=20000)
            if best is None or s.cost < best.cost:
                best = s
    a_pl, b_pl = best.x
    models['power_law'] = dict(params=dict(a=float(a_pl), b=float(b_pl)), **stats(a_pl*np.power(t,b_pl), 3))

    # line
    c = np.polyfit(t, y, 1)
    models['linear'] = dict(params=dict(slope=float(c[0]), intercept=float(c[1])), **stats(np.polyval(c, t), 3))

    # quadratic
    c2 = np.polyfit(t, y, 2)
    models['quadratic'] = dict(params=dict(a2=float(c2[0]), a1=float(c2[1]), a0=float(c2[2])), **stats(np.polyval(c2, t), 4))

    # saturation y = A(1-exp(-k t))
    def sat_res(p):
        A, k = p
        return A*(1-np.exp(-k*t)) - y
    bs = None
    for A0 in [1, 5, 10, 20]:
        for k0 in [0.05, 0.2, 0.5, 1.0, 2.0]:
            s = least_squares(sat_res, [A0, k0], method='lm', max_nfev=20000)
            if bs is None or s.cost < bs.cost:
                bs = s
    A_s, k_s = bs.x
    models['saturation'] = dict(params=dict(A=float(A_s), k=float(k_s)), **stats(A_s*(1-np.exp(-k_s*t)), 3))

    best_model = min(models, key=lambda m: models[m]['AICc'])
    res[fid] = dict(times=t.tolist(), y=y.tolist(), final_product_4h=final, mean_rate_over_4h=mean_rate,
                    segment_rates=[float(v) for v in seg], ratio_last_first=ratio,
                    models=models, best_model_by_AICc=best_model)

print(json.dumps(res, indent=2))
with open('/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/recomputed_kinetics.json','w') as f:
    json.dump(res, f, indent=2)

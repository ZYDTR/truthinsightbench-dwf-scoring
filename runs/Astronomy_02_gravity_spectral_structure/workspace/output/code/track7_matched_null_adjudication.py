import sys; sys.path.insert(0,'/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code')
import numpy as np
from common import load_traces, ls_power, freq_grid, ar1_params, ar1_surrogate
tr = load_traces()
fu, fh = freq_grid(200,3000,5.0)
rng = np.random.default_rng(4242)
groups = list(tr.groupby('pass_id'))
for tag,col in [('A','residual_model_A_mm_s'),('B','residual_model_B_mm_s'),('C','residual_model_C_mm_s')]:
    P = np.array([ls_power(g.relative_time_min.values, g[col].values, fh) for _,g in groups])
    obs = float(P.mean(axis=0).max())
    null = []
    for _ in range(120):
        Ps = []
        for _,g in groups:
            a,s = ar1_params(g[col].values)
            Ps.append(ls_power(g.relative_time_min.values, ar1_surrogate(len(g),a,s,rng), fh))
        null.append(float(np.array(Ps).mean(axis=0).max()))
    null = np.array(null)
    p = float((null>=obs).mean())
    k = int(np.argmax(P.mean(axis=0)))
    print(f"{tag}: obs_max_of_pass_mean={obs:.4f} null_q50={np.median(null):.4f} q95={np.percentile(null,95):.4f} p={p:.4f} peak={fu[k]:.0f}uHz inband={860<=fu[k]<=1310}")

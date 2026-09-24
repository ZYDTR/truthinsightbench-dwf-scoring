import numpy as np, pandas as pd, json

SRC = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/ultrafast_transient_traces.tsv"
df = pd.read_csv(SRC, sep="\t")
print(df.groupby("formulation_id").size())
print("delay range:", df.delay_ps.min(), df.delay_ps.max())

SEED = 20260919
NBOOT = 20000

def mbb_ci(x, L, nboot=NBOOT, seed=SEED):
    """Moving-block bootstrap CI (percentile) for the mean of x."""
    n = len(x)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(nboot, nblocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(nboot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), means

out = {}
for fid, g in df.groupby("formulation_id", sort=True):
    g = g.sort_values("delay_ps").reset_index(drop=True)
    d = g.delay_ps.values; y = g.normalized_transient_signal.values
    # peak
    imax = int(np.argmax(y))
    peak_val = float(y[imax]); peak_delay = float(d[imax])
    # also delay of max as reported (ties?)
    n_ties = int(np.sum(y == y[imax]))
    # >= 30 ps
    m = d >= 30.0
    yl = y[m]; n = int(m.sum())
    mean = float(yl.mean()); med = float(np.median(yl))
    frac_pos = float(np.mean(yl > 0))
    L = max(2, int(np.sqrt(n)))
    ci, boot = mbb_ci(yl, L)
    out[fid] = dict(n=n, mean=mean, median=med, frac_pos=frac_pos,
                    n_pos=int(np.sum(yl > 0)),
                    block_len=L, ci=[float(ci[0]), float(ci[1])],
                    peak_delay=peak_delay, peak_value=peak_val,
                    peak_ties=n_ties,
                    first_delay_ge30=float(d[m][0]), last_delay=float(d[m][-1]))
    print(fid, out[fid])

# difference of means, independent resampling of each trace's long-delay block
oms = {}
for fid in ["formulation_1", "formulation_2"]:
    g = df[df.formulation_id == fid].sort_values("delay_ps")
    y = g[g.delay_ps >= 30].normalized_transient_signal.values
    oms[fid] = y
y1, y2 = oms["formulation_1"], oms["formulation_2"]
n1, n2 = len(y1), len(y2)
L1 = max(2, int(np.sqrt(n1))); L2 = max(2, int(np.sqrt(n2)))
rng = np.random.default_rng(SEED + 1)
def resample_means(x, L, nboot, rng):
    n = len(x); nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(nboot, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(nboot, -1)[:, :n]
    return x[idx].mean(axis=1), L
b1, _ = resample_means(y1, L1, NBOOT, rng)
b2, _ = resample_means(y2, L2, NBOOT, rng)
diff = b1 - b2
dci = np.percentile(diff, [2.5, 97.5])
diff_point = float(y1.mean() - y2.mean())
out["difference"] = dict(point=diff_point, ci=[float(dci[0]), float(dci[1])],
                         block_len_1=int(L1), block_len_2=int(L2))
print("difference", out["difference"])

with open("/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/transient_verification.json", "w") as f:
    json.dump({"seed": SEED, "n_boot": NBOOT, "results": out}, f, indent=2)
print("done")

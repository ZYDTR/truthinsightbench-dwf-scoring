#!/usr/bin/env python3
"""Phase 1 profiling of anonymous trajectories (pure numpy)."""
import json
import numpy as np

DATA = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project/data/anonymous_trajectories.npz"
OUT_BLOCKVAR = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project/output/results/profile_blockvar.csv"
OUT_SUMMARY = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project/output/results/profile_summary.json"

def main():
    # a. Load and shape/dtype check
    with np.load(DATA) as z:
        X = np.asarray(z["trajectories"], dtype=float)
        T = np.asarray(z["time_index"], dtype=int)
    n_traj, n_steps = X.shape
    print("shape:", X.shape, "dtype:", X.dtype, "time_index:", T[:5], "...", T[-5:], "contiguous:", bool(np.all(np.diff(T) == 1)))
    start = X[:, 0]
    all_start_zero = bool(np.all(start == 0.0))
    print("all_start_zero:", all_start_zero, "n_start_not_zero:", int(np.sum(start != 0.0)))

    # b. Endpoints (last step)
    ep = X[:, -1]
    endpoint_mean = float(np.mean(ep))
    endpoint_sd = float(np.std(ep, ddof=0))
    endpoint_min = float(np.min(ep))
    endpoint_max = float(np.max(ep))
    endpoint_abs_mean = float(np.mean(np.abs(ep)))
    print("endpoint mean/sd/min/max/|mean|:", endpoint_mean, endpoint_sd, endpoint_min, endpoint_max, endpoint_abs_mean)

    # c. Per-trajectory increment variance (population, ddof=0)
    inc = np.diff(X, axis=1)  # (240, 127)
    inc_var = np.var(inc, axis=1)
    iv_mean = float(np.mean(inc_var))
    iv_median = float(np.median(inc_var))
    iv_min = float(np.min(inc_var))
    iv_max = float(np.max(inc_var))
    iv_sd = float(np.std(inc_var, ddof=0))
    iv_n_zero = int(np.sum(inc_var == 0.0))
    iv_n_above_1 = int(np.sum(inc_var >= 1.0))
    iv_n_below_0_05 = int(np.sum(inc_var <= 0.05))
    print("inc_var mean/median/min/max/sd:", iv_mean, iv_median, iv_min, iv_max, iv_sd)
    print("inc_var n_zero:", iv_n_zero, "n>=1:", iv_n_above_1, "n<=0.05:", iv_n_below_0_05)

    # d. Per-trajectory lag-1 increment autocorrelation (Pearson, manual)
    a = inc[:, :-1]
    b = inc[:, 1:]
    ac_mean = a.mean(axis=1)
    ac = (a - ac_mean[:, None])
    bc = (b - b.mean(axis=1)[:, None])
    na = np.linalg.norm(ac, axis=1)
    nb = np.linalg.norm(bc, axis=1)
    denom = na * nb
    with np.errstate(divide="ignore", invalid="ignore"):
        lag1 = np.sum(ac * bc, axis=1) / denom
    # handle degenerate zero-norm cases (variance 0 -> NaN)
    lag1 = np.where(np.isnan(lag1), 0.0, lag1)
    lag1_mean = float(np.mean(lag1))
    lag1_sd = float(np.std(lag1, ddof=0))
    print("lag1_autocorr mean/sd:", lag1_mean, lag1_sd)

    # e. Block-wise increment variance: 8 contiguous blocks of 16 steps.
    #    X has 128 points -> 127 increments. 8 blocks of 16 X-points each:
    #    X[0:16], X[16:32], ..., X[112:128]. Within each block, the first
    #    differences are inc[start:end-1] (15 values). Contiguous time used.
    B = 8
    BLEN = 16
    blockvar = np.empty((n_traj, B), dtype=float)
    for k in range(B):
        start = k * BLEN
        end = (k + 1) * BLEN
        blockvar[:, k] = np.var(inc[:, start:end - 1], axis=1)
    print("blockvar shape:", blockvar.shape)
    # write CSV
    with open(OUT_BLOCKVAR, "w") as f:
        f.write("traj_id,block0,block1,block2,block3,block4,block5,block6,block7\n")
        for i in range(n_traj):
            row = ",".join([repr(float(v)) for v in blockvar[i]])
            f.write(f"{i},{row}\n")

    # f. Multi-step displacement M(h) = mean over traj,t of (X[t+h]-X[t])^2.
    #    X[:, h:] - X[:, :128-h] gives X[t+h]-X[t] for t=0..127-h (elementwise,
    #    NOT np.diff(...,n=h) which would be the h-th order difference).
    M = {}
    for h in [1, 2, 4, 8, 16, 32, 64]:
        d = X[:, h:] - X[:, :128 - h]
        M[h] = float(np.mean(d * d))
        print(f"M_{h}:", M[h])

    # g. Per-trajectory max |increment| and max |second difference|
    max_abs_inc = np.max(np.abs(inc), axis=1)
    max_abs_2nd = np.max(np.abs(np.diff(inc, axis=1)), axis=1)
    max_abs_inc_mean = float(np.mean(max_abs_inc))
    max_abs_inc_min = float(np.min(max_abs_inc))
    max_abs_inc_max = float(np.max(max_abs_inc))
    max_abs_2nd_mean = float(np.mean(max_abs_2nd))
    max_abs_2nd_min = float(np.min(max_abs_2nd))
    max_abs_2nd_max = float(np.max(max_abs_2nd))
    print("max|inc| mean/min/max:", max_abs_inc_mean, max_abs_inc_min, max_abs_inc_max)
    print("max|2nd| mean/min/max:", max_abs_2nd_mean, max_abs_2nd_min, max_abs_2nd_max)

    # h. JSON summary
    summary = {
        "n_traj": int(n_traj),
        "n_steps": int(n_steps),
        "X_dtype": X.dtype.name,
        "all_start_zero": bool(all_start_zero),
        "endpoint_mean": endpoint_mean,
        "endpoint_sd": endpoint_sd,
        "endpoint_min": endpoint_min,
        "endpoint_max": endpoint_max,
        "endpoint_abs_mean": endpoint_abs_mean,
        "inc_var_mean": iv_mean,
        "inc_var_median": iv_median,
        "inc_var_min": iv_min,
        "inc_var_max": iv_max,
        "inc_var_sd": iv_sd,
        "inc_var_n_zero": iv_n_zero,
        "inc_var_n_above_1": iv_n_above_1,
        "inc_var_n_below_0_05": iv_n_below_0_05,
        "lag1_autocorr_mean": lag1_mean,
        "lag1_autocorr_sd": lag1_sd,
        "M_h_1": M[1],
        "M_h_2": M[2],
        "M_h_4": M[4],
        "M_h_8": M[8],
        "M_h_16": M[16],
        "M_h_32": M[32],
        "M_h_64": M[64],
        "max_abs_inc_mean": max_abs_inc_mean,
        "max_abs_inc_min": max_abs_inc_min,
        "max_abs_inc_max": max_abs_inc_max,
        "max_abs_2nd_mean": max_abs_2nd_mean,
        "max_abs_2nd_min": max_abs_2nd_min,
        "max_abs_2nd_max": max_abs_2nd_max,
        "blockvar_shape": [int(x) for x in blockvar.shape],
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", OUT_BLOCKVAR, "and", OUT_SUMMARY)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
panel_d — real temporal-network delay vs buffer in fig4b_RW_networks.

Network pairs (buffer, delay):
  Thiers13    -> linear buffer scan  (buffer_lin_Thiers13)
  Workplace15 -> log buffer scan     (buffer_log_Workplace15)

Tasks:
  1. Fit delay = a + b*buffer ; delay = a*exp(b*buffer) ; delay = a + b*log(buffer+1).
     Report coefficients and R^2.
  2. Monotonicity: is delay monotone decreasing in buffer? Total drop min->max buffer,
     per-unit slope.
  3. Compare shape under the two scan types (linear vs log) — apparent transition.
  4. Robustness: refit excluding first point; refit over top half of buffers.
  5. Interpretation: what the panel shows re: real-network delay dynamics vs a critical buffer.

No scipy -> exponential fit done by 1-D grid on exponent c with a solved analytically.
"""
import os, json
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project"
OUT_RESULTS = os.path.join(BASE, "output", "results")
os.makedirs(OUT_RESULTS, exist_ok=True)

# ---------------------------------------------------------------- load data
def load(path):
    return pd.read_csv(path)

th = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "fig_source_data", "Thiers13_source_data.csv"))
wp = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "fig_source_data", "Workplace15_source_data.csv"))

bl = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "data", "buffer_lin.csv"))
bp = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "data", "buffer_log.csv"))
td = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "data", "Thiers13_delays.csv"))
wd = load(os.path.join(BASE, "data", "source_data", "figures",
                       "fig4b_RW_networks", "data", "Workplace15_delays.csv"))

def first_col(df):
    return df.iloc[:, 0].to_numpy(dtype=float)

assert "buffer_lin_Thiers13" in th.columns and "Thiers13_delays" in th.columns
assert "buffer_log_Workplace15" in wp.columns and "Workplace15_delays" in wp.columns

networks = {
    "Thiers13":  {"buf": th["buffer_lin_Thiers13"].to_numpy(dtype=float),
                  "delay": th["Thiers13_delays"].to_numpy(dtype=float),
                  "scan": "linear",
                  "buf_standalone": first_col(bl), "delay_standalone": first_col(td)},
    "Workplace15": {"buf": wp["buffer_log_Workplace15"].to_numpy(dtype=float),
                    "delay": wp["Workplace15_delays"].to_numpy(dtype=float),
                    "scan": "log",
                    "buf_standalone": first_col(bp), "delay_standalone": first_col(wd)},
}

for name, d in networks.items():
    ok_b = np.allclose(sorted(d["buf"]), sorted(d["buf_standalone"]))
    ok_y = np.allclose(sorted(d["delay"]), sorted(d["delay_standalone"]))
    print(f"[consistency] {name}: buffer matches standalone={ok_b}, delay matches standalone={ok_y}")

# ---------------------------------------------------------------- fit helpers
def r2(y, yhat):
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    return (1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def ols(X, y):
    X = np.atleast_2d(X.astype(float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return beta.tolist(), r2(y, yhat)

def fit_linear(x, y):
    a, r2v = ols(np.column_stack([np.ones_like(x), x]), y)
    return a[0], a[1], r2v

def fit_log(x, y):
    a, r2v = ols(np.column_stack([np.ones_like(x), np.log(x + 1.0)]), y)
    return a[0], a[1], r2v

def fit_exp(x, y):
    """
    y = a*exp(c*x).  For a fixed c, optimal a = sum(exp(cx)*y)/sum(exp(cx)^2).
    Minimize SSE over c on a grid (c expected negative for decay).
    Returns (a, c, r2_original, r2_log, yhat).
    """
    y = y.astype(float)
    best = None
    for lo, hi, npts in [(-20, -0.01, 3000), (-0.01, 0.5, 400)]:
        cg = np.linspace(lo, hi, npts)
        for c in cg:
            ex = np.exp(c * x)
            denom = float(np.sum(ex * ex))
            if denom <= 0:
                continue
            a = float(np.sum(ex * y) / denom)
            yhat = a * ex
            ss_res = float(np.sum((y - yhat)**2))
            if best is None or ss_res < best[0]:
                best = (ss_res, a, c)
    ss_res, a, c = best
    yhat = a * np.exp(c * x)
    r2_orig = r2(y, yhat)
    m = y > 0
    if np.sum(m) >= 3:
        _, rl = ols(np.column_stack([np.ones_like(y[m]), c * x[m]]), np.log(y[m]))
        r2_log = rl
    else:
        r2_log = np.nan
    return a, c, r2_orig, r2_log, yhat

# ---------------------------------------------------------------- compute per network
rows = []

def run(name, buf, delay):
    X = {"network": name, "scan": networks[name]["scan"], "n_full": len(buf)}
    X["buf_min"] = float(np.min(buf)); X["buf_max"] = float(np.max(buf))
    X["delay_min"] = float(np.min(delay)); X["delay_max"] = float(np.max(delay))
    z = np.where(np.abs(buf) < 1e-9)[0]
    X["delay_at_buf0"] = float(delay[z[0]]) if len(z) else np.nan
    X["delay_first_point"] = float(delay[0])

    order = np.argsort(buf)
    bs, ds = buf[order], delay[order]
    diffs = np.diff(ds)
    n_dec = int(np.sum(diffs < 0)); n_inc = int(np.sum(diffs > 0))
    X["frac_decreasing"] = round(float(n_dec / max(1, len(diffs))), 6)
    X["frac_increasing"] = round(float(n_inc / max(1, len(diffs))), 6)
    X["n_increases"] = n_inc
    X["drop_min_to_max"] = round(float(ds[0] - ds[-1]), 6)
    X["per_unit_slope"] = round(float((ds[-1] - ds[0]) / (bs[-1] - bs[0])), 6)
    d0 = float(ds[0])
    half = 0.5 * d0
    idx = int(np.argmin(np.abs(ds - half)))
    X["buf_at_half_delay"] = round(float(bs[idx]), 4)
    X["delay_at_max_buffer"] = round(float(ds[-1]), 8)

    for variant, m in [("full", np.ones(len(buf), bool)),
                       ("excl_first", np.ones(len(buf), bool)),
                       ("top_half", None)]:
        if variant == "excl_first":
            m[0] = False
        elif variant == "top_half":
            top = np.argsort(buf)[-len(buf)//2:]
            m = np.zeros(len(buf), bool); m[top] = True
        if not np.any(m):
            continue
        xb, yb = buf[m], delay[m]
        if len(xb) < 3:
            continue
        a_lin, c_lin, r2_lin = fit_linear(xb, yb)
        a_exp, c_exp, r2_exp, r2_explog, _ = fit_exp(xb, yb)
        a_log, c_log, r2_logm = fit_log(xb, yb)
        for model, (a, c, r2v) in [("linear", (a_lin, c_lin, r2_lin)),
                                   ("exponential", (a_exp, c_exp, r2_exp)),
                                   ("log", (a_log, c_log, r2_logm))]:
            rows.append({"network": name, "scan": X["scan"], "variant": variant,
                         "n": int(len(xb)), "model": model, "a": a, "c": c,
                         "R2": r2v, "buf_min": float(np.min(xb)), "buf_max": float(np.max(xb))})
    return X

meta = {}
for name in networks:
    meta[name] = run(name, networks[name]["buf"], networks[name]["delay"])

df_res = pd.DataFrame(rows)
csv_path = os.path.join(OUT_RESULTS, "panel_d_results.csv")
df_res.to_csv(csv_path, index=False)
with open(os.path.join(OUT_RESULTS, "panel_d_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=float)

print("=== RESULTS ===")
print(df_res.to_string())
print("\nmeta:", json.dumps(meta, indent=2))
print("\nsaved:", csv_path)

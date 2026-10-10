#!/usr/bin/env python3
"""f2: spatial (coord_y) gradient on strength and distance; system-dependence test.

Uses pandas/numpy only. Pearson p-values computed via the t-distribution
CDF implemented with the regularized incomplete beta function (Lanczos
gamma + continued fraction), so no scipy is required.
"""
import json
import math

import numpy as np
import pandas as pd

DATA_ROOT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data"
OUT_ROOT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output"


# ---------- incomplete beta / t-distribution p-value (scipy-free) ----------
_LANCZOS_G = [76.18009172947146, -86.50532032941677, 24.01409824083091,
              -21.28271230987814, 1.150260885835548e1, -4.094387182770938e1,
              4.991256370743705e2, -3.59991109373941e3, 1.73611777933143e4,
              -5.007379898444474e3, 4.161456862178782e1, -1.901123046495472e-1]


def _log_gamma(z):
    """Lanczos approximation of log(Gamma(z))."""
    z -= 1
    x = _LANCZOS_G[0]
    for i in range(1, len(_LANCZOS_G)):
        x += _LANCZOS_G[i] / (z + i)
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(z + 1) - (z + 1) + math.log(x)


def _log_beta(a, b):
    return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


def _beta_cf(a, b, x):
    """Continued fraction for I_x(a,b) (Lentz's method)."""
    MAXIT = 200
    EPS = 3e-15
    FPMIN = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < EPS:
            break
    # I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF
    return math.exp(a * math.log(x) + b * math.log(1.0 - x) - math.log(a)
                    - _log_beta(a, b) + math.log(h))


def _t_sf(t, df):
    """Two-sided survival function (p-value) of Student t with df degrees of freedom."""
    x = df / (df + t * t)
    return _beta_cf(df / 2.0, 0.5, x)


def pearson(x, y):
    """Pearson r, t statistic, two-sided p-value, n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    if n < 3:
        return {"r": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    t = r * math.sqrt(n - 2) / math.sqrt(max(1.0 - r * r, 0.0))
    p = _t_sf(abs(t), n - 2)
    p = float(np.clip(2.0 * p, 0.0, 1.0))
    return {"r": r, "t": float(t), "p": p, "n": n}


def main():
    summary = pd.read_csv(
        DATA_ROOT + "/network_node_summary.tsv", sep="\t", dtype={"region_index": int}
    )
    profile = pd.read_csv(
        DATA_ROOT + "/network_profile_a.tsv", sep="\t", dtype={"region_index": int}
    )
    df = summary.merge(profile, on="region_index", how="inner")
    df = df.dropna(subset=["node_strength", "mean_connection_distance", "coord_x", "coord_y", "coord_z"])
    n_nodes = len(df)

    # overall Pearson correlations
    strength = df["node_strength"].values
    distance = df["mean_connection_distance"].values
    cx = df["coord_x"].values
    cy = df["coord_y"].values
    cz = df["coord_z"].values

    overall = {
        "strength_coord_y": pearson(strength, cy),
        "distance_coord_y": pearson(distance, cy),
        "strength_coord_x": pearson(strength, cx),
        "strength_coord_z": pearson(strength, cz),
    }

    # per-system (4 systems, region_index // 200)
    df["system"] = (df["region_index"] // 200).astype(int)
    per_system = {"strength_coord_y": {}, "distance_coord_y": {}}
    for s in sorted(df["system"].unique()):
        sub = df[df["system"] == s]
        per_system["strength_coord_y"][int(s)] = float(
            np.corrcoef(sub["node_strength"].values, sub["coord_y"].values)[0, 1]
        )
        per_system["distance_coord_y"][int(s)] = float(
            np.corrcoef(sub["mean_connection_distance"].values, sub["coord_y"].values)[0, 1]
        )

    # sign conflict: do per-system r's differ in sign?
    def sign_pattern(vals):
        signs = []
        for v in vals:
            if v > 1e-12:
                signs.append("+")
            elif v < -1e-12:
                signs.append("-")
            else:
                signs.append("0")
        uniq = sorted(set(signs))
        return signs, len(uniq) > 1

    sp_s, conflict_s = sign_pattern([per_system["strength_coord_y"][s] for s in range(4)])
    sp_d, conflict_d = sign_pattern([per_system["distance_coord_y"][s] for s in range(4)])
    sign_conflict = bool(conflict_s or conflict_d)

    # quadrant means: 2x2 split by coord_x / coord_y medians
    qx, qy = df["coord_x"].median(), df["coord_y"].median()
    df["qx"] = (df["coord_x"] >= qx).astype(int)
    df["qy"] = (df["coord_y"] >= qy).astype(int)
    df["q"] = df["qx"] * 2 + df["qy"]  # 0:(<, <), 1:(<, >=), 2:(>=, <), 3:(>=, >=)
    quadrant_means = {
        "strength": {int(q): float(df.loc[df["q"] == q, "node_strength"].mean()) for q in range(4)},
        "distance": {int(q): float(df.loc[df["q"] == q, "mean_connection_distance"].mean()) for q in range(4)},
    }

    result = {
        "overall": overall,
        "per_system": per_system,
        "sign_conflict": sign_conflict,
        "quadrant_means": quadrant_means,
        "n_nodes": n_nodes,
        "sign_pattern": {
            "strength_coord_y": sp_s,
            "distance_coord_y": sp_d,
        },
    }
    out_path = OUT_ROOT + "/f2_spatial_gradient_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    main()

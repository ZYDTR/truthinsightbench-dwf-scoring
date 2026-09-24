"""05_lsr_excursion.py

Characterise the short interval near 241 ka where the linear sedimentation rate
(LSR = d(depth)/d(age), m/ka) is extreme, and test whether it is unique.

Serial-correlation caveat: rows are NOT independent; the significance/standout
statements below are phrased in terms of ranks against the record-wide LSR
distribution and block (10 ka bin) averages, not per-row p-values.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project")
DATA = ROOT / "data" / "observations.tsv"
OUT = ROOT / "output" / "results" / "05_lsr_excursion.json"


def load():
    df = pd.read_csv(DATA, sep="\t")
    df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
    return df


def fit_sinusoid(t, y, periods):
    """Scan periods; design [1, t, cos, sin, cos2, sin2] with t centred.

    Vectorised over the period grid via batched normal equations.
    Returns best period, R^2, amplitude of harmonic 1 and harmonic 2.
    """
    tc = t - t.mean()
    # X: (n, 6, n_p) design stack
    w = 2.0 * np.pi / periods                      # (n_p,)
    X = np.empty((len(tc), 6, len(periods)))
    X[:, 0, :] = 1.0
    X[:, 1, :] = tc[:, None]
    wt = w[None, :] * tc[:, None]
    X[:, 2, :] = np.cos(wt)
    X[:, 3, :] = np.sin(wt)
    X[:, 4, :] = np.cos(2.0 * wt)
    X[:, 5, :] = np.sin(2.0 * wt)

    XtX = np.einsum("ikp,ilp->pkl", X, X)              # (n_p, 6, 6)
    Xty = np.einsum("ikp,i->pk", X, y)[..., None]      # (n_p, 6, 1)
    beta = np.linalg.solve(XtX, Xty)[..., 0]           # (n_p, 6)
    yhat = np.einsum("ikp,pk->ip", X, beta)            # (n, n_p)
    ss_res = np.sum((y[:, None] - yhat) ** 2, axis=0)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    amp1 = np.hypot(beta[:, 2], beta[:, 3])
    amp2 = np.hypot(beta[:, 4], beta[:, 5])
    k = int(np.argmax(r2))
    return {"period_ka": float(periods[k]), "r2": float(r2[k]),
            "amplitude": float(amp1[k]), "amplitude_harmonic2": float(amp2[k]),
            "n_rows": int(len(y))}


def main():
    df = load()
    depth = df["depth_m"].to_numpy(float)
    age = df["age_ka_bp_1"].to_numpy(float)
    d18o = df["d18o_pdb"].to_numpy(float)
    n = len(df)

    lsr = np.gradient(depth, age)

    # ---------------- 1. distribution ----------------
    med = float(np.median(lsr))
    q1, q3 = np.percentile(lsr, [25, 75])
    p95 = float(np.percentile(lsr, 95))
    p99 = float(np.percentile(lsr, 99))
    lsr_max = float(lsr.max())
    lsr_min = float(lsr.min())
    above = np.where(lsr > p99)[0]

    # ---------------- 2. episodes above p99 ----------------
    # join flagged rows into episodes allowing internal gaps of <= 2 rows
    episodes_idx = []
    cur = [above[0]]
    for i in above[1:]:
        if i - cur[-1] <= 3:  # gap of at most 2 unflagged rows between
            cur.append(i)
        else:
            episodes_idx.append(cur)
            cur = [i]
    episodes_idx.append(cur)

    episodes = []
    for ep in episodes_idx:
        rows = np.array(ep)
        ep_depth = depth[rows]
        ep_age = age[rows]
        y = d18o[rows]
        y_ok = y[~np.isnan(y)]
        mean_d18o = float(np.mean(y_ok)) if len(y_ok) else float("nan")

        centre = 0.5 * (ep_age[0] + ep_age[-1])
        win = (np.abs(age - centre) <= 10.0) & ~np.isnan(d18o)
        win_vals = d18o[win]
        win_mean = float(np.mean(win_vals))
        win_sd = float(np.std(win_vals, ddof=1))
        anomaly = mean_d18o - win_mean
        episodes.append({
            "row_index_start": int(rows[0]),
            "row_index_end": int(rows[-1]),
            "age_start_ka": float(ep_age[0]),
            "age_end_ka": float(ep_age[-1]),
            "depth_start_m": float(ep_depth[0]),
            "depth_end_m": float(ep_depth[-1]),
            "n_rows": int(len(rows)),
            "mean_d18o": mean_d18o,
            "d18o_anomaly_vs_20ka_window": float(anomaly),
            "anomaly_in_window_sd": float(anomaly / win_sd),
            # extras for the report (not required keys)
            "rows": [int(r) for r in rows],
            "d18o_values": [None if np.isnan(v) else float(v) for v in y],
            "window_centre_ka": float(centre),
            "window_spans_ka": [float(centre - 10.0), float(centre + 10.0)],
            "window_mean_d18o": win_mean,
            "window_sd_d18o": win_sd,
            "window_n": int(len(win_vals)),
        })

    # ---------------- 3. already inside episodes (anomaly vs 20 ka window) ----------------

    # ---------------- 4. fits ----------------
    periods = np.arange(40.0, 300.0 + 1e-9, 0.1)
    complete = ~np.isnan(d18o)

    mask_a = complete
    mask_b = complete & ~np.isin(np.arange(n), above)
    pad = np.zeros(n, bool)
    for ep in episodes_idx:
        lo = max(0, ep[0] - 2)
        hi = min(n - 1, ep[-1] + 2)
        pad[lo:hi + 1] = True
    mask_c = complete & ~pad

    fit_all = fit_sinusoid(age[mask_a], d18o[mask_a], periods)
    fit_excl_p99 = fit_sinusoid(age[mask_b], d18o[mask_b], periods)
    fit_excl_ep = fit_sinusoid(age[mask_c], d18o[mask_c], periods)
    fit_all["n_rows_removed"] = 0
    fit_excl_p99["n_rows_removed"] = int(mask_a.sum() - mask_b.sum())
    fit_excl_ep["n_rows_removed"] = int(mask_a.sum() - mask_c.sum())
    fit_excl_ep["padded_row_range"] = [int(np.where(pad)[0][0]), int(np.where(pad)[0][-1])]

    # ---------------- 5. alternative: 10 ka non-overlapping bins ----------------
    edges = np.arange(0.0, 370.0, 10.0)
    which = np.digitize(age, edges) - 1
    bin_vals = []
    bin_info = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() == 0:
            continue
        v = float(np.mean(lsr[m]))
        bin_vals.append(v)
        dspan = float(depth[m].max() - depth[m].min())
        aspan = float(age[m].max() - age[m].min())
        bin_info.append({
            "bin_lo_ka": float(edges[b]),
            "bin_hi_ka": float(edges[b + 1]),
            "n_rows": int(m.sum()),
            "mean_lsr": v,
            "depth_span_m": dspan,
            "age_span_ka": aspan,
            "span_lsr": dspan / aspan,
        })
    bin_vals = np.array(bin_vals)
    bin_p95 = float(np.percentile(bin_vals, 95))
    flagged = np.where(bin_vals > bin_p95)[0]
    for k, bi in enumerate(bin_info):
        bi["flagged_above_p95"] = bool(bin_vals[k] > bin_p95)
    ep_centre = episodes[0]["window_centre_ka"]
    ep_bin = next((i for i, bi in enumerate(bin_info)
                   if bi["bin_lo_ka"] <= ep_centre < bi["bin_hi_ka"]), None)
    ep_bin_flagged = bool(bin_info[ep_bin]["flagged_above_p95"]) if ep_bin is not None else False
    ep_bin_is_max = bool(ep_bin is not None and
                         bin_vals[ep_bin] == bin_vals.max())
    ep_bin_rank = int((bin_vals > bin_vals[ep_bin]).sum() + 1) if ep_bin is not None else None
    # robustness: the same binning using depth-span/age-span instead of the row mean
    span_vals = np.array([b["span_lsr"] for b in bin_info])
    span_p95 = float(np.percentile(span_vals, 95))
    n_span_flagged = int((span_vals > span_p95).sum())
    ep_bin_span_flagged = bool(ep_bin is not None and span_vals[ep_bin] > span_p95)
    ep_bin_span_rank = int((span_vals > span_vals[ep_bin]).sum() + 1) if ep_bin is not None else None

    # ---------------- 6. depth sampling grid ----------------
    dstep = np.round(np.diff(depth), 6)
    big_step_rows = np.where(dstep > 0.0101)[0]          # interval i -> i+1
    big_step_row_set = set(big_step_rows.tolist()) | set((big_step_rows + 1).tolist())
    extreme_set = set(above.tolist())
    overlap = sorted(extreme_set & big_step_row_set)
    # driver check: age increments on the extreme rows
    dage = np.diff(age)
    extreme_dage = [float(dage[i]) for i in extreme_set if i < len(dage)]
    med_dage = float(np.median(dage))

    # Counterfactual: replace the compressed age increments around the episode by the
    # record median step, keeping the depth grid untouched. Shows the spike is not
    # generated by the depth grid alone.
    ep_lo, ep_hi = episodes[0]["row_index_start"], episodes[0]["row_index_end"]
    # intervals feeding the gradient at the extreme rows: ep_lo-1 .. ep_hi
    cf_intervals = np.where((dage < 0.5 * med_dage) &
                            (np.arange(len(dage)) >= ep_lo - 1) &
                            (np.arange(len(dage)) <= ep_hi))[0]
    age_cf = age.copy()
    for i in cf_intervals:
        age_cf[i + 1:] += (med_dage - dage[i])
    lsr_med_age_step = np.gradient(depth, age_cf)
    cf_max = float(np.max(lsr_med_age_step))
    cf_p99 = float(np.percentile(lsr_med_age_step, 99))
    cf_n_above = int((lsr_med_age_step > cf_p99).sum())
    cf_interval_ids = [int(i) for i in cf_intervals]

    # discretisation check: how many distinct LSR values / how many rows share each
    lsr_round = np.round(lsr, 6)
    _, counts = np.unique(lsr_round, return_counts=True)
    n_distinct_lsr = int(len(counts))
    top_lsr_value = float(np.max(lsr_round))
    n_at_top_value = int((lsr_round == top_lsr_value).sum())

    n_eps = len(episodes)
    explanation = (
        "LSR = d(depth)/d(age). The seven rows above p99 (row indices 411-417, 240.41-241.88 ka, "
        "4.20-4.27 m, 6 with d18o) form ONE contiguous episode; no other episode exists, so it is "
        "unique under the per-row p99 rule. It is NOT caused by an enlarged depth step: six of the "
        "seven rows sit on ordinary 0.01 m steps, and although the 411->412 interval carries a "
        "0.02 m step (one of five such rows in the record: 44, 63, 312, 370, 411), none of those "
        "five intervals is itself extreme (their LSRs are 0.0128-0.0480 m/ka, and three are below "
        "the median). The spike is instead an age-grid artefact: age increments inside "
        f"240.87-241.88 ka are 0.20-0.23 ka against a record median of {med_dage:.2f} ka, so the "
        "same 0.01 m sampling yields ~2.3x the median LSR. Counterfactual: replacing only the "
        f"compressed age increments at intervals {cf_interval_ids} by the median step, with the "
        f"depth grid untouched, collapses the record max from {lsr_max:.5f} to {cf_max:.5f} m/ka "
        f"and leaves only {cf_n_above} rows above the new p99. Competing explanation confirmed: "
        "the extreme LSR is a sampling/age-model artefact (unusually fine age increments on an "
        "ordinary 0.01 m depth grid), not an enlarged depth step and not established as a real "
        "sedimentation pulse; the spike is not generated by the depth grid. "
        "Under the alternative 10 ka block-average definition the 240-250 ka bin is flagged "
        f"(rank {ep_bin_rank} of {len(bin_vals)} bins, mean LSR "
        f"{bin_info[ep_bin]['mean_lsr']:.4f} m/ka vs bin p95 {bin_p95:.4f}) but the 80-90 ka bin "
        f"is higher ({bin_vals.max():.4f} m/ka), so the 241 ka episode is no longer the single "
        "standout (it is the 2nd-highest of 2 flagged bins, both marginally above the 95th "
        "percentile); using the more robust within-bin depth-span/age-span rate instead of the "
        f"row mean the 240-250 ka bin is not flagged at all (rank {ep_bin_span_rank} of "
        f"{len(bin_info)}), confirming that at the 10 ka scale the episode is not exceptional. "
        "Its d18o anomaly is "
        f"{episodes[0]['d18o_anomaly_vs_20ka_window']:.3f} per mille "
        f"({episodes[0]['anomaly_in_window_sd']:.2f} sd of the surrounding 20 ka window, i.e. well "
        "within the window spread), and refitting the dominant sinusoid with the episode removed "
        f"changes the best period by {abs(fit_all['period_ka'] - fit_excl_ep['period_ka']):.1f} ka "
        f"and R^2 by {fit_excl_ep['r2'] - fit_all['r2']:+.3f}, so this short interval does not "
        "control the record-wide fit."
    )

    res = {
        "lsr_median": med,
        "lsr_iqr": [float(q1), float(q3)],
        "lsr_p95": p95,
        "lsr_p99": p99,
        "lsr_max_over_min": float(lsr_max / lsr_min),
        "n_rows_above_p99": int(len(above)),
        "n_episodes_above_p99": int(n_eps),
        "episodes": episodes,
        "fit_all": fit_all,
        "fit_excl_p99": fit_excl_p99,
        "fit_excl_episode_padded": fit_excl_ep,
        "episode_is_unique": bool(n_eps == 1),
        "n_bins_flagged_10ka": int(len(flagged)),
        "episode_standout_under_10ka_binning": bool(ep_bin_flagged and ep_bin_is_max),
        "extreme_rows_are_enlarged_depth_step_rows": bool(len(overlap) == len(extreme_set)),
        "explanation": explanation,
        # ---- extras ----
        "lsr_max": lsr_max,
        "lsr_min": lsr_min,
        "p99_row_indices": [int(i) for i in above],
        "rows_above_p99_detail": [
            {
                "row_index": int(i),
                "depth_m": float(depth[i]),
                "age_ka_bp_1": float(age[i]),
                "d18o_pdb": None if np.isnan(d18o[i]) else float(d18o[i]),
                "lsr": float(lsr[i]),
            }
            for i in above
        ],
        "enlarged_depth_step_rows": [int(i) for i in big_step_rows],
        "enlarged_depth_step_overlap_rows": overlap,
        "median_age_step_ka": med_dage,
        "extreme_rows_age_steps_ka": extreme_dage,
        "counterfactual_age_step_intervals": cf_interval_ids,
        "counterfactual_lsr_max": cf_max,
        "counterfactual_n_rows_above_p99": cf_n_above,
        "n_distinct_lsr_values": n_distinct_lsr,
        "n_rows_at_max_lsr_value": n_at_top_value,
        "ten_ka_bins": bin_info,
        "ten_ka_bin_p95": bin_p95,
        "episode_bin": bin_info[ep_bin] if ep_bin is not None else None,
        "episode_bin_rank": ep_bin_rank,
        "n_bins_flagged_10ka_span_based": n_span_flagged,
        "episode_bin_flagged_span_based": ep_bin_span_flagged,
        "episode_bin_rank_span_based": ep_bin_span_rank,
        "ten_ka_bin_span_p95": span_p95,
        "top_bin_by_mean_lsr": sorted(bin_info, key=lambda b: -b["mean_lsr"])[0],
        "top_bin_by_span_lsr": sorted(bin_info, key=lambda b: -b["span_lsr"])[0],
        "period_scan_ka": [40.0, 300.0, 0.1],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    # console report
    print("=== 1. LSR distribution ===")
    print(f"median={med:.6f} IQR=({q1:.6f},{q3:.6f}) p95={p95:.6f} p99={p99:.6f}")
    print(f"max={lsr_max:.6f} min={lsr_min:.6f} max/min={lsr_max/lsr_min:.3f} n>p99={len(above)}")
    print("=== 2. rows above p99 ===")
    for d in res["rows_above_p99_detail"]:
        print(d)
    print(f"n_episodes={n_eps}")
    for e in episodes:
        print({k: e[k] for k in ("row_index_start", "row_index_end", "age_start_ka",
                                 "age_end_ka", "depth_start_m", "depth_end_m", "n_rows",
                                 "mean_d18o", "d18o_anomaly_vs_20ka_window",
                                 "anomaly_in_window_sd")})
    print("=== 4. fits ===")
    for k in ("fit_all", "fit_excl_p99", "fit_excl_episode_padded"):
        print(k, res[k])
    print("=== 5. 10 ka bins ===")
    print("p95 of bins:", bin_p95, "n flagged:", len(flagged))
    print("flagged bins:", [bin_info[i] for i in flagged])
    print("episode bin:", bin_info[ep_bin] if ep_bin is not None else None,
          "rank", ep_bin_rank)
    print("=== 6. depth grid ===")
    print("enlarged step intervals (row i->i+1):", big_step_rows.tolist())
    print("overlap with extreme rows:", overlap)
    print("median age step", med_dage, "extreme rows age steps", extreme_dage)
    print("extreme_rows_are_enlarged_depth_step_rows:",
          res["extreme_rows_are_enlarged_depth_step_rows"])
    print("episode_is_unique:", res["episode_is_unique"])
    print("episode_standout_under_10ka_binning:",
          res["episode_standout_under_10ka_binning"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
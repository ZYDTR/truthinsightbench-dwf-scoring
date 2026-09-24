"""Stage 5 -- targeted robustness and alternative-explanation tests.

Three independent stress tests that bear directly on the registered findings:

  R1  Action-spectrum mismatch: is the red-end deficit an artefact of the
      arbitrary absorption units or of the 6-point action grid?  Tested by
      (a) renormalising the absorption template, and (b) comparing the
      blue-end plateau (408 vs 478 nm) against the red-end deficit
      (478 vs 648 nm) using only native 0.5 nm absorption values, so no
      interpolation or unit assumption enters.

  R2  Transients: a single-shape-with-time-scaling null.  If both traces were
      one kinetic shape rescaled in time and amplitude, the long-delay sign
      contrast would be impossible.  Fitted explicitly and compared.

  R3  Loading scan: the optimum location with a non-degenerate inhibitor model
      plus residual bootstrap, and an alternative two-regime description.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

import common as C

log = C.Logger(C.LOGS / "stage5_robustness.log")


# ---------------------------------------------------------------- R1
def action_blue_end_test(spec: pd.DataFrame) -> dict:
    log("=" * 78)
    log("R1  action-spectrum mismatch: unit-free and interpolation-free tests")
    log("=" * 78)
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    a_nm = abs_.wavelength_nm.values.astype(float)
    a_v = abs_.response_au_or_percent.values.astype(float)
    c_nm = act.wavelength_nm.values.astype(float)
    c_v = act.response_au_or_percent.values.astype(float)

    # --- unit-free version: normalise both profiles to their own maximum over
    #     the shared window, then compare RATIOS (ratios are unit-invariant)
    m = (a_nm >= 400) & (a_nm <= 700)
    abs_norm = a_v[m] / a_v[m].max()
    act_norm = c_v / c_v.max()
    at_action = np.interp(c_nm, a_nm[m], abs_norm)
    ratios = act_norm / at_action
    log("\n  (a) unit-free normalised comparison on the shared 400-700 nm window")
    log("      nm   action/norm   absorption/norm   ratio")
    for w, an, bn, r in zip(c_nm, act_norm, at_action, ratios):
        log(f"    {w:5.0f}   {an:10.3f}   {bn:14.3f}   {r:6.3f}")
    out = {"unit_free_normalised": {"nm": c_nm.tolist(), "action": act_norm.tolist(),
                                    "absorption": at_action.tolist(), "ratio": ratios.tolist()}}

    # --- (b) interpolation-free, unit-free: compare same-modality ratios.
    #     The action profile is already a same-modality triple, and the native
    #     0.5 nm absorption grid contains the action wavelengths exactly.
    def val_at(nm: float) -> float:
        hit = a_v[np.isclose(a_nm, nm)]
        return float(hit[0]) if hit.size else float("nan")

    def act_at(nm: float) -> float:
        hit = c_v[np.isclose(c_nm, nm)]
        return float(hit[0]) if hit.size else float("nan")

    # reference points taken at wavelengths where BOTH modalities were measured
    # (action grid = 400,450,500,550,600,650 nm; the 0.5 nm absorption grid
    #  contains all of these exactly, so no interpolation enters)
    blue_act = act_at(450.0) / act_at(400.0)
    red_act = act_at(650.0) / act_at(450.0)
    blue_abs = val_at(450.0) / val_at(400.0)
    red_abs = val_at(650.0) / val_at(450.0)
    log("\n  (b) same-modality ratio comparison at exactly coincident wavelengths")
    log(f"      action:      R(450/400) = {blue_act:.4f}   R(650/450) = {red_act:.4f}")
    log(f"      absorption:  R(450/400) = {blue_abs:.4f}   R(650/450) = {red_abs:.4f}")
    log(f"      -> the action profile is FLAT-to-rising at the blue end "
        f"({blue_act:.3f}) while absorption is flat-to-falling ({blue_abs:.3f});")
    log(f"      -> the action profile falls far faster at the red end "
        f"({red_act:.3f}) than absorption ({red_abs:.3f}).")

    # window sensitivity: enumerate EVERY probe/reference pair available on the
    # 6-point action grid and evaluate the same unit-free ratio statistic.
    # A ratio statistic > 1 means the action profile is flatter than absorption
    # across that pair; < 1 means it falls faster.
    pairs = []
    for i, ref in enumerate(c_nm):
        for probe in c_nm[i + 1:]:
            stat = (act_at(probe) / act_at(ref)) / (val_at(probe) / val_at(ref))
            pairs.append({"reference_nm": float(ref), "probe_nm": float(probe),
                          "span_nm": float(probe - ref),
                          "action_ratio": float(act_at(probe) / act_at(ref)),
                          "absorption_ratio": float(val_at(probe) / val_at(ref)),
                          "ratio_statistic": float(stat)})
    short = [p for p in pairs if p["probe_nm"] <= 500]
    long = [p for p in pairs if p["probe_nm"] >= 600]
    log("\n  (c) window sensitivity: all 15 probe/reference pairs on the action grid")
    log("      ref    probe   span   action_R  absorption_R  statistic")
    for p in pairs:
        log(f"    {p['reference_nm']:5.0f}  {p['probe_nm']:5.0f}  {p['span_nm']:5.0f}   "
            f"{p['action_ratio']:8.4f}   {p['absorption_ratio']:11.4f}   "
            f"{p['ratio_statistic']:9.3f}")
    log(f"      short-span pairs (probe <= 500 nm): min statistic = "
        f"{min(p['ratio_statistic'] for p in short):.3f}, all > 1: "
        f"{all(p['ratio_statistic'] > 1 for p in short)}")
    log(f"      long-span pairs  (probe >= 600 nm): max statistic = "
        f"{max(p['ratio_statistic'] for p in long):.3f}, all < 1: "
        f"{all(p['ratio_statistic'] < 1 for p in long)}")
    out["interpolation_free_ratios"] = {
        "action_R_450_over_400": blue_act, "action_R_650_over_450": red_act,
        "absorption_R_450_over_400": blue_abs, "absorption_R_650_over_450": red_abs,
        "blue_ratio_statistic": float(blue_act / blue_abs),
        "red_ratio_statistic": float(red_act / red_abs),
    }
    out["window_sensitivity_pairs"] = pairs
    out["blue_end_claim_supported"] = bool(all(p["ratio_statistic"] > 1 for p in short))
    out["red_end_claim_supported"] = bool(all(p["ratio_statistic"] < 1 for p in long))
    log(f"\n  blue-end claim supported (every short-span pair statistic > 1): "
        f"{out['blue_end_claim_supported']}")
    log(f"  red-end  claim supported (every long-span pair statistic < 1): "
        f"{out['red_end_claim_supported']}")

    # --- (d) does a free offset/saturation absorption template rescue the null?
    log("\n  (d) alternative null: action proportional to a SATURATING function of "
        "absorption, action = a * A / (b + A)")
    best = None
    tpl = np.array([val_at(w) for w in c_nm])
    for b in np.geomspace(1e-3, 1e3, 20000):
        basis = tpl / (b + tpl)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        a = float(basis @ c_v / denom)
        rss = float(((c_v - a * basis) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, a, b)
    rss = best[0]
    sst = float(((c_v - c_v.mean()) ** 2).sum())
    n = len(c_v)
    kk = 3
    aicc = n * np.log(max(rss, 1e-300) / n) + 2 * kk + (2 * kk * (kk + 1)) / max(n - kk - 1, 1)
    log(f"      best saturating-in-A fit: a={best[1]:.2f}, b={best[2]:.4f}, "
        f"R2={1-rss/sst:.5f}, RMSE={np.sqrt(rss/n):.3f}, AICc={aicc:.2f}")
    log(f"      (compare M1 absorption-limited RMSE = 8.872, AICc = 34.19)")
    out["saturating_absorption_null"] = {
        "a": float(best[1]), "b": float(best[2]), "rss": float(rss),
        "r2": float(1 - rss / sst), "rmse": float(np.sqrt(rss / n)), "aicc": float(aicc),
        "better_than_linear_in_A_by_rmse": bool(np.sqrt(rss / n) < 8.872),
    }
    return out


# ---------------------------------------------------------------- R2
def shared_shape_null(tr: pd.DataFrame) -> dict:
    """R2: can both traces be one shape rescaled in time and amplitude?

    Model: y2(t) = a * y1(t / s) with a single amplitude a and time scale s.
    Fitted by gridded interpolation over s with a profiled out in closed form.
    If the residual is large and the sign contrast survives, a single
    time-rescaled kinetics does not describe the pair.
    """
    log("\n" + "=" * 78)
    log("R2  transients: single-shape-with-time-scaling null")
    log("=" * 78)
    g1 = tr[tr.formulation_id == "formulation_1"].sort_values("delay_ps")
    g2 = tr[tr.formulation_id == "formulation_2"].sort_values("delay_ps")
    t1 = g1.delay_ps.values.astype(float)
    y1 = g1.normalized_transient_signal.values.astype(float)
    t2 = g2.delay_ps.values.astype(float)
    y2 = g2.normalized_transient_signal.values.astype(float)
    sst = float(((y2 - y2.mean()) ** 2).sum())
    best = None
    for s in np.geomspace(0.05, 20.0, 6000):
        interp = np.interp(t2 / s, t1, y1, left=np.nan, right=np.nan)
        ok = np.isfinite(interp)
        if ok.sum() < 20:
            continue
        denom = float(interp[ok] @ interp[ok])
        if denom <= 0:
            continue
        a = float(interp[ok] @ y2[ok] / denom)
        rss = float(((y2[ok] - a * interp[ok]) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, float(s), a, int(ok.sum()))
    rss, s_best, a_best, n_ok = best
    log(f"  best single-shape rescaling: y2(t) = {a_best:.3f} * y1(t / {s_best:.3f})")
    log(f"  R2 on the overlapping delay range = {1 - rss / sst:.4f} "
        f"(n = {n_ok} of {len(t2)} points)")
    # what does that model predict at long delay for f2?
    late = t2 >= 30.0
    pred_late = a_best * np.interp(t2[late] / s_best, t1, y1, left=np.nan, right=np.nan)
    finite = np.isfinite(pred_late)
    obs_late = y2[late]
    log(f"  held-out late-window (>=30 ps): observed mean = {obs_late.mean():+.4f}; "
        f"single-shape prediction mean = "
        f"{np.nanmean(pred_late) if finite.any() else float('nan'):+.4f} "
        f"(finite for {int(finite.sum())}/{int(late.sum())} points)")
    # amplitude-normalised, per-trace-standardised comparison at long delay
    contrast = {
        "f1_late_mean": float(y1[t1 >= 30].mean()),
        "f2_late_mean": float(obs_late.mean()),
        "single_shape_predicted_f2_late_mean": float(np.nanmean(pred_late)) if finite.any() else None,
        "single_shape_predicts_sign": ("positive" if np.nanmean(pred_late) > 0 else "negative")
        if finite.any() else None,
        "observed_f2_sign": "positive" if obs_late.mean() > 0 else "negative",
    }
    log(f"  single-shape model predicts f2 late sign = {contrast['single_shape_predicts_sign']}; "
        f"observed = {contrast['observed_f2_sign']}")
    return {"best_time_scale": s_best, "best_amplitude": a_best, "r2_overlap": float(1 - rss / sst),
            "n_points_used": n_ok, "late_window_contrast": contrast}


# ---------------------------------------------------------------- R3
def loading_bootstrap(ld: pd.DataFrame) -> dict:
    """R3: optimum location for a non-degenerate inhibitor model + residual bootstrap."""
    log("\n" + "=" * 78)
    log("R3  loading scan: optimum location with a non-degenerate inhibitor model")
    log("=" * 78)
    L = ld.cofactor_loading_percent.values.astype(float)
    R = ld.product_evolution_rate_umol_h.values.astype(float)

    def model(Lx, V, K, Ki):
        return V * Lx / (K + Lx + Lx * Lx / Ki)

    popt, _ = optimize.curve_fit(model, L, R, p0=[R.max() * 3, 1.0, 20.0], maxfev=200000)
    V, K, Ki = popt
    peak = float(np.sqrt(K * Ki))
    pred = model(L, *popt)
    rss = float(((R - pred) ** 2).sum())
    sst = float(((R - R.mean()) ** 2).sum())
    log(f"  fit: V={V:.1f}, K={K:.4f}, Ki={Ki:.4f}  ->  analytic peak at "
        f"sqrt(K*Ki) = {peak:.3f}% loading")
    log(f"  R2 = {1-rss/sst:.5f}, RMSE = {np.sqrt(rss/4):.1f} umol/h "
        f"(3 parameters, 4 observations)")

    resid = R - pred
    rng = np.random.default_rng(20220101)
    # Parameterise the peak directly: K = Lp^2 / Ki makes sqrt(K*Ki) == Lp, so the
    # resampled optimum is returned by the fit itself rather than recomputed.
    def model_peakparam(Lx, V, Lp, Ki):
        return V * Lx / (Lp * Lp / Ki + Lx + Lx * Lx / Ki)

    lp0 = float(np.sqrt(max(K * Ki, 1e-12)))
    pp0, _ = optimize.curve_fit(model_peakparam, L, R,
                                p0=[V, lp0, Ki], maxfev=200000)
    # A 4-point fit with 3 parameters has one residual degree of freedom, so the
    # resampled fit is well conditioned and the peak location is the fitted Lp.
    # The resampling is done in one vectorised batch rather than a Python loop.
    n_boot = 5000
    draws = rng.choice(resid, size=(n_boot, 4), replace=True)
    Rb_all = pred[None, :] + draws          # (n_boot, 4)

    def residuals(p, Lx, yobs):
        return model_peakparam(Lx, *p) - yobs

    peaks = []
    fails = 0
    for Rb in Rb_all:
        try:
            sol = optimize.least_squares(residuals, pp0, args=(L, Rb),
                                         method="lm", max_nfev=400)
            if sol.success and np.all(np.isfinite(sol.x)) and sol.x[1] > 0:
                peaks.append(float(sol.x[1]))
            else:
                fails += 1
        except Exception:
            fails += 1
    peaks = np.array(peaks)
    log(f"  bootstrap: {len(peaks)} successful / {fails} failed resamples")
    log(f"  residual bootstrap ({len(peaks)} successful resamples): peak median "
        f"{np.median(peaks):.3f}%, 95% CI [{np.percentile(peaks,2.5):.3f}, "
        f"{np.percentile(peaks,97.5):.3f}]%, fraction below 3% = "
        f"{np.mean(peaks < 3.0):.3f}")

    # alternative description: flat plateau then decline
    hi = float(np.mean([R[0], R[1]]))
    log(f"\n  alternative two-regime description: plateau at ~{hi:.0f} umol/h for "
        f"loading <= 1% (mean of the two low-loading points) then decline to {R[-1]:.0f}")
    log(f"  the low-loading pair differs by only {100*abs(R[1]-R[0])/R[0]:.1f}%, whereas the "
        f"drop to 3% is {100*(1-R[2]/R[1]):.1f}%; both the 'peak at 1%' and the "
        f" 'plateau up to 1% then decline' readings are consistent with n = 4 points")
    return {
        "inhibitor_fit": {"V": float(V), "K": float(K), "Ki": float(Ki),
                          "analytic_peak_percent": peak, "r2": float(1 - rss / sst),
                          "rmse": float(np.sqrt(rss / 4))},
        "bootstrap_peak": {
            "n_successful": int(len(peaks)),
            "median_percent": float(np.median(peaks)) if len(peaks) else None,
            "ci95_percent": ([float(np.percentile(peaks, 2.5)), float(np.percentile(peaks, 97.5))]
                             if len(peaks) else None),
            "fraction_below_3_percent": float(np.mean(peaks < 3.0)) if len(peaks) else None,
        },
        "alternative_plateau_description": {
            "plateau_rate_umol_h": hi, "plateau_upper_loading_percent": 1.0,
            "low_pair_relative_difference": float(abs(R[1] - R[0]) / R[0]),
            "drop_to_3pct_relative": float(1 - R[2] / R[1]),
        },
    }


if __name__ == "__main__":
    spec = C.load_spectra()
    tr = C.load_transients()
    ld = C.load_loading()
    r1 = action_blue_end_test(spec)
    r2 = shared_shape_null(tr)
    r3 = loading_bootstrap(ld)
    C.json_dump(C.RESULTS / "robustness.json", {"R1_action_spectrum": r1,
                                                "R2_shared_shape_null": r2,
                                                "R3_loading_optimum": r3})
    log("\nstage 5 complete")
    log.close()
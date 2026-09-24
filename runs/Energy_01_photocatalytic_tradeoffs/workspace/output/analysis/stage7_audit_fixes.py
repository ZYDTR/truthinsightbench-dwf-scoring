"""Stage 7 -- persist two checks that were previously reported without a stored
artifact, plus the normalisation-convention variants for the shared-shape test.

These were added in response to an independent audit of the registered findings.
Nothing here changes a conclusion; it makes two already-true statements traceable
to an artifact and removes an ambiguity between two normalisation conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

log = C.Logger(C.LOGS / "stage7_audit_fixes.log")


def normalised_shape_distances(tc: pd.DataFrame) -> dict:
    """Normalised RMS distance between every pair of product-evolution curves.

    Two conventions are stored explicitly because the audit flagged that the
    report quoted values from one convention while the previously stored artifact
    used another:
      - exclude_t0: normalise each curve by its own 4 h value, drop the shared
        (0, 0) origin
      - include_t0: same normalisation, keep the origin
    Both are reported for all six pairs so neither is ambiguous.
    """
    log("=" * 78)
    log("STAGE 7  normalised shape distances under both conventions")
    log("=" * 78)
    piv = tc.pivot(index="illumination_time_h", columns="formulation_id",
                   values="gaseous_product_umol")
    t = piv.index.values.astype(float)
    norm = piv.div(piv.iloc[-1], axis=1)
    ids = list(piv.columns)
    out = {"pairs": {}}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = norm[ids[i]].values
            b = norm[ids[j]].values
            ex = slice(1, None)  # drop the shared origin
            out["pairs"][f"{ids[i]}_vs_{ids[j]}"] = {
                "rms_exclude_t0": float(np.sqrt(((a[ex] - b[ex]) ** 2).mean())),
                "max_abs_exclude_t0": float(np.abs(a[ex] - b[ex]).max()),
                "rms_include_t0": float(np.sqrt(((a - b) ** 2).mean())),
                "max_abs_include_t0": float(np.abs(a - b).max()),
            }
            d = out["pairs"][f"{ids[i]}_vs_{ids[j]}"]
            log(f"  {ids[i]} vs {ids[j]}: rms_excl_t0={d['rms_exclude_t0']:.5f} "
                f"rms_incl_t0={d['rms_include_t0']:.5f}")
    # the two comparisons the report quotes
    out["quoted_3_vs_4_rms_exclude_t0"] = out["pairs"]["formulation_3_vs_formulation_4"]["rms_exclude_t0"]
    out["quoted_1_vs_3_rms_exclude_t0"] = out["pairs"]["formulation_1_vs_formulation_3"]["rms_exclude_t0"]
    out["closest_pair"] = min(out["pairs"], key=lambda k: out["pairs"][k]["rms_exclude_t0"])
    log(f"\n  quoted (exclude t0): 3 vs 4 = {out['quoted_3_vs_4_rms_exclude_t0']:.5f}, "
        f"1 vs 3 = {out['quoted_1_vs_3_rms_exclude_t0']:.5f}")
    log(f"  closest pair overall: {out['closest_pair']}")
    return out


def transmittance_bound_grid() -> dict:
    """Uniform-in-k check of the transmittance-conversion bound on the action ratio.

    Claim under test: for absorptance alpha(x) = 1 - 10^(-x) (concave), any scale
    k > 0 maps a pair of absorbances A_f > A_g to a ratio no smaller than A_g/A_f,
    i.e. a monotone transmittance conversion can only RAISE the predicted
    short->long action ratio, never lower it.  Since the absorption 450->650 ratio
    is 0.2334 and the observed action ratio is 0.0508, the observed value lies
    below the achievable floor and no such conversion can rescue the
    absorption-limited null.

    Swept over a wide k grid and several absorption pairs, including the actual
    measured pair and neighbouring wavelengths.
    """
    log("\n" + "-" * 78)
    log("Transmittance-conversion bound, swept over k and several absorption pairs")
    log("-" * 78)
    spec = C.load_spectra()
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    a_nm = abs_.wavelength_nm.values.astype(float)
    a_v = abs_.response_au_or_percent.values.astype(float)

    def val_at(nm: float) -> float:
        hit = a_v[np.isclose(a_nm, nm)]
        return float(hit[0]) if hit.size else float("nan")

    def alpha(x):
        return 1.0 - np.power(10.0, -x)

    pairs = [(450.0, 650.0), (400.0, 650.0), (450.0, 600.0), (500.0, 650.0), (550.0, 650.0)]
    k_grid = np.geomspace(1e-6, 1e2, 1200)
    rows = []
    violations = 0
    total = 0
    for short_nm, long_nm in pairs:
        A_short, A_long = val_at(short_nm), val_at(long_nm)
        raw = A_long / A_short
        best = None
        for k in k_grid:
            r = alpha(k * A_long) / alpha(k * A_short)
            total += 1
            if r < raw - 1e-12:
                violations += 1
            if best is None or r < best[0]:
                best = (r, k)
        rows.append({
            "short_nm": short_nm, "long_nm": long_nm,
            "A_short": A_short, "A_long": A_long,
            "raw_absorption_ratio": float(raw),
            "min_predicted_ratio_over_k_grid": float(best[0]),
            "k_at_minimum": float(best[1]),
            "bound_holds": bool(best[0] >= raw - 1e-12),
        })
        log(f"  {short_nm:.0f}->{long_nm:.0f} nm: A={A_short:.3f}->{A_long:.3f}  "
            f"raw ratio={raw:.4f}  min over k={best[0]:.4f} at k={best[1]:.2e}  "
            f"bound holds={rows[-1]['bound_holds']}")
    log(f"\n  grid size = {len(k_grid)} k values x {len(pairs)} pairs = {total} evaluations; "
        f"violations of the stated inequality = {violations}")
    log("  => a monotone transmittance conversion can only raise the predicted "
        "450->650 action ratio above the absorption ratio 0.2334; the observed 0.0508 "
        "is below that floor, so no k rescues the absorption-limited null.")
    return {"k_grid_points": int(len(k_grid)), "k_min": float(k_grid.min()),
            "k_max": float(k_grid.max()), "pairs_tested": len(pairs),
            "total_evaluations": int(total), "inequality_violations": int(violations),
            "per_pair": rows,
            "observed_action_ratio_450_650": 0.0508,
            "verdict": "bound holds uniformly; transmittance conversion cannot rescue the null"}


def loading_leaveout_restatement(ld: pd.DataFrame) -> dict:
    """Explicitly tabulate which leave-one-out subsets are monotone.

    The audit found the report's one-line summary of this was inverted.  This
    records the mapping directly so the wording can be checked against a file.
    """
    log("\n" + "-" * 78)
    log("Loading leave-one-out: which subsets are monotone decreasing")
    log("-" * 78)
    L = ld.cofactor_loading_percent.values.astype(float)
    R = ld.product_evolution_rate_umol_h.values.astype(float)
    out = {"cases": {}, "which_point_removed": {}}
    for i in range(len(L)):
        keep = np.ones(len(L), dtype=bool)
        keep[i] = False
        mono = bool(np.all(np.diff(R[keep]) < 0))
        is_low = L[i] <= 1.0
        out["cases"][f"drop_{L[i]:g}pct"] = {
            "removed_loading_percent": float(L[i]),
            "removed_is_low_loading": bool(is_low),
            "remaining_loading": L[keep].tolist(),
            "remaining_rate": R[keep].tolist(),
            "monotone_decreasing": mono,
        }
        out["which_point_removed"][f"drop_{L[i]:g}pct"] = "low" if is_low else "high"
        log(f"  drop {L[i]:5.2f}% ({'low' if is_low else 'high'} point): "
            f"remaining {R[keep].tolist()} -> monotone_decreasing={mono}")
    lows = [k for k, v in out["cases"].items() if v["removed_is_low_loading"]]
    highs = [k for k, v in out["cases"].items() if not v["removed_is_low_loading"]]
    out["summary"] = {
        "monotone_when_a_low_point_is_dropped": all(out["cases"][k]["monotone_decreasing"] for k in lows),
        "monotone_when_a_high_point_is_dropped": all(out["cases"][k]["monotone_decreasing"] for k in highs),
        "correct_statement": ("the strictly decreasing triple appears only when a LOW-loading "
                              "point (0.2 or 1.0 percent) is removed, i.e. when both high-loading "
                              "points are retained plus exactly one low point; removing either "
                              "high-loading point leaves the non-monotone rise-then-fall pattern"),
    }
    log(f"\n  {out['summary']['correct_statement']}")
    return out


if __name__ == "__main__":
    tc = C.load_timecourses()
    ld = C.load_loading()
    payload = {
        "normalised_shape_distances": normalised_shape_distances(tc),
        "transmittance_bound_grid": transmittance_bound_grid(),
        "loading_leaveout_restatement": loading_leaveout_restatement(ld),
    }
    C.json_dump(C.RESULTS / "audit_fixes.json", payload)
    log("\nwrote results/audit_fixes.json")
    log("\nstage 7 complete")
    log.close()
#!/usr/bin/env python3
"""
V1 -- Falsification of CLAIM D1.

Claim D1: In observation_set_1 (10 x 13 x 11 complete factorial, 1 rep/cell),
  numeric_feature_6 (y6) variance splits ~ main .521 / 2way .325 / 3way .154
  numeric_feature_7 (y7) variance splits ~ main .887 / 2way .076 / 3way .037
  y7 3-way "small but nonzero" (Tukey F ~ 9.9, p ~ .0017);
  y6 3-way "large" (F ~ 764).

TWO INDEPENDENT COMPUTATION PATHS:
  (P1) closed-form balanced-factorial sums of squares via cell means (the
       textbook orthogonal decomposition for a complete single-replicate
       factorial) -- exact, no matrix inversion, no library contrasts.
  (P2) explicit sum-to-zero effect coding built by hand + QR least squares on
       nested effect subspaces -- an independent numerical route.
Then: adversarial robustness (standardise / winsorise / rank transform),
distribution diagnostics, and an honest error-stratum discussion.

Everything is re-derived from the raw TSV; no number is taken from the claim.
"""
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy import stats

t0 = time.time()
BASE = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project"
OUT = BASE + "/output/verification"

d = pd.read_csv(BASE + "/data/observation_set_1.tsv", sep="\t")
FA, FB, FC = "numeric_feature_2", "numeric_feature_3", "numeric_feature_4"
FACTORS = [FA, FB, FC]
RESPONSES = ["numeric_feature_6", "numeric_feature_7"]
rec = {}

rec["file"] = BASE + "/data/observation_set_1.tsv"
with open(rec["file"], "rb") as fh:
    rec["md5"] = hashlib.md5(fh.read()).hexdigest()
rec["shape"] = list(d.shape)
rec["dtypes"] = {c: str(t) for c, t in d.dtypes.items()}
rec["n_missing"] = int(d.isna().sum().sum())

# ============================================================ 1. DESIGN CHECK
nA, nB, nC = (int(d[c].nunique()) for c in FACTORS)
N = nA * nB * nC
cell_counts = d.groupby(FACTORS, observed=True).size()
mi = ((d[FA].values * nB) + d[FB].values) * nC + d[FC].values
rec["design"] = {
    "n_levels": {FA: nA, FB: nB, FC: nC},
    "n_expected_cells": N,
    "n_rows": int(len(d)),
    "n_distinct_cells": int(len(cell_counts)),
    "reps_per_cell_min": int(cell_counts.min()),
    "reps_per_cell_max": int(cell_counts.max()),
    "complete_factorial_all_cells_exactly_once": bool(
        len(cell_counts) == N and (cell_counts == 1).all() and len(d) == N),
    "row_order_is_mixed_radix_full_factorial": bool((mi == np.arange(len(d))).all()),
    "factor_levels_exactly_0_to_k_minus_1": {
        FA: bool(np.array_equal(np.sort(d[FA].unique()), np.arange(nA))),
        FB: bool(np.array_equal(np.sort(d[FB].unique()), np.arange(nB))),
        FC: bool(np.array_equal(np.sort(d[FC].unique()), np.arange(nC))),
    },
    "response_columns_are_continuous_dup_counts": {
        r: {"n_unique": int(d[r].nunique()), "n_rows": int(len(d)),
            "is_continuous": bool(abs(float(d[r].std(ddof=1))) > 0 and
                                  d[r].nunique() == len(d))}
        for r in RESPONSES},
}
# corroborating structure: the other columns
rec["design"]["other_column_nunique"] = {c: int(d[c].nunique()) for c in d.columns}

# =============================== 2a. PATH P1: closed-form balanced-factorial SS
A = d[FA].values.astype(int); B = d[FB].values.astype(int); C = d[FC].values.astype(int)


def closed_form_ss(y):
    """Exact orthogonal (Yates) decomposition for a balanced single-replicate
    a x b x c factorial.  All terms are orthogonal, so each SS is simply
    n_cells_per_level_sum * sum of squared effect means."""
    y = np.asarray(y, float)
    grand = y.mean()
    total = float(((y - grand) ** 2).sum())
    # full-crossed cell means == raw values (1 rep/cell)
    def cellmean(idx):
        out = {}
        for k in np.unique(idx):
            out[k] = y[idx == k].mean()
        return out
    mA = np.array([y[A == i].mean() for i in range(nA)])
    mB = np.array([y[B == j].mean() for j in range(nB)])
    mC = np.array([y[C == k].mean() for k in range(nC)])
    mAB = np.array([[y[(A == i) & (B == j)].mean() for j in range(nB)] for i in range(nA)])
    mAC = np.array([[y[(A == i) & (C == k)].mean() for k in range(nC)] for i in range(nA)])
    mBC = np.array([[y[(B == j) & (C == k)].mean() for k in range(nC)] for j in range(nB)])
    mABC = np.array([[[y[(A == i) & (B == j) & (C == k)][0] for k in range(nC)]
                      for j in range(nB)] for i in range(nA)])

    ss_A = nB * nC * float(((mA - grand) ** 2).sum())
    ss_B = nA * nC * float(((mB - grand) ** 2).sum())
    ss_C = nA * nB * float(((mC - grand) ** 2).sum())
    ss_AB = nC * float(((mAB - mA[:, None] - mB[None, :] + grand) ** 2).sum())
    ss_AC = nB * float(((mAC - mA[:, None] - mC[None, :] + grand) ** 2).sum())
    ss_BC = nA * float(((mBC - mB[:, None] - mC[None, :] + grand) ** 2).sum())
    ss_ABC = float(((mABC - mAB[:, :, None] - mAC[:, None, :] - mBC[None, :, :]
                     + mA[:, None, None] + mB[None, :, None] + mC[None, None, :]
                     - grand) ** 2).sum())
    main = ss_A + ss_B + ss_C
    two = ss_AB + ss_AC + ss_BC
    three = ss_ABC
    resid = total - main - two - three
    return {"SS_main": main, "SS_two_way": two, "SS_three_way": three,
            "SS_residual": resid, "SS_total": total,
            "SS_components": {"A": ss_A, "B": ss_B, "C": ss_C,
                              "AB": ss_AB, "AC": ss_AC, "BC": ss_BC, "ABC": ss_ABC},
            "df": {"A": nA - 1, "B": nB - 1, "C": nC - 1,
                   "AB": (nA - 1) * (nB - 1), "AC": (nA - 1) * (nC - 1),
                   "BC": (nB - 1) * (nC - 1), "ABC": (nA - 1) * (nB - 1) * (nC - 1),
                   "residual": (N - 1) - ((nA - 1) + (nB - 1) + (nC - 1)
                                          + (nA - 1) * (nB - 1) + (nA - 1) * (nC - 1)
                                          + (nB - 1) * (nC - 1) + (nA - 1) * (nB - 1) * (nC - 1))}}


def eta_from_ss(ss, total):
    return {"main": ss["SS_main"] / total, "two_way": ss["SS_two_way"] / total,
            "three_way": ss["SS_three_way"] / total,
            "residual": ss["SS_residual"] / total}


P1 = {}
for r in RESPONSES:
    ss = closed_form_ss(d[r].values)
    P1[r] = {"SS": ss, "eta2": eta_from_ss(ss, ss["SS_total"])}
rec["path_P1_closed_form_balanced_SS"] = P1

# ==================================== 2b. PATH P2: hand-built effect-coding + QR
def effect_codes(k):
    M = np.zeros((k, k - 1)); M[: k - 1, :] = np.eye(k - 1); M[k - 1, :] = -1.0
    return M


MA, MB, MC = effect_codes(nA), effect_codes(nB), effect_codes(nC)
blk = {}
blk["int"] = np.ones((N, 1))
blk["A"] = MA[A]; blk["B"] = MB[B]; blk["C"] = MC[C]
blk["AB"] = (MA[A][:, :, None] * MB[B][:, None, :]).reshape(N, -1)
blk["AC"] = (MA[A][:, :, None] * MC[C][:, None, :]).reshape(N, -1)
blk["BC"] = (MB[B][:, :, None] * MC[C][:, None, :]).reshape(N, -1)
blk["ABC"] = (MA[A][:, :, None, None] * MB[B][:, None, :, None]
              * MC[C][:, None, None, :]).reshape(N, -1)
Xfull = np.hstack([blk[k] for k in ["int", "A", "B", "C", "AB", "AC", "BC", "ABC"]])


def qr_ss(y):
    """PATH P2 -- independent of the textbook cell-mean formulas in P1.
    Builds the explicit sum-to-zero effect-coding design and obtains each
    effect SS from a normal-equations projection (b = (X'X)^-1 X'y,
    SS = b'X'y) restricted to each effect block.  Valid here because the
    effect-coding blocks are mutually orthogonal by construction (verified
    numerically and reported), so X'X is block-diagonal."""
    yv = np.asarray(y, float)
    Ytot = float(((yv - yv.mean()) ** 2).sum())
    out, Gblk = {}, {}
    for k in ["A", "B", "C", "AB", "AC", "BC", "ABC"]:
        Xk = blk[k]
        Gk = Xk.T @ Xk
        ck = Xk.T @ yv
        bk = np.linalg.solve(Gk, ck)
        out[k] = float(bk @ ck)
        Gblk[k] = Gk
    total = Ytot
    main = out["A"] + out["B"] + out["C"]
    two = out["AB"] + out["AC"] + out["BC"]
    three = out["ABC"]
    return {"SS_main": main, "SS_two_way": two, "SS_three_way": three,
            "SS_residual": total - main - two - three, "SS_total": total,
            "SS_components": out,
            "block_singular_max_abs": float(max(
                np.abs(Gblk[k] - np.diag(np.diag(Gblk[k]))).max() / np.abs(Gblk[k]).max()
                for k in Gblk))}


P2 = {}
for r in RESPONSES:
    ss = qr_ss(d[r].values)
    P2[r] = {"SS": ss, "eta2": eta_from_ss(ss, ss["SS_total"])}
rec["path_P2_effect_coding_QR"] = P2

# record the explicit cross-block orthogonality (validates that P1==P2)
_Gblocks = {k: (blk[k].T @ blk[k]) for k in ["A", "B", "C", "AB", "AC", "BC", "ABC"]}
_crossmax = 0.0
_keys = list(_Gblocks)
for _i, _k1 in enumerate(_keys):
    for _k2 in _keys[_i + 1:]:
        _crossmax = max(_crossmax, float(np.abs(blk[_k1].T @ blk[_k2]).max()))
rec["effect_coding_design_orthogonal_max_abs_offdiag"] = _crossmax
rec["effect_coding_block_gram_offdiag_relative"] = float(max(
    np.abs(_Gblocks[k] - np.diag(np.diag(_Gblocks[k]))).max() / np.abs(_Gblocks[k]).max()
    for k in _Gblocks))

# agreement between the two paths
rec["path_agreement_max_abs_eta2_diff"] = float(max(
    max(abs(P1[r]["eta2"][k] - P2[r]["eta2"][k]) for k in P1[r]["eta2"])
    for r in RESPONSES))

# ============================================ 3. ADVERSARIAL ROBUSTNESS TESTS
def decompose_full(y):
    return closed_form_ss(y)


def _project(Xm, y):
    """OLS projection of y onto col(Xm) via normal equations (small, well
    conditioned designs only)."""
    G = Xm.T @ Xm
    c = Xm.T @ y
    b = np.linalg.solve(G, c)
    return Xm @ b


def tukey_1df_nonadditivity(y):
    """Tukey's 1-df test for non-additivity of the ADDITIVE (main-effects)
    model, using yhat^2 (orthogonalised to the additive space) as the single
    extra regressor."""
    y = np.asarray(y, float)
    Xa = np.hstack([blk[c] for c in ["int", "A", "B", "C"]])
    fit = _project(Xa, y)
    resid = y - fit
    z = fit ** 2
    zc = z - z.mean()
    zc = zc - _project(Xa, zc)
    bz = float((zc @ resid) / (zc @ zc))
    ss_nad = float((bz ** 2) * (zc @ zc))
    ss_err = float(((resid - bz * zc) ** 2).sum())
    df_err = len(y) - Xa.shape[1] - 1
    F = (ss_nad / 1) / (ss_err / df_err)
    return {"SS_nonadditivity": ss_nad, "SS_error": ss_err, "df_error": int(df_err),
            "F": float(F), "p": float(stats.f.sf(F, 1, df_err))}


def tukey_within_3way(y):
    """Split the (a-1)(b-1)(c-1)-df 3-way stratum into Tukey's 1-df
    nonadditivity component (predicted by yhat^2 from the additive+2way model)
    and the orthogonal remainder; F-test 1-df component vs remainder."""
    y = np.asarray(y, float)
    X2 = np.hstack([blk[c] for c in ["int", "A", "B", "C", "AB", "AC", "BC"]])
    fit = _project(X2, y)
    resid = y - fit                          # pure 3-way, df=(a-1)(b-1)(c-1)
    z = fit ** 2
    zc = z - z.mean()
    zc = zc - _project(X2, zc)
    bz = float((zc @ resid) / (zc @ zc))
    ss_nad = float((bz ** 2) * (zc @ zc))
    ss_rest = float(((resid - bz * zc) ** 2).sum())
    df_rest = len(y) - X2.shape[1] - 1
    F = (ss_nad / 1) / (ss_rest / df_rest)
    return {"SS_3way_total": float((resid ** 2).sum()),
            "SS_3way_nonadditivity_1df": ss_nad, "SS_3way_remainder": ss_rest,
            "df_3way_total": int(len(y) - X2.shape[1]), "df_remainder": int(df_rest),
            "F": float(F), "p": float(stats.f.sf(F, 1, df_rest))}


# candidate reproductions of the claimed "Tukey F ~ 9.9 / ~764"
cand_F = {}
for r in RESPONSES:
    y = d[r].values
    ss = P1[r]["SS"]
    dft = (nA - 1) * (nB - 1) * (nC - 1)
    other = N - 1 - dft
    c = {}
    c["A_MS3way_over_MS_pooled_all_other"] = float(
        (ss["SS_three_way"] / dft) /
        ((ss["SS_total"] - ss["SS_three_way"]) / (N - 1 - dft)))
    c["B_MS3way_over_MS_2way"] = float(
        (ss["SS_three_way"] / dft) /
        (ss["SS_two_way"] / ((nA - 1) * (nB - 1) + (nA - 1) * (nC - 1) + (nB - 1) * (nC - 1))))
    c["C_MS3way_over_MS_main"] = float(
        (ss["SS_three_way"] / dft) /
        (ss["SS_main"] / ((nA - 1) + (nB - 1) + (nC - 1))))
    c["D_MS2way_plus_3way_over_MS_resid_style"] = None
    # classic textbook: 3-way tested against pooled "within" -- undefined here
    c["E_tukey_1df_additive_model"] = tukey_1df_nonadditivity(y)
    c["F_tukey_split_within_3way"] = tukey_within_3way(y)
    c["SS_ratio_3way_over_2way"] = float(ss["SS_three_way"] / ss["SS_two_way"])
    c["SS_ratio_3way_over_main"] = float(ss["SS_three_way"] / ss["SS_main"])
    cand_F[r] = c
rec["F_statistic_search"] = cand_F

# --- the three adversarial transforms
def eta_of(y):
    ss = decompose_full(y)
    total = ss["SS_total"]
    return {"main": ss["SS_main"] / total, "two_way": ss["SS_two_way"] / total,
            "three_way": ss["SS_three_way"] / total,
            "residual": ss["SS_residual"] / total,
            "SS_total": total}


robust, variants_desc = {}, {}
for r in RESPONSES:
    y = d[r].values.astype(float)
    lo, hi = np.percentile(y, [1, 99])
    v = {
        "raw": y,
        "standardised_unit_variance": (y - y.mean()) / y.std(ddof=1),
        "winsorised_1_99pct": np.clip(y, lo, hi),
        "rank_transformed": stats.rankdata(y).astype(float),
        "rank_normal_scores": stats.norm.ppf((stats.rankdata(y) - 0.375) / (len(y) + 0.25)),
        "log_shifted_positive": np.log(y - y.min() + 1e-3),
    }
    robust[r] = {k: eta_of(vv) for k, vv in v.items()}
    variants_desc[r] = {"winsor_lo": float(lo), "winsor_hi": float(hi),
                        "n_clipped": int(((y < lo) | (y > hi)).sum())}
rec["adversarial_robustness_eta2"] = robust
rec["adversarial_variant_details"] = variants_desc

# Is the eta^2 change under rank transform driven by marginal shape or by
# real structure?  Compare each transformed response's eta2 to what you get
# from a rank transform of the FITTED values of the raw full model.
def saturated_residual_check(y):
    """The full 10x13x11 model with all interactions is saturated (1430 params
    = 1430 obs), so the fitted values are exactly y.  Verify this by solving
    the orthogonal-projection LS on the full block set and comparing."""
    yv = np.asarray(y, float)
    fit = np.zeros_like(yv)
    for c in ["int", "A", "B", "C", "AB", "AC", "BC", "ABC"]:
        Xk = blk[c]
        bk = np.linalg.solve(Xk.T @ Xk, Xk.T @ yv)
        fit += Xk @ bk
    return float(np.abs(fit - yv).max())


rec["saturated_model_reproduces_response_max_abs_err"] = {
    r: saturated_residual_check(d[r].values) for r in RESPONSES}

# --- how much of the rank-transform effect is due to ties in the ranks?
rec["rank_tie_info"] = {r: {"n_ties": int(len(d) - d[r].nunique())} for r in RESPONSES}

# ==================================================== 4. HONEST INFERENCE CHECK
inference = {}
for r in RESPONSES:
    ss = P1[r]["SS"]
    dft = (nA - 1) * (nB - 1) * (nC - 1)
    dfmain = (nA - 1) + (nB - 1) + (nC - 1)
    df2 = (nA - 1) * (nB - 1) + (nA - 1) * (nC - 1) + (nB - 1) * (nC - 1)
    ms_main = ss["SS_main"] / dfmain
    ms_2 = ss["SS_two_way"] / df2
    ms_3 = ss["SS_three_way"] / dft
    resid_df = ss["df"]["residual"]
    inference[r] = {
        "df": {"main": dfmain, "two_way": df2, "three_way": dft,
               "pure_error": 0, "residual_after_full_model": resid_df},
        "has_pure_error_replicate_variance": False,
        "note": ("Single replicate per cell: no pure-error stratum exists. "
                 "The ONLY usable error stratum is the 3-way interaction "
                 f"({dft} df), which is exactly the effect D1 calls 'small but "
                 "nonzero', so testing it against itself is impossible."),
        "MS": {"main": ms_main, "two_way": ms_2, "three_way": ms_3},
        "F_main_over_MS3way": float(ms_main / ms_3),
        "p_main_over_MS3way": float(stats.f.sf(ms_main / ms_3, dfmain, dft)),
        "F_two_way_over_MS3way": float(ms_2 / ms_3),
        "p_two_way_over_MS3way": float(stats.f.sf(ms_2 / ms_3, df2, dft)),
        "F_three_way_over_MS3way": 1.0,
        "p_three_way_over_MS3way": 1.0,
        "eta2_effect_size": P1[r]["eta2"],
        "omega2_3way_vs_3way_stratum_UNUSABLE": None,
        "cohens_f2_3way": float(ss["SS_three_way"] / (ss["SS_total"] - ss["SS_three_way"])),
        "tukey_1df_additive_model": tukey_1df_nonadditivity(d[r].values),
        "tukey_split_within_3way": tukey_within_3way(d[r].values),
        "r_squared_full_model_saturated": 1.0,
    }
rec["inference"] = inference

# ------------------------------------------------- permutation reality check
# With no pure error, the honest way to attach a p-value to the 3-way SS is a
# permutation test that assumes exchangeability of the OBSERVATIONS under a
# null of "additive + 2-way structure" -- i.e. permute the 3-way residuals.
def perm_p_3way(y, n_perm=500):
    yv = np.asarray(y, float)
    X2 = np.hstack([blk[c] for c in ["int", "A", "B", "C", "AB", "AC", "BC"]])
    resid = yv - _project(X2, yv)        # the 3-way stratum values
    obs = float((resid ** 2).sum())
    rng = np.random.default_rng(20260919)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = float((rng.permutation(resid) ** 2).sum())
    return {"observed_SS_3way": obs, "permutation_SS_min": float(perm.min()),
            "permutation_SS_max": float(perm.max()),
            "permutation_SS_std": float(perm.std()),
            "degenerate": bool(np.allclose(perm, obs)),
            "explanation": ("Permuting the 3-way residual vector only relabels "
                            "which cell holds which residual; the sum of squares "
                            "is invariant. No design-based error term exists.")}


rec["permutation_reality_check"] = {r: perm_p_3way(d[r].values, 500) for r in RESPONSES}

# distribution diagnostics for the adversarial question
diag = {}
for r in RESPONSES:
    y = d[r].values.astype(float)
    lo, hi = np.percentile(y, [1, 99])
    z = (y - y.mean()) / y.std(ddof=1)
    diag[r] = {
        "mean": float(y.mean()), "sd": float(y.std(ddof=1)),
        "min": float(y.min()), "max": float(y.max()),
        "range_over_sd": float((y.max() - y.min()) / y.std(ddof=1)),
        "skewness": float(stats.skew(y)), "excess_kurtosis": float(stats.kurtosis(y)),
        "shapiro_W": float(stats.shapiro(y).statistic),
        "shapiro_p": float(stats.shapiro(y).pvalue),
        "max_abs_z": float(np.abs(z).max()),
        "n_abs_z_gt_3": int((np.abs(z) > 3).sum()),
        "n_abs_z_gt_4": int((np.abs(z) > 4).sum()),
        "iqr_over_sd": float((np.percentile(y, 75) - np.percentile(y, 25)) / y.std(ddof=1)),
        "winsor_pull_l1_relative": float(np.abs(np.clip(y, lo, hi) - y).sum() / np.abs(y).sum()),
        "cell_mean_sd": None,
    }
    # spread of the cell means (the 'signal') on the RAW scale vs the sd
    cellm = d.groupby(FACTORS, observed=True)[r].mean()
    diag[r]["cell_mean_sd"] = float(cellm.std(ddof=1))
    diag[r]["cell_mean_sd_over_sd"] = float(cellm.std(ddof=1) / y.std(ddof=1))
rec["distribution_diagnostics"] = diag

# ---------------------------------------------------------------- output
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        fv = float(o)
        return fv if np.isfinite(fv) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None:
        return None
    return o


rec["runtime_sec"] = round(time.time() - t0, 2)
with open(OUT + "/V1_factorial_decomposition.json", "w") as fh:
    json.dump(clean(rec), fh, indent=2)

# ------------------------------------------------------------- report
print("=== 1. DESIGN ===")
print(json.dumps(clean(rec["design"]), indent=2))
print("\n=== 2. ETA^2 ===")
for r in RESPONSES:
    e1, e2 = P1[r]["eta2"], P2[r]["eta2"]
    print(f"{r}")
    print(f"   P1 main={e1['main']:.4f} 2way={e1['two_way']:.4f} "
          f"3way={e1['three_way']:.4f} resid={e1['residual']:.6f}")
    print(f"   P2 main={e2['main']:.4f} 2way={e2['two_way']:.4f} "
          f"3way={e2['three_way']:.4f} resid={e2['residual']:.6f}")
print(f"max |eta(P1)-eta(P2)| = {rec['path_agreement_max_abs_eta2_diff']:.3e}")
print(f"design orthogonality max |offdiag| = "
      f"{rec['effect_coding_design_orthogonal_max_abs_offdiag']:.3e}")
print("\n=== 3. ADVERSARIAL ROBUSTNESS ===")
for r in RESPONSES:
    print(r)
    for k, v in robust[r].items():
        print(f"   {k:28s} main={v['main']:.4f} 2way={v['two_way']:.4f} "
              f"3way={v['three_way']:.4f} resid={v['residual']:.5f}")
print("\n=== 3b. DISTRIBUTION DIAGNOSTICS ===")
for r in RESPONSES:
    print(r, json.dumps(clean(diag[r]), indent=2))
print("\n=== 4. INFERENCE / F SEARCH ===")
for r in RESPONSES:
    print(r)
    print("   F candidates:", json.dumps(
        {k: (vv if not isinstance(vv, dict) else
             {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in vv.items()})
         for k, vv in cand_F[r].items()}, indent=4))
print("\nruntime", rec["runtime_sec"], "s")
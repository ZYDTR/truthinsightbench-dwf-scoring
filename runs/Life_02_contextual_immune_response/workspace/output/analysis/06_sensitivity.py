#!/usr/bin/env python3
"""Sensitivity / resolution limits of the two negative results.

The task's two boundary results are:
  (a) the time-resolved file shows no detectable role difference at any of the
      7 time points (paired t, n = 4 pairs, all p >= 0.51);
  (b) the compartment ORDERING of fold change is not reproducible across assays
      (Kendall's W = 0.778 / p_chi2 = 0.097 for endpoint_count; W = 0.111 /
      p_chi2 = 0.717 for endpoint_fraction).

A null is only meaningful with a stated resolution, so this script computes the
smallest effect each design could have detected, the equivalence margin the data
themselves support, how much data the ordering result would need, and where the
boundary of the one positive ordering result lies.

Sections
  S1  MDE for the paired per-time-point contrast (n = 4 pairs, alpha .05
      two-sided, 80% power) by the closed form (t_crit + t_beta) * SD / sqrt(n)
      and by exact inversion of the noncentral-t power function.
  S2  TOST equivalence against (i) margin = observed paired SD at that time
      point and (ii) margin = 3 percentage points, using the 90% CI.
  S3  Assays required for the observed between-assay concordance of the
      compartment fold-change ordering to reach p < 0.05, under the chi-square
      approximation and under the exact permutation null, plus a bootstrap over
      the observed orderings.
  S4  Boundary of the intervention ordering: recompute Kendall's W and its
      p-value under plausible alternative orderings of the 5 settings (all
      single transpositions plus named scenarios) and report which still give
      p < 0.05, under both the chi-square approximation used in the core
      results and an exact permutation null.
  S5  Within-cell replicate SD of log2 fold change (compartment file)
      translated into the smallest reliably detectable between-compartment
      fold-change ratio difference (n = 4 replicates/cell, 3 compartments).

Run:  python3 output/analysis/06_sensitivity.py
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from common import (
    LOGS,
    ensure_dirs,
    kendall_w,
    load,
    pformat,
    write_json,
)

ensure_dirs()
SECTION_LOG: list[str] = []


def say(message: str = "") -> None:
    print(message)
    SECTION_LOG.append(message)


ALPHA = 0.05
POWER = 0.80
N_PAIRS = 4
M_SETS = 5          # intervention settings
K_ASSAYS = 3        # assays currently observed

T_CRIT_DF3 = float(stats.t.ppf(1 - ALPHA / 2, N_PAIRS - 1))   # 3.182446
T_BETA_DF3 = float(stats.t.ppf(POWER, N_PAIRS - 1))           # 0.978472
T_CRIT_DF6 = float(stats.t.ppf(1 - ALPHA / 2, 6))             # 2.446912
T_BETA_DF6 = float(stats.t.ppf(POWER, 6))                     # 0.905703
T_CRIT_DF54 = float(stats.t.ppf(1 - ALPHA / 2, 54))
T_BETA_DF54 = float(stats.t.ppf(POWER, 54))
CHI2_CRIT_DF2 = float(stats.chi2.ppf(1 - ALPHA, 2))
CHI2_CRIT_DF4 = float(stats.chi2.ppf(1 - ALPHA, 4))
Q_CRIT_3_6 = float(stats.studentized_range.ppf(1 - ALPHA, 3, 6))
Q_CRIT_3_54 = float(stats.studentized_range.ppf(1 - ALPHA, 3, 54))

FORMULAS = {
    "mde_closed_form": "MDE = (t_crit + t_beta) * SD / sqrt(n)",
    "mde_noncentral_t": (
        "solve delta such that power(delta) = P(T' > t_crit | df=n-1, "
        "ncp=delta*sqrt(n)/SD) + P(T' < -t_crit | df=n-1, "
        "ncp=delta*sqrt(n)/SD) = 0.80"
    ),
    "tost": (
        "two one-sided t tests at alpha each: p_lower = P(T_{n-1} >= "
        "(mean - low)/SE), p_upper = P(T_{n-1} <= (mean - high)/SE); TOST p = "
        "max(p_lower, p_upper); equivalence declared iff TOST p < alpha "
        "(equivalently the 90% CI lies inside [-margin, +margin])"
    ),
    "kendall_w_chi2": (
        "chi2 = k * (m - 1) * W with df = m - 1 (chi-square approximation used "
        "in the core results)"
    ),
    "kendall_w_exact": (
        "S = sum_j (S_j - k(m+1)/2)^2 for k independent uniform random orderings "
        "of m items; W = 12 S / (k^2 (m^3 - m)); p = P_perm(S >= S_obs), computed "
        "by exact enumeration of the null distribution of S"
    ),
    "assays_needed": (
        "S_obs(k) = W_obs * k^2 * (m^3 - m) / 12; required k = smallest k with "
        "S_obs(k) >= S_crit(k), where S_crit(k) is the permutation-null 5% "
        "critical value for k raters on m items"
    ),
    "tukey_hsd_mde": (
        "MDE_pair = q_crit(0.05, k_groups, df_error) * SD_within / sqrt(n_per_group)"
    ),
}

say("=" * 78)
say("Sensitivity analysis: resolution limits of the negative / boundary results")
say("=" * 78)
say(f"alpha = {ALPHA} two-sided, power = {POWER}")
say(f"critical values: t(.975,df=3) = {T_CRIT_DF3:.6f}, t(.80,df=3) = {T_BETA_DF3:.6f}")
say(f"                  t(.975,df=6) = {T_CRIT_DF6:.6f}, t(.80,df=6) = {T_BETA_DF6:.6f}")
say(f"                  t(.975,df=54) = {T_CRIT_DF54:.6f}, t(.80,df=54) = {T_BETA_DF54:.6f}")
say(f"                  chi2(.95,df=2) = {CHI2_CRIT_DF2:.6f}, "
    f"chi2(.95,df=4) = {CHI2_CRIT_DF4:.6f}")
say(f"                  q(.95,k=3,df=6) = {Q_CRIT_3_6:.6f}, "
    f"q(.95,k=3,df=54) = {Q_CRIT_3_54:.6f}")

# ==========================================================================
# S1  Minimum detectable effect for the paired time-point contrast
# ==========================================================================
say("\n" + "=" * 78)
say("S1  MDE for the paired per-time-point contrast (n = 4 pairs)")
say("=" * 78)

time_df = load("time_resolved")
wide = time_df.pivot_table(
    index=["time_units", "replicate_id"],
    columns="condition_role",
    values="normalized_response_percent",
).reset_index()
wide.columns.name = None
wide["difference"] = wide["perturbation"] - wide["baseline"]
times = sorted(wide["time_units"].unique())
assert wide["difference"].notna().all(), "paired differences must be complete"

say(f"design: {len(wide)} pairs, {N_PAIRS} replicates per role x time point, "
    f"times = {[float(t) for t in times]}")


def noncentral_t_power(delta: float, sd: float, n: int, df: int):
    """Two-sided power of the paired t test for a true difference ``delta``.

    Returns None when scipy's noncentral t underflows (large noncentrality).
    """
    t_crit = float(stats.t.ppf(1 - ALPHA / 2, df))
    ncp = delta * math.sqrt(n) / sd
    try:
        upper = float(stats.nct.sf(t_crit, df, ncp))
        lower = float(stats.nct.cdf(-t_crit, df, ncp))
    except Exception:
        return None
    if not (np.isfinite(upper) and np.isfinite(lower)):
        return None
    return upper + lower


def noncentral_t_mde(sd: float, n: int, df: int):
    """Invert the noncentral-t power function over a coarse grid, then refine.

    Grid-first bracketing is used because scipy's nct returns NaN for some
    large noncentralities; the grid skips those points.
    """
    hi = 12.0 * sd
    grid = np.linspace(1e-9, hi, 600)
    powers = [noncentral_t_power(float(d), sd, n, df) for d in grid]
    finite = [(float(d), p) for d, p in zip(grid, powers) if p is not None and np.isfinite(p)]
    crossing = None
    for i in range(1, len(finite)):
        if finite[i - 1][1] < POWER <= finite[i][1]:
            crossing = (finite[i - 1][0], finite[i][0])
            break
    if crossing is None:
        return None
    lo, high = crossing
    try:
        root = brentq(lambda d: (noncentral_t_power(d, sd, n, df) or 0.0) - POWER,
                      lo, high, xtol=1e-10, rtol=1e-12)
        return float(root)
    except Exception:
        return float((lo + high) / 2.0)


say(f"formula (closed form): {FORMULAS['mde_closed_form']}")
say(f"formula (exact)      : {FORMULAS['mde_noncentral_t']}")

mde_rows = []
for t in times:
    block = wide[wide["time_units"] == t]
    diff = block["difference"].to_numpy(dtype=float)
    n = len(diff)
    dfree = n - 1
    sd = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    t_crit = float(stats.t.ppf(1 - ALPHA / 2, dfree))
    t_beta = float(stats.t.ppf(POWER, dfree))
    mde_closed = (t_crit + t_beta) * sd / math.sqrt(n)
    mde_exact = noncentral_t_mde(sd, n, dfree)
    test = stats.ttest_rel(block["perturbation"], block["baseline"])
    mde_rows.append({
        "time_units": float(t),
        "n_pairs": int(n),
        "observed_difference_mean_pp": mean,
        "observed_paired_sd_pp": sd,
        "observed_sd_ddof": 1,
        "t_crit": t_crit,
        "t_beta": t_beta,
        "df": dfree,
        "mde_closed_form_pp": float(mde_closed),
        "mde_noncentral_t_pp": mde_exact,
        "ncp_at_mde_closed_form": float(mde_closed * math.sqrt(n) / sd),
        "paired_t": float(test.statistic),
        "paired_p": float(test.pvalue),
        "power_at_observed_effect": noncentral_t_power(abs(mean), sd, n, dfree),
        "observed_effect_inside_mde": bool(abs(mean) < mde_closed),
    })

mde_frame = pd.DataFrame(mde_rows)
for _, row in mde_frame.iterrows():
    nct_txt = (f"{row['mde_noncentral_t_pp']:.4f}"
               if row["mde_noncentral_t_pp"] is not None else "n/a")
    say(f"   t={row['time_units']:5.0f}  diff = {row['observed_difference_mean_pp']:+.3f} pp  "
        f"SD = {row['observed_paired_sd_pp']:8.4f}  "
        f"MDE(closed) = {row['mde_closed_form_pp']:8.4f} pp  MDE(nct) = {nct_txt:>8s} pp  "
        f"paired t = {row['paired_t']:+.3f} (p = {pformat(row['paired_p'])})")

md_median_closed = float(np.median(mde_frame["mde_closed_form_pp"]))
nct_vals = mde_frame["mde_noncentral_t_pp"].dropna()
md_median_exact = float(np.median(nct_vals)) if len(nct_vals) else None
sd_median = float(np.median(mde_frame["observed_paired_sd_pp"]))
sd_pooled = float(np.sqrt((mde_frame["observed_paired_sd_pp"].to_numpy() ** 2).mean()))
mde_from_pooled_sd = (T_CRIT_DF3 + T_BETA_DF3) * sd_pooled / math.sqrt(N_PAIRS)
late = mde_frame[mde_frame["time_units"] == 21].iloc[0]
late_inside = bool(abs(late["observed_difference_mean_pp"]) < late["mde_closed_form_pp"])

say(f"\n   observed paired SD: median = {sd_median:.4f} pp; pooled RMS = "
    f"{sd_pooled:.4f} pp (sqrt of the mean of the 7 per-time-point variances)")
say(f"   MDE computed from the pooled SD: {mde_from_pooled_sd:.4f} pp")
say(f"   median MDE across the 7 time points: closed form = {md_median_closed:.4f} pp; "
    f"noncentral t = "
    f"{md_median_exact if md_median_exact is not None else float('nan'):.4f} pp "
    f"(noncentral-t inversion available for {len(nct_vals)}/7 time points)")
say(f"   MDE range: {mde_frame['mde_closed_form_pp'].min():.4f} to "
    f"{mde_frame['mde_closed_form_pp'].max():.4f} pp")
say(f"   time points whose MDE is at or below 3 pp: "
    f"{int((mde_frame['mde_closed_form_pp'] <= 3.0).sum())}/7")
say(f"\n   STATEMENT: differences smaller than {md_median_closed:.2f} pp "
    f"(median MDE; range {mde_frame['mde_closed_form_pp'].min():.2f}-"
    f"{mde_frame['mde_closed_form_pp'].max():.2f} pp) could not have been detected "
    f"by this paired design at 80% power and alpha = 0.05 two-sided with n = 4 pairs.")
say(f"\n   late-time contrast t=21: observed difference = "
    f"{late['observed_difference_mean_pp']:+.3f} pp against an MDE of "
    f"{late['mde_closed_form_pp']:.3f} pp -> observed effect is "
    f"{'INSIDE (below)' if late_inside else 'OUTSIDE (above)'} the resolution floor")
say(f"   power of the t=21 test at the observed effect size: "
    f"{late['power_at_observed_effect'] if late['power_at_observed_effect'] is not None else float('nan'):.4f}")

# ==========================================================================
# S2  TOST equivalence testing
# ==========================================================================
say("\n" + "=" * 78)
say("S2  TOST equivalence at each time point")
say("=" * 78)
say(f"formula: {FORMULAS['tost']}")
say("margins reported: (i) margin = observed paired SD at that time point")
say("  (margin = one SD, derived from the data), (ii) margin = 3 pp (fixed scale).")

tost_rows = []
for _, row in mde_frame.iterrows():
    t = row["time_units"]
    block = wide[wide["time_units"] == t]
    diff = block["difference"].to_numpy(dtype=float)
    n = len(diff)
    dfree = n - 1
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(1 - ALPHA / 2, dfree))
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    entry = {
        "time_units": float(t),
        "n_pairs": int(n),
        "difference_mean_pp": mean,
        "difference_sd_pp": sd,
        "standard_error_pp": se,
        "df": dfree,
        "t_crit": t_crit,
        "ci90_low_pp": float(ci_low),
        "ci90_high_pp": float(ci_high),
        "ci90_half_width_pp": float(t_crit * se),
    }
    for label, margin in (("sd_margin", sd), ("margin_3pp", 3.0)):
        low, high = -margin, margin
        p_lower = float(stats.t.sf((mean - low) / se, dfree))    # H0: mean <= low
        p_upper = float(stats.t.cdf((mean - high) / se, dfree))  # H0: mean >= high
        entry[f"{label}_margin_pp"] = float(margin)
        entry[f"{label}_tost_p"] = float(max(p_lower, p_upper))
        entry[f"{label}_equivalent"] = bool(max(p_lower, p_upper) < ALPHA)
        entry[f"{label}_ci_inside_margin"] = bool(ci_low > low and ci_high < high)
    tost_rows.append(entry)

tost_frame = pd.DataFrame(tost_rows)
for _, row in tost_frame.iterrows():
    say(f"   t={row['time_units']:5.0f}  diff = {row['difference_mean_pp']:+.3f} pp  "
        f"90% CI [{row['ci90_low_pp']:+.3f}, {row['ci90_high_pp']:+.3f}] "
        f"(half-width {row['ci90_half_width_pp']:.3f})")
    say(f"          margin = 1 SD ({row['sd_margin_margin_pp']:.3f} pp): TOST p = "
        f"{pformat(row['sd_margin_tost_p'])} equivalent = {row['sd_margin_equivalent']}"
        f"   | margin = 3 pp: TOST p = {pformat(row['margin_3pp_tost_p'])} "
        f"equivalent = {row['margin_3pp_equivalent']}")

n_eq_sd = int(tost_frame["sd_margin_equivalent"].sum())
n_eq_3 = int(tost_frame["margin_3pp_equivalent"].sum())
min_p_3 = float(tost_frame["margin_3pp_tost_p"].min())
min_p_sd = float(tost_frame["sd_margin_tost_p"].min())
say(f"\n   equivalent within the 1-SD margin at {n_eq_sd}/7 time points; "
    f"equivalent within the +/-3 pp margin at {n_eq_3}/7 time points")
say(f"   smallest TOST p: {min_p_sd:.4f} (1-SD margin), {min_p_3:.4f} (3 pp margin)")
say(f"   structural reason for the 1-SD margin: with n = 4 the 90% CI half-width is "
    f"t(.95,3)*SD/2 = {float(stats.t.ppf(0.95, 3)) / 2.0:.4f} * SD > 1 * SD, so the CI "
    f"can never fit inside a one-SD margin - equivalence within one SD is "
    f"unreachable by construction at this n.")
say("   INTERPRETATION: failing to declare equivalence is NOT evidence of a "
    "difference. Every paired t is non-significant (all p >= 0.506); the design is "
    "too imprecise to certify either a difference or equivalence at any margin at "
    "or below one SD / 3 pp.")

# ==========================================================================
# S3  How much data would the compartment ordering result need?
# ==========================================================================
say("\n" + "=" * 78)
say("S3  Assays needed for the compartment fold-change ordering to reach p < 0.05")
say("=" * 78)
say(f"formula: {FORMULAS['assays_needed']}")

_EXACT_TABLE_CACHE: dict[tuple[int, int], dict] = {}


def kendall_exact_table(k_max: int, m: int) -> dict:
    """Exact null distributions of S for k = 2..k_max raters on m items.

    Built by sequential convolution of the uniform ordering distribution; works
    for m = 3 and small k because the state space is the set of reachable rank
    sums (O(k^2) states).
    """
    key = (k_max, m)
    if key in _EXACT_TABLE_CACHE:
        return _EXACT_TABLE_CACHE[key]
    perms = list(itertools.permutations(range(1, m + 1)))
    dist: dict[tuple, int] = {tuple([0] * m): 1}
    table: dict[int, dict] = {}
    for k in range(1, k_max + 1):
        new: dict[tuple, int] = defaultdict(int)
        for state, count in dist.items():
            for perm in perms:
                nxt = tuple(state[i] + perm[i] for i in range(m))
                new[nxt] += count
        dist = dict(new)
        if k < 2:
            continue
        total = sum(dist.values())
        center = k * (m + 1) / 2.0
        s_counts: dict[float, int] = defaultdict(int)
        for state, count in dist.items():
            s_counts[float(sum((x - center) ** 2 for x in state))] += count
        crit = None
        tail = 0.0
        for value in sorted(s_counts, reverse=True):
            tail += s_counts[value] / total
            if tail <= ALPHA:
                crit = value
            else:
                break
        table[k] = {"S_counts": dict(s_counts), "total": total, "crit": crit}
    _EXACT_TABLE_CACHE[key] = table
    return table


def kendall_exact_p(S_obs: float, k: int, m: int) -> float:
    table = kendall_exact_table(k, m)
    null = table[k]
    tail = sum(c for s, c in null["S_counts"].items() if s >= S_obs - 1e-9)
    return float(tail / null["total"])


def S_from_W(W: float, k: int, m: int) -> float:
    return float(W * k * k * (m ** 3 - m) / 12.0)


def W_from_S(S: float, k: int, m: int) -> float:
    return float(12.0 * S / (k * k * (m ** 3 - m)))


RESULTS_DIR = ("/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/"
               "project/output/results")
ordering = pd.read_csv(f"{RESULTS_DIR}/compartment_ordering_reproducibility.csv")
fc_ordering = ordering[ordering["measure"] == "fold_change"].copy()
cell = pd.read_csv(f"{RESULTS_DIR}/compartment_cell_effects.csv")
PRIMARY_FC = "fc_geo_mean_ratio"

KMAX_EXACT = 60      # exact-null search cap for m = 3 (S_crit grows ~ linearly)
KMAX_BOOT = 40       # bootstrap search cap

t0 = time.time()
s_crit_table = kendall_exact_table(KMAX_EXACT, 3)
say(f"exact permutation null table built for k = 2..{KMAX_EXACT}, m = 3 "
    f"({time.time() - t0:.2f} s); "
    f"S_crit(k=3) = {s_crit_table[3]['crit']:.1f}, "
    f"S_crit(k=5) = {s_crit_table[5]['crit']:.1f}, "
    f"S_crit(k=27) = {s_crit_table[27]['crit']:.1f}, "
    f"chi2-approx S_crit(k=3) = "
    f"{CHI2_CRIT_DF2 * 3 * 3 * 4 / 12.0:.2f}")

rng = np.random.default_rng(20200101)
assays_needed_rows = []
for _, record in fc_ordering.iterrows():
    endpoint = record["endpoint_type"]
    csv_kendall_w = float(record["kendall_w"])
    m = 3
    block = cell[cell["endpoint_type"] == endpoint]
    mat = block.pivot(index="assay_id", columns="compartment_id", values=PRIMARY_FC)
    ranks = mat.rank(axis=1)
    observed_rank_vectors = [tuple(int(v) for v in ranks.loc[a]) for a in ranks.index]
    # Recompute S from the integer rank matrix.  The result CSV stores W rounded
    # to 6 significant digits, and rounding 14/18 up to 0.777778 moves S just
    # above the achievable value 14, which would corrupt the exact tail
    # probability; S = sum_j (sum_i rank_ij - k(m+1)/2)^2 is exact.
    S_obs_k3 = float(((ranks.to_numpy(dtype=float).sum(axis=0)
                       - K_ASSAYS * (m + 1) / 2.0) ** 2).sum())
    W_obs = W_from_S(S_obs_k3, K_ASSAYS, m)

    # (a) chi-square approximation
    k_chi2 = None
    for k in range(2, 4001):
        if float(stats.chi2.sf(k * (m - 1) * W_obs, m - 1)) < ALPHA:
            k_chi2 = k
            break
    # (b) exact permutation null, search capped at KMAX_EXACT
    k_exact = None
    for k in range(2, KMAX_EXACT + 1):
        crit = s_crit_table[k]["crit"]
        if crit is not None and S_from_W(W_obs, k, m) >= crit:
            k_exact = k
            break

    # (c) bootstrap: resample complete observed orderings with replacement,
    #     using one seeded generator for the whole loop so the result is
    #     reproducible
    boot_found = []
    for _ in range(2000):
        hit = KMAX_BOOT + 1
        draws = rng.integers(0, len(observed_rank_vectors), KMAX_BOOT)
        draws = [observed_rank_vectors[int(idx)] for idx in draws]
        for k in range(2, KMAX_BOOT + 1):
            S_k = float(((np.array(draws[:k], dtype=float).sum(axis=0)
                          - k * (m + 1) / 2.0) ** 2).sum())
            if s_crit_table[k]["crit"] is not None and S_k >= s_crit_table[k]["crit"]:
                hit = k
                break
        boot_found.append(hit)

    boot_arr = np.array(boot_found, dtype=float)
    finite = boot_arr[boot_arr <= KMAX_BOOT]
    boot_summary = {
        "n_bootstrap": int(len(boot_found)),
        "max_k_searched": KMAX_BOOT,
        "min": int(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "p90": float(np.percentile(finite, 90)) if len(finite) else None,
        "max": int(finite.max()) if len(finite) else None,
        "n_not_reached_within_cap": int((boot_arr > KMAX_BOOT).sum()),
    }

    null3 = s_crit_table[3]
    p_at_k3 = kendall_exact_p(S_obs_k3, 3, m)
    row = {
        "endpoint_type": endpoint,
        "measure": "fold_change",
        "observed_kendall_w": W_obs,
        "observed_kendall_w_from_csv_rounded": csv_kendall_w,
        "observed_chi2": float(record["chi2"]),
        "observed_chi2_p": float(record["p_value"]),
        "observed_exact_p_k3": p_at_k3,
        "observed_S_k3": S_obs_k3,
        "n_assays_required_chi2_approx": k_chi2,
        "n_assays_required_exact_permutation": k_exact,
        "exact_search_capped_at": KMAX_EXACT,
        "exact_p_at_k_required": (
            kendall_exact_p(S_from_W(W_obs, k_exact, m), k_exact, m)
            if k_exact is not None else None
        ),
        "chi2_p_at_k_required": (
            float(stats.chi2.sf(k_chi2 * (m - 1) * W_obs, m - 1))
            if k_chi2 is not None else None
        ),
        "bootstrap_assays_needed": boot_summary,
        "observed_rank_vectors_by_assay": {
            str(a): [int(v) for v in ranks.loc[a]] for a in ranks.index
        },
        "exact_p_minimum_at_k3": (null3["S_counts"].get(18.0, 0) / null3["total"]),
        "note": (
            "S_obs(k) holds the observed W fixed as assays are added, so it is the "
            "curve implied by the observed concordance rather than a forecast"
        ),
    }
    assays_needed_rows.append(row)

    say(f"\n   {endpoint}: observed W = {W_obs:.4f} (S = {S_obs_k3:.1f}, "
        f"chi2 = {record['chi2']:.4f}, p_chi2 = {pformat(record['p_value'])}, "
        f"exact p = {p_at_k3:.6f})")
    say(f"      observed per-assay rank vectors (1 = lowest fold change): "
        + "; ".join(f"{a}:{observed_rank_vectors[i]}"
                    for i, a in enumerate(ranks.index)))
    say(f"      exact p at perfect concordance (W = 1, k = 3): "
        f"{null3['S_counts'].get(18.0, 0) / null3['total']:.6f}")
    say(f"      assays needed, chi-square approximation: {k_chi2} "
        f"(p there = "
        f"{pformat(float(stats.chi2.sf(k_chi2 * (m - 1) * W_obs, m - 1))) if k_chi2 else 'n/a'})")
    say(f"      assays needed, exact permutation null: "
        f"{k_exact if k_exact is not None else 'not reached within k = ' + str(KMAX_EXACT)}"
        + (f" (p there = {pformat(kendall_exact_p(S_from_W(W_obs, k_exact, m), k_exact, m))})"
           if k_exact is not None else ""))
    say(f"      bootstrap over observed orderings (2000 draws, cap k = {KMAX_BOOT}): "
        f"min = {boot_summary['min']}, median = {boot_summary['median']}, "
        f"p90 = {boot_summary['p90']}, cap not reached in "
        f"{boot_summary['n_not_reached_within_cap']}/2000 draws")

# ==========================================================================
# S4  Boundary check on the intervention ordering
# ==========================================================================
say("\n" + "=" * 78)
say("S4  Boundary of the intervention ordering: which orderings still give p < 0.05?")
say("=" * 78)

t1 = time.time()
kendall_exact_table(3, M_SETS)
say(f"(exact null table for k = 3, m = {M_SETS} built in {time.time() - t1:.2f} s)")

inter = load("intervention")
inter_mean = (
    inter.groupby(["assay_id", "intervention_setting_id"])["response_value"]
    .mean()
    .reset_index()
    .rename(columns={"response_value": "response_mean"})
)
mat_i = inter_mean.pivot(
    index="assay_id", columns="intervention_setting_id", values="response_mean"
)
rank_matrix = mat_i.rank(axis=1)
settings = list(mat_i.columns)
observed_ranks = rank_matrix.to_numpy(dtype=float)
W_i = kendall_w(observed_ranks)
S_i = float(((observed_ranks.sum(axis=0) - K_ASSAYS * (M_SETS + 1) / 2.0) ** 2).sum())
chi2_i = K_ASSAYS * (M_SETS - 1) * W_i
p_chi2_i = float(stats.chi2.sf(chi2_i, M_SETS - 1))
p_exact_i = kendall_exact_p(S_i, K_ASSAYS, M_SETS)
friedman_i = stats.friedmanchisquare(*[mat_i.loc[a].to_numpy() for a in mat_i.index])

say(f"observed: {K_ASSAYS} assays x {M_SETS} settings, S = {S_i:.1f}, W = {W_i:.4f}")
say(f"   chi2 = {chi2_i:.4f}, df = {M_SETS - 1}, p_chi2 = {pformat(p_chi2_i)} "
    f"-> p < 0.05: {p_chi2_i < ALPHA}")
say(f"   Friedman p = {pformat(float(friedman_i.pvalue))}")
say(f"   exact permutation p = {p_exact_i:.6f} -> p < 0.05: {p_exact_i < ALPHA}")
say("   rank matrix (1 = lowest response):")
for assay in rank_matrix.index:
    say(f"      {assay}: " + "  ".join(
        f"{s[-1]}={int(rank_matrix.loc[assay, s])}" for s in settings))

swap_rows = []
for assay_idx, assay in enumerate(rank_matrix.index):
    for i, j in itertools.combinations(range(M_SETS), 2):
        alt = observed_ranks.copy()
        alt[assay_idx, i], alt[assay_idx, j] = alt[assay_idx, j], alt[assay_idx, i]
        S_alt = float(((alt.sum(axis=0) - K_ASSAYS * (M_SETS + 1) / 2.0) ** 2).sum())
        W_alt = W_from_S(S_alt, K_ASSAYS, M_SETS)
        chi2_alt = K_ASSAYS * (M_SETS - 1) * W_alt
        p_chi2_alt = float(stats.chi2.sf(chi2_alt, M_SETS - 1))
        p_exact_alt = kendall_exact_p(S_alt, K_ASSAYS, M_SETS)
        swap_rows.append({
            "scenario": f"swap {settings[i]} <-> {settings[j]} in {assay}",
            "assay": str(assay),
            "settings_swapped": [str(settings[i]), str(settings[j])],
            "S": S_alt,
            "kendall_w": W_alt,
            "chi2": chi2_alt,
            "p_chi2": p_chi2_alt,
            "p_exact": p_exact_alt,
            "significant_chi2": bool(p_chi2_alt < ALPHA),
            "significant_exact": bool(p_exact_alt < ALPHA),
        })

swap_frame = pd.DataFrame(swap_rows).sort_values(
    "S", ascending=False).reset_index(drop=True)
say(f"\n   A. all {len(swap_frame)} single transpositions in one assay, ordered by S:")
for _, row in swap_frame.iterrows():
    say(f"      {row['scenario']:52s} S = {row['S']:5.1f}  W = {row['kendall_w']:.4f}  "
        f"p_chi2 = {pformat(row['p_chi2'])} [{'sig' if row['significant_chi2'] else 'ns '}]  "
        f"p_exact = {row['p_exact']:.6f} [{'sig' if row['significant_exact'] else 'ns '}]")
n_sig_chi2 = int(swap_frame["significant_chi2"].sum())
n_sig_exact = int(swap_frame["significant_exact"].sum())
n_swap = int(len(swap_frame))
row_ns_chi2 = swap_frame[~swap_frame["significant_chi2"]]
row_ns_exact = swap_frame[~swap_frame["significant_exact"]]
say(f"   -> {n_sig_chi2}/{n_swap} single swaps keep p_chi2 < 0.05; "
    f"{n_sig_exact}/{n_swap} keep p_exact < 0.05")
max_S_ns = float(row_ns_chi2["S"].max()) if len(row_ns_chi2) else float("nan")
min_S_sig = float(swap_frame.loc[swap_frame["significant_chi2"], "S"].min()) \
    if int(swap_frame["significant_chi2"].sum()) else float("nan")
say(f"   -> chi-square screen boundary: the largest S that FAILS p_chi2 < 0.05 is "
    f"{max_S_ns:.1f} (p_chi2 = "
    f"{pformat(row_ns_chi2.loc[row_ns_chi2['S'].idxmax(), 'p_chi2'])}); the smallest "
    f"S that PASSES is {min_S_sig:.1f} (p_chi2 = "
    f"{pformat(swap_frame.loc[swap_frame['significant_chi2'], 'p_chi2'].max())})")
max_S_ns_exact = float(row_ns_exact["S"].max()) if len(row_ns_exact) else float("nan")
say(f"   -> exact-null boundary: the largest S that FAILS exact p < 0.05 is "
    f"{max_S_ns_exact:.1f}; exact S_crit(k=3, m=5) = "
    f"{kendall_exact_table(3, M_SETS)[3]['crit']:.1f}")

scenario_rows = []


def add_scenario(name: str, mat: np.ndarray, note: str) -> None:
    k, m = mat.shape
    S = float(((mat.sum(axis=0) - k * (m + 1) / 2.0) ** 2).sum())
    W = W_from_S(S, k, m)
    chi2 = k * (m - 1) * W
    p_chi2 = float(stats.chi2.sf(chi2, m - 1))
    p_exact = kendall_exact_p(S, k, m)
    scenario_rows.append({
        "scenario": name,
        "S": S,
        "kendall_w": W,
        "chi2": chi2,
        "p_chi2": p_chi2,
        "p_exact": p_exact,
        "significant_chi2": bool(p_chi2 < ALPHA),
        "significant_exact": bool(p_exact < ALPHA),
        "note": note,
    })
    say(f"      {name:58s} S = {S:5.1f}  W = {W:.4f}  p_chi2 = {pformat(p_chi2)} "
        f"[{'sig' if p_chi2 < ALPHA else 'ns '}]  p_exact = {p_exact:.6f} "
        f"[{'sig' if p_exact < ALPHA else 'ns '}]")


say("\n   B. plausible alternative orderings:")
add_scenario("observed", observed_ranks.copy(), "core result")
add_scenario("perfect concordance (all assays = assay_1 order)",
             np.tile(observed_ranks[0], (K_ASSAYS, 1)), "W = 1.0")
i25 = settings.index("intervention_setting_2")
j25 = settings.index("intervention_setting_5")
for assay_idx, assay in enumerate(rank_matrix.index):
    alt = observed_ranks.copy()
    alt[assay_idx, i25], alt[assay_idx, j25] = alt[assay_idx, j25], alt[assay_idx, i25]
    add_scenario(f"swap settings 2<->5 in {assay} only", alt,
                 "this is the observed cross-assay rank spread" if assay == "assay_2"
                 else "")
for assay_idx, assay in enumerate(rank_matrix.index):
    alt = observed_ranks.copy()
    alt[assay_idx] = observed_ranks[1]
    add_scenario(f"{assay} adopts the assay_2 ordering", alt,
                 "single-assay replacement")
for assay_idx, assay in enumerate(rank_matrix.index):
    alt = observed_ranks.copy()
    alt[assay_idx] = alt[assay_idx][::-1]
    add_scenario(f"{assay} fully reversed", alt, "adversarial single assay")
alt = observed_ranks.copy()
alt[1], alt[2] = observed_ranks[2].copy(), observed_ranks[1].copy()
add_scenario("assay_2 and assay_3 orderings exchanged",
             alt, "rank-identical reshuffle, S unchanged")

scenario_frame = pd.DataFrame(scenario_rows)
say(f"\n   summary: {int(scenario_frame['significant_chi2'].sum())}/"
    f"{len(scenario_frame)} scenarios keep p_chi2 < 0.05; "
    f"{int(scenario_frame['significant_exact'].sum())}/{len(scenario_frame)} keep "
    f"p_exact < 0.05")
ve_chi2 = int(scenario_frame["significant_exact"].sum())
say(f"   the chi-square approximation is anti-conservative for this design "
    f"(observed W = {W_i:.4f}: p_chi2 = {pformat(p_chi2_i)} vs exact p = "
    f"{p_exact_i:.6f}); the exact null therefore keeps significance for more "
    f"perturbations of the ordering than the chi-square screen does")

# ==========================================================================
# S5  Replicate resolution of the compartment fold change
# ==========================================================================
say("\n" + "=" * 78)
say("S5  Within-cell replicate resolution of log2 fold change (compartment file)")
say("=" * 78)
say(f"formula: {FORMULAS['tukey_hsd_mde']}")

comp = load("compartment")
rep = comp.pivot_table(
    index=["assay_id", "endpoint_type", "compartment_id", "replicate_id"],
    columns="condition_role",
    values="response_value",
).reset_index()
rep.columns.name = None
rep["fold_change"] = rep["perturbation"] / rep["baseline"]
rep["log2_fold_change"] = np.log2(rep["fold_change"])

cell_keys = ["assay_id", "endpoint_type", "compartment_id"]
within = rep.groupby(cell_keys)["log2_fold_change"].agg(["std", "count"]).reset_index()
within = within.rename(columns={"std": "sd_log2fc", "count": "n_replicates"})
df_within = int(within["n_replicates"].sum() - len(within))
sd_pooled_repl = float(math.sqrt((within["sd_log2fc"] ** 2).mean()))
sd_median_repl = float(within["sd_log2fc"].median())
sd_min_repl = float(within["sd_log2fc"].min())
sd_max_repl = float(within["sd_log2fc"].max())
n_per_cell = int(within["n_replicates"].iloc[0])
# 95% CI on a chi-square(3)-based SD estimate, for the uncertainty caveat
sd_ci_low = sd_pooled_repl * math.sqrt(df_within / float(stats.chi2.ppf(0.975, df_within)))
sd_ci_high = sd_pooled_repl * math.sqrt(df_within / float(stats.chi2.ppf(0.025, df_within)))

say(f"   design: {len(within)} cells, {n_per_cell} replicates per cell, "
    f"df_within = {df_within}")
say(f"   within-cell SD of log2 fold change: pooled RMS = {sd_pooled_repl:.4f} log2 "
    f"units, median = {sd_median_repl:.4f} (range {sd_min_repl:.4f} - {sd_max_repl:.4f})")
say(f"   -> geometric SD of the per-replicate fold change = "
    f"2^{sd_pooled_repl:.4f} = {2 ** sd_pooled_repl:.4f}x")
say(f"   chi-square 95% CI on that SD (df = {df_within}): "
    f"{sd_ci_low:.4f} - {sd_ci_high:.4f}")

q_crit = Q_CRIT_3_6
mde_pair_log2_tukey = q_crit * sd_pooled_repl / math.sqrt(n_per_cell)
mde_pair_ratio = 2.0 ** mde_pair_log2_tukey
mde_two_sample_log2 = (T_CRIT_DF6 + T_BETA_DF6) * sd_pooled_repl * \
    math.sqrt(2.0 / n_per_cell)
mde_paired_log2 = (T_CRIT_DF3 + T_BETA_DF3) * sd_pooled_repl * math.sqrt(2.0) / \
    math.sqrt(N_PAIRS)
mde_tukey_df54 = Q_CRIT_3_54 * sd_pooled_repl / math.sqrt(n_per_cell)
mde_t_df54 = (T_CRIT_DF54 + T_BETA_DF54) * sd_pooled_repl * math.sqrt(2.0 / n_per_cell)

say(f"   MDE between two compartment means, Tukey-style (k = 3 groups, n = 4, "
    f"df_error = 6): q * SD / sqrt(n) = {q_crit:.4f} * {sd_pooled_repl:.4f} / "
    f"{math.sqrt(n_per_cell):.4f} = {mde_pair_log2_tukey:.4f} log2 "
    f"(= {mde_pair_ratio:.4f}x fold-change ratio)")
say(f"   MDE between two compartment means, two-sample t (df = 6): "
    f"{mde_two_sample_log2:.4f} log2 (= {2 ** mde_two_sample_log2:.4f}x)")
say(f"   MDE between two compartment means, Tukey with pooled df = {df_within}: "
    f"{mde_tukey_df54:.4f} log2 (= {2 ** mde_tukey_df54:.4f}x); "
    f"two-sample t with df = {df_within}: {mde_t_df54:.4f} log2 "
    f"(= {2 ** mde_t_df54:.4f}x)")
say(f"   MDE if the same biological replicate could be paired across compartments "
    f"(upper bound using SD_gap = sqrt(2)*SD_within): {mde_paired_log2:.4f} log2 "
    f"(= {2 ** mde_paired_log2:.4f}x)")
obs_span = 1.0385632027597602 - 0.7595626797853864
say(f"   observed compartment-level log2FC spread (core analysis): "
    f"compartment_1 = 1.0386, compartment_2 = 0.8137, compartment_3 = 0.7596; "
    f"span = {obs_span:.4f} log2 = {2 ** obs_span:.4f}x")
say(f"   -> the reliable floor ({mde_pair_log2_tukey:.3f} log2 = "
    f"{mde_pair_ratio:.3f}x) is much larger than the entire observed compartment "
    f"span ({obs_span:.3f} log2 = {2 ** obs_span:.3f}x); every bottom-two ordering "
    f"gap from the core results (0.0048 - 0.3370 log2) is far below that floor, "
    f"so the ordering is being read off differences the per-cell replicates "
    f"cannot resolve")

# ==========================================================================
# Persist
# ==========================================================================
mde_timepoints = {
    "design": {
        "n_pairs_per_timepoint": N_PAIRS,
        "n_timepoints": int(len(times)),
        "alpha": ALPHA,
        "two_sided": True,
        "power": POWER,
        "df": N_PAIRS - 1,
    },
    "critical_values": {
        "t_crit_df3": T_CRIT_DF3,
        "t_beta_df3": T_BETA_DF3,
        "t_crit_df6": T_CRIT_DF6,
        "t_beta_df6": T_BETA_DF6,
        "t_crit_df54": T_CRIT_DF54,
        "t_beta_df54": T_BETA_DF54,
        "chi2_crit_df2": CHI2_CRIT_DF2,
        "chi2_crit_df4": CHI2_CRIT_DF4,
        "q_crit_k3_df6": Q_CRIT_3_6,
        "q_crit_k3_df54": Q_CRIT_3_54,
    },
    "formulas": FORMULAS,
    "per_timepoint": mde_frame.to_dict(orient="records"),
    "pooled_paired_sd_pp": sd_pooled,
    "median_paired_sd_pp": sd_median,
    "median_mde_closed_form_pp": md_median_closed,
    "median_mde_noncentral_t_pp": md_median_exact,
    "mde_from_pooled_sd_pp": mde_from_pooled_sd,
    "mde_min_pp": float(mde_frame["mde_closed_form_pp"].min()),
    "mde_max_pp": float(mde_frame["mde_closed_form_pp"].max()),
    "timepoints_with_mde_at_or_below_3pp": int(
        (mde_frame["mde_closed_form_pp"] <= 3.0).sum()),
    "statement": (
        f"differences smaller than {md_median_closed:.2f} pp (median MDE across the "
        f"7 time points; range {mde_frame['mde_closed_form_pp'].min():.2f}-"
        f"{mde_frame['mde_closed_form_pp'].max():.2f} pp) could not have been "
        f"detected at 80% power, alpha = 0.05 two-sided, with n = 4 paired "
        f"replicates"
    ),
    "late_time_t21": {
        "observed_difference_pp": float(late["observed_difference_mean_pp"]),
        "mde_pp": float(late["mde_closed_form_pp"]),
        "inside_resolution": late_inside,
        "power_at_observed_effect": (
            float(late["power_at_observed_effect"])
            if late["power_at_observed_effect"] is not None else None),
        "interpretation": (
            f"the observed {late['observed_difference_mean_pp']:+.2f} pp at t = 21 "
            f"is INSIDE the {late['mde_closed_form_pp']:.2f} pp resolution floor: "
            f"it is smaller than the effect this design could detect with 80% "
            f"power, so the non-significant paired test at t = 21 is uninformative "
            f"about whether a true difference of that size exists"
        ),
    },
}

tost_payload = {
    "alpha_each_side": ALPHA,
    "ci_level_reported": 1 - 2 * ALPHA,
    "formula": FORMULAS["tost"],
    "margins": {
        "sd_margin": "margin = observed paired SD at that time point (one SD)",
        "margin_3pp": "margin = 3 percentage points (fixed scale)",
    },
    "per_timepoint": tost_frame.to_dict(orient="records"),
    "n_equivalent_within_sd_margin": n_eq_sd,
    "n_equivalent_within_3pp": n_eq_3,
    "n_timepoints": int(len(tost_frame)),
    "min_tost_p_sd_margin": min_p_sd,
    "min_tost_p_3pp_margin": min_p_3,
    "structural_note": (
        "with n = 4 the 90% CI half-width is t(.95,3)*SD/2 = "
        f"{float(stats.t.ppf(0.95, 3)) / 2.0:.4f} * SD, which exceeds 1 * SD, so "
        "equivalence within a one-SD margin is unreachable by construction at this "
        "sample size"
    ),
    "interpretation": (
        "equivalence is declared at 0/7 time points under either margin. This is a "
        "statement about precision, not about effect: failing to declare "
        "equivalence is NOT evidence of a difference. The paired t tests are all "
        "non-significant (min p = 0.506) and all TOST p-values exceed 0.07, so the "
        "time-resolved data are consistent both with no role difference and with a "
        "role difference whose size this design cannot resolve."
    ),
}

assays_needed_payload = {
    "formula": FORMULAS["assays_needed"],
    "model": (
        "hold the observed W fixed and ask how many concordant assays would be "
        "needed: S_obs(k) = W_obs * k^2 * (m^3 - m) / 12, judged against the exact "
        "permutation critical value S_crit(k) (and, separately, against the "
        "chi-square approximation chi2 = k(m-1)W with df = m-1)"
    ),
    "per_endpoint": assays_needed_rows,
    "observed_kendall_w": {
        row["endpoint_type"]: row["observed_kendall_w"] for row in assays_needed_rows},
    "observed_exact_p_k3": {
        row["endpoint_type"]: row["observed_exact_p_k3"] for row in assays_needed_rows},
    "observed_chi2_p_k3": {
        row["endpoint_type"]: row["observed_chi2_p"] for row in assays_needed_rows},
    "n_assays_required_chi2_approx": {
        row["endpoint_type"]: row["n_assays_required_chi2_approx"]
        for row in assays_needed_rows},
    "n_assays_required_exact_permutation": {
        row["endpoint_type"]: row["n_assays_required_exact_permutation"]
        for row in assays_needed_rows},
    "exact_search_capped_at": KMAX_EXACT,
    "bootstrap_capped_at": KMAX_BOOT,
    "interpretation": (
        "endpoint_count (W = 0.778, well above chance) would reach p < 0.05 with "
        "4 assays under the chi-square screen and 5 under the exact permutation "
        "null. endpoint_fraction (W = 0.111, near chance) needs 27 assays under "
        "the chi-square screen and 29 under the exact permutation null; because "
        "S grows as W*k^2 while S_crit(k) "
        "grows nearly linearly, that number is finite but the ordering itself is "
        "essentially chance-level (3 assays give W = 0.111 with exact p = 0.944). "
        "The bottleneck for the endpoint_fraction boundary is concordance, not "
        "sample size: more assays at the same concordance would make the test "
        "significant without making the ordering reproducible, which is exactly "
        "why the reproduction claim cannot be rescued by adding data."
    ),
}

intervention_payload = {
    "observed": {
        "n_assays": K_ASSAYS,
        "n_settings": M_SETS,
        "S": S_i,
        "kendall_w": W_i,
        "chi2": chi2_i,
        "df": M_SETS - 1,
        "p_chi2": p_chi2_i,
        "p_exact_permutation": p_exact_i,
        "friedman_p": float(friedman_i.pvalue),
        "significant_at_0.05_chi2": bool(p_chi2_i < ALPHA),
        "significant_at_0.05_exact": bool(p_exact_i < ALPHA),
    },
    "critical_values": {
        "S_crit_exact_k3_m5": kendall_exact_table(3, M_SETS)[3]["crit"],
        "S_crit_chi2_approx_k3_m5": float(
            CHI2_CRIT_DF4 * K_ASSAYS * M_SETS * (M_SETS + 1) / 12.0),
        "exact_p_at_S_crit": kendall_exact_p(
            kendall_exact_table(3, M_SETS)[3]["crit"], 3, M_SETS),
        "exact_p_at_perfect_concordance": kendall_exact_p(90.0, 3, M_SETS),
    },
    "single_swaps": swap_frame.to_dict(orient="records"),
    "n_single_swaps_total": n_swap,
    "n_single_swaps_significant_chi2": n_sig_chi2,
    "n_single_swaps_significant_exact": n_sig_exact,
    "named_scenarios": scenario_frame.to_dict(orient="records"),
    "boundary": (
        f"the observed S = {S_i:.0f}, W = {W_i:.4f} sits between the chi-square "
        f"5% cutoff (chi2 = {CHI2_CRIT_DF4:.4f} <=> S > "
        f"{float(CHI2_CRIT_DF4 * K_ASSAYS * M_SETS * (M_SETS + 1) / 12.0):.1f}) "
        f"and the exact permutation 5% cutoff (S_crit = "
        f"{kendall_exact_table(3, M_SETS)[3]['crit']:.0f}). Under the chi-square "
        "approximation used in the core results the conclusion survives only the "
        f"{n_sig_chi2}/{n_swap} single transpositions with S >= 74 of a maximum 90; "
        "transposing settings 1 and 3 in any assay (S = 70, p_chi2 = 0.053) or "
        "settings 4 and 5 in assay_1/assay_3 (S = 66, p_chi2 = 0.066) already "
        "breaks it. Under the exact permutation null more single transpositions "
        f"survive ({n_sig_exact}/{n_swap}). The p < 0.05 conclusion is therefore "
        "at the edge of what 3 assays x 5 settings can resolve, which is expected "
        "because a Kendall/Friedman test on 3 raters has very little power."
    ),
}

replicate_payload = {
    "design": {
        "cells": int(len(within)),
        "replicates_per_cell": n_per_cell,
        "df_within": df_within,
        "compartments": 3,
    },
    "within_cell_sd_log2fc": {
        "pooled_rms": sd_pooled_repl,
        "median": sd_median_repl,
        "min": sd_min_repl,
        "max": sd_max_repl,
        "chi2_95ci_low": sd_ci_low,
        "chi2_95ci_high": sd_ci_high,
        "fold_change_equivalent": float(2 ** sd_pooled_repl),
        "per_cell": within.to_dict(orient="records"),
    },
    "mde_between_compartments": {
        "tukey_k3_df6_log2": float(mde_pair_log2_tukey),
        "tukey_k3_df6_ratio": float(mde_pair_ratio),
        "two_sample_t_df6_log2": float(mde_two_sample_log2),
        "two_sample_t_df6_ratio": float(2 ** mde_two_sample_log2),
        "tukey_k3_df54_log2": float(mde_tukey_df54),
        "tukey_k3_df54_ratio": float(2 ** mde_tukey_df54),
        "two_sample_t_df54_log2": float(mde_t_df54),
        "two_sample_t_df54_ratio": float(2 ** mde_t_df54),
        "paired_within_replicate_log2_upper_bound": float(mde_paired_log2),
        "paired_within_replicate_ratio_upper_bound": float(2 ** mde_paired_log2),
        "q_crit_k3_df6": q_crit,
        "q_crit_k3_df54": Q_CRIT_3_54,
    },
    "observed_compartment_span_log2": float(obs_span),
    "observed_compartment_span_ratio": float(2 ** obs_span),
    "observed_ordering_gaps_log2": {
        "min": 0.00475864,
        "max": 0.337009,
        "source": "results/robustness_ordering_gaps.csv (gap_1_2 / gap_2_3 per "
                  "assay x endpoint, log2 fold-change scale)",
    },
    "interpretation": (
        f"with {n_per_cell} replicates per cell the within-cell SD of log2 fold "
        f"change is {sd_pooled_repl:.3f} log2 units ({2 ** sd_pooled_repl:.2f}x on "
        f"the ratio scale). Taking the within-cell SD as the compartment-level "
        f"error SD, two compartment mean fold changes can be reliably separated "
        f"only if they differ by about {mde_pair_log2_tukey:.3f} log2 = "
        f"{mde_pair_ratio:.2f}x or more (Tukey-style, k = 3 compartments, n = 4, "
        f"df_error = 6; {mde_tukey_df54:.2f}x using the pooled df = {df_within}). "
        f"This required difference is much larger than the entire observed "
        f"compartment span ({obs_span:.3f} log2 = {2 ** obs_span:.2f}x) and than "
        f"every observed ordering gap (0.005 - 0.337 log2), so the replicate "
        f"noise of a single cell is of the same size as the between-compartment "
        f"effect it would need to resolve. This is the proximate reason the "
        f"ordering cannot be reproduced across assays."
    ),
}

conclusion = (
    "Both boundary results have stated resolutions. (1) Time-resolved role "
    f"contrast: with n = 4 pairs and a pooled paired SD of {sd_pooled:.2f} pp, the "
    f"design could only detect differences of about {md_median_closed:.1f} pp "
    f"(median MDE; range {mde_frame['mde_closed_form_pp'].min():.1f}-"
    f"{mde_frame['mde_closed_form_pp'].max():.1f} pp). The observed late-time "
    f"difference at t = 21 ({late['observed_difference_mean_pp']:+.2f} pp) lies "
    "inside that resolution, and every time point is non-significant (min p = "
    "0.506). The data therefore cannot distinguish 'no role difference' from a "
    "difference of the size the data themselves suggest. Equivalence is declared "
    "at 0/7 time points under either a one-SD or a 3 pp margin (min TOST p = "
    f"{min_p_3:.3f} and {min_p_sd:.3f}); that failure must not be read as evidence "
    "of a difference. (2) Compartment ordering of fold change: the "
    "endpoint_fraction ordering is not reproducible (W = 0.111, exact permutation "
    f"p = {assays_needed_rows[1]['observed_exact_p_k3']:.3f} for the fraction "
    "endpoint) and would need 27 assays at that same concordance to reach p < "
    "0.05, which illustrates the boundary rather than rescuing it: a significant "
    "Kendall W after adding that many chance-level orderings would not make the "
    "ordering reproducible. The endpoint_count ordering is more concordant "
    "(W = 0.778) but still non-significant at k = 3 (exact p = 0.194) and would "
    "need about 5 concordant assays. Both exact p-values are much larger than the "
    "chi-square p-values used in the core results, so the ordering boundary is "
    "actually stronger than the core result states. (3) Replicate resolution: the "
    f"within-cell SD of log2 fold change is {sd_pooled_repl:.2f} log2 units, "
    "which puts the smallest reliably detectable between-compartment "
    f"fold-change difference at about {mde_pair_log2_tukey:.2f} log2 = "
    f"{mde_pair_ratio:.2f}x, far above the whole observed compartment span "
    f"({2 ** obs_span:.2f}x); the ordering is therefore read off differences "
    "below the per-cell replicate noise floor."
)

caveats = [
    "Data-precision note: the existing result CSV stores Kendall's W rounded to "
    "6 significant digits, and rounding 14/18 to 0.777778 moves S just above the "
    "achievable value 14. S is recomputed here from the integer rank matrix "
    "(14 for the count endpoint, 2 for the fraction endpoint) rather than from "
    "the rounded W; both the rounded CSV value and the exact value are stored in "
    "the report. No data file was modified.",
    "Computations that failed and how they were handled: scipy.stats.nct returns "
    "NaN for some large noncentralities (e.g. ncp = 8 with df = 3), so the "
    "noncentral-t MDE inversion brackets on a grid and skips NaN points; the "
    "closed-form MDE is reported for every time point and the exact inversion for "
    "all 7 in this run. scipy.stats.studentized_range was probed for a "
    "noncentral power calculation and does accept a noncentrality argument, but "
    "the Tukey MDE is reported with the standard central-q definition to stay "
    "conservative and reproducible from the printed q critical value.",
    "The exact permutation null for Kendall's W is discrete: at k = 3 raters the "
    f"smallest achievable p is {kendall_exact_table(3, 3)[3]['S_counts'].get(18.0, 0) / kendall_exact_table(3, 3)[3]['total']:.4f} "
    "for m = 3 and 6.9e-5 for m = 5. The chi-square approximation used in the "
    "core results is anti-conservative for both ordering analyses (compartment "
    f"count endpoint: p_chi2 = 0.097 vs exact p = 0.194; intervention: p_chi2 = "
    f"{pformat(p_chi2_i)} vs exact p = {p_exact_i:.6f}).",
    "S3 holds the observed W fixed as assays are added. That answers 'how much "
    "data at this concordance', not 'how much data will a real new assay need'; "
    "the bootstrap over the observed orderings (2000 draws) is reported as a "
    "second, non-parametric view. Both are capped (exact search at k = 60, "
    "bootstrap at k = 40) and the caps are reported.",
    "MDE calculations treat the observed SD as the true SD and the paired "
    "differences as normal. With n = 4 the SD has only df = 3; the 95% chi-square "
    "interval for the pooled within-cell compartment SD (df = 54) is "
    f"[{sd_ci_low:.3f}, {sd_ci_high:.3f}] log2, and the interval for the n = 4 "
    "paired SDs is far wider, so every MDE here is itself uncertain by roughly a "
    "factor of two or more.",
    "The one-SD TOST margin is wide by construction (and structurally unreachable "
    "at n = 4); it is reported because the task asked for a data-derived margin. "
    "The 3 pp margin was chosen because 3 pp is the scale of the observed "
    "late-time divergence.",
    "Noncentral-t power inversion uses scipy.stats.nct on a grid; for very large "
    "noncentrality scipy returns NaN and those grid points are skipped, so the "
    f"exact MDE is reported only where it was finite ({len(nct_vals)}/7 time "
    "points) with the closed-form value reported alongside everywhere.",
    "S4 operates on the 3 x 5 assay-by-setting matrix of mean responses; a "
    "Kendall/Friedman test on 3 raters has very low power, so 'p < 0.05 is lost "
    "under an additional single swap' describes the limits of this design rather "
    "than a claim about the underlying ordering. Conclusions are reported under "
    "both the chi-square screen and the exact permutation null because they "
    "disagree about which swaps are tolerable.",
    "No causal or mechanistic claim is made: compartment and intervention "
    "associations do not identify a unique cellular mechanism (DATA_GUIDE.md).",
]

report = {
    "script": "output/analysis/06_sensitivity.py",
    "task": "Life_02_contextual_immune_response",
    "purpose": (
        "resolution limits (MDE, TOST equivalence, assays needed, ordering "
        "boundary, replicate resolution) for the two negative / boundary results"
    ),
    "mde_timepoints": mde_timepoints,
    "tost": tost_payload,
    "assays_needed_ordering": assays_needed_payload,
    "intervention_boundary": intervention_payload,
    "replicate_resolution": replicate_payload,
    "conclusion": conclusion,
    "caveats": caveats,
}

write_json("sensitivity_report.json", report)
(LOGS / "06_sensitivity.log").write_text("\n".join(SECTION_LOG) + "\n", encoding="utf-8")
print("\n[done] 06_sensitivity.py")
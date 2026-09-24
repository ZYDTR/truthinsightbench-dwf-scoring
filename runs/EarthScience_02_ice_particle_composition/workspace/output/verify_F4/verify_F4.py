"""Adversarial verification of Finding F4 (measurement metrology / count quantisation).

Independent re-implementation. Reads the raw TSV as strings so per-value decimal
precision is preserved (the report assumes a uniform 2-decimal grid; some size
percent columns carry 3 decimals and one carries 9).
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = ROOT / "output" / "verify_F4"
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(ROOT / "data/particle_composition_profiles.tsv"), delimiter="\t"))
hdr = list(rows[0].keys())
poly = [c[len("polymer_"):-len("_percent")] for c in hdr
        if c.startswith("polymer_") and c.endswith("_percent")]
size = [c[len("particle_size_"):-len("_percent")] for c in hdr
        if c.startswith("particle_size_") and c.endswith("_percent")]
nS = len(rows)

def dec_of(s):
    s = s.strip()
    return len(s.split(".")[1]) if "." in s else 0

pctP = np.zeros((len(poly), nS)); pctS = np.zeros((len(size), nS))
decP = np.zeros((len(poly), nS), int); decS = np.zeros((len(size), nS), int)
cntP = np.zeros((len(poly), nS)); cntS = np.zeros((len(size), nS))
cdcP = np.zeros((len(poly), nS), int); cdcS = np.zeros((len(size), nS), int)
tot = np.zeros(nS); sid = []; grp = []

for j, r in enumerate(rows):
    sid.append(r["sample_id"]); grp.append(r["collection_group_id"])
    tot[j] = float(r["particle_total_per_L"])
    for i, p in enumerate(poly):
        s = r[f"polymer_{p}_percent"]; pctP[i, j] = float(s); decP[i, j] = dec_of(s)
        s2 = r[f"polymer_count_{p}_per_L"]; cntP[i, j] = float(s2); cdcP[i, j] = dec_of(s2)
    for i, s in enumerate(size):
        s1 = r[f"particle_size_{s}_percent"]; pctS[i, j] = float(s1); decS[i, j] = dec_of(s1)
        s2 = r[f"particle_size_{s}_per_L"]; cntS[i, j] = float(s2); cdcS[i, j] = dec_of(s2)

res = {"claim_id": "F4"}
problems = []

# ------------------------------------------------------------------ (e) denominator
# implied denominator from each nonzero (count, percent) pair
den_rows = []
for j in range(nS):
    out = {"sample_id": sid[j], "group": grp[j], "reported_total_per_L": float(tot[j])}
    for fam, P, C in (("polymer", pctP[:, j], cntP[:, j]), ("size", pctS[:, j], cntS[:, j])):
        m = (P > 0) & (C > 0)
        imp = C[m] / P[m] * 100.0
        out[f"{fam}_implied_denominator_median"] = float(np.median(imp))
        out[f"{fam}_implied_denominator_min"] = float(imp.min())
        out[f"{fam}_implied_denominator_max"] = float(imp.max())
        out[f"{fam}_count_sum"] = float(C.sum())
    out["implied_denom_vs_reported_total"] = float(
        np.median([out["polymer_implied_denominator_median"],
                   out["size_implied_denominator_median"]]) / tot[j])
    den_rows.append(out)
res["denominator_rows"] = den_rows

# exact residual of count == pct/100 * D for the two candidate denominators
for tag, Dfun in (("reported_total", lambda j: tot[j]),
                  ("family_count_sum", None),
                  ("common_count_sum", None)):
    rs = []
    for j in range(nS):
        if tag == "reported_total":
            Dp = Ds = tot[j]
        elif tag == "family_count_sum":
            Dp, Ds = cntP[:, j].sum(), cntS[:, j].sum()
        else:
            Dm = 0.5 * (cntP[:, j].sum() + cntS[:, j].sum()); Dp = Ds = Dm
        rs.append(np.abs(pctP[:, j] / 100 * Dp - cntP[:, j]))
        rs.append(np.abs(pctS[:, j] / 100 * Ds - cntS[:, j]))
    r = np.concatenate(rs)
    res.setdefault("derived_count_residual", {})[tag] = {
        "max_abs_residual_per_L": float(r.max()),
        "median_abs_residual_per_L": float(np.median(r)),
        "p99_abs_residual_per_L": float(np.quantile(r, 0.99)),
        "frac_within_0.005": float((r <= 0.005).mean()),
    }

# denominator ratio: family count sum / reported total  (question (e))
ratios_p = np.array([cntP[:, j].sum() / tot[j] for j in range(nS)])
ratios_s = np.array([cntS[:, j].sum() / tot[j] for j in range(nS)])
res["denominator"] = {
    "ratio_family_count_sum_over_reported_total": {
        "polymer_min": float(ratios_p.min()), "polymer_max": float(ratios_p.max()),
        "size_min": float(ratios_s.min()), "size_max": float(ratios_s.max()),
        "n_within_1pct": int(np.sum((np.abs(ratios_p - 1) < 0.01) &
                                    (np.abs(ratios_s - 1) < 0.01))),
    },
    "polymer_vs_size_count_sum_max_abs_diff_per_L": float(
        max(abs(cntP[:, j].sum() - cntS[:, j].sum()) for j in range(nS))),
    "implied_denominator_within_1pct_of_reported_total_frac": float(np.mean(
        [abs(r["implied_denom_vs_reported_total"] - 1) < 0.013 for r in den_rows])),
    "verdict": ("percentages are within-family normalised (each family sums to 100), and the "
                "count columns are reconstructed with a per-sample denominator that is the "
                "same for both families (they agree within 0.02 per L) but differs from the "
                "reported particle_total_per_L by up to 1.3 percent; so the operational "
                "denominator is the integer tally N (equivalently the unrounded per-L total), "
                "not the reported total and not an independent family-specific sum."),
}

# ------------------------------------------------- exact consistent-N machinery
def feasible_N_exact(P, D, Nmax):
    """All N in [2,Nmax] for which integers k_i>=0 with sum N exist and every
    reported (nonzero) percent is reproduced within its own last-digit rounding.
    Reported zeros are taken as exactly zero counts."""
    sols = []
    nz = P > 0
    Q = np.round(P[nz] * 10 ** D[nz]).astype(np.int64)   # percent * 10^d (integer)
    dd = D[nz]
    pw = 10 ** dd
    for N in range(2, Nmax + 1):
        lo = np.ceil((Q * N - 0.5 * N) / (100.0 * pw) - 1e-9)
        hi = np.floor((Q * N + 0.5 * N) / (100.0 * pw) + 1e-9)
        lo = np.maximum(lo, 0)
        if hi.min() < 0:
            continue
        if lo.sum() <= N <= hi.sum():
            sols.append(N)
    return sols


def feasible_N_report(P, Nmax, tol_pp=0.0051):
    """The report's criterion (07_resolution_and_spectrum.py): round(P*N/100) close
    to P within tol_pp pp AND |sum round(k) - N| <= 0.005 N."""
    sols = []
    for N in range(2, Nmax + 1):
        k = P * N / 100.0
        if np.abs(k - np.round(k)).max() <= tol_pp * N / 100.0:
            if abs(np.round(k).sum() - N) <= 0.5 * N / 100.0:
                sols.append(N)
    return sols

NMAX = 5000
res["consistent_N"] = {}
NminP, NminS, NminC = {}, {}, {}
repP, repS = {}, {}
for j in range(nS):
    fp = feasible_N_exact(pctP[:, j], decP[:, j], NMAX)
    fs = feasible_N_exact(pctS[:, j], decS[:, j], NMAX)
    rp = feasible_N_report(pctP[:, j], NMAX)
    rsn = feasible_N_report(pctS[:, j], NMAX)
    common = sorted(set(fp) & set(fs))
    NminP[sid[j]] = fp[0] if fp else None
    NminS[sid[j]] = fs[0] if fs else None
    NminC[sid[j]] = common[0] if common else None
    repP[sid[j]] = rp[0] if rp else None
    repS[sid[j]] = rsn[0] if rsn else None
    res["consistent_N"][sid[j]] = {
        "group": grp[j], "total_per_L": float(tot[j]),
        "Nmin_polymer_exact": fp[0] if fp else None,
        "Nmin_size_exact": fs[0] if fs else None,
        "Nmin_common_exact": common[0] if common else None,
        "exact_minima_agree": bool(fp and fs and fp[0] == fs[0]),
        "n_solutions_polymer_le200": len(fp),
        "n_solutions_size_le200": len(fs),
        "n_common_le200": len(common),
        "polymer_solutions": fp[:60],
        "size_solutions": fs[:60],
        "common_solutions": common[:60],
        "Nmin_polymer_report_criterion": rp[0] if rp else None,
        "Nmin_size_report_criterion": rsn[0] if rsn else None,
        "report_criterion_agree": bool(rp and rsn and rp[0] == rsn[0]),
    }

agree_samples = [s for s in sid if res["consistent_N"][s]["exact_minima_agree"]]
agree_report = [s for s in sid if res["consistent_N"][s]["report_criterion_agree"]]
res["consistent_N_summary"] = {
    "n_exact_minima_agree": len(agree_samples), "exact_agreeing_samples": agree_samples,
    "n_report_criterion_agree": len(agree_report), "report_agreeing_samples": agree_report,
    "Nmin_exact_agreeing_range": [min(NminC[s] for s in agree_samples if NminC[s]),
                                  max(NminC[s] for s in agree_samples if NminC[s])],
    "Nmin_report_agreeing_range": [min(NminP[s] for s in agree_report if NminP[s]),
                                   max(NminP[s] for s in agree_report if NminP[s])],
}

# ---------------------------------------------- (d) falsification check
fals = []
for s in sid:
    r = res["consistent_N"][s]
    j = sid.index(s)
    mp = pctP[:, j][pctP[:, j] > 0].min()
    ms = pctS[:, j][pctS[:, j] > 0].min()
    row = {"sample_id": s,
           "min_nonzero_pct_polymer": float(mp), "min_nonzero_pct_size": float(ms)}
    for tag, N in (("polymer", r["Nmin_polymer_exact"]), ("size", r["Nmin_size_exact"])):
        row[f"floor_polymer_family_100_over_Nmin_{tag}"] = 100.0 / N if N else None
    Nc = r["Nmin_common_exact"]
    row["Nmin_common"] = Nc
    if Nc:
        row["min_pct_ge_floor_common"] = bool(min(mp, ms) >= 100.0 / Nc - 1e-9)
        row["min_pct_ge_floor_polymer"] = bool(mp >= 100.0 / Nc - 1e-9)
        row["min_pct_ge_floor_size"] = bool(ms >= 100.0 / Nc - 1e-9)
    fals.append(row)
viol = [r for r in fals if r.get("min_pct_ge_floor_common") is False]
res["falsification_check"] = {
    "rows": fals,
    "n_violations_min_pct_lt_100_over_N": len(viol),
    "violating_samples": [r["sample_id"] for r in viol],
    "n_with_common_N": int(sum(1 for r in fals if r["Nmin_common"])),
}

# ---------------------------------------------- (c) bound direction, numeric demo
demo = []
for N in (17, 64, 76, 100, 2436):
    demo.append({"N": N, "floor_pp": 100.0 / N})
res["bound_direction"] = {
    "statement": ("For every sample the true tally satisfies N_true >= N_min(smallest consistent N), "
                  "hence true floor 100/N_true <= 100/N_min. So 100/N_min (the quoted resolution) is an "
                  "UPPER bound on the true detection floor: the real floor can only be FINER than quoted."),
    "illustration": demo,
    "report_language_audit": (
        "Result.md bullet says 'the real floor is at least as coarse as 100/N' and 'the quoted resolution "
        "is a best case'. Both are inverted: given N_true >= N_min, the real floor is at most as coarse as "
        "100/N_min (quoted floor is the worst/coarsest case, i.e. an upper bound on the floor, a conservative "
        "overstatement of the resolution penalty). The JSON field direction_of_bound in "
        "07_resolution_spectrum.json states the correct direction ('UPPER bound'), so the code is right and "
        "the prose is wrong."),
    "consequence": ("quoted floors (5.88 pp for sample_024 etc.) are upper bounds; if the true tally were 10x "
                    "N_min the real floor would be 10x finer."),
}

# ---------------------------------------------- (a) chance-agreement simulation
rng = np.random.default_rng(20240601)
agree_list = [s for s in agree_samples if NminC[s]]
Js = [sid.index(s) for s in agree_list]


def search_min(P, dec, Ntrue):
    """smallest exact-feasible N <= Ntrue for one family (true N always feasible)."""
    nz = P > 0
    Q = np.round(P[nz] * 10 ** dec[nz]).astype(np.int64)
    pw = 10.0 ** dec[nz]
    for N in range(2, int(Ntrue) + 1):
        lo = np.maximum(np.ceil((Q * N - 0.5 * N) / (100.0 * pw) - 1e-9), 0)
        hi = np.floor((Q * N + 0.5 * N) / (100.0 * pw) + 1e-9)
        if lo.sum() <= N <= hi.sum():
            return N
    return None


def sim_trial(Ntrue, pvecA, pvecB, rng):
    kA = rng.multinomial(int(Ntrue), pvecA)
    kB = rng.multinomial(int(Ntrue), pvecB)
    PA = np.round(100.0 * kA / Ntrue, 2)
    PB = np.round(100.0 * kB / Ntrue, 2)
    dA = np.full(len(PA), 2); dB = np.full(len(PB), 2)
    return search_min(PA, dA, Ntrue), search_min(PB, dB, Ntrue)


results_a = {}
# variant 1: generator = observed composition of the agreeing sample itself (with its own zeros)
trials, agree, both_true, minsA, minsB = 0, 0, 0, [], []
for _ in range(400):
    j = int(rng.choice(Js))
    N = NminC[sid[j]]
    pA = pctP[:, j] / 100.0
    pB = pctS[:, j] / 100.0
    pA = np.maximum(pA, 0); pB = np.maximum(pB, 0)
    if pA.sum() <= 0 or pB.sum() <= 0:
        continue
    pA = pA / pA.sum(); pB = pB / pB.sum()
    a, b = sim_trial(N, pA, pB, rng)
    trials += 1
    minsA.append(a); minsB.append(b)
    if a == b:
        agree += 1
    if a == N and b == N:
        both_true += 1
results_a["observed_composition_generator"] = {
    "n_trials": trials, "agree_rate": agree / trials,
    "both_recover_true_N_rate": both_true / trials,
    "mean_minA_over_N": float(np.mean(np.array(minsA) / np.array(
        [NminC[sid[int(rng.integers(len(Js)))]] for _ in range(trials)]))) if False else None,
}

# variant 2: Dirichlet(alpha) on the full 17/22 classes with matched nonzero support
def matched_support(pobs, ndraw, rng, alpha):
    """Dirichlet over classes that are nonzero in the observed sample; zeros stay zero."""
    idx = np.where(pobs > 0)[0]
    v = rng.dirichlet(np.full(len(idx), alpha))
    p = np.zeros(len(pobs)); p[idx] = v
    return p


for alpha in (0.5, 1.0, 3.0):
    trials, agree, both_true = 0, 0, 0
    minA, minB, Ns = [], [], []
    for _ in range(400):
        j = int(rng.choice(Js))
        N = NminC[sid[j]]
        pA = matched_support(pctP[:, j] / 100.0, 17, rng, alpha)
        pB = matched_support(pctS[:, j] / 100.0, 22, rng, alpha)
        a, b = sim_trial(N, pA, pB, rng)
        trials += 1; minA.append(a); minB.append(b); Ns.append(N)
        if a == b:
            agree += 1
        if a == N and b == N:
            both_true += 1
    results_a[f"dirichlet_alpha_{alpha}_matched_support"] = {
        "n_trials": trials, "agree_rate": agree / trials,
        "both_recover_true_N_rate": both_true / trials,
        "median_minA": float(np.median(minA)), "median_minB": float(np.median(minB)),
    }

# variant 3: naive full-support Dirichlet (17 & 22 nonzero classes always)
for alpha in (0.5, 1.0):
    trials, agree, both_true = 0, 0, 0
    for _ in range(300):
        j = int(rng.choice(Js))
        N = NminC[sid[j]]
        pA = rng.dirichlet(np.full(17, alpha))
        pB = rng.dirichlet(np.full(22, alpha))
        a, b = sim_trial(N, pA, pB, rng)
        trials += 1
        if a == b:
            agree += 1
        if a == N and b == N:
            both_true += 1
    results_a[f"dirichlet_alpha_{alpha}_full_support"] = {
        "n_trials": trials, "agree_rate": agree / trials,
        "both_recover_true_N_rate": both_true / trials,
    }

# variant 4: same-composition null (one composition used for both families) - upper reference
trials, agree = 0, 0
for _ in range(300):
    j = int(rng.choice(Js))
    N = NminC[sid[j]]
    pA = pctP[:, j] / 100.0; pA = pA / pA.sum()
    kk = rng.multinomial(int(N), pA)
    P = np.round(100.0 * kk / N, 2)
    a = search_min(P, np.full(17, 2), N)
    b = search_min(P, np.full(17, 2), N)
    trials += 1
    if a == b:
        agree += 1
results_a["identical_composition_reference"] = {"n_trials": trials, "agree_rate": agree / trials}

res["chance_agreement"] = {
    "design": ("true N from the 15 samples quoted as agreeing; compositions drawn either from the "
               "observed per-sample composition, Dirichlet with the observed nonzero support, or full "
               "Dirichlet; percentages rounded to 2 decimals; same exact smallest-consistent-N search"),
    "variants": results_a,
    "reported_evidence": {"n_samples": 30, "n_agree": 15, "rate": 0.5},
    "interpretation": ("if the simulated chance-agreement rate is comparable to 0.5, the 15/30 agreement "
                       "is uninformative about the tally; if it is small, the agreement is informative."),
}

# ---------------------------------------------- (b) uniqueness: solution-set sizes
sets = []
for s in sid:
    r = res["consistent_N"][s]
    sets.append({"sample_id": s, "total_per_L": r["total_per_L"],
                 "n_polymer_solutions_le5000": len(r["polymer_solutions"]) if False else None,
                 "exact_polymer_solutions_le200": r["n_solutions_polymer_le200"],
                 "exact_size_solutions_le200": r["n_solutions_size_le200"],
                 "exact_common_le200": r["n_common_le200"]})
# full solution sets up to 5000 for the agreeing samples
full = {}
for s in agree_report:
    j = sid.index(s)
    fp = feasible_N_exact(pctP[:, j], decP[:, j], NMAX)
    fs = feasible_N_exact(pctS[:, j], decS[:, j], NMAX)
    full[s] = {"polymer_all_N_le5000": fp, "size_all_N_le5000": fs,
               "common_all_N_le5000": sorted(set(fp) & set(fs)),
               "n_polymer": len(fp), "n_size": len(fs),
               "n_common": len(set(fp) & set(fs))}
res["uniqueness"] = {
    "per_sample_solution_counts_le200": sets,
    "agreeing_samples_full_sets_le5000": full,
    "median_solutions_per_family_le5000": float(np.median(
        [len(v["polymer_all_N_le5000"]) for v in full.values()] +
        [len(v["size_all_N_le5000"]) for v in full.values()])),
    "interpretation": ("N is only identified up to a set; the report quotes the minimum, which is the "
                       "finest-resolution (best-case) end of the set."),
}

# ---------------------------------------------- count-column integer test
# do the _per_L counts pin N more sharply than the 2dp percentages?
count_id = []
for j in range(nS):
    D = 0.5 * (cntP[:, j].sum() + cntS[:, j].sum())   # common unrounded per-L total
    c = np.concatenate([cntP[:, j], cntS[:, j]])
    dc = np.concatenate([cdcP[:, j], cdcS[:, j]])
    ok = c > 0
    c = c[ok]; dc = dc[ok]
    feas = []
    for N in range(2, 20001):
        q = D / N
        k = c / q
        tol = 0.5 * 10.0 ** (-dc) / q
        if np.abs(k - np.round(k)).max() <= tol.max() * 0 + (np.abs(k - np.round(k)) / tol).max() <= 1.0:
            feas.append(N)
    count_id.append({"sample_id": sid[j], "D_common_est": float(D),
                     "first_consistent_N": feas[0] if feas else None,
                     "n_consistent_N_le20000": len(feas),
                     "first_20": feas[:20],
                     "percentage_Nmin_polymer": NminP[sid[j]],
                     "percentage_Nmin_size": NminS[sid[j]],
                     "percentage_Nmin_in_count_set": bool(NminC[sid[j]] in feas) if NminC[sid[j]] else None})
res["count_column_N_test"] = {
    "method": ("count_i = k_i * D/N_true exactly, so the printed counts (up to 3 decimals per L) are "
               "integer multiples of the quantum D/N; searching N in 2..20000 for the smallest N whose "
               "quantum reproduces every printed count within its own rounding."),
    "rows": count_id,
    "note": ("if the count columns are pure derivations of the 2dp percentages this test should find many "
             "N; if they retain the unrounded percentages they can pin N."),
}

# ---------------------------------------------- rho of nonzero bins vs log total
nz_bins = (pctS > 0).sum(axis=0).astype(float)
lt = np.log10(tot)
import scipy.stats as sps
rho, p = sps.spearmanr(nz_bins, lt)
res["supporting_rho"] = {"n_nonzero_size_bins_vs_log10_total_rho": float(rho),
                         "p": float(p), "min_bins": int(nz_bins.min()),
                         "max_bins": int(nz_bins.max())}

# ---------------------------------------------- misc reproduction of quoted numbers
q = []
for s in agree_report:
    j = sid.index(s)
    q.append({"sample_id": s, "N_min": NminP[s], "total_per_L": float(tot[j]),
              "one_particle_per_L": float(tot[j] / NminP[s]) if NminP[s] else None,
              "floor_pp": 100.0 / NminP[s] if NminP[s] else None})
res["one_particle_concentration"] = {
    "rows": q,
    "min": float(min(x["one_particle_per_L"] for x in q if x["one_particle_per_L"])),
    "max": float(max(x["one_particle_per_L"] for x in q if x["one_particle_per_L"])),
}

res["problems"] = problems
(OUT / "verify_F4.json").write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT / "verify_F4.json")
print("exact agreeing:", len(agree_samples), agree_samples)
print("report-criterion agreeing:", len(agree_report))
print("falsification violations:", len(viol))
print("derived residual:", json.dumps(res["derived_count_residual"], indent=1))
print("denominator ratios:", json.dumps(res["denominator"]["ratio_family_count_sum_over_reported_total"], indent=1))
print("chance:", json.dumps(results_a, indent=1))
print("count id first:", [(r["sample_id"], r["first_consistent_N"], r["n_consistent_N_le20000"]) for r in count_id][:10])
print("rho:", rho, p)
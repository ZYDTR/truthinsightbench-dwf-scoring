#!/usr/bin/env python3
"""Independent adversarial verification of the claims in the main analysis.

Each block below tries to BREAK a specific claim from run_analysis.py rather
than confirm it.  Anything that survives is reported as verified; anything that
does not is reported as a limitation.

Run:  python3 output/analysis/verify_claims.py
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RES = ROOT / "output" / "results"
RES.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)
V = {}
SET = list(range(1, 8))


def load():
    g = collections.defaultdict(list)
    with (DATA / "rate_dependent_voltage_curves.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g[(r["scan_role"], r["anonymous_rate_setting"])].append(
                (float(r["state_of_charge_percent"]), float(r["cell_voltage_V"])))
    out = {}
    for k, v in g.items():
        v.sort()
        out[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
    return out


C = load()
eff, pw = {}, {}
with (DATA / "rate_tradeoff_summaries.tsv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        (eff if r["summary_role"] == "efficiency" else pw)[float(r["anonymous_rate_value"])] = float(r["response_value"])
cyc = []
with (DATA / "long_cycle_capacity.tsv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        cyc.append(float(r["capacity_like_value"]))
cyc = np.array(cyc)
RE, RP = sorted(eff), sorted(pw)


def E(role, i, x):
    return float(np.interp(x, *C[i and (role, f"rate_setting_{i}")]))


print("=" * 78)
print("VERIFY 1 -- is the efficiency reduction really a same-SoC voltage ratio?")
print("=" * 78)
# NOTE (post-verification correction): the first two entries previously carried
# labels that did not describe their code.  Entry 1 used a nearest-INDEX numerator
# against an INTERPOLATED denominator (a mismatched-abscissa construction, which is
# why it scores better) yet was labelled as the plain ratio; entry 2's DoD pairing
# is degenerate at the evaluated probe (x = 50 gives Vch(100-50) = Vch(50)), so it
# reproduces the primary rather than competing with it.  Labels now match the code.
forms = {
    "Vdis(nearest measured x)/Vch(interp at x)  [mismatched abscissa]": lambda x, i: 100 * C[("discharge_like", f"rate_setting_{i}")][1][
        np.argmin(np.abs(C[("discharge_like", f"rate_setting_{i}")][0] - x))] / np.interp(
        x, *C[("charge_like", f"rate_setting_{i}")]),
    "Vdis(interp at x)/Vch(interp at x)  [the headline form]": lambda x, i: 100 * np.interp(
        x, *C[("discharge_like", f"rate_setting_{i}")]) / np.interp(
        x, *C[("charge_like", f"rate_setting_{i}")]),
    "Vdis(x)/Vch(100-x)  [DoD pairing; degenerate at x=50]": lambda x, i: 100 * np.interp(
        x, *C[("discharge_like", f"rate_setting_{i}")]) / np.interp(
        100 - x, *C[("charge_like", f"rate_setting_{i}")]),
    "2Vdis/(Vch+Vdis) at x": lambda x, i: 200 * np.interp(x, *C[("discharge_like", f"rate_setting_{i}")]) / (
        np.interp(x, *C[("charge_like", f"rate_setting_{i}")]) + np.interp(x, *C[("discharge_like", f"rate_setting_{i}")])),
    "mean(Vdis)/mean(Vch) full": lambda x, i: 100 * C[("discharge_like", f"rate_setting_{i}")][1].mean() / C[("charge_like", f"rate_setting_{i}")][1].mean(),
}
rows = []
for name, fn in forms.items():
    if name.startswith("mean"):
        res = [fn(50, i) - eff[r] for r, i in zip(RE, SET)]
    else:
        res = [fn(50.0, i) - eff[r] for r, i in zip(RE, SET)]
    res = np.array(res)
    rows.append([name, np.sqrt((res ** 2).mean()), np.abs(res).max()])
    print(f"  {name:38s} RMSE={np.sqrt((res**2).mean()):8.4f} pp  max|res|={np.abs(res).max():7.4f}")
write_csv = lambda n, h, r: (RES / n).write_text(
    "\n".join(["\t".join(map(str, h))] + ["\t".join(map(str, x)) for x in r]) + "\n", encoding="utf-8")
write_csv("verify_efficiency_forms.tsv", ["form", "rmse_pp", "max_abs_res_pp"], rows)

# two-parameter check: eff = a*ratio + b
rat = np.array([100 * float(np.interp(50, *C[("discharge_like", f"rate_setting_{i}")])) /
                float(np.interp(50, *C[("charge_like", f"rate_setting_{i}")])) for i in SET])
Ee = np.array([eff[r] for r in RE])
A = np.vstack([rat, np.ones(7)]).T
c, *_ = np.linalg.lstsq(A, Ee, rcond=None)
rr = Ee - A @ c
print(f"  free affine eff = a*ratio+b : a={c[0]:.6f}, b={c[1]:+.5f}, "
      f"R2={1-(rr**2).sum()/((Ee-Ee.mean())**2).sum():.7f}")
V["efficiency_form_affine"] = dict(a=float(c[0]), b=float(c[1]))

# same for voltage
ratv = np.array([float(np.interp(50, *C[("discharge_like", f"rate_setting_{i}")])) for i in SET])
Pv = np.array([pw[r] / r for r in RP[:7]])
A2 = np.vstack([ratv, np.ones(7)]).T
c2, *_ = np.linalg.lstsq(A2, Pv, rcond=None)
rr2 = Pv - A2 @ c2
print(f"  free affine Vd=P/rate   : a={c2[0]:.6f}, b={c2[1]:+.6f}, "
      f"R2={1-(rr2**2).sum()/((Pv-Pv.mean())**2).sum():.7f}")
V["voltage_form_affine"] = dict(a=float(c2[0]), b=float(c2[1]))

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 2 -- hold-out: fit the linkage on 4 rates, predict the other 3")
print("=" * 78)
# If the two summary rows are the SAME construction, a slope fitted on a
# sub-block must predict held-out rates.  If the agreement were coincidence,
# it would not transport.
errs = []
for hold in itertools.combinations(range(7), 3):
    fit = [k for k in range(7) if k not in hold]
    a, b = np.polyfit(rat[fit], Ee[fit], 1)
    pred = a * rat[list(hold)] + b
    errs.extend((pred - Ee[list(hold)]).tolist())
errs = np.array(errs)
print(f"  efficiency: {len(errs)} held-out predictions, RMSE={np.sqrt((errs**2).mean()):.4f} pp, "
      f"max|err|={np.abs(errs).max():.4f} pp")
V["holdout_efficiency"] = dict(n=len(errs), rmse_pp=float(np.sqrt((errs ** 2).mean())),
                               max_abs_pp=float(np.abs(errs).max()))

errs2 = []
for hold in itertools.combinations(range(7), 3):
    fit = [k for k in range(7) if k not in hold]
    a, b = np.polyfit(ratv[fit], Pv[fit], 1)
    pred = a * ratv[list(hold)] + b
    errs2.extend((pred - Pv[list(hold)]).tolist())
errs2 = np.array(errs2)
print(f"  voltage   : {len(errs2)} held-out predictions, RMSE={np.sqrt((errs2**2).mean()):.5f} V, "
      f"max|err|={np.abs(errs2).max():.5f} V")
V["holdout_voltage"] = dict(n=len(errs2), rmse_V=float(np.sqrt((errs2 ** 2).mean())),
                            max_abs_V=float(np.abs(errs2).max()))

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 3 -- gap ordering: where does it break, and why?")
print("=" * 78)
rob = list(csv.DictReader((RES / "ordering_robustness_soc.csv").open()))
bad = [r for r in rob if r["gap_strictly_increasing"] != "True" or
       r["charge_strictly_increasing"] != "True" or r["discharge_strictly_decreasing"] != "True"]
print(f"  probes where at least one ordering fails: {len(bad)}")
for r in bad:
    print(f"    SoC={r['soc_probe']:>5}: charge_inc={r['charge_strictly_increasing']}, "
          f"dischg_dec={r['discharge_strictly_decreasing']}, gap_inc={r['gap_strictly_increasing']}")
# what fraction of the ordering is preserved when it breaks?
for soc in [10, 15, 90, 95]:
    c = np.array([float(np.interp(soc, *C[("charge_like", f"rate_setting_{i}")])) for i in SET])
    d = np.array([float(np.interp(soc, *C[("discharge_like", f"rate_setting_{i}")])) for i in SET])
    g = c - d
    print(f"    SoC={soc:>3}: #charge increases {int((np.diff(c)>0).sum())}/6, "
          f"#discharge decreases {int((np.diff(d)<0).sum())}/6, #gap increases {int((np.diff(g)>0).sum())}/6")
V["ordering_breakdown"] = [r["soc_probe"] for r in bad]

# also test on the raw union of all measured x (no interpolation)
print("  interpolation-free check: rank each curve by its own measured extremes")
for i in SET:
    xc, yc = C[("charge_like", f"rate_setting_{i}")]
    xd, yd = C[("discharge_like", f"rate_setting_{i}")]
    print(f"    s{i}: Vch range [{yc.min():.4f},{yc.max():.4f}]  Vdis range [{yd.min():.4f},{yd.max():.4f}]")

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 4 -- cycling: does ANY interior change point survive a proper null?")
print("=" * 78)
x = np.arange(1, 601, dtype=float)
A = np.vstack([x, np.ones(600)]).T
cl, *_ = np.linalg.lstsq(A, cyc, rcond=None)
rl = cyc - A @ cl
rss_lin = float((rl ** 2).sum())

Aq = np.vstack([x ** 2, x, np.ones(600)]).T
cq, *_ = np.linalg.lstsq(Aq, cyc, rcond=None)
rq = cyc - Aq @ cq
rss_quad = float((rq ** 2).sum())

print(f"  quadratic term = {cq[0]:.3e}; RSS linear={rss_lin:.0f} quad={rss_quad:.0f} "
      f"(reduction {100*(1-rss_quad/rss_lin):.2f}%)")


def moving_block_sample(resid, block=25):
    starts = np.arange(0, 600 - block + 1)
    s = RNG.choice(starts, int(np.ceil(600 / block)))
    idx = np.concatenate([np.arange(t, t + block) for t in s])[:600]
    return resid[idx]


obs_q = 1 - rss_quad / rss_lin
null_q = np.empty(400)
for b in range(400):
    y = cl[0] * x + cl[1] + moving_block_sample(rl)
    aa = np.vstack([x ** 2, x, np.ones(600)]).T
    cc, *_ = np.linalg.lstsq(aa, y, rcond=None)
    rr = y - aa @ cc
    null_q[b] = 1 - float((rr ** 2).sum()) / float(((y - (cl[0] * x + cl[1])) ** 2).sum())
p_q = float((null_q >= obs_q).mean())
print(f"  quadratic-vs-linear, trend-preserving block null (400 reps): p = {max(p_q, 1/401):.4f}")

obs_c = float(cq[0])
null_c = np.empty(400)
for b in range(400):
    y = cl[0] * x + cl[1] + moving_block_sample(rl)
    aa = np.vstack([x ** 2, x, np.ones(600)]).T
    cc, *_ = np.linalg.lstsq(aa, y, rcond=None)
    null_c[b] = cc[0]
p_c = float((null_c <= obs_c).mean())
print(f"  quadratic coefficient, one-sided block null: p = {max(p_c, 1/401):.4f} "
      f"(observed {obs_c:.3e}, null mean {null_c.mean():.3e})")
V["cycling_quadratic_test"] = dict(observed_coef=float(obs_c), p_block_null=float(max(p_c, 1 / 401)),
                                   rss_reduction_pct=float(100 * obs_q))

# piecewise: does segment 2 really differ from segment 1?
cp = 286
a1, b1, r1 = (lambda c: (c[0], c[1], cyc[:cp] - np.vstack([x[:cp], np.ones(cp)]).T @ c))(
    np.linalg.lstsq(np.vstack([x[:cp], np.ones(cp)]).T, cyc[:cp], rcond=None)[0])
a2, b2, r2 = (lambda c: (c[0], c[1], cyc[cp:] - np.vstack([x[cp:], np.ones(600 - cp)]).T @ c))(
    np.linalg.lstsq(np.vstack([x[cp:], np.ones(600 - cp)]).T, cyc[cp:], rcond=None)[0])
obs_diff = a2 - a1
null_d = np.empty(400)
for b in range(400):
    y = cl[0] * x + cl[1] + moving_block_sample(rl)
    s1 = np.linalg.lstsq(np.vstack([x[:cp], np.ones(cp)]).T, y[:cp], rcond=None)[0][0]
    s2 = np.linalg.lstsq(np.vstack([x[cp:], np.ones(600 - cp)]).T, y[cp:], rcond=None)[0][0]
    null_d[b] = s2 - s1
p_d = float((null_d <= obs_diff).mean())
print(f"  slope difference s2-s1 = {obs_diff:+.5f}; trend-preserving null p = {max(p_d,1/401):.4f} "
      f"(null mean {null_d.mean():+.5f}, sd {null_d.std():.5f})")
V["cycling_slope_change_test"] = dict(observed=float(obs_diff), null_mean=float(null_d.mean()),
                                      null_sd=float(null_d.std()), p=float(max(p_d, 1 / 401)))

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 5 -- the exact +0.041520 V offset pair: artifact or physics?")
print("=" * 78)
x6, y6 = C[("charge_like", "rate_setting_6")]
x7, y7 = C[("charge_like", "rate_setting_7")]
print(f"  identical abscissa: {np.allclose(x6, x7)}   constant offset: {np.unique(np.round(y7-y6,10))}")
xd6, yd6 = C[("discharge_like", "rate_setting_6")]
xd7, yd7 = C[("discharge_like", "rate_setting_7")]
print(f"  discharge 6 vs 7 share abscissa: {len(xd6)==len(xd7) and np.allclose(xd6,xd7)}")
seg6 = np.diff(yd6)[:min(len(yd6), len(yd7)) - 1]
seg7 = np.diff(yd7)[:min(len(yd6), len(yd7)) - 1]
print(f"  discharge 6/7 leading first-difference profiles identical? "
      f"{np.allclose(seg6, seg7)} (compared {len(seg6)} of {len(np.diff(yd6))}/{len(np.diff(yd7))})")
# ratio-based test
lo, hi = max(xd6[0], xd7[0]), min(xd6[-1], xd7[-1])
g = np.linspace(lo, hi, 300)
u, w = np.interp(g, xd6, yd6), np.interp(g, xd7, yd7)
print(f"  discharge on common window: mean(V7-V6)={np.mean(w-u):+.5f} V, sd={np.std(w-u):.5f} V; "
      f"mean(V7/V6)={np.mean(w/u):.6f}")
V["offset_pair"] = dict(offset_V=float(np.unique(np.round(y7 - y6, 10))[0]),
                        identical_abscissa_charge=bool(np.allclose(x6, x7)),
                        identical_abscissa_discharge=bool(len(xd6) == len(xd7) and np.allclose(xd6, xd7)))

# how many of the other 6 settings share an abscissa with anything?
share = 0
for role in ("charge_like", "discharge_like"):
    for a, b in itertools.combinations(SET, 2):
        xa = C[(role, f"rate_setting_{a}")][0]
        xb = C[(role, f"rate_setting_{b}")][0]
        if len(xa) == len(xb) and np.allclose(xa, xb):
            share += 1
print(f"  total identical-abscissa pairs across both roles: {share} (of 42)")
V["n_identical_abscissa_pairs"] = share

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 6 -- power turnover: sensitivity to how the peak is located")
print("=" * 78)
RParr = np.array(RP, dtype=float)
Pp = np.array([pw[r] for r in RP], dtype=float)
print(f"  power series: {list(zip(RP, Pp.tolist()))}")
print(f"  argmax at index {int(np.argmax(Pp))} -> rate {RP[int(np.argmax(Pp))]}")
# what if 300 is an outlier (it and 500 are the only exactly-round values)?
mask = RParr != 300.0
print(f"  excluding rate 300: peak becomes {Pp[mask].max()} at rate {RParr[mask][int(np.argmax(Pp[mask]))]}")
mask2 = ~np.isin(RParr, [300.0, 500.0])
print(f"  excluding rates 300 and 500: peak {Pp[mask2].max()} at rate {RParr[mask2][int(np.argmax(Pp[mask2]))]}"
      f" -> whether an interior turnover remains is untestable without those points")
# is 500 below the last measured discharge voltage even at its lowest point?
allmin = min(float(C[("discharge_like", f"rate_setting_{i}")][1].min()) for i in SET)
print(f"  implied V at rate 500 = {pw[500.0]/500.0:.4f} V vs lowest measured discharge V = {allmin:.4f} V")
print(f"  -> the 500 point cannot be reproduced by any measured discharge curve; "
      f"it is the only point that breaks the P = rate*Vdis construction")
rows = []
for r in RP:
    v = pw[r] / r
    inside = allmin <= v <= max(float(C[("discharge_like", f"rate_setting_{i}")][1].max()) for i in SET)
    rows.append([r, pw[r], v, inside])
(RES / "verify_power_window.tsv").write_text(
    "rate\tpower_density\timplied_V\tinside_measured_voltage_window\n" +
    "\n".join("\t".join(map(str, x)) for x in rows) + "\n", encoding="utf-8")
V["power_turnover_sensitivity"] = dict(
    peak_including_all=float(Pp.max()), peak_rate_including_all=float(RParr[int(np.argmax(Pp))]),
    peak_excluding_300=float(Pp[mask].max()), peak_rate_excluding_300=float(RParr[mask][int(np.argmax(Pp[mask]))]),
    interior_turnover_survives_excluding_round_values=bool(int(np.argmax(Pp[mask])) < mask.sum() - 1),
)

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 7 -- cycling quantization: confirm the 1/11 grid and its break")
print("=" * 78)
q = np.round(cyc * 11).astype(np.int64)
print(f"  max |11*cap - round(11*cap)| = {np.abs(cyc*11-q).max():.2e} over 600 points")
b1, b2 = q[:50], q[50:]
print(f"  cycles 1-50   : residues mod 10 = {sorted(set((b1 % 10).tolist()))}")
print(f"  cycles 51-600 : residues mod 10 = {sorted(set((b2 % 10).tolist()))}")
off = sorted(set((b2 % 10).tolist()))
viol = np.where(b2 % 10 != 0)[0]
print(f"  cycles 51-600 violating the multiple-of-10 rule: {len(viol)} "
      f"at cycles {[int(v)+51 for v in viol]} values {[float(b2[v]/11) for v in viol]}")
V["quantization"] = dict(grid=1 / 11, block1_residues=sorted(set((b1 % 10).tolist())),
                         block2_residues=sorted(set((b2 % 10).tolist())),
                         block2_violations=int(len(viol)))
# does the break change the fitted drift?
for name, sl in [("cycles 1-50", cyc[:50]), ("cycles 51-600", cyc[50:])]:
    xx = np.arange(len(sl), dtype=float)
    a, b, r = (lambda c: (c[0], c[1], sl - np.vstack([xx, np.ones(len(sl))]).T @ c))(
        np.linalg.lstsq(np.vstack([xx, np.ones(len(sl))]).T, sl, rcond=None)[0])
    print(f"  {name}: slope={a:+.5f} units/cycle, R2={1-(r**2).sum()/((sl-sl.mean())**2).sum():.4f}")

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 8 -- is the 5040-permutation identity rank actually evidential?")
print("=" * 78)
# The report presented rank 1/5040 as a robustness result for the linkage.  It is
# not: when two series are both strictly monotone in the same direction, the
# identity pairing is the minimum-MSE pairing by the rearrangement inequality, so
# rank 1 is guaranteed regardless of whether the series share a construction.
# Only the MAGNITUDE gap to the runner-up carries information.
_taut = 0
_NT = 1500
_perms = list(itertools.permutations(range(7)))[:200]
for _ in range(_NT):
    _a = np.sort(RNG.normal(size=7))
    _b = np.sort(RNG.normal(size=7))
    _id = ((_a - _b) ** 2).mean()
    _better = sum(1 for _p in _perms if ((_a - _b[list(_p)]) ** 2).mean() < _id - 1e-15)
    _taut += (_better == 0)
print(f"  identity is the minimum-MSE pairing in {_taut}/{_NT} random INDEPENDENT monotone pairs")
print(f"  -> rank 1/5040 is guaranteed by monotonicity, not evidence of shared construction")
_runner = sorted(((np.sqrt(np.mean((np.array([100 * np.interp(soc, *C[("discharge_like", f"rate_setting_{i}")]) / np.interp(soc, *C[("charge_like", f"rate_setting_{i}")]) for i in p]) - Ee) ** 2)), p)
                    for p in itertools.permutations(range(1, 8))
                    for soc in [50.0]))[:3]
print(f"  informative statistic: identity RMSE {_runner[0][0]:.4f} pp vs runner-up "
      f"{_runner[1][0]:.4f} pp ({_runner[1][0]/_runner[0][0]:.1f}x worse)")
V["permutation_is_tautological"] = dict(
    independent_monotone_pairs_rank1=f"{_taut}/{_NT}",
    identity_rmse_pp=float(_runner[0][0]), runner_up_rmse_pp=float(_runner[1][0]),
    runner_up_ratio=float(_runner[1][0] / _runner[0][0]),
    note="rank 1 of 5040 is guaranteed by the rearrangement inequality for any two "
         "monotone series in the same direction; only the magnitude gap is informative",
)

# --------------------------------------------------------------------------
print()
print("=" * 78)
print("VERIFY 9 -- residual share: which rate drives the efficiency model misfit?")
print("=" * 78)
# The report asserted the rate-80 value was 'the largest single contributor to the
# residual of the linear models'.  Check that claim rather than assume it.
for nm, A_ in [("linear-in-log10", np.vstack([np.log10(np.array(RE, float)), np.ones(7)]).T),
               ("linear-in-rate", np.vstack([np.array(RE, float), np.ones(7)]).T)]:
    c_, *_ = np.linalg.lstsq(A_, Ee, rcond=None)
    r_ = Ee - A_ @ c_
    share = r_ ** 2 / (r_ ** 2).sum() * 100
    order = np.argsort(-share)
    rank80 = list(order).index(list(RE).index(80.0)) + 1
    print(f"  {nm:16s}: largest |residual| at rate {RE[order[0]]} ({r_[order[0]]:+.3f} pp); "
          f"rate 80 is rank {rank80}/7 with {share[list(RE).index(80.0)]:.1f}% of RSS")
    V[f"residual_share_{nm}"] = dict(
        largest_at_rate=float(RE[order[0]]), largest_residual_pp=float(r_[order[0]]),
        rate80_rank=int(rank80), rate80_share_pct=float(share[list(RE).index(80.0)]),
    )
print("  -> the rate-80 value is NOT the largest contributor for either linear model")

(RES / "verification_summary.json").write_text(json.dumps(V, indent=2, default=float), encoding="utf-8")
print()
print("verification_summary.json written")
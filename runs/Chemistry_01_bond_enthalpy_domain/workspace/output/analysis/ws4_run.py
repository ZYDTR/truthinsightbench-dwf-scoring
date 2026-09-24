"""WS4 final run: tail census, descriptor test, crowded flag, weak-bond regions,
direction of failure, artifact + figure."""
from __future__ import annotations

import sys

sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')

import json
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger

import common
import ws4_failures as w
from common import CHANNELS

RDLogger.DisableLog("rdApp.*")

PROJECT = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
df = w.analysis_table()
N = len(df)
OUT = {}

# ==========================================================================
# 1. TAIL CENSUS
# ==========================================================================
census = {}
for c in CHANNELS:
    ae = df[c + '_ae']
    entry = {"n_gt5": int((ae > 5).sum()), "n_gt10": int((ae > 10).sum()),
             "n_gt15": int((ae > 15).sum()),
             "max_abs_err": round(float(ae.max()), 3),
             "frac_gt5": round(float((ae > 5).mean()), 4),
             "frac_gt10": round(float((ae > 10).mean()), 4)}
    for thr in (5, 10, 15):
        sub = df.loc[ae > thr].sort_values(ae.name.replace('_ae', '_err'), key=lambda s: s.abs(), ascending=False)
        entry[f"obs_gt{thr}"] = [{
            "observation_id": r.observation_id,
            "parent_smiles": r.parent_structure_smiles,
            "fragment_a": r.fragment_a_smiles,
            "fragment_b": r.fragment_b_smiles,
            "exp_bde": round(float(r.experimental_bde_kcal_mol), 2),
            "estimate": round(float(getattr(r, common.CHAN_COL[c])), 2),
            "signed_err": round(float(getattr(r, c + '_err')), 2),
            "abs_err": round(float(getattr(r, c + '_ae')), 2),
            "bond_class": r.bond_class,
            "hac": int(r.p_hac), "n_rings": int(r.p_n_rings),
            "n_arom_rings": int(r.p_n_arom_rings), "n_rotb": int(r.p_n_rotb),
            "same_frag": bool(r.same_frag_canon),
            "in_db_row": bool(r.in_db_row),
        } for r in sub.itertuples()]
    census[c] = entry
OUT["tail_census"] = census

union_any5 = np.any([(df[c + '_ae'] > 5).values for c in CHANNELS], axis=0)
union_any10 = np.any([(df[c + '_ae'] > 10).values for c in CHANNELS], axis=0)
union_any15 = np.any([(df[c + '_ae'] > 15).values for c in CHANNELS], axis=0)
all3_5 = np.all([(df[c + '_ae'] > 5).values for c in CHANNELS], axis=0)
all3_10 = np.all([(df[c + '_ae'] > 10).values for c in CHANNELS], axis=0)
all3_15 = np.all([(df[c + '_ae'] > 15).values for c in CHANNELS], axis=0)
OUT["tail_census"]["union"] = {
    "n_any_channel_gt5": int(union_any5.sum()), "n_any_channel_gt10": int(union_any10.sum()),
    "n_any_channel_gt15": int(union_any15.sum()),
    "n_all3_channels_gt5": int(all3_5.sum()), "n_all3_channels_gt10": int(all3_10.sum()),
    "n_all3_channels_gt15": int(all3_15.sum()),
    "worst_case_rows_gt10_union": [{
        "observation_id": r.observation_id, "parent_smiles": r.parent_structure_smiles,
        "fragment_a": r.fragment_a_smiles, "fragment_b": r.fragment_b_smiles,
        "exp_bde": float(r.experimental_bde_kcal_mol),
        "err_quantum": round(float(r.quantum_err), 2),
        "err_learned": round(float(r.learned_err), 2),
        "err_reference": round(float(r.reference_err), 2),
        "bond_class": r.bond_class, "hac": int(r.p_hac), "n_rings": int(r.p_n_rings),
        "n_arom_rings": int(r.p_n_arom_rings), "same_frag": bool(r.same_frag_canon),
        "crowded": bool(r.crowded), "in_db_row": bool(r.in_db_row),
    } for r in df.loc[union_any10].sort_values('experimental_bde_kcal_mol').itertuples()],
}

# ==========================================================================
# 2. DESCRIPTOR TEST: failure set vs rest, per descriptor, per channel
# ==========================================================================
desc_cols = ["p_hac", "p_n_rings", "p_n_arom_rings", "p_n_rotb", "p_n_bridge",
             "p_n_spiro", "p_n_fused", "p_n_hetero", "fsp3" if "fsp3" in df else "p_fsp3"]

def desc_test(frame, fail_mask, cols):
    rows = []
    for d in cols:
        a = frame.loc[fail_mask, d].astype(float).values
        b = frame.loc[~fail_mask, d].astype(float).values
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        rows.append({"descriptor": d, "fail_n": len(a), "rest_n": len(b),
                     "fail_mean": round(float(a.mean()), 4),
                     "fail_median": round(float(np.median(a)), 4),
                     "rest_mean": round(float(b.mean()), 4),
                     "rest_median": round(float(np.median(b)), 4),
                     "cliffs_delta": round(common.cliffs_delta(a, b), 4),
                     "mannwhitney_p": float(p)})
    return rows

OUT["descriptor_test"] = {}
for c in CHANNELS:
    fail = (df[c + '_ae'] > 5).values
    entry = {"threshold": 5, "n_fail": int(fail.sum()), "n_rest": int((~fail).sum()),
             "numeric": desc_test(df, fail, desc_cols)}
    # binary flags
    bins = []
    for f in ["same_frag_canon", "both_rad_in_ring", "any_rad_arom", "crowded", "in_db_row"]:
        flag = df[f].astype(bool).values
        tab = np.array([[int((flag & fail).sum()), int((flag & ~fail).sum())],
                        [int((~flag & fail).sum()), int((~flag & ~fail).sum())]])
        orr, p = stats.fisher_exact(tab)
        bins.append({"flag": f, "n_flag": int(flag.sum()),
                     "fail_rate_flag": round(float(fail[flag].mean()), 4),
                     "fail_rate_noflag": round(float(fail[~flag].mean()), 4),
                     "odds_ratio": round(float(orr), 4) if np.isfinite(orr) else None,
                     "fisher_p": float(p)})
    entry["binary"] = bins
    # 16-descriptor-family correction
    entry["n_tests_numeric"] = len(entry["numeric"])
    OUT["descriptor_test"][c] = entry

# Benjamini-Hochberg over the numeric descriptors of each channel
for c in CHANNELS:
    ps = np.array([r["mannwhitney_p"] for r in OUT["descriptor_test"][c]["numeric"]])
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, ps[idx] * m / k)
        q[idx] = prev
    for r, qq in zip(OUT["descriptor_test"][c]["numeric"], q):
        r["bh_q"] = float(qq)

# ==========================================================================
# 3. CROWDED / STERIC FLAG
# ==========================================================================
crowded_tbl = {"definition": "both fragment radical centres in rings OR parent ring count >= 3",
               "n_crowded": int(df.crowded.sum()), "n_not_crowded": int((~df.crowded).sum()), "channels": {}}
for c in CHANNELS:
    a = df.loc[df.crowded, c + '_ae'].values
    b = df.loc[~df.crowded, c + '_ae'].values
    crowded_tbl["channels"][c] = {
        "mae_crowded": round(float(a.mean()), 4), "n_crowded": len(a),
        "mae_not_crowded": round(float(b.mean()), 4), "n_not_crowded": len(b),
        "delta_mae": round(float(a.mean() - b.mean()), 4),
        "cliffs_delta": round(common.cliffs_delta(a, b), 4),
        "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative='two-sided')[1]),
        "bias_crowded": round(float(df.loc[df.crowded, c + '_err'].mean()), 4),
        "bias_not_crowded": round(float(df.loc[~df.crowded, c + '_err'].mean()), 4),
    }
# sub-flags
crowded_tbl["subflags"] = {}
for name, flag in [("both_rad_in_ring", df.both_rad_in_ring.astype(bool)),
                   ("parent_ring_count_ge3", df.p_n_rings >= 3),
                   ("parent_arom_rings_ge2", df.p_n_arom_rings >= 2),
                   ("fused_or_bridged", (df.p_n_fused > 0) | (df.p_n_bridge > 0) | (df.p_n_spiro > 0))]:
    sub = {"n_flag": int(flag.sum())}
    for c in CHANNELS:
        a = df.loc[flag, c + '_ae'].values
        b = df.loc[~flag, c + '_ae'].values
        sub[c] = {"mae_flag": round(float(a.mean()), 4), "n_flag": int(len(a)),
                  "mae_noflag": round(float(b.mean()), 4), "n_noflag": int(len(b)),
                  "delta_mae": round(float(a.mean() - b.mean()), 4),
                  "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative='two-sided')[1])}
    crowded_tbl["subflags"][name] = sub
# confounding control: stratify by in_db_row
crowded_tbl["stratified_by_db_membership"] = {}
for lbl, frame in [("in_db_row", df[df.in_db_row]), ("not_in_db_row", df[~df.in_db_row])]:
    blk = {"n": len(frame), "n_crowded": int(frame.crowded.sum())}
    for c in CHANNELS:
        a = frame.loc[frame.crowded, c + '_ae'].values
        b = frame.loc[~frame.crowded, c + '_ae'].values
        blk[c] = {"mae_crowded": round(float(a.mean()), 4), "n_crowded": int(len(a)),
                  "mae_not_crowded": round(float(b.mean()), 4), "n_not_crowded": int(len(b)),
                  "delta_mae": round(float(a.mean() - b.mean()), 4),
                  "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative='two-sided')[1])
                  if len(a) >= 2 and len(b) >= 2 else None}
    crowded_tbl["stratified_by_db_membership"][lbl] = blk
crowded_tbl["confound_note"] = (
    "93 of 95 crowded rows are NOT exact DB rows (crowded x in_db_row = 93/2); the crowded "
    "flag is almost a re-label of the out-of-database set.  The in-DB stratum has only 2 "
    "crowded rows so the stratified test is uninformative there.")
crowded_tbl["cross_tab_crowded_x_in_db"] = {
    "crowded_and_in_db": int((df.crowded & df.in_db_row).sum()),
    "crowded_and_not_in_db": int((df.crowded & ~df.in_db_row).sum()),
    "not_crowded_and_in_db": int((~df.crowded & df.in_db_row).sum()),
    "not_crowded_and_not_in_db": int((~df.crowded & ~df.in_db_row).sum())}
OUT["crowded_flag"] = crowded_tbl

# ==========================================================================
# 4. WEAK-BOND REGION TABLE
# ==========================================================================
regions = {}
for label, mask in [("lt60", df.experimental_bde_kcal_mol < 60),
                    ("lt70", df.experimental_bde_kcal_mol < 70),
                    ("lt80", df.experimental_bde_kcal_mol < 80),
                    ("ge80", df.experimental_bde_kcal_mol >= 80),
                    ("all", pd.Series(True, index=df.index))]:
    sub = df[mask]
    blk = {"n_obs": len(sub),
           "n_unique_parents": int(sub.parent_structure_smiles.nunique()),
           "n_unique_frag_pairs": int(sub.groupby(
               ['parent_structure_smiles', 'fragment_a_smiles', 'fragment_b_smiles']).ngroups),
           "exp_min": round(float(sub.experimental_bde_kcal_mol.min()), 2),
           "exp_max": round(float(sub.experimental_bde_kcal_mol.max()), 2),
           "exp_mean": round(float(sub.experimental_bde_kcal_mol.mean()), 2),
           "channels": {}}
    for c in CHANNELS:
        e = sub[c + '_err'].values
        ae = np.abs(e)
        tt, tp = stats.ttest_1samp(e, 0.0)
        wstat, wp = common.wilcoxon(e) if len(np.unique(e)) > 1 else (float('nan'), float('nan'))
        blk["channels"][c] = {
            "mae": round(float(ae.mean()), 4), "rmse": round(common.rmse(e), 4),
            "bias": round(float(e.mean()), 4), "median_signed": round(float(np.median(e)), 4),
            "n_gt5": int((ae > 5).sum()), "n_gt10": int((ae > 10).sum()),
            "n_positive": int((e > 0).sum()), "n_negative": int((e < 0).sum()),
            "t_p": float(tp), "wilcoxon_p": float(wp),
            "sign_test_p": float(stats.binomtest(int((e > 0).sum()),
                                                 int((e != 0).sum()), 0.5).pvalue)
            if int((e != 0).sum()) else None,
        }
    regions[label] = blk
OUT["region_table"] = regions

# divergence: |signed error| vs experimental BDE trend
trend = {}
for c in CHANNELS:
    r, p = stats.pearsonr(df.experimental_bde_kcal_mol, df[c + '_err'])
    rho, sp = stats.spearmanr(df.experimental_bde_kcal_mol, df[c + '_err'])
    trend[c] = {"pearson_r_vs_exp": round(float(r), 4), "pearson_p": float(p),
                "spearman_rho": round(float(rho), 4), "spearman_p": float(sp),
                "slope_kcal_per_kcal": round(float(np.polyfit(
                    df.experimental_bde_kcal_mol, df[c + '_err'], 1)[0]), 5)}
OUT["divergence_trend"] = trend

# ==========================================================================
# 5. DIRECTION OF FAILURE
# ==========================================================================
direction = {"worst_cases_any_gt15": {}, "weak_region_sign_tests": {}}
mask15 = union_any15
sub15 = df[mask15]
for c in CHANNELS:
    e = sub15[c + '_err'].values
    npos, nneg = int((e > 0).sum()), int((e < 0).sum())
    direction["worst_cases_any_gt15"][c] = {
        "n_rows": len(e), "errors": [round(float(x), 2) for x in e],
        "n_overshoot": npos, "n_undershoot": nneg,
        "binomial_p": float(stats.binomtest(npos, npos + nneg, 0.5).pvalue)}
for c in CHANNELS:
    top = df[c + '_err'].abs().nlargest(25)
    e = df.loc[top.index, c + '_err'].values
    npos, nneg = int((e > 0).sum()), int((e < 0).sum())
    direction[f"worst25_of_{c}"] = {
        "n_overshoot": npos, "n_undershoot": nneg,
        "binomial_p": float(stats.binomtest(npos, npos + nneg, 0.5).pvalue),
        "mean_signed_err": round(float(e.mean()), 3),
        "median_signed_err": round(float(np.median(e)), 3),
        "mean_abs_err": round(float(np.abs(e).mean()), 3),
        "observation_ids": [str(i) for i in df.loc[top.index, 'observation_id']]}
for thr in (99, 70, 80, 90):
    mask = (df.experimental_bde_kcal_mol < thr) if thr < 99 else (df.experimental_bde_kcal_mol > 120)
    lbl = f"exp_lt{thr}" if thr < 99 else "exp_gt120"
    sub = df[mask]
    blk = {"n": len(sub)}
    for c in CHANNELS:
        e = sub[c + '_err'].values
        npos, nneg = int((e > 0).sum()), int((e < 0).sum())
        blk[c] = {"n_overshoot": npos, "n_undershoot": nneg,
                  "binomial_p": float(stats.binomtest(npos, npos + nneg, 0.5).pvalue)
                  if npos + nneg else None,
                  "bias": round(float(e.mean()), 4), "mae": round(float(np.abs(e).mean()), 4)}
    direction["weak_region_sign_tests"][lbl] = blk
OUT["direction_of_failure"] = direction

# ---- composition of the extreme set (union of any-channel |err|>10) --------
comp = {}
for c in CHANNELS:
    top = df[c + '_ae'].nlargest(25)
    comp[c] = {
        "worst25_n_not_in_db_row": int((~df.loc[top.index, 'in_db_row']).sum()),
        "worst25_n_same_fragment_pair": int(df.loc[top.index, 'same_frag_canon'].sum()),
        "worst25_n_arom_rings_ge2": int((df.loc[top.index, 'p_n_arom_rings'] >= 2).sum()),
        "worst25_mean_exp_bde": round(float(df.loc[top.index, 'experimental_bde_kcal_mol'].mean()), 2),
        "worst25_n_overshoot": int((df.loc[top.index, c + '_err'] > 0).sum()),
    }
comp["union_any_gt10"] = {
    "n": int(union_any10.sum()),
    "n_not_in_db_row": int((~df.loc[union_any10, 'in_db_row']).sum()),
    "frac_not_in_db_row": round(float((~df.loc[union_any10, 'in_db_row']).mean()), 4),
    "n_arom_rings_ge2": int((df.loc[union_any10, 'p_n_arom_rings'] >= 2).sum()),
    "mean_exp_bde": round(float(df.loc[union_any10, 'experimental_bde_kcal_mol'].mean()), 2),
    "mean_hac": round(float(df.loc[union_any10, 'p_hac'].mean()), 2),
}
comp["benchmark_baseline"] = {
    "frac_not_in_db_row": round(float((~df.in_db_row).mean()), 4),
    "frac_arom_rings_ge2": round(float((df.p_n_arom_rings >= 2).mean()), 4),
    "mean_exp_bde": round(float(df.experimental_bde_kcal_mol.mean()), 2),
    "mean_hac": round(float(df.p_hac.mean()), 2),
}
comp["reference_tail_bimodality"] = {
    "n_err_pos_gt10": int((df.reference_err > 10).sum()),
    "n_err_neg_lt_minus10": int((df.reference_err < -10).sum()),
    "pos_tail_mean_exp_bde": round(float(df.loc[df.reference_err > 10, 'experimental_bde_kcal_mol'].mean()), 2),
    "neg_tail_mean_exp_bde": round(float(df.loc[df.reference_err < -10, 'experimental_bde_kcal_mol'].mean()), 2),
    "pos_tail_examples": [{"observation_id": r.observation_id, "smiles": r.parent_structure_smiles,
                           "err": round(float(r.reference_err), 2)}
                          for r in df.loc[df.reference_err > 10].nlargest(5, 'reference_err').itertuples()],
    "neg_tail_examples": [{"observation_id": r.observation_id, "smiles": r.parent_structure_smiles,
                           "err": round(float(r.reference_err), 2)}
                          for r in df.loc[df.reference_err < -10].nsmallest(5, 'reference_err').itertuples()],
}
OUT["extreme_set_composition"] = comp

# ==========================================================================
# 6. WHAT IS SUPPORTED / UNRESOLVED
# ==========================================================================
OUT["verdicts"] = {
    "data_supported": [
        "Reference channel has by far the widest tail: 260/543 (47.9%) rows exceed 5 kcal/mol, "
        "51 (9.4%) exceed 10, 12 (2.2%) exceed 15; quantum 49/8/3 and learned 71/12/4.",
        "Only 3 rows have all three channels wrong by more than 10 kcal/mol and none by more "
        "than 15: the channels fail on largely disjoint rows (union of any-channel>10 is 61 rows).",
        "Both calibrated channels systematically OVERSHOOT in the weak-bond region: for exp<80 "
        "quantum 107/147 positive (sign-test p=3.0e-8, bias +1.74) and learned 123/147 "
        "(p=3.2e-17, bias +3.12); the overshoot grows as the bond weakens "
        "(bias +4.57 and +6.87 for n=18 rows with exp<70, both p<0.01).",
        "The learned channel's extreme tail is almost perfectly one-sided: 24 of its worst-25 "
        "errors are overshoots (binomial p=1.6e-6).",
        "The learned channel degrades on polycyclic / crowded parents and this survives the "
        "DB-membership confound: within the 212 non-DB rows a parent ring count >=3 raises MAE "
        "from 2.71 to 4.23 (n=90 vs 122, MWU p=1.3e-4).",
        "Reference channel extreme tail is bimodal and structurally readable: large POSITIVE "
        "errors on strained small rings (obs_0049/0050 C1=CC1, obs_0072 C1C2CC12, obs_0113 "
        "C=CC(=C)C, +15 to +21.6) and large NEGATIVE errors on the triarylmethane / "
        "tetraphenylmethane family (obs_0508 -33.4, obs_0520 -31.0, obs_0509 -24.7).",
    ],
    "suggestive_only": [
        "A 'crowded' flag (both radical centres in rings OR ring count >=3) raises MAE for all "
        "channels (quantum +0.48 p=0.015, learned +2.13 p=4.3e-11, reference +1.78 p=0.0012) "
        "but 93/95 crowded rows are also out-of-database rows, so the unstratified contrast is "
        "confounded; it is decisive only for the learned channel.",
        "Heteroatom count, aromatic-ring count and HAC all shift upward in the failure set for "
        "every channel (MWU p from 0.02 to 5e-9) but the effects are small "
        "(Cliff's delta 0.12-0.43) and the descriptors are mutually correlated.",
        "Reference-channel failure is mildly associated with fewer heteroatoms and MORE "
        "rotatable bonds (Cliff's delta -0.12 and -0.11) - opposite in sign to the calibrated "
        "channels, but at these effect sizes it is a weak signal.",
    ],
    "unresolved": [
        "For the quantum channel the crowded/polycyclic penalty disappears after stratifying on "
        "DB membership (delta +0.047, p=0.52 within non-DB rows) - at 543 rows the quantum "
        "channel cannot be shown to have a steric failure family.",
        "The quantum channel's worst-case direction is NOT established: 9 overshoots vs 8 "
        "undershoots among the 17 rows with any channel error >15 (p=1.0), and 17 vs 8 in its "
        "own worst-25 (p=0.11).",
        "Reference-channel weak-bond direction is unresolved: 7 positive vs 11 negative for "
        "exp<70 (p=0.48) and 16 vs 9 in its worst-25 (p=0.23); its sign test only becomes "
        "marginal at exp<80 (p=0.057).",
        "The 'both radical centres in a ring' sub-flag has only n=6 rows: MAE 5.15 vs 2.06 "
        "(quantum) but p=0.16 - not resolvable at this sample size.",
        "Whether the triarylmethane/tetraphenylmethane reference undershoots are a transferability "
        "failure of the reference channel or errors in the heterogeneous experimental "
        "references cannot be decided from this workspace.",
        "The 543 rows are heterogeneous literature values, not repeat measurements, so all "
        "tail counts carry an unknown reference uncertainty; a single bad experimental datum "
        "would appear as a failure of all three channels at once (only 3 such rows exist).",
    ],
    "analysis_units": {
        "tail_census": "bond observations (543 rows); each row is one experimental bond observation",
        "descriptor_and_flag_tests": "bond observations, but descriptors are parent-molecule "
                                     "properties so rows sharing a parent are not independent",
        "unique_molecules": int(df.parent_structure_smiles.nunique()),
        "unique_fragment_pairs": int(df.groupby(
            ['parent_structure_smiles', 'fragment_a_smiles', 'fragment_b_smiles']).ngroups),
        "n_unique_fragment_a": int(df.fragment_a_smiles.nunique()),
        "n_unique_fragment_b": int(df.fragment_b_smiles.nunique()),
    },
}

# ==========================================================================
# FIGURE
# ==========================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
colors = {"quantum": "#1f77b4", "learned": "#d62728", "reference": "#2ca02c"}
for ax, c in zip(axes, CHANNELS):
    x = df.experimental_bde_kcal_mol.values
    y = df[c + '_err'].values
    ax.axhline(0, color='k', lw=0.8)
    for lv in (-15, -10, -5, 5, 10, 15):
        ax.axhline(lv, color='grey', lw=0.4, ls=':')
    ax.scatter(x, y, s=8, alpha=0.45, color=colors[c], edgecolors='none')
    ax.axvline(60, color='orange', lw=0.8, ls='--')
    ax.axvline(70, color='orange', lw=0.8, ls='--')
    ax.axvline(80, color='orange', lw=0.8, ls='--')
    worst = df[c + '_ae'] > 10
    ax.scatter(x[worst.values], y[worst.values], s=42, facecolors='none',
               edgecolors='black', linewidths=1.1, zorder=5)
    for r in df.loc[worst].itertuples():
        ax.annotate(r.observation_id.replace("bond_observation_", "obs"),
                    (r.experimental_bde_kcal_mol, getattr(r, c + '_err')),
                    fontsize=5.2, xytext=(2, 2), textcoords='offset points')
    mae = np.abs(y).mean()
    ax.set_title(f"{c}  MAE={mae:.2f}  bias={y.mean():+.2f}  "
                 f"n|e|>10={int(worst.sum())}", fontsize=10)
    ax.set_xlabel("experimental BDE (kcal/mol)")
    ax.set_ylim(-36, 30)
axes[0].set_ylabel("signed error, estimate - experiment (kcal/mol)")
fig.suptitle("WS4: signed error vs experimental BDE, 543 bond observations "
             "(circles = |error|>10; dashed lines = 60/70/80 kcal/mol)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{PROJECT}/output/figures/ws4_failures.png', dpi=150)
print("[ws4] wrote figure")

path = common.save_json(OUT, "ws4_failures.json")
print(json.dumps({
    "tail": {c: {k: census[c][k] for k in ("n_gt5", "n_gt10", "n_gt15", "max_abs_err")} for c in CHANNELS},
    "union": OUT["tail_census"]["union"]["n_any_channel_gt10"],
}, indent=2))

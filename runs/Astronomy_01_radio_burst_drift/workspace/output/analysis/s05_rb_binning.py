#!/usr/bin/env python3
"""S05 - ROBUSTNESS TEST 1: binning sensitivity of the drift-population counts.

Question attacked: "the drift-population counts show a multi-component (fast/slow
or multi-population) structure".

Two adversarial checks:

A. THE `drift_regime` LABEL IS A SOURCE-BLOCK ARTEFACT, NOT A MEASURED PROPERTY.
   The normalised view concatenates, per series, the workbook blocks
   '(-35<s<-10)' (coarse ~1.0 MHz/s bins) and '(-10<s<0)' + '(0<s<20)' (fine
   ~0.10 MHz/s bins).  The 'fast'/'slow' split is therefore a re-labelling of
   *which column of a spreadsheet the number came from*.  We verify this
   exactly, then re-bin every series onto a COMMON grid so the comparison the
   regime label invites can actually be made.

B. GRID DEPENDENCE.  Re-bin onto linear 0.1/0.25/0.5/1.0 MHz/s grids over
   [0, 28] plus a log-spaced grid, and for each grid report occupied bins, the
   mode, and whether a 2-component lognormal mixture beats 1 component by
   AIC/BIC.  If the answer (1 vs 2 components) flips with the grid, any
   claimed multi-population result is grid-dependent.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from s04_rb_common import (  # noqa: E402
    LOGDIR, RESULTS, concentration, mix_fit, mode_of, rebin, series_blocks, write_pair,
)

GRIDS = {
    "linear_0.10": np.round(np.arange(0.0, 28.0 + 1e-9, 0.10), 6),
    "linear_0.25": np.round(np.arange(0.0, 28.0 + 1e-9, 0.25), 6),
    "linear_0.50": np.round(np.arange(0.0, 28.0 + 1e-9, 0.50), 6),
    "linear_1.00": np.round(np.arange(0.0, 28.0 + 1e-9, 1.00), 6),
    "log": np.round(np.logspace(np.log10(0.05), np.log10(28.0), 60), 6),
}
# keep the log grid strictly increasing and inside range
assert np.all(np.diff(GRIDS["log"]) > 0)
for g in GRIDS.values():
    assert np.all(np.diff(g) > 0)


def main() -> None:
    log = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    P("=" * 78)
    P("S05 BINNING SENSITIVITY  (analysis unit: one complete source-polarization series)")
    P("=" * 78)

    blocks = series_blocks()
    P(f"source blocks recovered: {len(blocks)}")
    for b in blocks:
        P("  %-20s %-14s blk%d  n=%3d  dx=%.5f  range=[%.4f, %.4f]  events=%d"
          % (b["population_series_id"], b["drift_regime"], b["block_index"],
             b["n_bins"], b["dx"], b["lo"], b["hi"], int(b["events"])))

    # ---------------------------------------------------------------- A
    P("")
    P("--- A. is `drift_regime` a measured property or a source-block label? ---")
    tot_ev = sum(b["events"] for b in blocks)
    by_reg = {}
    for b in blocks:
        by_reg.setdefault(b["drift_regime"], []).append(b)
    reg_tab = []
    for reg, bs in sorted(by_reg.items()):
        dxs = sorted({round(b["dx"], 4) for b in bs})
        nblocks = sorted({b["block_index"] for b in bs})
        ev = sum(b["events"] for b in bs)
        lo = min(b["lo"] for b in bs)
        hi = max(b["hi"] for b in bs)
        reg_tab.append(dict(drift_regime=reg, n_blocks=len(bs), n_bins=sum(b["n_bins"] for b in bs),
                            dx_values=str(dxs), block_indices=str(nblocks), events=float(ev),
                            events_share=float(ev / tot_ev), range_lo=lo, range_hi=hi))
        P("  regime=%-5s blocks=%d bins=%3d distinct dx=%s  events=%6d (%.1f%%)  range=[%.3f, %.3f]"
          % (reg, len(bs), sum(b["n_bins"] for b in bs), dxs, ev, 100 * ev / tot_ev, lo, hi))
    write_pair("s05_regime_is_block_label", reg_tab)

    overlap = min(b["hi"] for b in by_reg["fast"]) if "fast" in by_reg else None
    slowmax = max(b["hi"] for b in by_reg["slow"]) if "slow" in by_reg else None
    P("  OVERLAP: min(fast hi)=%.3f  max(slow hi)=%.3f  -> the two 'regimes' share x-range [%.3f, %.3f]"
      % (overlap, slowmax, min(b["lo"] for b in by_reg["fast"]), min(overlap, slowmax)))
    P("  CONCLUSION(A): 'fast' == the '(-35<s<-10)' workbook block of each series;")
    P("                 'slow' == the CONCATENATION of the '(-10<s<0)' and '(0<s<20)' blocks.")
    P("                 Equal event_count in two regimes is NOT evidence of a bimodal drift")
    P("                 distribution, because the two regimes are different measurements.")

    # per-series check that slow is exactly the concatenation
    d = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parents[2]
                    / "data" / "normalized_views" / "drift_population_series.tsv", sep="\t")
    conc_rows = []
    for pid in sorted(d.population_series_id.unique()):
        s = d[(d.population_series_id == pid) & (d.drift_regime == "slow")]
        c = s.distribution_bin_center.to_numpy()
        y = s.event_count.to_numpy()
        cut = np.where(np.diff(c) < 0)[0] + 1
        ok = len(cut) == 1
        b1 = y[: cut[0]].sum() if ok else np.nan
        b2 = y[cut[0]:].sum() if ok else np.nan
        conc_rows.append(dict(population_series_id=pid, n_slow_bins=len(c),
                              single_negative_jump=bool(ok),
                              first_block_bins=int(cut[0]) if ok else None,
                              first_block_events=float(b1) if ok else None,
                              second_block_bins=int(len(c) - cut[0]) if ok else None,
                              second_block_events=float(b2) if ok else None))
    write_pair("s05_slow_is_two_concatenated_blocks", conc_rows)
    for r in conc_rows:
        P("  %s slow: single jump=%s  block1 %d bins/%d ev | block2 %d bins/%d ev"
          % (r["population_series_id"], r["single_negative_jump"], r["first_block_bins"],
             r["first_block_events"], r["second_block_bins"], r["second_block_events"]))

    # ---------------------------------------------------------------- B
    P("")
    P("--- B. re-bin every series onto common grids; fit 1- vs 2-lognormal ---")
    rows = []
    prof_rows = []
    for pid in sorted({b["population_series_id"] for b in blocks}):
        bs = [b for b in blocks if b["population_series_id"] == pid]
        grp = bs[0]["anonymous_source_group"]
        ch = bs[0]["polarization_channel"]
        for gname, edges in GRIDS.items():
            counts = rebin(bs, edges)
            occ = int((counts > 0).sum())
            mode = mode_of(edges, counts)
            c1 = mix_fit(edges, counts, 1)
            c2 = mix_fit(edges, counts, 2)
            r = dict(population_series_id=pid, anonymous_source_group=grp, polarization_channel=ch,
                     grid=gname, n_grid_bins=len(edges) - 1, n_occupied=occ,
                     frac_occupied=float(occ / (len(edges) - 1)),
                     mode_MHz_s=mode, total_events=float(counts.sum()),
                     aic1=c1["aic"], aic2=c2["aic"], bic1=c1["bic"], bic2=c2["bic"],
                     dAIC_2v1=float(c1["aic"] - c2["aic"]),
                     dBIC_2v1=float(c1["bic"] - c2["bic"]),
                     prefers_2_by_AIC=bool(c1["aic"] - c2["aic"] > 0),
                     prefers_2_by_BIC=bool(c1["bic"] - c2["bic"] > 0))
            if c2:
                p = c2["params"]
                w1 = 1 / (1 + np.exp(-p[0]))
                r.update(comp1_weight=w1, comp1_med_MHz_s=float(10 ** p[1]),
                         comp1_sigma_dex=float(np.exp(p[2])),
                         comp2_med_MHz_s=float(10 ** p[3]),
                         comp2_sigma_dex=float(np.exp(p[4])))
            rows.append(r)
            for e0, e1, c in zip(edges[:-1], edges[1:], counts):
                prof_rows.append(dict(population_series_id=pid, grid=gname,
                                      bin_lo=float(e0), bin_hi=float(e1),
                                      bin_center=float(np.sqrt(e0 * e1)), count=float(c)))

    tab = pd.DataFrame(rows)
    write_pair("s05_binning_grid_sensitivity", tab)
    write_pair("s05_rebinned_profiles", pd.DataFrame(prof_rows))

    P("")
    P("  per-grid summary over the 6 series (does a 2-component mixture beat 1?)")
    for gname in GRIDS:
        sub = tab[tab.grid == gname]
        P("   grid %-12s gridbins=%3d  occupied %s  dAIC>0 in %d/6  dBIC>0 in %d/6  modes=%s"
          % (gname, sub.n_grid_bins.iloc[0],
             str(sorted(sub.n_occupied.tolist())),
             int(sub.prefers_2_by_AIC.sum()), int(sub.prefers_2_by_BIC.sum()),
             str([round(x, 2) for x in sub.mode_MHz_s.tolist()])))

    P("")
    P("  SAME-SERIES comparison, fast-block-only vs slow-blocks-only vs all combined,")
    P("  evaluated on ONE common grid (linear 0.25 MHz/s):")
    edges = GRIDS["linear_0.25"]
    same = []
    for pid in sorted({b["population_series_id"] for b in blocks}):
        bs = [b for b in blocks if b["population_series_id"] == pid]
        for sel, nm in ((lambda b: b["drift_regime"] == "fast", "fast_block"),
                        (lambda b: b["drift_regime"] == "slow", "slow_blocks"),
                        (lambda b: True, "all_combined")):
            sub = [b for b in bs if sel(b)]
            counts = rebin(sub, edges)
            c1 = mix_fit(edges, counts, 1)
            c2 = mix_fit(edges, counts, 2)
            same.append(dict(population_series_id=pid, subset=nm, n_events=float(counts.sum()),
                             n_occupied=int((counts > 0).sum()), mode_MHz_s=mode_of(edges, counts),
                             dAIC_2v1=float(c1["aic"] - c2["aic"]),
                             dBIC_2v1=float(c1["bic"] - c2["bic"]),
                             prefers_2_by_BIC=bool(c1["bic"] - c2["bic"] > 0),
                             dispersion_vmr=float(counts.var(ddof=1) / max(counts.mean(), 1e-12))))
    st = pd.DataFrame(same)
    write_pair("s05_subset_grid025", st)
    for pid in sorted(st.population_series_id.unique()):
        sub = st[st.population_series_id == pid]
        P("   %s" % pid)
        for _, r in sub.iterrows():
            P("      %-13s events=%6d occ=%2d mode=%6.2f  dBIC=%9.2f  VMR=%8.3f"
              % (r["subset"], r["n_events"], r["n_occupied"], r["mode_MHz_s"],
                 r["dBIC_2v1"], r["dispersion_vmr"]))

    pd.DataFrame(rows).to_json(RESULTS / "s05_binning_summary.json", orient="records", indent=2)
    (LOGDIR / "s05_binning.log").write_text("\n".join(log) + "\n")
    print("\nwrote", RESULTS / "s05_binning_grid_sensitivity.csv")


if __name__ == "__main__":
    main()
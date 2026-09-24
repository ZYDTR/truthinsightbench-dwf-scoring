"""Figure 2: relation between normalized_spectral_response.tsv and the photon-flux grid.
Raw TSVs only; no other worker's artifacts are read."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/data'
OUT = '/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/output/figures/fig2_crossfile_relation.png'

g = pd.read_csv(f'{DATA}/photon_flux_grid.tsv', sep='\t')
r = pd.read_csv(f'{DATA}/normalized_spectral_response.tsv', sep='\t')

wg = np.sort(g.wavelength_nm.unique())      # 176 pts, 179.40 .. 850.00
wr = r.wavelength_nm.values                 # 175 pts, 179.31 .. 849.77

# ---------------------------------------------------------------- axis alignment
# index-aligned for i<=66; for i>=67 the grid index is j+1 (one grid wavelength,
# 321.0 nm, has no response counterpart -> response axis is index-shifted by +1)
idx_map = np.arange(len(wr))
idx_map[67:] = np.arange(67, len(wr)) + 1
wg_mapped = wg[idx_map]
maxres = float(np.max(np.abs(wg_mapped - wr)))

# ---------------------------------------------------- metallicity assignment
# best constant fit per level, searched over the full 3 T x 35 M grid
levels = [1, 2, 3, 4, 5]
Ms = {1: -0.90, 2: -0.50, 3: 0.00, 4: 0.50, 5: 0.90}
COL = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}


def flux_on(wl, T, M):
    """Linear interpolation of the grid photon flux onto wavelength array wl."""
    d = g[(g.effective_temperature_K == T) & (g.metallicity_index == M)]
    d = d.sort_values('wavelength_nm')
    return np.interp(wl, d.wavelength_nm.values, d.photon_flux_cm2.values)


OPT = wr >= 400.0          # fitted band: optical only
C = {}                     # fitted constant per level, T=5800 K
for L in levels:
    x = r[f'response_level_{L}'].values * wr / flux_on(wr, 5800, Ms[L])
    C[L] = float(np.median(x[OPT]))

# temperature failure: ratio normalised by the T=5800 constant
x5300, x6300 = {}, {}
for L in levels:
    y = r[f'response_level_{L}'].values
    x5300[L] = y * wr / flux_on(wr, 5300, Ms[L]) / C[L]
    x6300[L] = y * wr / flux_on(wr, 6300, Ms[L]) / C[L]

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)

# ---- (a) the five response series
axa = fig.add_subplot(gs[0, 0])
for L in levels:
    axa.plot(wr, r[f'response_level_{L}'].values, lw=1.5, color=COL[L],
             label=f'response_level_{L}  (M={Ms[L]:+.2f})')
axa.set_yscale('log')
axa.set_xlabel('wavelength (nm)')
axa.set_ylabel('normalized response (log scale)')
axa.set_title('(a) The five independent normalized response series (175 pts, 179.31-849.77 nm)')
axa.legend(fontsize=8, loc='lower right')
axa.grid(alpha=0.3, which='both')
axa.axvspan(300, 330, color='0.85', alpha=0.5, zorder=0)

# ---- (b) decisive test at T = 5800 K
axb = fig.add_subplot(gs[0, 1])
for L in levels:
    x = r[f'response_level_{L}'].values * wr / flux_on(wr, 5800, Ms[L])
    axb.plot(wr, x, lw=1.4, color=COL[L], label=f'level {L}, M={Ms[L]:+.2f}')
axb.axhline(C[3], color='k', ls='-', lw=1.4,
            label=f'fitted constant = {C[3]:.3e} (median, >=400 nm only)')
for b in (296, 299, 322, 327, 330):
    axb.axvline(b, color='k', ls='--', lw=0.9, alpha=0.7)
axb.text(297, 1.05e-12, '296/299', fontsize=7, rotation=90, va='top')
axb.text(323, 1.05e-12, '322', fontsize=7, rotation=90, va='top')
axb.text(331, 1.05e-12, '327/330', fontsize=7, rotation=90, va='top')
axb.set_yscale('log')
axb.set_xlim(179, 860)
axb.set_ylim(3.80e-13, 2.45e-12)
axb.set_xlabel('wavelength (nm)')
axb.set_ylabel(r'$response_k \cdot \lambda \; / \; F(T=5800\,K, M_k, \lambda)$')
axb.set_title('(b) DECISIVE TEST: photon-to-energy collapse at T=5800 K, and its UV breakdown')
axb.legend(fontsize=7, loc='lower left', ncol=2, framealpha=0.95)
axb.grid(alpha=0.3, which='both')
arrow = dict(arrowstyle='->', lw=0.8, color='0.25')
axb.annotate('all 5 levels collapse onto the same constant\nover 330-850 nm (max 0.4%, 129 pts)',
             xy=(640, C[3]), xytext=(408, 1.95e-12), fontsize=8, arrowprops=arrow)
axb.annotate('x5.01 plateau\n299.9-319.9 nm\n(21 pts, all levels)',
             xy=(313, 5.0 * C[3]), xytext=(212, 1.80e-12),
             fontsize=8, ha='center', arrowprops=arrow)
axb.annotate('x2.54 step\n322.4-326.4 nm', xy=(324, 2.54 * C[3]), xytext=(400, 8.6e-13),
             fontsize=8, ha='center', arrowprops=arrow)
axb.annotate('fitted constant\nC = %.3e (median, >=400 nm)' % C[3],
             xy=(795, C[3]), xytext=(690, 1.30e-12), fontsize=8, ha='center', arrowprops=arrow)
axb.annotate('smooth 2.7x decline below the 299 nm edge\n(3.10 C at 179.3 nm -> 1.14 C at 296.2 nm,\nno step: the response jumps, the flux does not)',
             xy=(255, r['response_level_3'].values[np.argmin(abs(wr - 255))] * 255
                / flux_on(np.array([255.]), 5800, 0.0)[0]),
             xytext=(196, 6.0e-13), fontsize=7.5, arrowprops=arrow)

# ---- (c) the same test at T = 5300 K and 6300 K
axc = fig.add_subplot(gs[1, 0])
for L in levels:
    axc.plot(wr, x5300[L], lw=1.2, color=COL[L], alpha=0.9)
    axc.plot(wr, x6300[L], lw=1.2, color=COL[L], ls=':', alpha=0.9)
axc.axhline(1.0, color='k', lw=1.3)
axc.set_yscale('log')
axc.set_xlim(179, 860)
axc.set_ylim(5.5e-3, 2.2e2)
axc.set_xlabel('wavelength (nm)')
axc.set_ylabel(r'$(response_k \cdot \lambda / F(T,M_k)) \; / \; C_{5800}$')
axc.set_title('(c) Same ratio at T=5300 K (solid) and T=6300 K (dotted) - neither is constant')
axc.grid(alpha=0.3, which='both')
axc.text(0.02, 0.97, 'solid = T 5300 K\ndotted = T 6300 K',
         transform=axc.transAxes, fontsize=8, va='top',
         bbox=dict(fc='w', ec='0.6', alpha=0.9))
axc.text(0.30, 0.62, 'no wavelength-independent constant:\nT=5300 K drifts 0.94 -> 1.58 (factor 1.5)\nT=6300 K drifts 0.70 -> 1.10 (factor 1.4)\nacross 400-850 nm at fixed metallicity',
         transform=axc.transAxes, fontsize=8, va='top', ha='left',
         bbox=dict(fc='w', ec='0.6', alpha=0.9))
axc.annotate('both wrong temperatures keep the SAME\nUV plateau/step shape (x5, x2.5):\nthat structure is a pure wavelength effect,\nindependent of temperature',
             xy=(305, 5.0), xytext=(0.30, 0.20), textcoords='axes fraction',
             fontsize=8, va='top', arrowprops=arrow)

# ---- (d) wavelength-grid misalignment
axd = fig.add_subplot(gs[1, 1])
axd.plot(np.arange(len(wg)), wg, lw=1.6, color='#1f77b4',
         label=f'photon_flux_grid.tsv ({len(wg)} pts, 179.40-850.00 nm)')
axd.plot(np.arange(len(wr)), wr, lw=1.6, color='#d62728',
         label=f'normalized_spectral_response.tsv ({len(wr)} pts, 179.31-849.77 nm)')
axd.axvline(67, color='k', ls='--', lw=1.0)
axd.text(74, 700, 'index 67: response axis\nshifts +1 grid index\n(grid 321.0 nm is skipped)', fontsize=8)
axd.set_xlabel('wavelength index')
axd.set_ylabel('recorded wavelength (nm)')
axd.set_title('(d) Wavelength-grid misalignment: recorded wavelength vs index')
axd.legend(fontsize=8, loc='upper left')
axd.grid(alpha=0.3)

axi = axd.inset_axes([0.30, 0.16, 0.66, 0.32])
d_index = wg[:len(wr)] - wr
d_true = wg_mapped - wr
axi.plot(np.arange(len(wr)), d_index, lw=1.2, color='#7f7f7f',
         label=r'$\lambda_{grid}[i]-\lambda_{resp}[i]$ (naive index)')
axi.plot(np.arange(len(wr)), d_true, lw=1.6, color='#2ca02c',
         label=r'$\lambda_{grid}[i{+}1]-\lambda_{resp}[i]$, i>=67')
axi.axvline(67, color='k', ls='--', lw=0.9)
axi.axhline(0, color='k', lw=0.7)
axi.set_xlabel('response index', fontsize=7)
axi.set_ylabel('delta (nm)', fontsize=7)
axi.tick_params(labelsize=6.5)
axi.legend(fontsize=6, loc='lower left')
axi.set_title('after correction |delta| <= 0.23 nm', fontsize=7)

fig.suptitle('Figure 2 - Independent normalized spectral response vs. bundled photon-flux grid\n'
             'Flux values are LINEARLY INTERPOLATED from the 176-point grid axis onto the 175-point '
             'response axis (axes differ: index-shifted by +1 past index 66). '
             'Fitted constant uses the >=400 nm band only.',
             fontsize=11)
fig.savefig(OUT, dpi=170, bbox_inches='tight')
print('saved', OUT)

# ------------------------------------------------------------------ numbers
print(f'max |grid[idx_map] - resp| after +1 shift = {maxres:.4f} nm')
for L in levels:
    x = r[f'response_level_{L}'].values * wr / flux_on(wr, 5800, Ms[L])
    print(f'level {L} M={Ms[L]:+.2f}: C={C[L]:.5e}  cv(>=400nm)={(x[OPT]/C[L]).std():.5f}')
print('UV plateau 299-319 nm  mean ratio/C:')
for L in levels:
    x = r[f'response_level_{L}'].values * wr / flux_on(wr, 5800, Ms[L])
    band = (wr >= 299.5) & (wr <= 319.91)
    print(f'  level {L}: {np.nanmean(x[band] / C[L]):.3f}  (n={band.sum()})')
    band2 = (wr >= 322.0) & (wr <= 326.5)
    print(f'  level {L}: 322-327 step {np.nanmean(x[band2] / C[L]):.3f}')
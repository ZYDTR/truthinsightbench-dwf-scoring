import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, common
j = pd.read_pickle('/tmp/bench_joined.pkl')
exp=j.experimental_bde_kcal_mol.to_numpy(float); q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float); r=j.reference_estimate_bde_kcal_mol.to_numpy(float)
inb=j.in_db_row.to_numpy(); db=j.db_bde.to_numpy(float)

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

# --- panel A: db vs quantum difference histogram ---------------------------
ax=axes[0,0]
d = db[inb]-q[inb]
ax.hist(d, bins=90, color='#2b6cb0', edgecolor='white', linewidth=0.3)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axvline(np.median(d), color='#c53030', lw=1.4, label=f'median = {np.median(d):+.4f}')
ax.set_yscale('log')
ax.set_xlabel('database BDE − quantum estimate (kcal/mol)')
ax.set_ylabel('count (log scale)')
ax.set_title(f'A. Quantum channel vs its database row\n(n={int(inb.sum())} matched obs; '
             f'{np.mean(np.abs(d)<=0.1)*100:.1f}% within 0.1, max |diff| {np.abs(d).max():.2f})')
ax.legend(fontsize=8, loc='upper left')
ax.text(0.98,0.95, f"mean {d.mean():+.4f}\nsd {d.std(ddof=1):.4f}", transform=ax.transAxes,
        ha='right', va='top', fontsize=8, bbox=dict(fc='white', ec='0.7', alpha=.9))

# inset: zoom on the core
axi = ax.inset_axes([0.55,0.45,0.42,0.42])
axi.hist(d[np.abs(d)<=0.3], bins=60, color='#68a0d0', edgecolor='white', linewidth=0.2)
axi.axvline(0, color='k', lw=0.7, ls='--')
axi.set_title('zoom |diff|<=0.3', fontsize=7); axi.tick_params(labelsize=6)

# --- panel B: db vs learned -------------------------------------------------
ax=axes[0,1]
dl = db[inb]-l[inb]
bins=np.linspace(min(d.min(),dl.min()), max(d.max(),dl.max()), 90)
ax.hist(d, bins=bins, alpha=.6, label=f'quantum  (|d|<=0.1: {np.mean(np.abs(d)<=0.1)*100:.0f}%)', color='#2b6cb0')
ax.hist(dl, bins=bins, alpha=.6, label=f'learned  (|d|<=0.1: {np.mean(np.abs(dl)<=0.1)*100:.0f}%)', color='#dd6b20')
ax.set_yscale('log'); ax.axvline(0,color='k',lw=.8,ls='--')
ax.set_xlabel('database BDE − channel estimate (kcal/mol)'); ax.set_ylabel('count (log scale)')
ax.set_title('B. Both channels against the database row they were trained on')
ax.legend(fontsize=8)
ax.text(0.02,0.05,f"quantum max|d|={np.abs(d).max():.2f}\nlearned  max|d|={np.abs(dl).max():.2f}",
        transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(fc='white',ec='0.7',alpha=.9))

# --- panel C: per-molecule experimental spread ------------------------------
ax=axes[1,0]
size = j.groupby('parent_structure_smiles').size()
multi = size[size>1].index
sub = j[j.parent_structure_smiles.isin(multi)]
ordr = sub.groupby('parent_structure_smiles').experimental_bde_kcal_mol.mean().sort_values().index
pos = {m:i for i,m in enumerate(ordr)}
for m in ordr:
    dd = sub[sub.parent_structure_smiles==m]
    x = pos[m]; yo = np.argsort(dd.experimental_bde_kcal_mol.to_numpy())
    ax.plot([x]*len(dd), dd.experimental_bde_kcal_mol.to_numpy()[yo], 'o-', ms=2.5, lw=0.5, color='#2b6cb0', alpha=.55)
    ax.plot([x]*len(dd), dd.quantum_estimate_bde_kcal_mol.to_numpy()[yo], 's-', ms=2.0, lw=0.5, color='#dd6b20', alpha=.55)
ax.plot([],[],'o-',color='#2b6cb0',label='experimental'); ax.plot([],[],'s-',color='#dd6b20',label='quantum')
g = sub.groupby('parent_structure_smiles').experimental_bde_kcal_mol.agg(['min','max'])
sp=(g['max']-g['min']).to_numpy()
ax.set_xlabel(f'molecule rank ({len(ordr)} molecules with >1 bond observation)')
ax.set_ylabel('BDE (kcal/mol)')
ax.set_title(f'C. Within-molecule experimental spread\nmean {sp.mean():.2f}, median {np.median(sp):.2f}, max {sp.max():.2f} kcal/mol')
ax.legend(fontsize=8, loc='upper left')
ax.text(0.98,0.03, f'rms spread {np.sqrt((sp**2).mean()):.2f}\nimplied sigma {np.sqrt((sp**2).mean()/2):.2f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, bbox=dict(fc='white',ec='0.7',alpha=.9))

# --- panel D: MAE vs the heterogeneity floor --------------------------------
ax=axes[1,1]
labels=['quantum','learned','reference']
maes=[np.abs(exp-q).mean(), np.abs(exp-l).mean(), np.abs(exp-r).mean()]
colors=['#2b6cb0','#dd6b20','#c53030']
floors={'co-tight\n(1.54)':1.538,'tight-quantum\n(1.91)':1.914,'decomposition\n(2.26)':2.260}
bars=ax.bar(labels, maes, color=colors, alpha=.85, width=.55)
for b,v in zip(bars,maes): ax.text(b.get_x()+b.get_width()/2, v+.06, f'{v:.3f}', ha='center', fontsize=9)
for i,(k,v) in enumerate(floors.items()):
    ax.axhline(v, ls='--', lw=1.1, color=['#38a169','#805ad5','#d69e2e'][i])
    ax.text(2.45, v+0.03, f'floor {k}', fontsize=7.5, color=['#38a169','#805ad5','#d69e2e'][i], va='bottom', ha='right')
ax.set_ylabel('MAE vs experimental (kcal/mol)')
ax.set_title('D. Channel MAE vs the reference-heterogeneity floor\n(543 bond observations)')
ax.set_ylim(0, 6.4)
ax.text(0.02,0.97,'quantum 2.098 sits BELOW the\ndecomposition floor 2.260', transform=ax.transAxes,
        fontsize=8, va='top', bbox=dict(fc='#fffbea', ec='#d69e2e', alpha=.95))
fig.suptitle('WS5 provenance audit: the quantum channel reproduces the database row; experimental reference heterogeneity bounds resolution',
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0,0,1,0.97])
out = common.FIGURES / 'ws5_provenance.png'
fig.savefig(out, dpi=160)
print("wrote", out)

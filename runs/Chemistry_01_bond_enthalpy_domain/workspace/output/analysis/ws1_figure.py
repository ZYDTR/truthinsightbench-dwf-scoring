"""WS1 figure: bond-class census (raw rows vs unique pairs) + HAC distributions."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import common

PROJ = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'
fig_pkl = pd.read_pickle(f'{PROJ}/output/cache/ws1_fig.pkl')
census = pd.DataFrame(fig_pkl['census'])
hac_cmp = fig_pkl['hac_cmp']
b = fig_pkl['b']
desc = pd.read_pickle(f'{PROJ}/output/cache/db_mol_desc.pkl')

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("WS1  Database inventory, structural domain and coverage boundary\n"
             "BDE database: 484,907 raw rows | 42,577 unique molecules | 290,664 unique fragment pairs",
             fontsize=13, fontweight='bold')

# -- (a) bond-class census: rows vs unique pairs --------------------------
ax = axes[0, 0]
x = np.arange(len(census)); w = 0.4
ax.bar(x - w/2, census.raw_rows, w, label='raw rows (484,907)', color='#3b6ea5')
ax.bar(x + w/2, census.unique_fragment_pairs, w, label='unique fragment pairs (290,664)', color='#e08a3c')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(census.bond_class, rotation=45, ha='right')
ax.set_ylabel('count (log scale)')
ax.set_title('(a) Bond-class census: raw rows vs unique fragment pairs')
ax.legend(fontsize=9)
for i, (r, p) in enumerate(zip(census.raw_rows, census.unique_fragment_pairs)):
    ax.text(i, r * 1.15, f"{r/p:.2f}x", ha='center', fontsize=8, color='#b3452a')
ax.grid(axis='y', alpha=0.25)

# -- (b) redundancy factor per class --------------------------------------
ax = axes[0, 1]
c2 = census.sort_values('redundancy_rows_per_pair')
ax.barh(c2.bond_class, c2.redundancy_rows_per_pair, color='#b3452a')
ax.axvline(1.0, color='k', ls='--', lw=1, label='no redundancy (1.0)')
ax.axvline(484907/290664, color='green', ls=':', lw=1.4, label='database overall 1.67')
ax.set_xlabel('raw rows per unique fragment pair')
ax.set_title('(b) Row redundancy by bond class')
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.25)
for i, (n, v) in enumerate(zip(c2.bond_class, c2.redundancy_rows_per_pair)):
    ax.text(v + 0.03, i, f"{v:.2f}", va='center', fontsize=8)

# -- (c) HAC distributions: DB molecules vs benchmark observations --------
ax = axes[1, 0]
hac = hac_cmp.hac.to_numpy()
ax.bar(hac - 0.2, hac_cmp.db_unique_molecules / 42577 * 100, 0.4,
       label='DB unique molecules (n=42,577)', color='#3b6ea5')
ax.bar(hac + 0.2, hac_cmp.benchmark_observations / 543 * 100, 0.4,
       label='benchmark observations (n=543)', color='#e08a3c')
ax.axvline(10.5, color='red', ls='--', lw=1.6)
ax.text(10.8, ax.get_ylim()[1]*0.85, 'DB HAC max = 10\n150 / 543 (27.6%) benchmark\nobservations beyond it',
        color='red', fontsize=9)
ax.set_xlabel('parent heavy-atom count (HAC)'); ax.set_ylabel('percent of unit')
ax.set_title('(c) HAC distribution: database vs benchmark')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)

# -- (d) coverage boundary waterfall --------------------------------------
ax = axes[1, 1]
labels = ['543 benchmark\nobservations',
          'exact fragment pair\nin DB',
          'parent molecule in DB\n(pair may be absent)',
          'parent HAC <= 10\n(inside DB domain)',
          'parent HAC > 10\n(outside DB domain)']
vals = [543, 331, 369, 393, 150]
colors = ['#777777', '#2e7d32', '#3b6ea5', '#3b6ea5', '#c62828']
bars = ax.bar(labels, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 8, f"{v}\n{100*v/543:.1f}%",
            ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('benchmark observations'); ax.set_ylim(0, 620)
ax.set_title('(d) Coverage boundary of the 543 benchmark observations')
ax.grid(axis='y', alpha=0.25)
plt.setp(ax.get_xticklabels(), fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = f'{PROJ}/output/figures/ws1_domain.png'
plt.savefig(out, dpi=140)
print("wrote", out)

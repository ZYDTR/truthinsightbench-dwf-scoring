import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json, time
import numpy as np, pandas as pd, common

t0=time.time()
bench = common.load_benchmark()
db = pd.read_pickle('/tmp/db.pkl')
print("loaded", round(time.time()-t0,1), flush=True)

c1,u1 = pd.factorize(db.molecule); c2,u2 = pd.factorize(db.fragment1); c3,u3 = pd.factorize(db.fragment2)
print("factorize", round(time.time()-t0,1), flush=True)
n2,n3 = np.int64(len(u2)), np.int64(len(u3))
comb = (c1.astype(np.int64)*n2 + c2.astype(np.int64))*n3 + c3.astype(np.int64)
uniq, inv = np.unique(comb, return_inverse=True)
print("codes", round(time.time()-t0,1), "nk", len(uniq), flush=True)

bde = db.bde.to_numpy(float)
s = pd.Series(bde).groupby(inv)
mn, mx, cnt, mean = s.min().to_numpy(), s.max().to_numpy(), s.size().to_numpy(), s.mean().to_numpy()
order = np.lexsort((bde, inv)); iv = inv[order]; bv = bde[order]
new = np.empty(len(iv), bool); new[0]=True; new[1:] = (iv[1:]!=iv[:-1]) | (bv[1:]!=bv[:-1])
nu = np.bincount(iv, weights=new.astype(float)).astype(int)
print("group stats", round(time.time()-t0,1), flush=True)

# bond_type per key (first) and n distinct
bt = db.bond_type.to_numpy()
bt_first = pd.Series(bt).groupby(inv, sort=True).first().to_numpy()
bt_nu = pd.Series(bt).groupby(inv).nunique().to_numpy()
print("bt stats", round(time.time()-t0,1), flush=True)

# bench codes: hash via dict built from unique combos (cheap: 290664 lookups)
kmol = {i:0 for i in ()}
print("build lookup dicts", round(time.time()-t0,1), flush=True)
u1i = {v:i for i,v in enumerate(u1)}; u2i = {v:i for i,v in enumerate(u2)}; u3i = {v:i for i,v in enumerate(u3)}
bcode = np.fromiter(
    ((u1i[m]*n2 + u2i[a])*n3 + u3i[bb] if (m in u1i and a in u2i and bb in u3i) else -1
     for m,a,bb in zip(bench.parent_structure_smiles, bench.fragment_a_smiles, bench.fragment_b_smiles)),
    dtype=np.int64, count=len(bench))
slot = pd.Series(np.arange(len(uniq), dtype=np.int64), index=uniq).reindex(bcode).to_numpy()  # -1 if absent
print("bench codes", round(time.time()-t0,1), "matched", int((~np.isnan(slot)).sum()) if slot.dtype.kind=='f' else int((slot>=0).sum()), flush=True)

valid = slot >= 0
j = bench.copy()
j['db_n']=np.where(valid, cnt[np.clip(slot,0,None)], np.nan)
j['db_nuniq']=np.where(valid, nu[np.clip(slot,0,None)], np.nan)
j['db_min']=np.where(valid, mn[np.clip(slot,0,None)], np.nan)
j['db_max']=np.where(valid, mx[np.clip(slot,0,None)], np.nan)
j['db_mean']=np.where(valid, mean[np.clip(slot,0,None)], np.nan)
j['db_bde']=j['db_mean']
j['db_bt']=np.where(valid, bt_first[np.clip(slot,0,None)], None)
j['db_bt_nuniq']=np.where(valid, bt_nu[np.clip(slot,0,None)], np.nan)
j['in_db_row']=valid
j['db_spread']=j['db_max']-j['db_min']
j.to_pickle('/tmp/bench_joined.pkl')
print("in_db_row", int(valid.sum()), "/", len(j), round(time.time()-t0,1), flush=True)

np.savez('/tmp/db_group.npz', mn=mn,mx=mx,cnt=cnt,mean=mean,nu=nu,bt_nu=bt_nu,nkeys=np.array([len(uniq)]))
json.dump({"n_unique_keys":int(len(uniq)), "n_db_rows":int(len(db)),
   "multi_row_keys":int((cnt>1).sum()), "rows_on_multi_row_keys":int(cnt[cnt>1].sum()),
   "multi_row_keys_with_multi_bde":int((nu[cnt>1]>1).sum()),
   "max_within_key_spread":float(np.max(mx-mn))}, open('/tmp/ws5_join.json','w'), indent=2)
print("DONE", round(time.time()-t0,1), flush=True)

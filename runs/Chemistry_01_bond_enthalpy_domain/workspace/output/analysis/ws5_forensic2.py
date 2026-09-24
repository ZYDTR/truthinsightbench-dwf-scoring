import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json, time
t0=time.time(); out={}
z = np.load('/tmp/db_group.npz', allow_pickle=True)
mn,mx,cnt,nu,bt_nu = z['mn'],z['mx'],z['cnt'],z['nu'],z['bt_nu']
u1,z_u2,u3 = z['u1'],z['u2'],z['u3']
n2=int(z['n2'][0]); n3=int(z['n3'][0]); nk=len(mn)
db = pd.read_pickle('/tmp/db.pkl')
i1={v:i for i,v in enumerate(u1)}; i2={v:i for i,v in enumerate(z_u2)}; i3={v:i for i,v in enumerate(u3)}
c1 = db.molecule.map(i1).to_numpy(np.int64); c2 = db.fragment1.map(i2).to_numpy(np.int64); c3 = db.fragment2.map(i3).to_numpy(np.int64)
comb = (c1*n2 + c2)*n3 + c3
uniq = np.load('/tmp/db_uniq.npy'); inv = np.searchsorted(uniq, comb)
bde = db.bde.to_numpy(float); bde_r = np.round(bde,6)
print("codes", round(time.time()-t0,1), flush=True)
out["db_key_multiplicity"] = {"n_rows":int(len(db)),"n_unique_keys":int(nk),
  "n_multi_row_keys":int((cnt>1).sum()),"rows_on_multi_row_keys":int(cnt[cnt>1].sum()),
  "max_rows_per_key":int(cnt.max()),"pct_rows_on_multi_row_keys":float(cnt[cnt>1].sum()/len(db))}
mp = cnt>1
# varying bond_index via lexsort
bidx = db.bond_index.to_numpy(np.int64)
o=np.lexsort((bidx,inv)); iv=inv[o]; vv=bidx[o]
nw=np.empty(len(iv),bool); nw[0]=True; nw[1:]=(iv[1:]!=iv[:-1])|(vv[1:]!=vv[:-1])
bi_nu=np.bincount(iv,weights=nw.astype(float)).astype(int)
dsnv = np.nan_to_num(db.delta_stereo.to_numpy(float), nan=-999.0)
o=np.lexsort((dsnv,inv)); iv=inv[o]; vv=dsnv[o]
nw=np.empty(len(iv),bool); nw[0]=True; nw[1:]=(iv[1:]!=iv[:-1])|(vv[1:]!=vv[:-1])
ds_nu=np.bincount(iv,weights=nw.astype(float)).astype(int)
out["db_multi_key_varying"] = {"n_multi_keys":int(mp.sum()),"varying_bond_index":int((bi_nu[mp]>1).sum()),
  "varying_delta_stereo":int((ds_nu[mp]>1).sum()),"varying_bond_type":int((bt_nu[mp]>1).sum()),
  "varying_bde":int((nu[mp]>1).sum())}
out["db_key_multiplicity_distribution"]={str(int(k)):int(v) for k,v in zip(*np.unique(cnt,return_counts=True))}
print(json.dumps(out["db_multi_key_varying"]), flush=True)

# ---- reference vs db bde value sets, integer fpair code -------------------
fpc = c2*n3 + c3
o = np.lexsort((bde_r, fpc)); f_s=fpc[o]; b_s=bde_r[o]
from collections import defaultdict
sets=defaultdict(set)
prev=-1
for a,bb in zip(f_s.tolist(), b_s.tolist()): sets[a].add(bb)
print("sets built", round(time.time()-t0,1), flush=True)
b = common.load_benchmark()
bi2 = b.fragment_a_smiles.map(i2); bi3 = b.fragment_b_smiles.map(i3)
bfpc = (bi2*n3 + bi3).to_numpy(float)
refr = np.round(b.reference_estimate_bde_kcal_mol.to_numpy(float),6)
allbde = set(bde_r.tolist())
hit_pair = sum(1 for c,rv in zip(bfpc,refr.tolist()) if not np.isnan(c) and rv in sets.get(int(c),()))
hit_any = sum(1 for rv in refr.tolist() if rv in allbde)
print("ref in db same-fpair bde:", hit_pair, "/543", flush=True)
print("ref in db any bde:", hit_any, "/543", flush=True)
out["ref_found_in_db_same_fpair"]=int(hit_pair); out["ref_found_in_db_anywhere"]=int(hit_any)
out["db_distinct_bde_values"]=int(db.bde.nunique())
expset=set(np.round(b.experimental_bde_kcal_mol.to_numpy(float),6).tolist())
out["ref_found_in_experimental_values"]=int(sum(1 for x in refr.tolist() if x in expset))
print("ref in experimental values:", out["ref_found_in_experimental_values"], flush=True)
# what IS the reference nearest to?
print("DONE", round(time.time()-t0,1))
json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_forensic2.json','w'), indent=2, default=float)

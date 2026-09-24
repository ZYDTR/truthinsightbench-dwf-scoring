import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json, time
import numpy as np, pandas as pd, common
t0=time.time()
db = pd.read_pickle('/tmp/db.pkl')
print("loaded", round(time.time()-t0,1), flush=True)
c1,u1 = pd.factorize(db.molecule); c2,u2 = pd.factorize(db.fragment1); c3,u3 = pd.factorize(db.fragment2)
n2,n3 = np.int64(len(u2)), np.int64(len(u3))
comb = (c1.astype(np.int64)*n2 + c2.astype(np.int64))*n3 + c3.astype(np.int64)
uniq, inv = np.unique(comb, return_inverse=True)
bde = db.bde.to_numpy(float)
s = pd.Series(bde).groupby(inv)
mn,mx,cnt,mean = s.min().to_numpy(), s.max().to_numpy(), s.size().to_numpy(), s.mean().to_numpy()
order = np.lexsort((bde, inv)); iv=inv[order]; bv=bde[order]
new = np.empty(len(iv), bool); new[0]=True; new[1:] = (iv[1:]!=iv[:-1])|(bv[1:]!=bv[:-1])
nu = np.bincount(iv, weights=new.astype(float)).astype(int)
# bond_type first-per-key and n-distinct via factorize+lexsort
bt_codes, bt_uniques = pd.factorize(db.bond_type)
o2 = np.lexsort((bt_codes, inv)); iv2=inv[o2]; bc2=bt_codes[o2]
new2 = np.empty(len(iv2),bool); new2[0]=True; new2[1:]=(iv2[1:]!=iv2[:-1])
first_bt_code = bc2[new2]
bt_nu = np.bincount(iv2, weights=new2.astype(float)).astype(int)
print("group stats", round(time.time()-t0,1), flush=True)
np.savez_compressed('/tmp/db_group.npz', mn=mn,mx=mx,cnt=cnt,mean=mean,nu=nu,bt_nu=bt_nu,
                    first_bt_code=first_bt_code, bt_uniques=np.array(bt_uniques,dtype=object), nkeys=np.array([len(uniq)]),
                    u1=np.array(u1,dtype=object), u2=np.array(u2,dtype=object), u3=np.array(u3,dtype=object),
                    n2=np.array([n2]), n3=np.array([n3]))
np.save('/tmp/db_uniq.npy', uniq)
print("saved", round(time.time()-t0,1), flush=True)
stats = {"n_db_rows":int(len(db)), "n_unique_keys":int(len(uniq)),
  "multi_row_keys":int((cnt>1).sum()), "rows_on_multi_row_keys":int(cnt[cnt>1].sum()),
  "multi_row_keys_with_multi_bde":int((nu[cnt>1]>1).sum()),
  "max_within_key_spread":float(np.max(mx-mn)),
  "multi_keys_varying_bt":int((bt_nu[cnt>1]>1).sum())}
json.dump(stats, open('/tmp/ws5_join_stats.json','w'), indent=2)
print(json.dumps(stats, indent=2), flush=True)
print("DONE A", round(time.time()-t0,1), flush=True)

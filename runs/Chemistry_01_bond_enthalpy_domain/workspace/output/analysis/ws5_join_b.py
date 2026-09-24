import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import json, time
import numpy as np, pandas as pd, common
t0=time.time()
bench = common.load_benchmark()
z = np.load('/tmp/db_group.npz', allow_pickle=True)
uniq = np.load('/tmp/db_uniq.npy')
mn,mx,cnt,mean,nu = z['mn'],z['mx'],z['cnt'],z['mean'],z['nu']
bt_nu = z['bt_nu']; first_bt_code = z['first_bt_code']; bt_uniques = z['bt_uniques']
u1 = {v:i for i,v in enumerate(z['u1'])}; u2 = {v:i for i,v in enumerate(z['u2'])}; u3 = {v:i for i,v in enumerate(z['u3'])}
n2 = int(z['n2'][0]); n3 = int(z['n3'][0])
print("loaded cache", round(time.time()-t0,1), "nkeys", len(uniq), flush=True)

codes = np.fromiter(
  ((u1[m]*n2 + u2[a])*n3 + u3[bb] if (m in u1 and a in u2 and bb in u3) else -1
   for m,a,bb in zip(bench.parent_structure_smiles, bench.fragment_a_smiles, bench.fragment_b_smiles)),
  dtype=np.int64, count=len(bench))
print("bench codes", round(time.time()-t0,1), flush=True)
slot = np.searchsorted(uniq, codes)
slot = np.where((slot < len(uniq)) & (uniq[np.clip(slot,0,len(uniq)-1)] == codes), slot, -1)
valid = slot >= 0
print("matched", int(valid.sum()), "of", len(bench), round(time.time()-t0,1), flush=True)
sl = np.clip(slot, 0, None)
j = bench.copy()
j['db_n']=np.where(valid, cnt[sl], np.nan)
j['db_nuniq']=np.where(valid, nu[sl], np.nan)
j['db_min']=np.where(valid, mn[sl], np.nan)
j['db_max']=np.where(valid, mx[sl], np.nan)
j['db_mean']=np.where(valid, mean[sl], np.nan)
j['db_bde']=j['db_mean']
j['db_bt']=np.where(valid, np.array(bt_uniques,dtype=object)[first_bt_code[sl]], None)
j['in_db_row']=valid
j['db_spread']=j['db_max']-j['db_min']
j.to_pickle('/tmp/bench_joined.pkl')
print("in_db_row", int(valid.sum()), "/", len(j), "saved", round(time.time()-t0,1), flush=True)

# sanity: does the recovered db_bt match our own bond_class labeler?
bc = pd.Series([common.bond_class(a,b) for a,b in zip(bench.fragment_a_smiles, bench.fragment_b_smiles)])
agree = int((bc[j.in_db_row] == pd.Series(j.db_bt[j.in_db_row])).sum())
print(f"labeler agreement with db bond_type on matched: {agree}/{int(valid.sum())}", flush=True)
json.dump({"n_matched":int(valid.sum()), "n_unmatched":int((~valid).sum()),
           "n_matched_multi_db_rows":int((j.db_n>1).sum()),
           "n_matched_multi_distinct_bde":int((j.db_nuniq>1).sum()),
           "max_matched_spread":float(j.db_spread.max()),
           "labeler_agreement":agree}, open('/tmp/ws5_join_b.json','w'), indent=2)
print("DONE B", round(time.time()-t0,1), flush=True)

import sys; sys.path.insert(0,'/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/analysis')
import numpy as np, pandas as pd, common, json
j = pd.read_pickle('/tmp/bench_joined.pkl')
out={}
exp=j.experimental_bde_kcal_mol.to_numpy(float)
q=j.quantum_estimate_bde_kcal_mol.to_numpy(float)
l=j.learned_estimate_bde_kcal_mol.to_numpy(float)
r=j.reference_estimate_bde_kcal_mol.to_numpy(float)
db=j.db_bde.to_numpy(float); inb=j.in_db_row.to_numpy()

def desc(x):
    x=np.asarray(x,float); a=np.abs(x)
    return {"n":int(len(x)),"mean":float(x.mean()),"sd":float(x.std(ddof=1)),"median":float(np.median(x)),
        "mean_abs":float(a.mean()),"median_abs":float(np.median(a)),"max_abs":float(a.max()),
        "q05":float(np.quantile(x,.05)),"q95":float(np.quantile(x,.95)),
        "frac_abs_le_0.5":float((a<=.5).mean()),"frac_abs_le_1":float((a<=1).mean()),
        "frac_abs_le_2":float((a<=2).mean()),"frac_abs_le_5":float((a<=5).mean())}
out['disagreement_all543'] = {k:desc(v) for k,v in {
  'quantum_minus_learned':q-l,'quantum_minus_reference':q-r,'learned_minus_reference':l-r,
  'quantum_minus_experimental':q-exp,'learned_minus_experimental':l-exp,'reference_minus_experimental':r-exp}.items()}
print("=== channel-vs-channel disagreement (543 bond observations) ===")
for k,v in out['disagreement_all543'].items():
    print(f"  {k:28s} mean={v['mean']:8.4f} sd={v['sd']:8.4f} MAE={v['mean_abs']:7.4f} medAE={v['median_abs']:7.4f} max={v['max_abs']:7.4f} <=1:{v['frac_abs_le_1']:.3f} <=2:{v['frac_abs_le_2']:.3f}")

out['mae_vs_experimental']={c:{"all543":float(np.abs(v-exp).mean()),"matched331":float(np.abs(v[inb]-exp[inb]).mean()),
   "unmatched212":float(np.abs(v[~inb]-exp[~inb]).mean()), "rmse_all543":float(np.sqrt(np.mean((v-exp)**2)))}
   for c,v in [('quantum',q),('learned',l),('reference',r)]}
print("\n=== MAE vs experimental ===")
for c,d in out['mae_vs_experimental'].items():
    print(f"  {c:10s} all={d['all543']:.4f} (RMSE {d['rmse_all543']:.4f})  matched={d['matched331']:.4f}  unmatched={d['unmatched212']:.4f}")

# paired tests: |err| differences
out['paired_abs_error_tests']={}
def ae(c): return np.abs({'quantum':q,'learned':l,'reference':r}[c]-exp)
print("\n=== paired |error| comparisons (543 bond obs) ===")
for a,b in [('quantum','learned'),('quantum','reference'),('learned','reference')]:
    d = ae(a)-ae(b); m,lo,hi = common.paired_bootstrap_ci(d); t,p = common.paired_t_test(d)
    w,pw = common.wilcoxon(d)
    cd = common.cliffs_delta(ae(a),ae(b))
    out['paired_abs_error_tests'][f'{a}_vs_{b}']={"n":int(len(d)),"mean_diff_abs_err":m,"ci_lo":lo,"ci_hi":hi,
       "t":t,"p_ttest":p,"wilcoxon_W":w,"p_wilcoxon":pw,"cliffs_delta":cd,"median_diff":float(np.median(d)),
       "n_a_better":int((d<0).sum()),"n_b_better":int((d>0).sum()),"n_ties":int((d==0).sum())}
    print(f"  MAE({a})-MAE({b}) = {m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p_t={p:.3g} p_w={pw:.3g} delta={cd:+.4f} "
          f"({a} better {int((d<0).sum())}, {b} better {int((d>0).sum())})")

# same on the 331 matched subset
inb_m = inb
out['paired_abs_error_tests_matched331']={}
for a,b in [('quantum','learned'),('quantum','reference'),('learned','reference')]:
    d = (ae(a)-ae(b))[inb_m]; m,lo,hi = common.paired_bootstrap_ci(d); t,p = common.paired_t_test(d)
    _,pw = common.wilcoxon(d)
    out['paired_abs_error_tests_matched331'][f'{a}_vs_{b}']={"n":int(len(d)),"mean_diff_abs_err":m,"ci_lo":lo,"ci_hi":hi,
       "p_ttest":p,"p_wilcoxon":pw,"n_a_better":int((d<0).sum()),"n_b_better":int((d>0).sum())}
    print(f"  [331] MAE({a})-MAE({b}) = {m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] p_t={p:.3g} p_w={pw:.3g}")

json.dump(out, open('/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project/output/results/ws5_channel_stats.json','w'), indent=2, default=float)
print("\nDONE")

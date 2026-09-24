import numpy as np, pandas as pd
from scipy import stats
df=pd.read_csv('/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv',sep='\t')
m=df.c08_age_a_ad_ce.notna().values
age=df.c08_age_a_ad_ce.values[m]; d1=df.c01_depth_top_m.values[m]; d6=df.c06_depth_w_e_m.values[m]
def des(f,x): return np.column_stack([np.ones_like(np.asarray(x,float)),f(np.asarray(x,float))])
def loo(f,x,y):
    e=np.empty(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool); tr[i]=False
        X=des(f,x[tr]); b,*_=np.linalg.lstsq(X,y[tr],rcond=None)
        e[i]=abs(np.asarray(des(f,x[i:i+1])@b,float).ravel()[0]-y[i])
    return e
e_pow=loo(lambda v:v**1.25,d1,age)
e_c06=loo(lambda v:v,d6,age)
e_c01=loo(lambda v:v,d1,age)
e_quad=loo(lambda v:np.column_stack([v,v**2]),d1,age)
print("MAE c01^1.25 = %.3f ; c06 = %.3f ; c01 = %.3f ; c01-quad = %.3f"%(e_pow.mean(),e_c06.mean(),e_c01.mean(),e_quad.mean()))
print("c01^1.25 beats c06 on %d/15 ; beats c01 on %d/15"%((e_pow<e_c06).sum(),(e_pow<e_c01).sum()))
print("sign test c01^1.25 vs c06 two-sided p=%.4g"%stats.binomtest(int((e_pow<e_c06).sum()),15,0.5,alternative='two-sided').pvalue)
order=np.argsort(d1); blocks=[slice(0,3),slice(3,6),slice(6,9),slice(9,12),slice(12,15)]
ep,ec=e_pow[order],e_c06[order]
print("blocks where c01^1.25 wins: %d/5"%(sum(1 for s in blocks if ep[s].mean()<ec[s].mean())))
rng=np.random.default_rng(7); dd=[]
for _ in range(5000):
    picks=rng.integers(0,5,5); ii=np.concatenate([[np.arange(s.start,s.stop) for s in blocks][p] for p in picks])
    dd.append(ec[ii].mean()-ep[ii].mean())
dd=np.array(dd); print("block-bootstrap MAE(c06)-MAE(c01^1.25): median %.2f yr, 95%% CI [%.2f, %.2f]"%(np.median(dd),np.percentile(dd,2.5),np.percentile(dd,97.5)))
# same for c06 vs c01
ecc=e_c06[order]; e11=e_c01[order]; dd2=[]
for _ in range(5000):
    picks=rng.integers(0,5,5); ii=np.concatenate([[np.arange(s.start,s.stop) for s in blocks][p] for p in picks])
    dd2.append(e11[ii].mean()-ecc[ii].mean())
dd2=np.array(dd2); print("block-bootstrap MAE(c01)-MAE(c06): median %.2f yr, 95%% CI [%.2f, %.2f]"%(np.median(dd2),np.percentile(dd2,2.5),np.percentile(dd2,97.5)))

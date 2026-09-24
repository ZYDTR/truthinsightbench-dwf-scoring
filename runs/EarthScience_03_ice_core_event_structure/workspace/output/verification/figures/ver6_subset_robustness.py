import numpy as np, pandas as pd
df=pd.read_csv('/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv',sep='\t')
m=df.c08_age_a_ad_ce.notna().values
sub=df[m].reset_index(drop=True)
def des(f,x): return np.column_stack([np.ones_like(np.asarray(x,float)),f(np.asarray(x,float))])
def loo(f,x,y):
    e=np.empty(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool); tr[i]=False
        b,*_=np.linalg.lstsq(des(f,x[tr]),y[tr],rcond=None)
        e[i]=abs(np.asarray(des(f,x[i:i+1])@b,float).ravel()[0]-y[i])
    return e
for label,sel in [("all 15", np.ones(15,bool)),
                  ("10 unidentified only", (sub.c07_signal=="unidentified_event").values),
                  ("5 named events only", (sub.c07_signal!="unidentified_event").values)]:
    a=sub.c08_age_a_ad_ce.values[sel]; d1=sub.c01_depth_top_m.values[sel]; d6=sub.c06_depth_w_e_m.values[sel]
    n=sel.sum()
    if n<4: 
        print(label,"n=%d too small"%n); continue
    e25=loo(lambda v:v**1.25,d1,a); e6=loo(lambda v:v,d6,a); e1=loo(lambda v:v,d1,a)
    print("%-24s n=%2d  c01^1.25 MAE=%7.2f  c06 MAE=%7.2f  c01 MAE=%7.2f  c01^1.25 wins %d/%d"%(
        label,n,e25.mean(),e6.mean(),e1.mean(),int((e25<e6).sum()),n))

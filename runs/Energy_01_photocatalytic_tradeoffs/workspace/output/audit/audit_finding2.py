import numpy as np, pandas as pd, re
D="/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/optical_action_spectra.tsv"
df=pd.read_csv(D,sep="\t")
ab=df[df.measurement_role=="absorption"].sort_values("wavelength_nm")
ac=df[df.measurement_role=="action_response"].sort_values("wavelength_nm")
A=ab.wavelength_nm.values; yA=ab.response_au_or_percent.values
wl=ac.wavelength_nm.values; act=ac.response_au_or_percent.values
print("action wl:",wl,"vals:",act)
print("abs at action:",[float(yA[np.isclose(A,w)][0]) for w in wl])
a=np.array([float(yA[np.isclose(A,w)][0]) for w in wl])
y=act.copy()
# free-scale through origin fit on all six
s=np.sum(a*y)/np.sum(a*a); pred=s*a; resid=y-pred
ss=1-np.sum(resid**2)/np.sum((y-y.mean())**2); rmse=np.sqrt(np.mean(resid**2))
print("FULL FIT s=%.6f r2=%.6f rmse=%.6f"%(s,ss,rmse))
print("pred:",np.round(pred,4)); print("resid:",np.round(resid,4))
print("pred650/obs650=%.6f pred650=%.6f"%(pred[-1]/y[-1],pred[-1]))
# ratios
def R(p,r): return (y[np.argmin(abs(wl-p))]/y[np.argmin(abs(wl-r))])/(a[np.argmin(abs(wl-p))]/a[np.argmin(abs(wl-r))])
print("400->650 %.6f  450->650 %.6f"%(R(650,400),R(650,450)))
print("action 650/450=%.6f abs 650/450=%.6f"%(y[-1]/y[1],a[-1]/a[1]))
# all 15 forward/backward pairs
pairs={}
for i,p in enumerate(wl):
    for j,r in enumerate(wl):
        if i!=j: pairs[(int(r),int(p))]=R(p,r)
fwd=[(k,v) for k,v in pairs.items() if k[1]>k[0] or k==(450,400)]
print("num pairs total",len(pairs))
probe_ge600=[v for k,v in pairs.items() if k[1]>=600]
print("pairs with probe>=600 (all ordered):",len(probe_ge600),"max",max(probe_ge600),"all<1",all(v<1 for v in probe_ge600))
fwd_ge600=[v for k,v in pairs.items() if k[1]>=600 and k[1]>k[0]]
print("forward probe>=600:",len(fwd_ge600),"max",max(fwd_ge600))
short=[(k,round(v,6)) for k,v in pairs.items() if k[1]<=500 and k[1]>k[0]]
print("forward probe<=500:",short)
# holdout <=500
m=wl<=500; sh=np.sum(a[m]*y[m])/np.sum(a[m]**2)
hold_pred=sh*a[~m]
print("holdout s=%.6f preds=%s obs=%s relerr=%s"%(sh,np.round(hold_pred,4),y[~m],np.round((hold_pred-y[~m])/y[~m]*100,3)))
# k-grid inequality
A450=float(a[1]); A650=float(a[-1])
def alpha(x): return 1-10.0**(-x)
ks=np.logspace(-9,3,250)
ratios=[alpha(k*A650)/alpha(k*A450) for k in ks]
print("A450=%.4f A650=%.4f raw ratio=%.8f"%(A450,A650,A650/A450))
print("min alpha(kA650)/alpha(kA450) over grid=%.8f at k=%.2e (at k->0 it equals raw)"%(min(ratios),ks[int(np.argmin(ratios))]))
print("any below raw ratio? ", min(ratios)<A650/A450-1e-12)
print("monotone increasing in k:", all(ratios[i]<=ratios[i+1]+1e-12 for i in range(len(ratios)-1)))
# check the report's literal inequality direction
viol=[(k,alpha(k*A450)/alpha(k*A650)) for k in ks]
print("literal alpha(kA450)/alpha(kA650): min %.4f max %.4f vs A450/A650=%.4f ; literal claim (>=A450/A650) holds?"%(min(v for _,v in viol),max(v for _,v in viol),A450/A650, all(v>=A450/A650-1e-9 for _,v in viol)))
# same but for any increasing concave T with T(0)=0 e.g. sqrt, checked
# precision check
print("action strings dp:",sorted(set(len(str(x).split('.')[-1]) for x in open(D).read().splitlines() if 'action_response' in x)))
import collections
ab_lines=[l for l in open(D).read().splitlines() if l.startswith('absorption')]
dpc=collections.Counter()
for l in ab_lines:
    v=l.split('\t')[2]
    dpc[len(v.split('.')[-1]) if '.' in v else 0]+=1
print("absorption dp hist:",dict(dpc))
print("absorption raw strings at action wl:",[l.split('\t')[2] for l in ab_lines for w in wl if abs(float(l.split('\t')[1])-w)<1e-9])

"""Explore possible 0.0083 / 0.0676 derivations for the Finding-1 audit."""
import numpy as np, json, os
ROOT="/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project"
raw={}
with open(os.path.join(ROOT,"data/conversion_timecourses.tsv")) as fh:
    fh.readline()
    for line in fh:
        f,t,y=line.split(); raw.setdefault(f,([],[])); raw[f][0].append(float(t)); raw[f][1].append(float(y))
t=np.array(raw['formulation_1'][0]); y={f:np.array(v[1]) for f,v in raw.items()}
f1,f2,f3,f4=y['formulation_1'],y['formulation_2'],y['formulation_3'],y['formulation_4']

# log-space RMS of normalised curves
def logrms(a,b):
    an,bn=a/a[-1],b/b[-1]
    la=np.log(np.clip(an,1e-12,None)); lb=np.log(np.clip(bn,1e-12,None))
    return float(np.sqrt(np.mean((la-lb)**2)))
print('log-rms 3v4',logrms(f3,f4),'1v3',logrms(f1,f3),'1v2',logrms(f1,f2),'1v4',logrms(f1,f4))

# pointwise fractional difference RMS (relative error normalised by the mean/final)
for name,(a,b) in {'3v4':(f3,f4),'1v3':(f1,f3),'1v2':(f1,f2)}.items():
    m=(a[-1]+b[-1])/2
    print('frac-of-mean rms',name,float(np.sqrt(np.mean(((a-b)/m)**2))))

# normalised by each curve's own final: diff RMS as fraction of curve? (relative)
for name,(a,b) in {'3v4':(f3,f4),'1v3':(f1,f3),'1v2':(f1,f2)}.items():
    an,bn=a/a[-1],b/b[-1]
    print('rel-to-own-scale rms', name, float(np.sqrt(np.mean(((a-b)/np.maximum((a+b)/2,1e-12))**2))))

# Does raw artifact contain 0.0083 anywhere? search all json/csv under output
import subprocess
print(subprocess.run(['grep','-rn','0.0083',os.path.join(ROOT,'output','results'),os.path.join(ROOT,'output','verification')],capture_output=True,text=True).stdout)

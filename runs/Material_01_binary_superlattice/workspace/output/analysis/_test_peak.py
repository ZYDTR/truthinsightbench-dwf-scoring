#!/usr/bin/env python3
import re, json
from pathlib import Path
import numpy as np

BASE = Path("/data/user/conversations/conv-d944711d-fa11-472b-99db-a1344a7c93b2/project")
man = json.loads((BASE/"data/structure_manifest.json").read_text())
HDR = re.compile(r"Box dimension:\s*(\d+(?:\.\d+)?)\(p\)")
FBOX = re.compile(r"\*?\d+(?:\.\d+)?\(f\)")

def parse(path):
    lines = path.read_text().splitlines()
    parts=[]; box=None; fbox=None
    for i,l in enumerate(lines):
        s=l.strip()
        if not s: continue
        if i==0 and re.match(r"^\d+\s*$", s): continue
        if s.startswith("R1=") or "Energy" in s:
            m=HDR.search(s); box=float(m.group(1)) if m else None
            fm=re.search(r"\d+(?:\.\d+)?\(f\)", s); fbox=float(fm.group(0).split("(")[0]) if fm else None
            continue
        p=s.split()
        if len(p)>=4:
            try: parts.append((p[0],float(p[1]),float(p[2]),float(p[3])))
            except: pass
    return box,fbox,parts

def first_peak(distances, bin0=0.0, width=0.5):
    d=np.array([x for x in distances if np.isfinite(x) and x>0])
    hist, edges = np.histogram(d, bins=np.arange(bin0, d.max()+width, width))
    # first local max: first i>=1 with hist[i] >= hist[i-1] and hist[i] > hist[i+1]
    for i in range(1,len(hist)-1):
        if hist[i] >= hist[i-1] and hist[i] > hist[i+1]:
            return (edges[i]+edges[i+1])/2, hist
    # fallback first drop
    for i in range(len(hist)-1):
        if hist[i] > hist[i+1]:
            return (edges[i]+edges[i+1])/2, hist
    return (edges[-2]+edges[-1])/2, hist

# Test structure_01
p = (BASE/"data/structures/structure_01.xyz").read_text().splitlines()
box,fbox,pts = parse(BASE/"data/structures/structure_01.xyz")
print("box p =", box, "box f =", fbox, "n =", len(pts))
N=len(pts)
def allpairs(pts, box, fbox, zfree):
    dlist=[]
    for a in range(N):
        for b in range(N):
            if a>=b: continue
            dx=pts[b][1]-pts[a][1]; dy=pts[b][2]-pts[a][2]; dz=pts[b][3]-pts[a][3]
            dx=((dx+box/2)%box)-box/2
            dy=((dy+box/2)%box)-box/2
            if not zfree:
                dz=((dz+fbox/2)%(fbox))-fbox/2
            dlist.append((dx*dx+dy*dy+dz*dz)**0.5)
    return dlist

for zfree in (True,False):
    dlist=allpairs(pts,box,fbox,zfree)
    c,h=first_peak(dlist)
    print(f"zfree={zfree}: first_peak={c}")

# also show histogram around 6.0-6.5 for zfree=True
dlist=allpairs(pts,box,fbox,True)
hist,edges=np.histogram(dlist, bins=np.arange(0, dlist.max()+0.5, 0.5))
print("hist bins 10-16 (edges):", [(i, edges[i], round(hist[i],1)) for i in range(10,17)])

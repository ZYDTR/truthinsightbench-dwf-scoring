#!/usr/bin/env python3
import re, json
from pathlib import Path
import numpy as np

BASE = Path("/data/user/conversations/conv-d944711d-fa11-472b-99db-a1344a7c93b2/project")
man = json.loads((BASE/"data/structure_manifest.json").read_text())
HDR = re.compile(r"Box dimension:\s*(\d+(?:\.\d+)?)\(p\)")

def parse(path):
    lines = path.read_text().splitlines()
    pts=[]; box=None
    for i,l in enumerate(lines):
        s=l.strip()
        if not s: continue
        if i==0 and re.match(r"^\d+\s*$", s): continue
        if s.startswith("R1=") or "Energy" in s:
            m=HDR.search(s); box=float(m.group(1)) if m else None
            continue
        p=s.split()
        if len(p)>=4:
            try: pts.append((p[0],float(p[1]),float(p[2]),float(p[3])))
            except: pass
    return box, pts

def first_peak(dlist, width=0.5):
    d=np.array([x for x in dlist if np.isfinite(x) and x>0])
    hist, edges = np.histogram(d, bins=np.arange(0.0, d.max()+width, width))
    for i in range(1,len(hist)-1):
        if hist[i] >= hist[i-1] and hist[i] > hist[i+1]:
            return (edges[i]+edges[i+1])/2
    return (edges[-2]+edges[-1])/2

for pid in ["structure_04","structure_03","structure_02","structure_01"]:
    box, pts = parse(BASE / f"data/structures/{pid}.xyz")
    types=[p[0] for p in pts]
    N=len(pts)
    dlist=[]
    for a in range(N):
        for b in range(N):
            if a>=b: continue
            dx=((pts[b][1]-pts[a][1]+box/2)%box)-box/2
            dy=((pts[b][2]-pts[a][2]+box/2)%box)-box/2
            dz=pts[b][3]-pts[a][3]
            dlist.append((dx*dx+dy*dy+dz*dz)**0.5)
    c=first_peak(dlist)
    same=cross=0
    for a in range(N):
        for b in range(N):
            if a>=b: continue
            dx=((pts[b][1]-pts[a][1]+box/2)%box)-box/2
            dy=((pts[b][2]-pts[a][2]+box/2)%box)-box/2
            dz=pts[b][3]-pts[a][3]
            d=(dx*dx+dy*dy+dz*dz)**0.5
            if d <= c:
                if types[a]==types[b]: same+=1
                else: cross+=1
    frac = same/(same+cross) if (same+cross) else 0.0
    print(pid, "c=", round(c,3), "same=", same, "cross=", cross, "frac=", round(frac,4))
    # check min pair distance
    print("  min pair dist =", round(min(dlist),3))

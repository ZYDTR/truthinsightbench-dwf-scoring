import re
from pathlib import Path
data = Path("data")
def test(p):
    lines = p.read_text().splitlines()
    box=None
    for i,line in enumerate(lines):
        line=line.strip()
        if not line: continue
        if i==0 and re.match(r"^\d+\s*$",line): continue
        if line.startswith("R1=") or line.startswith("Box") or "Energy" in line:
            m=re.search(r"Box dimension:\s*(\d+(?:\.\d+)?)(p)",line)
            if m: box=float(m.group(1))
    return box
for f in sorted((data/"structures").glob("structure_*.xyz")):
    print(f.name, "-> box =", test(f))

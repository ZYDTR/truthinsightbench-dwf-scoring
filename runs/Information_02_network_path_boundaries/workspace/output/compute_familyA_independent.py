#!/usr/bin/env python3
"""INDEPENDENT recomputation for Family A (strongly-connected flower).

Reads ONLY the raw observations JSON and the reference .net graph, and
recomputes the three claims required by the task. It does not import or
re-use analysis_flower.py.
"""
import json
from collections import defaultdict, deque

JSON_PATH = "data/source_data/computational_observations.json"
NET_PATH = "data/source_data/Figs/directed_sparse_reference.net"
N_LIST = [6, 8, 10, 12]


def load_actions():
    with open(JSON_PATH) as fh:
        data = json.load(fh)
    return data["actions"][0]["cases"]  # Family A cases


def parse_net(path):
    """Parse '*Vertices N' header + 'u v' arc lines."""
    nodes = None
    arcs = []
    in_arcs = False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("note"):
                continue
            if line.startswith("*Vertices"):
                parts = line.split()
                nodes = int(parts[1])
                in_arcs = False
                continue
            if line.startswith("*Arcs") or line.startswith("*Edges"):
                in_arcs = True
                continue
            if in_arcs:
                parts = line.split()
                if len(parts) >= 2:
                    arcs.append((int(parts[0]), int(parts[1])))
    return nodes, arcs


def bfs_distances(nodes, arcs):
    """Return dist[source][target] = shortest directed distance (or absent)."""
    adj = defaultdict(list)
    for u, v in arcs:
        adj[u].append(v)
    dists = {}
    for s in range(1, nodes + 1):
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(w)
        dists[s] = d
    return dists


def bfs_metrics(nodes, arcs):
    """avg path length, global efficiency (unreachable = 0), diameter."""
    dists = bfs_distances(nodes, arcs)
    finite = []
    for i in range(1, nodes + 1):
        for j in range(1, nodes + 1):
            if i == j:
                continue
            if j in dists[i]:
                finite.append(dists[i][j])
    n_pairs = nodes * (nodes - 1)
    avg_path = sum(finite) / n_pairs
    eff = sum(1.0 / d for d in finite) / n_pairs
    diameter = max(finite) if finite else None
    strongly_connected = all(
        j in dists[i]
        for i in range(1, nodes + 1)
        for j in range(1, nodes + 1)
        if i != j
    )
    return {
        "n_nodes": nodes,
        "n_arcs": len(arcs),
        "avg_path": avg_path,
        "global_efficiency": eff,
        "diameter": diameter,
        "strongly_connected": strongly_connected,
    }


def main():
    cases = load_actions()
    byN = defaultdict(list)
    for c in cases:
        byN[c["n_nodes"]].append(c)

    # ---------------------------------------------------------------
    # Monotonicity in L: path decreases, eff increases, diam non-increase
    # ---------------------------------------------------------------
    mono_violations = []
    for n in sorted(byN):
        rows = sorted(byN[n], key=lambda c: c["n_arcs"])
        prev = rows[0]
        for c in rows[1:]:
            if c["observed_path_length"] >= prev["observed_path_length"]:
                mono_violations.append(
                    ("path_not_decreasing", n, prev["n_arcs"], c["n_arcs"],
                     prev["observed_path_length"], c["observed_path_length"])
                )
            if c["observed_global_efficiency"] <= prev["observed_global_efficiency"]:
                mono_violations.append(
                    ("eff_not_increasing", n, prev["n_arcs"], c["n_arcs"],
                     prev["observed_global_efficiency"], c["observed_global_efficiency"])
                )
            if c["diameter"] > prev["diameter"]:
                mono_violations.append(
                    ("diameter_increasing", n, prev["n_arcs"], c["n_arcs"],
                     prev["diameter"], c["diameter"])
                )
            prev = c
    monotonicity_holds = len(mono_violations) == 0

    # ---------------------------------------------------------------
    # Cycle baseline at L=N: diam==N-1, path==N/2, eff==H_{N-1}/(N-1)
    # ---------------------------------------------------------------
    cycle_ok = True
    cycle_detail = []
    for n in N_LIST:
        rows = sorted(byN[n], key=lambda c: c["n_arcs"])
        Lmin_row = next((r for r in rows if r["n_arcs"] == n), None)
        if Lmin_row is None:
            cycle_ok = False
            cycle_detail.append(("no_row_at_L=N", n, None, n))
            continue
        H = sum(1.0 / k for k in range(1, n))  # H_{n-1}
        eff_cycle = H / (n - 1)
        if Lmin_row["diameter"] != n - 1:
            cycle_ok = False
            cycle_detail.append(("diam", n, Lmin_row["diameter"], n - 1))
        if abs(Lmin_row["observed_path_length"] - n / 2) > 1e-9:
            cycle_ok = False
            cycle_detail.append(("path", n, Lmin_row["observed_path_length"], n / 2))
        if abs(Lmin_row["observed_global_efficiency"] - eff_cycle) > 1e-9:
            cycle_ok = False
            cycle_detail.append(("eff", n, Lmin_row["observed_global_efficiency"], eff_cycle))

    # ---------------------------------------------------------------
    # diameter == 2 ONLY at L == 2N-2
    # ---------------------------------------------------------------
    diam2_L_values = []
    diam2_only_at_max = True
    for c in cases:
        if c["diameter"] == 2.0:
            diam2_L_values.append(c["n_arcs"])
            if c["n_arcs"] != 2 * c["n_nodes"] - 2:
                diam2_only_at_max = False

    # ---------------------------------------------------------------
    # Reference case N=8, L=11 from .net via BFS
    # ---------------------------------------------------------------
    nodes, arcs = parse_net(NET_PATH)
    ref = bfs_metrics(nodes, arcs)
    stored = next(
        (c for c in cases if c["n_nodes"] == 8 and c["n_arcs"] == 11), None
    )
    if stored is None:
        stored = {"observed_path_length": None,
                  "observed_global_efficiency": None, "diameter": None}

    ref_path = ref["avg_path"]
    ref_eff = ref["global_efficiency"]
    ref_diam = ref["diameter"]
    stored_path = stored["observed_path_length"]
    stored_eff = stored["observed_global_efficiency"]
    stored_diam = stored["diameter"]

    path_match = (abs(ref_path - stored_path) < 1e-12) if stored_path is not None else False
    eff_match = (abs(ref_eff - stored_eff) < 1e-12) if stored_eff is not None else False
    diam_match = (ref_diam == stored_diam) if stored_diam is not None else False
    reference_matches_stored = path_match and eff_match and diam_match

    result = {
        "family": "A_strongly_connected",
        "monotonicity_holds": monotonicity_holds,
        "monotonicity_violations": mono_violations,
        "cycle_extremum_ok": cycle_ok,
        "cycle_detail": cycle_detail,
        "diam2_only_at_max": diam2_only_at_max,
        "diam2_L_values": diam2_L_values,
        "reference": {
            "n_nodes": nodes,
            "n_arcs": len(arcs),
            "avg_path": ref_path,
            "global_efficiency": ref_eff,
            "diameter": ref_diam,
            "strongly_connected": ref["strongly_connected"],
            "stored_path": stored_path,
            "stored_eff": stored_eff,
            "stored_diam": stored_diam,
            "path_match": path_match,
            "eff_match": eff_match,
            "diam_match": diam_match,
            "reference_matches_stored": reference_matches_stored,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

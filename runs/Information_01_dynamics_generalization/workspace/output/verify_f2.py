import json

SRC = "/data/user/conversations/conv-07581fcf-2edd-4246-8da8-dc035efd1751/project/data/source_data/computational_observations.json"
OUT = "/data/user/conversations/conv-07581fcf-2edd-4246-8da8-dc035efd1751/project/output/results/verify_f2.json"
GRAPHS = ["training_graph", "equivalent_novel_graph", "higher_degree_graph"]
DIST = ["uniform_0_1", "beta_5_2", "beta_0_5_0_5_boundary"]

data = json.load(open(SRC))
pd = data["diagnostic_panels"]["within_domain"]

per_graph = {}
uniforms = []
betas = []
boundaries = []
for g in GRAPHS:
    m = {n: pd[g]["distributions"][n]["mean_percent"] for n in DIST}
    ub = m["beta_0_5_0_5_boundary"] / m["uniform_0_1"]
    b5 = m["beta_0_5_0_5_boundary"] / m["beta_5_2"]
    per_graph[g] = {"raw_means": m, "boundary_over_uniform": ub, "boundary_over_beta_5_2": b5}
    uniforms.append(m["uniform_0_1"]); betas.append(m["beta_5_2"]); boundaries.append(m["beta_0_5_0_5_boundary"])

ub_ratios = [per_graph[g]["boundary_over_uniform"] for g in GRAPHS]
b5_ratios = [per_graph[g]["boundary_over_beta_5_2"] for g in GRAPHS]
mu = sum(boundaries) / 3
mu_u = sum(uniforms) / 3
mu_b = sum(betas) / 3
summary_ub = mu / mu_u
summary_b5 = mu / mu_b
verdict = "confirmed" if (1.9 <= summary_ub <= 2.0 and 1.95 <= summary_b5 <= 1.97
                          and all(1.9 <= r <= 2.0 for r in ub_ratios)
                          and all(1.95 <= r <= 1.97 for r in b5_ratios)) else "refuted"

result = {
    "per_graph": per_graph,
    "ratio_ranges": {"boundary_over_uniform": {"min": min(ub_ratios), "max": max(ub_ratios)},
                     "boundary_over_beta_5_2": {"min": min(b5_ratios), "max": max(b5_ratios)}},
    "summary": {"uniform_mean": mu_u, "beta_5_2_mean": mu_b, "boundary_mean": mu,
                "boundary_over_uniform": summary_ub, "boundary_over_beta_5_2": summary_b5},
    "verdict": verdict,
}
json.dump(result, open(OUT, "w"), indent=2)
print(json.dumps(result, indent=2))

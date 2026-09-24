import numpy as np, pandas as pd, json, itertools

P = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/optical_action_spectra.tsv"
df = pd.read_csv(P, sep="\t")
print("roles:", df.measurement_role.value_counts().to_dict())
print("dtypes fine; any NaN:", df.isna().any().to_dict())

absn = df[df.measurement_role == "absorption"].sort_values("wavelength_nm")
act  = df[df.measurement_role == "action_response"].sort_values("wavelength_nm")

wl_abs = absn.wavelength_nm.to_numpy()
A_abs  = absn.response_au_or_percent.to_numpy()
wl_act = act.wavelength_nm.to_numpy()
A_act  = act.response_au_or_percent.to_numpy()
print("n_abs =", len(wl_abs), "range", wl_abs.min(), wl_abs.max(),
      "spacing uniq:", np.unique(np.round(np.diff(wl_abs), 9)))
print("action wl:", wl_act, "action val:", A_act)

# exact-grid lookup, assert exact coincidence
idx = {}
for w in wl_act:
    hits = np.where(np.isclose(wl_abs, w, atol=1e-9))[0]
    assert len(hits) == 1, (w, hits)
    idx[w] = hits[0]
abs_at_act = {float(w): float(A_abs[idx[w]]) for w in wl_act}
print("absorption at action wavelengths:", abs_at_act)

# ratio statistic for all ordered pairs (probe != ref)
pairs = {}
for wref, wprobe in itertools.permutations(wl_act, 2):
    num = A_act[list(wl_act).index(wprobe)] / A_act[list(wl_act).index(wref)]
    den = abs_at_act[float(wprobe)] / abs_at_act[float(wref)]
    pairs[f"{int(wref)}->{int(wprobe)}"] = float(num / den)

# best single-scale through-origin fit action = s * absorption
s_hat = float(np.sum(np.fromiter((A_abs[idx[w]] * A_act[i] for i, w in enumerate(wl_act)), float)) /
              np.sum(np.fromiter((A_abs[idx[w]] ** 2 for w in wl_act), float)))
pred = s_hat * np.array([abs_at_act[float(w)] for w in wl_act])
resid = A_act - pred
rmse = float(np.sqrt(np.mean(resid ** 2)))
ss_tot = float(np.sum((A_act - A_act.mean()) ** 2))
r2 = float(1 - np.sum(resid ** 2) / ss_tot)
# also free-intercept OLS for context
slope, intercept = np.polyfit([abs_at_act[float(w)] for w in wl_act], A_act, 1)

# holdout: fit s on wl <= 500
mask = wl_act <= 500
s_hold = float(np.sum([abs_at_act[float(w)] * A_act[i] for i, w in enumerate(wl_act) if mask[i]]) /
               np.sum([abs_at_act[float(w)] ** 2 for i, w in enumerate(wl_act) if mask[i]]))
hold = {}
for i, w in enumerate(wl_act):
    if not mask[i]:
        p = s_hold * abs_at_act[float(w)]
        hold[int(w)] = dict(observed=float(A_act[i]), predicted=float(p),
                            rel_err_pct=float(100 * (p - A_act[i]) / A_act[i]))

out = dict(
    absorption_at_action_wavelengths=abs_at_act,
    ratio_statistic_all_pairs=pairs,
    ratio_400_to_650=pairs["400->650"],
    ratio_450_to_650=pairs["450->650"],
    single_scale_fit=dict(s=s_hat, rmse=rmse, r2=r2,
                          s_free_intercept_slope=float(slope), intercept=float(intercept)),
    holdout_le500=dict(s_fit=s_hold, predictions=hold),
)
with open("/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/recomputed_values.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

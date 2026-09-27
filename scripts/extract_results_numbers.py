"""Extract the v6 numbers needed to rewrite the results chapter."""

import json
import os

import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/v6/hmm_results_improved.parquet")
codes = pd.read_csv("past-runs/v6/outputs/subject_code_mapping.csv").set_index("subject")["subject_code"]
df["code"] = df["subject"].map(codes)

# Representative subject of the thesis tables, referenced by anonymised code.
REP_CODE = os.environ.get("THESIS_REP_SUBJECT_CODE", "S15")
rep = df[df["code"] == REP_CODE]
if rep.empty:
    raise SystemExit(f"No subject with code {REP_CODE} in {sorted(codes.unique())}")

print(f"dataset: {len(df):,} obs, {df['subject'].nunique()} subjects, "
      f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"representative subject: {REP_CODE}, n={len(rep):,}\n")

col = "hmm_state_filtered"
means = rep.groupby(col)["activity"].mean().sort_values()
remap = {old: new for new, old in enumerate(means.index)}
seq = rep[col].dropna().map(remap).to_numpy().astype(int)

print("=== state occupancy (activity-ordered) ===")
occ = pd.Series(seq).value_counts(normalize=True).sort_index() * 100
for s, pct in occ.items():
    act = rep.loc[rep[col].map(remap) == s, "activity"].mean()
    print(f"  State {s}: {pct:5.2f}%   mean activity={act:.2f}")

print("\n=== transition matrix ===")
k = len(remap)
T = np.zeros((k, k))
for a, b in zip(seq[:-1], seq[1:]):
    T[a, b] += 1
T = T / T.sum(axis=1, keepdims=True)
print(pd.DataFrame(T).round(3).to_string())
print(f"  diagonal: {np.round(np.diag(T), 3)}  mean persistence={np.diag(T).mean():.3f}")

print("\n=== dwell times (representative subject) ===")
runs, vals = [], []
start = 0
for i in range(1, len(seq) + 1):
    if i == len(seq) or seq[i] != seq[start]:
        runs.append(i - start)
        vals.append(seq[start])
        start = i
d = pd.DataFrame({"state": vals, "bins": runs})
d["hours"] = d["bins"] * 0.25
print(d.groupby("state")["hours"].agg(["count", "mean", "median"]).round(2).to_string())

print("\n=== dwell times (all subjects pooled, by activity rank) ===")
allruns = []
for subj, g in df.groupby("subject"):
    m = g.groupby(col)["activity"].mean().sort_values()
    rm = {old: new for new, old in enumerate(m.index)}
    s = g[col].dropna().map(rm).to_numpy().astype(int)
    if len(s) == 0:
        continue
    st = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[st]:
            allruns.append({"state": s[st], "hours": (i - st) * 0.25})
            st = i
ar = pd.DataFrame(allruns)
print(ar.groupby("state")["hours"].agg(["count", "mean", "median", "skew"]).round(2).to_string())

print("\n=== sex effects ===")
if "sex" in df.columns:
    sub = df.groupby(["subject", "sex"])["activity_log"].mean().reset_index()
    print(sub.groupby("sex")["activity_log"].agg(["count", "mean", "std"]).round(4).to_string())
    from scipy import stats
    f = sub[sub["sex"].str.lower().str.startswith("f")]["activity_log"]
    m = sub[sub["sex"].str.lower().str.startswith("m")]["activity_log"]
    u, p = stats.mannwhitneyu(f, m)
    print(f"  Mann-Whitney U={u:.1f}, p={p:.4f}")

print("\n=== population cosinor (v6) ===")
r = json.load(open("past-runs/v6/outputs/updated_results.json", encoding="utf-8"))
pop = r["multi_component_cosinor"]["population"]
print(f"  MESOR={pop['mesor']:.4f}  R2={pop['r_squared']:.4f}  "
      f"F={pop['f_statistic']:.2f}  p={pop['p_value']:.3e}")
print(f"  single-component R2={r['multi_component_cosinor']['single_component_r2']:.4f}")
for c in pop["components"]:
    print(f"    T={c['period']:>5.1f}h  amp={c['amplitude']:.4f}  acro={c['acrophase_hours']:.2f}h")

ps = r["multi_component_cosinor"].get("per_subject", [])
if ps:
    p = pd.DataFrame(ps)
    for c in ("r_squared", "r_squared_single"):
        if c in p.columns:
            print(f"  per-subject {c}: min={p[c].min():.3f} max={p[c].max():.3f} mean={p[c].mean():.3f}")

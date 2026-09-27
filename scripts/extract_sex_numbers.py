"""Sex-effect descriptives for v6, using the same merge logic as the analytics stage."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Optional roster supplied by the data owner, used only to cross-check the sex
# assignment derived from the data files. Columns: subject,sex. Not distributed.
REFERENCE_SEX_FILE = Path("data/raw/subject_sex_reference.csv")

df = pd.read_parquet("data/processed/v6/hmm_results_improved.parquet")
codes = pd.read_csv("past-runs/v6/outputs/subject_code_mapping.csv").set_index("subject")["subject_code"]

meta = pd.read_csv("data/raw/ACT_Wet_Dry_Seasons_BR.hourly.csv", usecols=["subject", "sex"])
static = meta.groupby("subject")["sex"].first()
df["sex"] = df["subject"].map(static)

ov = pd.read_csv("data/raw/subject_sex_overrides.csv", usecols=["subject", "sex"])
ov_map = ov.drop_duplicates("subject").set_index("subject")["sex"]
df.loc[df["sex"].isna(), "sex"] = df.loc[df["sex"].isna(), "subject"].map(ov_map)

assigned = df.groupby("subject")["sex"].first()
if REFERENCE_SEX_FILE.exists():
    reference = (
        pd.read_csv(REFERENCE_SEX_FILE, usecols=["subject", "sex"])
        .drop_duplicates("subject")
        .set_index("subject")["sex"]
    )
    print("=== sex assignment vs. the reference roster ===")
    mismatch = []
    for subj, s in assigned.items():
        expected = reference.get(subj, "?")
        if s != expected:
            mismatch.append((codes[subj], s, expected))
        print(f"  {codes[subj]:>4s} assigned={str(s):8s} expected={expected:8s} "
              f"{'OK' if s == expected else 'MISMATCH'}")
    print(f"\nmismatches: {mismatch or 'NONE'}")
else:
    print(f"=== no reference roster at {REFERENCE_SEX_FILE}; skipping cross-check ===")
print(assigned.value_counts().to_string())

sub = df.groupby(["subject", "sex"])["activity_log"].mean().reset_index()
print("\n=== subject-level mean log activity by sex ===")
print(sub.groupby("sex")["activity_log"].agg(["count", "mean", "std"]).round(4).to_string())

f = sub[sub["sex"] == "Female"]["activity_log"]
m = sub[sub["sex"] == "Male"]["activity_log"]
u, p = stats.mannwhitneyu(f, m)
print(f"Mann-Whitney U={u:.1f}, p={p:.4f}")

col = "hmm_state_filtered"
ranked = []
for subj, g in df.groupby("subject"):
    mm = g.groupby(col)["activity"].mean().sort_values()
    rm = {old: new for new, old in enumerate(mm.index)}
    r = g[[col, "sex"]].copy()
    r["rank"] = r[col].map(rm)
    ranked.append(r)
r = pd.concat(ranked).dropna(subset=["rank", "sex"])

ct = pd.crosstab(r["sex"], r["rank"])
pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
print("\n=== state allocation by sex (percent) ===")
print(pct.to_string())

chi2, p2, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"\nchi2={chi2:.1f}, df={dof}, p={p2:.3e}, Cramers V={v:.4f}, n={n:,}")

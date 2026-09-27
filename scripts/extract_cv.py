"""Between-subject dispersion of mean activity, for the results text."""

import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/v6/hmm_results_improved.parquet")
m = df.groupby("subject_code")["activity"].mean()

print(f"n subjects      : {len(m)}")
print(f"cohort mean     : {m.mean():.3f}")
print(f"SD across means : {m.std(ddof=1):.3f}")
print(f"CV              : {100 * m.std(ddof=1) / m.mean():.1f}%")
print(f"range           : {m.min():.2f} ({m.idxmin()}) to {m.max():.2f} ({m.idxmax()})")
print(f"fold difference : {m.max() / m.min():.2f}x")

col = "hmm_state_filtered"
rows = []
for code, g in df.groupby("subject_code"):
    order = g.groupby(col)["activity"].mean().sort_values()
    remap = {old: new for new, old in enumerate(order.index)}
    r = g[col].map(remap).value_counts(normalize=True)
    rows.append({"code": code, **{f"state{k}": r.get(k, 0.0) for k in range(4)}})

p = pd.DataFrame(rows).set_index("code").sort_index()
print("\nstate occupancy range across subjects:")
for k in range(4):
    c = p[f"state{k}"]
    print(f"  state {k}: {100*c.min():5.1f}% to {100*c.max():5.1f}%  (CV {100*c.std(ddof=1)/c.mean():.0f}%)")

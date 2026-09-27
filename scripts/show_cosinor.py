"""Print cosinor detail for the v3 and v5 runs."""

import json

for v in ["v3", "v5"]:
    d = json.load(open(f"past-runs/{v}/outputs/updated_results.json", encoding="utf-8"))
    mc = d.get("multi_component_cosinor", {})
    pop = mc.get("population", {})
    print(f"--- {v} ---")
    print("  multi-component R2 :", pop.get("r2"))
    print("  single-component R2:", mc.get("single_component_r2"))
    print("  MESOR              :", pop.get("mesor"))
    print("  F / p              :", pop.get("f_statistic"), pop.get("p_value"))
    for c in pop.get("components", []):
        print(f"    T={c.get('period')}h  amp={c.get('amplitude')}  acro={c.get('acrophase')}")
    print("  top-level keys:", list(d.keys()))
    print()

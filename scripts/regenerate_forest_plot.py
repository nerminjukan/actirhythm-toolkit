"""
Regenerate the forest-plot figure for the 8 separate multi-component
cosinor LMMs (one per covariate x response combination).

Each panel (Activity / High-state) shows the main-effect coefficient
from each separate model, giving the same visual layout as the original
but reflecting the corrected separate-model analysis.
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── make src importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.effects import fit_multi_component_cosinor_lmm

# ── paths ────────────────────────────────────────────────────────────
PARQUET = "data/processed/hmm_results_improved.parquet"
META    = "data/raw/ACT_Wet_Dry_Seasons_BR.hourly.csv"
OUT     = "outputs/glmm_forest_plots.png"

# ── 1. Load & prepare hourly data (same as run_updated_analysis.py) ──
print("Loading HMM results …")
df = pd.read_parquet(PARQUET)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Loading & merging metadata …")
meta = pd.read_csv(META, usecols=['subject', 'sex', 'location', 'breeding', 'season',
                                   'day', 'hour'])
static = meta.groupby('subject').agg({'sex': 'first', 'location': 'first'}).reset_index()
df = df.merge(static, on='subject', how='left')

df['month'] = df['timestamp'].dt.month
df['season'] = df['month'].apply(lambda m: 'Dry' if 4 <= m <= 9 else 'Rainy')

# Breeding must be TIME-VARYING (not per-subject mode).
# Merge on subject + date + hour for proper temporal alignment.
meta['day'] = pd.to_datetime(meta['day']).dt.normalize()
meta_breed = meta[['subject', 'day', 'hour', 'breeding']].drop_duplicates()
meta_breed = meta_breed.rename(columns={'day': '_merge_date', 'hour': '_merge_hour'})
df['_merge_date'] = df['timestamp'].dt.normalize()
df['_merge_hour'] = df['timestamp'].dt.hour
df = df.merge(meta_breed, on=['subject', '_merge_date', '_merge_hour'], how='left')
df.drop(columns=['_merge_date', '_merge_hour'], inplace=True)
df['breeding'] = df['breeding'].fillna('Non-Reproductive')
print(f"Breeding distribution: {df['breeding'].value_counts().to_dict()}")

# Determine high-activity state
state_means = df.groupby('hmm_state_filtered')['activity_log'].mean()
high_state = state_means.idxmax()

print("Aggregating to hourly …")
hourly = df.groupby(['subject', 'hour', 'sex', 'location', 'season', 'breeding']).agg(
    mean_activity_log=('activity_log', 'mean'),
    prop_high_state=('hmm_state_filtered', lambda x: (x == high_state).mean()),
).reset_index()
hourly = hourly.dropna(subset=['sex', 'location', 'season', 'breeding'])
print(f"Hourly data: {len(hourly)} rows, {hourly['subject'].nunique()} subjects")

# ── 2. Fit the 8 separate LMMs ──────────────────────────────────────
responses = {
    'activity':   'mean_activity_log',
    'high_state': 'prop_high_state',
}
group_cols = ['sex', 'location', 'season', 'breeding']

results = {}   # key = (response_key, group_col)
for rkey, rcol in responses.items():
    for gcol in group_cols:
        label = f"{rkey}_{gcol}"
        print(f"\nFitting model: {label}")
        res = fit_multi_component_cosinor_lmm(
            hourly, response=rcol, group_column=gcol,
            subject_column='subject', hour_column='hour',
            n_components=5,
        )
        results[(rkey, gcol)] = res

# ── 3. Extract main-effect coefficients ─────────────────────────────
def extract_main_effect(res, gcol):
    """Return list of (label, coef, ci_lo, ci_hi, pval) for the
    grouping-variable main-effect levels (skip Intercept & harmonics)."""
    model_obj = res['model']
    fe = model_obj.fe_params
    ci = model_obj.conf_int()
    pv = model_obj.pvalues

    # Debug: print all fixed effect names
    print(f"  Fixed effects for {gcol}: {list(fe.index)}")

    out = []
    for name in fe.index:
        # keep only entries whose name starts with the group col
        # (the main effect, not the harmonic interactions)
        if ':' in name:
            continue  # skip interaction terms
        if name == 'Intercept':
            continue
        # Skip harmonic terms
        if name.startswith('sin_') or name.startswith('cos_'):
            continue
        # Pretty label
        label = name
        for prefix in [f'{gcol}[T.', f'C({gcol})[T.']:
            if prefix in name:
                label = name.replace(prefix, '').rstrip(']')
                break
        c = fe[name]
        lo = ci.loc[name].iloc[0]
        hi = ci.loc[name].iloc[1]
        p = pv[name]
        out.append((label, c, lo, hi, p))
        print(f"    {name}: coef={c:.4f}, CI=[{lo:.4f}, {hi:.4f}], p={p:.4f}")
    return out

# Build data for the two panels
panel_data = {}  # key = response_key, value = list of tuples
for rkey in responses:
    rows = []
    for gcol in group_cols:
        res = results[(rkey, gcol)]
        if 'error' in res:
            print(f"  WARNING: {rkey}_{gcol} failed: {res['error']}")
            continue
        effects = extract_main_effect(res, gcol)
        for label, coef, lo, hi, pv in effects:
            display = f"{gcol.capitalize()}: {label}"
            rows.append((display, coef, lo, hi, pv))
    panel_data[rkey] = rows

# ── 4. Plot ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def draw_forest(ax, rows, title):
    names  = [r[0] for r in rows]
    coefs  = [r[1] for r in rows]
    ci_lo  = [r[2] for r in rows]
    ci_hi  = [r[3] for r in rows]
    pvals  = [r[4] for r in rows]
    y_pos  = list(range(len(names)))
    colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in pvals]
    # Ensure non-negative error bars
    xerr_lo = [max(c - l, 0) for c, l in zip(coefs, ci_lo)]
    xerr_hi = [max(h - c, 0) for c, h in zip(ci_hi, coefs)]

    ax.barh(y_pos, coefs,
            xerr=[xerr_lo, xerr_hi],
            color=colors, alpha=0.7, edgecolor='black',
            linewidth=0.5, capsize=3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Coefficient (95 % CI)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, (c, p) in enumerate(zip(coefs, pvals)):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if marker:
            offset = max(abs(ci_hi[i]), abs(ci_lo[i])) * 0.05
            ax.text(ci_hi[i] + offset, i, marker,
                    va='center', fontsize=12, fontweight='bold')

draw_forest(ax1, panel_data['activity'],
            'Activity Level\n(separate LMMs, main-effect coefficients)')
draw_forest(ax2, panel_data['high_state'],
            'High-Activity State\n(separate LMMs, main-effect coefficients)')

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {OUT}")

"""
Updated analysis pipeline incorporating feedback:
1. Multi-component cosinor (5 harmonics) instead of single-component
2. Multi-component cosinor LMM with random slopes per subject
3. Shapiro-Wilk normality test for bout durations
4. Updated figures (sub-figure labels, optional subject exclusion, log-scale)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import warnings
import json
import os
from pathlib import Path
from scipy import stats
from loguru import logger

# Project imports
from src.cosinor import (
    fit_cosinor,
    fit_multi_component_cosinor,
    fit_multi_component_cosinor_per_subject,
    multi_component_cosinor_model,
)
from src.effects import (
    fit_multi_component_cosinor_lmm,
    test_bout_duration_normality,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

RUN_VERSION = os.environ.get('THESIS_RUN_VERSION', 'v3')
PROCESSED_BASE_DIR = Path(os.environ.get('THESIS_PROCESSED_BASE_DIR', 'data/processed'))
OUTPUT_BASE_DIR = Path(os.environ.get('THESIS_OUTPUT_BASE_DIR', 'past-runs'))
# Subjects dropped from the lower panels of the between-subject figure, e.g. a
# device whose activity scale is not comparable with the rest of the cohort.
EXCLUDED_SUBJECTS = [
    s.strip() for s in os.environ.get('THESIS_EXCLUDED_SUBJECTS', '').split(',') if s.strip()
]
PROCESSED_DIR = PROCESSED_BASE_DIR / RUN_VERSION
OUTPUT_DIR = OUTPUT_BASE_DIR / RUN_VERSION / 'outputs'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load HMM results and attach sex metadata when available."""
    logger.info("Loading HMM results...")
    hmm_input_file = Path(
        os.environ.get(
            'THESIS_HMM_INPUT_FILE',
            str(PROCESSED_DIR / 'hmm_results_improved.parquet'),
        )
    )
    df = pd.read_parquet(hmm_input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Attach sex metadata only if available in an external file.
    meta_path = Path('data/raw/ACT_Wet_Dry_Seasons_BR.hourly.csv')
    if meta_path.exists():
        try:
            meta = pd.read_csv(
                meta_path,
                usecols=['subject', 'sex'],
            )
            static_cols = [c for c in ['sex'] if c in meta.columns]
            if static_cols:
                static = meta.groupby('subject')[static_cols].first().reset_index()
                df = df.merge(static, on='subject', how='left')
        except Exception as exc:
            logger.warning(f"Metadata merge skipped due to read/format issue: {exc}")

    # Optional subject-level overrides for revised cohorts.
    # Expected columns: subject, sex (Male/Female)
    override_path = Path('data/raw/subject_sex_overrides.csv')
    if override_path.exists():
        try:
            ov = pd.read_csv(override_path, usecols=['subject', 'sex'])
            ov['sex'] = ov['sex'].astype(str).str.strip()
            ov = ov[ov['sex'].isin(['Male', 'Female'])]
            if not ov.empty:
                ov_map = ov.drop_duplicates('subject').set_index('subject')['sex']
                missing_mask = df['sex'].isna() if 'sex' in df.columns else pd.Series(True, index=df.index)
                df.loc[missing_mask, 'sex'] = df.loc[missing_mask, 'subject'].map(ov_map)
                logger.info(
                    f"Applied sex overrides from {override_path}: "
                    f"{ov['subject'].nunique()} subjects available"
                )
        except Exception as exc:
            logger.warning(f"Sex override file ignored due to read/format issue: {exc}")

    if 'sex' not in df.columns:
        df['sex'] = np.nan

    missing_subjects = sorted(df.loc[df['sex'].isna(), 'subject'].dropna().unique())
    if missing_subjects:
        logger.warning(
            "Missing sex labels for subjects (excluded from sex-based models): "
            f"{missing_subjects}. Add labels in data/raw/subject_sex_overrides.csv"
        )

    logger.info(
        f"Data loaded: {len(df):,} obs, {df['subject'].nunique()} subjects, "
        f"available metadata columns: "
        f"{[c for c in ['sex'] if c in df.columns]}"
    )
    return df


# ============================================================================
# 1. MULTI-COMPONENT COSINOR ANALYSIS
# ============================================================================

def run_multi_component_cosinor(df):
    """Fit 5-component cosinor to each subject and population-level."""
    logger.info("=== Running multi-component cosinor analysis ===")

    n_components = 5
    results = {}

    # --- Population-level fit ---
    hourly_pop = df.groupby('hour')['activity_log'].mean().reset_index()
    pop_result = fit_multi_component_cosinor(
        hourly_pop['hour'].values,
        hourly_pop['activity_log'].values,
        n_components=n_components,
    )
    results['population'] = pop_result
    logger.info(
        f"Population cosinor: R²={pop_result['r_squared']:.4f}, "
        f"MESOR={pop_result['mesor']:.3f}"
    )

    # --- Per-subject fits ---
    per_subject = fit_multi_component_cosinor_per_subject(
        df, time_column='timestamp', value_column='activity_log',
        subject_column='subject', n_components=n_components,
    )
    results['per_subject'] = per_subject
    logger.info(f"Per-subject cosinor: {len(per_subject)} subjects fitted")

    # --- Also fit old single-component for comparison ---
    pop_single = fit_cosinor(
        hourly_pop['hour'].values,
        hourly_pop['activity_log'].values,
        period=24.0,
    )
    results['single_component_r2'] = pop_single['r_squared']

    logger.info(
        f"R² comparison: single={pop_single['r_squared']:.4f} vs "
        f"5-component={pop_result['r_squared']:.4f}"
    )

    # --- Generate comparison figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Population fit comparison
    ax = axes[0]
    hours = np.linspace(0, 23.99, 200)
    y_data = hourly_pop['activity_log'].values
    hour_data = hourly_pop['hour'].values

    ax.scatter(hour_data, y_data, color='black', s=30, zorder=5, label='Observed')

    # Single component
    y_single = pop_single['mesor'] + pop_single['amplitude'] * np.cos(
        2 * np.pi * hours / 24 - pop_single['acrophase']
    )
    ax.plot(hours, y_single, 'b--', linewidth=1.5,
            label=f'1-component (R²={pop_single["r_squared"]:.3f})')

    # Multi component
    y_multi = multi_component_cosinor_model(
        hours, np.array(pop_result['popt']), pop_result['periods']
    )
    ax.plot(hours, y_multi, 'r-', linewidth=2,
            label=f'5-component (R²={pop_result["r_squared"]:.3f})')

    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Mean activity (log)')
    ax.set_title('(a) Population-level cosinor model comparison')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    # (b) Per-subject R² comparison
    ax = axes[1]
    # Also fit single-component per subject for comparison
    r2_single_list = []
    r2_multi_list = []
    subjects_list = []
    for subj in sorted(df['subject'].unique()):
        subj_data = df[df['subject'] == subj]
        hourly = subj_data.groupby('hour')['activity_log'].mean().reset_index()
        s_res = fit_cosinor(hourly['hour'].values, hourly['activity_log'].values)
        m_res = fit_multi_component_cosinor(
            hourly['hour'].values, hourly['activity_log'].values,
            n_components=n_components,
        )
        r2_single_list.append(s_res['r_squared'])
        r2_multi_list.append(m_res['r_squared'])
        subjects_list.append(subj)

    x = np.arange(len(subjects_list))
    width = 0.35
    ax.bar(x - width/2, r2_single_list, width, label='1-component', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, r2_multi_list, width, label='5-component', color='firebrick', alpha=0.8)
    ax.set_ylabel('R²')
    ax.set_title('(b) Per-subject cosinor R² comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects_list, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cosinor_multi_component.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved cosinor_multi_component.png")

    return results


# ============================================================================
# 2. MULTI-COMPONENT COSINOR LMM
# ============================================================================

def run_cosinor_lmm(df):
    """Run multi-component cosinor LMM for each grouping variable."""
    logger.info("=== Running multi-component cosinor LMM ===")

    candidate_group_vars = ['sex']
    available_group_vars = [
        g for g in candidate_group_vars
        if g in df.columns and df[g].dropna().nunique() >= 2
    ]
    if not available_group_vars:
        logger.warning("No grouping metadata with >=2 levels; skipping LMM group models")
        return {}

    # Aggregate to hourly means
    group_cols = ['subject', 'hour'] + available_group_vars
    df_hourly = df.groupby(group_cols).agg(
        mean_activity_log=('activity_log', 'mean'),
        prop_high_state=('hmm_state_filtered', lambda x: (x == 3.0).mean()),
    ).reset_index()

    logger.info(f"Hourly aggregated: {len(df_hourly)} rows, {df_hourly['subject'].nunique()} subjects")

    lmm_results = {}

    # Filter to subjects with complete metadata for available grouping variables
    df_hourly = df_hourly.dropna(subset=available_group_vars)
    logger.info(f"After metadata filter: {len(df_hourly)} rows, {df_hourly['subject'].nunique()} subjects")

    for group_var in available_group_vars:
        logger.info(f"--- LMM for group: {group_var} ---")

        # Model 1: Activity level
        result_act = fit_multi_component_cosinor_lmm(
            df_hourly,
            response='mean_activity_log',
            group_column=group_var,
            subject_column='subject',
            hour_column='hour',
            n_components=5,
        )
        lmm_results[f'activity_{group_var}'] = result_act

        # Model 2: High-state occupancy
        result_state = fit_multi_component_cosinor_lmm(
            df_hourly,
            response='prop_high_state',
            group_column=group_var,
            subject_column='subject',
            hour_column='hour',
            n_components=5,
        )
        lmm_results[f'high_state_{group_var}'] = result_state

    # Save summaries
    summary_rows = []
    for key, res in lmm_results.items():
        if 'error' not in res:
            summary_rows.append({
                'model': key,
                'n_obs': res['n_obs'],
                'n_groups': res['n_groups'],
                'r2_marginal': res['r2_marginal'],
                'r2_conditional': res['r2_conditional'],
                'icc': res['icc'],
                'aic': res['aic'],
                'bic': res['bic'],
                'converged': res['converged'],
            })
        else:
            summary_rows.append({
                'model': key,
                'error': res['error'],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'lmm_multi_component_results.csv', index=False)
    logger.info("Saved lmm_multi_component_results.csv")

    # Save full summaries as text
    with open(OUTPUT_DIR / 'lmm_summaries.txt', 'w', encoding='utf-8') as f:
        for key, res in lmm_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"MODEL: {key}\n")
            f.write(f"{'='*80}\n")
            if 'error' not in res:
                f.write(f"Formula: {res['formula']}\n")
                f.write(f"RE Formula: {res['re_formula']}\n")
                f.write(f"N obs: {res['n_obs']}, N groups: {res['n_groups']}\n")
                f.write(f"R2 marginal: {res['r2_marginal']:.4f}\n")
                f.write(f"R2 conditional: {res['r2_conditional']:.4f}\n")
                f.write(f"ICC: {res['icc']:.4f}\n")
                aic_val = res['aic']
                f.write(f"AIC: {aic_val}\n")
                f.write(f"\n{res['summary']}\n")
            else:
                f.write(f"ERROR: {res['error']}\n")
    logger.info("Saved lmm_summaries.txt")

    return lmm_results


# ============================================================================
# 3. BOUT DURATION NORMALITY TESTS
# ============================================================================

def run_bout_duration_analysis(df):
    """Compute bout durations and test for normality."""
    logger.info("=== Running bout duration + Shapiro-Wilk analysis ===")

    # Compute bout durations from hmm_state_filtered
    all_bout_results = {}
    all_bout_durations = {0: [], 1: [], 2: [], 3: []}

    for subj in sorted(df['subject'].unique()):
        subj_data = df[df['subject'] == subj].sort_values('timestamp')
        states = subj_data['hmm_state_filtered'].values

        # Run-length encoding
        bouts = []
        if len(states) > 0:
            current_state = states[0]
            current_len = 1
            for s in states[1:]:
                if s == current_state:
                    current_len += 1
                else:
                    bouts.append((current_state, current_len))
                    current_state = s
                    current_len = 1
            bouts.append((current_state, current_len))

        for state, length in bouts:
            if not np.isnan(state):
                duration_hours = length * 0.25  # 15-min intervals
                all_bout_durations[int(state)].append(duration_hours)

    # Test normality per state
    normality_results = {}
    for state in sorted(all_bout_durations.keys()):
        durations = np.array(all_bout_durations[state])
        if len(durations) >= 3:
            normality_results[state] = test_bout_duration_normality(durations)
            logger.info(
                f"State {state}: n={len(durations)}, "
                f"Shapiro-Wilk p={normality_results[state]['shapiro_wilk_p']:.4e}, "
                f"skew={normality_results[state]['skewness']:.2f}"
            )

    # Generate normality diagnostic figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    state_labels = ['Deep rest', 'Light rest', 'Moderate activity', 'High activity']

    for i, state in enumerate(sorted(all_bout_durations.keys())):
        durations = np.array(all_bout_durations[state])
        if len(durations) < 3:
            continue

        # Histogram
        ax = axes[0, i]
        ax.hist(durations, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='white')
        # Overlay normal fit
        mu, sigma = np.mean(durations), np.std(durations)
        x_range = np.linspace(0, np.percentile(durations, 99), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2)
        sw_p = normality_results[state].get('shapiro_wilk_p', np.nan)
        ax.set_title(f'({chr(97+i)}) State {state}: {state_labels[i]}', fontsize=10)
        ax.set_xlabel('Duration (hours)')
        ax.set_ylabel('Density')
        ax.text(0.95, 0.95, f'S-W p={sw_p:.2e}\nn={len(durations)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Q-Q plot
        ax = axes[1, i]
        stats.probplot(durations, dist='norm', plot=ax)
        ax.set_title(f'Q-Q plot: State {state}', fontsize=10)

    plt.suptitle('Bout Duration Normality Assessment', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'bout_duration_normality.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved bout_duration_normality.png")

    return normality_results, all_bout_durations


# ============================================================================
# 4. UPDATED BETWEEN-SUBJECT COMPARISON FIGURES
# ============================================================================

def generate_updated_between_subject_figures(df):
    """Generate updated between-subject figures (excluded subjects dropped from C/D, log-scale)."""
    logger.info("=== Generating updated between-subject figures ===")

    subjects = sorted(df['subject'].unique())

    # Subject-level summary
    subj_summary = df.groupby('subject').agg(
        mean_activity=('activity_log', 'mean'),
        std_activity=('activity_log', 'std'),
    ).reset_index()

    # Merge metadata when available
    available_meta = [c for c in ['sex'] if c in df.columns]
    if available_meta:
        meta = df.groupby('subject')[available_meta].first().reset_index()
        subj_summary = subj_summary.merge(meta, on='subject', how='left')

    # State proportions per subject
    state_props = df.groupby(['subject', 'hmm_state_filtered']).size().unstack(fill_value=0)
    state_props = state_props.div(state_props.sum(axis=1), axis=0)

    # Subjects retained in panels (c) and (d)
    subjects_kept = [s for s in subjects if s not in EXCLUDED_SUBJECTS]
    excl_note = ' (excl. outliers)' if EXCLUDED_SUBJECTS else ''

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (a) Activity boxplots - all subjects
    ax = fig.add_subplot(gs[0, 0])
    data_for_box = [df[df['subject'] == s]['activity_log'].values for s in subjects]
    bp = ax.boxplot(data_for_box, tick_labels=subjects, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Activity (log)')
    ax.set_title('(a) Activity distribution by subject')

    # (b) State proportions - all subjects
    ax = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(subjects))
    colors_list = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    state_labels = ['Deep rest', 'Light rest', 'Moderate', 'High activity']
    for state_idx in range(4):
        col = state_idx
        if col in state_props.columns:
            vals = [state_props.loc[s, col] if s in state_props.index else 0 for s in subjects]
        else:
            vals = [0] * len(subjects)
        ax.bar(range(len(subjects)), vals, bottom=bottom, color=colors_list[state_idx],
               label=state_labels[state_idx], alpha=0.8)
        bottom += vals
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Proportion')
    ax.set_title('(b) State distribution by subject')
    ax.legend(fontsize=7, loc='upper right')

    # (c) Activity boxplots - excluded subjects dropped, Y-axis log scale
    ax = fig.add_subplot(gs[1, 0])
    data_for_box_no_n = [
        df[(df['subject'] == s) & (df['activity'] > 0)]['activity'].values
        for s in subjects_kept
    ]
    bp = ax.boxplot(data_for_box_no_n, tick_labels=subjects_kept,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels(subjects_kept, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Activity (log scale)')
    ax.set_title(f'(c) Activity by subject{excl_note}, log scale')

    # (d) State proportions - excluded subjects dropped
    ax = fig.add_subplot(gs[1, 1])
    bottom = np.zeros(len(subjects_kept))
    for state_idx in range(4):
        col = state_idx
        if col in state_props.columns:
            vals = [state_props.loc[s, col] if s in state_props.index else 0
                    for s in subjects_kept]
        else:
            vals = [0] * len(subjects_kept)
        ax.bar(range(len(subjects_kept)), vals, bottom=bottom,
               color=colors_list[state_idx], label=state_labels[state_idx], alpha=0.8)
        bottom += vals
    ax.set_xticks(range(len(subjects_kept)))
    ax.set_xticklabels(subjects_kept, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Proportion')
    ax.set_title(f'(d) State distribution{excl_note}')
    ax.legend(fontsize=7, loc='upper right')

    plt.savefig(OUTPUT_DIR / 'between_subject_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved between_subject_comparison_v2.png")


# ============================================================================
# 5. UPDATED METADATA EFFECT FIGURES (consistent graph types, a/b/c labels)
# ============================================================================

def generate_updated_metadata_figures(df):
    """Generate updated metadata effect figures with consistent layout and a/b/c labels."""
    logger.info("=== Generating updated metadata effect figures ===")

    candidate_group_vars = ['sex']
    available_group_vars = [
        g for g in candidate_group_vars
        if g in df.columns and df[g].dropna().nunique() >= 2
    ]
    if not available_group_vars:
        logger.warning("No metadata groups with >=2 levels; skipping metadata effect figures")
        return

    subj_summary = df.groupby('subject').agg(
        mean_activity=('activity_log', 'mean'),
        std_activity=('activity_log', 'std'),
    ).reset_index()

    meta_cols = [c for c in candidate_group_vars if c in df.columns]
    meta = df.groupby('subject')[meta_cols].first().reset_index()
    subj_summary = subj_summary.merge(meta, on='subject')

    # State proportions per subject
    state_props = df.groupby(['subject', 'hmm_state_filtered']).size().unstack(fill_value=0)
    state_props = state_props.div(state_props.sum(axis=1), axis=0)
    state_props = state_props.reset_index()
    state_props = state_props.merge(meta, on='subject')

    for group_var in available_group_vars:
        groups = sorted(df[group_var].dropna().unique())
        n_groups = len(groups)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # (a) Activity boxplot by group
        ax = axes[0]
        group_data = [subj_summary[subj_summary[group_var] == g]['mean_activity'].values
                      for g in groups]
        bp = ax.boxplot(group_data, tick_labels=groups, patch_artist=True, showfliers=False)
        palette = plt.cm.Set2(np.linspace(0, 1, n_groups))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
        ax.set_ylabel('Mean activity (log)')
        ax.set_title(f'(a) Activity by {group_var}')

        # Statistical test
        # Filter out empty groups
        group_data_nonempty = [g for g in group_data if len(g) >= 1]
        if len(group_data_nonempty) >= 2:
            if n_groups == 2:
                stat, pval = stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                test_name = 'Mann-Whitney U'
            else:
                stat, pval = stats.kruskal(*group_data_nonempty)
                test_name = 'Kruskal-Wallis'
            ax.text(0.5, 0.02, f'{test_name}: p={pval:.4f}',
                    transform=ax.transAxes, ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.02, 'Insufficient groups for test',
                    transform=ax.transAxes, ha='center', fontsize=8)

        # (b) State proportions by group (stacked bar per group)
        ax = axes[1]
        colors_list = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        state_labels_plot = ['Deep rest', 'Light rest', 'Moderate', 'High']
        width = 0.6
        bottom = np.zeros(n_groups)
        for state_idx in range(4):
            col = state_idx
            vals = []
            for g in groups:
                mask = state_props[group_var] == g
                if col in state_props.columns and mask.any():
                    vals.append(state_props.loc[mask, col].mean())
                else:
                    vals.append(0)
            ax.bar(range(n_groups), vals, width, bottom=bottom,
                   color=colors_list[state_idx], label=state_labels_plot[state_idx], alpha=0.8)
            bottom += np.array(vals)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(groups)
        ax.set_ylabel('Mean proportion')
        ax.set_title(f'(b) State distribution by {group_var}')
        ax.legend(fontsize=7, loc='upper right')

        # (c) Diel activity profile by group
        ax = axes[2]
        for i, g in enumerate(groups):
            g_data = df[df[group_var] == g]
            hourly = g_data.groupby('hour')['activity_log'].mean()
            ax.plot(hourly.index, hourly.values, '-o', markersize=3,
                    color=palette[i], label=g, linewidth=1.5)
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Mean activity (log)')
        ax.set_title(f'(c) Diel profile by {group_var}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 23)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f'{group_var}_effects_v2.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {group_var}_effects_v2.png")


# ============================================================================
# 6. UPDATED COSINOR RHYTHM FIGURE (with multi-component)
# ============================================================================

def generate_cosinor_figure(df, cosinor_results):
    """Generate updated cosinor rhythm figure showing multi-component fit."""
    logger.info("=== Generating updated cosinor figure ===")

    hourly = df.groupby('hour')['activity_log'].mean().reset_index()
    hours_fine = np.linspace(0, 23.99, 200)

    pop = cosinor_results['population']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Fit with data
    ax = axes[0]
    ax.scatter(hourly['hour'], hourly['activity_log'],
               color='black', s=40, zorder=5, label='Hourly means')

    y_multi = multi_component_cosinor_model(
        hours_fine, np.array(pop['popt']), pop['periods']
    )
    ax.plot(hours_fine, y_multi, 'r-', linewidth=2,
            label=f'5-component fit (R²={pop["r_squared"]:.3f})')
    ax.axhline(pop['mesor'], color='gray', linestyle=':', alpha=0.5, label=f'MESOR={pop["mesor"]:.2f}')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Mean activity (log)')
    ax.set_title('(a) Multi-component cosinor: population')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    # (b) Component decomposition
    ax = axes[1]
    for j, comp in enumerate(pop['components']):
        T = comp['period']
        amp = comp['amplitude']
        phi = comp['acrophase_rad']
        y_comp = amp * np.cos(2 * np.pi * hours_fine / T - phi)
        ax.plot(hours_fine, y_comp, linewidth=1.5,
                label=f'Component {j+1} ({T:.1f}h, amp={abs(amp):.3f})')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Amplitude contribution')
    ax.set_title('(b) Harmonic component decomposition')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cosinor_rhythm_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved cosinor_rhythm_v2.png")


# ============================================================================
# 7. SAVE RESULTS SUMMARY
# ============================================================================

def save_results_summary(cosinor_results, lmm_results, normality_results):
    """Save all numerical results to a JSON file for LaTeX integration."""
    logger.info("=== Saving results summary ===")

    summary = {
        'multi_component_cosinor': {
            'population': {
                'mesor': cosinor_results['population']['mesor'],
                'r_squared': cosinor_results['population']['r_squared'],
                'f_statistic': cosinor_results['population']['f_statistic'],
                'p_value': cosinor_results['population']['p_value'],
                'n_components': cosinor_results['population']['n_components'],
                'components': cosinor_results['population']['components'],
            },
            'single_component_r2': cosinor_results['single_component_r2'],
        },
        'lmm_results': {},
        'bout_normality': {},
    }

    # LMM summary
    for key, res in lmm_results.items():
        if 'error' not in res:
            summary['lmm_results'][key] = {
                'r2_marginal': res['r2_marginal'],
                'r2_conditional': res['r2_conditional'],
                'icc': res['icc'],
                'aic': res['aic'],
                'bic': res['bic'],
                'n_obs': res['n_obs'],
                'n_groups': res['n_groups'],
                'converged': res['converged'],
            }
        else:
            summary['lmm_results'][key] = {'error': res['error']}

    # Normality summary
    for state, res in normality_results.items():
        summary['bout_normality'][str(state)] = {
            'n': res['n'],
            'mean': res['mean'],
            'std': res['std'],
            'skewness': res['skewness'],
            'kurtosis': res['kurtosis'],
            'shapiro_wilk_stat': res.get('shapiro_wilk_stat'),
            'shapiro_wilk_p': res.get('shapiro_wilk_p'),
            'shapiro_wilk_normal': res.get('shapiro_wilk_normal'),
        }

    # Per-subject cosinor summary
    if 'per_subject' in cosinor_results:
        ps_df = cosinor_results['per_subject']
        summary['multi_component_cosinor']['per_subject'] = ps_df.to_dict(orient='records')

    with open(OUTPUT_DIR / 'updated_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved updated_results.json")

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("UPDATED ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # Load data
    df = load_data()

    # 1. Multi-component cosinor
    cosinor_results = run_multi_component_cosinor(df)

    # 2. Multi-component cosinor LMM
    lmm_results = run_cosinor_lmm(df)

    # 3. Bout duration normality
    normality_results, bout_durations = run_bout_duration_analysis(df)

    # 4. Updated between-subject figures
    generate_updated_between_subject_figures(df)

    # 5. Updated metadata figures
    generate_updated_metadata_figures(df)

    # 6. Updated cosinor figure
    generate_cosinor_figure(df, cosinor_results)

    # 7. Save results
    summary = save_results_summary(cosinor_results, lmm_results, normality_results)

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)

    # Print key results
    pop = cosinor_results['population']
    print("\n=== KEY RESULTS ===")
    print(f"Multi-component cosinor R2: {pop['r_squared']:.4f} "
          f"(vs single: {cosinor_results['single_component_r2']:.4f})")
    print(f"F-statistic: {pop['f_statistic']:.2f}, p={pop['p_value']:.2e}")
    print(f"Components:")
    for j, comp in enumerate(pop['components']):
        print(f"  {j+1}. Period={comp['period']:.1f}h, "
              f"Amplitude={comp['amplitude']:.4f}, "
              f"Acrophase={comp['acrophase_hours']:.2f}h")

    print(f"\nLMM models fitted: {len(lmm_results)}")
    for key, res in lmm_results.items():
        if 'error' not in res:
            print(f"  {key}: R2m={res['r2_marginal']:.4f}, "
                  f"R2c={res['r2_conditional']:.4f}, ICC={res['icc']:.4f}")

    print(f"\nBout duration normality (Shapiro-Wilk):")
    for state, res in normality_results.items():
        print(f"  State {state}: W={res.get('shapiro_wilk_stat', 'N/A'):.4f}, "
              f"p={res.get('shapiro_wilk_p', 'N/A'):.2e}, "
              f"Normal={res.get('shapiro_wilk_normal', 'N/A')}")


if __name__ == '__main__':
    main()

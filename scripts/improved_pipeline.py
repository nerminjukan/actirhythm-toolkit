#!/usr/bin/env python
"""
IMPROVED PIPELINE - Correctly handles multi-subject, long time-series data.

CRITICAL FIXES:
1. Per-subject feature standardization (z-scoring within each individual)
2. Per-subject HMM fitting (separate models for each subject)
3. Proper handling of rolling features (computed per subject)
4. Log transformation to handle zero-inflated activity data
5. Biological validation of HMM states

This replaces the flawed example_pipeline.py that pooled all subjects together.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import pipeline modules
from src import io, qc, features, models_hmm, cosinor
from src.config import load_config_or_defaults
from src.logging_config import setup_logging

# Setup
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure logging
logger = setup_logging(log_level="INFO")
logger.info("Starting IMPROVED pipeline for multi-subject accelerometer data")

# Load configuration
config = load_config_or_defaults()

RUN_VERSION = os.environ.get("THESIS_RUN_VERSION", "v3")
PROCESSED_BASE_DIR = Path(os.environ.get("THESIS_PROCESSED_BASE_DIR", "data/processed"))
OUTPUT_BASE_DIR = Path(os.environ.get("THESIS_OUTPUT_BASE_DIR", "past-runs"))
PROCESSED_DIR = PROCESSED_BASE_DIR / RUN_VERSION
OUTPUT_DIR = OUTPUT_BASE_DIR / RUN_VERSION / "outputs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_SIGNAL = os.environ.get("THESIS_ACTIVITY_SIGNAL", "ActMindata")
# 2 bins = 30 min: removes decoder flicker without imposing a floor above the
# unfiltered median bout length (0.75 h). See dwell_time_sensitivity.csv.
MIN_DWELL = int(os.environ.get("THESIS_MIN_DWELL", "2"))
# Reported as a sensitivity analysis: 1, 2 and 4 bins = 15, 30 and 60 minutes.
MIN_DWELL_CANDIDATES = [1, 2, 4]

def run_improved_pipeline():
    """Run the corrected analysis pipeline."""
    
    # ==========================================================================
    # Stage 1: Data Intake
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 1: Data Intake")
    logger.info("=" * 70)
    
    # Load accelerometer data
    revised_dir = Path(os.environ.get('THESIS_DATA_REVISED_DIR', 'data-revised'))
    raw_data_file = Path(
        os.environ.get('THESIS_RAW_DATA_FILE', 'data/raw/sample_accelerometer_data.csv')
    )
    if revised_dir.exists():
        df = io.load_accelerometer_data_from_directory(
            revised_dir,
            pattern='*.ACT.csv'
        )
    else:
        df = io.load_accelerometer_data(raw_data_file)
    logger.info(f"Loaded {len(df):,} rows for {df['subject'].nunique()} subjects")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # Only the per-axis averages were exported, so their magnitude tracks posture
    # (r = -0.96 against ActMindata hourly); ActMindata is the real movement measure.
    if ACTIVITY_SIGNAL not in df.columns:
        raise ValueError(
            f"Configured activity signal '{ACTIVITY_SIGNAL}' not found. "
            f"Available: {[c for c in df.columns if c.startswith('activity')]}"
        )
    df['activity'] = df[ACTIVITY_SIGNAL]
    logger.success(f"Using {ACTIVITY_SIGNAL} as primary activity signal")
    
    # ==========================================================================
    # Stage 2: Quality Control
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 2: Quality Control")
    logger.info("=" * 70)
    
    df_clean = qc.remove_duplicates(df)

    logger.info("Applying documented deployment windows...")
    qc.load_deployment_windows()
    df_clean = qc.apply_deployment_windows(df_clean)

    df_clean, subject_codes = qc.assign_subject_codes(df_clean)
    pd.Series(subject_codes, name='subject_code').rename_axis('subject').to_csv(
        OUTPUT_DIR / 'subject_code_mapping.csv'
    )

    # Subject-level QC report
    for subject in df_clean['subject'].unique():
        subj_data = df_clean[df_clean['subject'] == subject]
        logger.info(f"{subject_codes[subject]} ({subject}): {len(subj_data):,} rows, "
                   f"mean activity={subj_data['activity'].mean():.2f}, "
                   f"zeros={100*(subj_data['activity']==0).sum()/len(subj_data):.1f}%")

    logger.success(
        f"Analysis dataset: {len(df_clean):,} observations from "
        f"{df_clean['subject'].nunique()} subjects, "
        f"{df_clean['timestamp'].min().date()} to {df_clean['timestamp'].max().date()}"
    )
    
    # ==========================================================================
    # Stage 3: Feature Engineering (PER SUBJECT)
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 3: Feature Engineering (Per-Subject)")
    logger.info("=" * 70)
    
    # Log transform to handle zeros
    df_features = features.log_transform_activity(df_clean, 'activity', offset=1.0)
    logger.success("Applied log(activity + 1) transformation")
    
    # Add rolling statistics PER SUBJECT
    # Using shorter windows appropriate for behavioral states
    df_features = features.add_rolling_statistics_per_individual(
        df_features,
        'activity_log',
        subject_column='subject',
        windows=[4, 8, 12],  # Shorter windows (1-3 hours at 15-min sampling)
        stats=['mean', 'std']
    )
    logger.success("Added per-subject rolling features")
    
    # Add time features
    df_features = features.add_time_features(
        df_features,
        'timestamp',
        features=['hour', 'day_of_week']
    )
    
    # CRITICAL: Standardize ALL features PER SUBJECT
    feature_cols_to_standardize = ['activity_log'] + [
        col for col in df_features.columns 
        if 'rolling' in col and col.endswith(('_4', '_8', '_12'))
    ]
    
    df_features = features.standardize_features(
        df_features,
        feature_cols_to_standardize,
        group_by='subject',  # CRITICAL: per-subject z-scoring
        overwrite=True
    )
    logger.success(f"Standardized {len(feature_cols_to_standardize)} features PER SUBJECT")
    
    io.save_processed_data(df_features, PROCESSED_DIR / 'feature_data_improved.parquet')
    
    # ==========================================================================
    # Stage 4: HMM Analysis (PER SUBJECT)
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 4: HMM Analysis (Per-Subject)")
    logger.info("=" * 70)
    
    # Use standardized features for HMM
    hmm_feature_cols = ['activity_log_standardized']
    
    # Add best rolling features if available
    rolling_std_cols = [col for col in df_features.columns 
                       if col.endswith('_standardized') and 'rolling_std' in col]
    if rolling_std_cols:
        hmm_feature_cols.append(rolling_std_cols[0])  # Add one rolling std feature
        logger.info(f"Using features for HMM: {hmm_feature_cols}")
    
    # FIT HMM PER SUBJECT (CORRECT APPROACH)
    hmm_results = models_hmm.fit_hmm_per_subject(
        df_features,
        feature_columns=hmm_feature_cols,
        subject_column='subject',
        state_range=range(2, 5),  # Test 2-4 states
        random_state=RANDOM_STATE,
        standardize=False  # Already standardized
    )
    
    # Combine results back into dataframe
    df_with_states = models_hmm.combine_per_subject_states(
        df_features,
        hmm_results,
        subject_column='subject'
    )
    
    # Apply minimum dwell time per subject, for every candidate threshold
    logger.info("Applying minimum dwell time filters per subject...")
    dwell_columns = {d: f'hmm_state_filtered_{d}' for d in MIN_DWELL_CANDIDATES}
    for col in dwell_columns.values():
        df_with_states[col] = np.nan
    df_with_states['hmm_state_filtered'] = np.nan

    for subject in df_with_states['subject'].unique():
        mask = df_with_states['subject'] == subject
        subject_data = df_with_states.loc[mask, 'hmm_state'].dropna()

        if len(subject_data) > 0:
            states = subject_data.values.astype(int)
            for dwell, col in dwell_columns.items():
                df_with_states.loc[subject_data.index, col] = (
                    models_hmm.apply_minimum_dwell_time(states, min_dwell=dwell)
                )

    df_with_states['hmm_state_filtered'] = df_with_states[dwell_columns[MIN_DWELL]]

    dwell_summary = summarise_dwell_sensitivity(df_with_states, dwell_columns)
    dwell_summary.to_csv(OUTPUT_DIR / 'dwell_time_sensitivity.csv', index=False)
    logger.info(f"Dwell-time sensitivity summary:\n{dwell_summary.to_string(index=False)}")

    io.save_processed_data(df_with_states, PROCESSED_DIR / 'hmm_results_improved.parquet')
    
    # ==========================================================================
    # Stage 5: Visualizations
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 5: Creating Improved Visualizations")
    logger.info("=" * 70)
    
    create_improved_visualizations(df_with_states, hmm_results)
    
    # ==========================================================================
    # Stage 6: State Validation
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Stage 6: Biological Validation of States")
    logger.info("=" * 70)
    
    validate_states(df_with_states)
    validate_states_with_posture(df_with_states)
    
    logger.success("=" * 70)
    logger.success("IMPROVED PIPELINE COMPLETE")
    logger.success("=" * 70)
    logger.success("Key outputs:")
    logger.success(f"  - {PROCESSED_DIR / 'feature_data_improved.parquet'}")
    logger.success(f"  - {PROCESSED_DIR / 'hmm_results_improved.parquet'}")
    logger.success(f"  - {OUTPUT_DIR / 'improved_*.png'}")


def validate_states_with_posture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check decoded states against the posture channel, which is independent of
    ActMindata.

    The stored axes are bin-averaged ADC values, so their vector magnitude is
    gravity-dominated: it is longest when the animal holds one orientation and
    shortens as orientation varies within the bin. Resting states should
    therefore show a higher and less variable magnitude than active states.
    """
    if 'activity_xyz' not in df.columns:
        logger.warning("activity_xyz unavailable; skipping posture validation")
        return pd.DataFrame()

    state_col = 'hmm_state_filtered'
    valid = df[df[state_col].notna()].copy()

    # Rank states by activity within subject so they are comparable across models.
    valid['state_rank'] = (
        valid.groupby('subject')[state_col]
        .transform(lambda s: s.map(valid.loc[s.index].groupby(s)['activity'].mean().rank(method='dense')))
    )

    summary = valid.groupby('state_rank').agg(
        n=('activity', 'size'),
        activity_mean=('activity', 'mean'),
        posture_mag_mean=('activity_xyz', 'mean'),
        posture_mag_sd=('activity_xyz', 'std'),
    ).round(2)

    corr = valid.groupby('subject').apply(
        lambda g: g['activity'].corr(g['activity_xyz']), include_groups=False
    )

    logger.info("Posture validation of decoded states (rank 1 = least active):")
    logger.info(f"\n{summary.to_string()}")
    logger.info(
        f"Per-subject corr(activity, posture magnitude): "
        f"mean={corr.mean():.3f}, range {corr.min():.3f} to {corr.max():.3f}, "
        f"negative in {int((corr < 0).sum())}/{len(corr)} subjects"
    )

    summary.to_csv(OUTPUT_DIR / 'posture_state_validation.csv')
    return summary


def summarise_dwell_sensitivity(
    df: pd.DataFrame,
    dwell_columns: dict,
    bin_minutes: float = 15.0,
) -> pd.DataFrame:
    """Compare state sequences produced by different minimum dwell-time filters."""
    rows = []
    raw = df['hmm_state']

    for dwell, col in dwell_columns.items():
        filtered = df[col]
        valid = raw.notna() & filtered.notna()

        bout_lengths = []
        n_bouts = 0
        for _, group in df.groupby('subject', sort=False):
            seq = group[col].dropna().to_numpy()
            if seq.size == 0:
                continue
            starts = np.concatenate(([0], np.where(np.diff(seq) != 0)[0] + 1))
            lengths = np.diff(np.concatenate((starts, [seq.size])))
            bout_lengths.append(lengths)
            n_bouts += lengths.size

        lengths = np.concatenate(bout_lengths) if bout_lengths else np.array([0])
        changed = (raw[valid] != filtered[valid]).mean() * 100

        rows.append({
            'min_dwell_bins': dwell,
            'min_dwell_minutes': dwell * bin_minutes,
            'pct_observations_reassigned': round(changed, 2),
            'n_bouts': int(n_bouts),
            'mean_bout_hours': round(lengths.mean() * bin_minutes / 60, 3),
            'median_bout_hours': round(float(np.median(lengths)) * bin_minutes / 60, 3),
            'n_states_used': int(filtered.dropna().nunique()),
        })

    return pd.DataFrame(rows)


def create_improved_visualizations(df: pd.DataFrame, hmm_results: dict):
    """Create publication-quality visualizations."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # -------------------------------------------------------------------------
    # 1. Per-Subject Activity Distributions (with and without log transform)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw activity
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject]['activity']
        axes[0].hist(subj_data, bins=30, alpha=0.3, label=subject)
    axes[0].set_xlabel('Activity (ActMindata)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Raw Activity Distribution by Subject')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    # Log-transformed activity
    for subject in df['subject'].unique():
        subj_data = df[df['subject'] == subject]['activity_log']
        axes[1].hist(subj_data, bins=30, alpha=0.3, label=subject)
    axes[1].set_xlabel('log(Activity + 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Activity by Subject')
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'improved_activity_distributions.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved: {out_path}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. Per-Subject State Characteristics
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot optimal K per subject
    subjects = list(hmm_results.keys())
    optimal_ks = [hmm_results[s]['optimal_k'] for s in subjects]
    
    axes[0].bar(range(len(subjects)), optimal_ks, color='steelblue')
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_xticklabels(subjects, rotation=45, ha='right')
    axes[0].set_ylabel('Optimal K')
    axes[0].set_title('Optimal Number of States per Subject')
    axes[0].set_ylim([0, max(optimal_ks) + 1])
    for i, k in enumerate(optimal_ks):
        axes[0].text(i, k + 0.1, str(k), ha='center', va='bottom')
    
    # Plot state proportions for each subject
    state_props = []
    for subject in subjects:
        mask = df['subject'] == subject
        states = pd.Series(df.loc[mask, 'hmm_state_filtered']).dropna()
        if len(states) > 0:
            props = [100 * (states == s).sum() / len(states) 
                    for s in range(int(states.max()) + 1)]
            state_props.append(props)
        else:
            state_props.append([])
    
    # Create stacked bar chart
    max_states = max(len(p) for p in state_props)
    state_data = np.zeros((len(subjects), max_states))
    for i, props in enumerate(state_props):
        state_data[i, :len(props)] = props
    
    bottom = np.zeros(len(subjects))
    colors = plt.colormaps.get_cmap('Set3')(np.linspace(0, 1, max_states))
    for state_idx in range(max_states):
        axes[1].bar(range(len(subjects)), state_data[:, state_idx], 
                   bottom=bottom, label=f'State {state_idx}', color=colors[state_idx])
        bottom += state_data[:, state_idx]
    
    axes[1].set_xticks(range(len(subjects)))
    axes[1].set_xticklabels(subjects, rotation=45, ha='right')
    axes[1].set_ylabel('Percentage')
    axes[1].set_title('State Proportions by Subject')
    axes[1].legend(loc='upper right', fontsize=8)
    
    # Plot mean activity per state per subject
    for subject in subjects:
        mask = (df['subject'] == subject) & df['hmm_state_filtered'].notna()
        if mask.sum() > 0:
            subj_data = df.loc[mask].groupby('hmm_state_filtered')['activity'].mean()
            axes[2].plot(subj_data.index, subj_data.values, marker='o', alpha=0.6, label=subject)
    
    axes[2].set_xlabel('HMM State')
    axes[2].set_ylabel('Mean Activity')
    axes[2].set_title('Mean Activity per State by Subject')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'improved_hmm_per_subject.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved: {out_path}")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. Sample time series for selected subjects
    # -------------------------------------------------------------------------
    sample_subjects = subjects[:4]  # First 4 subjects
    fig, axes = plt.subplots(len(sample_subjects), 1, figsize=(16, 3 * len(sample_subjects)))
    
    for idx, subject in enumerate(sample_subjects):
        mask = df['subject'] == subject
        subj_data = df[mask].copy()
        
        # Plot first 14 days
        max_points = min(len(subj_data), 14 * 96)  # ~14 days at 15-min intervals
        plot_data = subj_data.iloc[:max_points]
        
        ax = axes[idx] if len(sample_subjects) > 1 else axes
        
        # Plot activity
        ax2 = ax.twinx()
        ax.plot(plot_data['timestamp'], plot_data['activity'], 
               color='gray', alpha=0.5, linewidth=0.5, label='Activity')
        
        # Plot states as colored background
        if 'hmm_state_filtered' in plot_data.columns:
            states = plot_data['hmm_state_filtered'].fillna(-1)
            unique_states = states[states >= 0].unique()
            colors = plt.colormaps.get_cmap('Set3')(np.linspace(0, 1, len(unique_states)))
            
            for i, state in enumerate(sorted(unique_states)):
                state_mask = states == state
                ax2.fill_between(plot_data['timestamp'], 0, 1, where=state_mask,
                               alpha=0.3, color=colors[i], label=f'State {int(state)}',
                               transform=ax2.get_xaxis_transform())
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity', color='gray')
        ax.set_title(f'{subject} - Sample Time Series (14 days)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('HMM State')
        ax2.set_yticks([])
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'improved_time_series_samples.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved: {out_path}")
    plt.close()


def validate_states(df: pd.DataFrame):
    """Validate HMM states using biological criteria."""
    
    logger.info("\nBiological Validation of HMM States:")
    logger.info("=" * 70)
    
    for subject in df['subject'].unique():
        mask = (df['subject'] == subject) & df['hmm_state_filtered'].notna()
        if mask.sum() == 0:
            continue
        
        subj_data = df[mask].copy()
        
        logger.info(f"\n{subject}:")
        logger.info("-" * 70)
        
        # Analyze each state
        for state in sorted(subj_data['hmm_state_filtered'].unique()):
            state_data = subj_data[subj_data['hmm_state_filtered'] == state]
            
            mean_act = state_data['activity'].mean()
            std_act = state_data['activity'].std()
            n_obs = len(state_data)
            pct = 100 * n_obs / len(subj_data)
            
            # Diel pattern
            hour_dist = state_data['hour'].value_counts()
            peak_hour = hour_dist.idxmax() if len(hour_dist) > 0 else -1
            day_pct = 100 * ((state_data['hour'] >= 6) & (state_data['hour'] < 18)).sum() / len(state_data)
            
            logger.info(
                f"  State {int(state)}: "
                f"mean_activity={mean_act:.2f}±{std_act:.2f}, "
                f"n={n_obs} ({pct:.1f}%), "
                f"peak_hour={peak_hour}h, "
                f"daytime={day_pct:.1f}%"
            )
        
        # Biological interpretation
        state_means = subj_data.groupby('hmm_state_filtered')['activity'].mean().sort_values()
        if len(state_means) >= 2:
            logger.info(f"  → State interpretation:")
            logger.info(f"    - State {int(state_means.idxmin())}: RESTING (lowest activity)")
            logger.info(f"    - State {int(state_means.idxmax())}: ACTIVE (highest activity)")


if __name__ == "__main__":
    run_improved_pipeline()

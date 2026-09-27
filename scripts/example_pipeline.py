#!/usr/bin/env python
"""
Example script demonstrating the thesis analysis pipeline with accelerometer data.

This script shows how to use the pipeline modules programmatically
rather than through Jupyter notebooks.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import pipeline modules
from src import io, qc, features, models_hmm, cosinor, effects, eval
from src.config import load_config_or_defaults
from src.logging_config import setup_logging

# Setup
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure logging
logger = setup_logging(log_level="INFO")
logger.info("Starting pipeline example with accelerometer data")

# Load configuration
config = load_config_or_defaults()
logger.info(f"Configuration loaded: random_state={config['random_state']}")


def run_pipeline():
    """Run the complete analysis pipeline with accelerometer data."""
    
    # 1. Data Intake
    logger.info("=" * 50)
    logger.info("Stage 1: Data Intake - Accelerometer Data")
    logger.info("=" * 50)
    
    # Load accelerometer data
    df = io.load_accelerometer_data('data/raw/sample_accelerometer_data.csv')
    
    # CORRECT APPROACH: Use ActMin as primary activity signal
    # ActMindata is the tag-derived activity metric (validated by manufacturer)
    if 'ActMindata' in df.columns:
        df['activity'] = df['ActMindata']
        logger.info("Using ActMindata as primary activity signal (CORRECT)")
    else:
        logger.error("ActMindata column not found - cannot proceed with correct analysis")
        raise ValueError("ActMindata column required for proper analysis")
    
    # Add relative movement features from accelerometer axes
    # These capture changes and variability, NOT absolute magnitude
    if all(col in df.columns for col in ['XAccel', 'YAccel', 'ZAccel']):
        df = features.add_accelerometer_movement_features(
            df, 
            subject_column='subject',
            windows=[6, 12, 24],
            standardize=True
        )
        logger.info("Added relative movement features from XAccel/YAccel/ZAccel (variance, std, changes)")
    
    io.save_processed_data(df, 'data/processed/intake_data.parquet')
    
    # 2. Quality Control
    logger.info("=" * 50)
    logger.info("Stage 2: Quality Control")
    logger.info("=" * 50)
    
    qc_report = qc.qc_report(df, time_column='timestamp')
    df_clean = qc.remove_duplicates(df)
    io.save_processed_data(df_clean, 'data/processed/qc_data.parquet')
    
    # 3. Feature Engineering
    logger.info("=" * 50)
    logger.info("Stage 3: Feature Engineering")
    logger.info("=" * 50)
    
    # Add rolling statistics on ActMin (primary activity signal)
    df_features = features.add_rolling_statistics_per_individual(
        df_clean, 
        'activity',
        subject_column='subject',
        windows=config['features']['rolling_windows'],
        stats=config['features']['rolling_stats']
    )
    df_features = features.add_time_features(
        df_features,
        'timestamp',
        features=config['features']['time_features']
    )
    
    # Optional: Log transform and standardize activity for HMM
    df_features = features.log_transform_activity(df_features, 'activity', offset=1.0)
    df_features = features.standardize_features(
        df_features, 
        ['activity_log'],
        group_by='subject',
        overwrite=True
    )
    
    io.save_processed_data(df_features, 'data/processed/feature_data.parquet')
    
    # 4. HMM Analysis
    logger.info("=" * 50)
    logger.info("Stage 4: HMM Analysis")
    logger.info("=" * 50)
    
    # Prepare features for HMM: ActMin + relative movement features
    # Use standardized features for stable HMM performance
    feature_cols = ['activity_log_standardized']
    
    # Add standardized movement features if available
    movement_feature_cols = [
        col for col in df_features.columns 
        if '_standardized' in col and any(
            pattern in col for pattern in ['movement_var', 'abs_diff', 'rolling_var']
        )
    ]
    if movement_feature_cols:
        feature_cols.extend(movement_feature_cols[:3])  # Limit to top 3 to avoid overfitting
        logger.info(f"Using {len(feature_cols)} features for HMM: {feature_cols}")
    
    # Remove NaN values (from rolling/diff operations)
    df_hmm = df_features[feature_cols].dropna()
    X = df_hmm.values
    
    hmm_results = models_hmm.fit_hmm_pipeline(
        X,
        state_range=range(2, 6),
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Optimal K: {hmm_results['optimal_k']}")
    logger.info(f"BIC scores: {hmm_results['bic_scores']}")
    logger.warning(
        "NOTE: For long time series, use biological validation instead of IC-based selection. "
        "See METHODOLOGY_CORRECTIONS.md for details."
    )
    
    # 5. Cosinor Analysis (AFTER state identification)
    logger.info("=" * 50)
    logger.info("Stage 5: Cosinor Analysis (State-Based)")
    logger.info("=" * 50)
    
    # Add HMM states to dataframe
    df_clean_subset = df_clean.iloc[:len(hmm_results['states'])].copy()
    df_clean_subset['hmm_state'] = hmm_results['states']
    
    # Apply state-based cosinor (CORRECT approach)
    if 'timestamp' in df_clean_subset.columns:
        cosinor_results = cosinor.fit_cosinor_per_state(
            df_clean_subset,
            time_column='timestamp',
            state_column='hmm_state',
            activity_column='activity',
            period=config['cosinor']['period']
        )
        
        logger.info("State-based cosinor results:")
        logger.info(f"Active probability cosinor: {cosinor_results['active_probability_cosinor']}")
        logger.info(f"Per-state cosinor: {cosinor_results['per_state_cosinor']}")
    
    # 6. Machine Learning
    logger.info("=" * 50)
    logger.info("Stage 6: Machine Learning")
    logger.info("=" * 50)
    
    # Prepare features
    feature_cols_ml = ['hour', 'day_of_week']
    X_train, X_test, y_train, y_test = effects.prepare_ml_data(
        df_features,
        target_column='activity',
        feature_columns=feature_cols_ml,
        test_size=config['ml']['test_size'],
        random_state=RANDOM_STATE
    )
    
    # Random Forest
    rf_results = effects.fit_random_forest(
        X_train, y_train, X_test, y_test,
        task='regression',
        n_estimators=config['ml']['rf']['n_estimators'],
        max_depth=config['ml']['rf']['max_depth'],
        random_state=RANDOM_STATE
    )
    
    # XGBoost
    xgb_results = effects.fit_xgboost(
        X_train, y_train, X_test, y_test,
        task='regression',
        n_estimators=config['ml']['xgb']['n_estimators'],
        learning_rate=config['ml']['xgb']['learning_rate'],
        random_state=RANDOM_STATE
    )
    
    # Compare models
    comparison = effects.compare_models(
        X_train, y_train, X_test, y_test,
        task='regression',
        random_state=RANDOM_STATE
    )
    
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison}")
    
    # Save results
    comparison.to_csv('outputs/model_comparison.csv', index=False)
    
    logger.success("Pipeline completed successfully!")
    logger.info("Results saved to outputs/")
    logger.info(
        "\nKEY METHODOLOGICAL NOTES:\n"
        "1. Used ActMin as primary activity signal (NOT magnitude from axes)\n"
        "2. Added relative movement features from XAccel/YAccel/ZAccel\n"
        "3. Standardized features per individual before HMM\n"
        "4. Applied cosinor AFTER state identification\n"
        "See METHODOLOGY_CORRECTIONS.md for full details."
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

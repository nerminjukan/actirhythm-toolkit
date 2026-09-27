"""Feature engineering module for creating activity metrics and rolling statistics.

ACTIVITY SIGNAL:
- Activity is derived from accelerometer axes: activity_xy = sqrt(X² + Y²).
- X axis captures forward/backward movement; Y axis captures sideways/rotational movement.
- Z axis is excluded from the primary metric because it is absent in some
  collar deployments; using only X and Y ensures cross-dataset comparability.
- ActMindata (firmware-compressed 0-15 index) is preserved for optional
  comparison but is NOT used as the primary signal.

PIPELINE:
1. Compute activity_xy during data loading (see src/io.py)
2. Log-transform: log(activity_xy + 1)
3. Add rolling statistics per individual
4. Standardize features per individual before HMM/ML
5. Apply cosinor AFTER behavioral state identification

Rolling features are intended for modeling input, not for global visualization
across multi-year datasets. For visualization, compute rolling features on short temporal
windows (days to weeks) to avoid unreadable plots.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from loguru import logger
from sklearn.preprocessing import StandardScaler

# Constants
VARIANCE_THRESHOLD = 1e-8  # Minimum variance threshold for standardization


def calculate_activity_metric(
    df: pd.DataFrame,
    value_column: str,
    method: str = 'magnitude'
) -> pd.Series:
    """
    Calculate activity metric from raw sensor data.
    
    WARNING: Do NOT use 'magnitude' or 'squared' methods with unsigned relative
    accelerometer axes (e.g., XAccel/YAccel/ZAccel ranging 0-255). These methods
    assume zero-centered calibrated data and will produce incorrect results.
    
    For accelerometer tags that provide a derived activity metric (e.g., ActMin,
    ActMindata), use that column directly instead of this function.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str or list of str
        Column(s) containing sensor values
    method : str, default='magnitude'
        Method to calculate activity ('magnitude', 'abs', 'squared')
        WARNING: 'magnitude' and 'squared' are INVALID for unsigned relative axes
        
    Returns
    -------
    pd.Series
        Activity metric
        
    Notes
    -----
    Correct usage:
    - Use tag-derived activity metric (ActMin) as primary signal
    - Use add_accelerometer_movement_features() for relative movement patterns
    - Do NOT compute magnitude from unsigned accelerometer axes
    """
    if method == 'magnitude':
        if isinstance(value_column, list):
            logger.warning(
                "Computing magnitude from multiple columns. "
                "If these are unsigned accelerometer axes (0-255), this is INCORRECT. "
                "Use ActMin as activity signal instead."
            )
            # Calculate Euclidean magnitude for multi-axis data
            activity = np.sqrt(sum(df[col]**2 for col in value_column))
        else:
            activity = df[value_column].abs()
    elif method == 'abs':
        if isinstance(value_column, list):
            activity = sum(df[col].abs() for col in value_column)
        else:
            activity = df[value_column].abs()
    elif method == 'squared':
        if isinstance(value_column, list):
            logger.warning(
                "Squaring and summing columns. "
                "If these are unsigned accelerometer axes (0-255), this is INCORRECT. "
                "Use ActMin as activity signal instead."
            )
            activity = sum(df[col]**2 for col in value_column)
        else:
            activity = df[value_column]**2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Activity metric calculated using {method} method")
    return activity


def calculate_activity_xy(
    df: pd.DataFrame,
    x_col: str = 'XAccel',
    y_col: str = 'YAccel',
) -> pd.Series:
    """
    Compute activity from X and Y accelerometer axes.

    activity_xy = sqrt(X² + Y²)

    Z axis is excluded to ensure comparability across datasets where
    ZAccel may be missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with accelerometer columns
    x_col : str, default='XAccel'
        Column name for X-axis values
    y_col : str, default='YAccel'
        Column name for Y-axis values

    Returns
    -------
    pd.Series
        activity_xy values
    """
    activity = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)
    logger.info(f"Computed activity_xy from {x_col}, {y_col}")
    return activity


def add_accelerometer_movement_features(
    df: pd.DataFrame,
    x_col: str = 'XAccel',
    y_col: str = 'YAccel',
    z_col: str = 'ZAccel',
    subject_column: Optional[str] = 'subject',
    windows: List[int] = [6, 12, 24],
    standardize: bool = True
) -> pd.DataFrame:
    """
    Compute relative movement features from unsigned accelerometer axes.
    
    IMPORTANT: This function correctly handles accelerometer data recorded as
    unsigned relative values (0-255). These values encode device orientation
    and gravity - absolute values are NOT meaningful. Only CHANGES and VARIABILITY
    reflect actual movement.
    
    Computed features (per axis and combined):
    - Rolling variance: captures movement intensity
    - Rolling standard deviation: movement variability
    - Absolute first differences: instantaneous change
    - All features are computed per individual and z-scored
    
    DO NOT use this for:
    - Computing VeDBA or ODBA (requires calibrated acceleration)
    - Raw magnitude (assumes zero-centered data)
    - Direct axis combinations (ignores gravity/orientation bias)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with accelerometer columns
    x_col : str, default='XAccel'
        Column name for X-axis values
    y_col : str, default='YAccel'
        Column name for Y-axis values
    z_col : str, default='ZAccel'
        Column name for Z-axis values
    subject_column : str, optional, default='subject'
        Column identifying individuals for per-subject standardization.
        If None, standardizes globally (not recommended for multi-subject data).
    windows : list of int, default=[6, 12, 24]
        Window sizes for rolling statistics
    standardize : bool, default=True
        Whether to z-score features per individual
        
    Returns
    -------
    pd.DataFrame
        Data with added movement features:
        - {axis}_rolling_var_{window}: rolling variance per axis
        - {axis}_rolling_std_{window}: rolling std per axis
        - {axis}_abs_diff: absolute first difference per axis
        - combined_movement_var_{window}: sum of rolling variances
        - All features suffixed with '_standardized' if standardize=True
        
    Examples
    --------
    >>> # Correct approach: ActMin + relative movement features
    >>> df['activity'] = df['ActMindata']  # Primary activity signal
    >>> df = add_accelerometer_movement_features(df, subject_column='subject')
    >>> # Use 'activity' and '*_standardized' features for HMM
    
    Notes
    -----
    Assumptions:
    1. XAccel/YAccel/ZAccel are unsigned relative values (0-255)
    2. Absolute axis values encode orientation + gravity (not movement)
    3. Movement is detected via variance and change over time
    4. Features are computed per individual to avoid cross-subject contamination
    5. Z-scoring ensures features are on comparable scales for HMM/ML
    
    References
    ----------
    Brown et al. (2013). "Observing the unwatchable through acceleration logging 
    of animal behavior." Animal Biotelemetry 1:20.
    - Documents that raw accelerometer values encode posture, not activity intensity
    """
    logger.info("Computing relative movement features from accelerometer axes")
    logger.info(
        "ASSUMPTION: XAccel/YAccel/ZAccel are unsigned relative values (0-255), "
        "encoding orientation + gravity. Only variance and change reflect movement."
    )
    
    df_out = df.copy()
    axes = [(x_col, 'x'), (y_col, 'y'), (z_col, 'z')]
    
    # Compute per-axis features
    for col, axis_name in axes:
        if col not in df_out.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        
        # Absolute first difference (instantaneous change)
        diff_col = f'{axis_name}_abs_diff'
        if subject_column and subject_column in df_out.columns:
            df_out[diff_col] = df_out.groupby(subject_column)[col].transform(
                lambda x: x.diff().abs()
            )
        else:
            df_out[diff_col] = df_out[col].diff().abs()
        
        # Rolling variance and std per window
        for window in windows:
            var_col = f'{axis_name}_rolling_var_{window}'
            std_col = f'{axis_name}_rolling_std_{window}'
            
            if subject_column and subject_column in df_out.columns:
                df_out[var_col] = df_out.groupby(subject_column)[col].transform(
                    lambda x: x.rolling(window=window).var()
                )
                df_out[std_col] = df_out.groupby(subject_column)[col].transform(
                    lambda x: x.rolling(window=window).std()
                )
            else:
                df_out[var_col] = df_out[col].rolling(window=window).var()
                df_out[std_col] = df_out[col].rolling(window=window).std()
    
    # Combined movement features: sum of variances across axes
    for window in windows:
        combined_var_col = f'combined_movement_var_{window}'
        var_cols = [f'{axis}_rolling_var_{window}' for _, axis in axes]
        existing_var_cols = [c for c in var_cols if c in df_out.columns]
        
        if len(existing_var_cols) > 0:
            df_out[combined_var_col] = df_out[existing_var_cols].sum(axis=1)
    
    # Standardize features per individual
    if standardize:
        feature_cols = [
            col for col in df_out.columns 
            if any(pattern in col for pattern in [
                '_rolling_var_', '_rolling_std_', '_abs_diff', 'combined_movement_var_'
            ]) and col in df_out.columns and df_out[col].notna().sum() > 0
        ]
        
        if len(feature_cols) > 0:
            df_out = standardize_features(
                df_out, 
                feature_cols, 
                group_by=subject_column,
                overwrite=True
            )
            logger.success(
                f"Computed and standardized {len(feature_cols)} relative movement features "
                f"per {subject_column if subject_column else 'globally'}"
            )
        else:
            logger.warning("No movement features to standardize")
    else:
        logger.info("Standardization skipped (standardize=False)")
    
    return df_out


def add_rolling_statistics(
    df: pd.DataFrame,
    value_column: str,
    windows: List[int] = [6, 12, 24],
    stats: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Add rolling statistics for specified windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column to calculate statistics on
    windows : list of int, default=[6, 12, 24]
        Window sizes in number of observations
    stats : list of str, default=['mean', 'std', 'min', 'max']
        Statistics to calculate
        
    Returns
    -------
    pd.DataFrame
        Data with added rolling statistics
    """
    df_out = df.copy()
    
    for window in windows:
        for stat in stats:
            col_name = f'{value_column}_rolling_{stat}_{window}'
            
            if stat == 'mean':
                df_out[col_name] = df[value_column].rolling(window=window).mean()
            elif stat == 'std':
                df_out[col_name] = df[value_column].rolling(window=window).std()
            elif stat == 'min':
                df_out[col_name] = df[value_column].rolling(window=window).min()
            elif stat == 'max':
                df_out[col_name] = df[value_column].rolling(window=window).max()
            elif stat == 'median':
                df_out[col_name] = df[value_column].rolling(window=window).median()
            elif stat == 'sum':
                df_out[col_name] = df[value_column].rolling(window=window).sum()
            elif stat == 'var':
                df_out[col_name] = df[value_column].rolling(window=window).var()
    
    logger.info(f"Added rolling statistics for {len(windows)} windows and {len(stats)} stats")
    return df_out


def add_time_features(
    df: pd.DataFrame,
    time_column: str,
    features: List[str] = ['hour', 'day_of_week', 'month']
) -> pd.DataFrame:
    """
    Extract time-based features from timestamp column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_column : str
        Name of timestamp column
    features : list of str
        Time features to extract
        
    Returns
    -------
    pd.DataFrame
        Data with added time features
    """
    df_out = df.copy()
    df_out[time_column] = pd.to_datetime(df_out[time_column])
    
    if 'hour' in features:
        df_out['hour'] = df_out[time_column].dt.hour
    if 'day_of_week' in features:
        df_out['day_of_week'] = df_out[time_column].dt.dayofweek
    if 'month' in features:
        df_out['month'] = df_out[time_column].dt.month
    if 'day_of_year' in features:
        df_out['day_of_year'] = df_out[time_column].dt.dayofyear
    if 'week_of_year' in features:
        df_out['week_of_year'] = df_out[time_column].dt.isocalendar().week
    if 'is_weekend' in features:
        df_out['is_weekend'] = (df_out[time_column].dt.dayofweek >= 5).astype(int)
    
    logger.info(f"Added {len(features)} time features")
    return df_out


def add_lag_features(
    df: pd.DataFrame,
    value_column: str,
    lags: List[int] = [1, 2, 3, 6, 12, 24]
) -> pd.DataFrame:
    """
    Add lagged features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column to create lags for
    lags : list of int
        Lag periods
        
    Returns
    -------
    pd.DataFrame
        Data with lagged features
    """
    df_out = df.copy()
    
    for lag in lags:
        col_name = f'{value_column}_lag_{lag}'
        df_out[col_name] = df[value_column].shift(lag)
    
    logger.info(f"Added {len(lags)} lag features")
    return df_out


def add_difference_features(
    df: pd.DataFrame,
    value_column: str,
    periods: List[int] = [1, 24]
) -> pd.DataFrame:
    """
    Add differenced features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column to difference
    periods : list of int
        Difference periods
        
    Returns
    -------
    pd.DataFrame
        Data with differenced features
    """
    df_out = df.copy()
    
    for period in periods:
        col_name = f'{value_column}_diff_{period}'
        df_out[col_name] = df[value_column].diff(periods=period)
    
    logger.info(f"Added {len(periods)} difference features")
    return df_out


def create_feature_set(
    df: pd.DataFrame,
    value_column: str,
    time_column: Optional[str] = None,
    include_rolling: bool = True,
    include_lags: bool = True,
    include_time: bool = True
) -> pd.DataFrame:
    """
    Create a comprehensive feature set.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Main value column
    time_column : str, optional
        Timestamp column
    include_rolling : bool, default=True
        Include rolling statistics
    include_lags : bool, default=True
        Include lag features
    include_time : bool, default=True
        Include time features
        
    Returns
    -------
    pd.DataFrame
        Data with all features
    """
    df_out = df.copy()
    
    if include_rolling:
        df_out = add_rolling_statistics(df_out, value_column)
    
    if include_lags:
        df_out = add_lag_features(df_out, value_column)
    
    if include_time and time_column is not None:
        df_out = add_time_features(df_out, time_column)
    
    logger.success("Feature engineering complete")
    return df_out


def standardize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    group_by: Optional[str] = None,
    overwrite: bool = True
) -> pd.DataFrame:
    """
    Standardize features using z-score normalization.
    
    This is critical for HMM/HSMM to prevent noisy observations and ensure
    features are on comparable scales. Should be applied before state modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_columns : list of str
        Columns to standardize
    group_by : str, optional
        Column to group by for per-group standardization (e.g., 'subject')
    overwrite : bool, default=True
        If True and standardized columns already exist, overwrite them.
        If False and columns exist, raise an error.
        
    Returns
    -------
    pd.DataFrame
        Data with standardized features
        
    Examples
    --------
    >>> # Standardize globally
    >>> df_std = standardize_features(df, ['activity', 'rolling_mean'])
    >>> # Standardize per individual
    >>> df_std = standardize_features(df, ['activity'], group_by='subject')
    """
    df_out = df.copy()
    
    # Check for existing columns
    std_cols = [f'{col}_standardized' for col in feature_columns]
    existing = [col for col in std_cols if col in df_out.columns]
    if existing and not overwrite:
        raise ValueError(
            f"Standardized columns already exist: {existing}. "
            "Set overwrite=True to replace them."
        )
    
    if group_by is not None:
        # Standardize per group
        for col in feature_columns:
            df_out[f'{col}_standardized'] = df_out.groupby(group_by)[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > VARIANCE_THRESHOLD else (x - x.mean())
            )
        logger.info(f"Standardized {len(feature_columns)} features per {group_by}")
    else:
        # Global standardization
        scaler = StandardScaler()
        standardized_values = scaler.fit_transform(df_out[feature_columns])
        for i, col in enumerate(feature_columns):
            df_out[f'{col}_standardized'] = standardized_values[:, i]
        logger.info(f"Standardized {len(feature_columns)} features globally")
    
    return df_out


def log_transform_activity(
    df: pd.DataFrame,
    value_column: str,
    offset: float = 1.0
) -> pd.DataFrame:
    """
    Apply log transformation to activity data.
    
    Log transformation reduces the impact of extreme values and can help
    reduce noise in HMM state identification. Common for count/activity data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column to transform
    offset : float, default=1.0
        Offset to add before log (to handle zeros): log(x + offset)
        
    Returns
    -------
    pd.DataFrame
        Data with log-transformed column
        
    Examples
    --------
    >>> df = log_transform_activity(df, 'activity', offset=1.0)
    >>> # Creates 'activity_log' column
    """
    df_out = df.copy()
    df_out[f'{value_column}_log'] = np.log(df_out[value_column] + offset)
    logger.info(f"Log-transformed {value_column} with offset={offset}")
    return df_out


def add_rolling_statistics_per_individual(
    df: pd.DataFrame,
    value_column: str,
    subject_column: str = 'subject',
    windows: List[int] = [6, 12, 24],
    stats: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Add rolling statistics computed separately per individual.
    
    This ensures rolling windows don't span across different individuals,
    which is critical for proper feature engineering in multi-subject studies.
    
    NOTE: Rolling features are for MODELING, not global visualization.
    For visualization, compute on short time windows (days-weeks).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column to calculate statistics on
    subject_column : str, default='subject'
        Column identifying different subjects
    windows : list of int, default=[6, 12, 24]
        Window sizes in number of observations
    stats : list of str, default=['mean', 'std', 'min', 'max']
        Statistics to calculate
        
    Returns
    -------
    pd.DataFrame
        Data with added per-individual rolling statistics
        
    Examples
    --------
    >>> df = add_rolling_statistics_per_individual(
    ...     df, 'activity', subject_column='subject', windows=[24]
    ... )
    """
    df_out = df.copy()
    
    for window in windows:
        for stat in stats:
            col_name = f'{value_column}_rolling_{stat}_{window}'
            
            if stat == 'mean':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).mean()
                )
            elif stat == 'std':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).std()
                )
            elif stat == 'min':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).min()
                )
            elif stat == 'max':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).max()
                )
            elif stat == 'median':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).median()
                )
            elif stat == 'sum':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).sum()
                )
            elif stat == 'var':
                df_out[col_name] = df_out.groupby(subject_column)[value_column].transform(
                    lambda x: x.rolling(window=window).var()
                )
    
    logger.info(
        f"Added rolling statistics per {subject_column} for "
        f"{len(windows)} windows and {len(stats)} stats"
    )
    return df_out

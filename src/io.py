"""Data ingestion module for loading and saving data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from loguru import logger


def load_raw_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load raw data from various file formats.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data file
    **kwargs
        Additional arguments passed to pandas read functions
        
    Returns
    -------
    pd.DataFrame
        Loaded data
        
    Examples
    --------
    >>> df = load_raw_data('data/raw/sample.csv')
    >>> df = load_raw_data('data/raw/sample.xlsx', sheet_name='Sheet1')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif suffix == '.json':
        df = pd.read_json(filepath, **kwargs)
    elif suffix == '.feather':
        df = pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    logger.info(f"Loaded data with shape {df.shape}")
    return df


def save_processed_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = 'parquet',
    **kwargs
) -> None:
    """
    Save processed data to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str or Path
        Output file path
    format : str, default='parquet'
        Output format ('parquet', 'csv', 'excel', 'feather')
    **kwargs
        Additional arguments passed to pandas write functions
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to {filepath}")
    
    if format == 'parquet':
        df.to_parquet(filepath, **kwargs)
    elif format == 'csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif format == 'excel':
        df.to_excel(filepath, index=False, **kwargs)
    elif format == 'feather':
        df.to_feather(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.success(f"Data saved successfully to {filepath}")


def load_sample_data(
    filepath: Union[str, Path],
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load a sample of the data for testing.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data file
    sample_size : int, optional
        Number of rows to sample. If None, loads all data
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Sampled data
    """
    df = load_raw_data(filepath)
    
    if sample_size is not None and sample_size < len(df):
        logger.info(f"Sampling {sample_size} rows from {len(df)} total rows")
        df = df.sample(n=sample_size, random_state=random_state)
    
    return df


def load_accelerometer_data(
    filepath: Union[str, Path],
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load accelerometer data and compute activity from X/Y accelerometer axes.
    
    Expected columns: ActMindata, XAccel, YAccel, ZAccel, PostChg, PostCt,
                     ACTEndTimeAllS, serial, subject
    
    DATA FORMAT:
    - XAccel/YAccel: Difference-based activity values (0-255) measured 4x/sec
      and averaged over the sampling interval. These represent relative
      mean acceleration per axis over the bin, in ADC units.
    - ZAccel: Same format; present for every record in the study dataset.
    - ActMindata: Firmware count of minutes scored active within the bin
      (0-15), derived from a step/threshold rule. This is the only genuine
      movement measure in the export and is the pipeline's activity signal.

    Note on the axis columns: because only the *average* of each axis is
    stored, their vector magnitude is dominated by gravity and encodes body
    orientation, not movement intensity. It shortens as within-bin
    orientation varies, so it is negatively correlated with real activity
    (hourly profile r = -0.96 against ActMindata). activity_xy and
    activity_xyz are therefore computed as posture descriptors only and must
    not be used as intensity indices. A true intensity metric (e.g. ODBA or
    VeDBA) would require within-bin variance, which this export does not
    retain.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the accelerometer data CSV file
    parse_dates : bool, default=True
        Whether to parse the ACTEndTimeAllS column as datetime
        
    Returns
    -------
    pd.DataFrame
        Loaded accelerometer data with:
        - 'ActMindata': minutes active per bin (primary activity signal)
        - 'activity_xy' / 'activity_xyz': mean-acceleration vector magnitude,
          retained as posture descriptors only
        - 'activity': alias for the configured primary signal
        
    Examples
    --------
    >>> df = load_accelerometer_data('data/raw/sample_accelerometer_data.csv')
    >>> # activity_xy is already computed and set as 'activity'
    >>> print(df['activity'].describe())
    """
    logger.info(f"Loading accelerometer data from {filepath}")
    
    # Load CSV data
    df = pd.read_csv(filepath)
    
    # Parse timestamp column if requested - use fast format parsing
    if parse_dates and 'ACTEndTimeAllS' in df.columns:
        # Specify format for faster parsing (example: "7/8/23 10:46")
        try:
            df['timestamp'] = pd.to_datetime(df['ACTEndTimeAllS'], format='%m/%d/%y %H:%M')
        except Exception:
            # Fallback to slower auto-detection if format doesn't match
            logger.warning("Fast date parsing failed, using slower method...")
            df['timestamp'] = pd.to_datetime(df['ACTEndTimeAllS'], format='mixed')
        logger.info("Parsed ACTEndTimeAllS to timestamp column")
    
    # Handle NA values in numeric columns
    numeric_cols = ['ActMindata', 'XAccel', 'YAccel', 'ZAccel', 'PostChg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- Detect accelerometer axis columns ---
    x_col = _detect_column(df, ['XAccel', 'ActX', 'X_accel', 'x_accel'])
    y_col = _detect_column(df, ['YAccel', 'ActY', 'Y_accel', 'y_accel'])
    z_col = _detect_column(df, ['ZAccel', 'ActZ', 'Z_accel', 'z_accel'])
    
    if x_col is None or y_col is None:
        raise ValueError(
            f"Required accelerometer columns (X and Y) not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    # --- Posture descriptors: magnitude of the mean acceleration vector ---
    # Gravity-dominated, so these describe orientation, not movement intensity.
    df['activity_xy'] = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)
    logger.info(
        f"Computed posture descriptor activity_xy = sqrt({x_col}² + {y_col}²). "
        f"Range: {df['activity_xy'].min():.2f} – {df['activity_xy'].max():.2f}"
    )

    if z_col is not None:
        df['activity_xyz'] = np.sqrt(
            df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2
        )
        n_missing_z = int(df[z_col].isna().sum())
        logger.info(
            f"Computed posture descriptor activity_xyz "
            f"({n_missing_z} missing {z_col} values)."
        )

    # The firmware minutes-active count is the only genuine movement measure here.
    if 'ActMindata' in df.columns:
        df['activity'] = df['ActMindata']
        logger.info("Set ActMindata (minutes active per bin) as the activity signal")
    else:
        df['activity'] = df['activity_xy']
        logger.warning(
            "ActMindata not found; falling back to the activity_xy posture "
            "descriptor, which is NOT a movement-intensity measure."
        )
    
    logger.info(f"Loaded accelerometer data with shape {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Subjects: {df['subject'].unique() if 'subject' in df.columns else 'N/A'}")
    
    return df


def load_accelerometer_data_from_directory(
    directory: Union[str, Path],
    pattern: str = '*.csv',
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate per-subject accelerometer files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing per-subject CSV files
    pattern : str, default='*.csv'
        Glob pattern for files to load
    parse_dates : bool, default=True
        Whether to parse ACTEndTimeAllS into timestamp

    Returns
    -------
    pd.DataFrame
        Concatenated accelerometer dataframe for all subjects
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' in {directory}"
        )

    logger.info(
        f"Loading accelerometer data from directory {directory} "
        f"({len(files)} files)"
    )

    parts = []
    for fp in files:
        part = load_accelerometer_data(fp, parse_dates=parse_dates)

        # Fallback in case subject column is missing in some files.
        if 'subject' not in part.columns or part['subject'].isna().all():
            inferred_subject = fp.stem.split('.')[0]
            part['subject'] = inferred_subject

        parts.append(part)

    df = pd.concat(parts, axis=0, ignore_index=True)
    logger.success(
        f"Loaded concatenated accelerometer data: {len(df):,} rows, "
        f"{df['subject'].nunique() if 'subject' in df.columns else 0} subjects"
    )
    return df


def _detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_subject_metadata(
    filepath: Union[str, Path],
) -> pd.DataFrame:
    """
    Load the hourly subject-metadata file (ACT_Wet_Dry_Seasons_BR.hourly.csv).

    The file contains one row per subject × day × hour with columns:
        subject, day, hour, ACT, inactive, month, jday, weekday, weekend,
        sex, location, breeding, week_iso, season

    Static columns (constant per subject):
        sex        – Male / Female
        location   – Farmlands / Captive / Canastra

    Time-varying columns (change across dates for a subject):
        breeding   – Reproductive / Non-Reproductive
        season     – Dry / Rainy

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Metadata with parsed date column and all original columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    logger.info(f"Loading subject metadata from {filepath}")
    meta = pd.read_csv(filepath)

    # Parse 'day' to datetime
    meta['day'] = pd.to_datetime(meta['day'])
    logger.info(
        f"Loaded metadata: {meta.shape[0]:,} rows, "
        f"{meta['subject'].nunique()} subjects, "
        f"date range {meta['day'].min().date()} – {meta['day'].max().date()}"
    )
    return meta


def merge_metadata(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    subject_col: str = 'subject',
) -> pd.DataFrame:
    """
    Merge subject metadata into accelerometer data on subject + date + hour.

    Static columns (sex, location) are joined by subject name alone so that
    subjects present in the accelerometer file but absent from the metadata
    still get NaN rather than losing rows.

    Time-varying columns (breeding, season) are joined on
    (subject, date, hour) since they change over time.

    Parameters
    ----------
    df : pd.DataFrame
        Accelerometer dataframe with ``timestamp_col`` (datetime) and
        ``subject_col`` columns.
    meta : pd.DataFrame
        Metadata dataframe returned by :func:`load_subject_metadata`.
    timestamp_col : str
        Name of the datetime column in *df*.
    subject_col : str
        Name of the subject column in both dataframes.

    Returns
    -------
    pd.DataFrame
        *df* with added columns: sex, location, breeding, season.
    """
    df_out = df.copy()

    # ---- 1. Static columns: sex, location (constant per subject) ----
    static_cols = ['sex', 'location']
    available_static = [c for c in static_cols if c in meta.columns]

    if available_static:
        static_map = (
            meta.groupby(subject_col)[available_static]
            .first()
            .reset_index()
        )
        df_out = df_out.merge(static_map, on=subject_col, how='left')
        n_mapped = df_out[available_static[0]].notna().sum()
        logger.info(
            f"Merged static metadata ({', '.join(available_static)}): "
            f"{n_mapped:,}/{len(df_out):,} rows mapped"
        )

    # ---- 2. Time-varying columns: breeding, season ----
    time_cols = ['breeding', 'season']
    available_time = [c for c in time_cols if c in meta.columns]

    if available_time:
        # Build join keys from timestamp
        df_out['_merge_date'] = df_out[timestamp_col].dt.normalize()
        df_out['_merge_hour'] = df_out[timestamp_col].dt.hour

        meta_tv = meta[[subject_col, 'day', 'hour'] + available_time].copy()
        meta_tv = meta_tv.rename(columns={'day': '_merge_date', 'hour': '_merge_hour'})
        meta_tv['_merge_date'] = pd.to_datetime(meta_tv['_merge_date']).dt.normalize()

        df_out = df_out.merge(
            meta_tv,
            on=[subject_col, '_merge_date', '_merge_hour'],
            how='left',
        )

        df_out.drop(columns=['_merge_date', '_merge_hour'], inplace=True)

        n_mapped = df_out[available_time[0]].notna().sum()
        logger.info(
            f"Merged time-varying metadata ({', '.join(available_time)}): "
            f"{n_mapped:,}/{len(df_out):,} rows mapped"
        )

    return df_out

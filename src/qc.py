"""Quality control module for detecting gaps, duplicates, and data issues."""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import pandas as pd
import numpy as np
from loguru import logger


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
        
    Returns
    -------
    pd.DataFrame
        Summary of missing values per column
    """
    missing_summary = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary['missing_count'] > 0]
    
    if len(missing_summary) > 0:
        logger.warning(f"Found missing values in {len(missing_summary)} columns")
    else:
        logger.info("No missing values detected")
    
    return missing_summary


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> Tuple[pd.DataFrame, int]:
    """
    Detect and optionally remove duplicate rows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    subset : list of str, optional
        Columns to consider for duplicates. If None, uses all columns
    keep : str, default='first'
        Which duplicates to keep ('first', 'last', False)
        
    Returns
    -------
    tuple
        (DataFrame with duplicates marked, number of duplicates)
    """
    duplicates = df.duplicated(subset=subset, keep=keep)
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate rows")
    else:
        logger.info("No duplicates detected")
    
    return df[duplicates], n_duplicates


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    subset : list of str, optional
        Columns to consider for duplicates
    keep : str, default='first'
        Which duplicates to keep
        
    Returns
    -------
    pd.DataFrame
        Data with duplicates removed
    """
    initial_len = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = initial_len - len(df_clean)
    
    logger.info(f"Removed {n_removed} duplicate rows")
    return df_clean


# Documented device-removal times and confirmed pre-explant battery failures.
# Deliberately empty in the public repository: removal times are study metadata
# that identify individual animals. Populate it with load_deployment_windows(),
# or pass an explicit mapping to apply_deployment_windows().
DEPLOYMENT_END_OVERRIDES: Dict[str, str] = {}

DEPLOYMENT_WINDOWS_FILE = 'deployment_windows.csv'


def load_deployment_windows(
    path: Union[str, Path] = DEPLOYMENT_WINDOWS_FILE,
    update_default: bool = True,
) -> Dict[str, str]:
    """
    Load documented end-of-deployment timestamps from a local CSV file.

    The file is not distributed with the package. It must contain the columns
    ``subject`` and ``end_timestamp``, one row per subject, e.g.::

        subject,end_timestamp,reason
        A,2024-04-02 00:00,battery failure before explant
        B,2023-07-07 10:15,documented device removal

    Parameters
    ----------
    path : str or Path, default='deployment_windows.csv'
        Location of the CSV file. A missing file yields an empty mapping.
    update_default : bool, default=True
        Also populate the module-level DEPLOYMENT_END_OVERRIDES.

    Returns
    -------
    dict
        Mapping of subject -> cut-off timestamp
    """
    path = Path(path)
    if not path.exists():
        logger.warning(
            f"No deployment-window file at '{path}'; post-removal records "
            "cannot be trimmed and will be treated as valid observations"
        )
        return {}

    table = pd.read_csv(path)
    missing = {'subject', 'end_timestamp'} - set(table.columns)
    if missing:
        raise ValueError(
            f"'{path}' is missing required column(s): {sorted(missing)}"
        )

    windows = {
        str(row.subject): str(row.end_timestamp)
        for row in table.itertuples(index=False)
    }
    logger.info(f"Loaded {len(windows)} deployment windows from '{path}'")

    if update_default:
        DEPLOYMENT_END_OVERRIDES.clear()
        DEPLOYMENT_END_OVERRIDES.update(windows)

    return windows


def apply_deployment_windows(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, str]] = None,
    subject_column: str = 'subject',
    time_column: str = 'timestamp',
) -> pd.DataFrame:
    """
    Trim each subject's record at its documented end of deployment.

    Post-removal records cannot be identified from the signal alone: while a
    retrieved biologger is being transported it still registers varying posture
    and occasional activity, which is indistinguishable from genuine low-activity
    behaviour. Documented removal times are therefore required.

    Parameters
    ----------
    df : pd.DataFrame
        Data with subject and timestamp columns
    overrides : dict, optional
        Mapping of subject -> cut-off timestamp. Defaults to
        DEPLOYMENT_END_OVERRIDES.
    subject_column : str, default='subject'
    time_column : str, default='timestamp'

    Returns
    -------
    pd.DataFrame
        Data with out-of-deployment records removed
    """
    overrides = DEPLOYMENT_END_OVERRIDES if overrides is None else overrides

    if subject_column not in df.columns or time_column not in df.columns:
        logger.warning(
            f"Cannot apply deployment windows: missing '{subject_column}' or "
            f"'{time_column}'"
        )
        return df

    drop_mask = pd.Series(False, index=df.index)

    for subject, cutoff in overrides.items():
        cut = pd.Timestamp(cutoff)
        subject_mask = df[subject_column] == subject
        if not subject_mask.any():
            logger.warning(f"Deployment override for '{subject}' matched no rows")
            continue

        remove = subject_mask & (df[time_column] >= cut)
        n = int(remove.sum())
        drop_mask |= remove
        logger.info(f"  {subject}: removed {n} records from {cut}")

    n_total = int(drop_mask.sum())
    logger.success(
        f"Deployment-window filter removed {n_total} records "
        f"({100 * n_total / len(df):.2f}% of {len(df):,})"
    )
    return df.loc[~drop_mask].copy()


def assign_subject_codes(
    df: pd.DataFrame,
    subject_column: str = 'subject',
    prefix: str = 'S',
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Replace subject names with anonymous codes, ordered by first observation.

    Individual names are not published for these animals, so all reported output
    uses codes instead.

    Returns
    -------
    (pd.DataFrame, dict)
        Data with a 'subject_code' column, and the name -> code mapping.
    """
    if 'timestamp' in df.columns:
        order = (
            df.groupby(subject_column)['timestamp'].min().sort_values().index.tolist()
        )
    else:
        order = sorted(df[subject_column].unique())

    width = len(str(len(order)))
    mapping = {name: f"{prefix}{i:0{width}d}" for i, name in enumerate(order, start=1)}

    df = df.copy()
    df['subject_code'] = df[subject_column].map(mapping)
    logger.info(f"Assigned anonymous codes to {len(mapping)} subjects")
    return df, mapping


def detect_time_gaps(
    df: pd.DataFrame,
    time_column: str,
    expected_freq: str = '1H',
    tolerance: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect gaps in time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_column : str
        Name of the timestamp column
    expected_freq : str, default='1H'
        Expected frequency of observations
    tolerance : str, optional
        Tolerance for missing observations
        
    Returns
    -------
    pd.DataFrame
        Information about detected gaps
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    
    # Calculate time differences
    time_diffs = df[time_column].diff()
    expected_diff = pd.Timedelta(expected_freq)
    
    if tolerance is None:
        tolerance = expected_diff * 0.1
    else:
        tolerance = pd.Timedelta(tolerance)
    
    # Detect gaps larger than expected
    gaps = time_diffs > (expected_diff + tolerance)
    gap_info = df[gaps][[time_column]].copy()
    gap_info['gap_duration'] = time_diffs[gaps]
    
    if len(gap_info) > 0:
        logger.warning(f"Found {len(gap_info)} time gaps in the data")
    else:
        logger.info("No significant time gaps detected")
    
    return gap_info


def validate_data_ranges(
    df: pd.DataFrame,
    range_dict: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Validate that numeric columns fall within expected ranges.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    range_dict : dict
        Dictionary mapping column names to (min, max) tuples
        
    Returns
    -------
    pd.DataFrame
        Rows with values outside expected ranges
    """
    out_of_range = pd.DataFrame()
    
    for col, (min_val, max_val) in range_dict.items():
        if col in df.columns:
            mask = (df[col] < min_val) | (df[col] > max_val)
            if mask.any():
                logger.warning(
                    f"Column '{col}': {mask.sum()} values outside "
                    f"range [{min_val}, {max_val}]"
                )
                out_of_range = pd.concat([out_of_range, df[mask]])
    
    return out_of_range.drop_duplicates()


def qc_report(df: pd.DataFrame, time_column: Optional[str] = None) -> Dict:
    """
    Generate a comprehensive quality control report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    time_column : str, optional
        Name of the timestamp column for time-based checks
        
    Returns
    -------
    dict
        QC report with various metrics
    """
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': detect_missing_values(df).to_dict(),
        'n_duplicates': detect_duplicates(df)[1],
        'dtypes': df.dtypes.to_dict(),
    }
    
    if time_column and time_column in df.columns:
        gaps = detect_time_gaps(df, time_column)
        report['n_time_gaps'] = len(gaps)
        report['time_gaps'] = gaps.to_dict() if len(gaps) > 0 else {}
    
    logger.info("QC report generated successfully")
    return report

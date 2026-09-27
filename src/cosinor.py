"""Cosinor analysis for circadian rhythm detection."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import optimize, stats
from loguru import logger


def cosinor_model(t: np.ndarray, mesor: float, amplitude: float, acrophase: float) -> np.ndarray:
    """
    Cosinor model function.
    
    Parameters
    ----------
    t : np.ndarray
        Time values (in hours)
    mesor : float
        Midline Estimating Statistic Of Rhythm (mean level)
    amplitude : float
        Half the difference between peak and trough
    acrophase : float
        Time of peak (in radians)
        
    Returns
    -------
    np.ndarray
        Predicted values
    """
    omega = 2 * np.pi / 24  # 24-hour period
    return mesor + amplitude * np.cos(omega * t - acrophase)


def fit_cosinor(
    t: np.ndarray,
    y: np.ndarray,
    period: float = 24.0
) -> Dict[str, float]:
    """
    Fit cosinor model to time series data.
    
    Parameters
    ----------
    t : np.ndarray
        Time values (in hours)
    y : np.ndarray
        Observed values
    period : float, default=24.0
        Period of the rhythm (in hours)
        
    Returns
    -------
    dict
        Fitted parameters: mesor, amplitude, acrophase
    """
    logger.info(f"Fitting cosinor model with period={period}h")
    
    # Initial parameter estimates
    mesor_init = np.mean(y)
    amplitude_init = (np.max(y) - np.min(y)) / 2
    acrophase_init = 0.0
    
    initial_params = [mesor_init, amplitude_init, acrophase_init]
    
    # Fit the model
    def model_wrapper(t, mesor, amplitude, acrophase):
        omega = 2 * np.pi / period
        return mesor + amplitude * np.cos(omega * t - acrophase)
    
    try:
        popt, pcov = optimize.curve_fit(
            model_wrapper,
            t,
            y,
            p0=initial_params,
            maxfev=10000
        )
        
        mesor, amplitude, acrophase = popt

        # A negative amplitude is equivalent to a positive one shifted by half a
        # period; normalise so the reported acrophase is the true activity peak.
        if amplitude < 0:
            amplitude = -amplitude
            acrophase = acrophase + np.pi

        acrophase = acrophase % (2 * np.pi)

        # Convert acrophase to hours (0-24)
        acrophase_hours = (acrophase / (2 * np.pi) * period) % period
        
        results = {
            'mesor': mesor,
            'amplitude': amplitude,
            'acrophase': acrophase,
            'acrophase_hours': acrophase_hours,
            'period': period,
        }
        
        # Calculate R-squared
        y_pred = model_wrapper(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        results['r_squared'] = r_squared
        
        logger.success(
            f"Cosinor fitted: MESOR={mesor:.3f}, "
            f"Amplitude={amplitude:.3f}, "
            f"Acrophase={acrophase_hours:.2f}h, "
            f"R²={r_squared:.3f}"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Cosinor fitting failed: {e}")
        return {
            'mesor': np.nan,
            'amplitude': np.nan,
            'acrophase': np.nan,
            'acrophase_hours': np.nan,
            'period': period,
            'r_squared': np.nan,
        }


def fit_multi_period_cosinor(
    t: np.ndarray,
    y: np.ndarray,
    periods: list = [24.0, 12.0]
) -> Dict[float, Dict[str, float]]:
    """
    Fit cosinor models with multiple periods.
    
    Parameters
    ----------
    t : np.ndarray
        Time values (in hours)
    y : np.ndarray
        Observed values
    periods : list, default=[24.0, 12.0]
        List of periods to test
        
    Returns
    -------
    dict
        Results for each period
    """
    results = {}
    
    for period in periods:
        results[period] = fit_cosinor(t, y, period)
    
    return results


def multi_component_cosinor_model(t: np.ndarray, params: np.ndarray,
                                   periods: List[float]) -> np.ndarray:
    """
    Multi-component cosinor model: MESOR + sum of cosine harmonics.

    y(t) = M + sum_{j=1}^{n} A_j * cos(2*pi*t/T_j - phi_j)

    Parameters
    ----------
    t : np.ndarray
        Time values (in hours)
    params : np.ndarray
        [mesor, amp1, phase1, amp2, phase2, ...]
    periods : list of float
        Periods for each component (e.g. [24, 12, 8, 6, 4.8])

    Returns
    -------
    np.ndarray
        Predicted values
    """
    mesor = params[0]
    y = np.full_like(t, mesor, dtype=float)
    for j, T in enumerate(periods):
        amp = params[1 + 2 * j]
        phi = params[2 + 2 * j]
        y += amp * np.cos(2 * np.pi * t / T - phi)
    return y


def fit_multi_component_cosinor(
    t: np.ndarray,
    y: np.ndarray,
    n_components: int = 5,
    base_period: float = 24.0,
) -> Dict:
    """
    Fit a multi-component cosinor model with n_components harmonics.

    The periods are base_period / k for k = 1, ..., n_components,
    i.e. 24h, 12h, 8h, 6h, 4.8h for the default 5 components.

    This captures non-sinusoidal rhythms (e.g. bimodal crepuscular peaks)
    that a single cosine cannot represent.

    Parameters
    ----------
    t : np.ndarray
        Time values (in hours, typically hour-of-day 0..23)
    y : np.ndarray
        Observed values
    n_components : int, default=5
        Number of harmonic components
    base_period : float, default=24.0
        Fundamental period (hours)

    Returns
    -------
    dict
        Fitted parameters including per-component amplitude/acrophase,
        overall R², F-test, and the periods used.
    """
    periods = [base_period / k for k in range(1, n_components + 1)]
    n_params = 1 + 2 * n_components  # mesor + (amp, phase) per component

    logger.info(
        f"Fitting {n_components}-component cosinor, "
        f"periods={[f'{p:.1f}h' for p in periods]}"
    )

    # Initial estimates
    mesor_init = float(np.mean(y))
    amp_init = float((np.max(y) - np.min(y)) / 2)
    p0 = [mesor_init]
    for j in range(n_components):
        p0.append(amp_init / (j + 1))  # decreasing amplitude guess
        p0.append(0.0)                 # phase guess

    def model_fn(t, *params):
        return multi_component_cosinor_model(t, np.array(params), periods)

    try:
        popt, pcov = optimize.curve_fit(
            model_fn, t, y, p0=p0, maxfev=20000
        )

        mesor = popt[0]
        components = []
        for j in range(n_components):
            amp = popt[1 + 2 * j]
            phi = popt[2 + 2 * j]

            # curve_fit can return a negative amplitude; A*cos(wt - p) with A < 0
            # equals |A|*cos(wt - (p + pi)), so fold the sign into the phase to
            # keep amplitudes positive and acrophases interpretable.
            if amp < 0:
                amp = -amp
                phi = phi + np.pi

            phi = phi % (2 * np.pi)
            acro_hours = (phi / (2 * np.pi) * periods[j]) % periods[j]
            components.append({
                'period': periods[j],
                'amplitude': amp,
                'acrophase_rad': phi,
                'acrophase_hours': acro_hours,
            })

        y_pred = model_fn(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # F-test: model vs. flat mean
        n = len(y)
        k = n_params
        if n > k and ss_tot > 0:
            f_stat = ((ss_tot - ss_res) / (k - 1)) / (ss_res / (n - k))
            p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            f_stat = np.nan
            p_value = np.nan

        result = {
            'mesor': mesor,
            'n_components': n_components,
            'periods': periods,
            'components': components,
            'r_squared': r_squared,
            'f_statistic': f_stat,
            'p_value': p_value,
            'n_params': n_params,
            'popt': popt.tolist(),
        }

        # Also store the dominant (fundamental, 24h) component separately
        result['amplitude'] = components[0]['amplitude']
        result['acrophase_hours'] = components[0]['acrophase_hours']

        logger.success(
            f"Multi-component cosinor: MESOR={mesor:.3f}, "
            f"R²={r_squared:.3f}, F={f_stat:.1f}, p={p_value:.2e}, "
            f"dominant amp={components[0]['amplitude']:.3f}"
        )
        return result

    except Exception as e:
        logger.error(f"Multi-component cosinor fitting failed: {e}")
        return {
            'mesor': np.nan,
            'n_components': n_components,
            'periods': periods,
            'components': [],
            'r_squared': np.nan,
            'f_statistic': np.nan,
            'p_value': np.nan,
            'amplitude': np.nan,
            'acrophase_hours': np.nan,
        }


def fit_multi_component_cosinor_per_subject(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    subject_column: str = 'subject',
    n_components: int = 5,
    base_period: float = 24.0,
) -> pd.DataFrame:
    """
    Fit multi-component cosinor per subject.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time, value, and subject columns
    time_column : str
        Timestamp column name
    value_column : str
        Activity value column name
    subject_column : str
        Subject identifier column
    n_components : int
        Number of harmonic components
    base_period : float
        Fundamental period (hours)

    Returns
    -------
    pd.DataFrame
        One row per subject with cosinor parameters
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df['hour_of_day'] = df[time_column].dt.hour + df[time_column].dt.minute / 60

    rows = []
    for subj in sorted(df[subject_column].unique()):
        subj_data = df[df[subject_column] == subj]
        hourly = subj_data.groupby('hour_of_day')[value_column].mean().reset_index()

        result = fit_multi_component_cosinor(
            hourly['hour_of_day'].values,
            hourly[value_column].values,
            n_components=n_components,
            base_period=base_period,
        )

        row = {
            'subject': subj,
            'mesor': result['mesor'],
            'r_squared': result['r_squared'],
            'f_statistic': result.get('f_statistic', np.nan),
            'p_value': result.get('p_value', np.nan),
        }
        for j, comp in enumerate(result.get('components', [])):
            row[f'amplitude_{j+1}'] = comp['amplitude']
            row[f'acrophase_hours_{j+1}'] = comp['acrophase_hours']
            row[f'period_{j+1}'] = comp['period']

        rows.append(row)

    return pd.DataFrame(rows)


def extract_cosinor_features(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    period: float = 24.0
) -> Dict[str, float]:
    """
    Extract cosinor features from a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    value_column : str
        Column containing values to analyze
    time_column : str
        Column containing time values
    period : float, default=24.0
        Period to analyze
        
    Returns
    -------
    dict
        Cosinor parameters
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert time to hours from start
    t_start = df[time_column].min()
    t_hours = (df[time_column] - t_start).dt.total_seconds() / 3600
    
    # Extract hour of day for 24h period analysis
    if period == 24.0:
        t_values = df[time_column].dt.hour + df[time_column].dt.minute / 60
    else:
        t_values = t_hours.values
    
    y_values = df[value_column].values
    
    return fit_cosinor(t_values, y_values, period)


def test_rhythmicity(
    t: np.ndarray,
    y: np.ndarray,
    period: float = 24.0,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Test statistical significance of rhythmicity using permutation test.
    
    Parameters
    ----------
    t : np.ndarray
        Time values
    y : np.ndarray
        Observed values
    period : float, default=24.0
        Period to test
    n_permutations : int, default=1000
        Number of permutations
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    dict
        Test results including p-value
    """
    logger.info("Testing rhythmicity significance")
    
    # Fit actual data
    actual_result = fit_cosinor(t, y, period)
    actual_amplitude = actual_result['amplitude']
    
    # Permutation test
    np.random.seed(random_state)
    permuted_amplitudes = []
    
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        perm_result = fit_cosinor(t, y_permuted, period)
        permuted_amplitudes.append(perm_result['amplitude'])
    
    # Calculate p-value
    p_value = np.mean(np.array(permuted_amplitudes) >= actual_amplitude)
    
    result = {
        'actual_amplitude': actual_amplitude,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
    
    logger.info(f"Rhythmicity test: p-value={p_value:.4f}")
    return result


def fit_cosinor_per_state(
    df: pd.DataFrame,
    time_column: str,
    state_column: str = 'hmm_state',
    activity_column: str = 'activity',
    period: float = 24.0,
    active_states: Optional[List[int]] = None
) -> Dict:
    """
    Fit cosinor models per behavioral state and for active state probability.
    
    CORRECT USAGE: Cosinor should be applied AFTER behavioral state identification,
    not to raw activity which mixes multiple behavioral modes.
    
    This function computes:
    1. Cosinor on probability of being in active states: P(state ∈ {active states})
    2. State-conditioned mean activity: E[activity | state]
    3. Per-state cosinor parameters (MESOR, amplitude, acrophase)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with timestamps, states, and activity
    time_column : str
        Name of timestamp column
    state_column : str, default='hmm_state'
        Column with HMM state assignments
    activity_column : str, default='activity'
        Column with activity values
    period : float, default=24.0
        Period in hours
    active_states : list of int, optional
        Which states are considered "active" (e.g., [2, 3] for states 2 and 3).
        If None, uses states with above-median mean activity.
        
    Returns
    -------
    dict
        Results including:
        - 'active_probability_cosinor': cosinor fit to P(active state)
        - 'per_state_cosinor': dict of cosinor fits per state
        - 'state_mean_activity': mean activity per state
        
    Notes
    -----
    This is the biologically valid approach. Raw activity cosinor is INCORRECT
    because it assumes a single dominant rhythm, which is violated when data
    mixes rest, low activity, and high activity states.
    """
    logger.info("Fitting cosinor per behavioral state (CORRECT METHOD)")
    
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Extract hour of day
    df['hour_of_day'] = df[time_column].dt.hour + df[time_column].dt.minute / 60
    
    # Determine active states if not provided
    if active_states is None:
        state_means = df.groupby(state_column)[activity_column].mean()
        median_activity = state_means.median()
        active_states = state_means[state_means > median_activity].index.tolist()
        logger.info(f"Auto-detected active states (above median): {active_states}")
    
    # Create binary active indicator
    df['is_active'] = df[state_column].isin(active_states).astype(int)
    
    # Aggregate by hour to get probability and mean activity
    hourly = df.groupby('hour_of_day').agg({
        'is_active': 'mean',  # P(active state)
        activity_column: 'mean',  # E[activity]
        state_column: lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    # Fit cosinor to active state probability
    logger.info("Fitting cosinor to P(active state)")
    active_prob_cosinor = fit_cosinor(
        hourly['hour_of_day'].values,
        hourly['is_active'].values,
        period=period
    )
    
    # Fit per-state cosinor
    per_state_results = {}
    unique_states = sorted(df[state_column].unique())
    
    for state in unique_states:
        logger.info(f"Fitting cosinor for state {state}")
        state_data = df[df[state_column] == state]
        
        state_hourly = state_data.groupby('hour_of_day')[activity_column].mean().reset_index()
        
        if len(state_hourly) >= 12:  # Need sufficient data points
            per_state_results[state] = fit_cosinor(
                state_hourly['hour_of_day'].values,
                state_hourly[activity_column].values,
                period=period
            )
        else:
            logger.warning(f"Insufficient data for state {state} cosinor")
            per_state_results[state] = {
                'mesor': np.nan,
                'amplitude': np.nan,
                'acrophase_hours': np.nan,
                'r_squared': np.nan,
            }
    
    # Calculate state mean activities
    state_means = df.groupby(state_column)[activity_column].mean().to_dict()
    
    results = {
        'active_probability_cosinor': active_prob_cosinor,
        'per_state_cosinor': per_state_results,
        'state_mean_activity': state_means,
        'active_states': active_states,
    }
    
    logger.success("State-based cosinor analysis complete")
    return results


def fit_cosinor_per_individual_per_state(
    df: pd.DataFrame,
    time_column: str,
    subject_column: str = 'subject',
    state_column: str = 'hmm_state',
    activity_column: str = 'activity',
    period: float = 24.0,
    active_states: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Fit cosinor per individual and per state for population analysis.
    
    This enables statistical testing of individual differences in circadian
    rhythms across behavioral states.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with timestamps, subjects, states, and activity
    time_column : str
        Timestamp column
    subject_column : str, default='subject'
        Subject identifier column
    state_column : str, default='hmm_state'
        State column
    activity_column : str, default='activity'
        Activity column
    period : float, default=24.0
        Period in hours
    active_states : list of int, optional
        Active states for probability calculation
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with cosinor parameters per individual per state:
        - subject, state, mesor, amplitude, acrophase_hours, r_squared
        
    Examples
    --------
    >>> cosinor_df = fit_cosinor_per_individual_per_state(df, 'timestamp')
    >>> # Analyze with GLMM or compare groups
    """
    logger.info("Fitting cosinor per individual per state")
    
    results = []
    
    for subject in df[subject_column].unique():
        subject_data = df[df[subject_column] == subject]
        
        # Fit per-state for this subject
        state_results = fit_cosinor_per_state(
            subject_data,
            time_column=time_column,
            state_column=state_column,
            activity_column=activity_column,
            period=period,
            active_states=active_states
        )
        
        # Extract per-state cosinor
        for state, cosinor_params in state_results['per_state_cosinor'].items():
            results.append({
                'subject': subject,
                'state': state,
                'mesor': cosinor_params.get('mesor', np.nan),
                'amplitude': cosinor_params.get('amplitude', np.nan),
                'acrophase_hours': cosinor_params.get('acrophase_hours', np.nan),
                'r_squared': cosinor_params.get('r_squared', np.nan),
            })
    
    df_results = pd.DataFrame(results)
    logger.success(f"Computed cosinor for {len(df_results)} subject-state combinations")
    
    return df_results

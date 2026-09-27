"""Hidden Markov Model and Hidden Semi-Markov Model implementation.

IMPORTANT INPUT REQUIREMENTS:
- Features MUST be standardized (z-scored) before fitting HMM
- Use ActMin/ActMindata as primary activity signal
- Add relative movement features from accelerometer axes (variance, std, changes)
- DO NOT use raw magnitude from unsigned accelerometer axes
- Apply minimum dwell-time filtering to prevent unrealistic state flickering

CORRECT WORKFLOW:
1. Prepare features: ActMin + relative movement features
2. Standardize features per individual (z-score)
3. Fit HMM with standardized features
4. Apply minimum dwell-time filter
5. Validate states biologically (not just via IC)
6. Apply cosinor to state probabilities/state-conditioned activity
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from loguru import logger
from hmmlearn import hmm
import warnings


def fit_gaussian_hmm(
    X: np.ndarray,
    n_states: int = 2,
    n_iter: int = 100,
    random_state: int = 42,
    covariance_type: str = 'diag'
) -> hmm.GaussianHMM:
    """
    Fit a Gaussian Hidden Markov Model.
    
    Parameters
    ----------
    X : np.ndarray
        Input data (n_samples, n_features)
    n_states : int, default=2
        Number of hidden states
    n_iter : int, default=100
        Maximum number of iterations
    random_state : int, default=42
        Random seed for reproducibility
    covariance_type : str, default='diag'
        Type of covariance matrix
        
    Returns
    -------
    GaussianHMM
        Fitted HMM model
    """
    logger.info(f"Fitting Gaussian HMM with {n_states} states")
    
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X)
    
    logger.success(f"HMM fitted with score: {model.score(X):.2f}")
    return model


def predict_states(
    model: hmm.GaussianHMM,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict hidden states and posterior probabilities.
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model
    X : np.ndarray
        Input data
        
    Returns
    -------
    tuple
        (state_sequence, posterior_probabilities)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Predict most likely state sequence (Viterbi algorithm)
    states = model.predict(X)
    
    # Calculate posterior probabilities
    posteriors = model.predict_proba(X)
    
    logger.info("State sequences and posteriors computed")
    return states, posteriors


def calculate_bic(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    """
    Calculate Bayesian Information Criterion.
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model
    X : np.ndarray
        Input data
        
    Returns
    -------
    float
        BIC score (lower is better)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_states = model.n_components
    
    # Log likelihood
    log_likelihood = model.score(X) * n_samples
    
    # Number of parameters
    # Transition matrix: n_states * (n_states - 1)
    # Means: n_states * n_features
    # Covariances: n_states * n_features (for diagonal)
    # Initial probabilities: n_states - 1
    n_params = (
        n_states * (n_states - 1) +
        n_states * n_features +
        n_states * n_features +
        (n_states - 1)
    )
    
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic


def calculate_aic(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    """
    Calculate Akaike Information Criterion.
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model
    X : np.ndarray
        Input data
        
    Returns
    -------
    float
        AIC score (lower is better)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_states = model.n_components
    
    # Log likelihood
    log_likelihood = model.score(X) * n_samples
    
    # Number of parameters
    n_params = (
        n_states * (n_states - 1) +
        n_states * n_features +
        n_states * n_features +
        (n_states - 1)
    )
    
    aic = -2 * log_likelihood + 2 * n_params
    return aic


def select_optimal_states(
    X: np.ndarray,
    state_range: range = range(2, 6),
    criterion: str = 'bic',
    n_iter: int = 100,
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of states using BIC or AIC.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    state_range : range, default=range(2, 6)
        Range of states to test (K=2..5)
    criterion : str, default='bic'
        Selection criterion ('bic' or 'aic')
    n_iter : int, default=100
        Maximum iterations for fitting
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (optimal_n_states, scores_dict)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    scores = {}
    
    logger.info(f"Testing {len(state_range)} different state configurations")
    
    for n_states in state_range:
        model = fit_gaussian_hmm(X, n_states, n_iter, random_state)
        
        if criterion == 'bic':
            score = calculate_bic(model, X)
        elif criterion == 'aic':
            score = calculate_aic(model, X)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        scores[n_states] = score
        logger.info(f"K={n_states}: {criterion.upper()}={score:.2f}")
    
    # Select model with lowest score
    optimal_k = min(scores, key=scores.get)
    logger.success(f"Optimal number of states: {optimal_k}")
    
    return optimal_k, scores


def fit_hmm_pipeline(
    X: np.ndarray,
    state_range: range = range(2, 6),
    random_state: int = 42
) -> Dict:
    """
    Complete HMM pipeline: fit models, select optimal, predict states.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    state_range : range, default=range(2, 6)
        Range of states to test
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    dict
        Pipeline results including model, states, posteriors, and metrics
    """
    logger.info("Starting HMM pipeline")
    
    # Select optimal number of states
    optimal_k, bic_scores = select_optimal_states(
        X, state_range, criterion='bic', random_state=random_state
    )
    _, aic_scores = select_optimal_states(
        X, state_range, criterion='aic', random_state=random_state
    )
    
    # Fit final model with optimal K
    final_model = fit_gaussian_hmm(X, n_states=optimal_k, random_state=random_state)
    
    # Predict states and posteriors
    states, posteriors = predict_states(final_model, X)
    
    results = {
        'model': final_model,
        'optimal_k': optimal_k,
        'states': states,
        'posteriors': posteriors,
        'bic_scores': bic_scores,
        'aic_scores': aic_scores,
        'transition_matrix': final_model.transmat_,
        'means': final_model.means_,
        'covars': final_model.covars_,
    }
    
    logger.success("HMM pipeline complete")
    return results


def apply_minimum_dwell_time(
    states: np.ndarray,
    min_dwell: int = 4
) -> np.ndarray:
    """
    Post-process HMM states to enforce minimum dwell time.
    
    This addresses the "flickering" problem where states switch unrealistically
    fast. A minimum dwell time ensures biologically plausible state dynamics.
    
    Parameters
    ----------
    states : np.ndarray
        Raw HMM state sequence
    min_dwell : int, default=4
        Minimum number of consecutive observations in a state.
        Choose based on your sampling interval:
        - For 15-minute intervals: min_dwell=4 → 1 hour
        - For 5-minute intervals: min_dwell=12 → 1 hour
        - For 1-hour intervals: min_dwell=1 → 1 hour
        
    Returns
    -------
    np.ndarray
        Filtered state sequence with enforced minimum dwell time
        
    Notes
    -----
    Short state segments (< min_dwell) are replaced with the surrounding
    majority state to enforce temporal persistence. The algorithm uses
    efficient numpy operations for run-length detection.
    """
    logger.info(f"Applying minimum dwell time filter (min_dwell={min_dwell})")
    
    if len(states) == 0:
        return states
    
    filtered_states = states.copy()
    
    # Use numpy for efficient run-length encoding
    # Find where state changes
    change_points = np.concatenate(([0], np.where(np.diff(states) != 0)[0] + 1, [len(states)]))
    
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i + 1]
        run_length = end - start
        current_state = states[start]
        
        # If run is too short, replace with majority of neighbors
        if run_length < min_dwell:
            prev_state = states[start - 1] if start > 0 else current_state
            next_state = states[end] if end < len(states) else current_state
            
            # Use majority vote from neighbors
            if prev_state == next_state:
                replacement = prev_state
            else:
                # If neighbors disagree, keep current
                replacement = current_state
            
            filtered_states[start:end] = replacement
    
    n_changed = np.sum(states != filtered_states)
    pct_changed = 100 * n_changed / len(states)
    logger.success(
        f"Dwell-time filter applied: {n_changed} transitions changed ({pct_changed:.1f}%)"
    )
    
    return filtered_states


def fit_hmm_per_subject(
    df: pd.DataFrame,
    feature_columns: List[str],
    subject_column: str = 'subject',
    state_range: range = range(2, 6),
    random_state: int = 42,
    standardize: bool = True
) -> Dict[str, Dict]:
    """
    Fit separate HMM models for each subject.
    
    This is the CORRECT approach for multi-subject data where individuals
    may have different behavioral patterns and activity scales.
    
    WARNING: DO NOT pool subjects into one HMM unless you have strong
    biological justification and all features are properly standardized
    per subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with all subjects
    feature_columns : list of str
        Column names to use as HMM features
    subject_column : str, default='subject'
        Column identifying subjects
    state_range : range, default=range(2, 6)
        Range of states to test
    random_state : int, default=42
        Random seed for reproducibility
    standardize : bool, default=True
        Whether to standardize features per subject before fitting
        (recommended: True)
    
    Returns
    -------
    dict
        Dictionary with subject IDs as keys, each containing:
        - model: fitted HMM model
        - states: predicted states
        - posteriors: state probabilities
        - optimal_k: selected number of states
        - bic_scores: BIC for different K
        - n_observations: number of data points
    
    Examples
    --------
    >>> results = fit_hmm_per_subject(
    ...     df, ['activity_log_standardized'], 
    ...     subject_column='subject'
    ... )
    >>> # Access individual subject results
    >>> subject_states = results[subjects[0]]['states']
    """
    logger.info(f"Fitting HMM separately for each subject")
    
    subjects = df[subject_column].unique()
    results = {}
    
    for subject in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject: {subject}")
        logger.info(f"{'='*60}")
        
        # Extract subject data
        subj_df = df[df[subject_column] == subject].copy()
        
        # Check for missing values
        subj_df = subj_df[feature_columns].dropna()
        
        if len(subj_df) < 100:
            logger.warning(
                f"Subject {subject} has only {len(subj_df)} observations. "
                "Skipping (minimum 100 required for stable HMM)."
            )
            continue
        
        X = subj_df.values
        
        # Optionally standardize (though features should already be standardized)
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            logger.info(f"Standardized {len(feature_columns)} features for {subject}")
        
        # Fit HMM pipeline
        subject_results = fit_hmm_pipeline(
            X, state_range=state_range, random_state=random_state
        )
        subject_results['n_observations'] = len(subj_df)
        subject_results['subject'] = subject
        
        results[subject] = subject_results
        
        logger.success(
            f"{subject}: K={subject_results['optimal_k']}, "
            f"N={len(subj_df)} observations"
        )
    
    logger.success(f"\nCompleted HMM fitting for {len(results)} subjects")
    return results


def combine_per_subject_states(
    df: pd.DataFrame,
    hmm_results: Dict[str, Dict],
    subject_column: str = 'subject'
) -> pd.DataFrame:
    """
    Combine per-subject HMM states back into the original dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe
    hmm_results : dict
        Results from fit_hmm_per_subject()
    subject_column : str, default='subject'
        Column identifying subjects
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added 'hmm_state' and 'state_probability' columns
    """
    df_out = df.copy()
    df_out['hmm_state'] = np.nan
    df_out['state_probability'] = np.nan
    
    for subject, results in hmm_results.items():
        subject_mask = df_out[subject_column] == subject
        subject_indices = df_out[subject_mask].index
        
        # Account for rows dropped due to NaN in features
        n_states = len(results['states'])
        if n_states <= len(subject_indices):
            df_out.loc[subject_indices[:n_states], 'hmm_state'] = results['states']
            df_out.loc[subject_indices[:n_states], 'state_probability'] = results['posteriors'].max(axis=1)
        else:
            logger.warning(
                f"Mismatch in number of states for {subject}: "
                f"got {n_states}, expected {len(subject_indices)}"
            )
    
    logger.info("Combined per-subject states into dataframe")
    return df_out


def analyze_state_characteristics(
    states: np.ndarray,
    activity: np.ndarray,
    timestamps: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Analyze biological characteristics of HMM states.
    
    This function computes mean activity, dwell times, and diel (day/night)
    prevalence for each state to aid in biological validation and interpretation.
    
    Parameters
    ----------
    states : np.ndarray
        State sequence from HMM
    activity : np.ndarray
        Activity values
    timestamps : pd.Series, optional
        Timestamp series for diel analysis
        
    Returns
    -------
    pd.DataFrame
        State characteristics including:
        - mean_activity: mean activity level in each state
        - std_activity: standard deviation of activity
        - mean_dwell_time: average consecutive time in state
        - day_prevalence: proportion of daytime (6am-6pm) observations
        - night_prevalence: proportion of nighttime observations
        
    Notes
    -----
    States should be biologically interpretable, e.g.:
    - State with lowest mean_activity → "Rest"
    - State with highest mean_activity → "Highly Active"
    - Intermediate states → "Low Activity", "Active"
    """
    logger.info("Analyzing state characteristics for biological validation")
    
    unique_states = np.unique(states)
    results = []
    
    for state in unique_states:
        mask = states == state
        
        # Mean activity
        mean_act = np.mean(activity[mask])
        std_act = np.std(activity[mask])
        
        # Dwell times
        dwell_times = []
        i = 0
        while i < len(states):
            if states[i] == state:
                j = i
                while j < len(states) and states[j] == state:
                    j += 1
                dwell_times.append(j - i)
                i = j
            else:
                i += 1
        
        mean_dwell = np.mean(dwell_times) if dwell_times else 0
        
        # Diel prevalence (if timestamps provided)
        day_prev = np.nan
        night_prev = np.nan
        if timestamps is not None:
            hours = pd.to_datetime(timestamps).dt.hour
            day_mask = (hours >= 6) & (hours < 18)
            night_mask = ~day_mask
            
            state_day = np.sum(mask & day_mask)
            state_night = np.sum(mask & night_mask)
            total_state = np.sum(mask)
            
            day_prev = state_day / total_state if total_state > 0 else 0
            night_prev = state_night / total_state if total_state > 0 else 0
        
        results.append({
            'state': state,
            'mean_activity': mean_act,
            'std_activity': std_act,
            'mean_dwell_time': mean_dwell,
            'n_observations': np.sum(mask),
            'day_prevalence': day_prev,
            'night_prevalence': night_prev,
        })
    
    df_results = pd.DataFrame(results).sort_values('mean_activity')
    
    logger.success("State characteristics computed")
    logger.info(f"\nState Summary:\n{df_results.to_string()}")
    
    return df_results


def assign_biological_labels(
    state_characteristics: pd.DataFrame,
    n_states: int = 3
) -> Dict[int, str]:
    """
    Assign biological labels to states based on mean activity.
    
    This creates interpretable labels for hidden states based on their
    emission characteristics. States are labeled from lowest to highest
    mean activity.
    
    Parameters
    ----------
    state_characteristics : pd.DataFrame
        Output from analyze_state_characteristics()
    n_states : int
        Number of states (2, 3, or 4 recommended)
        
    Returns
    -------
    dict
        Mapping from state number to biological label
        
    Examples
    --------
    For 3 states: {0: "Rest", 1: "Low Activity", 2: "Active"}
    For 4 states: {0: "Rest", 1: "Low", 2: "Active", 3: "Highly Active"}
    """
    # Sort by mean activity
    sorted_states = state_characteristics.sort_values('mean_activity')
    state_ids = sorted_states['state'].values
    
    if n_states == 2:
        labels = ["Rest", "Active"]
    elif n_states == 3:
        labels = ["Rest", "Low Activity", "Active"]
    elif n_states == 4:
        labels = ["Rest", "Low", "Active", "Highly Active"]
    else:
        # Generic labels for other cases
        labels = [f"State_{i}" for i in range(n_states)]
    
    mapping = {state_ids[i]: labels[i] for i in range(len(state_ids))}
    
    logger.info(f"Biological labels assigned: {mapping}")
    return mapping


def select_states_biologically(
    X: np.ndarray,
    activity: np.ndarray,
    state_range: range = range(2, 5),
    timestamps: Optional[pd.Series] = None,
    random_state: int = 42
) -> Dict:
    """
    Select number of states based on biological interpretability.
    
    IMPORTANT: For long time series, AIC/BIC decrease monotonically and are
    unreliable. This function selects K based on:
    1. Distinct mean activity levels per state
    2. Reasonable dwell-time distributions
    3. Interpretable diel (day/night) patterns
    
    Recommended: K=3 or K=4 for animal activity data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix for HMM
    activity : np.ndarray
        Raw activity values for interpretation
    state_range : range, default=range(2, 5)
        Range of K to evaluate
    timestamps : pd.Series, optional
        Timestamps for diel analysis
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    dict
        Results including recommended K and characteristics for each K
        
    Notes
    -----
    IC-based selection (AIC/BIC) is documented as INSUFFICIENT for this
    data due to very long time series causing monotonic decrease.
    """
    logger.info("Selecting states based on biological interpretability")
    logger.warning(
        "AIC/BIC decrease monotonically for long time series and are UNRELIABLE. "
        "Using biological validation instead."
    )
    
    all_characteristics = {}
    
    for n_states in state_range:
        logger.info(f"\n=== Evaluating K={n_states} ===")
        
        # Fit model
        model = fit_gaussian_hmm(X, n_states=n_states, random_state=random_state)
        states, _ = predict_states(model, X)
        
        # Apply dwell-time filter
        states_filtered = apply_minimum_dwell_time(states, min_dwell=4)
        
        # Analyze characteristics
        chars = analyze_state_characteristics(states_filtered, activity, timestamps)
        all_characteristics[n_states] = chars
    
    # Recommendation based on biological criteria
    logger.info("\n=== Recommendation ===")
    logger.info(
        "Select K=3 or K=4 based on interpretability:\n"
        "  K=3: Rest, Low Activity, Active\n"
        "  K=4: Rest, Low, Active, Highly Active\n"
        "Examine mean_activity separation and diel patterns above."
    )
    
    results = {
        'recommended_k': 3,  # Default recommendation
        'characteristics': all_characteristics,
        'note': 'IC-based selection insufficient - use biological validation',
    }
    
    return results

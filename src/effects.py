"""Mixed effects models and machine learning implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from loguru import logger
from scipy import stats


def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_column : str
        Target variable column
    feature_columns : list of str
        Feature columns
    test_size : float, default=0.2
        Proportion for test set
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Remove rows with missing values
    df_clean = df[feature_columns + [target_column]].dropna()
    
    X = df_clean[feature_columns].values
    y = df_clean[target_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(
        f"Data prepared: {len(X_train)} train samples, "
        f"{len(X_test)} test samples, {len(feature_columns)} features"
    )
    
    return X_train, X_test, y_train, y_test


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = 'regression',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **kwargs
) -> Dict:
    """
    Fit Random Forest model.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test targets
    task : str, default='regression'
        'regression' or 'classification'
    n_estimators : int, default=100
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    random_state : int, default=42
        Random seed
    **kwargs
        Additional parameters for RandomForest
        
    Returns
    -------
    dict
        Model and results
    """
    logger.info(f"Fitting Random Forest ({task})")
    
    if task == 'regression':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    results = {
        'model': model,
        'train_score': train_score,
        'test_score': test_score,
        'feature_importances': model.feature_importances_,
        'n_estimators': n_estimators,
        'random_state': random_state,
    }
    
    logger.success(
        f"Random Forest fitted: train_score={train_score:.4f}, "
        f"test_score={test_score:.4f}"
    )
    
    return results


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = 'regression',
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Dict:
    """
    Fit XGBoost model.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test targets
    task : str, default='regression'
        'regression' or 'classification'
    n_estimators : int, default=100
        Number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.1
        Learning rate
    random_state : int, default=42
        Random seed
    **kwargs
        Additional parameters for XGBoost
        
    Returns
    -------
    dict
        Model and results
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install the package with the 'ml' extra to use fit_xgboost."
        ) from exc

    logger.info(f"Fitting XGBoost ({task})")
    
    if task == 'regression':
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    results = {
        'model': model,
        'train_score': train_score,
        'test_score': test_score,
        'feature_importances': model.feature_importances_,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'random_state': random_state,
    }
    
    logger.success(
        f"XGBoost fitted: train_score={train_score:.4f}, "
        f"test_score={test_score:.4f}"
    )
    
    return results


def fit_glmm(
    df: pd.DataFrame,
    formula: str,
    random_effects: str,
    family: str = 'gaussian'
) -> Dict:
    """
    Fit Generalized Linear Mixed Model using pymer4.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    formula : str
        R-style formula for fixed effects
    random_effects : str
        Random effects specification
    family : str, default='gaussian'
        Distribution family
        
    Returns
    -------
    dict
        Model results
    """
    try:
        from pymer4.models import Lmer
        
        logger.info("Fitting GLMM")
        
        # Combine formula with random effects
        full_formula = f"{formula} + {random_effects}"
        
        model = Lmer(full_formula, data=df, family=family)
        fitted = model.fit()
        
        results = {
            'model': model,
            'fitted': fitted,
            'formula': full_formula,
            'aic': fitted['AIC'][0] if 'AIC' in fitted else None,
            'bic': fitted['BIC'][0] if 'BIC' in fitted else None,
        }
        
        logger.success("GLMM fitted successfully")
        return results
        
    except ImportError:
        logger.warning("pymer4 not available, skipping GLMM")
        return {
            'error': 'pymer4 not installed',
            'formula': formula,
        }
    except Exception as e:
        logger.error(f"GLMM fitting failed: {e}")
        return {
            'error': str(e),
            'formula': formula,
        }


def fit_multi_component_cosinor_lmm(
    df: pd.DataFrame,
    response: str,
    group_column: str,
    subject_column: str = 'subject',
    hour_column: str = 'hour',
    n_components: int = 5,
    base_period: float = 24.0,
    covariates: Optional[List[str]] = None,
) -> Dict:
    """
    Fit a multi-component cosinor LMM analogous to:

        ACT ~ amp_acro(hour, period=periods[1:5], n_components=5,
              group=group) + (1 + amp_acro1 + ... + amp_acro5 | subject)

    Uses statsmodels MixedLM.  Fixed effects include sin/cos harmonics
    for each component, interacted with group, plus optional covariates.
    Random effects include random intercept and random slopes for
    the harmonic terms per subject.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per observation (e.g. hourly aggregated).
    response : str
        Name of the response column.
    group_column : str
        Categorical covariate to interact with harmonics (e.g. 'sex',
        'location', 'season', 'breeding').
    subject_column : str
        Grouping variable for random effects.
    hour_column : str
        Column with hour-of-day (numeric, 0-23).
    n_components : int
        Number of harmonic components (default 5 → 24h,12h,8h,6h,4.8h).
    base_period : float
        Fundamental period in hours.
    covariates : list of str, optional
        Additional fixed-effect covariates.

    Returns
    -------
    dict
        Model summary, coefficients, variance components, R² etc.
    """
    import statsmodels.formula.api as smf

    logger.info(
        f"Fitting {n_components}-component cosinor LMM: "
        f"{response} ~ harmonics * {group_column} + (harmonics | {subject_column})"
    )

    periods = [base_period / k for k in range(1, n_components + 1)]
    df = df.copy()

    # Create sin/cos harmonic columns
    harmonic_cols = []
    for j, T in enumerate(periods, start=1):
        sin_col = f'sin_{j}'
        cos_col = f'cos_{j}'
        df[sin_col] = np.sin(2 * np.pi * df[hour_column] / T)
        df[cos_col] = np.cos(2 * np.pi * df[hour_column] / T)
        harmonic_cols.extend([sin_col, cos_col])

    # Build fixed-effects formula
    # Harmonics interacted with group
    fixed_terms = []
    for col in harmonic_cols:
        fixed_terms.append(col)
        fixed_terms.append(f'{col}:{group_column}')

    if covariates:
        fixed_terms.extend(covariates)

    fixed_terms.append(group_column)
    formula = f'{response} ~ ' + ' + '.join(fixed_terms)

    # Random effects: random intercept + random slopes for harmonics
    re_formula = '1 + ' + ' + '.join(harmonic_cols)

    try:
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df[subject_column],
            re_formula=re_formula,
        )
        result = model.fit(reml=True, method='lbfgs')

        # Extract key information
        fixed_effects = result.fe_params.to_dict()
        random_effects = {
            str(k): v for k, v in result.random_effects.items()
        }

        # Compute marginal and conditional R²  (Nakagawa & Schielzeth 2013)
        y = df[response].values
        y_pred_fixed = result.fittedvalues.values
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(result.resid.values ** 2)

        # Variance components
        var_fixed = np.var(y_pred_fixed - np.mean(y_pred_fixed))
        var_resid = result.scale
        # Random effect variance (sum of diagonal of random effects cov)
        re_cov = result.cov_re
        var_random = np.trace(re_cov) if re_cov is not None else 0.0

        r2_marginal = var_fixed / (var_fixed + var_random + var_resid)
        r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

        # ICC
        icc = var_random / (var_random + var_resid) if (var_random + var_resid) > 0 else 0.0

        model_results = {
            'formula': formula,
            're_formula': re_formula,
            'n_components': n_components,
            'periods': periods,
            'group_column': group_column,
            'n_obs': int(result.nobs),
            'n_groups': len(result.random_effects),
            'fixed_effects': fixed_effects,
            'random_effects_summary': random_effects,
            'aic': result.aic,
            'bic': result.bic,
            'log_likelihood': result.llf,
            'r2_marginal': r2_marginal,
            'r2_conditional': r2_conditional,
            'icc': icc,
            'var_fixed': var_fixed,
            'var_random': var_random,
            'var_resid': var_resid,
            'converged': result.converged,
            'summary': str(result.summary()),
            'model': result,
        }

        logger.success(
            f"Multi-component cosinor LMM fitted: "
            f"R²m={r2_marginal:.3f}, R²c={r2_conditional:.3f}, "
            f"ICC={icc:.3f}, AIC={result.aic:.1f}"
        )
        return model_results

    except Exception as e:
        logger.error(f"Multi-component cosinor LMM failed: {e}")
        return {
            'error': str(e),
            'formula': formula,
            'n_components': n_components,
        }


def test_bout_duration_normality(
    bout_durations: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Test bout durations for normality using Shapiro-Wilk and
    Kolmogorov-Smirnov tests, plus skewness/kurtosis descriptives.

    Parameters
    ----------
    bout_durations : np.ndarray
        Array of bout durations (in hours or time steps).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test statistics and interpretation.
    """
    n = len(bout_durations)
    result = {
        'n': n,
        'mean': float(np.mean(bout_durations)),
        'std': float(np.std(bout_durations, ddof=1)),
        'median': float(np.median(bout_durations)),
        'skewness': float(stats.skew(bout_durations)),
        'kurtosis': float(stats.kurtosis(bout_durations)),
    }

    # Shapiro-Wilk (recommended for n < 5000)
    if 3 <= n <= 5000:
        sw_stat, sw_p = stats.shapiro(bout_durations)
        result['shapiro_wilk_stat'] = float(sw_stat)
        result['shapiro_wilk_p'] = float(sw_p)
        result['shapiro_wilk_normal'] = sw_p > alpha
    else:
        result['shapiro_wilk_stat'] = np.nan
        result['shapiro_wilk_p'] = np.nan
        result['shapiro_wilk_normal'] = None

    # Kolmogorov-Smirnov (for any sample size)
    if n >= 3:
        mean = result['mean']
        std = result['std']
        if std > 0:
            standardized = (bout_durations - mean) / std
            ks_stat, ks_p = stats.kstest(standardized, stats.norm.cdf)
            result['ks_stat'] = float(ks_stat)
            result['ks_p'] = float(ks_p)
            result['ks_normal'] = ks_p > alpha
        else:
            result['ks_stat'] = np.nan
            result['ks_p'] = np.nan
            result['ks_normal'] = None

    logger.info(
        f"Normality tests (n={n}): "
        f"Shapiro-Wilk p={result.get('shapiro_wilk_p', 'N/A')}, "
        f"K-S p={result.get('ks_p', 'N/A')}, "
        f"skew={result['skewness']:.2f}, kurtosis={result['kurtosis']:.2f}"
    )
    return result


test_bout_duration_normality.__test__ = False


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = 'regression',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple machine learning models.
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test targets
    task : str, default='regression'
        Type of task
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    logger.info("Comparing models")
    
    results = []
    
    # Random Forest
    rf_results = fit_random_forest(
        X_train, y_train, X_test, y_test,
        task=task, random_state=random_state
    )
    results.append({
        'model': 'RandomForest',
        'train_score': rf_results['train_score'],
        'test_score': rf_results['test_score'],
    })
    
    # XGBoost
    xgb_results = fit_xgboost(
        X_train, y_train, X_test, y_test,
        task=task, random_state=random_state
    )
    results.append({
        'model': 'XGBoost',
        'train_score': xgb_results['train_score'],
        'test_score': xgb_results['test_score'],
    })
    
    comparison_df = pd.DataFrame(results)
    logger.success("Model comparison complete")
    
    return comparison_df

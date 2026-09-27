"""Evaluation utilities for model assessment and visualization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from loguru import logger


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    if not np.any(y_true == 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape
    
    logger.info(
        f"Regression metrics - RMSE: {metrics['rmse']:.4f}, "
        f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}"
    )
    
    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict:
    """
    Evaluate classification model performance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str, default='weighted'
        Averaging method for multi-class metrics
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    logger.info(
        f"Classification metrics - Accuracy: {metrics['accuracy']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )
    
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs actual values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot residuals for regression model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    save_path : str, optional
        Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    # Residuals histogram
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list, optional
        Class labels
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance.
    
    Parameters
    ----------
    feature_names : list of str
        Feature names
    importances : np.ndarray
        Feature importance values
    top_n : int, default=20
        Number of top features to display
    save_path : str, optional
        Path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'r2'
) -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model
        Sklearn-compatible model
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    cv : int, default=5
        Number of folds
    scoring : str, default='r2'
        Scoring metric
        
    Returns
    -------
    dict
        Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores,
    }
    
    logger.info(
        f"Cross-validation ({cv}-fold): "
        f"mean={results['mean_score']:.4f}, std={results['std_score']:.4f}"
    )
    
    return results

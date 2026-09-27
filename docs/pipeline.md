# Pipeline Documentation

## Overview

This document describes the reproducible data analysis pipeline for the master's thesis project.

## Pipeline Workflow

The pipeline consists of 6 sequential stages, each implemented in a separate Jupyter notebook:

### Stage 0: Data Intake (00_intake.ipynb)

**Purpose**: Load and prepare raw data for analysis

**Inputs**:
- Raw data files (CSV, Excel, Parquet, JSON, Feather)

**Outputs**:
- `data/processed/intake_data.parquet`

**Key Functions**:
- `io.load_raw_data()`: Load data from various formats
- `io.save_processed_data()`: Save data in efficient format

**Parameters Logged**:
- `random_state`: Random seed (default: 42)
- `n_samples`: Number of samples loaded
- `n_features`: Number of features

---

### Stage 1: Quality Control & EDA (01_qc_eda.ipynb)

**Purpose**: Detect and fix data quality issues, perform exploratory analysis

**Inputs**:
- `data/processed/intake_data.parquet`

**Outputs**:
- `data/processed/qc_data.parquet`
- `outputs/timeseries_plot.png`
- `outputs/distribution_plot.png`

**Quality Checks**:
1. Missing value detection
2. Duplicate detection and removal
3. Time gap analysis
4. Data range validation

**Key Functions**:
- `qc.detect_missing_values()`: Find missing data
- `qc.detect_duplicates()`: Find duplicate rows
- `qc.detect_time_gaps()`: Find gaps in time series
- `qc.qc_report()`: Generate comprehensive QC report

**Parameters Logged**:
- `random_state`: 42
- `initial_samples`: Sample count before cleaning
- `final_samples`: Sample count after cleaning
- `duplicates_removed`: Number of duplicates removed

---

### Stage 2: Feature Engineering (02_features.ipynb)

**Purpose**: Create activity metrics and derived features

**Inputs**:
- `data/processed/qc_data.parquet`

**Outputs**:
- `data/processed/feature_data.parquet`
- `outputs/rolling_features.png`

**Features Created**:

1. **Activity Metrics**:
   - Magnitude, absolute value, or squared metrics
   - `features.calculate_activity_metric()`

2. **Rolling Statistics** (windows: 6, 12, 24):
   - Mean, std, min, max
   - `features.add_rolling_statistics()`

3. **Time Features**:
   - Hour of day, day of week, weekend indicator
   - `features.add_time_features()`

4. **Lag Features**:
   - Lags: 1, 6, 24 periods
   - `features.add_lag_features()`

5. **Difference Features**:
   - First and seasonal differences
   - `features.add_difference_features()`

**Key Functions**:
- `features.create_feature_set()`: Create comprehensive feature set

**Parameters Logged**:
- `random_state`: 42
- `rolling_windows`: [6, 12, 24]
- `lag_periods`: [1, 6, 24]

---

### Stage 3: HMM/HSMM Analysis (03_hmm_hsmm.ipynb)

**Purpose**: Fit Hidden Markov Models to detect hidden states

**Inputs**:
- `data/processed/feature_data.parquet`

**Outputs**:
- `data/processed/hmm_data.parquet`
- `outputs/hmm_model_selection.png`
- `outputs/hmm_states.png`
- `outputs/hmm_transition_matrix.png`

**Analysis Steps**:

1. **Model Selection**:
   - Test K=2, 3, 4, 5 states
   - Evaluate using BIC and AIC
   - Select optimal K (lowest BIC/AIC)

2. **State Prediction**:
   - Viterbi algorithm for most likely state sequence
   - Posterior probabilities for each state

3. **Model Interpretation**:
   - Transition matrix analysis
   - State means and covariances
   - State duration statistics

**Key Functions**:
- `models_hmm.fit_gaussian_hmm()`: Fit Gaussian HMM
- `models_hmm.select_optimal_states()`: Model selection
- `models_hmm.predict_states()`: State sequence prediction
- `models_hmm.fit_hmm_pipeline()`: Complete pipeline

**Parameters Logged**:
- `random_state`: 42
- `state_range`: '2-5'
- `optimal_k`: Selected number of states
- `n_samples`: Number of observations

---

### Stage 4: Cosinor Analysis (04_cosinor.ipynb)

**Purpose**: Detect and quantify circadian rhythms

**Inputs**:
- `data/processed/hmm_data.parquet`

**Outputs**:
- `data/processed/cosinor_data.parquet`
- `outputs/cosinor_fit.png`

**Analysis Steps**:

1. **Cosinor Fitting**:
   - Fit sinusoidal model: y = MESOR + Amplitude × cos(ωt - φ)
   - Default period: 24 hours
   - Non-linear least squares optimization

2. **Parameters Extracted**:
   - **MESOR**: Midline Estimating Statistic Of Rhythm (mean level)
   - **Amplitude**: Half the difference between peak and trough
   - **Acrophase**: Time of peak (in hours)
   - **R²**: Goodness of fit

3. **Multi-Period Analysis**:
   - Test multiple periods (24h, 12h)
   - Compare fit quality

4. **Significance Testing**:
   - Permutation test for rhythmicity
   - p-value calculation

**Key Functions**:
- `cosinor.fit_cosinor()`: Fit cosinor model
- `cosinor.extract_cosinor_features()`: Extract parameters from dataframe
- `cosinor.fit_multi_period_cosinor()`: Test multiple periods
- `cosinor.test_rhythmicity()`: Statistical significance testing

**Parameters Logged**:
- `random_state`: 42
- `period`: 24.0 hours
- `mesor`: Mean level
- `amplitude`: Rhythm amplitude
- `acrophase_hours`: Peak time
- `r_squared`: Fit quality

---

### Stage 5: GLMM & Machine Learning (05_glmm_ml.ipynb)

**Purpose**: Build predictive models using mixed effects and ML

**Inputs**:
- `data/processed/cosinor_data.parquet`

**Outputs**:
- `outputs/model_results.csv`
- `outputs/rf_feature_importance.png`
- `outputs/xgb_predictions.png`
- `outputs/xgb_residuals.png`

**Models Implemented**:

1. **Random Forest**:
   - n_estimators: 100
   - max_depth: 10
   - Task: regression or classification

2. **XGBoost**:
   - n_estimators: 100
   - max_depth: 6
   - learning_rate: 0.1

3. **GLMM (optional)**:
   - Generalized Linear Mixed Models
   - Fixed and random effects
   - Requires pymer4 library

**Evaluation Metrics**:

- **Regression**: RMSE, MAE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1

**Key Functions**:
- `effects.prepare_ml_data()`: Prepare train/test split
- `effects.fit_random_forest()`: Train Random Forest
- `effects.fit_xgboost()`: Train XGBoost
- `effects.fit_glmm()`: Fit mixed effects model
- `effects.compare_models()`: Compare multiple models
- `eval.evaluate_regression()`: Calculate regression metrics
- `eval.plot_predictions()`: Visualize predictions
- `eval.plot_feature_importance()`: Show important features

**Parameters Logged**:
- `random_state`: 42
- `n_features`: Number of input features
- `test_size`: 0.2 (20% held out)
- `rf_n_estimators`: 100
- `rf_max_depth`: 10
- `xgb_n_estimators`: 100
- `xgb_learning_rate`: 0.1
- `xgb_max_depth`: 6

---

## Reproducibility Guidelines

### Random Seeds

All random processes use a fixed seed (default: 42):
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

### Parameter Logging

Every notebook logs:
1. All hyperparameters
2. Random seeds
3. Data dimensions
4. Model performance metrics

### Data Versioning

- Raw data goes in `data/raw/`
- Processed data saved at each stage in `data/processed/`
- All intermediate outputs preserved for debugging

### Environment

- Python 3.8+
- Dependencies pinned in `requirements.txt`
- Virtual environment recommended

---

## Running the Full Pipeline

### Sequential Execution

```bash
cd notebooks
jupyter notebook
```

Execute in order:
1. 00_intake.ipynb
2. 01_qc_eda.ipynb
3. 02_features.ipynb
4. 03_hmm_hsmm.ipynb
5. 04_cosinor.ipynb
6. 05_glmm_ml.ipynb

### Automated Execution (optional)

Convert notebooks to scripts:
```bash
jupyter nbconvert --to script *.ipynb
python 00_intake.py
python 01_qc_eda.py
# etc.
```

---

## Troubleshooting

### Missing Data Files

Ensure data files are in `data/raw/` before running intake notebook.

### Import Errors

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Memory Issues

- Process data in chunks
- Use sampling in `io.load_sample_data()`
- Reduce feature window sizes

### GLMM Errors

If pymer4 fails to install or R is not available:
- GLMM section will be skipped
- RF and XGBoost will still run

---

## Output Files

| Stage | File | Description |
|-------|------|-------------|
| 0 | `intake_data.parquet` | Initial loaded data |
| 1 | `qc_data.parquet` | Cleaned data |
| 1 | `timeseries_plot.png` | Time series visualization |
| 1 | `distribution_plot.png` | Value distribution |
| 2 | `feature_data.parquet` | Data with engineered features |
| 2 | `rolling_features.png` | Rolling statistics plot |
| 3 | `hmm_data.parquet` | Data with HMM states |
| 3 | `hmm_model_selection.png` | BIC/AIC comparison |
| 3 | `hmm_states.png` | State sequences |
| 3 | `hmm_transition_matrix.png` | Transition heatmap |
| 4 | `cosinor_data.parquet` | Data with cosinor parameters |
| 4 | `cosinor_fit.png` | Cosinor curve fit |
| 5 | `model_results.csv` | Model performance summary |
| 5 | `rf_feature_importance.png` | RF feature importance |
| 5 | `xgb_predictions.png` | Predictions vs actual |
| 5 | `xgb_residuals.png` | Residual analysis |

---

## Best Practices

1. **Always use the same random seed** for reproducibility
2. **Log all parameters** in each notebook
3. **Save intermediate outputs** at each stage
4. **Document any manual interventions** or data modifications
5. **Version control** your notebooks and code
6. **Keep raw data unchanged** - only modify processed data
7. **Use meaningful variable names** and comments
8. **Test on sample data** before full dataset
9. **Monitor resource usage** (memory, CPU)
10. **Back up results** regularly

---

## References

- Hidden Markov Models: hmmlearn documentation
- Cosinor Analysis: Cornelissen, 2014
- Mixed Models: pymer4 documentation
- XGBoost: Chen & Guestrin, 2016
- Random Forest: Breiman, 2001

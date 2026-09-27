# Quick Start Guide: Corrected Analysis Pipeline

## Overview
This guide provides a quick reference for running the corrected analysis pipeline with biologically and statistically valid methods.

## Prerequisites
```bash
pip install -r requirements.txt
```

## Pipeline Execution Order

Run notebooks in this exact order:

1. `00_intake.ipynb` - Data loading
2. `01_qc_eda.ipynb` - Quality control
3. `02_features.ipynb` - Feature engineering with standardization ⚠️
4. `03_hmm_hsmm.ipynb` - HMM with biological validation ⚠️
5. `04_cosinor.ipynb` - State-based cosinor analysis ⚠️
6. `05_glmm_ml.ipynb` - Environmental effects

⚠️ = Contains critical methodological corrections

## Key Methodological Changes

### 1. Feature Engineering (Notebook 02)

**What Changed:**
- Features are now **standardized** (z-scored) before HMM
- Rolling features computed **per individual**
- Optional **log transformation** for activity data

**Key Functions:**
```python
# Standardize features for HMM input
df = features.standardize_features(df, ['activity_log'], group_by='subject')

# Log transform activity (optional)
df = features.log_transform_activity(df, 'activity', offset=1.0)

# Per-individual rolling features
df = features.add_rolling_statistics_per_individual(
    df, 'activity', subject_column='subject', windows=[6, 12, 24]
)
```

**Why It Matters:**
- Reduces HMM state "flickering"
- Ensures features are on comparable scales
- Prevents rolling windows from mixing individuals

---

### 2. HMM Analysis (Notebook 03)

**What Changed:**
- State selection based on **biological criteria**, NOT AIC/BIC
- **Minimum dwell-time filter** applied (≥4 samples = 1 hour)
- States **validated and labeled** (Rest, Low, Active, Highly Active)

**Key Functions:**
```python
# Biological state selection (NOT IC-based)
bio_results = models_hmm.select_states_biologically(
    X, activity, state_range=range(2, 5), timestamps=timestamps
)

# Apply dwell-time filter to prevent flickering
states_filtered = models_hmm.apply_minimum_dwell_time(states_raw, min_dwell=4)

# Analyze state characteristics
state_chars = models_hmm.analyze_state_characteristics(
    states_filtered, activity, timestamps
)

# Assign biological labels
bio_labels = models_hmm.assign_biological_labels(state_chars, n_states=3)
```

**How to Select K:**
1. Run biological validation for K=2,3,4
2. Examine mean activity separation between states
3. Check dwell-time distributions (should be ≥1 hour)
4. Look at diel (day/night) patterns
5. Select K=3 or K=4 based on interpretability

**Why AIC/BIC Don't Work Here:**
- Time series is too long (100,000+ observations)
- IC decrease monotonically → no stopping criterion
- Leads to over-parameterized models
- **Solution:** Use biological validation instead

---

### 3. Cosinor Analysis (Notebook 04)

**What Changed:**
- Cosinor applied **AFTER state identification** (correct order)
- Analysis of **state probability** and **state-conditioned activity**
- **Per-individual per-state** parameters for population analysis

**Key Functions:**
```python
# State-based cosinor (CORRECT approach)
state_cosinor = cosinor.fit_cosinor_per_state(
    df,
    time_column='timestamp',
    state_column='hmm_state',
    activity_column='activity',
    period=24.0
)

# Per-individual per-state cosinor for population analysis
individual_cosinor = cosinor.fit_cosinor_per_individual_per_state(
    df,
    time_column='timestamp',
    subject_column='subject',
    state_column='hmm_state',
    activity_column='activity'
)
```

**What You Get:**
1. **Active State Probability Cosinor**: Rhythms in *when* animal is active
2. **Per-State Cosinor**: Separate MESOR/amplitude/acrophase for each state
3. **Individual-Level Parameters**: Can be used in downstream GLMM/ML

**Why Raw Activity Cosinor Is Wrong:**
- Mixes rest, low activity, and high activity states
- Violates cosinor assumption of single dominant rhythm
- Results are biologically meaningless
- **Solution:** Apply cosinor to states, not raw data

---

## Quick Validation Checklist

After running the pipeline, verify:

### Feature Engineering
- [ ] Features have mean≈0, std≈1 after standardization
- [ ] Log-transformed activity has reasonable distribution
- [ ] Rolling features computed per individual (if multi-subject)

### HMM States
- [ ] States have distinct mean activity levels (not overlapping)
- [ ] Mean dwell time ≥4 samples (≥1 hour)
- [ ] States show interpretable day/night patterns
- [ ] Biological labels make sense (Rest < Low < Active)

### Cosinor
- [ ] R² values are reasonable (not ~0)
- [ ] Acrophase values align with expected activity peaks
- [ ] Per-state cosinor shows different rhythms per state
- [ ] Active state probability shows clear circadian pattern

---

## Common Issues and Solutions

### Issue: "State flickering" (rapid state switches)
**Solution:**
- Ensure features are standardized before HMM
- Apply dwell-time filter with min_dwell=4
- Consider log-transforming activity data

### Issue: Too many states (K>4)
**Solution:**
- Don't use min(AIC) or min(BIC) for selection
- Use biological validation instead
- Select K=3 or K=4 based on interpretability

### Issue: Cosinor R² is very low (~0)
**Solution:**
- Make sure you're using state-based cosinor, not raw activity
- Check that states show circadian patterns (visualize prevalence by hour)
- Verify data has sufficient temporal coverage (multiple days)

### Issue: States don't make biological sense
**Solution:**
- Check that features are standardized
- Verify dwell-time filter is applied
- Examine state characteristics (mean activity, diel patterns)
- Consider collapsing similar states

---

## Parameter Recommendations

### Feature Engineering
```python
# Log transform offset
offset = 1.0  # For activity data with zeros

# Rolling windows (in samples, not hours)
windows = [6, 12, 24]  # For 15-min sampling: 1.5h, 3h, 6h

# Standardization
group_by = 'subject'  # Per-individual standardization
```

### HMM
```python
# Number of states
K = 3  # Rest, Low Activity, Active
# or
K = 4  # Rest, Low, Active, Highly Active

# Dwell-time filter
min_dwell = 4  # 1 hour for 15-min intervals

# HMM iterations
n_iter = 100  # Usually sufficient
```

### Cosinor
```python
# Period
period = 24.0  # 24-hour circadian rhythm

# Active state threshold
# Auto-detected as states with above-median mean activity
# Can also specify manually: active_states = [2, 3]
```

---

## Expected Output Files

After running the pipeline:

1. `data/processed/qc_data.parquet` - Quality-controlled data
2. `data/processed/feature_data.parquet` - Standardized features
3. `data/processed/hmm_data.parquet` - HMM states (filtered)
4. `data/processed/cosinor_per_individual_state.parquet` - Cosinor params
5. `outputs/state_based_rhythms.png` - Visualization

---

## Typical Analysis Flow

```python
# 1. Load and QC data
df = io.load_accelerometer_data('raw_data.csv')
df = qc.detect_and_handle_duplicates(df)

# 2. Feature engineering with standardization
df['activity'] = features.calculate_accelerometer_activity(df)
df = features.log_transform_activity(df, 'activity')
df = features.standardize_features(df, ['activity_log'], group_by='subject')

# 3. HMM with biological validation
X = df['activity_log_standardized'].values.reshape(-1, 1)
bio_results = models_hmm.select_states_biologically(X, df['activity'].values)
# Examine results, select K=3

model = models_hmm.fit_gaussian_hmm(X, n_states=3)
states, _ = models_hmm.predict_states(model, X)
states_filtered = models_hmm.apply_minimum_dwell_time(states)

# 4. State-based cosinor
df['hmm_state'] = states_filtered
state_cosinor = cosinor.fit_cosinor_per_state(df, 'timestamp', 'hmm_state', 'activity')

# 5. Use results in downstream analysis
# MESOR, amplitude, acrophase are now available per state
```

---

## Additional Resources

- **Full Documentation**: `METHODOLOGY_CORRECTIONS.md`
- **Example Data**: `data/raw/sample_accelerometer_data.csv`
- **Outputs**: `outputs/` directory

---

## Questions?

For methodological questions, refer to:
1. `METHODOLOGY_CORRECTIONS.md` - Detailed explanations
2. Inline comments in notebooks
3. Function docstrings in source code

For implementation issues:
- Check notebook execution order
- Verify input data format
- Review parameter settings above

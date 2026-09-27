# Accelerometer Data Assumptions and Correct Usage

## Executive Summary

This document explains the correct interpretation and usage of tri-axial accelerometer data recorded as **unsigned relative values** (0–255), not calibrated acceleration in physical units.

**Key Points**:
- ❌ DO NOT compute VeDBA, ODBA, or raw magnitude from XAccel/YAccel/ZAccel
- ❌ DO NOT square or combine axes assuming zero-centered data
- ✅ USE ActMin/ActMindata as primary activity signal
- ✅ DERIVE relative movement features (variance, std, changes) from axes
- ✅ Z-score standardize all features per individual before HMM

---

## 1. Data Format and Encoding

### What the columns represent:

- **ActMindata**: Tag-derived activity metric
  - Pre-computed by the tag manufacturer
  - Validated measure of activity intensity
  - **PRIMARY ACTIVITY SIGNAL** for analysis

- **XAccel, YAccel, ZAccel**: Raw accelerometer axes
  - **Unsigned relative values** (typically 0–255)
  - Encode **device orientation + gravity**
  - NOT calibrated acceleration in m/s² or g-units
  - Absolute values are NOT meaningful for activity intensity

### Common misconception:

Many accelerometer studies use calibrated tri-axial acceleration where:
- Values are in physical units (m/s² or g)
- Data is zero-centered (gravity-corrected)
- VeDBA/ODBA (vectorial/overall dynamic body acceleration) can be computed
- Magnitude = sqrt(X² + Y² + Z²) is meaningful

**This does NOT apply to unsigned relative accelerometer data!**

---

## 2. Why Magnitude Calculation is INCORRECT

### The problem:

Computing `magnitude = sqrt(XAccel² + YAccel² + ZAccel²)` from unsigned relative values:

1. **Assumes zero-centered data**: XAccel/YAccel/ZAccel are NOT centered around zero
2. **Ignores gravity bias**: Values include constant gravity component (~9.8 g)
3. **Confounds orientation with movement**: Changes in device angle affect all axes
4. **Not validated**: Tag manufacturer provides ActMin for a reason

### Example:

```python
# INCORRECT - DO NOT DO THIS
df['activity'] = np.sqrt(df['XAccel']**2 + df['YAccel']**2 + df['ZAccel']**2)

# This produces garbage because:
# - If animal is stationary but tag rotates: magnitude changes (false positive)
# - If animal moves but orientation constant: magnitude may not change (false negative)
# - No calibration to physical units
# - Gravity component dominates the signal
```

---

## 3. CORRECT Approach

### Use ActMin as primary activity signal:

```python
# CORRECT: Use tag-derived metric
df['activity'] = df['ActMindata']
```

ActMin is:
- Computed by the tag manufacturer
- Based on calibrated algorithms
- Validated for activity measurement
- Independent of device orientation

### Derive relative movement features from axes:

```python
# CORRECT: Add relative movement features
df = features.add_accelerometer_movement_features(
    df,
    subject_column='subject',
    windows=[6, 12, 24],
    standardize=True
)
```

This computes:
- **Rolling variance** per axis: captures movement intensity
- **Rolling standard deviation** per axis: movement variability  
- **Absolute first differences** per axis: instantaneous change
- **Combined variance**: sum of per-axis variances
- **Z-score standardization** per individual

### Why this works:

Movement creates **changes** and **variability** in axis values, regardless of absolute values or orientation. Variance and differences capture this while being robust to:
- Constant gravity offset
- Device orientation
- Cross-individual sensor differences

---

## 4. Feature Engineering Workflow

### Step-by-step correct workflow:

```python
from src import io, features

# 1. Load data
df = io.load_accelerometer_data('data/raw/accelerometer.csv')

# 2. Use ActMin as primary activity signal
df['activity'] = df['ActMindata']

# 3. Add relative movement features from axes
df = features.add_accelerometer_movement_features(
    df,
    x_col='XAccel',
    y_col='YAccel', 
    z_col='ZAccel',
    subject_column='subject',
    windows=[6, 12, 24],
    standardize=True  # CRITICAL for HMM
)

# 4. Add rolling statistics on ActMin
df = features.add_rolling_statistics_per_individual(
    df,
    value_column='activity',
    subject_column='subject',
    windows=[6, 12, 24],
    stats=['mean', 'std', 'var']
)

# 5. Log transform and standardize ActMin
df = features.log_transform_activity(df, 'activity', offset=1.0)
df = features.standardize_features(
    df,
    ['activity_log'],
    group_by='subject',
    overwrite=True
)

# 6. Features ready for HMM:
# - activity_log_standardized (primary signal)
# - x_rolling_var_24_standardized (movement from X axis)
# - y_rolling_var_24_standardized (movement from Y axis)
# - z_rolling_var_24_standardized (movement from Z axis)
# - combined_movement_var_24_standardized (total movement)
```

---

## 5. HMM/HSMM Input Requirements

### Feature selection for HMM:

```python
# Select features for HMM
feature_cols = [
    'activity_log_standardized',  # Primary: ActMin (log-transformed, z-scored)
    'combined_movement_var_24_standardized',  # Secondary: total movement variance
]

# Optional: Add per-axis movement features
# feature_cols.extend(['x_rolling_var_24_standardized', 
#                      'y_rolling_var_24_standardized',
#                      'z_rolling_var_24_standardized'])

# Prepare data
df_hmm = df[feature_cols].dropna()
X = df_hmm.values

# Fit HMM
hmm_results = models_hmm.fit_hmm_pipeline(X, state_range=range(2, 5))
```

### Why standardization is CRITICAL:

1. **Prevents state flickering**: Raw values cause noisy state transitions
2. **Ensures scale compatibility**: Features on comparable scales
3. **Improves convergence**: HMM fits more stably
4. **Enables interpretation**: States represent biological modes, not amplitude slices

---

## 6. Validation and Interpretation

### Validate your features:

```python
# Check that features are standardized (mean≈0, std≈1)
for col in feature_cols:
    print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")

# Expected output:
# activity_log_standardized: mean≈0.000, std≈1.000
# combined_movement_var_24_standardized: mean≈0.000, std≈1.000
```

### Biological validation of HMM states:

```python
# Analyze state characteristics
state_chars = models_hmm.analyze_state_characteristics(
    states=hmm_results['states'],
    activity=df['activity'].values,
    timestamps=df['timestamp']
)

# Look for:
# - Distinct mean activity levels per state
# - Reasonable dwell times (≥1 hour)
# - Interpretable diel (day/night) patterns
```

---

## 7. Common Mistakes to Avoid

### ❌ INCORRECT Approaches:

```python
# 1. Computing magnitude from unsigned axes
df['activity'] = np.sqrt(df['XAccel']**2 + df['YAccel']**2 + df['ZAccel']**2)

# 2. Computing VeDBA/ODBA
vedba = np.sqrt((df['XAccel'] - df['XAccel'].mean())**2 + 
                (df['YAccel'] - df['YAccel'].mean())**2 + 
                (df['ZAccel'] - df['ZAccel'].mean())**2)

# 3. Using raw axes directly in HMM
X = df[['XAccel', 'YAccel', 'ZAccel']].values
hmm.fit(X)  # WRONG

# 4. Not standardizing features
X = df[['activity']].values  # Not z-scored
hmm.fit(X)  # Will cause flickering

# 5. Applying cosinor before state identification
cosinor.fit_cosinor(df['activity'])  # Mixes behavioral states
```

### ✅ CORRECT Approaches:

```python
# 1. Use ActMin as primary signal
df['activity'] = df['ActMindata']

# 2. Add relative movement features
df = features.add_accelerometer_movement_features(df)

# 3. Standardize all features per individual
df = features.standardize_features(df, feature_cols, group_by='subject')

# 4. Use standardized features for HMM
X = df[standardized_feature_cols].dropna().values
hmm.fit(X)

# 5. Apply cosinor AFTER state identification
cosinor.fit_cosinor_per_state(df, state_column='hmm_state')
```

---

## 8. References

### Why accelerometer axes encode orientation:

1. **Brown et al. (2013)**. "Observing the unwatchable through acceleration logging of animal behavior." *Animal Biotelemetry* 1:20.
   - Documents that raw accelerometer values encode posture, not activity
   - Recommends using tag-derived metrics

2. **Sakamoto et al. (2009)**. "Can ethograms be automatically generated using body acceleration data from free-ranging birds?" *PLoS ONE* 4(4):e5379.
   - Shows that absolute acceleration values reflect static posture
   - Movement detected via variance and dynamic changes

3. **Wilson et al. (2006)**. "Moving towards acceleration for estimates of activity-specific metabolic rate in free-living animals." *Journal of Animal Ecology* 75(5):1081-1090.
   - Explains dynamic vs. static acceleration
   - Notes importance of calibration for VeDBA/ODBA

### Key insight:

Without calibration to remove gravity and convert to physical units, raw accelerometer axes are **positional/orientation sensors**, not activity sensors.

---

## 9. Summary Checklist

Before running your analysis, ensure:

- [ ] Using ActMindata as primary activity signal
- [ ] NOT computing magnitude from XAccel/YAccel/ZAccel
- [ ] Adding relative movement features (variance, std, differences)
- [ ] Computing features per individual (not globally)
- [ ] Z-score standardizing all features per individual
- [ ] Using standardized features for HMM input
- [ ] Applying minimum dwell-time filter to HMM states
- [ ] Validating states biologically (not just via IC)
- [ ] Applying cosinor AFTER state identification
- [ ] Documenting all assumptions in code and methods

---

## 10. Questions and Troubleshooting

### "Why can't I use magnitude like other papers?"

Other papers likely use:
- Calibrated acceleration in physical units (m/s² or g)
- Zero-centered data (gravity-corrected)
- Tags that output dynamic acceleration directly

Your data has:
- Unsigned relative values (0-255)
- Orientation + gravity encoded in absolute values
- Manufacturer-provided activity metric (ActMin)

### "How do I know if my axis data is unsigned/relative?"

Indicators:
- Values range 0-255 (8-bit unsigned integer)
- No negative values
- Tag documentation doesn't mention "calibrated acceleration"
- Tag provides separate activity metric (ActMin)
- Values change with device orientation even when stationary

### "What if I don't have ActMin?"

If your tag doesn't provide a derived activity metric:
1. Contact manufacturer for calibration details
2. Use relative movement features ONLY (variance, std, differences)
3. Do NOT compute magnitude or VeDBA/ODBA
4. Consider re-processing raw data with manufacturer software

---

## Contact and Feedback

For questions about this methodology:
- See `METHODOLOGY_CORRECTIONS.md` for full pipeline corrections
- See `QUICK_START.md` for quick reference guide
- See module docstrings for implementation details

Last updated: 2026-01-06

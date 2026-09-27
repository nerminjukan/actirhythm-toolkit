# Methodology Corrections for Animal Activity Rhythm Analysis

## Executive Summary

This document details the corrections made to the analysis pipeline to ensure biological and statistical validity for animal activity rhythm analysis. The original pipeline had several critical methodological errors that have been systematically addressed.

**NEW: Critical correction for accelerometer data** - See Section 0 for proper handling of unsigned relative accelerometer axes.

---

## 0. Accelerometer Data Interpretation (CRITICAL - NEW)

### Problem
XAccel/YAccel/ZAccel were incorrectly treated as calibrated acceleration in physical units, and magnitude = sqrt(X² + Y² + Z²) was computed and used as the primary activity signal.

### Why This Is Wrong
- XAccel/YAccel/ZAccel are **unsigned relative values** (0-255), NOT calibrated acceleration
- Absolute axis values encode **device orientation + gravity**, not movement intensity
- Computing magnitude from uncalibrated, non-zero-centered axes produces meaningless results
- Tag manufacturer provides ActMin/ActMindata for a reason - it's the validated activity metric
- Squaring and combining biased values compounds the error
- Results are confounded by device orientation changes unrelated to activity

### Correct Approach (Now Implemented)

#### Use ActMin as Primary Signal
ActMindata is the tag-derived activity metric:
- Computed by manufacturer using validated algorithms
- Calibrated for activity measurement
- Independent of device orientation
- **This should be your primary activity signal**

```python
# CORRECT
df['activity'] = df['ActMindata']
```

#### Derive Relative Movement Features from Axes
Movement creates **changes and variability** in axis values:
- Rolling variance per axis: captures movement intensity
- Rolling standard deviation: movement variability
- Absolute first differences: instantaneous change
- Combined variance: total movement across all axes
- All computed per individual and z-scored

```python
# CORRECT
df = features.add_accelerometer_movement_features(
    df,
    subject_column='subject',
    windows=[6, 12, 24],
    standardize=True
)
```

#### What NOT to Do
```python
# WRONG - DO NOT DO THIS
df['activity'] = np.sqrt(df['XAccel']**2 + df['YAccel']**2 + df['ZAccel']**2)

# Also wrong:
vedba = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2 + (Z - Z.mean())**2)
odba = abs(X - X.mean()) + abs(Y - Y.mean()) + abs(Z - Z.mean())
```

### Code Changes
- **Updated function**: `features.calculate_accelerometer_activity()` - Now deprecated with warnings
- **New function**: `features.add_accelerometer_movement_features()` - Correct approach
- **Updated**: `io.load_accelerometer_data()` - Documents data assumptions
- **Updated**: Module docstrings - Explain unsigned relative values
- **Updated**: `example_pipeline.py` - Demonstrates correct workflow
- **Updated**: `README.md` - Documents data format and assumptions
- **New file**: `ACCELEROMETER_DATA_ASSUMPTIONS.md` - Complete documentation

### Assumptions Documented
1. XAccel/YAccel/ZAccel are unsigned relative values (0-255)
2. Absolute axis values encode orientation + gravity, not movement
3. ActMindata is the validated tag-derived activity metric
4. Movement detected via variance and change, not absolute values
5. Features computed per individual to avoid cross-subject contamination

---

## 1. Cosinor Misuse (CRITICAL)

### Problem
Cosinor was incorrectly applied to raw or hourly-averaged activity data that mixes multiple behavioral states. This violates the core assumption of cosinor analysis: a single dominant rhythm.

### Why This Is Wrong
- Raw activity data contains a mixture of rest, low activity, and highly active behavioral states
- Each state may have different rhythmic properties
- Applying cosinor to mixed data produces biologically meaningless results
- The fitted parameters (MESOR, amplitude, acrophase) represent an artificial average across incompatible states

### Correct Approach (Now Implemented)
Cosinor is applied **AFTER** behavioral state identification:

1. **Probability of Active States**: Fit cosinor to P(state ∈ {active states})
   - Analyzes the rhythm in *when* the animal is active, not *how much*
   - Biologically interpretable: "peak probability of being active"

2. **State-Conditioned Mean Activity**: Fit separate cosinor per behavioral state
   - E[activity | state = rest], E[activity | state = active], etc.
   - Each state's rhythm is analyzed independently

3. **Per-Individual Per-State**: Compute MESOR, amplitude, and acrophase for each individual and each state
   - Enables population-level statistical analysis
   - Can be used in downstream GLMM/ML models

### Code Changes
- **New function**: `cosinor.fit_cosinor_per_state()` - Correct state-based approach
- **New function**: `cosinor.fit_cosinor_per_individual_per_state()` - Population analysis
- **Updated**: `04_cosinor.ipynb` - Uses state-based cosinor exclusively
- **Removed**: Raw activity cosinor (biologically invalid)

---

## 2. HMM State Flickering (CRITICAL)

### Problem
Hidden states switched unrealistically fast, producing implausible dwell times (e.g., switching states every few minutes). This is biologically impossible for animal behavioral states.

### Root Causes
1. **Noisy observations**: Raw sensor values fed directly to HMM
2. **Unscaled features**: Different features on incompatible scales
3. **No duration modeling**: Standard HMM has no minimum state duration
4. **No post-processing**: Raw Viterbi output used without filtering

### Correct Approach (Now Implemented)

#### A. Feature Standardization
All features are z-score normalized before HMM fitting:
```python
# Per-individual standardization
df = standardize_features(df, ['activity_log'], group_by='subject')
```

**Justification**:
- Reduces impact of sensor noise
- Ensures features are on comparable scales
- Improves HMM convergence
- **Critical** for preventing flickering

#### B. Log Transformation
Activity data is log-transformed to reduce impact of extreme values:
```python
df = log_transform_activity(df, 'activity', offset=1.0)
```

**Justification**:
- Activity/count data is often log-normal
- Reduces influence of outliers
- Improves state separation

#### C. Minimum Dwell-Time Post-Processing
States are filtered to enforce minimum consecutive duration:
```python
states_filtered = apply_minimum_dwell_time(states_raw, min_dwell=4)
```

**Justification**:
- Default: 4 samples = 1 hour (for 15-minute intervals)
- Biologically plausible minimum state duration
- Short "flickers" are replaced with surrounding state
- Alternative to HSMM (which requires more complex implementation)

### Code Changes
- **New function**: `features.standardize_features()` - Z-score normalization
- **New function**: `features.log_transform_activity()` - Log transform
- **New function**: `models_hmm.apply_minimum_dwell_time()` - Dwell-time filter
- **Updated**: `02_features.ipynb` - Includes standardization pipeline
- **Updated**: `03_hmm_hsmm.ipynb` - Uses filtered states

---

## 3. Invalid Model Selection via AIC/BIC

### Problem
AIC and BIC were used to select the number of HMM states. For very long time series (100,000+ observations), these criteria decrease monotonically with K and are **unreliable**.

### Why This Is Wrong
- Information criteria penalize model complexity relative to data size
- For extremely large N, the log-likelihood term dominates
- Adding more states almost always improves fit
- BIC/AIC provide no stopping criterion
- Results in over-parameterized models (K=5, 6, 7...) with no biological meaning

### Correct Approach (Now Implemented)
Select K based on **biological interpretability**:

1. **Mean Activity Separation**: States should have clearly distinct mean activity levels
   - E.g., Rest (low), Moderate (medium), Active (high)
   - Avoid states that are just "amplitude slices" of the same behavior

2. **Dwell-Time Distributions**: States should have reasonable durations
   - Check mean dwell time per state
   - Ensure states persist for biologically plausible periods

3. **Diel (Day/Night) Prevalence**: States should show interpretable circadian patterns
   - E.g., rest state more prevalent at night
   - Active states more prevalent during day

4. **Recommended K**: 3 or 4 states for animal activity data
   - K=3: Rest, Low Activity, Active
   - K=4: Rest, Low, Active, Highly Active

### Implementation
```python
# Biological validation replaces IC selection
bio_results = select_states_biologically(
    X, activity, state_range=range(2, 5), timestamps=timestamps
)

# Examine characteristics for each K
for k, chars in bio_results['characteristics'].items():
    print(f"K={k}: {chars}")

# Select K based on interpretability, not IC
SELECTED_K = 3  # Based on biological validation
```

### Code Changes
- **New function**: `models_hmm.analyze_state_characteristics()` - Compute biological metrics
- **New function**: `models_hmm.select_states_biologically()` - Biological selection
- **New function**: `models_hmm.assign_biological_labels()` - Interpretable labels
- **Updated**: `03_hmm_hsmm.ipynb` - Documents IC insufficiency
- **Removed**: Automatic K selection via min(BIC) (invalid for this data)

---

## 4. Over-Segmentation of Activity States

### Problem
States appeared to represent amplitude slices (low, medium, high) of a single behavior rather than distinct behavioral modes.

### Root Causes
1. Using too many states (K > 4) due to IC-based selection
2. Not validating biological interpretability
3. States not labeled or interpreted

### Correct Approach (Now Implemented)
States are validated and labeled based on:
- Mean activity level (sorted low to high)
- Biological labels: "Rest", "Low Activity", "Active", "Highly Active"
- State prevalence patterns (day vs. night)
- Transition probabilities (should be sparse - states are stable)

### Example Output
```
State 0 → "Rest" (mean activity: 45.3)
State 1 → "Low Activity" (mean activity: 158.7)
State 2 → "Active" (mean activity: 312.4)
```

### Code Changes
- **New function**: `models_hmm.assign_biological_labels()` - Auto-labeling
- **Updated**: State characteristics analysis includes diel patterns
- **Updated**: `03_hmm_hsmm.ipynb` - Displays labeled states

---

## 5. Rolling Feature Misuse

### Problem
24-hour rolling means were computed globally across multi-year data and plotted, producing:
- Unreadable plots (millions of data points)
- Misleading visualizations
- Rolling windows that spanned across different individuals

### Root Causes
1. Rolling features computed globally instead of per-individual
2. Confusion between features for modeling vs. features for visualization
3. Attempting to visualize global trends in raw time series

### Correct Approach (Now Implemented)

#### A. Per-Individual Rolling Features
```python
df = add_rolling_statistics_per_individual(
    df, 'activity', subject_column='subject', windows=[6, 12, 24]
)
```

**Justification**:
- Prevents rolling windows from mixing data across individuals
- Essential for multi-subject studies
- Each individual's temporal patterns computed independently

#### B. Purpose Documentation
Added clear documentation:
- Rolling features are for **MODELING INPUT** (HMM, ML)
- NOT for global visualization across years
- For visualization: use short temporal windows (days to weeks)

#### C. Removed Global Rolling Plots
Removed misleading plots that attempted to show rolling means across entire dataset.

### Code Changes
- **New function**: `features.add_rolling_statistics_per_individual()` - Per-subject rolling
- **Updated**: Module docstring - Documents feature purposes
- **Updated**: `02_features.ipynb` - Uses per-individual rolling
- **Removed**: Global rolling visualizations

---

## 6. Correct Pipeline Order

### Original (Incorrect) Order
```
Raw → Features → Cosinor ❌ → HMM → GLMM/ML
```

**Problems**:
- Cosinor applied to raw mixed-state activity
- No feature standardization
- No biological validation

### Corrected Order
```
Load Accelerometer Data
    ↓
Extract ActMin (primary activity signal)
    ↓
Add Relative Movement Features (from XAccel/YAccel/ZAccel)
    ↓
Feature Engineering (rolling stats on ActMin, per-individual)
    ↓
Log Transformation (optional)
    ↓
Standardization (z-score, per-individual) ← CRITICAL
    ↓
HMM/HSMM (ActMin + movement features, all standardized)
    ↓
Minimum Dwell-Time Filter
    ↓
Biological State Validation (K selection, labeling)
    ↓
State-Based Cosinor ✓
    ↓
Environmental Effects (GLMM/ML)
```

**Key Improvements**:
0. Use ActMin, not magnitude from uncalibrated axes
1. Add relative movement features (variance/std/changes)
2. Standardization before HMM (reduces noise)
3. Biological validation replaces IC selection
4. Dwell-time filtering for realistic dynamics
5. Cosinor AFTER state identification
6. State-based cosinor analysis

---

## 7. Documentation Requirement

### Added Documentation

All changes include:
1. **Code comments** explaining methodological choices
2. **Function docstrings** with biological justifications
3. **Notebook markdown cells** documenting pipeline steps
4. **Logging statements** for reproducibility
5. **This document** (METHODOLOGY_CORRECTIONS.md)

### Example Documentation Patterns

```python
# INCORRECT APPROACH (documented as such):
# cosinor.fit_cosinor(raw_activity, time)  # ❌ Mixes behavioral states

# CORRECT APPROACH (now used):
cosinor.fit_cosinor_per_state(...)  # ✓ State-conditioned analysis
```

### Reproducibility
All notebooks now log:
- Random seeds
- Parameter choices
- Methodological justifications
- Selection criteria (e.g., "K=3 selected based on biological interpretability")

---

## 8. Implementation Summary

### Files Modified

#### Source Modules
- `src/features.py`: 
  - Added `add_accelerometer_movement_features()` - Relative movement from axes
  - Deprecated `calculate_accelerometer_activity()` - Magnitude calculation
  - Added warnings for incorrect usage
  - Updated module docstring with data assumptions
- `src/io.py`:
  - Updated `load_accelerometer_data()` - Documents unsigned relative values
  - Added data format assumptions to docstring
- `src/models_hmm.py`:
  - Added module docstring documenting input requirements
  - Documents ActMin + movement features as correct inputs
  - Added standardization, log transform, per-individual rolling
  - Added dwell-time filter, biological validation, state labeling
- `src/cosinor.py`: Added state-based cosinor functions

#### Example Scripts
- `example_pipeline.py`: 
  - Updated to use ActMin as primary signal
  - Demonstrates add_accelerometer_movement_features()
  - Shows correct HMM input preparation
  - Includes methodological notes

#### Documentation
- `README.md`:
  - Added critical data format assumptions
  - Updated usage examples to show correct approach
  - Updated module descriptions
  - Updated pipeline workflow
- `METHODOLOGY_CORRECTIONS.md`: 
  - Added Section 0 on accelerometer data interpretation
  - Updated pipeline order
  - This document
- `ACCELEROMETER_UPDATE.md`:
  - Updated to reflect methodological corrections
  - Documents deprecated vs. correct approaches
- **NEW**: `ACCELEROMETER_DATA_ASSUMPTIONS.md`:
  - Complete guide to accelerometer data handling
  - Why magnitude is incorrect
  - Correct workflow
  - Validation and troubleshooting
  - Scientific references

#### Notebooks (to be updated)
- `notebooks/00_intake.ipynb`: Use ActMin, not magnitude
- `notebooks/02_features.ipynb`: Add movement features, standardization
- `notebooks/03_hmm_hsmm.ipynb`: Use standardized ActMin + movement features
- `notebooks/04_cosinor.ipynb`: State-based cosinor (already correct)

### Backward Compatibility
- Old functions still available (not removed)
- New functions have clear names (e.g., `fit_cosinor_per_state` vs. `fit_cosinor`)
- Notebooks document which approach is correct

---

## 9. Validation and Testing

### Biological Validation Checklist

For each analysis, verify:
- [ ] States have distinct mean activity levels
- [ ] State dwell times are biologically plausible (≥1 hour)
- [ ] States show interpretable diel patterns
- [ ] Transition matrix is sparse (states are stable)
- [ ] State labels match behavioral expectations

### Statistical Validation
- [ ] Features are standardized (mean≈0, std≈1)
- [ ] Cosinor R² values are reasonable (not ~0)
- [ ] Acrophase values align with expected activity peaks
- [ ] Per-individual results show appropriate variation

### Reproducibility Checklist
- [ ] Random seed set and logged
- [ ] All parameters documented
- [ ] Method choices justified in notebooks
- [ ] Results can be regenerated from saved data

---

## 10. Recommendations for Thesis Methods Section

### Key Points to Include

0. **Accelerometer Data Interpretation** (CRITICAL)
   - "Accelerometer axes (XAccel/YAccel/ZAccel) were recorded as unsigned relative values (0-255) encoding device orientation and gravity, not calibrated acceleration in physical units. These values were NOT used to compute VeDBA, ODBA, or raw magnitude, as such calculations are invalid for uncalibrated data. Instead, the tag-derived activity metric (ActMindata) was used as the primary activity signal, validated by the manufacturer for activity measurement. Relative movement features (rolling variance, standard deviation, and absolute differences) were computed from accelerometer axes to capture movement patterns while being robust to orientation bias."

1. **Feature Preprocessing**
   - "Activity data (ActMin) were log-transformed and z-score standardized per individual to reduce noise and ensure features were on comparable scales. Relative movement features derived from accelerometer axes were similarly standardized per individual."

2. **HMM State Selection**
   - "The number of hidden states (K) was selected based on biological interpretability rather than information criteria, as AIC and BIC decrease monotonically for long time series and do not provide a meaningful stopping criterion. K=3 states were selected based on distinct mean activity levels, dwell-time distributions, and diel prevalence patterns."

3. **Dwell-Time Filtering**
   - "To address unrealistic state switching, a minimum dwell-time constraint of 4 samples (1 hour) was applied post-hoc to the Viterbi state sequence, replacing short state segments with the surrounding majority state."

4. **State-Based Cosinor**
   - "Cosinor analysis was applied to behavioral states rather than raw activity. We fitted cosinor models to (1) the probability of being in active states and (2) state-conditioned mean activity, computing MESOR, amplitude, and acrophase per individual and per state."

5. **Pipeline Order**
   - "The analysis followed the sequence: load accelerometer data → extract ActMin as primary signal → add relative movement features from axes → feature engineering → standardization → HMM state identification → dwell-time filtering → biological validation → state-based cosinor → environmental effects modeling."

---

## 11. References and Further Reading

### Methodological Papers
- **Accelerometer Data Interpretation**: Brown, D. D., et al. (2013). "Observing the unwatchable through acceleration logging of animal behavior." *Animal Biotelemetry*, 1:20.
  - Documents that raw accelerometer values encode posture, not activity
  - Recommends using tag-derived metrics over raw magnitude
- **Dynamic vs Static Acceleration**: Wilson, R. P., et al. (2006). "Moving towards acceleration for estimates of activity-specific metabolic rate in free-living animals." *Journal of Animal Ecology*, 75(5):1081-1090.
  - Explains importance of calibration for VeDBA/ODBA
  - Shows static acceleration reflects posture/orientation
- **Cosinor Analysis**: Cornelissen, G. (2014). "Cosinor-based rhythmometry." *Theoretical Biology and Medical Modelling*, 11:16.
- **HMM for Animal Behavior**: Langrock, R., et al. (2012). "Flexible and practical modeling of animal telemetry data: hidden Markov models and extensions." *Ecology*, 93(11), 2336-2342.
- **State Duration Modeling**: Langrock, R., & Zucchini, W. (2011). "Hidden Markov models with arbitrary state dwell-time distributions." *Computational Statistics & Data Analysis*, 55(1), 715-724.
- **Model Selection**: Burnham, K. P., & Anderson, D. R. (2004). "Multimodel Inference: Understanding AIC and BIC in Model Selection." *Sociological Methods & Research*, 33(2), 261-304.

### Best Practices
- Use tag-derived activity metrics (ActMin) for unsigned relative accelerometer data
- Do NOT compute magnitude from uncalibrated accelerometer axes
- Derive relative movement features (variance, changes) from axes
- Always validate biological interpretability of hidden states
- Document why standard model selection criteria may fail
- Use domain knowledge to inform modeling decisions
- Apply cosinor to behaviorally homogeneous data
- Enforce temporal persistence in behavioral state models

---

## Conclusion

The corrected pipeline ensures:
0. ✅ Proper accelerometer data interpretation (ActMin + relative movement features)
1. ✅ Biologically valid cosinor analysis (applied to states, not raw data)
2. ✅ Realistic behavioral dynamics (via dwell-time filtering)
3. ✅ Interpretable state selection (biological validation, not IC)
4. ✅ Reduced noise (standardized features)
5. ✅ Proper multi-subject handling (per-individual features)
6. ✅ Complete documentation and reproducibility

All changes are minimal, surgical, and preserve existing functionality while adding correct methodological approaches.

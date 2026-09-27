"""Diagnostic analysis to identify data issues."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the data
print("Loading data...")
df = pd.read_csv('data/raw/sample_accelerometer_data.csv')
df['timestamp'] = pd.to_datetime(df['ACTEndTimeAllS'])

print("\n" + "="*60)
print("DATA STRUCTURE ANALYSIS")
print("="*60)
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"Number of subjects: {df['subject'].nunique()}")
print(f"Subjects: {df['subject'].unique()}")

print("\n" + "="*60)
print("ACTIVITY DISTRIBUTION (ActMindata)")
print("="*60)
print(df['ActMindata'].describe())
print(f"\nZero activity count: {(df['ActMindata'] == 0).sum()} ({100*(df['ActMindata'] == 0).sum()/len(df):.1f}%)")
print(f"Non-zero activity count: {(df['ActMindata'] > 0).sum()} ({100*(df['ActMindata'] > 0).sum()/len(df):.1f}%)")

print("\n" + "="*60)
print("SUBJECT-LEVEL STATISTICS")
print("="*60)
for subject in df['subject'].unique():
    subj_data = df[df['subject'] == subject]
    print(f"\n{subject}:")
    print(f"  Rows: {len(subj_data):,}")
    print(f"  Date range: {subj_data['timestamp'].min()} to {subj_data['timestamp'].max()}")
    print(f"  Duration: {(subj_data['timestamp'].max() - subj_data['timestamp'].min()).days} days")
    print(f"  Mean ActMindata: {subj_data['ActMindata'].mean():.2f}")
    print(f"  Zero activity %: {100*(subj_data['ActMindata'] == 0).sum()/len(subj_data):.1f}%")

print("\n" + "="*60)
print("TEMPORAL PATTERNS")
print("="*60)
df['hour'] = df['timestamp'].dt.hour  # type: ignore[attr-defined]
hourly_avg = df.groupby('hour')['ActMindata'].mean()
print("\nAverage activity by hour:")
print(hourly_avg)

print("\n" + "="*60)
print("FEATURE ENGINEERING CONCERNS")
print("="*60)

# Check if we have enough variation per subject
print("\nPer-subject activity variance:")
for subject in df['subject'].unique():
    subj_data = df[df['subject'] == subject]
    print(f"  {subject}: {subj_data['ActMindata'].var():.2f}")

# Check accelerometer axes
print("\nAccelerometer axes statistics:")
for col in ['XAccel', 'YAccel', 'ZAccel']:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}")

print("\n" + "="*60)
print("POTENTIAL ISSUES DETECTED")
print("="*60)

issues = []

# Check 1: Multiple subjects with wildly different patterns
n_subjects = df['subject'].nunique()
if n_subjects > 1:
    subject_means = df.groupby('subject')['ActMindata'].mean()
    subject_vars = df.groupby('subject')['ActMindata'].var()
    if subject_vars.max() / subject_vars.min() > 100:
        issues.append(f"⚠️  CRITICAL: Subject variance differs by {subject_vars.max()/subject_vars.min():.0f}x - needs per-subject standardization")
    if subject_means.max() / subject_means.min() > 10:
        issues.append(f"⚠️  CRITICAL: Subject means differ by {subject_means.max()/subject_means.min():.0f}x - needs per-subject standardization")

# Check 2: Long time series causing rolling feature issues
duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
if duration_days > 365:
    issues.append(f"⚠️  CRITICAL: {duration_days} day dataset - rolling features on entire dataset are meaningless")
    issues.append("   → Need to compute rolling features on shorter windows or per-subject basis")

# Check 3: Many zeros
zero_pct = 100 * (df['ActMindata'] == 0).sum() / len(df)
if zero_pct > 50:
    issues.append(f"⚠️  WARNING: {zero_pct:.1f}% zero activity - consider log(x+1) transformation")

# Check 4: Insufficient data points per subject
min_points = df.groupby('subject').size().min()
if min_points < 1000:
    issues.append(f"⚠️  WARNING: Smallest subject has only {min_points} data points - may be insufficient for HMM")

# Check 5: Check if features would have different scales
if n_subjects > 1:
    # Simulate a rolling feature
    test_rolling = df.groupby('subject')['ActMindata'].rolling(window=6, min_periods=1).mean()
    rolling_by_subject = test_rolling.groupby(df['subject']).mean()
    if rolling_by_subject.max() / rolling_by_subject.min() > 5:
        issues.append(f"⚠️  CRITICAL: Rolling features differ by {rolling_by_subject.max()/rolling_by_subject.min():.1f}x across subjects")
        issues.append("   → MUST standardize features per-subject BEFORE any modeling")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✓ No critical issues detected")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print("1. Standardize ALL features per-subject (z-score within each individual)")
print("2. Apply HMM per-subject OR use a mixed-effects framework")
print("3. For long time series, compute rolling features on shorter windows (days-weeks)")
print("4. Use log(ActMindata + 1) transformation to handle zeros")
print("5. Validate that HMM states align with biological expectations (rest/active)")
print("6. Apply cosinor AFTER state assignment, not on raw activity")

# Create diagnostic plot
print("\n" + "="*60)
print("CREATING DIAGNOSTIC PLOTS")
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Activity distribution
axes[0, 0].hist(df['ActMindata'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('ActMindata')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Activity Distribution (Raw)')
axes[0, 0].axvline(df['ActMindata'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Plot 2: Activity distribution (log scale)
axes[0, 1].hist(np.log1p(df['ActMindata']), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('log(ActMindata + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Activity Distribution (Log-transformed)')

# Plot 3: Per-subject activity
subject_data = df.groupby('subject')['ActMindata'].agg(['mean', 'std', 'count'])
axes[1, 0].bar(range(len(subject_data)), subject_data['mean'], yerr=subject_data['std'], capsize=5)
axes[1, 0].set_xticks(range(len(subject_data)))
axes[1, 0].set_xticklabels(subject_data.index, rotation=45)
axes[1, 0].set_ylabel('Mean Activity')
axes[1, 0].set_title('Activity by Subject (Mean ± SD)')

# Plot 4: Hourly pattern
hourly_by_subject = df.groupby(['subject', 'hour'])['ActMindata'].mean().unstack(level=0)
for subject in hourly_by_subject.columns:
    axes[1, 1].plot(hourly_by_subject.index, hourly_by_subject[subject], marker='o', label=subject)
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Mean Activity')
axes[1, 1].set_title('Circadian Pattern by Subject')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Time series sample (first subject, first 7 days)
first_subject = df['subject'].iloc[0]
sample_data = df[df['subject'] == first_subject].head(7 * 96)  # ~7 days at 15-min intervals
axes[2, 0].plot(sample_data['timestamp'], sample_data['ActMindata'])
axes[2, 0].set_xlabel('Time')
axes[2, 0].set_ylabel('Activity')
axes[2, 0].set_title(f'Sample Time Series ({first_subject}, 7 days)')
axes[2, 0].tick_params(axis='x', rotation=45)

# Plot 6: Coefficient of variation by subject
cv_by_subject = df.groupby('subject')['ActMindata'].agg(['mean', 'std'])
cv_by_subject['cv'] = cv_by_subject['std'] / cv_by_subject['mean']
axes[2, 1].bar(range(len(cv_by_subject)), cv_by_subject['cv'])
axes[2, 1].set_xticks(range(len(cv_by_subject)))
axes[2, 1].set_xticklabels(cv_by_subject.index, rotation=45)
axes[2, 1].set_ylabel('Coefficient of Variation')
axes[2, 1].set_title('Within-Subject Variability')
axes[2, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/diagnostic_analysis.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

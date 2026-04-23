"""
COGS Distribution Analysis: sales.csv vs sample_submission.csv

Check whether COGS values are significantly different between training and test periods.
This helps validate whether COGS is a safe exogenous feature for both periods.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

print(f"\n{'='*70}")
print("COGS DISTRIBUTION ANALYSIS")
print(f"{'='*70}")

# Load data
train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"\nData loaded:")
print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")

# Extract COGS
train_cogs = train['COGS'].dropna()
test_cogs = test['COGS'].dropna()

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

print(f"\n{'='*70}")
print("DESCRIPTIVE STATISTICS")
print(f"{'='*70}")

print(f"\nTRAIN COGS (sales.csv):")
print(f"  Count:      {len(train_cogs):,}")
print(f"  Mean:       {train_cogs.mean():,.2f}")
print(f"  Std:        {train_cogs.std():,.2f}")
print(f"  Min:        {train_cogs.min():,.2f}")
print(f"  Q1 (25%):   {train_cogs.quantile(0.25):,.2f}")
print(f"  Median:     {train_cogs.median():,.2f}")
print(f"  Q3 (75%):   {train_cogs.quantile(0.75):,.2f}")
print(f"  Max:        {train_cogs.max():,.2f}")
print(f"  IQR:        {train_cogs.quantile(0.75) - train_cogs.quantile(0.25):,.2f}")

print(f"\nTEST COGS (sample_submission.csv):")
print(f"  Count:      {len(test_cogs):,}")
print(f"  Mean:       {test_cogs.mean():,.2f}")
print(f"  Std:        {test_cogs.std():,.2f}")
print(f"  Min:        {test_cogs.min():,.2f}")
print(f"  Q1 (25%):   {test_cogs.quantile(0.25):,.2f}")
print(f"  Median:     {test_cogs.median():,.2f}")
print(f"  Q3 (75%):   {test_cogs.quantile(0.75):,.2f}")
print(f"  Max:        {test_cogs.max():,.2f}")
print(f"  IQR:        {test_cogs.quantile(0.75) - test_cogs.quantile(0.25):,.2f}")

# ============================================================================
# DISTRIBUTION DIFFERENCES
# ============================================================================

print(f"\n{'='*70}")
print("DISTRIBUTION DIFFERENCES")
print(f"{'='*70}")

mean_diff = test_cogs.mean() - train_cogs.mean()
mean_pct_change = (mean_diff / train_cogs.mean()) * 100
std_diff = test_cogs.std() - train_cogs.std()
std_pct_change = (std_diff / train_cogs.std()) * 100

print(f"\nMean:")
print(f"  Train:      {train_cogs.mean():,.2f}")
print(f"  Test:       {test_cogs.mean():,.2f}")
print(f"  Difference: {mean_diff:,.2f} ({mean_pct_change:+.2f}%)")

print(f"\nStandard Deviation:")
print(f"  Train:      {train_cogs.std():,.2f}")
print(f"  Test:       {test_cogs.std():,.2f}")
print(f"  Difference: {std_diff:,.2f} ({std_pct_change:+.2f}%)")

print(f"\nRange (Max - Min):")
print(f"  Train:      {train_cogs.max() - train_cogs.min():,.2f}")
print(f"  Test:       {test_cogs.max() - test_cogs.min():,.2f}")

# ============================================================================
# CORRELATION: COGS vs REVENUE
# ============================================================================

print(f"\n{'='*70}")
print("COGS vs REVENUE CORRELATION")
print(f"{'='*70}")

train_corr = train[['COGS', 'Revenue']].corr().iloc[0, 1]
test_corr = test[['COGS', 'Revenue']].corr().iloc[0, 1]

print(f"\nTrain (sales.csv):")
print(f"  COGS-Revenue correlation: {train_corr:.4f}")
print(f"  Interpretation: {'Very strong' if train_corr > 0.9 else 'Strong' if train_corr > 0.7 else 'Moderate'} positive")

print(f"\nTest (sample_submission.csv):")
print(f"  COGS-Revenue correlation: {test_corr:.4f}")
print(f"  Interpretation: {'Very strong' if test_corr > 0.9 else 'Strong' if test_corr > 0.7 else 'Moderate'} positive")

# COGS markup analysis
print(f"\nCOGS Markup (Revenue / COGS):")
train_markup = (train['Revenue'] / train['COGS']).dropna()
test_markup = (test['Revenue'] / test['COGS']).dropna()

print(f"  Train markup: {train_markup.mean():.4f}x ± {train_markup.std():.4f}")
print(f"  Test markup:  {test_markup.mean():.4f}x ± {test_markup.std():.4f}")
print(f"  Difference:   {test_markup.mean() - train_markup.mean():+.4f}x")

# ============================================================================
# DISTRIBUTION SHAPE ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print("DISTRIBUTION SHAPE")
print(f"{'='*70}")

train_skew = train_cogs.skew()
test_skew = test_cogs.skew()
train_kurtosis = train_cogs.kurtosis()
test_kurtosis = test_cogs.kurtosis()

print(f"\nSkewness (negative = left-skewed, positive = right-skewed):")
print(f"  Train: {train_skew:.4f}")
print(f"  Test:  {test_skew:.4f}")

print(f"\nKurtosis (higher = more outliers):")
print(f"  Train: {train_kurtosis:.4f}")
print(f"  Test:  {test_kurtosis:.4f}")

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

print(f"\n{'='*70}")
print("OUTLIER ANALYSIS")
print(f"{'='*70}")

def count_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), len(outliers) / len(series) * 100

train_outliers, train_pct = count_outliers_iqr(train_cogs)
test_outliers, test_pct = count_outliers_iqr(test_cogs)

print(f"\nOutliers (using 1.5 × IQR rule):")
print(f"  Train: {train_outliers} rows ({train_pct:.2f}%)")
print(f"  Test:  {test_outliers} rows ({test_pct:.2f}%)")

# ============================================================================
# ASSESSMENT
# ============================================================================

print(f"\n{'='*70}")
print("ASSESSMENT")
print(f"{'='*70}")

# Check if distributions are similar enough to use COGS as exogenous feature
mean_shift_acceptable = abs(mean_pct_change) < 20  # Within 20%
std_shift_acceptable = abs(std_pct_change) < 30    # Within 30%
corr_acceptable = train_corr > 0.85 and test_corr > 0.85
outlier_acceptable = train_pct < 10 and test_pct < 10

print(f"\nDistribution Similarity Checks:")
print(f"  Mean shift < 20%:       {mean_shift_acceptable} ({mean_pct_change:+.2f}%)")
print(f"  Std shift < 30%:        {std_shift_acceptable} ({std_pct_change:+.2f}%)")
print(f"  COGS-Revenue corr > 0.85: {corr_acceptable} (train={train_corr:.4f}, test={test_corr:.4f})")
print(f"  Outliers < 10%:         {outlier_acceptable} (train={train_pct:.2f}%, test={test_pct:.2f}%)")

all_checks_pass = mean_shift_acceptable and std_shift_acceptable and corr_acceptable and outlier_acceptable

print(f"\n{'✓ SAFE' if all_checks_pass else '⚠ CAUTION'} for using COGS as exogenous feature:")
if all_checks_pass:
    print(f"  - COGS distributions are similar between train and test")
    print(f"  - COGS-Revenue relationship is stable and strong")
    print(f"  - No excessive outliers")
    print(f"  → Using COGS in model is appropriate")
else:
    warnings = []
    if not mean_shift_acceptable:
        warnings.append(f"    • Mean shifted {mean_pct_change:+.2f}% (expect consistency)")
    if not std_shift_acceptable:
        warnings.append(f"    • Std shifted {std_pct_change:+.2f}% (expect consistency)")
    if not corr_acceptable:
        warnings.append(f"    • COGS-Revenue correlation inconsistent across periods")
    if not outlier_acceptable:
        warnings.append(f"    • Excessive outliers in one or both periods")
    
    print(f"  Warnings:")
    for w in warnings:
        print(w)
    print(f"  → Consider COGS usage carefully or apply robust scaling")

print(f"\n{'='*70}\n")

import pandas as pd
import numpy as np
import os

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

# Load data
train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# Extract COGS
train_cogs = train['COGS'].dropna()
test_cogs = test['COGS'].dropna()
train_revenue = train['Revenue'].dropna()
test_revenue = test['Revenue'].dropna()

print(f'\n{'='*70}')
print('COGS DISTRIBUTION ANALYSIS')
print(f'{'='*70}')

print(f'Train Mean: {train_cogs.mean():,.2f}, Std: {train_cogs.std():,.2f}')
print(f'Test Mean:  {test_cogs.mean():,.2f}, Std: {test_cogs.std():,.2f}')

mean_pct_change = (test_cogs.mean() - train_cogs.mean()) / train_cogs.mean() * 100
std_pct_change = (test_cogs.std() - train_cogs.std()) / train_cogs.std() * 100

print(f'\nMean Shift: {mean_pct_change:+.2f}%')
print(f'Std Shift:  {std_pct_change:+.2f}%')

train_corr = train[['COGS', 'Revenue']].corr().iloc[0, 1]
test_corr = test[['COGS', 'Revenue']].corr().iloc[0, 1]
print(f'\nCOGS-Revenue Correlation:')
print(f'  Train: {train_corr:.4f}')
print(f'  Test:  {test_corr:.4f}')

def count_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers) / len(series) * 100

train_outlier_pct = count_outliers_iqr(train_cogs)
test_outlier_pct = count_outliers_iqr(test_cogs)
print(f'\nOutlier Percentage:')
print(f'  Train: {train_outlier_pct:.2f}%')
print(f'  Test:  {test_outlier_pct:.2f}%')

print(f'\n{'='*70}')
print('CONCLUSION')
print(f'{'='*70}')
if abs(mean_pct_change) < 20 and abs(std_pct_change) < 30 and train_corr > 0.85 and test_corr > 0.85:
    print('COGS distribution is stable. Safe to use as exogenous feature.')
else:
    print('CAUTION: COGS distribution shows significant shifts or inconsistencies.')

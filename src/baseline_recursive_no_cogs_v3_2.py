"""
Recursive Single-Table Baseline v3.2 -- Controlled Rollback + Repair
====================================================================

CHANGES FROM v3.1:
  - PATH B: EWMA REMOVED ENTIRELY (train/inference parity cannot be guaranteed)
  - Restored v2 feature groups: rolling median, trend/ratios, expanding_mean+std,
    count_positive_7/30, avg_revenue_by_dayofweek, avg_revenue_by_month
  - Removed excess long lags (182, 365, 371) -> max lag = 364, burn-in = 364
  - Removed active_frac_30/90, recent_same_dow_mean
  - Kept: strict continuity, dynamic horizon backtest, no-COGS, recursive framework
  - NaN policy: LightGBM native NaN handling, no blanket fillna(0)
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_DIR = 'outputs'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# v2-style lags (max=364), no excess long lags
LAG_WINDOWS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 364]
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
BURN_IN_DAYS = 364

print(f"\n{'='*80}")
print("BASELINE RECURSIVE v3.2 -- ROLLBACK + REPAIR (NO EWMA)")
print(f"{'='*80}")

# ============================================================================
# 1. LOAD DATA & CONTINUITY CHECK
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

    print(f"\nData Loading:")
    print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")

    # Strict continuity
    date_diff = (train['Date'].max() - train['Date'].min()).days + 1
    if date_diff == len(train):
        print(f"  [OK] Daily continuity verified. Row-based lags = day-based lags.")
    else:
        raise ValueError(f"Continuity FAILED: {date_diff - len(train)} missing days.")

    # Dynamic horizon
    assert test['Date'].nunique() == len(test), "Test dates not unique!"
    horizon_days = len(test)
    print(f"  Inferred Horizon: {horizon_days} days")

    return train, test, horizon_days


# ============================================================================
# 2. CALENDAR FEATURES
# ============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df


# ============================================================================
# 3. VECTORIZED FEATURE ENGINEERING (TRAINING)
# ============================================================================

def prepare_train_features_vectorized(train_df: pd.DataFrame, burnin: int = BURN_IN_DAYS) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = train_df[['Date', 'Revenue']].copy()
    print(f"\nFeature Engineering (Vectorized Training):")

    # CALENDAR (8)
    df = add_calendar_features(df)
    shifted_rev = df['Revenue'].shift(1)

    # LAGS (11)
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)

    # ROLLING STATS (25 = 5 windows x 5 stats: mean/std/min/max/median)
    for win in ROLLING_WINDOWS:
        df[f'roll_mean_{win}'] = shifted_rev.rolling(window=win).mean()
        df[f'roll_std_{win}'] = shifted_rev.rolling(window=win).std(ddof=1)
        df[f'roll_min_{win}'] = shifted_rev.rolling(window=win).min()
        df[f'roll_max_{win}'] = shifted_rev.rolling(window=win).max()
        df[f'roll_median_{win}'] = shifted_rev.rolling(window=win).median()

    # EXPANDING (2)
    df['expanding_mean'] = shifted_rev.expanding().mean()
    df['expanding_std'] = shifted_rev.expanding().std(ddof=1)

    # TREND / RATIO (4)
    mean_7 = shifted_rev.rolling(window=7).mean()
    mean_30 = shifted_rev.rolling(window=30).mean()
    mean_90 = shifted_rev.rolling(window=90).mean()
    df['mean_7_minus_mean_30'] = mean_7 - mean_30
    df['mean_7_over_mean_30'] = mean_7 / (mean_30 + 1e-6)
    df['mean_30_over_mean_90'] = mean_30 / (mean_90 + 1e-6)
    df['volatility_ratio'] = shifted_rev.rolling(window=30).std(ddof=1) / (mean_30 + 1e-6)

    # ACTIVITY (3)
    revenue_positive = (df['Revenue'] > 0).astype(int)
    df['count_positive_7'] = revenue_positive.shift(1).rolling(window=7).sum()
    df['count_positive_30'] = revenue_positive.shift(1).rolling(window=30).sum()

    def days_since_nonzero_col(rev_series):
        result = [np.nan]
        for i in range(1, len(rev_series)):
            for j in range(i - 1, -1, -1):
                if rev_series.iloc[j] > 0:
                    result.append(i - j)
                    break
            else:
                result.append(np.nan)
        return result
    df['days_since_nonzero'] = days_since_nonzero_col(df['Revenue'])

    # SEASONAL HISTORY (2) -- all-history, v2-style
    def avg_revenue_by_dow(df):
        result = []
        for i in range(len(df)):
            if i == 0:
                result.append(np.nan)
            else:
                mask = df.iloc[:i]['dayofweek'] == df.iloc[i]['dayofweek']
                result.append(df.iloc[:i][mask]['Revenue'].mean() if mask.any() else np.nan)
        return result

    def avg_revenue_by_month(df):
        result = []
        for i in range(len(df)):
            if i == 0:
                result.append(np.nan)
            else:
                mask = df.iloc[:i]['month'] == df.iloc[i]['month']
                result.append(df.iloc[:i][mask]['Revenue'].mean() if mask.any() else np.nan)
        return result

    df['avg_revenue_by_dayofweek'] = avg_revenue_by_dow(df)
    df['avg_revenue_by_month'] = avg_revenue_by_month(df)

    # DROP BURN-IN
    df = df.iloc[burnin:].reset_index(drop=True)

    feature_names = [c for c in df.columns if c not in ['Date', 'Revenue']]

    # NaN audit
    nan_counts = df[feature_names].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"  NaN audit after burn-in ({burnin} days):")
        for col, cnt in nan_cols.items():
            print(f"    {col}: {cnt} NaNs")
        print(f"  -> LightGBM will handle these natively.")
    else:
        print(f"  [OK] No NaNs after burn-in ({burnin} days).")

    print(f"  Total features: {len(feature_names)}")
    print(f"  Breakdown: 8 cal + {len(LAG_WINDOWS)} lags + {5*len(ROLLING_WINDOWS)} rolling + 2 expanding + 4 trend + 3 activity + 2 seasonal = {len(feature_names)}")

    X_df = df[feature_names]
    y = df['Revenue'].values
    return X_df, y, feature_names


# ============================================================================
# 4. ROW-BY-ROW FEATURE ENGINEERING (RECURSIVE INFERENCE)
# ============================================================================

def engineer_features_for_row(history_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the LAST row. Exact parity with vectorized training."""
    df = history_df.copy()
    df = add_calendar_features(df)
    idx = len(df) - 1

    # LAGS
    for lag in LAG_WINDOWS:
        if idx >= lag:
            df.loc[idx, f'lag_{lag}'] = df.loc[idx - lag, 'Revenue']
        else:
            df.loc[idx, f'lag_{lag}'] = np.nan

    # ROLLING STATS (25) -- ddof=1 for std parity
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{win}'] = past.mean()
            df.loc[idx, f'roll_std_{win}'] = np.std(past, ddof=1) if len(past) > 1 else np.nan
            df.loc[idx, f'roll_min_{win}'] = past.min()
            df.loc[idx, f'roll_max_{win}'] = past.max()
            df.loc[idx, f'roll_median_{win}'] = np.median(past)
        else:
            for s in ['mean', 'std', 'min', 'max', 'median']:
                df.loc[idx, f'roll_{s}_{win}'] = np.nan

    # EXPANDING (2)
    past_all = df.loc[0:idx - 1, 'Revenue'].values
    if len(past_all) > 0:
        df.loc[idx, 'expanding_mean'] = past_all.mean()
        df.loc[idx, 'expanding_std'] = np.std(past_all, ddof=1) if len(past_all) > 1 else np.nan
    else:
        df.loc[idx, 'expanding_mean'] = np.nan
        df.loc[idx, 'expanding_std'] = np.nan

    # TREND / RATIO (4)
    if idx >= 7:
        m7 = df.loc[max(0, idx - 7):idx - 1, 'Revenue'].mean()
    else:
        m7 = np.nan
    if idx >= 30:
        m30 = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].mean()
    else:
        m30 = np.nan
    if idx >= 90:
        m90 = df.loc[max(0, idx - 90):idx - 1, 'Revenue'].mean()
    else:
        m90 = np.nan

    df.loc[idx, 'mean_7_minus_mean_30'] = m7 - m30 if not np.isnan(m7) and not np.isnan(m30) else np.nan
    df.loc[idx, 'mean_7_over_mean_30'] = m7 / (m30 + 1e-6) if not np.isnan(m7) and not np.isnan(m30) else np.nan
    df.loc[idx, 'mean_30_over_mean_90'] = m30 / (m90 + 1e-6) if not np.isnan(m30) and not np.isnan(m90) else np.nan

    if idx >= 30:
        past_30 = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].values
        vol_30 = np.std(past_30, ddof=1) if len(past_30) > 1 else np.nan
        df.loc[idx, 'volatility_ratio'] = vol_30 / (m30 + 1e-6) if not np.isnan(vol_30) else np.nan
    else:
        df.loc[idx, 'volatility_ratio'] = np.nan

    # ACTIVITY (3)
    if idx >= 7:
        past_7 = df.loc[max(0, idx - 7):idx - 1, 'Revenue'].values
        df.loc[idx, 'count_positive_7'] = (past_7 > 0).sum()
    else:
        df.loc[idx, 'count_positive_7'] = np.nan
    if idx >= 30:
        past_30_act = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].values
        df.loc[idx, 'count_positive_30'] = (past_30_act > 0).sum()
    else:
        df.loc[idx, 'count_positive_30'] = np.nan

    if idx > 0:
        for j in range(idx - 1, -1, -1):
            if df.loc[j, 'Revenue'] > 0:
                df.loc[idx, 'days_since_nonzero'] = idx - j
                break
        else:
            df.loc[idx, 'days_since_nonzero'] = np.nan
    else:
        df.loc[idx, 'days_since_nonzero'] = np.nan

    # SEASONAL HISTORY (2) -- all-history v2-style
    current_dow = df.loc[idx, 'dayofweek']
    current_month = df.loc[idx, 'month']

    mask_dow = df.loc[0:idx - 1, 'dayofweek'] == current_dow
    if mask_dow.any():
        df.loc[idx, 'avg_revenue_by_dayofweek'] = df.loc[0:idx - 1][mask_dow]['Revenue'].mean()
    else:
        df.loc[idx, 'avg_revenue_by_dayofweek'] = np.nan

    mask_month = df.loc[0:idx - 1, 'month'] == current_month
    if mask_month.any():
        df.loc[idx, 'avg_revenue_by_month'] = df.loc[0:idx - 1][mask_month]['Revenue'].mean()
    else:
        df.loc[idx, 'avg_revenue_by_month'] = np.nan

    return df


# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

def train_model(X_df: pd.DataFrame, y: np.ndarray) -> object:
    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=300,
            random_state=RANDOM_STATE,
            verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            learning_rate=0.05, max_depth=6, n_estimators=300, random_state=RANDOM_STATE
        )
    model.fit(X_df, y)
    return model


# ============================================================================
# 6. DYNAMIC HORIZON-ALIGNED BACKTEST
# ============================================================================

def rolling_origin_backtest(train_df: pd.DataFrame, horizon_days: int) -> Dict:
    print(f"\n{'='*80}")
    print(f"DYNAMIC HORIZON-ALIGNED BACKTEST ({horizon_days} DAYS PER FOLD)")
    print(f"{'='*80}")

    max_date = train_df['Date'].max()

    # Fold 1 (recent)
    val_end_1 = max_date
    val_start_1 = val_end_1 - pd.Timedelta(days=horizon_days - 1)
    train_end_1 = val_start_1 - pd.Timedelta(days=1)

    # Fold 2 (past)
    val_end_2 = train_end_1
    val_start_2 = val_end_2 - pd.Timedelta(days=horizon_days - 1)
    train_end_2 = val_start_2 - pd.Timedelta(days=1)

    folds_def = [
        {'train_end': train_end_1, 'val_start': val_start_1, 'val_end': val_end_1, 'label': 'Fold 1 (Recent)'},
        {'train_end': train_end_2, 'val_start': val_start_2, 'val_end': val_end_2, 'label': 'Fold 2 (Past)'},
    ]

    results = []

    for fold_def in folds_def:
        train_cutoff = pd.Timestamp(fold_def['train_end'])
        val_start = pd.Timestamp(fold_def['val_start'])
        val_end = pd.Timestamp(fold_def['val_end'])

        train_fold = train_df[train_df['Date'] <= train_cutoff].copy()
        val_fold = train_df[(train_df['Date'] >= val_start) & (train_df['Date'] <= val_end)].copy()

        print(f"\n{fold_def['label']}:")
        print(f"  Train: {train_fold['Date'].min().date()} to {train_fold['Date'].max().date()} ({len(train_fold):,} rows)")
        print(f"  Val:   {val_fold['Date'].min().date()} to {val_fold['Date'].max().date()} ({len(val_fold):,} rows)")

        X_train, y_train, feature_names = prepare_train_features_vectorized(train_fold, burnin=BURN_IN_DAYS)
        print(f"  Training: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")

        if X_train.shape[0] < 100:
            print(f"  Skipped (insufficient training rows after burnin)")
            continue

        model = train_model(X_train, y_train)

        # Recursive validation
        history = train_fold[['Date', 'Revenue']].copy()
        preds = []
        actuals = []

        for val_idx in range(len(val_fold)):
            val_date = val_fold.iloc[val_idx]['Date']
            val_revenue = val_fold.iloc[val_idx]['Revenue']

            new_row = pd.DataFrame({'Date': [val_date], 'Revenue': [np.nan]})
            history = pd.concat([history, new_row], ignore_index=True)
            history = engineer_features_for_row(history)

            X_last = history.iloc[-1:][feature_names]
            y_pred = max(0, model.predict(X_last)[0])

            preds.append(y_pred)
            actuals.append(val_revenue)
            history.loc[len(history) - 1, 'Revenue'] = y_pred

            if (val_idx + 1) % 100 == 0:
                print(f"    Val step {val_idx + 1}/{len(val_fold)}")

        mae = mean_absolute_error(actuals, preds)
        print(f"  MAE: {mae:,.0f}")
        results.append({'fold': fold_def['label'], 'MAE': mae, 'Length': len(val_fold)})

    mean_mae = np.mean([r['MAE'] for r in results]) if results else np.inf
    print(f"\n{'='*80}")
    print(f"Backtest Summary: Mean MAE = {mean_mae:,.0f}")
    print(f"{'='*80}")

    return {'folds': results, 'mean_mae': mean_mae}


# ============================================================================
# 7. RECURSIVE TEST PREDICTION
# ============================================================================

def recursive_predict_test(train_df, test_df, model, feature_names):
    print(f"\nRecursive Test Prediction ({len(test_df)} rows)...")
    history = train_df[['Date', 'Revenue']].copy()
    preds = []

    for test_idx in range(len(test_df)):
        test_date = test_df.iloc[test_idx]['Date']
        new_row = pd.DataFrame({'Date': [test_date], 'Revenue': [np.nan]})
        history = pd.concat([history, new_row], ignore_index=True)
        history = engineer_features_for_row(history)

        X_last = history.iloc[-1:][feature_names]
        y_pred = max(0, model.predict(X_last)[0])

        preds.append(y_pred)
        history.loc[len(history) - 1, 'Revenue'] = y_pred

        if (test_idx + 1) % 100 == 0:
            print(f"  Predicted {test_idx + 1}/{len(test_df)}")

    preds = np.array(preds)
    print(f"  Predictions: min={preds.min():,.0f}, max={preds.max():,.0f}, mean={preds.mean():,.0f}")
    return preds


# ============================================================================
# 8. SAVE SUBMISSION
# ============================================================================

def save_submission(test_df, predictions):
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values if 'COGS' in test_df.columns else 0
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'submission_no_cogs_v3_2.csv')
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved: {output_file}")
    return output_file


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    start_time = datetime.now()
    train, test, horizon_days = load_data()

    backtest_result = rolling_origin_backtest(train, horizon_days)

    print(f"\n{'='*80}")
    print("PHASE 2: FINAL TRAINING ON FULL DATA")
    print(f"{'='*80}")
    X_full, y_full, feature_names = prepare_train_features_vectorized(train, burnin=BURN_IN_DAYS)
    model_final = train_model(X_full, y_full)

    print(f"\n{'='*80}")
    print("PHASE 3: TEST PREDICTION (RECURSIVE)")
    print(f"{'='*80}")
    test_preds = recursive_predict_test(train, test, model_final, feature_names)

    output_file = save_submission(test, test_preds)

    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    for fold in backtest_result['folds']:
        print(f"  {fold['fold']}: MAE = {fold['MAE']:,.0f}")
    print(f"  Mean MAE: {backtest_result['mean_mae']:,.0f}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nExecution time: {elapsed:.1f} seconds")
    print(f"\n[OK] Complete! Ready for Kaggle submission.")

if __name__ == '__main__':
    main()

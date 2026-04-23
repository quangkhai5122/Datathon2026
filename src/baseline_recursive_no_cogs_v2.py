"""
Recursive Single-Table Baseline v2 — Improved NO_COGS with 55 Features
=====================================================================

IMPROVEMENTS OVER PHASE 3 BASELINE:

1. Roll_std Consistency Fix:
   - Training: df['Revenue'].shift(1).rolling(win).std(ddof=1)
   - Inference: np.std(past_values, ddof=1)
   - CONSISTENCY GUARANTEE: Both use ddof=1 (unbiased sample std)

2. Horizon-Aligned Rolling-Origin Backtest:
   - 3 folds with annual validation windows (~365 days each)
   - Fold 1: Train to 2019-12-31, Validate 2020 (366 days)
   - Fold 2: Train to 2020-12-31, Validate 2021 (365 days)
   - Fold 3: Train to 2021-12-31, Validate 2022 (365 days)
   - Prioritizes recent folds (closer to competition dates)
   - Recursive validation matches true forecasting procedure

3. Stronger Revenue-Only Feature Set (55 features total):
   - Calendar/Seasonality: 8 features
   - Revenue Lags: 11 features (1,2,5,7,14,28,30,60,90,180,364)
   - Rolling Statistics: 25 features (mean/std/min/max/median for windows 7,14,30,60,90)
   - Expanding Statistics: 2 features (mean, std)
   - Trend/Ratio Features: 4 features
   - Activity/Sparsity: 3 features
   - Seasonal History: 2 features (avg by dayofweek, avg by month)

Time-Leakage Prevention:
   - All features computed from history only (shift(1) where needed)
   - No current-row or future Revenue used
   - Recursive inference matches training exactly
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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Feature windows (optimized set)
LAG_WINDOWS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 364]  # 11 lags
ROLLING_WINDOWS = [7, 14, 30, 60, 90]  # 5 windows × 5 stats = 25 rolling features

print(f"\n{'='*80}")
print("BASELINE RECURSIVE v2 — IMPROVED NO_COGS (55 FEATURES)")
print(f"{'='*80}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate training and test data."""
    train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    
    print(f"\nData Loading:")
    print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")
    print(f"  Revenue range: {train['Revenue'].min():,.0f} to {train['Revenue'].max():,.0f}")
    
    # Verify continuity
    date_diff = (train['Date'].max() - train['Date'].min()).days + 1
    if date_diff == len(train):
        print(f"  ✓ Daily continuity: 100% (no missing dates)")
    else:
        print(f"  ! Missing dates: {date_diff - len(train)} days")
    
    return train, test


# ============================================================================
# 2. VECTORIZED FEATURE ENGINEERING FOR TRAINING
# ============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 calendar/seasonality features."""
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


def prepare_train_features_vectorized(train_df: pd.DataFrame, burnin: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training features using full vectorization.
    All rows computed simultaneously via shift/rolling/expanding.
    
    CONSISTENCY: roll_std uses ddof=1 (unbiased).
                 Recursive inference must match with np.std(..., ddof=1).
    """
    df = train_df[['Date', 'Revenue']].copy()
    
    print(f"\nFeature Engineering (Vectorized Training):")
    
    # =====================================================================
    # GROUP 1: CALENDAR FEATURES (8)
    # =====================================================================
    df = add_calendar_features(df)
    print(f"  ✓ Calendar features: 8")
    
    # =====================================================================
    # GROUP 2: REVENUE LAG FEATURES (11)
    # =====================================================================
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)
    print(f"  ✓ Revenue lag features: {len(LAG_WINDOWS)}")
    
    # =====================================================================
    # GROUP 3: ROLLING STATISTICS (25 = 5 windows × 5 stats)
    # =====================================================================
    # CONSISTENCY: ddof=1 for unbiased std computation
    # shift(1) excludes current row in the window
    for win in ROLLING_WINDOWS:
        shifted_rev = df['Revenue'].shift(1)
        df[f'roll_mean_{win}'] = shifted_rev.rolling(window=win).mean()
        df[f'roll_std_{win}'] = shifted_rev.rolling(window=win).std(ddof=1)  # CONSISTENCY: ddof=1
        df[f'roll_min_{win}'] = shifted_rev.rolling(window=win).min()
        df[f'roll_max_{win}'] = shifted_rev.rolling(window=win).max()
        df[f'roll_median_{win}'] = shifted_rev.rolling(window=win).median()
    print(f"  ✓ Rolling statistics: {5 * len(ROLLING_WINDOWS)}")
    
    # =====================================================================
    # GROUP 4: EXPANDING STATISTICS (2)
    # =====================================================================
    df['expanding_mean'] = df['Revenue'].shift(1).expanding().mean()
    df['expanding_std'] = df['Revenue'].shift(1).expanding().std(ddof=1)  # CONSISTENCY: ddof=1
    print(f"  ✓ Expanding statistics: 2")
    
    # =====================================================================
    # GROUP 5: TREND / RATIO FEATURES (4)
    # =====================================================================
    mean_7 = df['Revenue'].shift(1).rolling(window=7).mean()
    mean_30 = df['Revenue'].shift(1).rolling(window=30).mean()
    mean_90 = df['Revenue'].shift(1).rolling(window=90).mean()
    
    df['mean_7_minus_mean_30'] = mean_7 - mean_30
    df['mean_7_over_mean_30'] = mean_7 / (mean_30 + 1e-6)
    df['mean_30_over_mean_90'] = mean_30 / (mean_90 + 1e-6)
    df['volatility_ratio'] = df['Revenue'].shift(1).rolling(window=30).std(ddof=1) / (mean_30 + 1e-6)  # CONSISTENCY: ddof=1
    print(f"  ✓ Trend/ratio features: 4")
    
    # =====================================================================
    # GROUP 6: ACTIVITY / SPARSITY FEATURES (3)
    # =====================================================================
    revenue_positive = (df['Revenue'] > 0).astype(int)
    df['count_positive_7'] = revenue_positive.shift(1).rolling(window=7).sum()
    df['count_positive_30'] = revenue_positive.shift(1).rolling(window=30).sum()
    
    # Days since last nonzero
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
    print(f"  ✓ Activity/sparsity features: 3")
    
    # =====================================================================
    # GROUP 7: SEASONAL HISTORY FEATURES (2)
    # =====================================================================
    # Average revenue by dayofweek (computed from all past data at each point)
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
    print(f"  ✓ Seasonal history features: 2")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    
    # Drop burn-in rows
    df = df.iloc[burnin:].reset_index(drop=True)
    
    # Get feature names
    feature_names = [c for c in df.columns if c not in ['Date', 'Revenue']]
    print(f"\n  Total features: {len(feature_names)}")
    print(f"  Breakdown: 8 calendar + 11 lags + 25 rolling + 2 expanding + 4 trend + 3 activity + 2 seasonal = 55")
    
    # Extract X, y
    X = df[feature_names].fillna(0).values
    y = df['Revenue'].values
    
    return X, y, feature_names


# ============================================================================
# 3. ROW-BY-ROW FEATURE ENGINEERING FOR RECURSIVE INFERENCE
# ============================================================================

def engineer_features_for_row(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the LAST row of history_df.
    Used during recursive validation and test-time prediction.
    
    CONSISTENCY GUARANTEE:
    - roll_std uses np.std(..., ddof=1) to match training's ddof=1
    - Window behavior matches training: shift(1) logic respected
    """
    df = history_df.copy()
    df = add_calendar_features(df)
    
    idx = len(df) - 1  # Last row
    
    # =====================================================================
    # CALENDAR: Already added above
    # =====================================================================
    
    # =====================================================================
    # LAGS
    # =====================================================================
    for lag in LAG_WINDOWS:
        if idx >= lag:
            df.loc[idx, f'lag_{lag}'] = df.loc[idx - lag, 'Revenue']
        else:
            df.loc[idx, f'lag_{lag}'] = np.nan
    
    # =====================================================================
    # ROLLING STATISTICS (25)
    # =====================================================================
    # CONSISTENCY: Use np.std(..., ddof=1) to match training's ddof=1
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{win}'] = past.mean()
            df.loc[idx, f'roll_std_{win}'] = np.std(past, ddof=1) if len(past) > 1 else np.nan
            df.loc[idx, f'roll_min_{win}'] = past.min()
            df.loc[idx, f'roll_max_{win}'] = past.max()
            df.loc[idx, f'roll_median_{win}'] = np.median(past)
        else:
            df.loc[idx, f'roll_mean_{win}'] = np.nan
            df.loc[idx, f'roll_std_{win}'] = np.nan
            df.loc[idx, f'roll_min_{win}'] = np.nan
            df.loc[idx, f'roll_max_{win}'] = np.nan
            df.loc[idx, f'roll_median_{win}'] = np.nan
    
    # =====================================================================
    # EXPANDING (2)
    # =====================================================================
    past_all = df.loc[0:idx - 1, 'Revenue'].values
    if len(past_all) > 0:
        df.loc[idx, 'expanding_mean'] = past_all.mean()
        df.loc[idx, 'expanding_std'] = np.std(past_all, ddof=1) if len(past_all) > 1 else np.nan
    else:
        df.loc[idx, 'expanding_mean'] = np.nan
        df.loc[idx, 'expanding_std'] = np.nan
    
    # =====================================================================
    # TREND / RATIO (4)
    # =====================================================================
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
        df.loc[idx, 'volatility_ratio'] = vol_30 / (m30 + 1e-6) if not np.isnan(vol_30) and m30 > 0 else np.nan
    else:
        df.loc[idx, 'volatility_ratio'] = np.nan
    
    # =====================================================================
    # ACTIVITY (3)
    # =====================================================================
    if idx >= 7:
        past_7 = df.loc[max(0, idx - 7):idx - 1, 'Revenue'].values
        df.loc[idx, 'count_positive_7'] = (past_7 > 0).sum()
    else:
        df.loc[idx, 'count_positive_7'] = np.nan
    
    if idx >= 30:
        past_30 = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].values
        df.loc[idx, 'count_positive_30'] = (past_30 > 0).sum()
    else:
        df.loc[idx, 'count_positive_30'] = np.nan
    
    # Days since nonzero
    if idx > 0:
        for j in range(idx - 1, -1, -1):
            if df.loc[j, 'Revenue'] > 0:
                df.loc[idx, 'days_since_nonzero'] = idx - j
                break
        else:
            df.loc[idx, 'days_since_nonzero'] = np.nan
    else:
        df.loc[idx, 'days_since_nonzero'] = np.nan
    
    # =====================================================================
    # SEASONAL HISTORY (2)
    # =====================================================================
    current_dow = df.loc[idx, 'dayofweek']
    current_month = df.loc[idx, 'month']
    
    # Average revenue by dayofweek from prior dates
    mask_dow = df.loc[0:idx - 1, 'dayofweek'] == current_dow
    if mask_dow.any():
        df.loc[idx, 'avg_revenue_by_dayofweek'] = df.loc[0:idx - 1][mask_dow]['Revenue'].mean()
    else:
        df.loc[idx, 'avg_revenue_by_dayofweek'] = np.nan
    
    # Average revenue by month from prior dates
    mask_month = df.loc[0:idx - 1, 'month'] == current_month
    if mask_month.any():
        df.loc[idx, 'avg_revenue_by_month'] = df.loc[0:idx - 1][mask_month]['Revenue'].mean()
    else:
        df.loc[idx, 'avg_revenue_by_month'] = np.nan
    
    return df


# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

def train_model(X: np.ndarray, y: np.ndarray, verbose: bool = False) -> object:
    """Train LightGBM model (or fallback)."""
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
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            random_state=RANDOM_STATE
        )
    
    model.fit(X, y)
    return model


# ============================================================================
# 5. ROLLING-ORIGIN BACKTEST
# ============================================================================

def rolling_origin_backtest(train_df: pd.DataFrame) -> Dict:
    """
    Horizon-aligned rolling-origin backtest (3 folds).
    
    Folds:
    - Fold 1: Train to 2019-12-31, Validate 2020 (366 days)
    - Fold 2: Train to 2020-12-31, Validate 2021 (365 days)
    - Fold 3: Train to 2021-12-31, Validate 2022 (365 days)
    
    Returns:
        dict with 'folds' list and 'mean_mae'
    """
    print(f"\n{'='*80}")
    print("ROLLING-ORIGIN BACKTEST (3 FOLDS, HORIZON-ALIGNED)")
    print(f"{'='*80}")
    
    # Define fold boundaries
    folds_def = [
        {'train_end': '2019-12-31', 'val_start': '2020-01-01', 'val_end': '2020-12-31', 'label': 'Fold 1 (2020)'},
        {'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2021-12-31', 'label': 'Fold 2 (2021)'},
        {'train_end': '2021-12-31', 'val_start': '2022-01-01', 'val_end': '2022-12-31', 'label': 'Fold 3 (2022)'},
    ]
    
    results = []
    
    for fold_def in folds_def:
        train_cutoff = pd.Timestamp(fold_def['train_end'])
        val_start = pd.Timestamp(fold_def['val_start'])
        val_end = pd.Timestamp(fold_def['val_end'])
        
        train_fold = train_df[train_df['Date'] <= train_cutoff].copy()
        val_fold = train_df[(train_df['Date'] >= val_start) & (train_df['Date'] <= val_end)].copy()
        
        if len(val_fold) < 100:
            print(f"\n{fold_def['label']}: Skipped (insufficient validation rows: {len(val_fold)})")
            continue
        
        print(f"\n{fold_def['label']}:")
        print(f"  Train: {train_fold['Date'].min().date()} to {train_fold['Date'].max().date()} ({len(train_fold):,} rows)")
        print(f"  Val:   {val_fold['Date'].min().date()} to {val_fold['Date'].max().date()} ({len(val_fold):,} rows)")
        
        # Prepare training data
        X_train, y_train, feature_names = prepare_train_features_vectorized(train_fold, burnin=30)
        print(f"  Training: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        
        if X_train.shape[0] < 100:
            print(f"  Skipped (insufficient training rows after burnin)")
            continue
        
        # Train model
        model = train_model(X_train, y_train, verbose=False)
        
        # Recursive validation
        history = train_fold[['Date', 'Revenue']].copy()
        preds = []
        actuals = []
        
        for val_idx in range(len(val_fold)):
            val_date = val_fold.iloc[val_idx]['Date']
            val_revenue = val_fold.iloc[val_idx]['Revenue']
            
            # Add placeholder row
            new_row = pd.DataFrame({
                'Date': [val_date],
                'Revenue': [np.nan]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Engineer features for last row
            history = engineer_features_for_row(history)
            
            # Predict
            X_last = history.iloc[-1:][feature_names].fillna(0).values
            y_pred = model.predict(X_last)[0]
            y_pred = max(0, y_pred)  # Clip to non-negative
            
            preds.append(y_pred)
            actuals.append(val_revenue)
            
            # Update history with prediction
            history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        # Compute metrics
        preds = np.array(preds)
        actuals = np.array(actuals)
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        
        print(f"  MAE:  {mae:,.0f}")
        print(f"  RMSE: {rmse:,.0f}")
        
        results.append({'fold': fold_def['label'], 'MAE': mae, 'RMSE': rmse})
    
    # Summary
    mean_mae = np.mean([r['MAE'] for r in results]) if results else np.inf
    print(f"\n{'='*80}")
    print(f"Backtest Summary: Mean MAE = {mean_mae:,.0f}")
    print(f"{'='*80}")
    
    return {'folds': results, 'mean_mae': mean_mae}


# ============================================================================
# 6. RECURSIVE TEST PREDICTION
# ============================================================================

def recursive_predict_test(train_df: pd.DataFrame, test_df: pd.DataFrame, model: object, feature_names: List[str]) -> np.ndarray:
    """
    Generate recursive predictions for test period.
    """
    print(f"\nRecursive Test Prediction ({len(test_df)} rows)...")
    
    history = train_df[['Date', 'Revenue']].copy()
    preds = []
    
    for test_idx in range(len(test_df)):
        test_date = test_df.iloc[test_idx]['Date']
        
        # Add placeholder
        new_row = pd.DataFrame({
            'Date': [test_date],
            'Revenue': [np.nan]
        })
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Engineer features
        history = engineer_features_for_row(history)
        
        # Predict
        X_last = history.iloc[-1:][feature_names].fillna(0).values
        y_pred = model.predict(X_last)[0]
        y_pred = max(0, y_pred)
        
        preds.append(y_pred)
        
        # Update history
        history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        if (test_idx + 1) % 100 == 0:
            print(f"  Predicted {test_idx + 1}/{len(test_df)}")
    
    preds = np.array(preds)
    print(f"Test predictions: min={preds.min():,.0f}, max={preds.max():,.0f}, mean={preds.mean():,.0f}")
    
    return preds


# ============================================================================
# 7. SAVE SUBMISSION
# ============================================================================

def save_submission(test_df: pd.DataFrame, predictions: np.ndarray) -> str:
    """Save submission file."""
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values
    })
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'submission_no_cogs_v2.csv')
    submission.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved: {output_file}")
    print(f"  Shape: {submission.shape}")
    print(f"  Nulls: {submission.isnull().sum().sum()}")
    print(f"  Revenue: min={submission['Revenue'].min():,.0f}, max={submission['Revenue'].max():,.0f}, mean={submission['Revenue'].mean():,.0f}")
    
    return output_file


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    start_time = datetime.now()
    
    # Load data
    train, test = load_data()
    
    # Backtest (3-fold horizon-aligned)
    print(f"\n{'='*80}")
    print("PHASE 1: BACKTEST (3 FOLDS, HORIZON-ALIGNED)")
    print(f"{'='*80}")
    backtest_result = rolling_origin_backtest(train)
    
    # Final training on full dataset
    print(f"\n{'='*80}")
    print("PHASE 2: FINAL TRAINING ON FULL DATA")
    print(f"{'='*80}")
    X_full, y_full, feature_names = prepare_train_features_vectorized(train, burnin=30)
    print(f"Training data: {X_full.shape[0]:,} rows × {X_full.shape[1]} features")
    print(f"Target: {len(y_full):,} values")
    
    model_final = train_model(X_full, y_full, verbose=False)
    print(f"✓ Model trained on {X_full.shape[0]:,} rows")
    
    # Test prediction (recursive)
    print(f"\n{'='*80}")
    print("PHASE 3: TEST PREDICTION (RECURSIVE)")
    print(f"{'='*80}")
    test_preds = recursive_predict_test(train, test, model_final, feature_names)
    
    # Save submission
    print(f"\n{'='*80}")
    print("PHASE 4: SAVE SUBMISSION")
    print(f"{'='*80}")
    output_file = save_submission(test, test_preds)
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"\nBacktest Results (3 Folds):")
    for fold in backtest_result['folds']:
        print(f"  {fold['fold']}: MAE = {fold['MAE']:,.0f}")
    print(f"  Mean MAE: {backtest_result['mean_mae']:,.0f}")
    
    print(f"\nFeature Summary:")
    print(f"  Total features: {len(feature_names)}")
    print(f"    - Calendar/Seasonality: 8")
    print(f"    - Revenue Lags: {len(LAG_WINDOWS)}")
    print(f"    - Rolling Statistics: 25 (5 windows × 5 stats)")
    print(f"    - Expanding Statistics: 2")
    print(f"    - Trend/Ratio Features: 4")
    print(f"    - Activity/Sparsity: 3")
    print(f"    - Seasonal History: 2")
    
    print(f"\nModel:")
    print(f"  Type: LightGBM")
    print(f"  Training rows (after burnin): {X_full.shape[0]:,}")
    print(f"  Learning rate: 0.05")
    print(f"  Num leaves: 31")
    print(f"  Estimators: 300")
    
    print(f"\nTest Predictions:")
    print(f"  Rows: {len(test_preds)}")
    print(f"  Mean: {test_preds.mean():,.0f}")
    print(f"  Min: {test_preds.min():,.0f}")
    print(f"  Max: {test_preds.max():,.0f}")
    
    print(f"\nSubmission File:")
    print(f"  {output_file}")
    print(f"  Format: Date, Revenue, COGS")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nExecution time: {elapsed:.1f} seconds")
    print(f"\n✓ Complete! Ready for Kaggle submission.")


if __name__ == '__main__':
    main()

"""
Recursive Single-Table Baseline Forecasting Model (FIXED)
==========================================================

PATCH: Fixed Critical Leakage in Feature Generation
- OLD: Concatenated train+test, computed lags using future Revenue placeholders
- NEW: Recursive forecasting - predict one day at a time, write prediction back to history

Objective: Forecast daily Revenue using only sales.csv (no multi-table joins).

Strategy:
  1. Load sales.csv (3833 daily rows, 2012-07-04 to 2022-12-31)
  2. Create time-safe features: calendar, lags, rolling stats, expanding means
  3. Validate using rolling-origin backtesting (not single holdout)
     - For each fold, predict future dates recursively
     - Use true history + previous predictions for lag features
  4. Report fold-by-fold MAE
  5. Retrain on full 2012-2022 history
  6. Predict test (2023-2024) recursively with LightGBM
  7. Export submission in required format

Model: LightGBM Regressor (or XGBoost as fallback)
- Reason: Faster, better tuned for time-series than GradientBoosting

Metric: MAE on Revenue

Time leakage prevention:
  - All lag features built from history + previous predictions ONLY
  - No future unknown Revenue values used in feature generation
  - Validation procedure matches test procedure exactly (recursive)
  - COGS NOT used (not provably known in advance)
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Tuple, Dict, List

# Modeling
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_DIR = 'outputs'
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission_recursive_single_table.csv')
BACKTEST_FILE = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
AUDIT_FILE = os.path.join(OUTPUT_DIR, 'audit_report_recursive.txt')

# Feature engineering parameters
LAG_WINDOWS = [1, 2, 3, 7, 14, 30, 90]
ROLLING_WINDOWS = [7, 14, 30, 90]

# Model hyperparameters
RANDOM_STATE = 42

# Choose model
MODEL_TYPE = 'lgbm'  # Options: 'lgbm', 'xgb', 'gbm'

# LightGBM parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': RANDOM_STATE,
    'verbose': -1,
}

# ============================================================================
# 1. Data Loading & Inspection
# ============================================================================

def load_and_inspect_data(train_path, test_path):
    """Load and inspect train and test files."""
    print("\n" + "="*70)
    print("1. DATA LOADING & INSPECTION")
    print("="*70)
    
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date'])
    
    print(f"\nTrain: {train.shape[0]} rows, {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Test:  {test.shape[0]} rows, {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"\nTrain Revenue: mean={train['Revenue'].mean():,.0f}, std={train['Revenue'].std():,.0f}")
    print(f"Train Revenue: min={train['Revenue'].min():,.0f}, max={train['Revenue'].max():,.0f}")
    
    return train.sort_values('Date').reset_index(drop=True), test.sort_values('Date').reset_index(drop=True)


# ============================================================================
# 2. Feature Engineering (Time-Safe)
# ============================================================================

def create_calendar_features(df):
    """Create calendar features from date column."""
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['week'] = df['Date'].dt.isocalendar().week
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['day'] = df['Date'].dt.day
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    return df


def add_lag_features_to_row(history_df, target_col='Revenue', windows=None):
    """
    Add lag features to the most recent row of history_df based on its history.
    Returns the last row with lag features computed from past rows.
    """
    if windows is None:
        windows = LAG_WINDOWS
    
    df = history_df.copy()
    last_idx = len(df) - 1
    
    for lag in windows:
        if last_idx >= lag:
            df.loc[last_idx, f'lag_{lag}_{target_col}'] = df.loc[last_idx - lag, target_col]
        else:
            df.loc[last_idx, f'lag_{lag}_{target_col}'] = np.nan
    
    return df


def add_rolling_features_to_row(history_df, target_col='Revenue', windows=None):
    """
    Add rolling window features to the most recent row of history_df.
    Shifted by 1 day to avoid future leakage.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    
    df = history_df.copy()
    last_idx = len(df) - 1
    
    for win in windows:
        if last_idx >= win:
            past_values = df.loc[max(0, last_idx - win + 1):last_idx - 1, target_col].values
            if len(past_values) > 0:
                df.loc[last_idx, f'rolling_mean_{win}_{target_col}'] = past_values.mean()
                df.loc[last_idx, f'rolling_std_{win}_{target_col}'] = past_values.std()
                df.loc[last_idx, f'rolling_min_{win}_{target_col}'] = past_values.min()
                df.loc[last_idx, f'rolling_max_{win}_{target_col}'] = past_values.max()
                df.loc[last_idx, f'rolling_median_{win}_{target_col}'] = np.median(past_values)
            else:
                df.loc[last_idx, f'rolling_mean_{win}_{target_col}'] = np.nan
                df.loc[last_idx, f'rolling_std_{win}_{target_col}'] = np.nan
                df.loc[last_idx, f'rolling_min_{win}_{target_col}'] = np.nan
                df.loc[last_idx, f'rolling_max_{win}_{target_col}'] = np.nan
                df.loc[last_idx, f'rolling_median_{win}_{target_col}'] = np.nan
        else:
            for stat in ['mean', 'std', 'min', 'max', 'median']:
                df.loc[last_idx, f'rolling_{stat}_{win}_{target_col}'] = np.nan
    
    return df


def add_expanding_features_to_row(history_df, target_col='Revenue'):
    """Add expanding window features to the most recent row."""
    df = history_df.copy()
    last_idx = len(df) - 1
    
    if last_idx >= 1:
        past_values = df.loc[0:last_idx - 1, target_col].values
        if len(past_values) > 0:
            df.loc[last_idx, f'expanding_mean_{target_col}'] = past_values.mean()
            df.loc[last_idx, f'expanding_median_{target_col}'] = np.median(past_values)
            df.loc[last_idx, f'expanding_std_{target_col}'] = past_values.std()
        else:
            df.loc[last_idx, f'expanding_mean_{target_col}'] = np.nan
            df.loc[last_idx, f'expanding_median_{target_col}'] = np.nan
            df.loc[last_idx, f'expanding_std_{target_col}'] = np.nan
    else:
        df.loc[last_idx, f'expanding_mean_{target_col}'] = np.nan
        df.loc[last_idx, f'expanding_median_{target_col}'] = np.nan
        df.loc[last_idx, f'expanding_std_{target_col}'] = np.nan
    
    return df


def add_trend_features_to_row(history_df, target_col='Revenue'):
    """Add trend features to the most recent row."""
    df = history_df.copy()
    last_idx = len(df) - 1
    
    if last_idx >= 90:
        past_7 = df.loc[max(0, last_idx - 6):last_idx - 1, target_col].mean()
        past_30 = df.loc[max(0, last_idx - 29):last_idx - 1, target_col].mean()
        past_90 = df.loc[max(0, last_idx - 89):last_idx - 1, target_col].mean()
        
        df.loc[last_idx, 'momentum_7_30'] = past_30 - past_7
        df.loc[last_idx, 'trend_30_90_ratio'] = past_30 / (past_90 + 1e-6)
    else:
        df.loc[last_idx, 'momentum_7_30'] = np.nan
        df.loc[last_idx, 'trend_30_90_ratio'] = np.nan
    
    # Active days
    if last_idx >= 29:
        active = (df.loc[max(0, last_idx - 29):last_idx - 1, target_col] > 0).sum()
        df.loc[last_idx, 'active_days_30'] = active
    else:
        df.loc[last_idx, 'active_days_30'] = np.nan
    
    return df


def engineer_features_for_row(history_df, target_col='Revenue'):
    """
    Engineer all features for the most recent row using ONLY historical data.
    This is the core recursive feature function.
    """
    df = history_df.copy()
    df = create_calendar_features(df)
    df = add_lag_features_to_row(df, target_col=target_col)
    df = add_rolling_features_to_row(df, target_col=target_col)
    df = add_expanding_features_to_row(df, target_col=target_col)
    df = add_trend_features_to_row(df, target_col=target_col)
    return df


def get_feature_columns():
    """Get list of all feature column names."""
    features = [
        'year', 'month', 'quarter', 'week', 'dayofweek', 'day', 'dayofyear',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
    ]
    features += [f'lag_{w}_Revenue' for w in LAG_WINDOWS]
    for w in ROLLING_WINDOWS:
        features += [f'rolling_{stat}_{w}_Revenue' for stat in ['mean', 'std', 'min', 'max', 'median']]
    features += ['expanding_mean_Revenue', 'expanding_median_Revenue', 'expanding_std_Revenue']
    features += ['momentum_7_30', 'trend_30_90_ratio', 'active_days_30']
    return features


# ============================================================================
# 3. Training Data Preparation
# ============================================================================

def prepare_training_data(train_df, target_col='Revenue', burnin_days=90):
    """
    Prepare training data with engineered features.
    First burnin_days rows dropped due to NaN in lag features.
    """
    print(f"\nPreparing training data with {burnin_days}-day burn-in...")
    
    # Engineer all rows
    df = train_df.copy()
    for col in get_feature_columns():
        df[col] = np.nan
    
    for idx in range(len(df)):
        if idx > 0:
            df = engineer_features_for_row(df.iloc[:idx + 1], target_col=target_col)
    
    # Drop burn-in
    df = df.iloc[burnin_days:].reset_index(drop=True)
    
    # Extract X, y
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    dates = df['Date'].copy()
    
    print(f"Training data shape: {X.shape}")
    print(f"Non-null features: {X.notna().sum().sum()} / {X.size}")
    
    return X, y, dates, df, feature_cols


# ============================================================================
# 4. Model Training
# ============================================================================

def train_model(X, y, model_type='lgbm'):
    """Train LightGBM or XGBoost model."""
    print(f"\nTraining {model_type.upper()} model...")
    
    if model_type == 'lgbm' and HAVE_LGBM:
        model = lgb.LGBMRegressor(**LGBM_PARAMS, n_estimators=300)
        model.fit(X, y, verbose_eval=False)
    elif model_type == 'xgb' and HAVE_XGB:
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        model.fit(X, y)
    else:
        raise ValueError(f"Model {model_type} not available. Ensure lightgbm or xgboost is installed.")
    
    print("Training complete.")
    return model


# ============================================================================
# 5. Rolling-Origin Backtesting
# ============================================================================

def rolling_origin_backtest(train_df, test_periods=4, target_col='Revenue', model_type='lgbm'):
    """
    Perform rolling-origin backtesting with recursive forecasting.
    
    Args:
        train_df: Full training DataFrame with Date and Revenue
        test_periods: Number of periods to holdout for validation
        target_col: Target column name
        model_type: 'lgbm' or 'xgb'
    
    Returns:
        List of dicts with fold results
    """
    print("\n" + "="*70)
    print("5. ROLLING-ORIGIN BACKTESTING")
    print("="*70)
    
    train_df = train_df.sort_values('Date').reset_index(drop=True)
    results = []
    
    # Define folds by quarter
    unique_years = train_df['Date'].dt.year.unique()
    folds = []
    
    # Create folds: roughly equal size test periods
    total_rows = len(train_df)
    fold_size = total_rows // (test_periods + 1)
    
    for i in range(test_periods):
        train_end_idx = fold_size * (i + 1)
        val_end_idx = min(fold_size * (i + 2), total_rows)
        
        if val_end_idx - train_end_idx < 30:
            continue  # Skip if validation fold too small
        
        train_fold = train_df.iloc[:train_end_idx].copy()
        val_fold = train_df.iloc[train_end_idx:val_end_idx].copy()
        
        if len(val_fold) < 5:
            continue
        
        folds.append((train_fold, val_fold))
    
    # Run each fold
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1} ---")
        print(f"Train: {train_fold['Date'].min().date()} to {train_fold['Date'].max().date()} ({len(train_fold)} rows)")
        print(f"Val:   {val_fold['Date'].min().date()} to {val_fold['Date'].max().date()} ({len(val_fold)} rows)")
        
        # Prepare training data
        X_train, y_train, dates_train, train_prepared, feature_cols = prepare_training_data(
            train_fold, target_col=target_col
        )
        
        # Train model
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Recursive validation forecast
        history = train_fold[['Date', target_col]].reset_index(drop=True).copy()
        predictions = []
        actuals = []
        dates_list = []
        
        for val_idx in range(len(val_fold)):
            val_row_date = val_fold.iloc[val_idx]['Date']
            val_row_revenue = val_fold.iloc[val_idx][target_col]
            
            # Engineer features using history (no future data)
            history_with_placeholder = history.copy()
            new_row = pd.DataFrame({
                'Date': [val_row_date],
                target_col: [np.nan]  # Placeholder, will be filled
            })
            history_with_placeholder = pd.concat([history_with_placeholder, new_row], ignore_index=True, sort=False)
            
            # Add features
            history_with_placeholder = engineer_features_for_row(
                history_with_placeholder, target_col=target_col
            )
            
            # Get features for last row
            X_last = history_with_placeholder.iloc[-1:][feature_cols].copy()
            
            # Fill NaNs with 0 (safe for tree models)
            X_last = X_last.fillna(0)
            
            # Predict
            y_pred = model.predict(X_last)[0]
            
            # Store
            predictions.append(y_pred)
            actuals.append(val_row_revenue)
            dates_list.append(val_row_date)
            
            # Write prediction back to history
            new_hist_row = pd.DataFrame({
                'Date': [val_row_date],
                target_col: [y_pred]
            })
            history = pd.concat([history, new_hist_row], ignore_index=True, sort=False)
        
        # Compute metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        fold_mae = mean_absolute_error(actuals, predictions)
        fold_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        fold_r2 = r2_score(actuals, predictions)
        
        print(f"Fold MAE: {fold_mae:,.2f}")
        print(f"Fold RMSE: {fold_rmse:,.2f}")
        print(f"Fold R²: {fold_r2:.4f}")
        
        results.append({
            'fold': fold_idx + 1,
            'train_start': train_fold['Date'].min().date(),
            'train_end': train_fold['Date'].max().date(),
            'val_start': val_fold['Date'].min().date(),
            'val_end': val_fold['Date'].max().date(),
            'val_rows': len(val_fold),
            'MAE': fold_mae,
            'RMSE': fold_rmse,
            'R2': fold_r2,
        })
    
    # Summary
    mean_mae = np.mean([r['MAE'] for r in results])
    std_mae = np.std([r['MAE'] for r in results])
    
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    for r in results:
        print(f"Fold {r['fold']:2d}: MAE = {r['MAE']:>10,.2f}  |  {r['val_start']} to {r['val_end']}")
    print(f"\nMean MAE: {mean_mae:,.2f} (+/- {std_mae:,.2f})")
    
    return results


# ============================================================================
# 6. Recursive Test Prediction
# ============================================================================

def recursive_predict(train_df, test_df, model, feature_cols, target_col='Revenue', model_type='lgbm'):
    """
    Recursively predict test period one day at a time.
    Uses previous predictions as history for lag features.
    """
    print("\n" + "="*70)
    print("6. RECURSIVE TEST PREDICTION")
    print("="*70)
    
    predictions = []
    history = train_df[['Date', target_col]].reset_index(drop=True).copy()
    
    for test_idx in range(len(test_df)):
        test_date = test_df.iloc[test_idx]['Date']
        
        # Create row with placeholder revenue
        new_row = pd.DataFrame({
            'Date': [test_date],
            target_col: [np.nan]
        })
        
        # Add to history
        history_extended = pd.concat([history, new_row], ignore_index=True, sort=False)
        
        # Engineer features using history only
        history_extended = engineer_features_for_row(history_extended, target_col=target_col)
        
        # Get features for last row
        X_test_row = history_extended.iloc[-1:][feature_cols].copy()
        X_test_row = X_test_row.fillna(0)
        
        # Predict
        y_pred = model.predict(X_test_row)[0]
        
        # Ensure non-negative
        y_pred = max(0, y_pred)
        
        predictions.append(y_pred)
        
        # Write prediction back to history
        new_hist_row = pd.DataFrame({
            'Date': [test_date],
            target_col: [y_pred]
        })
        history = pd.concat([history, new_hist_row], ignore_index=True, sort=False)
        
        if (test_idx + 1) % 100 == 0:
            print(f"Predicted {test_idx + 1}/{len(test_df)} days")
    
    print(f"Predictions complete. Range: [{min(predictions):,.0f}, {max(predictions):,.0f}]")
    return np.array(predictions)


# ============================================================================
# 7. Submission
# ============================================================================

def create_submission(test_df, predictions, output_file=None):
    """Create and save submission."""
    print("\n" + "="*70)
    print("7. SUBMISSION GENERATION")
    print("="*70)
    
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values
    })
    
    print(f"\nSubmission shape: {submission.shape}")
    print(f"Revenue range: {submission['Revenue'].min():,.0f} to {submission['Revenue'].max():,.0f}")
    print(f"Nulls: {submission.isnull().sum().sum()}")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        submission.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
    
    return submission


# ============================================================================
# 8. Main Pipeline
# ============================================================================

def main():
    """Execute the full recursive baseline pipeline."""
    print("\n" + "="*70)
    print("RECURSIVE SINGLE-TABLE BASELINE (FIXED LEAKAGE)")
    print("="*70)
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Target: Revenue")
    print(f"Metric: MAE")
    print(f"Strategy: Recursive forecasting with rolling-origin CV")
    
    # 1. Load data
    train, test = load_and_inspect_data(TRAIN_FILE, TEST_FILE)
    
    # 2. Backtesting
    backtest_results = rolling_origin_backtest(train, test_periods=3, model_type=MODEL_TYPE)
    
    # Save backtest results
    backtest_df = pd.DataFrame(backtest_results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backtest_df.to_csv(BACKTEST_FILE, index=False)
    print(f"\nBacktest results saved to: {BACKTEST_FILE}")
    
    # 3. Prepare full training data
    X_full, y_full, dates_full, train_prepared, feature_cols = prepare_training_data(train)
    print(f"\nFull training set: {X_full.shape}")
    
    # 4. Train final model on full history
    print("\n" + "="*70)
    print("4. FINAL MODEL TRAINING")
    print("="*70)
    model_final = train_model(X_full, y_full, model_type=MODEL_TYPE)
    
    # Feature importance
    if hasattr(model_final, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model_final.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 15 features:")
        print(importance_df.head(15).to_string(index=False))
    
    # 5. Recursive prediction on test
    test_predictions = recursive_predict(train, test, model_final, feature_cols, model_type=MODEL_TYPE)
    
    # 6. Create submission
    submission = create_submission(test, test_predictions, output_file=SUBMISSION_FILE)
    
    # 7. Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    mean_backtest_mae = backtest_df['MAE'].mean()
    print(f"\nBacktest mean MAE: {mean_backtest_mae:,.2f}")
    print(f"Test submissions prediction range: [{test_predictions.min():,.0f}, {test_predictions.max():,.0f}]")
    print(f"\nSubmission saved: {SUBMISSION_FILE}")
    
    return {
        'train': train,
        'test': test,
        'model': model_final,
        'backtest_results': backtest_df,
        'submission': submission,
        'feature_cols': feature_cols,
    }


if __name__ == '__main__':
    results = main()
    print("\n✓ Pipeline complete.")

"""
Recursive Single-Table Baseline (SIMPLIFIED & FIXED)
====================================================

CRITICAL PATCH:
  - Removed leakage from future Revenue placeholders
  - Implemented true recursive forecasting (predict one day, update history)
  - Validation matches test-time procedure exactly
  - Clean, testable implementation

Forecasting Strategy:
  1. Train LightGBM on historical lags (2012-2021)
  2. Validate with expanding-window backtesting:
     - For each checkpoint, predict forward recursively
     - Use true history + predicted values for lag features
  3. Retrain on full 2012-2022
  4. Predict 2023-2024 recursively
  5. Export submission

Time Leakage Prevention:
  - All lags computed from history ONLY
  - No future placeholders used in feature generation
  - Validation procedure matches test procedure
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple

try:
    import lightgbm as lgb
    MODEL_LGBM = lgb.LGBMRegressor
except:
    MODEL_LGBM = None

try:
    import xgboost as xgb
    MODEL_XGB = xgb.XGBRegressor
except:
    MODEL_XGB = None

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_DIR = 'outputs'

LAG_WINDOWS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]
RANDOM_STATE = 42

print(f"\n{'='*70}")
print("RECURSIVE SINGLE-TABLE BASELINE (FIXED LEAKAGE)")
print(f"{'='*70}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data():
    train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    
    print(f"\nTrain: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")
    print(f"Revenue range: {train['Revenue'].min():,.0f} to {train['Revenue'].max():,.0f}")
    
    return train, test

# ============================================================================
# 2. FEATURE ENGINEERING (TIME-SAFE)
# ============================================================================

def add_calendar_features(df):
    """Add calendar features."""
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    return df

def engineer_features(history_df):
    """
    Engineer features for the last row of history_df.
    Uses ONLY past rows (no future data).
    Modifies history_df in place, returns it.
    """
    df = history_df.copy()
    df = add_calendar_features(df)
    
    n = len(df)
    idx = n - 1  # Last row
    
    # Lags
    for lag in LAG_WINDOWS:
        if idx >= lag:
            df.loc[idx, f'lag_{lag}'] = df.loc[idx - lag, 'Revenue']
        else:
            df.loc[idx, f'lag_{lag}'] = np.nan
    
    # Rolling statistics (shifted to avoid current value)
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{win}'] = past.mean()
            df.loc[idx, f'roll_std_{win}'] = past.std()
            df.loc[idx, f'roll_min_{win}'] = past.min()
            df.loc[idx, f'roll_max_{win}'] = past.max()
        else:
            df.loc[idx, f'roll_mean_{win}'] = np.nan
            df.loc[idx, f'roll_std_{win}'] = np.nan
            df.loc[idx, f'roll_min_{win}'] = np.nan
            df.loc[idx, f'roll_max_{win}'] = np.nan
    
    # Trend
    if idx >= 30:
        m7 = df.loc[idx - 7:idx - 1, 'Revenue'].mean()
        m30 = df.loc[idx - 30:idx - 1, 'Revenue'].mean()
        df.loc[idx, 'momentum'] = m30 - m7
    else:
        df.loc[idx, 'momentum'] = np.nan
    
    return df

def get_feature_names():
    """Get list of all feature column names."""
    features = ['year', 'month', 'quarter', 'day', 'dayofweek', 'dayofyear', 'is_month_start', 'is_month_end']
    features += [f'lag_{w}' for w in LAG_WINDOWS]
    features += [f'roll_mean_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_std_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_min_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_max_{w}' for w in ROLLING_WINDOWS]
    features += ['momentum']
    return features

# ============================================================================
# 3. PREPARE TRAINING DATA
# ============================================================================

def prepare_train_data(train_df, burnin=30):
    """Prepare training features. Drop first burnin rows due to NaN lags."""
    df = train_df[['Date', 'Revenue']].copy()
    
    # Engineer features for each row
    for idx in range(len(df)):
        df = engineer_features(df)
    
    # Drop burn-in
    df = df.iloc[burnin:].reset_index(drop=True)
    
    feature_names = get_feature_names()
    X = df[feature_names].fillna(0)
    y = df['Revenue'].values
    
    print(f"\nTrain features shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    
    return X, y, feature_names

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

def train_model(X, y, model_type='lgbm'):
    """Train LightGBM or XGBoost."""
    print(f"\nTraining {model_type.upper()}...")
    
    if model_type == 'lgbm' and MODEL_LGBM:
        model = MODEL_LGBM(
            objective='regression',
            metric='mae',
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=300,
            random_state=RANDOM_STATE,
            verbose=-1
        )
    elif model_type == 'xgb' and MODEL_XGB:
        model = MODEL_XGB(
            objective='reg:absoluteerror',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            random_state=RANDOM_STATE,
            verbosity=0
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
# 5. BACKTESTING
# ============================================================================

def backtest_recursive(train_df, feature_names, num_folds=3):
    """Rolling-origin backtesting with recursive forecasting."""
    print(f"\n{'='*70}")
    print("ROLLING-ORIGIN BACKTESTING")
    print(f"{'='*70}")
    
    results = []
    fold_size = len(train_df) // (num_folds + 1)
    
    for fold in range(num_folds):
        train_idx = (fold + 1) * fold_size
        test_start_idx = train_idx
        test_end_idx = min((fold + 2) * fold_size, len(train_df))
        
        train_fold = train_df.iloc[:train_idx].copy()
        val_fold = train_df.iloc[test_start_idx:test_end_idx].copy()
        
        if len(val_fold) < 10:
            continue
        
        print(f"\nFold {fold + 1}: Train to {train_fold['Date'].max().date()}, Val {val_fold['Date'].min().date()}-{val_fold['Date'].max().date()}")
        
        # Train on this fold
        X_train, y_train, _ = prepare_train_data(train_fold)
        model = train_model(X_train, y_train, model_type='lgbm')
        
        # Recursive validation
        history = train_fold[['Date', 'Revenue']].copy()
        preds = []
        actuals = []
        
        for idx in range(len(val_fold)):
            val_date = val_fold.iloc[idx]['Date']
            val_revenue = val_fold.iloc[idx]['Revenue']
            
            # Add placeholder row to history
            new_row = pd.DataFrame({
                'Date': [val_date],
                'Revenue': [np.nan]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Engineer features for last row
            history = engineer_features(history)
            
            # Get features
            X_last = history.iloc[-1:][feature_names].fillna(0)
            
            # Predict
            y_pred = model.predict(X_last)[0]
            y_pred = max(0, y_pred)  # Non-negative
            
            preds.append(y_pred)
            actuals.append(val_revenue)
            
            # Update history with prediction
            history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        # Metrics
        preds = np.array(preds)
        actuals = np.array(actuals)
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        
        print(f"  MAE: {mae:,.0f}, RMSE: {rmse:,.0f}")
        results.append({'fold': fold + 1, 'MAE': mae, 'RMSE': rmse})
    
    mean_mae = np.mean([r['MAE'] for r in results])
    print(f"\nBacktest Mean MAE: {mean_mae:,.0f}")
    
    return results

# ============================================================================
# 6. RECURSIVE PREDICTION
# ============================================================================

def recursive_predict_test(train_df, test_df, model, feature_names):
    """Predict test period recursively."""
    print(f"\n{'='*70}")
    print("RECURSIVE TEST PREDICTION")
    print(f"{'='*70}")
    
    history = train_df[['Date', 'Revenue']].copy()
    preds = []
    
    for idx in range(len(test_df)):
        test_date = test_df.iloc[idx]['Date']
        
        # Add placeholder
        new_row = pd.DataFrame({
            'Date': [test_date],
            'Revenue': [np.nan]
        })
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Engineer features
        history = engineer_features(history)
        
        # Predict
        X_last = history.iloc[-1:][feature_names].fillna(0)
        y_pred = model.predict(X_last)[0]
        y_pred = max(0, y_pred)
        
        preds.append(y_pred)
        
        # Update
        history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        if (idx + 1) % 100 == 0:
            print(f"  Predicted {idx + 1}/{len(test_df)}")
    
    preds = np.array(preds)
    print(f"Predictions: min={preds.min():,.0f}, max={preds.max():,.0f}, mean={preds.mean():,.0f}")
    
    return preds

# ============================================================================
# 7. SUBMISSION
# ============================================================================

def save_submission(test_df, predictions):
    """Save submission file."""
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values
    })
    
    output_file = os.path.join(OUTPUT_DIR, 'submission_recursive_fixed.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved: {output_file}")
    print(f"Shape: {submission.shape}")
    print(f"Nulls: {submission.isnull().sum().sum()}")
    
    return submission

# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    # Load
    train, test = load_data()
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Backtest
    print(f"\n{'='*70}")
    print("PHASE 1: BACKTESTING")
    print(f"{'='*70}")
    backtest_results = backtest_recursive(train, feature_names, num_folds=2)
    
    # Prepare full training data
    print(f"\n{'='*70}")
    print("PHASE 2: FINAL TRAINING")
    print(f"{'='*70}")
    X_full, y_full, _ = prepare_train_data(train)
    
    # Train final model
    model_final = train_model(X_full, y_full, model_type='lgbm')
    
    # Predict test
    print(f"\n{'='*70}")
    print("PHASE 3: TEST PREDICTION")
    print(f"{'='*70}")
    test_preds = recursive_predict_test(train, test, model_final, feature_names)
    
    # Save submission
    print(f"\n{'='*70}")
    print("PHASE 4: SUBMISSION")
    print(f"{'='*70}")
    submission = save_submission(test, test_preds)
    
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

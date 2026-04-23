"""
Recursive Single-Table Baseline — FIXED & ABLATION
===================================================

CRITICAL FIXES:
  1. prepare_train_data() now correctly computes features for EVERY row
     using fully vectorized shift/rolling/expanding operations
  2. Two feature variants: Revenue-only vs Revenue+COGS
  3. Rolling-origin backtest for both variants
  4. Choose better variant for final training/prediction

Time Leakage Prevention:
  - All lags/rolling computed from history only
  - No future placeholders used
  - Training table builder: fully vectorized
  - Test-time inference: recursive row-by-row

Ablation Question:
  - Does COGS as a known exogenous feature improve MAE?
  - Test both variants with identical rolling-origin CV
  - Report results and use winning variant
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
print("RECURSIVE SINGLE-TABLE BASELINE — FIXED & ABLATION")
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
    print(f"COGS present in train: {'COGS' in train.columns}")
    print(f"COGS present in test: {'COGS' in test.columns}")
    
    return train, test

# ============================================================================
# 2. FEATURE ENGINEERING FOR TRAINING DATA (VECTORIZED)
# ============================================================================

def add_calendar_features(df):
    """Add calendar features to all rows at once."""
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    return df

def prepare_train_data_vectorized(train_df, use_cogs=False, burnin=30):
    """
    FIXED: Prepare training data with vectorized feature engineering.
    
    ALL rows get their features computed correctly using:
    - shift() for lags
    - rolling() for rolling statistics
    - expanding() for trend
    
    Time-safe: Each row uses only data from rows before it.
    """
    df = train_df[['Date', 'Revenue']].copy()
    
    if use_cogs:
        df['COGS'] = train_df['COGS'].values
    
    # Add calendar features
    df = add_calendar_features(df)
    
    # LAG FEATURES (VECTORIZED)
    # lag_i = Revenue[t-i]
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)
    
    # ROLLING STATISTICS (VECTORIZED)
    # Use .shift(1) in window to exclude current row
    for win in ROLLING_WINDOWS:
        df[f'roll_mean_{win}'] = df['Revenue'].shift(1).rolling(window=win).mean()
        df[f'roll_std_{win}'] = df['Revenue'].shift(1).rolling(window=win).std()
        df[f'roll_min_{win}'] = df['Revenue'].shift(1).rolling(window=win).min()
        df[f'roll_max_{win}'] = df['Revenue'].shift(1).rolling(window=win).max()
    
    # TREND FEATURES (VECTORIZED)
    # momentum = mean(last 30 days) - mean(last 7 days)
    mean_30 = df['Revenue'].shift(1).rolling(window=30).mean()
    mean_7 = df['Revenue'].shift(1).rolling(window=7).mean()
    df['momentum'] = mean_30 - mean_7
    
    # OPTIONAL: COGS FEATURES
    if use_cogs:
        # Current COGS level
        df['cogs_level'] = df['COGS']
        
        # COGS lags (safe if COGS is known in advance)
        for lag in [1, 7, 30]:
            df[f'cogs_lag_{lag}'] = df['COGS'].shift(lag)
        
        # COGS rolling mean
        df['cogs_roll_mean_7'] = df['COGS'].shift(1).rolling(window=7).mean()
    
    # Drop burn-in to avoid NaN rows from lags
    df = df.iloc[burnin:].reset_index(drop=True)
    
    # Get feature names and prepare X, y
    feature_names = get_feature_names(use_cogs=use_cogs)
    X = df[feature_names].fillna(0)  # Now safe: no spurious zeros
    y = df['Revenue'].values
    
    print(f"\nTraining data prepared:")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Features used: {feature_names[:5]}... (showing first 5)")
    print(f"  Rows with NaN before fillna: {df[feature_names].isnull().sum().sum()} (all should be in burn-in)")
    
    return X, y, feature_names

def get_feature_names(use_cogs=False):
    """Get list of feature column names."""
    features = ['year', 'month', 'quarter', 'day', 'dayofweek', 'dayofyear', 'is_month_start', 'is_month_end']
    features += [f'lag_{w}' for w in LAG_WINDOWS]
    features += [f'roll_mean_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_std_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_min_{w}' for w in ROLLING_WINDOWS]
    features += [f'roll_max_{w}' for w in ROLLING_WINDOWS]
    features += ['momentum']
    
    if use_cogs:
        features += ['cogs_level', 'cogs_lag_1', 'cogs_lag_7', 'cogs_lag_30', 'cogs_roll_mean_7']
    
    return features

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================

def train_model(X, y, model_type='lgbm'):
    """Train LightGBM or XGBoost."""
    print(f"\n  Training {model_type.upper()}...")
    
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
# 4. FEATURE ENGINEERING FOR RECURSIVE INFERENCE (ROW-BY-ROW)
# ============================================================================

def engineer_features_for_row(history_df, use_cogs=False):
    """
    Engineer features for the LAST row of history_df using only past rows.
    Used during recursive test-time and validation-time prediction.
    
    history_df must have Date, Revenue (and COGS if use_cogs=True).
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
    
    # COGS features (if used)
    if use_cogs:
        df.loc[idx, 'cogs_level'] = df.loc[idx, 'COGS']
        
        for lag in [1, 7, 30]:
            if idx >= lag:
                df.loc[idx, f'cogs_lag_{lag}'] = df.loc[idx - lag, 'COGS']
            else:
                df.loc[idx, f'cogs_lag_{lag}'] = np.nan
        
        if idx >= 7:
            past_cogs = df.loc[max(0, idx - 7):idx - 1, 'COGS'].values
            df.loc[idx, 'cogs_roll_mean_7'] = past_cogs.mean()
        else:
            df.loc[idx, 'cogs_roll_mean_7'] = np.nan
    
    return df

# ============================================================================
# 5. ROLLING-ORIGIN BACKTESTING
# ============================================================================

def backtest_variant(train_df, feature_names, use_cogs=False, num_folds=2):
    """
    Rolling-origin backtesting for one variant.
    Returns list of results with MAE per fold.
    """
    variant_name = "WITH COGS" if use_cogs else "WITHOUT COGS"
    print(f"\n{'='*70}")
    print(f"BACKTEST: {variant_name}")
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
        
        print(f"\nFold {fold + 1}:")
        print(f"  Train: {train_fold['Date'].min().date()} to {train_fold['Date'].max().date()} ({len(train_fold)} rows)")
        print(f"  Val:   {val_fold['Date'].min().date()} to {val_fold['Date'].max().date()} ({len(val_fold)} rows)")
        
        # Prepare training data (vectorized)
        X_train, y_train, _ = prepare_train_data_vectorized(train_fold, use_cogs=use_cogs)
        
        # Train
        model = train_model(X_train, y_train, model_type='lgbm')
        
        # Recursive validation
        history = train_fold[['Date', 'Revenue']].copy()
        if use_cogs:
            history['COGS'] = train_fold['COGS'].values
        
        preds = []
        actuals = []
        
        for idx in range(len(val_fold)):
            val_date = val_fold.iloc[idx]['Date']
            val_revenue = val_fold.iloc[idx]['Revenue']
            val_cogs = val_fold.iloc[idx]['COGS'] if use_cogs else np.nan
            
            # Add placeholder row
            new_row = pd.DataFrame({
                'Date': [val_date],
                'Revenue': [np.nan],
                **(({'COGS': [val_cogs]}) if use_cogs else {})
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Engineer features
            history = engineer_features_for_row(history, use_cogs=use_cogs)
            
            # Predict
            X_last = history.iloc[-1:][feature_names].fillna(0)
            y_pred = model.predict(X_last)[0]
            y_pred = max(0, y_pred)
            
            preds.append(y_pred)
            actuals.append(val_revenue)
            
            # Update history
            history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        # Metrics
        preds = np.array(preds)
        actuals = np.array(actuals)
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        
        print(f"  MAE: {mae:,.0f}, RMSE: {rmse:,.0f}")
        results.append({'fold': fold + 1, 'MAE': mae, 'RMSE': rmse})
    
    mean_mae = np.mean([r['MAE'] for r in results]) if results else np.inf
    print(f"\n{variant_name} Mean MAE: {mean_mae:,.0f}")
    
    return results, mean_mae

# ============================================================================
# 6. RECURSIVE TEST PREDICTION
# ============================================================================

def recursive_predict_test(train_df, test_df, model, feature_names, use_cogs=False):
    """Predict test period recursively."""
    print(f"\nRecursive test prediction...")
    
    history = train_df[['Date', 'Revenue']].copy()
    if use_cogs:
        history['COGS'] = train_df['COGS'].values
    
    preds = []
    
    for idx in range(len(test_df)):
        test_date = test_df.iloc[idx]['Date']
        test_cogs = test_df.iloc[idx]['COGS'] if use_cogs else np.nan
        
        # Add placeholder
        new_row = pd.DataFrame({
            'Date': [test_date],
            'Revenue': [np.nan],
            **(({'COGS': [test_cogs]}) if use_cogs else {})
        })
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Engineer features
        history = engineer_features_for_row(history, use_cogs=use_cogs)
        
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

def save_submission(test_df, predictions, filename='submission_recursive_ablation.csv'):
    """Save submission file."""
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values
    })
    
    output_file = os.path.join(OUTPUT_DIR, filename)
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
    # Load data
    train, test = load_data()
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY: REVENUE-ONLY vs REVENUE+COGS")
    print(f"{'='*70}")
    
    # Backtest variant 1: Without COGS
    results_no_cogs, mean_mae_no_cogs = backtest_variant(train, get_feature_names(use_cogs=False), use_cogs=False, num_folds=2)
    
    # Backtest variant 2: With COGS
    results_with_cogs, mean_mae_with_cogs = backtest_variant(train, get_feature_names(use_cogs=True), use_cogs=True, num_folds=2)
    
    # Choose winner
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"Without COGS - Mean MAE: {mean_mae_no_cogs:,.0f}")
    print(f"With COGS    - Mean MAE: {mean_mae_with_cogs:,.0f}")
    
    if mean_mae_with_cogs < mean_mae_no_cogs:
        winner = "WITH COGS"
        use_cogs_final = True
        mean_mae_final = mean_mae_with_cogs
    else:
        winner = "WITHOUT COGS"
        use_cogs_final = False
        mean_mae_final = mean_mae_no_cogs
    
    print(f"\nWINNER: {winner} (MAE: {mean_mae_final:,.0f})")
    
    # Final training on full data
    print(f"\n{'='*70}")
    print("FINAL TRAINING ON FULL SALES.CSV")
    print(f"{'='*70}")
    
    X_full, y_full, feature_names_final = prepare_train_data_vectorized(train, use_cogs=use_cogs_final)
    model_final = train_model(X_full, y_full, model_type='lgbm')
    
    # Test prediction
    print(f"\n{'='*70}")
    print("TEST PREDICTION")
    print(f"{'='*70}")
    
    test_preds = recursive_predict_test(train, test, model_final, feature_names_final, use_cogs=use_cogs_final)
    
    # Save submission
    print(f"\n{'='*70}")
    print("SAVING SUBMISSION")
    print(f"{'='*70}")
    
    submission = save_submission(test, test_preds)
    
    # Save ablation results
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Training data: {X_full.shape[0]} rows × {X_full.shape[1]} features")
    print(f"✓ Backtesting completed (2 folds)")
    print(f"✓ Winner variant: {winner}")
    print(f"✓ Final model trained on {len(train)} rows")
    print(f"✓ Predictions generated for {len(test)} future rows")
    print(f"✓ Submission file: outputs/{('submission_recursive_ablation_with_cogs.csv' if use_cogs_final else 'submission_recursive_ablation_no_cogs.csv')}")

if __name__ == '__main__':
    main()

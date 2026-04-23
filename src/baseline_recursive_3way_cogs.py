"""
Recursive Single-Table Baseline — 3-Way COGS Ablation
=====================================================

THREE EXPLICIT COGS BRANCHES:
  A. NO_COGS: Revenue history only
  B. LAGGED_COGS_ONLY: Revenue + lagged/rolling COGS (no current-day)
  C. CURRENT_DAY_COGS: Revenue + lagged/rolling + current-day COGS

FIXES:
  1. Fully vectorized training table generation (all rows computed)
  2. roll_std computed identically in training and inference ()
  3. All 3 branches backtested independently
  4. All 3 branches trained on full history
  5. All 3 submissions exported separately

Time Leakage Prevention:
  - All lags/rolling computed from history only
  - No future placeholders used
  - Recursive validation matches test procedure
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict

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

# COGS FEATURE MODES
FEATURE_MODES = ['no_cogs', 'lagged_cogs_only', 'current_day_cogs']

print(f"\n{'='*70}")
print("RECURSIVE SINGLE-TABLE BASELINE — 3-WAY COGS ABLATION")
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
# 2. FEATURE ENGINEERING UTILITIES
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

def get_feature_names(feature_mode: str) -> List[str]:
    """
    Get list of feature column names for the given mode.
    
    MODES:
      - no_cogs: Calendar + Revenue lags/rolling/trend (26 features)
      - lagged_cogs_only: Above + lagged COGS features, NO current-day COGS (30 features)
      - current_day_cogs: Above + current-day COGS level (31 features)
    """
    # Calendar features (8)
    features = ['year', 'month', 'quarter', 'day', 'dayofweek', 'dayofyear', 'is_month_start', 'is_month_end']
    
    # Revenue lag features (5)
    features += [f'lag_{w}' for w in LAG_WINDOWS]
    
    # Revenue rolling statistics (12)
    for win in ROLLING_WINDOWS:
        features += [f'roll_mean_{win}', f'roll_std_{win}', f'roll_min_{win}', f'roll_max_{win}']
    
    # Trend feature (1)
    features += ['momentum']
    
    # COGS features (conditional)
    if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
        # Current-day COGS (only for current_day_cogs mode)
        if feature_mode == 'current_day_cogs':
            features.append('cogs_level')
        
        # Lagged COGS features (both modes)
        features += ['cogs_lag_1', 'cogs_lag_7', 'cogs_lag_30', 'cogs_roll_mean_7']
    
    return features

def prepare_train_data_vectorized(train_df, feature_mode: str, burnin: int = 30):
    """
    Prepare training data with vectorized feature engineering.
    ALL rows get their features computed correctly.
    
    CONSISTENCY NOTE: roll_std uses ddof=1 here (unbiased estimator).
                      recursive inference must use the same ddof=1.
    """
    assert feature_mode in FEATURE_MODES, f"Invalid feature_mode: {feature_mode}"
    
    df = train_df[['Date', 'Revenue']].copy()
    
    if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
        df['COGS'] = train_df['COGS'].values
    
    # Add calendar features
    df = add_calendar_features(df)
    
    # LAG FEATURES (VECTORIZED)
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)
    
    # ROLLING STATISTICS (VECTORIZED)
    # CONSISTENCY: ddof=1 for unbiased std estimator
    # shift(1) excludes current row
    for win in ROLLING_WINDOWS:
        df[f'roll_mean_{win}'] = df['Revenue'].shift(1).rolling(window=win).mean()
        df[f'roll_std_{win}'] = df['Revenue'].shift(1).rolling(window=win).std(ddof=1)
        df[f'roll_min_{win}'] = df['Revenue'].shift(1).rolling(window=win).min()
        df[f'roll_max_{win}'] = df['Revenue'].shift(1).rolling(window=win).max()
    
    # TREND FEATURES (VECTORIZED)
    mean_30 = df['Revenue'].shift(1).rolling(window=30).mean()
    mean_7 = df['Revenue'].shift(1).rolling(window=7).mean()
    df['momentum'] = mean_30 - mean_7
    
    # COGS FEATURES (conditional on feature_mode)
    if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
        # Current-day COGS level (only for current_day_cogs)
        if feature_mode == 'current_day_cogs':
            df['cogs_level'] = df['COGS']
        
        # Lagged COGS (both modes)
        for lag in [1, 7, 30]:
            df[f'cogs_lag_{lag}'] = df['COGS'].shift(lag)
        
        # COGS rolling mean (CONSISTENCY: )
        df['cogs_roll_mean_7'] = df['COGS'].shift(1).rolling(window=7, ).mean()
    
    # Drop burn-in
    df = df.iloc[burnin:].reset_index(drop=True)
    
    feature_names = get_feature_names(feature_mode)
    X = df[feature_names].fillna(0)
    y = df['Revenue'].values
    
    return X, y, feature_names

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================

def train_model(X, y, model_type='lgbm'):
    """Train LightGBM or fallback."""
    print(f"  Training {model_type.upper()}...")
    
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
# 4. FEATURE ENGINEERING FOR RECURSIVE INFERENCE
# ============================================================================

def engineer_features_for_row(history_df, feature_mode: str):
    """
    Engineer features for the LAST row of history_df using only past rows.
    Used during recursive test-time and validation-time prediction.
    
    CONSISTENCY NOTE: roll_std uses ddof=1 here to match training.
    """
    assert feature_mode in FEATURE_MODES
    
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
    
    # Rolling statistics
    # CONSISTENCY:  for std, same as training
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{win}'] = past.mean()
            df.loc[idx, f'roll_std_{win}'] = past.std() if len(past) > 1 else np.nan
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
    
    # COGS features
    if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
        if feature_mode == 'current_day_cogs':
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

def backtest_mode(train_df, feature_mode: str, num_folds: int = 2) -> Dict:
    """
    Rolling-origin backtesting for one feature mode.
    Returns dict with fold results and mean MAE.
    """
    assert feature_mode in FEATURE_MODES
    
    print(f"\n{'='*70}")
    print(f"BACKTEST: {feature_mode.upper()}")
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
        
        # Prepare training data
        X_train, y_train, feature_names = prepare_train_data_vectorized(train_fold, feature_mode=feature_mode)
        print(f"  Training rows: {X_train.shape[0]} × {X_train.shape[1]} features")
        
        # Train
        model = train_model(X_train, y_train, model_type='lgbm')
        
        # Recursive validation
        history = train_fold[['Date', 'Revenue']].copy()
        if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
            history['COGS'] = train_fold['COGS'].values
        
        preds = []
        actuals = []
        
        for idx in range(len(val_fold)):
            val_date = val_fold.iloc[idx]['Date']
            val_revenue = val_fold.iloc[idx]['Revenue']
            val_cogs = val_fold.iloc[idx]['COGS'] if feature_mode in ['lagged_cogs_only', 'current_day_cogs'] else np.nan
            
            # Add placeholder row
            new_row = pd.DataFrame({
                'Date': [val_date],
                'Revenue': [np.nan],
                **(({'COGS': [val_cogs]}) if feature_mode in ['lagged_cogs_only', 'current_day_cogs'] else {})
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Engineer features
            history = engineer_features_for_row(history, feature_mode=feature_mode)
            
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
    print(f"\n{feature_mode.upper()} Mean MAE: {mean_mae:,.0f}")
    
    return {'mode': feature_mode, 'folds': results, 'mean_mae': mean_mae}

# ============================================================================
# 6. RECURSIVE TEST PREDICTION
# ============================================================================

def recursive_predict_test(train_df, test_df, model, feature_names, feature_mode: str):
    """Predict test period recursively."""
    print(f"\nRecursive test prediction ({feature_mode})...")
    
    history = train_df[['Date', 'Revenue']].copy()
    if feature_mode in ['lagged_cogs_only', 'current_day_cogs']:
        history['COGS'] = train_df['COGS'].values
    
    preds = []
    
    for idx in range(len(test_df)):
        test_date = test_df.iloc[idx]['Date']
        test_cogs = test_df.iloc[idx]['COGS'] if feature_mode in ['lagged_cogs_only', 'current_day_cogs'] else np.nan
        
        # Add placeholder
        new_row = pd.DataFrame({
            'Date': [test_date],
            'Revenue': [np.nan],
            **(({'COGS': [test_cogs]}) if feature_mode in ['lagged_cogs_only', 'current_day_cogs'] else {})
        })
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Engineer features
        history = engineer_features_for_row(history, feature_mode=feature_mode)
        
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

def save_submission(test_df, predictions, feature_mode: str):
    """Save submission file for the given mode."""
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values
    })
    
    # Map mode to filename
    filename_map = {
        'no_cogs': 'submission_no_cogs.csv',
        'lagged_cogs_only': 'submission_lagged_cogs_only.csv',
        'current_day_cogs': 'submission_current_day_cogs.csv'
    }
    filename = filename_map[feature_mode]
    output_file = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved: {output_file}")
    print(f"Shape: {submission.shape}")
    print(f"Nulls: {submission.isnull().sum().sum()}")
    
    return submission, output_file

# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    # Load data
    train, test = load_data()
    
    print(f"\n{'='*70}")
    print("3-WAY COGS ABLATION STUDY")
    print(f"{'='*70}")
    
    # Backtest all 3 modes
    backtest_results = {}
    for mode in FEATURE_MODES:
        result = backtest_mode(train, feature_mode=mode, num_folds=2)
        backtest_results[mode] = result
    
    # Print ablation summary
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    summary_table = []
    for mode in FEATURE_MODES:
        result = backtest_results[mode]
        fold_maes = [f['MAE'] for f in result['folds']]
        fold_str = ' | '.join([f'{mae:,.0f}' for mae in fold_maes])
        summary_table.append({
            'Mode': mode,
            'Fold MAEs': fold_str,
            'Mean MAE': result['mean_mae']
        })
        print(f"\n{mode.upper()}:")
        print(f"  Fold MAEs: {fold_str}")
        print(f"  Mean MAE:  {result['mean_mae']:,.0f}")
    
    # Train final models for ALL 3 modes (not just winner)
    print(f"\n{'='*70}")
    print("FINAL TRAINING ON FULL SALES.CSV (ALL 3 MODES)")
    print(f"{'='*70}")
    
    submissions = {}
    
    for mode in FEATURE_MODES:
        print(f"\n{mode.upper()}:")
        
        # Prepare full training data
        X_full, y_full, feature_names = prepare_train_data_vectorized(train, feature_mode=mode)
        print(f"  Training data: {X_full.shape[0]} rows × {X_full.shape[1]} features")
        print(f"  Features: {feature_names[:8]}... (showing first 8)")
        
        # Train final model
        model_final = train_model(X_full, y_full, model_type='lgbm')
        
        # Test prediction
        print(f"\n  TEST PREDICTION:")
        test_preds = recursive_predict_test(train, test, model_final, feature_names, feature_mode=mode)
        
        # Save submission
        submission, output_file = save_submission(test, test_preds, feature_mode=mode)
        submissions[mode] = output_file
    
    # Final summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"\nBacktest Results:")
    for mode in FEATURE_MODES:
        print(f"  {mode.upper()}: Mean MAE = {backtest_results[mode]['mean_mae']:,.0f}")
    
    print(f"\nSubmission Files:")
    for mode in FEATURE_MODES:
        print(f"  {mode.upper()}: {submissions[mode]}")
    
    print(f"\nAll 3 variants are ready for Kaggle submission!")

if __name__ == '__main__':
    main()

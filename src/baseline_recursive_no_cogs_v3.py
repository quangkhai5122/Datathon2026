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

# Feature windows
LAG_WINDOWS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 182, 364, 365, 371]  # Added half-year and yearly variations
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
EWMA_WINDOWS = [7, 30, 90]
QUANTILE_WINDOWS = [30, 90]

# Burn-in must cover the maximum lag / window
BURN_IN_DAYS = 371 

print(f"\n{'='*80}")
print("BASELINE RECURSIVE v3 — NO_COGS, STRICT CONTINUITY & ALIGNED HORIZON")
print(f"{'='*80}")

# ============================================================================
# 1. LOAD DATA & CONTINUITY CHECK
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate training and test data."""
    train = pd.read_csv(TRAIN_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(TEST_FILE, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    
    print(f"\nData Loading:")
    print(f"  Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    print(f"  Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")
    
    # PRIORITY 5: STRICTER CONTINUITY HANDLING
    date_diff = (train['Date'].max() - train['Date'].min()).days + 1
    if date_diff == len(train):
        print(f"  OK Daily continuity: 100% (no missing dates). Row-based lags are perfectly equivalent to day-based lags.")
    else:
        raise ValueError(f"Strict Continuity Failed! Missing {date_diff - len(train)} days. Row-based lags would be invalid.")
        
    return train, test


# ============================================================================
# 2. VECTORIZED FEATURE ENGINEERING FOR TRAINING
# ============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features."""
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


def prepare_train_features_vectorized(train_df: pd.DataFrame, burnin: int = BURN_IN_DAYS) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training features using full vectorization.
    CONSISTENCY: roll_std uses ddof=1 (unbiased).
    """
    df = train_df[['Date', 'Revenue']].copy()
    
    print(f"\nFeature Engineering (Vectorized Training):")
    
    df = add_calendar_features(df)
    
    shifted_rev = df['Revenue'].shift(1)
    
    # ---------------------------------------------------------
    # A. REVENUE LAG FEATURES
    # ---------------------------------------------------------
    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)
        
    # ---------------------------------------------------------
    # B. ROLLING STATISTICS (mean, std, min, max)
    # ---------------------------------------------------------
    for win in ROLLING_WINDOWS:
        df[f'roll_mean_{win}'] = shifted_rev.rolling(window=win).mean()
        df[f'roll_std_{win}'] = shifted_rev.rolling(window=win).std(ddof=1)
        df[f'roll_min_{win}'] = shifted_rev.rolling(window=win).min()
        df[f'roll_max_{win}'] = shifted_rev.rolling(window=win).max()

    # Rolling quantiles
    for win in QUANTILE_WINDOWS:
        df[f'roll_q25_{win}'] = shifted_rev.rolling(window=win).quantile(0.25)
        df[f'roll_median_{win}'] = shifted_rev.rolling(window=win).median()
        df[f'roll_q75_{win}'] = shifted_rev.rolling(window=win).quantile(0.75)
        
    # ---------------------------------------------------------
    # C. RECENCY-WEIGHTED FEATURES (EWMA)
    # ---------------------------------------------------------
    for span in EWMA_WINDOWS:
        df[f'ewma_{span}'] = shifted_rev.ewm(span=span, adjust=False).mean()

    # Ratios
    df['ewma_7_over_30'] = df['ewma_7'] / (df['ewma_30'] + 1e-6)
    df['ewma_30_over_90'] = df['ewma_30'] / (df['ewma_90'] + 1e-6)
    
    # ---------------------------------------------------------
    # D. STRONGER ACTIVITY / RECENCY FEATURES
    # ---------------------------------------------------------
    revenue_positive = (df['Revenue'] > 0).astype(int)
    for win in QUANTILE_WINDOWS:
        df[f'active_frac_{win}'] = revenue_positive.shift(1).rolling(window=win).mean()
        
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
    
    # ---------------------------------------------------------
    # E. BETTER SEASONAL-HISTORY FEATURES (RECENT PAST ONLY)
    # ---------------------------------------------------------
    # Instead of all-history average, we compute trailing same-DOW mean over last 12 weeks
    # For a given row, we want the mean of (i-7, i-14, ..., i-84)
    # Vectorized approach using shift
    dow_lags = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
    dow_series = []
    for l in dow_lags:
        if f'lag_{l}' in df.columns:
            dow_series.append(df[f'lag_{l}'])
        else:
            dow_series.append(df['Revenue'].shift(l))
    df['recent_same_dow_mean'] = pd.concat(dow_series, axis=1).mean(axis=1)

    # Same-month mean approximations: roughly last 3 occurrences of ~30 days apart
    # Wait, month isn't exactly 30 days. Let's just use rolling mean of the 30-day period one year ago
    df['recent_same_month_last_year'] = shifted_rev.shift(365-15).rolling(window=30).mean()
    
    # ---------------------------------------------------------
    # SUMMARY & DROP BURN-IN
    # ---------------------------------------------------------
    
    # PRIORITY 1: Drop burn-in rows to eliminate structural NaNs safely
    df = df.iloc[burnin:].reset_index(drop=True)
    
    feature_names = [c for c in df.columns if c not in ['Date', 'Revenue']]
    print(f"  Total features: {len(feature_names)}")
    
    # Fill remaining accidental NaNs (e.g. from early days_since_nonzero or division)
    X = df[feature_names].fillna(0).values
    y = df['Revenue'].values
    
    return X, y, feature_names


# ============================================================================
# 3. ROW-BY-ROW FEATURE ENGINEERING FOR RECURSIVE INFERENCE
# ============================================================================

def engineer_features_for_row(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the LAST row of history_df.
    """
    df = history_df.copy()
    df = add_calendar_features(df)
    
    idx = len(df) - 1 
    
    # ---------------------------------------------------------
    # A. REVENUE LAG FEATURES
    # ---------------------------------------------------------
    for lag in LAG_WINDOWS:
        if idx >= lag:
            df.loc[idx, f'lag_{lag}'] = df.loc[idx - lag, 'Revenue']
        else:
            df.loc[idx, f'lag_{lag}'] = np.nan
            
    # ---------------------------------------------------------
    # B. ROLLING STATISTICS
    # ---------------------------------------------------------
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{win}'] = past.mean()
            df.loc[idx, f'roll_std_{win}'] = np.std(past, ddof=1) if len(past) > 1 else np.nan
            df.loc[idx, f'roll_min_{win}'] = past.min()
            df.loc[idx, f'roll_max_{win}'] = past.max()
        else:
            df.loc[idx, f'roll_mean_{win}'] = np.nan
            df.loc[idx, f'roll_std_{win}'] = np.nan
            df.loc[idx, f'roll_min_{win}'] = np.nan
            df.loc[idx, f'roll_max_{win}'] = np.nan

    for win in QUANTILE_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'roll_q25_{win}'] = np.percentile(past, 25)
            df.loc[idx, f'roll_median_{win}'] = np.median(past)
            df.loc[idx, f'roll_q75_{win}'] = np.percentile(past, 75)
        else:
            df.loc[idx, f'roll_q25_{win}'] = np.nan
            df.loc[idx, f'roll_median_{win}'] = np.nan
            df.loc[idx, f'roll_q75_{win}'] = np.nan

    # ---------------------------------------------------------
    # C. RECENCY-WEIGHTED FEATURES (EWMA)
    # ---------------------------------------------------------
    for span in EWMA_WINDOWS:
        if idx >= 1:
            alpha = 2 / (span + 1)
            # Reconstruct EWMA up to the previous row
            # To be efficient, we approximate by calculating from the past 3*span points
            lookback = min(idx, span * 3)
            past_series = df.loc[idx - lookback:idx - 1, 'Revenue']
            ewma_val = past_series.ewm(span=span, adjust=False).mean().iloc[-1]
            df.loc[idx, f'ewma_{span}'] = ewma_val
        else:
            df.loc[idx, f'ewma_{span}'] = np.nan
            
    if idx >= 1:
        e7 = df.loc[idx, 'ewma_7']
        e30 = df.loc[idx, 'ewma_30']
        e90 = df.loc[idx, 'ewma_90']
        df.loc[idx, 'ewma_7_over_30'] = e7 / (e30 + 1e-6) if not pd.isna(e7) and not pd.isna(e30) else np.nan
        df.loc[idx, 'ewma_30_over_90'] = e30 / (e90 + 1e-6) if not pd.isna(e30) and not pd.isna(e90) else np.nan
    else:
        df.loc[idx, 'ewma_7_over_30'] = np.nan
        df.loc[idx, 'ewma_30_over_90'] = np.nan

    # ---------------------------------------------------------
    # D. ACTIVITY / SPARSITY
    # ---------------------------------------------------------
    for win in QUANTILE_WINDOWS: # reusing [30, 90]
        if idx >= win:
            past_win = df.loc[max(0, idx - win):idx - 1, 'Revenue'].values
            df.loc[idx, f'active_frac_{win}'] = (past_win > 0).mean()
        else:
            df.loc[idx, f'active_frac_{win}'] = np.nan
            
    if idx > 0:
        for j in range(idx - 1, -1, -1):
            if df.loc[j, 'Revenue'] > 0:
                df.loc[idx, 'days_since_nonzero'] = idx - j
                break
        else:
            df.loc[idx, 'days_since_nonzero'] = np.nan
    else:
        df.loc[idx, 'days_since_nonzero'] = np.nan
        
    # ---------------------------------------------------------
    # E. BETTER SEASONAL-HISTORY
    # ---------------------------------------------------------
    dow_lags = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
    recent_dow_vals = []
    for l in dow_lags:
        if idx >= l:
            recent_dow_vals.append(df.loc[idx - l, 'Revenue'])
    df.loc[idx, 'recent_same_dow_mean'] = np.mean(recent_dow_vals) if recent_dow_vals else np.nan
    
    if idx >= 365 + 15:
        past_month_vals = df.loc[idx - 365 - 14 : idx - 365 + 15, 'Revenue'].values
        df.loc[idx, 'recent_same_month_last_year'] = past_month_vals.mean()
    else:
        df.loc[idx, 'recent_same_month_last_year'] = np.nan

    return df


# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

def train_model(X: np.ndarray, y: np.ndarray, verbose: bool = False) -> object:
    """Train LightGBM model."""
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
# 5. HORIZON-ALIGNED ROLLING-ORIGIN BACKTEST
# ============================================================================

def rolling_origin_backtest(train_df: pd.DataFrame) -> Dict:
    """
    PRIORITY 2: HORIZON-ALIGNED BACKTEST
    True horizon = 548 days.
    """
    print(f"\n{'='*80}")
    print("HORIZON-ALIGNED BACKTEST (548 DAYS PER FOLD)")
    print(f"{'='*80}")
    
    # 548-day folds backwards from 2022-12-31
    folds_def = [
        {'train_end': '2021-07-01', 'val_start': '2021-07-02', 'val_end': '2022-12-31', 'label': 'Fold 1 (Recent)'},
        {'train_end': '2020-01-01', 'val_start': '2020-01-02', 'val_end': '2021-07-01', 'label': 'Fold 2 (Past)'},
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
        print(f"  Training: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        
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
            
            X_last = history.iloc[-1:][feature_names].fillna(0).values
            y_pred = max(0, model.predict(X_last)[0])
            
            preds.append(y_pred)
            actuals.append(val_revenue)
            history.loc[len(history) - 1, 'Revenue'] = y_pred
            
            if (val_idx + 1) % 100 == 0:
                print(f"    Val step {val_idx + 1}/{len(val_fold)}")
                
        preds = np.array(preds)
        actuals = np.array(actuals)
        mae = mean_absolute_error(actuals, preds)
        
        print(f"  MAE:  {mae:,.0f}")
        results.append({'fold': fold_def['label'], 'MAE': mae, 'Length': len(val_fold)})
    
    mean_mae = np.mean([r['MAE'] for r in results]) if results else np.inf
    print(f"\n{'='*80}")
    print(f"Backtest Summary: Mean MAE = {mean_mae:,.0f}")
    print(f"{'='*80}")
    
    return {'folds': results, 'mean_mae': mean_mae}


# ============================================================================
# 6. RECURSIVE TEST PREDICTION
# ============================================================================

def recursive_predict_test(train_df: pd.DataFrame, test_df: pd.DataFrame, model: object, feature_names: List[str]) -> np.ndarray:
    print(f"\nRecursive Test Prediction ({len(test_df)} rows)...")
    history = train_df[['Date', 'Revenue']].copy()
    preds = []
    
    for test_idx in range(len(test_df)):
        test_date = test_df.iloc[test_idx]['Date']
        
        new_row = pd.DataFrame({'Date': [test_date], 'Revenue': [np.nan]})
        history = pd.concat([history, new_row], ignore_index=True)
        history = engineer_features_for_row(history)
        
        X_last = history.iloc[-1:][feature_names].fillna(0).values
        y_pred = max(0, model.predict(X_last)[0])
        
        preds.append(y_pred)
        history.loc[len(history) - 1, 'Revenue'] = y_pred
        
        if (test_idx + 1) % 100 == 0:
            print(f"  Predicted {test_idx + 1}/{len(test_df)}")
            
    return np.array(preds)


# ============================================================================
# 7. SAVE SUBMISSION
# ============================================================================

def save_submission(test_df: pd.DataFrame, predictions: np.ndarray) -> str:
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values if 'COGS' in test_df.columns else 0
    })
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'submission_no_cogs_v3.csv')
    submission.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved: {output_file}")
    return output_file


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    start_time = datetime.now()
    train, test = load_data()
    
    backtest_result = rolling_origin_backtest(train)
    
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
    print(f"\nOK Complete! Ready for Kaggle submission.")

if __name__ == '__main__':
    main()

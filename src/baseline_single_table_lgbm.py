"""
Single-Table Baseline Forecasting Model
========================================

Objective: Forecast daily Revenue using only sales.csv (no multi-table joins).

Strategy:
  - Load sales.csv (3833 daily rows, 2012-07-04 to 2022-12-31)
  - Create time-safe features: calendar, lags, rolling stats, expanding means
  - Validate on 2022 (holdout year)
  - Train on 2012-2021 history, evaluate on 2022
  - Refit on full 2012-2022 history
  - Predict test rows (2023-01-01 to 2024-07-01)
  - Export submission in required format

Model: Scikit-learn GradientBoostingRegressor
  (LightGBM and XGBoost not available; this is the next-best tree-ensemble method)

Metric: MAE on Revenue

Time leakage prevention:
  - All lag features built from past observations only
  - Rolling features shifted to avoid future data
  - No data from test period used in feature engineering
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

# Modeling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = 'data/raw'
TRAIN_FILE = os.path.join(DATA_DIR, 'sales.csv')
TEST_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_DIR = 'outputs'
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission_single_table.csv')

# Feature engineering parameters
LAG_WINDOWS = [1, 2, 3, 7, 14, 30, 90]  # days
ROLLING_WINDOWS = [7, 14, 30, 90]  # days

# Model hyperparameters
RANDOM_STATE = 42
GB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.9,
    'random_state': RANDOM_STATE,
    'verbose': 0,
}

# ============================================================================
# 1. Data Loading & Inspection
# ============================================================================

def load_and_inspect_data(train_path, test_path):
    """Load and inspect train and test files."""
    print("\n" + "="*70)
    print("1. DATA LOADING & INSPECTION")
    print("="*70)
    
    # Load train
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date'])
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Train date range: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"Train nulls:\n{train.isnull().sum()}")
    print(f"\nTrain columns: {list(train.columns)}")
    print(f"\nTrain head:")
    print(train.head(3))
    
    print(f"\n\nTest shape: {test.shape}")
    print(f"Test date range: {test['Date'].min().date()} → {test['Date'].max().date()}")
    print(f"Test nulls:\n{test.isnull().sum()}")
    print(f"\nTest head:")
    print(test.head(3))
    
    # Basic stats
    print(f"\n\nTrain Revenue stats:")
    print(train['Revenue'].describe())
    print(f"\nTrain COGS stats:")
    print(train['COGS'].describe())
    
    return train, test


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
    df['dayofweek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day'] = df['Date'].dt.day
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
    return df


def create_lag_features(df, target_col='Revenue', windows=None):
    """
    Create lagged features for target_col.
    
    Args:
        df: DataFrame with datetime index
        target_col: Column to lag (e.g., 'Revenue')
        windows: List of lag window sizes in days
    
    Returns:
        DataFrame with lag features added
    """
    if windows is None:
        windows = LAG_WINDOWS
    
    df = df.copy()
    for lag in windows:
        df[f'lag_{lag}_{target_col}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df, target_col='Revenue', windows=None):
    """
    Create rolling window statistics (shifted to avoid leakage).
    
    Args:
        df: DataFrame with datetime index
        target_col: Column to compute rolling stats on
        windows: List of rolling window sizes in days
    
    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    
    df = df.copy()
    for win in windows:
        # Shift by 1 to avoid using current day's value
        df[f'rolling_mean_{win}_{target_col}'] = (
            df[target_col].shift(1).rolling(window=win, min_periods=1).mean()
        )
        df[f'rolling_std_{win}_{target_col}'] = (
            df[target_col].shift(1).rolling(window=win, min_periods=1).std()
        )
        df[f'rolling_min_{win}_{target_col}'] = (
            df[target_col].shift(1).rolling(window=win, min_periods=1).min()
        )
        df[f'rolling_max_{win}_{target_col}'] = (
            df[target_col].shift(1).rolling(window=win, min_periods=1).max()
        )
        df[f'rolling_median_{win}_{target_col}'] = (
            df[target_col].shift(1).rolling(window=win, min_periods=1).median()
        )
    
    return df


def create_expanding_features(df, target_col='Revenue'):
    """
    Create expanding window statistics (cumulative up to but not including current row).
    
    Args:
        df: DataFrame with datetime index
        target_col: Column to compute expanding stats on
    
    Returns:
        DataFrame with expanding features added
    """
    df = df.copy()
    
    # Expanding mean up to previous day
    df[f'expanding_mean_{target_col}'] = (
        df[target_col].shift(1).expanding(min_periods=1).mean()
    )
    df[f'expanding_median_{target_col}'] = (
        df[target_col].shift(1).expanding(min_periods=1).median()
    )
    df[f'expanding_std_{target_col}'] = (
        df[target_col].shift(1).expanding(min_periods=1).std()
    )
    
    return df


def create_trend_features(df, target_col='Revenue'):
    """Create trend-based features."""
    df = df.copy()
    
    # Recent (30-day) vs long (90-day) mean ratio
    rolling_30 = df[target_col].shift(1).rolling(window=30, min_periods=1).mean()
    rolling_90 = df[target_col].shift(1).rolling(window=90, min_periods=1).mean()
    df['trend_30_90_ratio'] = rolling_30 / (rolling_90 + 1e-6)  # Avoid division by zero
    
    # Momentum: recent mean minus very-recent mean
    rolling_7 = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    df['momentum_7_30'] = rolling_30 - rolling_7
    
    # Count of positive revenue days in last 30 days (proxy for active trading days)
    df['active_days_30'] = (
        (df[target_col].shift(1) > 0).rolling(window=30, min_periods=1).sum()
    )
    
    return df


def engineer_features(df, target_col='Revenue'):
    """Master feature engineering function."""
    print("\n" + "="*70)
    print("2. FEATURE ENGINEERING (TIME-SAFE)")
    print("="*70)
    
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nStarting with {df.shape[1]} columns, {df.shape[0]} rows")
    
    # Calendar features
    df = create_calendar_features(df)
    print(f"After calendar features: {df.shape[1]} columns")
    
    # Lag features
    df = create_lag_features(df, target_col=target_col)
    print(f"After lag features: {df.shape[1]} columns")
    
    # Rolling features
    df = create_rolling_features(df, target_col=target_col)
    print(f"After rolling features: {df.shape[1]} columns")
    
    # Expanding features
    df = create_expanding_features(df, target_col=target_col)
    print(f"After expanding features: {df.shape[1]} columns")
    
    # Trend features
    df = create_trend_features(df, target_col=target_col)
    print(f"After trend features: {df.shape[1]} columns")
    
    return df


# ============================================================================
# 3. Data Preparation for Modeling
# ============================================================================

def prepare_modeling_data(df, target_col='Revenue', drop_initial_nulls=True):
    """
    Prepare data for modeling: handle NaNs, select features, create X and y.
    
    Args:
        df: DataFrame with engineered features
        target_col: Target column name
        drop_initial_nulls: Drop rows with NaN in features (expected from lags/rolling)
    
    Returns:
        X, y, feature_names
    """
    print("\n" + "="*70)
    print("3. DATA PREPARATION FOR MODELING")
    print("="*70)
    
    df = df.copy()
    
    # Identify feature columns (exclude Date, target, and COGS)
    exclude_cols = {'Date', target_col, 'COGS'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nTotal columns: {df.shape[1]}")
    print(f"Excluded columns: {exclude_cols}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"\nFirst 20 features: {feature_cols[:20]}")
    
    # Handle NaNs in feature columns
    print(f"\nNull counts in first 10 feature columns:")
    for col in feature_cols[:10]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  {col}: {null_count}")
    
    if drop_initial_nulls:
        # Drop rows where lags/rolling features are NaN (initial burn-in period)
        df_before = df.shape[0]
        df = df.dropna(subset=feature_cols, how='any')
        print(f"\nDropped {df_before - df.shape[0]} rows due to NaN in features")
        print(f"Modeling dataset: {df.shape[0]} rows")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y stats:\n{y.describe()}")
    
    return X, y, feature_cols, df


# ============================================================================
# 4. Training & Validation
# ============================================================================

def train_validate_model(X, y, dates, val_year=2022, train_year_min=2012):
    """
    Train and validate using time-based split.
    
    Args:
        X: Features
        y: Target
        dates: Date column aligned with X, y
        val_year: Year to use as validation set
        train_year_min: Minimum year for training set
    
    Returns:
        train_indices, val_indices, model, train_metrics, val_metrics
    """
    print("\n" + "="*70)
    print("4. TRAINING & VALIDATION (TIME-BASED SPLIT)")
    print("="*70)
    
    # Create time-based split
    train_mask = (dates.dt.year >= train_year_min) & (dates.dt.year < val_year)
    val_mask = dates.dt.year == val_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    train_dates = dates[train_mask]
    val_dates = dates[val_mask]
    
    print(f"\nTrain period: {train_dates.min().date()} → {train_dates.max().date()}")
    print(f"Train size: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"\nValidation period: {val_dates.min().date()} → {val_dates.max().date()}")
    print(f"Validation size: {X_val.shape[0]} rows")
    
    # Train model
    print(f"\nTraining GradientBoostingRegressor...")
    model = GradientBoostingRegressor(**GB_PARAMS)
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n--- TRAIN METRICS ---")
    print(f"MAE:  {train_mae:,.2f}")
    print(f"RMSE: {train_rmse:,.2f}")
    print(f"R²:   {train_r2:.4f}")
    
    print(f"\n--- VALIDATION METRICS (2022) ---")
    print(f"MAE:  {val_mae:,.2f}")
    print(f"RMSE: {val_rmse:,.2f}")
    print(f"R²:   {val_r2:.4f}")
    
    train_metrics = {'MAE': train_mae, 'RMSE': train_rmse, 'R2': train_r2}
    val_metrics = {'MAE': val_mae, 'RMSE': val_rmse, 'R2': val_r2}
    
    return X_train, y_train, X_val, y_val, model, train_metrics, val_metrics


def get_feature_importance(model, feature_names, top_n=20):
    """Extract and print feature importance."""
    print(f"\n--- TOP {top_n} MOST IMPORTANT FEATURES ---")
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(top_n).to_string(index=False))
    return importance


# ============================================================================
# 5. Final Model Training on Full History
# ============================================================================

def retrain_on_full_history(X, y, feature_names, train_year_min=2012, train_year_max=2022):
    """Retrain model on full available history."""
    print("\n" + "="*70)
    print("5. RETRAINING ON FULL HISTORY")
    print("="*70)
    
    # Limit to training years only (exclude test period data)
    # In this case, limit to years up to train_year_max
    print(f"\nRetraining on full history: {train_year_min}-{train_year_max}")
    print(f"Full dataset size: {X.shape[0]} rows, {X.shape[1]} features")
    
    model = GradientBoostingRegressor(**GB_PARAMS)
    model.fit(X, y)
    print("Retraining complete.")
    
    return model


# ============================================================================
# 6. Prediction & Submission
# ============================================================================

def generate_predictions(model, X, feature_names):
    """Generate predictions for given X."""
    y_pred = model.predict(X)
    return y_pred


def create_submission(test_df, predictions, target_col='Revenue', output_file=None):
    """
    Create and save submission file.
    
    Args:
        test_df: Test DataFrame with Date column
        predictions: Array of predictions
        target_col: Name of target column for submission
        output_file: Path to save submission
    
    Returns:
        Submission DataFrame
    """
    print("\n" + "="*70)
    print("6. SUBMISSION GENERATION")
    print("="*70)
    
    submission = pd.DataFrame({
        'Date': test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': predictions,
        'COGS': test_df['COGS'].values  # Keep COGS as provided in test file
    })
    
    # Optionally clip negative predictions (Revenue should be >= 0)
    if (predictions < 0).any():
        print(f"\nWarning: {(predictions < 0).sum()} negative predictions found.")
        print(f"Min prediction: {predictions.min():,.2f}")
        # Keep them as-is for now; can clip if domain knowledge suggests it
    
    print(f"\nSubmission shape: {submission.shape}")
    print(f"Date range: {submission['Date'].min()} → {submission['Date'].max()}")
    print(f"Revenue prediction range: {submission['Revenue'].min():,.2f} → {submission['Revenue'].max():,.2f}")
    print(f"\nSubmission head:")
    print(submission.head(10))
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        submission.to_csv(output_file, index=False)
        print(f"\nSubmission saved to: {output_file}")
    
    return submission


# ============================================================================
# 7. Main Pipeline
# ============================================================================

def main():
    """Execute the full baseline pipeline."""
    print("\n" + "="*70)
    print("SINGLE-TABLE BASELINE FORECASTING PIPELINE")
    print("="*70)
    print(f"Model: scikit-learn GradientBoostingRegressor")
    print(f"Target: Revenue")
    print(f"Metric: MAE")
    print(f"Random state: {RANDOM_STATE}")
    
    # 1. Load data
    train, test = load_and_inspect_data(TRAIN_FILE, TEST_FILE)
    
    # 2. Engineer features on full training data
    train_engineered = engineer_features(train)
    
    # 3. Prepare modeling data
    X, y, feature_names, train_with_features = prepare_modeling_data(
        train_engineered, target_col='Revenue'
    )
    
    # 4. Train and validate
    X_train, y_train, X_val, y_val, model, train_metrics, val_metrics = train_validate_model(
        X, y, train_with_features['Date'], val_year=2022
    )
    
    # 5. Feature importance
    importance_df = get_feature_importance(model, feature_names, top_n=20)
    
    # 6. Retrain on full available history (2012-2022)
    # First, we need X and y from the full train data
    model_full = retrain_on_full_history(X, y, feature_names)
    
    # 7. Engineer features for test data
    # We need to align test data with the same feature engineering process
    # But we only have test data starting from 2023-01-01
    # We need to use the last values from train to compute lags
    
    print("\n" + "="*70)
    print("7. PREPARING TEST DATA FOR PREDICTIONS")
    print("="*70)
    
    # Combine train and test for feature engineering continuity
    # But keep track of which rows are test
    combined = pd.concat([train_engineered, test], ignore_index=False, sort=False)
    # Re-index to handle the concat
    combined = combined.reset_index(drop=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Re-engineer features on combined data to ensure continuity of lags
    combined_engineered = engineer_features(combined)
    
    # Extract test features (rows corresponding to test dates)
    test_engineered = combined_engineered[combined_engineered['Date'].isin(test['Date'])].copy()
    
    # Select only feature columns
    exclude_cols = {'Date', 'Revenue', 'COGS'}
    feature_cols = [c for c in test_engineered.columns if c not in exclude_cols]
    X_test = test_engineered[feature_cols].copy()
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test feature columns: {len(feature_cols)}")
    print(f"Null counts in test features:")
    null_counts = X_test.isnull().sum()
    if null_counts.sum() > 0:
        print(null_counts[null_counts > 0])
    else:
        print("No nulls!")
    
    # Generate predictions
    y_test_pred = generate_predictions(model_full, X_test, feature_cols)
    
    # 8. Create and save submission
    submission = create_submission(
        test,
        y_test_pred,
        target_col='Revenue',
        output_file=SUBMISSION_FILE
    )
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\nTraining data: {train.shape[0]} rows")
    print(f"Test data: {test.shape[0]} rows")
    print(f"Features engineered: {len(feature_names)}")
    print(f"\nValidation (2022) MAE: {val_metrics['MAE']:,.2f}")
    print(f"Validation (2022) RMSE: {val_metrics['RMSE']:,.2f}")
    print(f"\nSubmission saved to: {SUBMISSION_FILE}")
    
    return {
        'train': train,
        'test': test,
        'model': model_full,
        'val_metrics': val_metrics,
        'submission': submission,
        'feature_importance': importance_df,
    }


if __name__ == '__main__':
    results = main()
    print("\n✓ Pipeline execution complete.")

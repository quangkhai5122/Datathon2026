"""
Multi-Table Baseline v1 -- Composition Templates + Single-Table Features
========================================================================
Track 1 only: forecast-time-safe features.
Row-level as-of template construction for training.
Frozen templates for validation/test.
10 composition features + 55 single-table features = 65 total.
"""
import os, warnings, numpy as np, pandas as pd
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

DATA_DIR = 'data/raw'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LAG_WINDOWS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 364]
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
BURN_IN_DAYS = 364

# Names of the 10 composition template features
MONTHLY_COMP_COLS = [
    'order_count', 'avg_order_value', 'avg_basket_size',
    'discount_intensity', 'cancel_rate', 'category_concentration'
]
DOW_COMP_COLS = [
    'order_count', 'avg_order_value', 'avg_basket_size', 'discount_intensity'
]

print(f"\n{'='*80}")
print("MULTI-TABLE BASELINE v1 -- COMPOSITION TEMPLATES")
print(f"{'='*80}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_all_data():
    sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), parse_dates=['order_date'])
    items = pd.read_csv(os.path.join(DATA_DIR, 'order_items.csv'), low_memory=False)
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))

    # Continuity check
    date_diff = (sales['Date'].max() - sales['Date'].min()).days + 1
    assert date_diff == len(sales), "Continuity failed"
    assert test['Date'].nunique() == len(test), "Test dates not unique"
    horizon = len(test)

    print(f"  Sales: {len(sales)} rows, Orders: {len(orders)}, Items: {len(items)}, Products: {len(products)}")
    print(f"  Horizon: {horizon} days")
    return sales, test, orders, items, products, horizon


# ============================================================================
# 2. BUILD DAILY COMPOSITION SERIES
# ============================================================================
def build_daily_composition(orders_df, items_df, products_df):
    """Build daily composition metrics from orders+items+products."""
    merged = items_df.merge(orders_df[['order_id', 'order_date', 'order_status']], on='order_id')
    merged = merged.merge(products_df[['product_id', 'category']], on='product_id', how='left')
    merged['line_revenue'] = merged['quantity'] * merged['unit_price'] - merged['discount_amount']

    daily = merged.groupby('order_date').agg(
        total_orders=('order_id', 'nunique'),
        total_items=('quantity', 'sum'),
        total_revenue=('line_revenue', 'sum'),
        avg_unit_price=('unit_price', 'mean'),
        total_discount=('discount_amount', 'sum'),
        total_gross=pd.NamedAgg(column='line_revenue', aggfunc=lambda x: (x + merged.loc[x.index, 'discount_amount']).sum()),
        unique_products=('product_id', 'nunique'),
    ).reset_index()

    # Cancel rate per day
    cancel_daily = merged[merged['order_status'] == 'cancelled'].groupby('order_date')['order_id'].nunique().reset_index()
    cancel_daily.columns = ['order_date', 'cancel_orders']
    daily = daily.merge(cancel_daily, on='order_date', how='left')
    daily['cancel_orders'] = daily['cancel_orders'].fillna(0)

    # Category concentration (HHI)
    cat_rev = merged.groupby(['order_date', 'category'])['line_revenue'].sum().reset_index()
    cat_total = cat_rev.groupby('order_date')['line_revenue'].transform('sum')
    cat_rev['share'] = cat_rev['line_revenue'] / (cat_total + 1e-9)
    hhi = (cat_rev['share'] ** 2).groupby(cat_rev['order_date']).sum().reset_index()
    hhi.columns = ['order_date', 'category_hhi']
    daily = daily.merge(hhi, on='order_date', how='left')

    # Derived metrics
    daily['order_count'] = daily['total_orders']
    daily['avg_order_value'] = daily['total_revenue'] / (daily['total_orders'] + 1e-9)
    daily['avg_basket_size'] = daily['total_items'] / (daily['total_orders'] + 1e-9)
    daily['discount_intensity'] = daily['total_discount'] / (daily['total_gross'] + 1e-9)
    daily['cancel_rate'] = daily['cancel_orders'] / (daily['total_orders'] + 1e-9)
    daily['category_concentration'] = daily['category_hhi']

    daily = daily.rename(columns={'order_date': 'Date'}).sort_values('Date').reset_index(drop=True)
    daily['month'] = daily['Date'].dt.month
    daily['dayofweek'] = daily['Date'].dt.dayofweek

    keep_cols = ['Date', 'month', 'dayofweek'] + MONTHLY_COMP_COLS
    print(f"  Daily composition: {len(daily)} rows, {len(MONTHLY_COMP_COLS)} metrics")
    return daily[keep_cols]


# ============================================================================
# 3. EXPANDING AS-OF TEMPLATE FEATURES (ROW-LEVEL)
# ============================================================================
def compute_expanding_templates(daily_comp, cutoff_date=None):
    """
    For each row at date t, compute expanding same-month and same-dow mean
    of composition metrics using only data from dates < t.
    O(n) via running sums.
    If cutoff_date is set, only use rows with Date <= cutoff_date.
    """
    df = daily_comp.copy()
    if cutoff_date is not None:
        df = df[df['Date'] <= cutoff_date].copy()
    df = df.sort_values('Date').reset_index(drop=True)

    months = df['month'].values
    dows = df['dayofweek'].values

    # Monthly templates
    for col in MONTHLY_COMP_COLS:
        vals = df[col].values.astype(float)
        result = np.full(len(vals), np.nan)
        sums = np.zeros(13)
        counts = np.zeros(13, dtype=int)
        for i in range(len(vals)):
            m = months[i]
            if counts[m] > 0:
                result[i] = sums[m] / counts[m]
            if not np.isnan(vals[i]):
                sums[m] += vals[i]
                counts[m] += 1
        df[f'tmpl_{col}_month'] = result

    # DOW templates
    for col in DOW_COMP_COLS:
        vals = df[col].values.astype(float)
        result = np.full(len(vals), np.nan)
        sums = np.zeros(7)
        counts = np.zeros(7, dtype=int)
        for i in range(len(vals)):
            d = dows[i]
            if counts[d] > 0:
                result[i] = sums[d] / counts[d]
            if not np.isnan(vals[i]):
                sums[d] += vals[i]
                counts[d] += 1
        df[f'tmpl_{col}_dow'] = result

    tmpl_cols = [f'tmpl_{c}_month' for c in MONTHLY_COMP_COLS] + [f'tmpl_{c}_dow' for c in DOW_COMP_COLS]
    return df[['Date'] + tmpl_cols]


def compute_frozen_templates(daily_comp, cutoff_date):
    """Freeze monthly/dow averages at cutoff. For val/test lookup."""
    df = daily_comp[daily_comp['Date'] <= cutoff_date].copy()

    monthly = {}
    for col in MONTHLY_COMP_COLS:
        monthly[f'tmpl_{col}_month'] = df.groupby('month')[col].mean().to_dict()

    dow = {}
    for col in DOW_COMP_COLS:
        dow[f'tmpl_{col}_dow'] = df.groupby('dayofweek')[col].mean().to_dict()

    return monthly, dow


def apply_frozen_templates(dates_df, monthly_dict, dow_dict):
    """Apply frozen templates to a DataFrame with Date column."""
    df = dates_df.copy()
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    for feat, lookup in monthly_dict.items():
        df[feat] = df['month'].map(lookup)
    for feat, lookup in dow_dict.items():
        df[feat] = df['dayofweek'].map(lookup)
    df = df.drop(columns=['month', 'dayofweek'], errors='ignore')
    return df


# ============================================================================
# 4. SINGLE-TABLE FEATURES (from v3.2)
# ============================================================================
def add_calendar_features(df):
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


def add_single_table_features(df, burnin):
    """Add all v3.2 single-table features to a df with Date and Revenue."""
    df = add_calendar_features(df)
    shifted = df['Revenue'].shift(1)

    for lag in LAG_WINDOWS:
        df[f'lag_{lag}'] = df['Revenue'].shift(lag)
    for win in ROLLING_WINDOWS:
        df[f'roll_mean_{win}'] = shifted.rolling(win).mean()
        df[f'roll_std_{win}'] = shifted.rolling(win).std(ddof=1)
        df[f'roll_min_{win}'] = shifted.rolling(win).min()
        df[f'roll_max_{win}'] = shifted.rolling(win).max()
        df[f'roll_median_{win}'] = shifted.rolling(win).median()

    df['expanding_mean'] = shifted.expanding().mean()
    df['expanding_std'] = shifted.expanding().std(ddof=1)

    mean_7 = shifted.rolling(7).mean()
    mean_30 = shifted.rolling(30).mean()
    mean_90 = shifted.rolling(90).mean()
    df['mean_7_minus_mean_30'] = mean_7 - mean_30
    df['mean_7_over_mean_30'] = mean_7 / (mean_30 + 1e-6)
    df['mean_30_over_mean_90'] = mean_30 / (mean_90 + 1e-6)
    df['volatility_ratio'] = shifted.rolling(30).std(ddof=1) / (mean_30 + 1e-6)

    rev_pos = (df['Revenue'] > 0).astype(int)
    df['count_positive_7'] = rev_pos.shift(1).rolling(7).sum()
    df['count_positive_30'] = rev_pos.shift(1).rolling(30).sum()

    dsn = [np.nan]
    for i in range(1, len(df)):
        for j in range(i - 1, -1, -1):
            if df['Revenue'].iloc[j] > 0:
                dsn.append(i - j)
                break
        else:
            dsn.append(np.nan)
    df['days_since_nonzero'] = dsn

    def expanding_seasonal(df, key_col, val_col):
        result = []
        for i in range(len(df)):
            if i == 0:
                result.append(np.nan)
            else:
                mask = df.iloc[:i][key_col] == df.iloc[i][key_col]
                result.append(df.iloc[:i][mask][val_col].mean() if mask.any() else np.nan)
        return result

    df['avg_revenue_by_dayofweek'] = expanding_seasonal(df, 'dayofweek', 'Revenue')
    df['avg_revenue_by_month'] = expanding_seasonal(df, 'month', 'Revenue')
    return df


# ============================================================================
# 5. PREPARE FULL TRAINING FEATURES
# ============================================================================
def prepare_training_data(sales_df, daily_comp, burnin=BURN_IN_DAYS, cutoff_date=None):
    """Build training features with row-level as-of templates."""
    if cutoff_date:
        df = sales_df[sales_df['Date'] <= cutoff_date].copy()
    else:
        df = sales_df.copy()

    # Single-table features
    df = add_single_table_features(df, burnin)

    # Row-level as-of composition templates
    tmpl = compute_expanding_templates(daily_comp, cutoff_date=cutoff_date)
    df = df.merge(tmpl, on='Date', how='left')

    # Drop burn-in
    df = df.iloc[burnin:].reset_index(drop=True)

    feature_names = [c for c in df.columns if c not in ['Date', 'Revenue', 'COGS']]

    # NaN audit
    nan_counts = df[feature_names].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"  NaN after burn-in: {dict(nan_cols)}")
        print(f"  -> LightGBM handles natively.")
    else:
        print(f"  [OK] No NaNs after burn-in.")

    print(f"  Features: {len(feature_names)} ({len(feature_names) - 10} single-table + 10 composition)")
    return df[feature_names], df['Revenue'].values, feature_names


# ============================================================================
# 6. ROW-BY-ROW INFERENCE FEATURES
# ============================================================================
def engineer_row_features(history_df):
    """Single-table features for last row (recursive inference). Same as v3.2."""
    df = history_df.copy()
    df = add_calendar_features(df)
    idx = len(df) - 1

    for lag in LAG_WINDOWS:
        df.loc[idx, f'lag_{lag}'] = df.loc[idx - lag, 'Revenue'] if idx >= lag else np.nan
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

    past_all = df.loc[0:idx - 1, 'Revenue'].values
    df.loc[idx, 'expanding_mean'] = past_all.mean() if len(past_all) > 0 else np.nan
    df.loc[idx, 'expanding_std'] = np.std(past_all, ddof=1) if len(past_all) > 1 else np.nan

    m7 = df.loc[max(0, idx - 7):idx - 1, 'Revenue'].mean() if idx >= 7 else np.nan
    m30 = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].mean() if idx >= 30 else np.nan
    m90 = df.loc[max(0, idx - 90):idx - 1, 'Revenue'].mean() if idx >= 90 else np.nan
    df.loc[idx, 'mean_7_minus_mean_30'] = m7 - m30 if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, 'mean_7_over_mean_30'] = m7 / (m30 + 1e-6) if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, 'mean_30_over_mean_90'] = m30 / (m90 + 1e-6) if not (np.isnan(m30) or np.isnan(m90)) else np.nan
    if idx >= 30:
        p30 = df.loc[max(0, idx - 30):idx - 1, 'Revenue'].values
        v30 = np.std(p30, ddof=1) if len(p30) > 1 else np.nan
        df.loc[idx, 'volatility_ratio'] = v30 / (m30 + 1e-6) if not np.isnan(v30) else np.nan
    else:
        df.loc[idx, 'volatility_ratio'] = np.nan

    if idx >= 7:
        df.loc[idx, 'count_positive_7'] = (df.loc[max(0, idx - 7):idx - 1, 'Revenue'].values > 0).sum()
    else:
        df.loc[idx, 'count_positive_7'] = np.nan
    if idx >= 30:
        df.loc[idx, 'count_positive_30'] = (df.loc[max(0, idx - 30):idx - 1, 'Revenue'].values > 0).sum()
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

    cur_dow = df.loc[idx, 'dayofweek']
    cur_month = df.loc[idx, 'month']
    mask_dow = df.loc[0:idx - 1, 'dayofweek'] == cur_dow
    df.loc[idx, 'avg_revenue_by_dayofweek'] = df.loc[0:idx - 1][mask_dow]['Revenue'].mean() if mask_dow.any() else np.nan
    mask_month = df.loc[0:idx - 1, 'month'] == cur_month
    df.loc[idx, 'avg_revenue_by_month'] = df.loc[0:idx - 1][mask_month]['Revenue'].mean() if mask_month.any() else np.nan

    return df


# ============================================================================
# 7. TRAIN MODEL
# ============================================================================
def train_model(X_df, y):
    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            objective='regression', metric='mae', learning_rate=0.05,
            num_leaves=31, n_estimators=300, random_state=RANDOM_STATE, verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            learning_rate=0.05, max_depth=6, n_estimators=300, random_state=RANDOM_STATE
        )
    model.fit(X_df, y)
    return model


# ============================================================================
# 8. BACKTEST + PREDICT
# ============================================================================
def run_backtest(sales_df, daily_comp, horizon):
    print(f"\n{'='*80}")
    print(f"HORIZON-ALIGNED BACKTEST ({horizon} DAYS PER FOLD)")
    print(f"{'='*80}")

    max_date = sales_df['Date'].max()
    val_end_1 = max_date
    val_start_1 = val_end_1 - pd.Timedelta(days=horizon - 1)
    train_end_1 = val_start_1 - pd.Timedelta(days=1)
    val_end_2 = train_end_1
    val_start_2 = val_end_2 - pd.Timedelta(days=horizon - 1)
    train_end_2 = val_start_2 - pd.Timedelta(days=1)

    folds = [
        {'train_end': train_end_1, 'val_start': val_start_1, 'val_end': val_end_1, 'label': 'Fold 1 (Recent)'},
        {'train_end': train_end_2, 'val_start': val_start_2, 'val_end': val_end_2, 'label': 'Fold 2 (Past)'},
    ]
    results = []

    for fd in folds:
        cutoff = pd.Timestamp(fd['train_end'])
        vs = pd.Timestamp(fd['val_start'])
        ve = pd.Timestamp(fd['val_end'])
        train_fold = sales_df[sales_df['Date'] <= cutoff].copy()
        val_fold = sales_df[(sales_df['Date'] >= vs) & (sales_df['Date'] <= ve)].copy()

        print(f"\n{fd['label']}:")
        print(f"  Train: {train_fold['Date'].min().date()} to {cutoff.date()} ({len(train_fold)} rows)")
        print(f"  Val:   {vs.date()} to {ve.date()} ({len(val_fold)} rows)")

        # Training: row-level as-of templates
        X_tr, y_tr, feat_names = prepare_training_data(sales_df, daily_comp, BURN_IN_DAYS, cutoff)
        print(f"  Train after burn-in: {X_tr.shape[0]} rows x {X_tr.shape[1]} features")

        model = train_model(X_tr, y_tr)

        # Validation: frozen templates
        monthly_d, dow_d = compute_frozen_templates(daily_comp, cutoff)

        history = train_fold[['Date', 'Revenue']].copy()
        preds, actuals = [], []

        for vi in range(len(val_fold)):
            vd = val_fold.iloc[vi]['Date']
            vr = val_fold.iloc[vi]['Revenue']
            new_row = pd.DataFrame({'Date': [vd], 'Revenue': [np.nan]})
            history = pd.concat([history, new_row], ignore_index=True)
            history = engineer_row_features(history)

            # Apply frozen templates to last row
            row = history.iloc[-1:].copy()
            row['month'] = vd.month
            row['dayofweek'] = vd.dayofweek
            for feat, lkp in monthly_d.items():
                row[feat] = lkp.get(vd.month, np.nan)
            for feat, lkp in dow_d.items():
                row[feat] = lkp.get(vd.dayofweek, np.nan)

            X_last = row[feat_names]
            yp = max(0, model.predict(X_last)[0])
            preds.append(yp)
            actuals.append(vr)
            history.loc[len(history) - 1, 'Revenue'] = yp

            if (vi + 1) % 100 == 0:
                print(f"    Val step {vi + 1}/{len(val_fold)}")

        mae = mean_absolute_error(actuals, preds)
        print(f"  MAE: {mae:,.0f}")
        results.append({'fold': fd['label'], 'MAE': mae})

    mean_mae = np.mean([r['MAE'] for r in results])
    print(f"\n{'='*80}")
    print(f"Backtest Mean MAE: {mean_mae:,.0f}")
    print(f"{'='*80}")
    return results, mean_mae


def run_final_predict(sales_df, test_df, daily_comp, model, feat_names):
    print(f"\nFinal Test Prediction ({len(test_df)} rows)...")
    monthly_d, dow_d = compute_frozen_templates(daily_comp, sales_df['Date'].max())
    history = sales_df[['Date', 'Revenue']].copy()
    preds = []

    for ti in range(len(test_df)):
        td = test_df.iloc[ti]['Date']
        new_row = pd.DataFrame({'Date': [td], 'Revenue': [np.nan]})
        history = pd.concat([history, new_row], ignore_index=True)
        history = engineer_row_features(history)

        row = history.iloc[-1:].copy()
        row['month'] = td.month
        row['dayofweek'] = td.dayofweek
        for feat, lkp in monthly_d.items():
            row[feat] = lkp.get(td.month, np.nan)
        for feat, lkp in dow_d.items():
            row[feat] = lkp.get(td.dayofweek, np.nan)

        X_last = row[feat_names]
        yp = max(0, model.predict(X_last)[0])
        preds.append(yp)
        history.loc[len(history) - 1, 'Revenue'] = yp

        if (ti + 1) % 100 == 0:
            print(f"  Predicted {ti + 1}/{len(test_df)}")

    preds = np.array(preds)
    print(f"  Predictions: min={preds.min():,.0f}, max={preds.max():,.0f}, mean={preds.mean():,.0f}")
    return preds


# ============================================================================
# 9. MAIN
# ============================================================================
def main():
    start = datetime.now()
    sales, test, orders, items, products, horizon = load_all_data()

    print("\nBuilding daily composition series...")
    daily_comp = build_daily_composition(orders, items, products)

    bt_results, bt_mae = run_backtest(sales, daily_comp, horizon)

    print(f"\n{'='*80}")
    print("FINAL TRAINING ON FULL DATA")
    print(f"{'='*80}")
    X_full, y_full, feat_names = prepare_training_data(sales, daily_comp, BURN_IN_DAYS)
    model = train_model(X_full, y_full)

    print(f"\n{'='*80}")
    print("FINAL TEST PREDICTION")
    print(f"{'='*80}")
    preds = run_final_predict(sales, test, daily_comp, model, feat_names)

    sub = pd.DataFrame({
        'Date': test['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': preds,
        'COGS': test['COGS'].values if 'COGS' in test.columns else 0
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'submission_multitable_v1.csv')
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved: {out}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in bt_results:
        print(f"  {r['fold']}: MAE = {r['MAE']:,.0f}")
    print(f"  Mean MAE: {bt_mae:,.0f}")
    print(f"  Single-table v3.2 benchmark: Mean MAE = 1,126,823")
    print(f"  Execution time: {(datetime.now() - start).total_seconds():.1f}s")
    print(f"\n[OK] Complete!")


if __name__ == '__main__':
    main()

"""
Baseline Recursive v3.5 -- Billing Cycle + Tet + Projected Promo
=================================================================
Builds on v3.3. Adds 4 deterministic projectable features:
  - days_to_eom: billing cycle (revenue surges at month boundaries)
  - week_of_month: coarser billing cycle encoding
  - days_to_tet: distance to nearest Lunar New Year
  - is_tet_week: binary flag for Tet +/- 3 days
Key insight: Promotions follow a PERFECTLY REPEATING annual calendar.
  - 4 core campaigns: Spring Sale, Mid-Year Sale, Fall Launch, Year-End Sale (every year)
  - 2 biennial campaigns: Urban Blowout, Rural Special (odd years only)
  - Fixed dates (e.g., Spring Sale always Mar 18 - Apr 17)
  - Fixed discount values
These can be PROJECTED into 2023-2024 as genuine future-known features.

Also adds: cyclical encoding, is_weekend, yoy_ratio, range ratios.
Same model hyperparameters as v3.2 (no model tuning).
"""
import os, warnings, numpy as np, pandas as pd
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

LAG_WINDOWS = [1,2,5,7,14,28,30,60,90,180,364]
ROLLING_WINDOWS = [7,14,30,60,90]
BURN_IN = 364

# ---- Recurring promo calendar (from data analysis) ----
RECURRING_PROMOS = [
    # (name, start_mmdd, end_mmdd, discount_pct, frequency)
    ('spring_sale',   '03-18', '04-17', 12.0, 'annual'),
    ('midyear_sale',  '06-23', '07-22', 18.0, 'annual'),
    ('fall_launch',   '08-30', '10-01', 10.0, 'annual'),
    ('yearend_sale',  '11-18', '01-02', 20.0, 'annual'),   # crosses year boundary
    ('urban_blowout', '07-30', '09-02', 50.0, 'odd'),      # odd years only
    ('rural_special', '01-30', '03-01', 15.0, 'odd'),      # odd years only
]

# Tet (Lunar New Year) dates - deterministic, projectable
TET_DATES = {
    2012:'2012-01-23',2013:'2013-02-10',2014:'2014-01-31',2015:'2015-02-19',
    2016:'2016-02-08',2017:'2017-01-28',2018:'2018-02-16',2019:'2019-02-05',
    2020:'2020-01-25',2021:'2021-02-12',2022:'2022-02-01',2023:'2023-01-22',
    2024:'2024-02-10',2025:'2025-01-29',2026:'2026-02-17',
}

def compute_days_to_tet(date):
    """Compute days to nearest Tet date (past or future)."""
    best = 999
    for yr, td in TET_DATES.items():
        diff = abs((date - pd.Timestamp(td)).days)
        if diff < best:
            best = diff
    return best

print(f"\n{'='*80}")
print("BASELINE RECURSIVE v3.5 -- BILLING CYCLE + TET + PROMO")
print(f"{'='*80}")

# ============================================================================
# 1. BUILD PROJECTED PROMO CALENDAR
# ============================================================================
def build_projected_promo_calendar(min_date, max_date):
    """Build daily promo features for ANY date range, including future."""
    dates = pd.date_range(min_date, max_date, freq='D')
    df = pd.DataFrame({'Date': dates})
    df['promo_active'] = 0
    df['promo_discount_pct'] = 0.0
    df['promo_count'] = 0
    df['max_discount'] = 0.0

    for name, start_md, end_md, disc, freq in RECURRING_PROMOS:
        for year in range(min_date.year, max_date.year + 2):
            # Check frequency
            if freq == 'odd' and year % 2 == 0:
                continue

            try:
                s = pd.Timestamp(f'{year}-{start_md}')
                if end_md < start_md:  # crosses year boundary (yearend_sale)
                    e = pd.Timestamp(f'{year+1}-{end_md}')
                else:
                    e = pd.Timestamp(f'{year}-{end_md}')
            except:
                continue

            mask = (df['Date'] >= s) & (df['Date'] <= e)
            df.loc[mask, 'promo_active'] = 1
            df.loc[mask, 'promo_count'] += 1
            df.loc[mask, 'max_discount'] = np.maximum(df.loc[mask, 'max_discount'], disc)

    # Derived features
    df['days_into_promo'] = 0
    df['days_until_promo_end'] = 0
    in_promo = False
    promo_start_idx = 0
    for i in range(len(df)):
        if df.loc[i, 'promo_active'] == 1:
            if not in_promo:
                promo_start_idx = i
                in_promo = True
            df.loc[i, 'days_into_promo'] = i - promo_start_idx + 1
        else:
            in_promo = False

    # days_until_promo_end: find the next transition from 1->0
    promo_arr = df['promo_active'].values
    dtpe = np.zeros(len(df), dtype=int)
    for i in range(len(df)-1, -1, -1):
        if promo_arr[i] == 1:
            # Find how many more promo days after this
            j = i
            while j < len(df) and promo_arr[j] == 1:
                j += 1
            dtpe[i] = j - i
    df['days_until_promo_end'] = dtpe

    return df


# ============================================================================
# 2. CALENDAR FEATURES
# ============================================================================
def add_calendar(df):
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # NEW: billing cycle features
    df['days_to_eom'] = df['Date'].dt.days_in_month - df['day']
    df['week_of_month'] = (df['day'] - 1) // 7
    # NEW: Tet features
    df['days_to_tet'] = df['Date'].apply(compute_days_to_tet)
    df['is_tet_week'] = (df['days_to_tet'] <= 3).astype(int)
    return df


# ============================================================================
# 3. VECTORIZED TRAINING FEATURES
# ============================================================================
def prepare_train_features(train_df, promo_cal, burnin=BURN_IN):
    df = train_df[['Date','Revenue']].copy()
    df = add_calendar(df)
    sh = df['Revenue'].shift(1)

    for l in LAG_WINDOWS:
        df[f'lag_{l}'] = df['Revenue'].shift(l)
    for w in ROLLING_WINDOWS:
        df[f'roll_mean_{w}'] = sh.rolling(w).mean()
        df[f'roll_std_{w}'] = sh.rolling(w).std(ddof=1)
        df[f'roll_min_{w}'] = sh.rolling(w).min()
        df[f'roll_max_{w}'] = sh.rolling(w).max()
        df[f'roll_median_{w}'] = sh.rolling(w).median()

    df['expanding_mean'] = sh.expanding().mean()
    df['expanding_std'] = sh.expanding().std(ddof=1)

    m7=sh.rolling(7).mean(); m30=sh.rolling(30).mean(); m90=sh.rolling(90).mean()
    df['mean_7_minus_mean_30'] = m7 - m30
    df['mean_7_over_mean_30'] = m7 / (m30 + 1e-6)
    df['mean_30_over_mean_90'] = m30 / (m90 + 1e-6)
    df['volatility_ratio'] = sh.rolling(30).std(ddof=1) / (m30 + 1e-6)
    df['yoy_ratio'] = df['lag_1'] / (df['lag_364'] + 1e-6)

    for w in [30,90]:
        rng = sh.rolling(w).max() - sh.rolling(w).min()
        df[f'roll_range_ratio_{w}'] = rng / (sh.rolling(w).mean() + 1e-6)

    rp = (df['Revenue'] > 0).astype(int)
    df['count_positive_7'] = rp.shift(1).rolling(7).sum()
    df['count_positive_30'] = rp.shift(1).rolling(30).sum()

    dsn = [np.nan]
    for i in range(1, len(df)):
        for j in range(i-1,-1,-1):
            if df['Revenue'].iloc[j] > 0: dsn.append(i-j); break
        else: dsn.append(np.nan)
    df['days_since_nonzero'] = dsn

    def exp_seas(df, k, v):
        r = []
        for i in range(len(df)):
            if i == 0: r.append(np.nan)
            else:
                mask = df.iloc[:i][k] == df.iloc[i][k]
                r.append(df.iloc[:i][mask][v].mean() if mask.any() else np.nan)
        return r
    df['avg_revenue_by_dayofweek'] = exp_seas(df, 'dayofweek', 'Revenue')
    df['avg_revenue_by_month'] = exp_seas(df, 'month', 'Revenue')

    # Merge promo calendar features
    promo_cols = ['promo_active','promo_count','max_discount','days_into_promo','days_until_promo_end']
    df = df.merge(promo_cal[['Date'] + promo_cols], on='Date', how='left')
    for c in promo_cols:
        df[c] = df[c].fillna(0)

    df = df.iloc[burnin:].reset_index(drop=True)
    feat = [c for c in df.columns if c not in ['Date','Revenue','COGS']]

    nan_c = df[feat].isnull().sum()
    nan_c = nan_c[nan_c > 0]
    if len(nan_c) > 0:
        print(f"  NaN after burn-in: {dict(nan_c)}")
    else:
        print(f"  [OK] No NaNs after burn-in.")
    print(f"  Features: {len(feat)}")
    return df[feat], df['Revenue'].values, feat


# ============================================================================
# 4. ROW-BY-ROW INFERENCE
# ============================================================================
def engineer_row(history_df, promo_cal_dict):
    df = history_df.copy()
    df = add_calendar(df)
    idx = len(df) - 1
    dt = df.loc[idx, 'Date']

    for l in LAG_WINDOWS:
        df.loc[idx, f'lag_{l}'] = df.loc[idx-l, 'Revenue'] if idx >= l else np.nan
    for w in ROLLING_WINDOWS:
        if idx >= w:
            p = df.loc[max(0,idx-w):idx-1, 'Revenue'].values
            df.loc[idx, f'roll_mean_{w}'] = p.mean()
            df.loc[idx, f'roll_std_{w}'] = np.std(p,ddof=1) if len(p)>1 else np.nan
            df.loc[idx, f'roll_min_{w}'] = p.min()
            df.loc[idx, f'roll_max_{w}'] = p.max()
            df.loc[idx, f'roll_median_{w}'] = np.median(p)
        else:
            for s in ['mean','std','min','max','median']:
                df.loc[idx, f'roll_{s}_{w}'] = np.nan

    pa = df.loc[0:idx-1,'Revenue'].values
    df.loc[idx,'expanding_mean'] = pa.mean() if len(pa)>0 else np.nan
    df.loc[idx,'expanding_std'] = np.std(pa,ddof=1) if len(pa)>1 else np.nan

    m7 = df.loc[max(0,idx-7):idx-1,'Revenue'].mean() if idx>=7 else np.nan
    m30 = df.loc[max(0,idx-30):idx-1,'Revenue'].mean() if idx>=30 else np.nan
    m90 = df.loc[max(0,idx-90):idx-1,'Revenue'].mean() if idx>=90 else np.nan
    df.loc[idx,'mean_7_minus_mean_30'] = m7-m30 if not(np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_7_over_mean_30'] = m7/(m30+1e-6) if not(np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_30_over_mean_90'] = m30/(m90+1e-6) if not(np.isnan(m30) or np.isnan(m90)) else np.nan
    if idx>=30:
        p30 = df.loc[max(0,idx-30):idx-1,'Revenue'].values
        v30 = np.std(p30,ddof=1) if len(p30)>1 else np.nan
        df.loc[idx,'volatility_ratio'] = v30/(m30+1e-6) if not np.isnan(v30) else np.nan
    else: df.loc[idx,'volatility_ratio'] = np.nan

    l1 = df.loc[idx,'lag_1'] if 'lag_1' in df.columns else np.nan
    l364 = df.loc[idx,'lag_364'] if 'lag_364' in df.columns else np.nan
    df.loc[idx,'yoy_ratio'] = l1/(l364+1e-6) if not(pd.isna(l1) or pd.isna(l364)) else np.nan

    for w in [30,90]:
        if idx>=w:
            pw = df.loc[max(0,idx-w):idx-1,'Revenue'].values
            df.loc[idx,f'roll_range_ratio_{w}'] = (pw.max()-pw.min())/(pw.mean()+1e-6)
        else: df.loc[idx,f'roll_range_ratio_{w}'] = np.nan

    if idx>=7: df.loc[idx,'count_positive_7'] = (df.loc[max(0,idx-7):idx-1,'Revenue'].values>0).sum()
    else: df.loc[idx,'count_positive_7'] = np.nan
    if idx>=30: df.loc[idx,'count_positive_30'] = (df.loc[max(0,idx-30):idx-1,'Revenue'].values>0).sum()
    else: df.loc[idx,'count_positive_30'] = np.nan
    if idx>0:
        for j in range(idx-1,-1,-1):
            if df.loc[j,'Revenue']>0: df.loc[idx,'days_since_nonzero']=idx-j; break
        else: df.loc[idx,'days_since_nonzero']=np.nan
    else: df.loc[idx,'days_since_nonzero']=np.nan

    cd=df.loc[idx,'dayofweek']; cm=df.loc[idx,'month']
    mk=df.loc[0:idx-1,'dayofweek']==cd
    df.loc[idx,'avg_revenue_by_dayofweek']=df.loc[0:idx-1][mk]['Revenue'].mean() if mk.any() else np.nan
    mm=df.loc[0:idx-1,'month']==cm
    df.loc[idx,'avg_revenue_by_month']=df.loc[0:idx-1][mm]['Revenue'].mean() if mm.any() else np.nan

    # Promo features from projected calendar (works for ANY date)
    promo_row = promo_cal_dict.get(dt, {})
    for c in ['promo_active','promo_count','max_discount','days_into_promo','days_until_promo_end']:
        df.loc[idx, c] = promo_row.get(c, 0)

    return df


# ============================================================================
# 5. TRAIN + BACKTEST + PREDICT
# ============================================================================
def train_model(X, y):
    if HAS_LGBM:
        m = lgb.LGBMRegressor(objective='regression',metric='mae',learning_rate=0.05,
            num_leaves=31,n_estimators=300,random_state=RANDOM_STATE,verbose=-1)
    else:
        m = GradientBoostingRegressor(learning_rate=0.05,max_depth=6,n_estimators=300,random_state=RANDOM_STATE)
    m.fit(X,y); return m


def run_backtest(sales, promo_cal, promo_cal_dict, horizon):
    print(f"\n{'='*80}")
    print(f"HORIZON-ALIGNED BACKTEST ({horizon} DAYS PER FOLD)")
    print(f"{'='*80}")

    max_d = sales['Date'].max()
    ve1=max_d; vs1=ve1-pd.Timedelta(days=horizon-1); te1=vs1-pd.Timedelta(days=1)
    ve2=te1; vs2=ve2-pd.Timedelta(days=horizon-1); te2=vs2-pd.Timedelta(days=1)
    folds = [
        {'te':te1,'vs':vs1,'ve':ve1,'label':'Fold 1 (Recent)'},
        {'te':te2,'vs':vs2,'ve':ve2,'label':'Fold 2 (Past)'},
    ]
    results = []

    for fd in folds:
        cutoff=pd.Timestamp(fd['te']); vs=pd.Timestamp(fd['vs']); ve=pd.Timestamp(fd['ve'])
        tf = sales[sales['Date']<=cutoff].copy()
        vf = sales[(sales['Date']>=vs)&(sales['Date']<=ve)].copy()
        pf = promo_cal[promo_cal['Date']<=cutoff]

        print(f"\n{fd['label']}:")
        print(f"  Train: {tf['Date'].min().date()} to {cutoff.date()} ({len(tf)} rows)")
        print(f"  Val:   {vs.date()} to {ve.date()} ({len(vf)} rows)")

        X_tr,y_tr,fnames = prepare_train_features(tf, pf, BURN_IN)
        print(f"  Training: {X_tr.shape[0]} rows x {X_tr.shape[1]} features")
        model = train_model(X_tr, y_tr)

        history = tf[['Date','Revenue']].copy()
        preds, acts = [], []
        for vi in range(len(vf)):
            vd=vf.iloc[vi]['Date']; vr=vf.iloc[vi]['Revenue']
            history=pd.concat([history,pd.DataFrame({'Date':[vd],'Revenue':[np.nan]})],ignore_index=True)
            history=engineer_row(history, promo_cal_dict)
            yp=max(0, model.predict(history.iloc[-1:][fnames])[0])
            preds.append(yp); acts.append(vr)
            history.loc[len(history)-1,'Revenue']=yp
            if(vi+1)%100==0: print(f"    Step {vi+1}/{len(vf)}")

        mae=mean_absolute_error(acts,preds)
        print(f"  MAE: {mae:,.0f}")
        results.append({'fold':fd['label'],'MAE':mae})

    mmae=np.mean([r['MAE'] for r in results])
    print(f"\n{'='*80}")
    print(f"Mean MAE: {mmae:,.0f}")
    print(f"{'='*80}")
    return results, mmae


def run_predict(sales, test, promo_cal, promo_cal_dict, model, fnames):
    print(f"\nRecursive Test Prediction ({len(test)} rows)...")
    history = sales[['Date','Revenue']].copy()
    preds = []
    for ti in range(len(test)):
        td=test.iloc[ti]['Date']
        history=pd.concat([history,pd.DataFrame({'Date':[td],'Revenue':[np.nan]})],ignore_index=True)
        history=engineer_row(history, promo_cal_dict)
        yp=max(0, model.predict(history.iloc[-1:][fnames])[0])
        preds.append(yp)
        history.loc[len(history)-1,'Revenue']=yp
        if(ti+1)%100==0: print(f"  Predicted {ti+1}/{len(test)}")
    preds=np.array(preds)
    print(f"  min={preds.min():,.0f}, max={preds.max():,.0f}, mean={preds.mean():,.0f}")
    return preds


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    start = datetime.now()
    sales = pd.read_csv(os.path.join(DATA_DIR,'sales.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(os.path.join(DATA_DIR,'sample_submission.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    assert (sales['Date'].max()-sales['Date'].min()).days+1 == len(sales)
    horizon = len(test)
    print(f"  Train: {len(sales)} rows, Test: {horizon} days")

    # Build projected promo calendar covering train + test
    min_d = sales['Date'].min()
    max_d = test['Date'].max()
    promo_cal = build_projected_promo_calendar(min_d, max_d)
    promo_cal_dict = {}
    for _, row in promo_cal.iterrows():
        promo_cal_dict[row['Date']] = row.to_dict()

    # Audit promo projection
    train_promo = promo_cal[promo_cal['Date'] <= sales['Date'].max()]['promo_active'].sum()
    test_promo = promo_cal[(promo_cal['Date'] > sales['Date'].max()) & (promo_cal['Date'] <= test['Date'].max())]['promo_active'].sum()
    print(f"  Promo days: {train_promo} train, {test_promo} test (projected)")
    # Show projected test promos
    test_pcal = promo_cal[(promo_cal['Date'] > sales['Date'].max()) & (promo_cal['promo_active']==1)]
    if len(test_pcal) > 0:
        print(f"  Test promo periods:")
        in_promo = False
        ps = None
        for _, row in test_pcal.iterrows():
            if not in_promo:
                ps = row['Date']
                in_promo = True
                pd_val = row['max_discount']
            pe = row['Date']
            # Check if next row is not promo
            next_idx = test_pcal.index.get_loc(_) + 1 if _ != test_pcal.index[-1] else len(test_pcal)
            if next_idx >= len(test_pcal) or (test_pcal.iloc[next_idx]['Date'] - pe).days > 1:
                print(f"    {ps.date()} to {pe.date()} ({(pe-ps).days+1} days, discount={pd_val}%)")
                in_promo = False

    bt_results, bt_mae = run_backtest(sales, promo_cal, promo_cal_dict, horizon)

    print(f"\n{'='*80}")
    print("FINAL TRAINING ON FULL DATA")
    print(f"{'='*80}")
    X_full, y_full, fnames = prepare_train_features(sales, promo_cal, BURN_IN)
    model = train_model(X_full, y_full)

    # Feature importance (top 15)
    if HAS_LGBM:
        imp = pd.Series(model.feature_importances_, index=fnames).sort_values(ascending=False)
        print("\n  Top 15 features:")
        for f, v in imp.head(15).items():
            print(f"    {f}: {v}")

    print(f"\n{'='*80}")
    print("FINAL TEST PREDICTION")
    print(f"{'='*80}")
    preds = run_predict(sales, test, promo_cal, promo_cal_dict, model, fnames)

    sub = pd.DataFrame({
        'Date': test['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': preds,
        'COGS': test['COGS'].values if 'COGS' in test.columns else 0
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'submission_no_cogs_v3_5.csv')
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved: {out}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in bt_results:
        print(f"  {r['fold']}: MAE = {r['MAE']:,.0f}")
    print(f"  Mean MAE: {bt_mae:,.0f}")
    print(f"  v3.2 benchmark: Mean MAE = 1,126,823")
    print(f"  v3.3 benchmark: Mean MAE = 892,527")
    print(f"  Time: {(datetime.now()-start).total_seconds():.1f}s")

if __name__ == '__main__':
    main()

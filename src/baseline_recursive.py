"""
Baseline Recursive -- Dual Target (Revenue + COGS)
===================================================
3 scenarios compared:
  S1 (v3.2-style): calendar + lag/roll of own target only
  S2 (v3.3-style): + projected promo calendar
  S3 (v3.5-style): + billing cycle (days_to_eom, week_of_month) + Tet features
Each scenario trains 2 independent LGBM models (Revenue, COGS) recursively.
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

DATA_DIR = 'data/raw'; OUTPUT_DIR = 'outputs'; RS = 42; np.random.seed(RS)
LAG_W = [1,2,5,7,14,28,30,60,90,180,364]
ROLL_W = [7,14,30,60,90]
BURN = 364

# ---- PROMO CALENDAR ----
PROMOS = [('spring','03-18','04-17',12.0,'annual'),('midyear','06-23','07-22',18.0,'annual'),
('fall','08-30','10-01',10.0,'annual'),('yearend','11-18','01-02',20.0,'annual'),
('urban','07-30','09-02',50.0,'odd'),('rural','01-30','03-01',15.0,'odd')]

def build_promo_cal(mn, mx):
    dates = pd.date_range(mn, mx, freq='D')
    df = pd.DataFrame({'Date': dates}); df['promo_active'] = 0; df['max_discount'] = 0.0; df['promo_count'] = 0
    for _, smd, emd, disc, freq in PROMOS:
        for yr in range(mn.year, mx.year + 2):
            if freq == 'odd' and yr % 2 == 0: continue
            try:
                s = pd.Timestamp(f'{yr}-{smd}')
                e = pd.Timestamp(f'{yr+1}-{emd}') if emd < smd else pd.Timestamp(f'{yr}-{emd}')
            except: continue
            m = (df['Date'] >= s) & (df['Date'] <= e)
            df.loc[m, 'promo_active'] = 1; df.loc[m, 'promo_count'] += 1
            df.loc[m, 'max_discount'] = np.maximum(df.loc[m, 'max_discount'], disc)
    return df

# ---- TET DATES ----
TET = {2012:'2012-01-23',2013:'2013-02-10',2014:'2014-01-31',2015:'2015-02-19',
2016:'2016-02-08',2017:'2017-01-28',2018:'2018-02-16',2019:'2019-02-05',
2020:'2020-01-25',2021:'2021-02-12',2022:'2022-02-01',2023:'2023-01-22',
2024:'2024-02-10',2025:'2025-01-29',2026:'2026-02-17'}

def days_to_tet(date):
    best = 999
    for _, td in TET.items():
        d = abs((date - pd.Timestamp(td)).days)
        if d < best: best = d
    return best

# ---- CALENDAR ----
def add_cal_s1(df):
    """S1: basic calendar (v3.2 style)."""
    df = df.copy()
    df['month'] = df['Date'].dt.month; df['quarter'] = df['Date'].dt.quarter
    df['day'] = df['Date'].dt.day; df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def add_cal_s2(df):
    """S2: + sin/cos + is_weekend (v3.3 style)."""
    df = add_cal_s1(df)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df

def add_cal_s3(df):
    """S3: + billing cycle + Tet (v3.5 style)."""
    df = add_cal_s2(df)
    df['days_to_eom'] = df['Date'].dt.days_in_month - df['day']
    df['week_of_month'] = (df['day'] - 1) // 7
    df['days_to_tet'] = df['Date'].apply(days_to_tet)
    df['is_tet_week'] = (df['days_to_tet'] <= 3).astype(int)
    return df

CAL_FN = {'S1': add_cal_s1, 'S2': add_cal_s2, 'S3': add_cal_s3}

# ---- FEATURE ENGINEERING (target-agnostic) ----
def build_features_vec(df_in, target_col, cal_fn, promo_cal, use_promo, burnin=BURN):
    df = df_in.copy(); df = cal_fn(df)
    sh = df[target_col].shift(1)
    for l in LAG_W: df[f'lag_{l}'] = df[target_col].shift(l)
    for w in ROLL_W:
        df[f'roll_mean_{w}'] = sh.rolling(w).mean()
        df[f'roll_std_{w}'] = sh.rolling(w).std(ddof=1)
        df[f'roll_min_{w}'] = sh.rolling(w).min()
        df[f'roll_max_{w}'] = sh.rolling(w).max()
        df[f'roll_median_{w}'] = sh.rolling(w).median()
    df['expanding_mean'] = sh.expanding().mean()
    df['expanding_std'] = sh.expanding().std(ddof=1)
    m7 = sh.rolling(7).mean(); m30 = sh.rolling(30).mean(); m90 = sh.rolling(90).mean()
    df['m7_m30'] = m7 - m30; df['m7_o_m30'] = m7 / (m30 + 1e-6)
    df['m30_o_m90'] = m30 / (m90 + 1e-6)
    df['vol_ratio'] = sh.rolling(30).std(ddof=1) / (m30 + 1e-6)
    df['yoy_ratio'] = df['lag_1'] / (df['lag_364'] + 1e-6)
    for w in [30, 90]:
        rng = sh.rolling(w).max() - sh.rolling(w).min()
        df[f'range_ratio_{w}'] = rng / (sh.rolling(w).mean() + 1e-6)
    rp = (df[target_col] > 0).astype(int)
    df['cnt_pos_7'] = rp.shift(1).rolling(7).sum()
    df['cnt_pos_30'] = rp.shift(1).rolling(30).sum()
    # Expanding seasonal averages
    def exp_seas(df, k, v):
        r = []
        for i in range(len(df)):
            if i == 0: r.append(np.nan)
            else:
                mk = df.iloc[:i][k] == df.iloc[i][k]
                r.append(df.iloc[:i][mk][v].mean() if mk.any() else np.nan)
        return r
    df['avg_by_dow'] = exp_seas(df, 'dayofweek', target_col)
    df['avg_by_month'] = exp_seas(df, 'month', target_col)
    # Promo
    if use_promo:
        pc = promo_cal[['Date','promo_active','promo_count','max_discount']].copy()
        df = df.merge(pc, on='Date', how='left')
        for c in ['promo_active','promo_count','max_discount']: df[c] = df[c].fillna(0)
    df = df.iloc[burnin:].reset_index(drop=True)
    excl = ['Date','Revenue','COGS']
    feat = [c for c in df.columns if c not in excl]
    return df[feat], df[target_col].values, feat

def engineer_row(hist, idx, target_col, cal_fn, pdict, use_promo):
    df = hist; dt = df.loc[idx, 'Date']
    # Calendar (apply only to current row manually)
    df.loc[idx, 'month'] = dt.month; df.loc[idx, 'quarter'] = dt.quarter
    df.loc[idx, 'day'] = dt.day; df.loc[idx, 'dayofweek'] = dt.dayofweek
    df.loc[idx, 'dayofyear'] = dt.dayofyear
    df.loc[idx, 'is_month_start'] = int(dt.is_month_start)
    df.loc[idx, 'is_month_end'] = int(dt.is_month_end)
    df.loc[idx, 'weekofyear'] = dt.isocalendar().week
    if cal_fn in [add_cal_s2, add_cal_s3]:
        df.loc[idx, 'sin_doy'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
        df.loc[idx, 'cos_doy'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
        df.loc[idx, 'is_weekend'] = int(dt.dayofweek >= 5)
    if cal_fn == add_cal_s3:
        import calendar
        dim = calendar.monthrange(dt.year, dt.month)[1]
        df.loc[idx, 'days_to_eom'] = dim - dt.day
        df.loc[idx, 'week_of_month'] = (dt.day - 1) // 7
        df.loc[idx, 'days_to_tet'] = days_to_tet(dt)
        df.loc[idx, 'is_tet_week'] = int(days_to_tet(dt) <= 3)
    # Lags
    tc = target_col
    for l in LAG_W:
        df.loc[idx, f'lag_{l}'] = df.loc[idx-l, tc] if idx >= l else np.nan
    for w in ROLL_W:
        if idx >= w:
            p = df.loc[max(0, idx-w):idx-1, tc].values
            df.loc[idx, f'roll_mean_{w}'] = p.mean()
            df.loc[idx, f'roll_std_{w}'] = np.std(p, ddof=1) if len(p) > 1 else np.nan
            df.loc[idx, f'roll_min_{w}'] = p.min()
            df.loc[idx, f'roll_max_{w}'] = p.max()
            df.loc[idx, f'roll_median_{w}'] = np.median(p)
        else:
            for s in ['mean','std','min','max','median']:
                df.loc[idx, f'roll_{s}_{w}'] = np.nan
    pa = df.loc[0:idx-1, tc].values
    df.loc[idx, 'expanding_mean'] = pa.mean() if len(pa) > 0 else np.nan
    df.loc[idx, 'expanding_std'] = np.std(pa, ddof=1) if len(pa) > 1 else np.nan
    m7 = df.loc[max(0,idx-7):idx-1, tc].mean() if idx >= 7 else np.nan
    m30 = df.loc[max(0,idx-30):idx-1, tc].mean() if idx >= 30 else np.nan
    m90 = df.loc[max(0,idx-90):idx-1, tc].mean() if idx >= 90 else np.nan
    df.loc[idx, 'm7_m30'] = m7 - m30 if not(np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, 'm7_o_m30'] = m7 / (m30 + 1e-6) if not(np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, 'm30_o_m90'] = m30 / (m90 + 1e-6) if not(np.isnan(m30) or np.isnan(m90)) else np.nan
    if idx >= 30:
        p30 = df.loc[max(0,idx-30):idx-1, tc].values
        df.loc[idx, 'vol_ratio'] = np.std(p30, ddof=1) / (m30 + 1e-6) if len(p30) > 1 else np.nan
    else:
        df.loc[idx, 'vol_ratio'] = np.nan
    l1 = df.loc[idx, 'lag_1'] if not pd.isna(df.loc[idx, 'lag_1']) else np.nan
    l364 = df.loc[idx, 'lag_364'] if idx >= 364 and not pd.isna(df.loc[idx, 'lag_364']) else np.nan
    df.loc[idx, 'yoy_ratio'] = l1 / (l364 + 1e-6) if not(pd.isna(l1) or pd.isna(l364)) else np.nan
    for w in [30, 90]:
        if idx >= w:
            pw = df.loc[max(0,idx-w):idx-1, tc].values
            df.loc[idx, f'range_ratio_{w}'] = (pw.max() - pw.min()) / (pw.mean() + 1e-6)
        else:
            df.loc[idx, f'range_ratio_{w}'] = np.nan
    if idx >= 7: df.loc[idx, 'cnt_pos_7'] = (df.loc[max(0,idx-7):idx-1, tc].values > 0).sum()
    else: df.loc[idx, 'cnt_pos_7'] = np.nan
    if idx >= 30: df.loc[idx, 'cnt_pos_30'] = (df.loc[max(0,idx-30):idx-1, tc].values > 0).sum()
    else: df.loc[idx, 'cnt_pos_30'] = np.nan
    cd = dt.dayofweek; cm = dt.month
    mk = df.loc[0:idx-1, 'dayofweek'] == cd
    df.loc[idx, 'avg_by_dow'] = df.loc[0:idx-1][mk][tc].mean() if mk.any() else np.nan
    mm = df.loc[0:idx-1, 'month'] == cm
    df.loc[idx, 'avg_by_month'] = df.loc[0:idx-1][mm][tc].mean() if mm.any() else np.nan
    if use_promo:
        pr = pdict.get(dt, {})
        for c in ['promo_active','promo_count','max_discount']:
            df.loc[idx, c] = pr.get(c, 0)
    return df

def train_mdl(X, y):
    if HAS_LGBM:
        m = lgb.LGBMRegressor(objective='regression', metric='mae', learning_rate=0.05,
            num_leaves=31, n_estimators=300, random_state=RS, verbose=-1)
    else:
        m = GradientBoostingRegressor(learning_rate=0.05, max_depth=6, n_estimators=300, random_state=RS)
    m.fit(X, y); return m

# ---- SCENARIO RUNNER ----
def run_scenario(scenario, sales, pcal, pdict, H):
    cal_fn = CAL_FN[scenario]
    use_promo = scenario in ['S2', 'S3']
    mx = sales['Date'].max()
    ve1 = mx; vs1 = ve1 - pd.Timedelta(days=H-1); te1 = vs1 - pd.Timedelta(days=1)
    ve2 = te1; vs2 = ve2 - pd.Timedelta(days=H-1); te2 = vs2 - pd.Timedelta(days=1)
    folds = [{'te':te1,'vs':vs1,'ve':ve1,'lbl':'Fold1'},{'te':te2,'vs':vs2,'ve':ve2,'lbl':'Fold2'}]

    all_res = {}
    for target in ['Revenue', 'COGS']:
        fold_maes = []
        for fd in folds:
            co = pd.Timestamp(fd['te']); vs = pd.Timestamp(fd['vs']); ve = pd.Timestamp(fd['ve'])
            tf = sales[sales['Date'] <= co].copy()
            vf = sales[(sales['Date'] >= vs) & (sales['Date'] <= ve)].copy()
            acts = vf[target].values
            X_tr, y_tr, fnames = build_features_vec(tf, target, cal_fn, pcal, use_promo, BURN)
            mdl = train_mdl(X_tr, y_tr)
            hist = tf[['Date', target]].copy()
            preds = []
            for vi in range(len(vf)):
                vd = vf.iloc[vi]['Date']
                hist = pd.concat([hist, pd.DataFrame({'Date': [vd], target: [np.nan]})], ignore_index=True)
                idx = len(hist) - 1
                hist = engineer_row(hist, idx, target, cal_fn, pdict, use_promo)
                yp = max(0, mdl.predict(hist.iloc[-1:][fnames])[0])
                preds.append(yp); hist.loc[idx, target] = yp
            mae = mean_absolute_error(acts, preds)
            fold_maes.append(mae)
            print(f"    {scenario} {target} {fd['lbl']}: MAE={mae:,.0f}")
        mm = np.mean(fold_maes)
        all_res[target] = {'f1': fold_maes[0], 'f2': fold_maes[1], 'mean': mm}
    return all_res

def generate_submission(scenario, sales, test, pcal, pdict):
    cal_fn = CAL_FN[scenario]
    use_promo = scenario in ['S2', 'S3']
    rev_preds = []; cogs_preds = []
    for target, preds_list in [('Revenue', rev_preds), ('COGS', cogs_preds)]:
        X_tr, y_tr, fnames = build_features_vec(sales, target, cal_fn, pcal, use_promo, BURN)
        mdl = train_mdl(X_tr, y_tr)
        hist = sales[['Date', target]].copy()
        for ti in range(len(test)):
            td = test.iloc[ti]['Date']
            hist = pd.concat([hist, pd.DataFrame({'Date': [td], target: [np.nan]})], ignore_index=True)
            idx = len(hist) - 1
            hist = engineer_row(hist, idx, target, cal_fn, pdict, use_promo)
            yp = max(0, mdl.predict(hist.iloc[-1:][fnames])[0])
            preds_list.append(yp); hist.loc[idx, target] = yp
            if (ti + 1) % 100 == 0: print(f"      {scenario} {target}: {ti+1}/{len(test)}")
    return rev_preds, cogs_preds

# ---- MAIN ----
def main():
    t0 = datetime.now()
    sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    H = len(test)
    pcal = build_promo_cal(sales['Date'].min(), test['Date'].max())
    pdict = {r['Date']: r.to_dict() for _, r in pcal.iterrows()}
    print(f"Train={len(sales)}, Test={H}")

    # Backtest all scenarios
    all_results = {}
    for sc in ['S1', 'S2', 'S3']:
        print(f"\n{'='*60}")
        print(f"  SCENARIO {sc}")
        print(f"{'='*60}")
        all_results[sc] = run_scenario(sc, sales, pcal, pdict, H)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Scenario':<10} {'Rev F1':>12} {'Rev F2':>12} {'Rev Mean':>12} | {'COGS F1':>12} {'COGS F2':>12} {'COGS Mean':>12}")
    print('-' * 86)
    for sc in ['S1', 'S2', 'S3']:
        rv = all_results[sc]['Revenue']; cg = all_results[sc]['COGS']
        print(f"{sc:<10} {rv['f1']:>12,.0f} {rv['f2']:>12,.0f} {rv['mean']:>12,.0f} | {cg['f1']:>12,.0f} {cg['f2']:>12,.0f} {cg['mean']:>12,.0f}")

    # Generate submissions for all 3 scenarios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sc in ['S1', 'S2', 'S3']:
        print(f"\n  Generating submission for {sc}...")
        rp, cp = generate_submission(sc, sales, test, pcal, pdict)
        sub = pd.DataFrame({
            'Date': test['Date'].dt.strftime('%Y-%m-%d'),
            'Revenue': rp, 'COGS': cp
        })
        fn = os.path.join(OUTPUT_DIR, f'submission_{sc.lower()}.csv')
        sub.to_csv(fn, index=False)
        print(f"    Saved: {fn}")

    print(f"\nTotal time: {(datetime.now()-t0).total_seconds():.0f}s")

if __name__ == '__main__':
    main()

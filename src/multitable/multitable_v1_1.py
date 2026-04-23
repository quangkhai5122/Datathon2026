"""
Multi-Table v1.1 -- Composition-Only Ablation
4 variants: SINGLE_TABLE, FULL_OLD, COMP_MINIMAL, COMP_PLUS
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

LAG_WINDOWS = [1,2,5,7,14,28,30,60,90,180,364]
ROLLING_WINDOWS = [7,14,30,60,90]
BURN_IN = 364

# ---- TEMPLATE FEATURE GROUP DEFINITIONS ----
# Demand-level (old v1 style)
DEMAND_MONTHLY = ['order_count','avg_order_value','avg_basket_size']
DEMAND_DOW = ['order_count','avg_order_value','avg_basket_size']

# Composition-only features
COMP_MONTHLY_MINIMAL = ['discount_intensity','cancel_rate','category_concentration']
COMP_DOW_MINIMAL = ['discount_intensity']

COMP_MONTHLY_PLUS = ['discount_intensity','cancel_rate','category_concentration','avg_unit_price']
COMP_DOW_PLUS = ['discount_intensity','cancel_rate','category_concentration']

# Variant definitions
VARIANTS = {
    'A_SINGLE_TABLE': {'monthly': [], 'dow': []},
    'B_FULL_OLD':     {'monthly': DEMAND_MONTHLY + COMP_MONTHLY_MINIMAL,
                       'dow': DEMAND_DOW + COMP_DOW_MINIMAL},
    'C_COMP_MINIMAL': {'monthly': COMP_MONTHLY_MINIMAL, 'dow': COMP_DOW_MINIMAL},
    'D_COMP_PLUS':    {'monthly': COMP_MONTHLY_PLUS, 'dow': COMP_DOW_PLUS},
}

# All possible composition metrics we need to compute daily
ALL_COMP_METRICS = list(set(
    DEMAND_MONTHLY + COMP_MONTHLY_PLUS + DEMAND_DOW + COMP_DOW_PLUS
))

print(f"\n{'='*80}")
print("MULTI-TABLE v1.1 -- COMPOSITION-ONLY ABLATION")
print(f"{'='*80}")

# ============================================================================
# DATA LOADING
# ============================================================================
def load_all_data():
    sales = pd.read_csv(os.path.join(DATA_DIR,'sales.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test = pd.read_csv(os.path.join(DATA_DIR,'sample_submission.csv'), parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    orders = pd.read_csv(os.path.join(DATA_DIR,'orders.csv'), parse_dates=['order_date'])
    items = pd.read_csv(os.path.join(DATA_DIR,'order_items.csv'), low_memory=False)
    products = pd.read_csv(os.path.join(DATA_DIR,'products.csv'))
    assert (sales['Date'].max()-sales['Date'].min()).days+1 == len(sales)
    assert test['Date'].nunique() == len(test)
    horizon = len(test)
    print(f"  Sales={len(sales)}, Orders={len(orders)}, Items={len(items)}, Horizon={horizon}")
    return sales, test, orders, items, products, horizon

# ============================================================================
# DAILY COMPOSITION SERIES
# ============================================================================
def build_daily_composition(orders_df, items_df, products_df):
    m = items_df.merge(orders_df[['order_id','order_date','order_status']], on='order_id')
    m = m.merge(products_df[['product_id','category']], on='product_id', how='left')
    m['line_rev'] = m['quantity']*m['unit_price'] - m['discount_amount']
    m['line_gross'] = m['quantity']*m['unit_price']

    d = m.groupby('order_date').agg(
        order_count=('order_id','nunique'),
        total_items=('quantity','sum'),
        total_rev=('line_rev','sum'),
        total_gross=('line_gross','sum'),
        total_discount=('discount_amount','sum'),
        avg_unit_price=('unit_price','mean'),
        unique_products=('product_id','nunique'),
    ).reset_index()

    canc = m[m['order_status']=='cancelled'].groupby('order_date')['order_id'].nunique().reset_index()
    canc.columns = ['order_date','cancel_orders']
    d = d.merge(canc, on='order_date', how='left')
    d['cancel_orders'] = d['cancel_orders'].fillna(0)

    cat_rev = m.groupby(['order_date','category'])['line_rev'].sum().reset_index()
    ct = cat_rev.groupby('order_date')['line_rev'].transform('sum')
    cat_rev['share'] = cat_rev['line_rev']/(ct+1e-9)
    hhi = (cat_rev['share']**2).groupby(cat_rev['order_date']).sum().reset_index()
    hhi.columns = ['order_date','category_hhi']
    d = d.merge(hhi, on='order_date', how='left')

    d['avg_order_value'] = d['total_rev']/(d['order_count']+1e-9)
    d['avg_basket_size'] = d['total_items']/(d['order_count']+1e-9)
    d['discount_intensity'] = d['total_discount']/(d['total_gross']+1e-9)
    d['cancel_rate'] = d['cancel_orders']/(d['order_count']+1e-9)
    d['category_concentration'] = d['category_hhi']

    d = d.rename(columns={'order_date':'Date'}).sort_values('Date').reset_index(drop=True)
    d['month'] = d['Date'].dt.month
    d['dayofweek'] = d['Date'].dt.dayofweek
    keep = ['Date','month','dayofweek'] + [c for c in ALL_COMP_METRICS if c in d.columns]
    return d[keep]

# ============================================================================
# TEMPLATE COMPUTATION
# ============================================================================
def compute_expanding_templates(daily_comp, monthly_cols, dow_cols, cutoff=None):
    df = daily_comp.copy()
    if cutoff: df = df[df['Date']<=cutoff].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    months = df['month'].values
    dows = df['dayofweek'].values

    for col in monthly_cols:
        vals = df[col].values.astype(float)
        res = np.full(len(vals), np.nan)
        s, c = np.zeros(13), np.zeros(13, dtype=int)
        for i in range(len(vals)):
            m = months[i]
            if c[m]>0: res[i] = s[m]/c[m]
            if not np.isnan(vals[i]): s[m]+=vals[i]; c[m]+=1
        df[f'tmpl_{col}_month'] = res

    for col in dow_cols:
        vals = df[col].values.astype(float)
        res = np.full(len(vals), np.nan)
        s, c = np.zeros(7), np.zeros(7, dtype=int)
        for i in range(len(vals)):
            d = dows[i]
            if c[d]>0: res[i] = s[d]/c[d]
            if not np.isnan(vals[i]): s[d]+=vals[i]; c[d]+=1
        df[f'tmpl_{col}_dow'] = res

    tcols = [f'tmpl_{c}_month' for c in monthly_cols]+[f'tmpl_{c}_dow' for c in dow_cols]
    return df[['Date']+tcols]

def compute_frozen_templates(daily_comp, monthly_cols, dow_cols, cutoff):
    df = daily_comp[daily_comp['Date']<=cutoff]
    md = {f'tmpl_{c}_month': df.groupby('month')[c].mean().to_dict() for c in monthly_cols}
    dd = {f'tmpl_{c}_dow': df.groupby('dayofweek')[c].mean().to_dict() for c in dow_cols}
    return md, dd

def apply_frozen(row_date, md, dd):
    feats = {}
    for f,lkp in md.items(): feats[f] = lkp.get(row_date.month, np.nan)
    for f,lkp in dd.items(): feats[f] = lkp.get(row_date.dayofweek, np.nan)
    return feats

# ============================================================================
# SINGLE-TABLE FEATURES (v3.2)
# ============================================================================
def add_calendar(df):
    df = df.copy()
    df['month']=df['Date'].dt.month; df['quarter']=df['Date'].dt.quarter
    df['day']=df['Date'].dt.day; df['dayofweek']=df['Date'].dt.dayofweek
    df['dayofyear']=df['Date'].dt.dayofyear
    df['is_month_start']=df['Date'].dt.is_month_start.astype(int)
    df['is_month_end']=df['Date'].dt.is_month_end.astype(int)
    df['weekofyear']=df['Date'].dt.isocalendar().week.astype(int)
    return df

def add_st_features(df):
    df = add_calendar(df)
    sh = df['Revenue'].shift(1)
    for l in LAG_WINDOWS: df[f'lag_{l}']=df['Revenue'].shift(l)
    for w in ROLLING_WINDOWS:
        df[f'roll_mean_{w}']=sh.rolling(w).mean()
        df[f'roll_std_{w}']=sh.rolling(w).std(ddof=1)
        df[f'roll_min_{w}']=sh.rolling(w).min()
        df[f'roll_max_{w}']=sh.rolling(w).max()
        df[f'roll_median_{w}']=sh.rolling(w).median()
    df['expanding_mean']=sh.expanding().mean()
    df['expanding_std']=sh.expanding().std(ddof=1)
    m7=sh.rolling(7).mean(); m30=sh.rolling(30).mean(); m90=sh.rolling(90).mean()
    df['mean_7_minus_mean_30']=m7-m30
    df['mean_7_over_mean_30']=m7/(m30+1e-6)
    df['mean_30_over_mean_90']=m30/(m90+1e-6)
    df['volatility_ratio']=sh.rolling(30).std(ddof=1)/(m30+1e-6)
    rp=(df['Revenue']>0).astype(int)
    df['count_positive_7']=rp.shift(1).rolling(7).sum()
    df['count_positive_30']=rp.shift(1).rolling(30).sum()
    dsn=[np.nan]
    for i in range(1,len(df)):
        for j in range(i-1,-1,-1):
            if df['Revenue'].iloc[j]>0: dsn.append(i-j); break
        else: dsn.append(np.nan)
    df['days_since_nonzero']=dsn
    def exp_seas(df,k,v):
        r=[]
        for i in range(len(df)):
            if i==0: r.append(np.nan)
            else:
                mask=df.iloc[:i][k]==df.iloc[i][k]
                r.append(df.iloc[:i][mask][v].mean() if mask.any() else np.nan)
        return r
    df['avg_revenue_by_dayofweek']=exp_seas(df,'dayofweek','Revenue')
    df['avg_revenue_by_month']=exp_seas(df,'month','Revenue')
    return df

def engineer_row(history_df):
    df=history_df.copy(); df=add_calendar(df); idx=len(df)-1
    for l in LAG_WINDOWS:
        df.loc[idx,f'lag_{l}']=df.loc[idx-l,'Revenue'] if idx>=l else np.nan
    for w in ROLLING_WINDOWS:
        if idx>=w:
            p=df.loc[max(0,idx-w):idx-1,'Revenue'].values
            df.loc[idx,f'roll_mean_{w}']=p.mean()
            df.loc[idx,f'roll_std_{w}']=np.std(p,ddof=1) if len(p)>1 else np.nan
            df.loc[idx,f'roll_min_{w}']=p.min()
            df.loc[idx,f'roll_max_{w}']=p.max()
            df.loc[idx,f'roll_median_{w}']=np.median(p)
        else:
            for s in['mean','std','min','max','median']: df.loc[idx,f'roll_{s}_{w}']=np.nan
    pa=df.loc[0:idx-1,'Revenue'].values
    df.loc[idx,'expanding_mean']=pa.mean() if len(pa)>0 else np.nan
    df.loc[idx,'expanding_std']=np.std(pa,ddof=1) if len(pa)>1 else np.nan
    m7=df.loc[max(0,idx-7):idx-1,'Revenue'].mean() if idx>=7 else np.nan
    m30=df.loc[max(0,idx-30):idx-1,'Revenue'].mean() if idx>=30 else np.nan
    m90=df.loc[max(0,idx-90):idx-1,'Revenue'].mean() if idx>=90 else np.nan
    df.loc[idx,'mean_7_minus_mean_30']=m7-m30 if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_7_over_mean_30']=m7/(m30+1e-6) if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_30_over_mean_90']=m30/(m90+1e-6) if not(np.isnan(m30)or np.isnan(m90)) else np.nan
    if idx>=30:
        p30=df.loc[max(0,idx-30):idx-1,'Revenue'].values
        v30=np.std(p30,ddof=1) if len(p30)>1 else np.nan
        df.loc[idx,'volatility_ratio']=v30/(m30+1e-6) if not np.isnan(v30) else np.nan
    else: df.loc[idx,'volatility_ratio']=np.nan
    if idx>=7: df.loc[idx,'count_positive_7']=(df.loc[max(0,idx-7):idx-1,'Revenue'].values>0).sum()
    else: df.loc[idx,'count_positive_7']=np.nan
    if idx>=30: df.loc[idx,'count_positive_30']=(df.loc[max(0,idx-30):idx-1,'Revenue'].values>0).sum()
    else: df.loc[idx,'count_positive_30']=np.nan
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
    return df

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================
def prepare_train(sales_df, daily_comp, monthly_cols, dow_cols, cutoff=None):
    df = sales_df[sales_df['Date']<=cutoff].copy() if cutoff else sales_df.copy()
    df = add_st_features(df)
    if monthly_cols or dow_cols:
        tmpl = compute_expanding_templates(daily_comp, monthly_cols, dow_cols, cutoff)
        df = df.merge(tmpl, on='Date', how='left')
    df = df.iloc[BURN_IN:].reset_index(drop=True)
    feat = [c for c in df.columns if c not in ['Date','Revenue','COGS']]
    return df[feat], df['Revenue'].values, feat

# ============================================================================
# TRAIN
# ============================================================================
def train_model(X, y):
    if HAS_LGBM:
        m = lgb.LGBMRegressor(objective='regression',metric='mae',learning_rate=0.05,
            num_leaves=31,n_estimators=300,random_state=RANDOM_STATE,verbose=-1)
    else:
        m = GradientBoostingRegressor(learning_rate=0.05,max_depth=6,n_estimators=300,random_state=RANDOM_STATE)
    m.fit(X,y); return m

# ============================================================================
# BACKTEST ONE VARIANT
# ============================================================================
def backtest_variant(name, sales, daily_comp, horizon, monthly_cols, dow_cols):
    print(f"\n--- Variant {name} ---")
    tcols = [f'tmpl_{c}_month' for c in monthly_cols]+[f'tmpl_{c}_dow' for c in dow_cols]
    print(f"  Template features: {tcols if tcols else 'NONE'}")

    max_d = sales['Date'].max()
    ve1=max_d; vs1=ve1-pd.Timedelta(days=horizon-1); te1=vs1-pd.Timedelta(days=1)
    ve2=te1; vs2=ve2-pd.Timedelta(days=horizon-1); te2=vs2-pd.Timedelta(days=1)
    folds=[
        {'te':te1,'vs':vs1,'ve':ve1,'label':'Fold1(Recent)'},
        {'te':te2,'vs':vs2,'ve':ve2,'label':'Fold2(Past)'},
    ]
    results=[]
    for fd in folds:
        cutoff=pd.Timestamp(fd['te'])
        vstart=pd.Timestamp(fd['vs']); vend=pd.Timestamp(fd['ve'])
        tf=sales[sales['Date']<=cutoff]; vf=sales[(sales['Date']>=vstart)&(sales['Date']<=vend)]
        X_tr,y_tr,fnames=prepare_train(sales,daily_comp,monthly_cols,dow_cols,cutoff)
        model=train_model(X_tr,y_tr)

        if monthly_cols or dow_cols:
            md,dd=compute_frozen_templates(daily_comp,monthly_cols,dow_cols,cutoff)
        else:
            md,dd={},{}

        history=tf[['Date','Revenue']].copy()
        preds,acts=[],[]
        for vi in range(len(vf)):
            vd=vf.iloc[vi]['Date']; vr=vf.iloc[vi]['Revenue']
            history=pd.concat([history,pd.DataFrame({'Date':[vd],'Revenue':[np.nan]})],ignore_index=True)
            history=engineer_row(history)
            row=history.iloc[-1:].copy()
            if md or dd:
                for f,lkp in md.items(): row[f]=lkp.get(vd.month,np.nan)
                for f,lkp in dd.items(): row[f]=lkp.get(vd.dayofweek,np.nan)
            yp=max(0,model.predict(row[fnames])[0])
            preds.append(yp); acts.append(vr)
            history.loc[len(history)-1,'Revenue']=yp
            if(vi+1)%200==0: print(f"    {fd['label']} step {vi+1}/{len(vf)}")

        mae=mean_absolute_error(acts,preds)
        print(f"  {fd['label']}: MAE={mae:,.0f}")
        results.append({'fold':fd['label'],'MAE':mae})

    mmae=np.mean([r['MAE'] for r in results])
    print(f"  Mean MAE: {mmae:,.0f}")
    return results, mmae

# ============================================================================
# FINAL PREDICT
# ============================================================================
def final_predict(sales, test, daily_comp, monthly_cols, dow_cols):
    X_full,y_full,fnames=prepare_train(sales,daily_comp,monthly_cols,dow_cols)
    model=train_model(X_full,y_full)
    if monthly_cols or dow_cols:
        md,dd=compute_frozen_templates(daily_comp,monthly_cols,dow_cols,sales['Date'].max())
    else:
        md,dd={},{}
    history=sales[['Date','Revenue']].copy()
    preds=[]
    for ti in range(len(test)):
        td=test.iloc[ti]['Date']
        history=pd.concat([history,pd.DataFrame({'Date':[td],'Revenue':[np.nan]})],ignore_index=True)
        history=engineer_row(history)
        row=history.iloc[-1:].copy()
        if md or dd:
            for f,lkp in md.items(): row[f]=lkp.get(td.month,np.nan)
            for f,lkp in dd.items(): row[f]=lkp.get(td.dayofweek,np.nan)
        yp=max(0,model.predict(row[fnames])[0])
        preds.append(yp)
        history.loc[len(history)-1,'Revenue']=yp
        if(ti+1)%100==0: print(f"  Predicted {ti+1}/{len(test)}")
    return np.array(preds), fnames

def save_sub(test, preds, fname):
    sub=pd.DataFrame({'Date':test['Date'].dt.strftime('%Y-%m-%d'),'Revenue':preds,
        'COGS':test['COGS'].values if 'COGS' in test.columns else 0})
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    path=os.path.join(OUTPUT_DIR,fname)
    sub.to_csv(path,index=False)
    print(f"  Saved: {path}")
    return path

# ============================================================================
# MAIN
# ============================================================================
def main():
    start=datetime.now()
    sales,test,orders,items,products,horizon=load_all_data()

    print("\nBuilding daily composition...")
    dc=build_daily_composition(orders,items,products)

    # Run all 4 variants
    all_results={}
    for vname, vcfg in VARIANTS.items():
        res, mmae = backtest_variant(vname, sales, dc, horizon, vcfg['monthly'], vcfg['dow'])
        all_results[vname] = {'folds':res, 'mean_mae':mmae}

    # Comparison table
    print(f"\n{'='*80}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'Fold1(Recent)':>15} {'Fold2(Past)':>15} {'Mean MAE':>15}")
    print('-'*70)
    for vn,vr in all_results.items():
        f1=vr['folds'][0]['MAE']; f2=vr['folds'][1]['MAE']
        print(f"{vn:<25} {f1:>15,.0f} {f2:>15,.0f} {vr['mean_mae']:>15,.0f}")

    # Find best composition-only variant
    comp_variants = {k:v for k,v in all_results.items() if k.startswith('C_') or k.startswith('D_')}
    best_comp = min(comp_variants, key=lambda k: comp_variants[k]['mean_mae'])
    best_cfg = VARIANTS[best_comp]

    # Generate submissions for best comp variant and old-style
    print(f"\n{'='*80}")
    print(f"FINAL PREDICTIONS: best={best_comp}")
    print(f"{'='*80}")

    print(f"\nGenerating submission for {best_comp}...")
    preds_best, _ = final_predict(sales, test, dc, best_cfg['monthly'], best_cfg['dow'])
    save_sub(test, preds_best, 'submission_multitable_v1_1_best.csv')

    print(f"\nGenerating submission for B_FULL_OLD...")
    old_cfg = VARIANTS['B_FULL_OLD']
    preds_old, _ = final_predict(sales, test, dc, old_cfg['monthly'], old_cfg['dow'])
    save_sub(test, preds_old, 'submission_multitable_v1_1_oldstyle.csv')

    elapsed=(datetime.now()-start).total_seconds()
    print(f"\nTotal time: {elapsed:.0f}s")
    print("[OK] Complete!")

if __name__=='__main__':
    main()

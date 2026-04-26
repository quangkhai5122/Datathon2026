"""
Dual-target business-constrained experiment.
A=Independent, B=COGS+Margin, C=COGS+LogMarkup, D=Rev with predicted COGS
Composite = (MAE_Rev + MAE_COGS) / 2
"""
import os,sys,warnings,numpy as np,pandas as pd,calendar as cal_mod
from datetime import datetime
try:
    import lightgbm as lgb; HAS_LGBM=True
except: HAS_LGBM=False
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

RECURRING_PROMOS = [
    # (name, start_mmdd, end_mmdd, discount_pct, frequency)
    ('spring_sale',   '03-18', '04-17', 12.0, 'annual'),
    ('midyear_sale',  '06-23', '07-22', 18.0, 'annual'),
    ('fall_launch',   '08-30', '10-01', 10.0, 'annual'),
    ('yearend_sale',  '11-18', '01-02', 20.0, 'annual'), 
    ('urban_blowout', '07-30', '09-02', 50.0, 'odd'),    
    ('rural_special', '01-30', '03-01', 15.0, 'odd'),   
]

TET_DATES = {
    2012:'2012-01-23',2013:'2013-02-10',2014:'2014-01-31',2015:'2015-02-19',
    2016:'2016-02-08',2017:'2017-01-28',2018:'2018-02-16',2019:'2019-02-05',
    2020:'2020-01-25',2021:'2021-02-12',2022:'2022-02-01',2023:'2023-01-22',
    2024:'2024-02-10',2025:'2025-01-29',2026:'2026-02-17',
}
EPS=1.0;MARKUP_CLIP=2.0

def days_to_tet(date):
    """Compute days to nearest Tet date (past or future)."""
    best = 999
    for yr, td in TET_DATES.items():
        diff = abs((date - pd.Timestamp(td)).days)
        if diff < best:
            best = diff
    return best

def build_promo_calendar(min_date, max_date):
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

def add_cal(df):
    df=df.copy()
    df['month']=df['Date'].dt.month
    df['quarter']=df['Date'].dt.quarter
    df['day']=df['Date'].dt.day
    df['dayofweek']=df['Date'].dt.dayofweek
    df['dayofyear']=df['Date'].dt.dayofyear
    df['is_month_start']=df['Date'].dt.is_month_start.astype(int)
    df['is_month_end']=df['Date'].dt.is_month_end.astype(int)
    df['weekofyear']=df['Date'].dt.isocalendar().week.astype(int)
    df['sin_doy']=np.sin(2*np.pi*df['dayofyear']/365.25)
    df['cos_doy']=np.cos(2*np.pi*df['dayofyear']/365.25)
    df['is_weekend']=(df['dayofweek']>=5).astype(int)
    df['days_to_eom']=df['Date'].dt.days_in_month-df['day']
    df['week_of_month']=(df['day']-1)//7
    df['days_to_tet']=df['Date'].apply(days_to_tet)
    df['is_tet_week']=(df['days_to_tet']<=5).astype(int)
    return df

def add_cal_row(df,idx):
    dt=df.loc[idx,'Date']
    df.loc[idx,'month']=dt.month
    df.loc[idx,'quarter']=dt.quarter
    df.loc[idx,'day']=dt.day
    df.loc[idx,'dayofweek']=dt.dayofweek
    df.loc[idx,'dayofyear']=dt.dayofyear
    df.loc[idx,'is_month_start']=int(dt.is_month_start)
    df.loc[idx,'is_month_end']=int(dt.is_month_end)
    df.loc[idx,'weekofyear']=dt.isocalendar().week
    df.loc[idx,'sin_doy']=np.sin(2*np.pi*dt.dayofyear/365.25)
    df.loc[idx,'cos_doy']=np.cos(2*np.pi*dt.dayofyear/365.25)
    df.loc[idx,'is_weekend']=int(dt.dayofweek>=5)
    dim=cal_mod.monthrange(dt.year,dt.month)[1]
    df.loc[idx,'days_to_eom']=dim-dt.day
    df.loc[idx,'week_of_month']=(dt.day-1)//7
    df.loc[idx,'days_to_tet']=days_to_tet(dt)
    df.loc[idx,'is_tet_week']=int(days_to_tet(dt)<=5)
    return df

EXCL={'Date','Revenue','COGS','margin','log_markup','markup'}

def build_features_vec(df_in, target_col, promo_cal, burnin=BURN_IN):
    df=df_in.copy()
    df=add_cal(df)
    sh=df[target_col].shift(1)
    for l in LAG_WINDOWS:
        df[f'lag_{l}'] = df[target_col].shift(l)
    for w in ROLLING_WINDOWS:
        df[f'roll_mean_{w}'] = sh.rolling(w).mean()
        df[f'roll_std_{w}'] = sh.rolling(w).std(ddof=1)
        df[f'roll_min_{w}'] = sh.rolling(w).min()
        df[f'roll_max_{w}'] = sh.rolling(w).max()
        df[f'roll_median_{w}'] = sh.rolling(w).median()
    df['expanding_mean']=sh.expanding().mean()
    df['expanding_std']=sh.expanding().std(ddof=1)
    m7=sh.rolling(7).mean()
    m30=sh.rolling(30).mean()
    m90=sh.rolling(90).mean()
    df['mean_7_minus_mean_30']=m7-m30
    df['mean_7_over_mean_30']=m7/(m30+1e-6)
    df['mean_30_over_mean_90']=m30/(m90+1e-6)
    df['volatility_ratio']=sh.rolling(30).std(ddof=1)/(m30+1e-6)
    df['yoy_ratio']=df['lag_1']/(df['lag_364']+1e-6)
    for w in[30,90]:
        rg=sh.rolling(w).max()-sh.rolling(w).min()
        df[f'roll_range_ratio_{w}']=rg/(sh.rolling(w).mean()+1e-6)
    rp=(df[target_col]>0).astype(int)
    df['count_positive_7']=rp.shift(1).rolling(7).sum()
    df['count_positive_30']=rp.shift(1).rolling(30).sum()
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
    pc = promo_cal[['Date','promo_active','promo_count','max_discount','days_into_promo','days_until_promo_end']].copy()
    df = df.merge(pc, on='Date', how='left')
    for c in ['promo_active','promo_count','max_discount','days_into_promo','days_until_promo_end']: 
        df[c] = df[c].fillna(0)
    df = df.iloc[burnin:].reset_index(drop=True)
    feat=[c for c in df.columns if c not in EXCL]
    print(f"  Features: {len(feat)}")
    return df[feat], df[target_col].values, feat

def row_feats(hist, idx, target_col, pcal_dict):
    """Feature engineering for a single row at index `idx` in `hist`."""
    df=hist.copy()
    if 'dayofweek' not in df.columns:
        df['dayofweek'] = df['Date'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['Date'].dt.month
    df=add_cal_row(df,idx)
    for l in LAG_WINDOWS:
        df.loc[idx,f'lag_{l}']=df.loc[idx-l,target_col] if idx>=l else np.nan
    for w in ROLLING_WINDOWS:
        if idx>=w:
            p = df.loc[max(0, idx-w):idx-1, target_col].values
            df.loc[idx, f'roll_mean_{w}'] = p.mean()
            df.loc[idx, f'roll_std_{w}'] = np.std(p, ddof=1) if len(p) > 1 else np.nan
            df.loc[idx, f'roll_min_{w}'] = p.min()
            df.loc[idx, f'roll_max_{w}'] = p.max()
            df.loc[idx, f'roll_median_{w}'] = np.median(p)
        else:
            for s in ['mean','std','min','max','median']:
                df.loc[idx, f'roll_{s}_{w}'] = np.nan
    pa=df.loc[0:idx-1,target_col].values
    df.loc[idx,'expanding_mean']=pa.mean() if len(pa)>0 else np.nan
    df.loc[idx,'expanding_std']=np.std(pa,ddof=1) if len(pa)>1 else np.nan
    m7=df.loc[max(0,idx-7):idx-1,target_col].mean() if idx>=7 else np.nan
    m30=df.loc[max(0,idx-30):idx-1,target_col].mean() if idx>=30 else np.nan
    m90=df.loc[max(0,idx-90):idx-1,target_col].mean() if idx>=90 else np.nan
    df.loc[idx,'mean_7_minus_mean_30']=m7-m30 if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_7_over_mean_30']=m7/(m30+1e-6) if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'mean_30_over_mean_90']=m30/(m90+1e-6) if not(np.isnan(m30)or np.isnan(m90)) else np.nan
    if idx>=30:
        p30=df.loc[max(0,idx-30):idx-1,target_col].values
        df.loc[idx,'volatility_ratio']=np.std(p30,ddof=1)/(m30+1e-6) if len(p30)>1 else np.nan
    else:df.loc[idx,'volatility_ratio']=np.nan
    l1v=df.loc[idx,'lag_1'] if not pd.isna(df.loc[idx,'lag_1']) else np.nan
    l364v=df.loc[idx,'lag_364'] if idx>=364 and not pd.isna(df.loc[idx,'lag_364']) else np.nan
    df.loc[idx,'yoy_ratio']=l1v/(l364v+1e-6) if not(pd.isna(l1v)or pd.isna(l364v)) else np.nan
    for w in [30,90]:
        if idx>=w:
            pw=df.loc[max(0,idx-w):idx-1,target_col].values
            df.loc[idx,f'roll_range_ratio_{w}']=(pw.max()-pw.min())/(pw.mean()+1e-6)
        else:df.loc[idx,f'roll_range_ratio_{w}']=np.nan
    if idx>=7:
        df.loc[idx,'count_positive_7']=(df.loc[max(0,idx-7):idx-1,target_col].values>0).sum()
    else:
        df.loc[idx,'count_positive_7']=np.nan
    if idx>=30:
        df.loc[idx,'count_positive_30']=(df.loc[max(0,idx-30):idx-1,target_col].values>0).sum()
    else:
        df.loc[idx,'count_positive_30']=np.nan
    cd=df.loc[idx,'Date'].dayofweek
    cm=df.loc[idx,'Date'].month
    mk=df.loc[0:idx-1,'dayofweek']==cd
    df.loc[idx,'avg_by_dow']=df.loc[0:idx-1][mk][target_col].mean() if mk.any() else np.nan
    mm=df.loc[0:idx-1,'month']==cm
    df.loc[idx,'avg_by_month']=df.loc[0:idx-1][mm][target_col].mean() if mm.any() else np.nan
    # Promo
    pr=pcal_dict.get(df.loc[idx,'Date'],{})
    for c in ['promo_active','promo_count','max_discount','days_into_promo','days_until_promo_end']:
        df.loc[idx, c] = pr.get(c, 0)

    return df

def train_model(X, y):
    if HAS_LGBM:
        m = lgb.LGBMRegressor(objective='regression', metric='mae', learning_rate=0.05,
            num_leaves=31, n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
    else:
        m = GradientBoostingRegressor(learning_rate=0.05, max_depth=6, n_estimators=300, random_state=RANDOM_STATE)
    m.fit(X, y); return m

def recursive_predict(sales_col, target_col, mdl, fn, pcal_dict, nrows):
    hist = sales_col.copy()
    preds = []
    for i in range(nrows):
        dt = hist.iloc[-1]['Date']+pd.Timedelta(days=1)
        hist = pd.concat([hist,pd.DataFrame({'Date':[dt],target_col:[np.nan]})],ignore_index=True)
        idx = len(hist)-1
        hist = row_feats(hist,idx,target_col,pcal_dict)
        yp = max(0,mdl.predict(hist.iloc[-1:][fn])[0])
        preds.append(yp)
        hist.loc[idx,target_col] = yp
    return preds,hist

def run_cogs_model(train_df,val_dates,pcal,pcal_dict):
    X,y,fn=build_features_vec(train_df,'COGS',pcal,BURN_IN)
    mdl=train_model(X,y)
    hist=train_df[['Date','COGS']].copy();preds=[]
    for vi in range(len(val_dates)):
        vd=val_dates[vi]
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'COGS':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=row_feats(hist,idx,'COGS',pcal_dict)
        yp=max(0,mdl.predict(hist.iloc[-1:][fn])[0])
        preds.append(yp)
        hist.loc[idx,'COGS']=yp
    return np.array(preds),mdl,fn

def run_rev_model(train_df,val_dates,pcal,pcal_dict):
    X,y,fn=build_features_vec(train_df,'Revenue',pcal,BURN_IN)
    mdl=train_model(X,y)
    hist=train_df[['Date','Revenue']].copy();preds=[]
    for vi in range(len(val_dates)):
        vd=val_dates[vi]
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'Revenue':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=row_feats(hist,idx,'Revenue',pcal_dict)
        yp=max(0,mdl.predict(hist.iloc[-1:][fn])[0])
        preds.append(yp)
        hist.loc[idx,'Revenue']=yp
    return np.array(preds)

# ---- SCENARIO A: INDEPENDENT ----
def scenario_A(train,val_dates,val_rev,val_cogs,pcal,pcal_dict):
    cp=run_cogs_model(train,val_dates,pcal,pcal_dict)[0]
    rp=run_rev_model(train,val_dates,pcal,pcal_dict)
    return rp,cp

# ---- SCENARIO B: COGS + MARGIN ----
def scenario_B(train,val_dates,val_rev,val_cogs,pcal,pcal_dict):
    cp,_,_=run_cogs_model(train,val_dates,pcal,pcal_dict)
    # Train margin model on training data
    tr=train.copy()
    tr['margin'] = tr['Revenue']-tr['COGS']
    X,y,fn=build_features_vec(tr,'margin',pcal,BURN_IN)
    mdl=train_model(X,y)
    hist=tr[['Date','margin']].copy()
    mp=[]
    for vi in range(len(val_dates)):
        vd=val_dates[vi]
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'margin':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=row_feats(hist,idx,'margin',pcal_dict)
        yp=mdl.predict(hist.iloc[-1:][fn])[0]
        mp.append(yp)
        hist.loc[idx,'margin']=yp
    rp=cp+np.array(mp)
    rp=np.maximum(rp,0)
    return rp,cp

# ---- SCENARIO C: COGS + LOG-MARKUP ----
def scenario_C(train,val_dates,val_rev,val_cogs,pcal,pcal_dict):
    cp,_,_=run_cogs_model(train,val_dates,pcal,pcal_dict)
    tr=train.copy()
    tr['log_markup']=np.log((tr['Revenue']+EPS)/(tr['COGS']+EPS))
    tr['log_markup']=tr['log_markup'].clip(-MARKUP_CLIP,MARKUP_CLIP)
    X,y,fn=build_features_vec(tr,'log_markup',pcal,BURN_IN)
    mdl=train_model(X,y)
    hist=tr[['Date','log_markup']].copy()
    lmp=[]
    for vi in range(len(val_dates)):
        vd=val_dates[vi]
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'log_markup':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=row_feats(hist,idx,'log_markup',pcal_dict)
        yp=np.clip(mdl.predict(hist.iloc[-1:][fn])[0],-MARKUP_CLIP,MARKUP_CLIP)
        lmp.append(yp)
        hist.loc[idx,'log_markup']=yp
    rp=(cp+EPS)*np.exp(np.array(lmp))-EPS
    rp=np.maximum(rp,0)
    return rp,cp

# ---- SCENARIO D: REV WITH PREDICTED COGS ----
def scenario_D(train,val_dates,val_rev,val_cogs,pcal,pcal_dict):
    cp,_,_=run_cogs_model(train,val_dates,pcal,pcal_dict)
    # Train revenue model with extra COGS-derived features
    tr=train.copy()
    tr['margin_hist']=tr['Revenue']-tr['COGS']
    tr['markup_hist']=tr['Revenue']/(tr['COGS']+EPS)
    # Build standard revenue features + cross features
    X_base,_,fn_base=build_features_vec(tr,'Revenue',pcal,BURN_IN)
    # Add cross features from COGS history
    tr2=tr.iloc[BURN_IN:].reset_index(drop=True)
    for l in[1,7,30]:
        X_base[f'cogs_l{l}']=tr2['COGS'].shift(l)
        X_base[f'margin_l{l}']=tr2['margin_hist'].shift(l)
        X_base[f'markup_l{l}']=tr2['markup_hist'].shift(l)
    X_base['cogs_rm7']=tr2['COGS'].shift(1).rolling(7).mean()
    X_base['cogs_rm30']=tr2['COGS'].shift(1).rolling(30).mean()
    fn_all=list(X_base.columns)
    y=tr2['Revenue'].values
    mdl=train_model(X_base,y)
    # Recursive inference
    hist_r=tr[['Date','Revenue']].copy()
    hist_c=tr[['Date','COGS']].copy()
    hist_r['margin_hist']=tr['margin_hist'];hist_r['markup_hist']=tr['markup_hist']
    hist_r['COGS']=tr['COGS']
    rp=[]
    for vi in range(len(val_dates)):
        vd=val_dates[vi];cogs_pred=cp[vi]
        hist_r=pd.concat([hist_r,pd.DataFrame({'Date':[vd],'Revenue':[np.nan],'COGS':[cogs_pred],
            'margin_hist':[np.nan],'markup_hist':[np.nan]})],ignore_index=True)
        idx=len(hist_r)-1
        hist_r=row_feats(hist_r,idx,'Revenue',pcal_dict)
        # Cross features
        for l in[1,7,30]:
            hist_r.loc[idx,f'cogs_l{l}']=hist_r.loc[idx-l,'COGS'] if idx>=l else np.nan
            hist_r.loc[idx,f'margin_l{l}']=hist_r.loc[idx-l,'margin_hist'] if idx>=l else np.nan
            hist_r.loc[idx,f'markup_l{l}']=hist_r.loc[idx-l,'markup_hist'] if idx>=l else np.nan
        if idx>=7:hist_r.loc[idx,'cogs_rm7']=hist_r.loc[max(0,idx-7):idx-1,'COGS'].mean()
        else:hist_r.loc[idx,'cogs_rm7']=np.nan
        if idx>=30:hist_r.loc[idx,'cogs_rm30']=hist_r.loc[max(0,idx-30):idx-1,'COGS'].mean()
        else:hist_r.loc[idx,'cogs_rm30']=np.nan
        yp=max(0,mdl.predict(hist_r.iloc[-1:][fn_all])[0])
        rp.append(yp);hist_r.loc[idx,'Revenue']=yp
        hist_r.loc[idx,'margin_hist']=yp-cogs_pred
        hist_r.loc[idx,'markup_hist']=yp/(cogs_pred+EPS)
    return np.array(rp),cp

SCENARIOS={'A_INDEP':scenario_A,'B_MARGIN':scenario_B,'C_LOGMARKUP':scenario_C,'D_CROSSFEAT':scenario_D}

def main():
    t0=datetime.now()
    sales=pd.read_csv(os.path.join(DATA_DIR,'sales.csv'),parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test=pd.read_csv(os.path.join(DATA_DIR,'sample_submission.csv'),parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    H=len(test);pcal=build_promo_calendar(sales['Date'].min(),test['Date'].max())
    pd_={r['Date']:r.to_dict() for _,r in pcal.iterrows()}
    print(f"Train={len(sales)}, Test={H}")
    mx=sales['Date'].max()
    ve1=mx;vs1=ve1-pd.Timedelta(days=H-1);te1=vs1-pd.Timedelta(days=1)
    ve2=te1;vs2=ve2-pd.Timedelta(days=H-1);te2=vs2-pd.Timedelta(days=1)
    folds=[{'te':te1,'vs':vs1,'ve':ve1,'lbl':'F1'},{'te':te2,'vs':vs2,'ve':ve2,'lbl':'F2'}]
    results={}
    for sn,sfn in SCENARIOS.items():
        print(f"\n--- {sn} ---")
        fr,fc=[],[]
        for fd in folds:
            co=pd.Timestamp(fd['te']);vs=pd.Timestamp(fd['vs']);ve=pd.Timestamp(fd['ve'])
            tf=sales[sales['Date']<=co].copy()
            vf=sales[(sales['Date']>=vs)&(sales['Date']<=ve)]
            vd=vf['Date'].values;ar=vf['Revenue'].values;ac=vf['COGS'].values
            rp,cp=sfn(tf,vd,ar,ac,pcal,pd_)
            mr=mean_absolute_error(ar,rp);mc=mean_absolute_error(ac,cp)
            comp=(mr+mc)/2
            print(f"  {fd['lbl']}: Rev={mr:,.0f} COGS={mc:,.0f} Comp={comp:,.0f}")
            fr.append(mr);fc.append(mc)
        results[sn]={'r1':fr[0],'r2':fr[1],'rm':np.mean(fr),'c1':fc[0],'c2':fc[1],'cm':np.mean(fc),
            'comp':(np.mean(fr)+np.mean(fc))/2}
    # Table
    print(f"\n{'='*90}")
    print(f"{'Scenario':<15} {'RevF1':>10} {'RevF2':>10} {'RevM':>10} | {'CogsF1':>10} {'CogsF2':>10} {'CogsM':>10} | {'Comp':>10}")
    print('-'*90)
    for sn,r in results.items():
        print(f"{sn:<15} {r['r1']:>10,.0f} {r['r2']:>10,.0f} {r['rm']:>10,.0f} | {r['c1']:>10,.0f} {r['c2']:>10,.0f} {r['cm']:>10,.0f} | {r['comp']:>10,.0f}")
    # Submissions for top 3
    ranked=sorted(results.items(),key=lambda x:x[1]['comp'])
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    for sn,_ in ranked[:3]:
        print(f"\nSubmission: {sn}...")
        co=sales['Date'].max()
        rp,cp=SCENARIOS[sn](sales,test['Date'].values,None,None,pcal,pd_)
        sub=pd.DataFrame({'Date':test['Date'].dt.strftime('%Y-%m-%d'),'Revenue':rp,'COGS':cp})
        fn=os.path.join(OUTPUT_DIR,f'submission_dual_{sn.lower()}.csv')
        sub.to_csv(fn,index=False);print(f"  Saved: {fn}")
    print(f"\nTime: {(datetime.now()-t0).total_seconds():.0f}s")

if __name__=='__main__':
    main()

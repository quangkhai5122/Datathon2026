"""
v3.4 -- Decomposition + Residual LGBM
Log-additive: log1p(Rev) = level + seasonal_doy + dow + promo + residual
4 variants: A=v3.3 direct, B=baseline only, C=log-additive+LGBM, D=ratio+LGBM
"""
import os,warnings,numpy as np,pandas as pd
from datetime import datetime
try:
    import lightgbm as lgb
    HAS_LGBM=True
except ImportError:
    HAS_LGBM=False
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')
DATA_DIR='data/raw'; OUTPUT_DIR='outputs'; RS=42; np.random.seed(RS)
LAG_W=[1,2,5,7,14,28,30,60,90,180,364]; ROLL_W=[7,14,30,60,90]; BURN=364

PROMOS=[('spring','03-18','04-17',12.0,'annual'),('midyear','06-23','07-22',18.0,'annual'),
('fall','08-30','10-01',10.0,'annual'),('yearend','11-18','01-02',20.0,'annual'),
('urban','07-30','09-02',50.0,'odd'),('rural','01-30','03-01',15.0,'odd')]

def build_promo_cal(mn,mx):
    dates=pd.date_range(mn,mx,freq='D')
    df=pd.DataFrame({'Date':dates}); df['promo_active']=0; df['max_discount']=0.0; df['promo_count']=0
    for _,smd,emd,disc,freq in PROMOS:
        for yr in range(mn.year,mx.year+2):
            if freq=='odd' and yr%2==0: continue
            try:
                s=pd.Timestamp(f'{yr}-{smd}')
                e=pd.Timestamp(f'{yr+1}-{emd}') if emd<smd else pd.Timestamp(f'{yr}-{emd}')
            except: continue
            m=(df['Date']>=s)&(df['Date']<=e)
            df.loc[m,'promo_active']=1; df.loc[m,'promo_count']+=1
            df.loc[m,'max_discount']=np.maximum(df.loc[m,'max_discount'],disc)
    return df

def add_cal(df):
    df=df.copy()
    df['month']=df['Date'].dt.month; df['quarter']=df['Date'].dt.quarter
    df['day']=df['Date'].dt.day; df['dayofweek']=df['Date'].dt.dayofweek
    df['dayofyear']=df['Date'].dt.dayofyear
    df['is_month_start']=df['Date'].dt.is_month_start.astype(int)
    df['is_month_end']=df['Date'].dt.is_month_end.astype(int)
    df['weekofyear']=df['Date'].dt.isocalendar().week.astype(int)
    df['sin_doy']=np.sin(2*np.pi*df['dayofyear']/365.25)
    df['cos_doy']=np.cos(2*np.pi*df['dayofyear']/365.25)
    df['is_weekend']=(df['dayofweek']>=5).astype(int)
    return df

# ---- DECOMPOSITION ----
def fit_baseline(train_df, promo_cal):
    """Fit log-additive baseline from training data only."""
    df=train_df[['Date','Revenue']].copy()
    df=df.merge(promo_cal[['Date','promo_active']],on='Date',how='left')
    df['promo_active']=df['promo_active'].fillna(0)
    df['log_rev']=np.log1p(df['Revenue'])
    df['level']=df['log_rev'].rolling(90,min_periods=30).median()
    frozen_level=df['level'].iloc[-1]
    df['dev']=df['log_rev']-df['level']
    df['doy']=df['Date'].dt.dayofyear
    doy_prof=df.dropna(subset=['dev']).groupby('doy')['dev'].mean().to_dict()
    df['after_seas']=df['dev']-df['doy'].map(doy_prof).fillna(0)
    df['dow']=df['Date'].dt.dayofweek
    dow_prof=df.dropna(subset=['after_seas']).groupby('dow')['after_seas'].mean().to_dict()
    df['after_dow']=df['after_seas']-df['dow'].map(dow_prof).fillna(0)
    pe=df[df['promo_active']==1]['after_dow'].mean()
    if pd.isna(pe): pe=0.0
    return {'level':frozen_level,'doy':doy_prof,'dow':dow_prof,'promo_eff':pe}

def get_baseline_val(date, bp, promo_active):
    b=bp['level']
    b+=bp['doy'].get(date.dayofyear,0)
    b+=bp['dow'].get(date.dayofweek,0)
    if promo_active: b+=bp['promo_eff']
    return b

# ---- FEATURE ENGINEERING ----
def build_features_vec(df_in, target_col, promo_cal, burnin=BURN):
    """Vectorized features for training. target_col is 'Revenue' or 'residual'."""
    df=df_in.copy(); df=add_cal(df)
    sh=df[target_col].shift(1)
    for l in LAG_W: df[f'lag_{l}']=df[target_col].shift(l)
    for w in ROLL_W:
        df[f'roll_mean_{w}']=sh.rolling(w).mean()
        df[f'roll_std_{w}']=sh.rolling(w).std(ddof=1)
        df[f'roll_min_{w}']=sh.rolling(w).min()
        df[f'roll_max_{w}']=sh.rolling(w).max()
        df[f'roll_median_{w}']=sh.rolling(w).median()
    df['expanding_mean']=sh.expanding().mean()
    df['expanding_std']=sh.expanding().std(ddof=1)
    m7=sh.rolling(7).mean();m30=sh.rolling(30).mean();m90=sh.rolling(90).mean()
    df['m7_m30']=m7-m30; df['m7_o_m30']=m7/(m30+1e-6); df['m30_o_m90']=m30/(m90+1e-6)
    df['vol_ratio']=sh.rolling(30).std(ddof=1)/(m30+1e-6)
    df['yoy_ratio']=df['lag_1']/(df['lag_364']+1e-6)
    for w in[30,90]:
        rng=sh.rolling(w).max()-sh.rolling(w).min()
        df[f'range_ratio_{w}']=rng/(sh.rolling(w).mean()+1e-6)
    rp=(df[target_col]>0).astype(int) if target_col=='Revenue' else None
    if rp is not None:
        df['cnt_pos_7']=rp.shift(1).rolling(7).sum()
        df['cnt_pos_30']=rp.shift(1).rolling(30).sum()
    def exp_seas(df,k,v):
        r=[]
        for i in range(len(df)):
            if i==0: r.append(np.nan)
            else:
                mk=df.iloc[:i][k]==df.iloc[i][k]
                r.append(df.iloc[:i][mk][v].mean() if mk.any() else np.nan)
        return r
    df['avg_by_dow']=exp_seas(df,'dayofweek',target_col)
    df['avg_by_month']=exp_seas(df,'month',target_col)
    # promo (skip merge if already present)
    if 'promo_active' not in df.columns:
        pc=promo_cal[['Date','promo_active','promo_count','max_discount']].copy()
        df=df.merge(pc,on='Date',how='left')
    for c in['promo_active','promo_count','max_discount']:
        if c in df.columns: df[c]=df[c].fillna(0)
    df=df.iloc[burnin:].reset_index(drop=True)
    feat=[c for c in df.columns if c not in['Date','Revenue','COGS','residual','log_rev','baseline','ratio','baseline_rev','promo_active_y']]
    return df[feat],df[target_col].values if target_col in df.columns else None,feat

def engineer_row_feats(hist, idx, target_col):
    """Row-level features for recursive inference."""
    df=hist; dt=df.loc[idx,'Date']
    df.loc[idx,'month']=dt.month; df.loc[idx,'quarter']=dt.quarter; df.loc[idx,'day']=dt.day
    df.loc[idx,'dayofweek']=dt.dayofweek; df.loc[idx,'dayofyear']=dt.dayofyear
    df.loc[idx,'is_month_start']=int(dt.is_month_start); df.loc[idx,'is_month_end']=int(dt.is_month_end)
    df.loc[idx,'weekofyear']=dt.isocalendar().week
    df.loc[idx,'sin_doy']=np.sin(2*np.pi*dt.dayofyear/365.25)
    df.loc[idx,'cos_doy']=np.cos(2*np.pi*dt.dayofyear/365.25)
    df.loc[idx,'is_weekend']=int(dt.dayofweek>=5)
    for l in LAG_W:
        df.loc[idx,f'lag_{l}']=df.loc[idx-l,target_col] if idx>=l else np.nan
    for w in ROLL_W:
        if idx>=w:
            p=df.loc[max(0,idx-w):idx-1,target_col].values
            df.loc[idx,f'roll_mean_{w}']=p.mean()
            df.loc[idx,f'roll_std_{w}']=np.std(p,ddof=1) if len(p)>1 else np.nan
            df.loc[idx,f'roll_min_{w}']=p.min(); df.loc[idx,f'roll_max_{w}']=p.max()
            df.loc[idx,f'roll_median_{w}']=np.median(p)
        else:
            for s in['mean','std','min','max','median']: df.loc[idx,f'roll_{s}_{w}']=np.nan
    pa=df.loc[0:idx-1,target_col].values
    df.loc[idx,'expanding_mean']=pa.mean() if len(pa)>0 else np.nan
    df.loc[idx,'expanding_std']=np.std(pa,ddof=1) if len(pa)>1 else np.nan
    m7=df.loc[max(0,idx-7):idx-1,target_col].mean() if idx>=7 else np.nan
    m30=df.loc[max(0,idx-30):idx-1,target_col].mean() if idx>=30 else np.nan
    m90=df.loc[max(0,idx-90):idx-1,target_col].mean() if idx>=90 else np.nan
    df.loc[idx,'m7_m30']=m7-m30 if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'m7_o_m30']=m7/(m30+1e-6) if not(np.isnan(m7)or np.isnan(m30)) else np.nan
    df.loc[idx,'m30_o_m90']=m30/(m90+1e-6) if not(np.isnan(m30)or np.isnan(m90)) else np.nan
    if idx>=30:
        p30=df.loc[max(0,idx-30):idx-1,target_col].values
        df.loc[idx,'vol_ratio']=np.std(p30,ddof=1)/(m30+1e-6) if len(p30)>1 else np.nan
    else: df.loc[idx,'vol_ratio']=np.nan
    l1v=df.loc[idx,'lag_1'] if not pd.isna(df.loc[idx,'lag_1']) else np.nan
    l364v=df.loc[idx,'lag_364'] if idx>=364 and not pd.isna(df.loc[idx,'lag_364']) else np.nan
    df.loc[idx,'yoy_ratio']=l1v/(l364v+1e-6) if not(pd.isna(l1v)or pd.isna(l364v)) else np.nan
    for w in[30,90]:
        if idx>=w:
            pw=df.loc[max(0,idx-w):idx-1,target_col].values
            df.loc[idx,f'range_ratio_{w}']=(pw.max()-pw.min())/(pw.mean()+1e-6)
        else: df.loc[idx,f'range_ratio_{w}']=np.nan
    if target_col=='Revenue':
        if idx>=7: df.loc[idx,'cnt_pos_7']=(df.loc[max(0,idx-7):idx-1,'Revenue'].values>0).sum()
        else: df.loc[idx,'cnt_pos_7']=np.nan
        if idx>=30: df.loc[idx,'cnt_pos_30']=(df.loc[max(0,idx-30):idx-1,'Revenue'].values>0).sum()
        else: df.loc[idx,'cnt_pos_30']=np.nan
    cd=dt.dayofweek;cm=dt.month
    mk=df.loc[0:idx-1,'dayofweek']==cd
    df.loc[idx,'avg_by_dow']=df.loc[0:idx-1][mk][target_col].mean() if mk.any() else np.nan
    mm=df.loc[0:idx-1,'month']==cm
    df.loc[idx,'avg_by_month']=df.loc[0:idx-1][mm][target_col].mean() if mm.any() else np.nan
    return df

def train_mdl(X,y):
    if HAS_LGBM:
        m=lgb.LGBMRegressor(objective='regression',metric='mae',learning_rate=0.05,
            num_leaves=31,n_estimators=300,random_state=RS,verbose=-1)
    else:
        m=GradientBoostingRegressor(learning_rate=0.05,max_depth=6,n_estimators=300,random_state=RS)
    m.fit(X,y); return m

# ---- VARIANT RUNNERS ----
def run_variant_A(sales,pcal,pdict,cutoff,val_df):
    """v3.3 direct."""
    tf=sales[sales['Date']<=cutoff].copy()
    X,y,fn=build_features_vec(tf,'Revenue',pcal,BURN)
    mdl=train_mdl(X,y)
    hist=tf[['Date','Revenue']].copy()
    preds=[]
    for vi in range(len(val_df)):
        vd=val_df.iloc[vi]['Date']
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'Revenue':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=engineer_row_feats(hist,idx,'Revenue')
        pr=pdict.get(vd,{})
        for c in['promo_active','promo_count','max_discount']: hist.loc[idx,c]=pr.get(c,0)
        yp=max(0,mdl.predict(hist.iloc[-1:][fn])[0])
        preds.append(yp); hist.loc[idx,'Revenue']=yp
    return preds

def run_variant_B(sales,pcal,pdict,cutoff,val_df):
    """Baseline only, no LGBM."""
    tf=sales[sales['Date']<=cutoff].copy()
    bp=fit_baseline(tf,pcal[pcal['Date']<=cutoff])
    preds=[]
    for vi in range(len(val_df)):
        vd=val_df.iloc[vi]['Date']
        pa=pdict.get(vd,{}).get('promo_active',0)
        bl=get_baseline_val(vd,bp,pa)
        yp=max(0,np.expm1(bl))
        preds.append(yp)
    return preds

def run_variant_C(sales,pcal,pdict,cutoff,val_df):
    """Log-additive decomp + residual LGBM."""
    tf=sales[sales['Date']<=cutoff].copy()
    bp=fit_baseline(tf,pcal[pcal['Date']<=cutoff])
    # compute residual for training
    tf2=tf.copy()
    tf2=tf2.merge(pcal[['Date','promo_active']],on='Date',how='left')
    tf2['promo_active']=tf2['promo_active'].fillna(0)
    tf2['log_rev']=np.log1p(tf2['Revenue'])
    tf2['baseline']=tf2.apply(lambda r: get_baseline_val(r['Date'],bp,r['promo_active']),axis=1)
    tf2['residual']=tf2['log_rev']-tf2['baseline']
    X,_,fn=build_features_vec(tf2,'residual',pcal,BURN)
    y=tf2.iloc[BURN:]['residual'].values
    mdl=train_mdl(X,y)
    # recursive validation
    hist=tf2[['Date','Revenue','residual']].copy()
    preds=[]
    for vi in range(len(val_df)):
        vd=val_df.iloc[vi]['Date']
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'Revenue':[np.nan],'residual':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=engineer_row_feats(hist,idx,'residual')
        pr=pdict.get(vd,{})
        for c in['promo_active','promo_count','max_discount']: hist.loc[idx,c]=pr.get(c,0)
        res_pred=mdl.predict(hist.iloc[-1:][fn])[0]
        pa=pr.get('promo_active',0)
        bl=get_baseline_val(vd,bp,pa)
        yp=max(0,np.expm1(bl+res_pred))
        preds.append(yp)
        hist.loc[idx,'Revenue']=yp
        hist.loc[idx,'residual']=np.log1p(yp)-bl
    return preds

def run_variant_D(sales,pcal,pdict,cutoff,val_df):
    """Ratio decomp + LGBM."""
    tf=sales[sales['Date']<=cutoff].copy()
    bp=fit_baseline(tf,pcal[pcal['Date']<=cutoff])
    tf2=tf.copy()
    tf2=tf2.merge(pcal[['Date','promo_active']],on='Date',how='left')
    tf2['promo_active']=tf2['promo_active'].fillna(0)
    tf2['baseline']=tf2.apply(lambda r: get_baseline_val(r['Date'],bp,r['promo_active']),axis=1)
    tf2['baseline_rev']=np.expm1(tf2['baseline'])
    tf2['ratio']=tf2['Revenue']/(tf2['baseline_rev']+1e-6)
    X,_,fn=build_features_vec(tf2,'ratio',pcal,BURN)
    y=tf2.iloc[BURN:]['ratio'].values
    mdl=train_mdl(X,y)
    hist=tf2[['Date','Revenue','ratio']].copy()
    preds=[]
    for vi in range(len(val_df)):
        vd=val_df.iloc[vi]['Date']
        hist=pd.concat([hist,pd.DataFrame({'Date':[vd],'Revenue':[np.nan],'ratio':[np.nan]})],ignore_index=True)
        idx=len(hist)-1
        hist=engineer_row_feats(hist,idx,'ratio')
        pr=pdict.get(vd,{})
        for c in['promo_active','promo_count','max_discount']: hist.loc[idx,c]=pr.get(c,0)
        ratio_pred=mdl.predict(hist.iloc[-1:][fn])[0]
        pa=pr.get('promo_active',0)
        bl=get_baseline_val(vd,bp,pa)
        bl_rev=max(1,np.expm1(bl))
        yp=max(0,bl_rev*ratio_pred)
        preds.append(yp)
        hist.loc[idx,'Revenue']=yp
        hist.loc[idx,'ratio']=yp/(bl_rev+1e-6)
    return preds

# ---- MAIN ----
def main():
    t0=datetime.now()
    sales=pd.read_csv(os.path.join(DATA_DIR,'sales.csv'),parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    test=pd.read_csv(os.path.join(DATA_DIR,'sample_submission.csv'),parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    assert (sales['Date'].max()-sales['Date'].min()).days+1==len(sales)
    H=len(test)
    pcal=build_promo_cal(sales['Date'].min(),test['Date'].max())
    pdict={r['Date']:r.to_dict() for _,r in pcal.iterrows()}
    print(f"Train={len(sales)}, Test={H}, Promos projected")

    mx=sales['Date'].max()
    ve1=mx;vs1=ve1-pd.Timedelta(days=H-1);te1=vs1-pd.Timedelta(days=1)
    ve2=te1;vs2=ve2-pd.Timedelta(days=H-1);te2=vs2-pd.Timedelta(days=1)
    folds=[{'te':te1,'vs':vs1,'ve':ve1,'lbl':'Fold1'},{'te':te2,'vs':vs2,'ve':ve2,'lbl':'Fold2'}]

    variants={'A_DIRECT':run_variant_A,'B_BASELINE':run_variant_B,
              'C_LOG_DECOMP':run_variant_C,'D_RATIO_DECOMP':run_variant_D}
    results={}

    for vn,vfn in variants.items():
        print(f"\n--- {vn} ---")
        fold_maes=[]
        for fd in folds:
            co=pd.Timestamp(fd['te']);vs=pd.Timestamp(fd['vs']);ve=pd.Timestamp(fd['ve'])
            vf=sales[(sales['Date']>=vs)&(sales['Date']<=ve)]
            acts=vf['Revenue'].values
            print(f"  {fd['lbl']}: train to {co.date()}, val {vs.date()}-{ve.date()} ({len(vf)} rows)")
            preds=vfn(sales,pcal,pdict,co,vf)
            mae=mean_absolute_error(acts,preds)
            print(f"    MAE: {mae:,.0f}")
            fold_maes.append(mae)
        mm=np.mean(fold_maes)
        print(f"  Mean MAE: {mm:,.0f}")
        results[vn]={'f1':fold_maes[0],'f2':fold_maes[1],'mean':mm}

    print(f"\n{'='*70}")
    print(f"{'Variant':<20} {'Fold1':>12} {'Fold2':>12} {'Mean':>12}")
    print('-'*56)
    for vn,vr in results.items():
        print(f"{vn:<20} {vr['f1']:>12,.0f} {vr['f2']:>12,.0f} {vr['mean']:>12,.0f}")

    # Final submission from best decomp variant
    best_decomp=min(['C_LOG_DECOMP','D_RATIO_DECOMP'],key=lambda k:results[k]['mean'])
    print(f"\nBest decomp: {best_decomp}")
    # Also check if any decomp beats direct
    best_all=min(results,key=lambda k:results[k]['mean'])
    print(f"Best overall: {best_all}")

    # Generate submission from C_LOG_DECOMP for leaderboard testing
    sub_variant = 'C_LOG_DECOMP'
    print(f"\nGenerating submission from {sub_variant}...")
    co=sales['Date'].max()
    sub_fn=variants[sub_variant]
    sub_preds=sub_fn(sales,pcal,pdict,co,test)
    sub=pd.DataFrame({'Date':test['Date'].dt.strftime('%Y-%m-%d'),'Revenue':sub_preds,
        'COGS':test['COGS'].values if 'COGS' in test.columns else 0})
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    out=os.path.join(OUTPUT_DIR,'submission_no_cogs_v3_4.csv')
    sub.to_csv(out,index=False)
    print(f"Saved: {out}")
    print(f"Time: {(datetime.now()-t0).total_seconds():.0f}s")

if __name__=='__main__':
    main()

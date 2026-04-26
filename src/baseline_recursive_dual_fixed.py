"""
Dual-target business-constrained experiment -- FIXED
===================================================
Scenarios:
  A_INDEP      : independent Revenue and COGS recursive models
  B_MARGIN     : predict COGS first, then margin = Revenue - COGS
  C_MARKUP     : predict COGS first, then raw markup = Revenue / COGS
  D_LOGMARKUP  : predict COGS first, then log-markup = log(Revenue / COGS)
  E_CROSSFEAT  : predict COGS first, then Revenue directly using predicted COGS + lagged cross-target features

Important fixes vs baseline_recursive_dual.py:
  1) row_feats uses dayofweek/month consistently; no missing 'dow' column.
  2) Scenario E no longer leaks same-day margin_hist / markup_hist into Revenue training.
  3) Scenario E uses train-time pseudo-predicted COGS for predicted_cogs_t,
     instead of actual same-day COGS.
  4) Added raw markup scenario.
  5) Preserves original sample_submission row order.
"""
import os
import warnings
import calendar as cal_mod
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

DATA_DIR = "data/raw"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LAG_WINDOWS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 364]
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
BURN_IN = 364
EPS = 1.0
MARKUP_CLIP = (0.50, 5.00)
LOG_MARKUP_CLIP = (-1.00, 2.00)
TET_WINDOW_DAYS = 3  # match the best S3/v3.5-style baseline unless changed deliberately

RECURRING_PROMOS = [
    ("spring_sale",   "03-18", "04-17", 12.0, "annual"),
    ("midyear_sale",  "06-23", "07-22", 18.0, "annual"),
    ("fall_launch",   "08-30", "10-01", 10.0, "annual"),
    ("yearend_sale",  "11-18", "01-02", 20.0, "annual"),
    ("urban_blowout", "07-30", "09-02", 50.0, "odd"),
    ("rural_special", "01-30", "03-01", 15.0, "odd"),
]

TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17",
}

BASE_EXCL = {"Date", "Revenue", "COGS", "margin", "markup", "log_markup"}
DERIVED_COLS_TO_EXCLUDE = {
    "margin_hist", "markup_hist", "log_markup_hist", "predicted_cogs_t",
}


def days_to_tet(date: pd.Timestamp) -> int:
    date = pd.Timestamp(date)
    return min(abs((date - pd.Timestamp(td)).days) for td in TET_DATES.values())


def build_promo_calendar(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    df = pd.DataFrame({"Date": dates})
    df["promo_active"] = 0
    df["promo_count"] = 0
    df["max_discount"] = 0.0

    for _, start_md, end_md, disc, freq in RECURRING_PROMOS:
        for year in range(min_date.year, max_date.year + 2):
            if freq == "odd" and year % 2 == 0:
                continue
            s = pd.Timestamp(f"{year}-{start_md}")
            e = pd.Timestamp(f"{year + 1}-{end_md}") if end_md < start_md else pd.Timestamp(f"{year}-{end_md}")
            mask = (df["Date"] >= s) & (df["Date"] <= e)
            df.loc[mask, "promo_active"] = 1
            df.loc[mask, "promo_count"] += 1
            df.loc[mask, "max_discount"] = np.maximum(df.loc[mask, "max_discount"], disc)

    df["days_into_promo"] = 0
    in_promo = False
    promo_start_idx = 0
    for i in range(len(df)):
        if df.loc[i, "promo_active"] == 1:
            if not in_promo:
                promo_start_idx = i
                in_promo = True
            df.loc[i, "days_into_promo"] = i - promo_start_idx + 1
        else:
            in_promo = False

    promo_arr = df["promo_active"].values
    dtpe = np.zeros(len(df), dtype=int)
    for i in range(len(df) - 1, -1, -1):
        if promo_arr[i] == 1:
            j = i
            while j < len(df) and promo_arr[j] == 1:
                j += 1
            dtpe[i] = j - i
    df["days_until_promo_end"] = dtpe
    return df


def add_cal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["days_to_eom"] = df["Date"].dt.days_in_month - df["day"]
    df["week_of_month"] = (df["day"] - 1) // 7
    df["days_to_tet"] = df["Date"].apply(days_to_tet)
    df["is_tet_week"] = (df["days_to_tet"] <= TET_WINDOW_DAYS).astype(int)
    return df


def ensure_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure historical rows also have dayofweek/month for expanding seasonal averages.
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["Date"]).dt.month
    if "dayofweek" not in df.columns:
        df["dayofweek"] = pd.to_datetime(df["Date"]).dt.dayofweek
    return df


def add_cal_row(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    df = ensure_calendar_columns(df)
    dt = pd.Timestamp(df.loc[idx, "Date"])
    df.loc[idx, "month"] = dt.month
    df.loc[idx, "quarter"] = dt.quarter
    df.loc[idx, "day"] = dt.day
    df.loc[idx, "dayofweek"] = dt.dayofweek
    df.loc[idx, "dayofyear"] = dt.dayofyear
    df.loc[idx, "is_month_start"] = int(dt.is_month_start)
    df.loc[idx, "is_month_end"] = int(dt.is_month_end)
    df.loc[idx, "weekofyear"] = dt.isocalendar().week
    df.loc[idx, "sin_doy"] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
    df.loc[idx, "cos_doy"] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
    df.loc[idx, "is_weekend"] = int(dt.dayofweek >= 5)
    dim = cal_mod.monthrange(dt.year, dt.month)[1]
    df.loc[idx, "days_to_eom"] = dim - dt.day
    df.loc[idx, "week_of_month"] = (dt.day - 1) // 7
    df.loc[idx, "days_to_tet"] = days_to_tet(dt)
    df.loc[idx, "is_tet_week"] = int(days_to_tet(dt) <= TET_WINDOW_DAYS)
    return df


def build_features_vec(df_in: pd.DataFrame, target_col: str, promo_cal: pd.DataFrame, burnin: int = BURN_IN) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Vectorized S3/v3.5-style features for one target. Extra leakage-prone columns are excluded."""
    df = df_in.copy()
    df = add_cal(df)
    sh = df[target_col].shift(1)

    for lag in LAG_WINDOWS:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for win in ROLLING_WINDOWS:
        df[f"roll_mean_{win}"] = sh.rolling(win).mean()
        df[f"roll_std_{win}"] = sh.rolling(win).std(ddof=1)
        df[f"roll_min_{win}"] = sh.rolling(win).min()
        df[f"roll_max_{win}"] = sh.rolling(win).max()
        df[f"roll_median_{win}"] = sh.rolling(win).median()

    df["expanding_mean"] = sh.expanding().mean()
    df["expanding_std"] = sh.expanding().std(ddof=1)
    m7 = sh.rolling(7).mean()
    m30 = sh.rolling(30).mean()
    m90 = sh.rolling(90).mean()
    df["mean_7_minus_mean_30"] = m7 - m30
    df["mean_7_over_mean_30"] = m7 / (m30 + 1e-6)
    df["mean_30_over_mean_90"] = m30 / (m90 + 1e-6)
    df["volatility_ratio"] = sh.rolling(30).std(ddof=1) / (m30 + 1e-6)
    df["yoy_ratio"] = df["lag_1"] / (df["lag_364"] + 1e-6)
    for win in [30, 90]:
        rng = sh.rolling(win).max() - sh.rolling(win).min()
        df[f"roll_range_ratio_{win}"] = rng / (sh.rolling(win).mean() + 1e-6)

    pos = (df[target_col] > 0).astype(int)
    df["count_positive_7"] = pos.shift(1).rolling(7).sum()
    df["count_positive_30"] = pos.shift(1).rolling(30).sum()

    def exp_seasonal(key_col: str) -> List[float]:
        out = []
        for i in range(len(df)):
            if i == 0:
                out.append(np.nan)
            else:
                mask = df.iloc[:i][key_col] == df.iloc[i][key_col]
                out.append(df.iloc[:i].loc[mask, target_col].mean() if mask.any() else np.nan)
        return out

    df["avg_by_dow"] = exp_seasonal("dayofweek")
    df["avg_by_month"] = exp_seasonal("month")

    promo_cols = ["promo_active", "promo_count", "max_discount", "days_into_promo", "days_until_promo_end"]
    df = df.merge(promo_cal[["Date"] + promo_cols], on="Date", how="left")
    for col in promo_cols:
        df[col] = df[col].fillna(0)

    df = df.iloc[burnin:].reset_index(drop=True)
    exclude = BASE_EXCL | DERIVED_COLS_TO_EXCLUDE
    feature_names = [c for c in df.columns if c not in exclude]
    return df[feature_names], df[target_col].values, feature_names


def row_feats(hist: pd.DataFrame, idx: int, target_col: str, pcal_dict: Dict[pd.Timestamp, dict]) -> pd.DataFrame:
    """Row-by-row recursive features. Fixes the old 'dow' bug and calendar-history issue."""
    df = hist.copy()
    df = add_cal_row(df, idx)

    for lag in LAG_WINDOWS:
        df.loc[idx, f"lag_{lag}"] = df.loc[idx - lag, target_col] if idx >= lag else np.nan
    for win in ROLLING_WINDOWS:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, target_col].values
            df.loc[idx, f"roll_mean_{win}"] = past.mean()
            df.loc[idx, f"roll_std_{win}"] = np.std(past, ddof=1) if len(past) > 1 else np.nan
            df.loc[idx, f"roll_min_{win}"] = past.min()
            df.loc[idx, f"roll_max_{win}"] = past.max()
            df.loc[idx, f"roll_median_{win}"] = np.median(past)
        else:
            for stat in ["mean", "std", "min", "max", "median"]:
                df.loc[idx, f"roll_{stat}_{win}"] = np.nan

    past_all = df.loc[0:idx - 1, target_col].values
    df.loc[idx, "expanding_mean"] = past_all.mean() if len(past_all) > 0 else np.nan
    df.loc[idx, "expanding_std"] = np.std(past_all, ddof=1) if len(past_all) > 1 else np.nan

    m7 = df.loc[max(0, idx - 7):idx - 1, target_col].mean() if idx >= 7 else np.nan
    m30 = df.loc[max(0, idx - 30):idx - 1, target_col].mean() if idx >= 30 else np.nan
    m90 = df.loc[max(0, idx - 90):idx - 1, target_col].mean() if idx >= 90 else np.nan
    df.loc[idx, "mean_7_minus_mean_30"] = m7 - m30 if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, "mean_7_over_mean_30"] = m7 / (m30 + 1e-6) if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    df.loc[idx, "mean_30_over_mean_90"] = m30 / (m90 + 1e-6) if not (np.isnan(m30) or np.isnan(m90)) else np.nan
    if idx >= 30:
        p30 = df.loc[max(0, idx - 30):idx - 1, target_col].values
        df.loc[idx, "volatility_ratio"] = np.std(p30, ddof=1) / (m30 + 1e-6) if len(p30) > 1 else np.nan
    else:
        df.loc[idx, "volatility_ratio"] = np.nan

    l1v = df.loc[idx, "lag_1"] if "lag_1" in df.columns and not pd.isna(df.loc[idx, "lag_1"]) else np.nan
    l364v = df.loc[idx, "lag_364"] if idx >= 364 and "lag_364" in df.columns and not pd.isna(df.loc[idx, "lag_364"]) else np.nan
    df.loc[idx, "yoy_ratio"] = l1v / (l364v + 1e-6) if not (pd.isna(l1v) or pd.isna(l364v)) else np.nan

    for win in [30, 90]:
        if idx >= win:
            past = df.loc[max(0, idx - win):idx - 1, target_col].values
            df.loc[idx, f"roll_range_ratio_{win}"] = (past.max() - past.min()) / (past.mean() + 1e-6)
        else:
            df.loc[idx, f"roll_range_ratio_{win}"] = np.nan

    df.loc[idx, "count_positive_7"] = (df.loc[max(0, idx - 7):idx - 1, target_col].values > 0).sum() if idx >= 7 else np.nan
    df.loc[idx, "count_positive_30"] = (df.loc[max(0, idx - 30):idx - 1, target_col].values > 0).sum() if idx >= 30 else np.nan

    cur_dow = int(df.loc[idx, "dayofweek"])
    cur_month = int(df.loc[idx, "month"])
    mask_dow = df.loc[0:idx - 1, "dayofweek"] == cur_dow
    mask_month = df.loc[0:idx - 1, "month"] == cur_month
    df.loc[idx, "avg_by_dow"] = df.loc[0:idx - 1].loc[mask_dow, target_col].mean() if mask_dow.any() else np.nan
    df.loc[idx, "avg_by_month"] = df.loc[0:idx - 1].loc[mask_month, target_col].mean() if mask_month.any() else np.nan

    promo = pcal_dict.get(pd.Timestamp(df.loc[idx, "Date"]), {})
    for col in ["promo_active", "promo_count", "max_discount", "days_into_promo", "days_until_promo_end"]:
        df.loc[idx, col] = promo.get(col, 0)
    return df


def train_model(X: pd.DataFrame, y: np.ndarray):
    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            objective="regression", metric="mae", learning_rate=0.05,
            num_leaves=31, n_estimators=300, random_state=RANDOM_STATE, verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            learning_rate=0.05, max_depth=6, n_estimators=300, random_state=RANDOM_STATE
        )
        # sklearn GBDT does not handle NaN. Keep fallback usable.
        X = X.fillna(0)
    model.fit(X, y)
    return model


def model_predict(model, X: pd.DataFrame) -> np.ndarray:
    if HAS_LGBM:
        return model.predict(X)
    return model.predict(X.fillna(0))


def run_cogs_model(train_df: pd.DataFrame, val_dates: np.ndarray, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]) -> Tuple[np.ndarray, object, List[str]]:
    X, y, fnames = build_features_vec(train_df[["Date", "COGS"]].copy(), "COGS", promo_cal, BURN_IN)
    model = train_model(X, y)
    hist = train_df[["Date", "COGS"]].copy()
    preds = []
    for vd in val_dates:
        hist = pd.concat([hist, pd.DataFrame({"Date": [pd.Timestamp(vd)], "COGS": [np.nan]})], ignore_index=True)
        idx = len(hist) - 1
        hist = row_feats(hist, idx, "COGS", pcal_dict)
        yp = max(0, model_predict(model, hist.iloc[-1:][fnames])[0])
        preds.append(yp)
        hist.loc[idx, "COGS"] = yp
    return np.array(preds), model, fnames


def run_revenue_model(train_df: pd.DataFrame, val_dates: np.ndarray, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]) -> np.ndarray:
    X, y, fnames = build_features_vec(train_df[["Date", "Revenue"]].copy(), "Revenue", promo_cal, BURN_IN)
    model = train_model(X, y)
    hist = train_df[["Date", "Revenue"]].copy()
    preds = []
    for vd in val_dates:
        hist = pd.concat([hist, pd.DataFrame({"Date": [pd.Timestamp(vd)], "Revenue": [np.nan]})], ignore_index=True)
        idx = len(hist) - 1
        hist = row_feats(hist, idx, "Revenue", pcal_dict)
        yp = max(0, model_predict(model, hist.iloc[-1:][fnames])[0])
        preds.append(yp)
        hist.loc[idx, "Revenue"] = yp
    return np.array(preds)


def add_historical_cross_features(X: pd.DataFrame, aligned_hist: pd.DataFrame, use_predicted_cogs: bool = False) -> pd.DataFrame:
    """Add lagged/rolling cross-target features. No same-day actual Revenue/COGS leakage."""
    X = X.copy()
    h = aligned_hist.reset_index(drop=True).copy()
    h["margin_hist"] = h["Revenue"] - h["COGS"]
    h["markup_hist"] = h["Revenue"] / (h["COGS"] + EPS)
    h["log_markup_hist"] = np.log((h["Revenue"] + EPS) / (h["COGS"] + EPS))

    for lag in [1, 7, 30]:
        X[f"cogs_lag_{lag}"] = h["COGS"].shift(lag)
        X[f"rev_lag_{lag}"] = h["Revenue"].shift(lag)
        X[f"margin_lag_{lag}"] = h["margin_hist"].shift(lag)
        X[f"markup_lag_{lag}"] = h["markup_hist"].shift(lag)
        X[f"log_markup_lag_{lag}"] = h["log_markup_hist"].shift(lag)

    for win in [7, 30]:
        X[f"cogs_roll_mean_{win}"] = h["COGS"].shift(1).rolling(win).mean()
        X[f"cogs_roll_std_{win}"] = h["COGS"].shift(1).rolling(win).std(ddof=1)
        X[f"margin_roll_mean_{win}"] = h["margin_hist"].shift(1).rolling(win).mean()
        X[f"markup_roll_mean_{win}"] = h["markup_hist"].shift(1).rolling(win).mean()
        X[f"log_markup_roll_mean_{win}"] = h["log_markup_hist"].shift(1).rolling(win).mean()

    if use_predicted_cogs:
        if "predicted_cogs_t" not in h.columns:
            raise ValueError("predicted_cogs_t is required but missing from aligned_hist")
        X["predicted_cogs_t"] = h["predicted_cogs_t"].values
    return X


def add_cross_features_row(row: pd.DataFrame, hist: pd.DataFrame, idx: int, include_predicted_cogs: bool = False) -> pd.DataFrame:
    """Add current-row cross features to one-row feature frame."""
    out = row.copy()
    for lag in [1, 7, 30]:
        out[f"cogs_lag_{lag}"] = hist.loc[idx - lag, "COGS"] if idx >= lag else np.nan
        out[f"rev_lag_{lag}"] = hist.loc[idx - lag, "Revenue"] if idx >= lag else np.nan
        out[f"margin_lag_{lag}"] = hist.loc[idx - lag, "margin_hist"] if idx >= lag else np.nan
        out[f"markup_lag_{lag}"] = hist.loc[idx - lag, "markup_hist"] if idx >= lag else np.nan
        out[f"log_markup_lag_{lag}"] = hist.loc[idx - lag, "log_markup_hist"] if idx >= lag else np.nan

    for win in [7, 30]:
        if idx >= win:
            c = hist.loc[max(0, idx - win):idx - 1, "COGS"].values
            m = hist.loc[max(0, idx - win):idx - 1, "margin_hist"].values
            mk = hist.loc[max(0, idx - win):idx - 1, "markup_hist"].values
            lmk = hist.loc[max(0, idx - win):idx - 1, "log_markup_hist"].values
            out[f"cogs_roll_mean_{win}"] = np.nanmean(c)
            out[f"cogs_roll_std_{win}"] = np.nanstd(c, ddof=1) if np.sum(~np.isnan(c)) > 1 else np.nan
            out[f"margin_roll_mean_{win}"] = np.nanmean(m)
            out[f"markup_roll_mean_{win}"] = np.nanmean(mk)
            out[f"log_markup_roll_mean_{win}"] = np.nanmean(lmk)
        else:
            for name in ["cogs_roll_mean", "cogs_roll_std", "margin_roll_mean", "markup_roll_mean", "log_markup_roll_mean"]:
                out[f"{name}_{win}"] = np.nan

    if include_predicted_cogs:
        out["predicted_cogs_t"] = hist.loc[idx, "COGS"]
    return out


def make_pseudo_predicted_cogs(train_df: pd.DataFrame, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict], n_splits: int = 4) -> pd.Series:
    """
    Time-series pseudo predictions for same-day COGS during Revenue-model training.
    This avoids training Revenue on actual same-day COGS when inference uses predicted COGS.
    Early rows fall back to lag-1 COGS, never actual same-day COGS.
    """
    n = len(train_df)
    pseudo = pd.Series(index=train_df.index, dtype=float)
    pseudo[:] = train_df["COGS"].shift(1)

    min_train = max(BURN_IN + 90, int(n * 0.35))
    if min_train >= n - 1:
        return pseudo

    boundaries = np.linspace(min_train, n, n_splits + 1, dtype=int)
    for k in range(n_splits):
        start = boundaries[k]
        end = boundaries[k + 1]
        if end <= start:
            continue
        sub_train = train_df.iloc[:start].copy()
        val_dates = train_df.iloc[start:end]["Date"].values
        preds, _, _ = run_cogs_model(sub_train, val_dates, promo_cal, pcal_dict)
        pseudo.iloc[start:end] = preds
    return pseudo


# -----------------------------------------------------------------------------
# Scenario A: independent targets
# -----------------------------------------------------------------------------
def scenario_A(train: pd.DataFrame, val_dates: np.ndarray, val_rev, val_cogs, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]) -> Tuple[np.ndarray, np.ndarray]:
    cogs_pred = run_cogs_model(train, val_dates, promo_cal, pcal_dict)[0]
    rev_pred = run_revenue_model(train, val_dates, promo_cal, pcal_dict)
    return rev_pred, cogs_pred


# -----------------------------------------------------------------------------
# Generic transformed-target scenario: margin / markup / log-markup
# -----------------------------------------------------------------------------
def train_transformed_model(train: pd.DataFrame, transform: str, promo_cal: pd.DataFrame):
    tr = train[["Date", "Revenue", "COGS"]].copy()
    tr["margin"] = tr["Revenue"] - tr["COGS"]
    tr["markup"] = (tr["Revenue"] / (tr["COGS"] + EPS)).clip(*MARKUP_CLIP)
    tr["log_markup"] = np.log((tr["Revenue"] + EPS) / (tr["COGS"] + EPS)).clip(*LOG_MARKUP_CLIP)

    X_base, y, fnames = build_features_vec(tr[["Date", transform]].copy(), transform, promo_cal, BURN_IN)
    aligned = tr.iloc[BURN_IN:].reset_index(drop=True)
    X = add_historical_cross_features(X_base, aligned, use_predicted_cogs=False)
    fnames = list(X.columns)
    model = train_model(X, y)
    return model, fnames


def predict_transformed_after_cogs(
    train: pd.DataFrame,
    val_dates: np.ndarray,
    cogs_pred: np.ndarray,
    transform: str,
    model,
    fnames: List[str],
    pcal_dict: Dict[pd.Timestamp, dict],
) -> Tuple[np.ndarray, np.ndarray]:
    hist = train[["Date", "Revenue", "COGS"]].copy()
    hist["margin_hist"] = hist["Revenue"] - hist["COGS"]
    hist["markup_hist"] = hist["Revenue"] / (hist["COGS"] + EPS)
    hist["log_markup_hist"] = np.log((hist["Revenue"] + EPS) / (hist["COGS"] + EPS))
    hist[transform] = hist[transform + "_hist"] if transform != "margin" else hist["margin_hist"]

    tpreds, rev_preds = [], []
    for i, vd in enumerate(val_dates):
        vd = pd.Timestamp(vd)
        new = pd.DataFrame({
            "Date": [vd], "Revenue": [np.nan], "COGS": [float(cogs_pred[i])],
            "margin_hist": [np.nan], "markup_hist": [np.nan], "log_markup_hist": [np.nan],
            transform: [np.nan],
        })
        hist = pd.concat([hist, new], ignore_index=True)
        idx = len(hist) - 1
        hist = row_feats(hist, idx, transform, pcal_dict)
        row = add_cross_features_row(hist.iloc[-1:][[c for c in hist.columns if c in fnames or c in ["Date", transform]]].copy(), hist, idx, include_predicted_cogs=False)
        # Ensure all expected columns exist.
        for col in fnames:
            if col not in row.columns:
                row[col] = np.nan
        pred_t = model_predict(model, row[fnames])[0]
        if transform == "markup":
            pred_t = float(np.clip(pred_t, *MARKUP_CLIP))
            rev = float(cogs_pred[i]) * pred_t
        elif transform == "log_markup":
            pred_t = float(np.clip(pred_t, *LOG_MARKUP_CLIP))
            rev = (float(cogs_pred[i]) + EPS) * np.exp(pred_t) - EPS
        elif transform == "margin":
            rev = float(cogs_pred[i]) + float(pred_t)
        else:
            raise ValueError(transform)
        rev = max(0.0, rev)
        tpreds.append(pred_t)
        rev_preds.append(rev)
        hist.loc[idx, "Revenue"] = rev
        hist.loc[idx, "margin_hist"] = rev - float(cogs_pred[i])
        hist.loc[idx, "markup_hist"] = rev / (float(cogs_pred[i]) + EPS)
        hist.loc[idx, "log_markup_hist"] = np.log((rev + EPS) / (float(cogs_pred[i]) + EPS))
        hist.loc[idx, transform] = pred_t
    return np.array(rev_preds), np.array(tpreds)


def scenario_B(train, val_dates, val_rev, val_cogs, promo_cal, pcal_dict):
    cogs_pred = run_cogs_model(train, val_dates, promo_cal, pcal_dict)[0]
    model, fnames = train_transformed_model(train, "margin", promo_cal)
    rev_pred, _ = predict_transformed_after_cogs(train, val_dates, cogs_pred, "margin", model, fnames, pcal_dict)
    return rev_pred, cogs_pred


def scenario_C(train, val_dates, val_rev, val_cogs, promo_cal, pcal_dict):
    # Raw markup scenario added.
    cogs_pred = run_cogs_model(train, val_dates, promo_cal, pcal_dict)[0]
    model, fnames = train_transformed_model(train, "markup", promo_cal)
    rev_pred, _ = predict_transformed_after_cogs(train, val_dates, cogs_pred, "markup", model, fnames, pcal_dict)
    return rev_pred, cogs_pred


def scenario_D(train, val_dates, val_rev, val_cogs, promo_cal, pcal_dict):
    # Log-markup scenario.
    cogs_pred = run_cogs_model(train, val_dates, promo_cal, pcal_dict)[0]
    model, fnames = train_transformed_model(train, "log_markup", promo_cal)
    rev_pred, _ = predict_transformed_after_cogs(train, val_dates, cogs_pred, "log_markup", model, fnames, pcal_dict)
    return rev_pred, cogs_pred


# -----------------------------------------------------------------------------
# Scenario E: direct Revenue model with prediction-based same-day COGS feature.
# -----------------------------------------------------------------------------
def train_revenue_cross_model(train: pd.DataFrame, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]):
    tr = train[["Date", "Revenue", "COGS"]].copy()
    tr["predicted_cogs_t"] = make_pseudo_predicted_cogs(tr, promo_cal, pcal_dict)

    # Build pure Revenue S3 features first. This avoids same-day margin/markup leakage.
    X_base, _, _ = build_features_vec(tr[["Date", "Revenue"]].copy(), "Revenue", promo_cal, BURN_IN)
    aligned = tr.iloc[BURN_IN:].reset_index(drop=True)
    X = add_historical_cross_features(X_base, aligned, use_predicted_cogs=True)
    y = aligned["Revenue"].values
    fnames = list(X.columns)
    model = train_model(X, y)
    return model, fnames


def scenario_E(train, val_dates, val_rev, val_cogs, promo_cal, pcal_dict):
    cogs_pred = run_cogs_model(train, val_dates, promo_cal, pcal_dict)[0]
    model, fnames = train_revenue_cross_model(train, promo_cal, pcal_dict)

    hist = train[["Date", "Revenue", "COGS"]].copy()
    hist["margin_hist"] = hist["Revenue"] - hist["COGS"]
    hist["markup_hist"] = hist["Revenue"] / (hist["COGS"] + EPS)
    hist["log_markup_hist"] = np.log((hist["Revenue"] + EPS) / (hist["COGS"] + EPS))

    rev_preds = []
    for i, vd in enumerate(val_dates):
        vd = pd.Timestamp(vd)
        new = pd.DataFrame({
            "Date": [vd], "Revenue": [np.nan], "COGS": [float(cogs_pred[i])],
            "margin_hist": [np.nan], "markup_hist": [np.nan], "log_markup_hist": [np.nan],
        })
        hist = pd.concat([hist, new], ignore_index=True)
        idx = len(hist) - 1
        hist = row_feats(hist, idx, "Revenue", pcal_dict)
        row = hist.iloc[-1:].copy()
        row = add_cross_features_row(row, hist, idx, include_predicted_cogs=True)
        for col in fnames:
            if col not in row.columns:
                row[col] = np.nan
        yp = max(0.0, model_predict(model, row[fnames])[0])
        rev_preds.append(yp)
        hist.loc[idx, "Revenue"] = yp
        hist.loc[idx, "margin_hist"] = yp - float(cogs_pred[i])
        hist.loc[idx, "markup_hist"] = yp / (float(cogs_pred[i]) + EPS)
        hist.loc[idx, "log_markup_hist"] = np.log((yp + EPS) / (float(cogs_pred[i]) + EPS))
    return np.array(rev_preds), cogs_pred


SCENARIOS = {
    "A_INDEP": scenario_A,
    "B_MARGIN": scenario_B,
    "C_MARKUP": scenario_C,
    "D_LOGMARKUP": scenario_D,
    "E_CROSSFEAT": scenario_E,
}


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), parse_dates=["Date"])
    test_raw["__orig_order"] = np.arange(len(test_raw))
    test_sorted = test_raw.sort_values("Date").reset_index(drop=True)
    assert test_sorted["Date"].is_unique, "This script assumes unique daily dates in sample_submission.csv"
    return sales, test_raw, test_sorted


def build_folds(sales: pd.DataFrame, horizon: int):
    max_date = sales["Date"].max()
    val_end_1 = max_date
    val_start_1 = val_end_1 - pd.Timedelta(days=horizon - 1)
    train_end_1 = val_start_1 - pd.Timedelta(days=1)
    val_end_2 = train_end_1
    val_start_2 = val_end_2 - pd.Timedelta(days=horizon - 1)
    train_end_2 = val_start_2 - pd.Timedelta(days=1)
    return [
        {"train_end": train_end_1, "val_start": val_start_1, "val_end": val_end_1, "label": "F1"},
        {"train_end": train_end_2, "val_start": val_start_2, "val_end": val_end_2, "label": "F2"},
    ]


def evaluate_scenarios(sales: pd.DataFrame, test_sorted: pd.DataFrame, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]):
    horizon = len(test_sorted)
    folds = build_folds(sales, horizon)
    results = {}
    for name, func in SCENARIOS.items():
        print(f"\n--- {name} ---")
        rev_maes, cogs_maes = [], []
        for fold in folds:
            cutoff = pd.Timestamp(fold["train_end"])
            vs = pd.Timestamp(fold["val_start"])
            ve = pd.Timestamp(fold["val_end"])
            train_fold = sales[sales["Date"] <= cutoff].copy()
            val_fold = sales[(sales["Date"] >= vs) & (sales["Date"] <= ve)].copy()
            rev_pred, cogs_pred = func(
                train_fold, val_fold["Date"].values,
                val_fold["Revenue"].values, val_fold["COGS"].values,
                promo_cal, pcal_dict,
            )
            mr = mean_absolute_error(val_fold["Revenue"].values, rev_pred)
            mc = mean_absolute_error(val_fold["COGS"].values, cogs_pred)
            print(f"  {fold['label']}: Rev={mr:,.0f} COGS={mc:,.0f} Comp={(mr + mc) / 2:,.0f}")
            rev_maes.append(mr)
            cogs_maes.append(mc)
        results[name] = {
            "rev_f1": rev_maes[0], "rev_f2": rev_maes[1], "rev_mean": float(np.mean(rev_maes)),
            "cogs_f1": cogs_maes[0], "cogs_f2": cogs_maes[1], "cogs_mean": float(np.mean(cogs_maes)),
            "composite": float((np.mean(rev_maes) + np.mean(cogs_maes)) / 2),
        }
    return results


def save_submission_for_scenario(name: str, func, sales: pd.DataFrame, test_raw: pd.DataFrame, test_sorted: pd.DataFrame, promo_cal: pd.DataFrame, pcal_dict: Dict[pd.Timestamp, dict]) -> str:
    rev_pred, cogs_pred = func(sales.copy(), test_sorted["Date"].values, None, None, promo_cal, pcal_dict)
    pred_sorted = pd.DataFrame({
        "Date": test_sorted["Date"].values,
        "Revenue_pred": rev_pred,
        "COGS_pred": cogs_pred,
    })
    out_df = test_raw[["Date", "__orig_order"]].merge(pred_sorted, on="Date", how="left").sort_values("__orig_order")
    sub = pd.DataFrame({
        "Date": out_df["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": out_df["Revenue_pred"].clip(lower=0).values,
        "COGS": out_df["COGS_pred"].clip(lower=0).values,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"submission_dual_{name.lower()}.csv")
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def main():
    start = datetime.now()
    sales, test_raw, test_sorted = load_data()
    horizon = len(test_sorted)
    promo_cal = build_promo_calendar(sales["Date"].min(), test_sorted["Date"].max())
    pcal_dict = {pd.Timestamp(row["Date"]): row.to_dict() for _, row in promo_cal.iterrows()}
    print(f"Train={len(sales)}, Test={horizon}, LightGBM={HAS_LGBM}")

    results = evaluate_scenarios(sales, test_sorted, promo_cal, pcal_dict)

    print(f"\n{'=' * 110}")
    print(f"{'Scenario':<15} {'RevF1':>11} {'RevF2':>11} {'RevMean':>11} | {'COGSF1':>11} {'COGSF2':>11} {'COGSMean':>11} | {'Composite':>11}")
    print("-" * 110)
    for name, r in results.items():
        print(f"{name:<15} {r['rev_f1']:>11,.0f} {r['rev_f2']:>11,.0f} {r['rev_mean']:>11,.0f} | {r['cogs_f1']:>11,.0f} {r['cogs_f2']:>11,.0f} {r['cogs_mean']:>11,.0f} | {r['composite']:>11,.0f}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["composite"])
    print(f"\nTop 3 scenarios by composite MAE: {[x[0] for x in ranked[:3]]}")
    for name, _ in ranked[:3]:
        print(f"\nSubmission: {name}...")
        save_submission_for_scenario(name, SCENARIOS[name], sales, test_raw, test_sorted, promo_cal, pcal_dict)

    print(f"\nTime: {(datetime.now() - start).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

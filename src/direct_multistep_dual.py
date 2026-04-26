"""
Direct multi-step experiment for DATATHON 2026 Round 1
=====================================================

Two non-recursive/direct-horizon scenarios:

1) DIRECT_REV_COGS
   - Direct model for Revenue(h)
   - Direct model for COGS(h)

2) DIRECT_REV_RATIO
   - Direct model for Revenue(h)
   - Direct model for ratio(h) = COGS / Revenue
   - Reconstruct COGS_hat = Revenue_hat * ratio_hat

Design principles:
- No recursive 548-step feedback loop.
- Each training row is (origin_date, horizon_h) -> target_date = origin_date + h.
- All features are known at origin_date or deterministic for target_date.
- Uses target-date calendar, projected promo calendar, origin-state features, seasonal anchors,
  and trailing seasonal templates.
- Horizon-aligned validation with the same test horizon length.

Expected data files:
- data/raw/sales.csv with columns Date, Revenue, COGS
- data/raw/sample_submission.csv with columns Date, Revenue, COGS

Outputs:
- outputs/submission_direct_rev_cogs.csv
- outputs/submission_direct_rev_ratio.csv
- outputs/direct_multistep_summary.csv
"""

from __future__ import annotations

import os
import math
import calendar as cal_mod
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = "data/raw"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Direct training controls
ORIGIN_STEP_DAYS = 7          # use weekly origins to keep training table manageable
MIN_HISTORY_DAYS = 728        # enough history for lag_728 anchors/templates
TRAILING_TEMPLATE_DAYS = 730  # recent seasonal templates from last ~2 years

# Feature switches
USE_PROMO = True
USE_TET = True
USE_VN_HOLIDAY_BASIC = True
USE_RATIO_SCENARIO = True

# Direct feature windows
ORIGIN_LAGS = [1, 2, 5, 7, 14, 28, 30, 60, 90, 180, 364, 728]
ORIGIN_ROLL_WINDOWS = [7, 14, 30, 60, 90, 180]
SEASONAL_ANCHOR_LAGS = [364, 365, 728, 729, 1092]

EPS = 1.0
RATIO_CLIP_FALLBACK = (0.50, 1.50)

# LightGBM defaults. Kept simple; direct formulation is the main experiment.
LGB_PARAMS = dict(
    objective="regression_l1",
    metric="mae",
    learning_rate=0.03,
    num_leaves=31,
    n_estimators=1200,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    verbose=-1,
)

# Recurring promo calendar inferred from the provided promotions history.
RECURRING_PROMOS = [
    # name, start_mmdd, end_mmdd, discount_pct, frequency
    ("spring_sale",   "03-18", "04-17", 12.0, "annual"),
    ("midyear_sale",  "06-23", "07-22", 18.0, "annual"),
    ("fall_launch",   "08-30", "10-01", 10.0, "annual"),
    ("yearend_sale",  "11-18", "01-02", 20.0, "annual"),  # crosses year boundary
    ("urban_blowout", "07-30", "09-02", 50.0, "odd"),
    ("rural_special", "01-30", "03-01", 15.0, "odd"),
]

TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17",
}

# A lightweight deterministic holiday set. This is intentionally simple.
VN_FIXED_HOLIDAYS = [
    (1, 1),   # New Year
    (4, 30),  # Reunification Day
    (5, 1),   # International Workers' Day
    (9, 2),   # National Day
]


# =============================================================================
# Utilities
# =============================================================================
@dataclass
class FoldSpec:
    label: str
    origin: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def ensure_lgbm() -> None:
    if not HAS_LGBM:
        raise ImportError(
            "LightGBM is required for this direct multi-step experiment because "
            "the feature matrix intentionally contains NaNs for unavailable seasonal anchors. "
            "Install it with: pip install lightgbm"
        )


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"), parse_dates=["Date"])
    test = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), parse_dates=["Date"])

    # Keep original test order for submission, but use sorted dates for forecasting checks.
    test["__orig_order"] = np.arange(len(test))
    sales = sales.sort_values("Date").reset_index(drop=True)

    required_sales = {"Date", "Revenue", "COGS"}
    missing = required_sales - set(sales.columns)
    if missing:
        raise ValueError(f"sales.csv missing required columns: {missing}")
    if "Date" not in test.columns:
        raise ValueError("sample_submission.csv must contain Date")

    # Strict daily continuity for sales.
    expected_n = (sales["Date"].max() - sales["Date"].min()).days + 1
    if expected_n != len(sales):
        raise ValueError(
            f"sales.csv is not continuous daily data: expected {expected_n}, got {len(sales)}"
        )
    if test["Date"].nunique() != len(test):
        raise ValueError("sample_submission.csv has duplicate dates; direct daily setup assumes unique dates")

    return sales, test


def days_to_tet(date: pd.Timestamp) -> int:
    best = 9999
    for td in TET_DATES.values():
        d = abs((date - pd.Timestamp(td)).days)
        best = min(best, d)
    return int(best)


def days_to_fixed_holiday(date: pd.Timestamp) -> int:
    if not USE_VN_HOLIDAY_BASIC:
        return 999
    candidates = []
    for yr in [date.year - 1, date.year, date.year + 1]:
        for m, d in VN_FIXED_HOLIDAYS:
            candidates.append(pd.Timestamp(year=yr, month=m, day=d))
    return int(min(abs((date - x).days) for x in candidates))


def build_promo_calendar(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    df = pd.DataFrame({"Date": dates})
    df["promo_active"] = 0
    df["promo_count"] = 0
    df["max_discount"] = 0.0

    for name, _, _, _, _ in RECURRING_PROMOS:
        df[f"promo_{name}"] = 0
        df[f"discount_{name}"] = 0.0

    for name, start_md, end_md, discount, freq in RECURRING_PROMOS:
        for year in range(min_date.year, max_date.year + 2):
            if freq == "odd" and year % 2 == 0:
                continue
            try:
                start = pd.Timestamp(f"{year}-{start_md}")
                end = pd.Timestamp(f"{year + 1}-{end_md}") if end_md < start_md else pd.Timestamp(f"{year}-{end_md}")
            except Exception:
                continue
            mask = (df["Date"] >= start) & (df["Date"] <= end)
            df.loc[mask, "promo_active"] = 1
            df.loc[mask, "promo_count"] += 1
            df.loc[mask, "max_discount"] = np.maximum(df.loc[mask, "max_discount"], discount)
            df.loc[mask, f"promo_{name}"] = 1
            df.loc[mask, f"discount_{name}"] = discount

    # Promo phase features across the union of active campaigns.
    df["days_into_promo"] = 0
    df["days_until_promo_end"] = 0
    active = df["promo_active"].values.astype(int)

    in_block = False
    start_idx = 0
    for i, is_active in enumerate(active):
        if is_active:
            if not in_block:
                start_idx = i
                in_block = True
            df.loc[i, "days_into_promo"] = i - start_idx + 1
        else:
            in_block = False

    for i in range(len(df) - 1, -1, -1):
        if active[i]:
            j = i
            while j < len(df) and active[j]:
                j += 1
            df.loc[i, "days_until_promo_end"] = j - i

    df["is_first_week_of_promo"] = ((df["promo_active"] == 1) & (df["days_into_promo"] <= 7)).astype(int)
    df["is_last_week_of_promo"] = ((df["promo_active"] == 1) & (df["days_until_promo_end"] <= 7)).astype(int)
    return df


def make_promo_dict(promo_calendar: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
    return {row["Date"]: row.drop(labels=["Date"]).to_dict() for _, row in promo_calendar.iterrows()}


def calendar_features(date: pd.Timestamp, prefix: str = "target") -> Dict[str, float]:
    doy = int(date.dayofyear)
    week = int(date.isocalendar().week)
    dim = cal_mod.monthrange(date.year, date.month)[1]
    feats: Dict[str, float] = {
        f"{prefix}_year": int(date.year),
        f"{prefix}_year_idx": int(date.year - 2012),
        f"{prefix}_is_odd_year": int(date.year % 2 == 1),
        f"{prefix}_month": int(date.month),
        f"{prefix}_quarter": int((date.month - 1) // 3 + 1),
        f"{prefix}_day": int(date.day),
        f"{prefix}_dayofweek": int(date.dayofweek),
        f"{prefix}_dayofyear": doy,
        f"{prefix}_weekofyear": week,
        f"{prefix}_is_weekend": int(date.dayofweek >= 5),
        f"{prefix}_is_month_start": int(date.is_month_start),
        f"{prefix}_is_month_end": int(date.is_month_end),
        f"{prefix}_days_to_eom": int(dim - date.day),
        f"{prefix}_week_of_month": int((date.day - 1) // 7),
        f"{prefix}_sin_doy_1": math.sin(2 * math.pi * doy / 365.25),
        f"{prefix}_cos_doy_1": math.cos(2 * math.pi * doy / 365.25),
        f"{prefix}_sin_doy_2": math.sin(2 * math.pi * 2 * doy / 365.25),
        f"{prefix}_cos_doy_2": math.cos(2 * math.pi * 2 * doy / 365.25),
        f"{prefix}_sin_doy_3": math.sin(2 * math.pi * 3 * doy / 365.25),
        f"{prefix}_cos_doy_3": math.cos(2 * math.pi * 3 * doy / 365.25),
        f"{prefix}_sin_week": math.sin(2 * math.pi * int(date.dayofweek) / 7),
        f"{prefix}_cos_week": math.cos(2 * math.pi * int(date.dayofweek) / 7),
        f"{prefix}_is_payday_window": int(date.day in list(range(1, 4)) + list(range(10, 13)) + list(range(28, 32))),
    }
    if USE_TET:
        dtt = days_to_tet(date)
        feats[f"{prefix}_days_to_tet"] = dtt
        feats[f"{prefix}_is_tet_week"] = int(dtt <= 5)
    if USE_VN_HOLIDAY_BASIC:
        dth = days_to_fixed_holiday(date)
        feats[f"{prefix}_days_to_fixed_holiday"] = dth
        feats[f"{prefix}_is_fixed_holiday_window"] = int(dth <= 1)
    return feats


def horizon_features(h: int, horizon: int) -> Dict[str, float]:
    return {
        "h": h,
        "h_norm": h / horizon,
        "h_log1p": math.log1p(h),
        "h_week": h // 7,
        "h_mod_7": h % 7,
        "h_mod_30": h % 30,
        "h_sin_7": math.sin(2 * math.pi * h / 7),
        "h_cos_7": math.cos(2 * math.pi * h / 7),
        "h_sin_365": math.sin(2 * math.pi * h / 365.25),
        "h_cos_365": math.cos(2 * math.pi * h / 365.25),
    }


def promo_features(date: pd.Timestamp, promo_dict: Dict[pd.Timestamp, Dict[str, float]]) -> Dict[str, float]:
    if not USE_PROMO:
        return {}
    raw = promo_dict.get(date, {})
    return {f"target_{k}": v for k, v in raw.items()}


def add_ratio_column(sales: pd.DataFrame) -> pd.DataFrame:
    out = sales.copy()
    out["COGS_REV_RATIO"] = out["COGS"] / (out["Revenue"] + EPS)
    return out


def value_at(series: pd.Series, date: pd.Timestamp) -> float:
    try:
        return float(series.loc[date])
    except KeyError:
        return np.nan


def origin_state_features(
    hist: pd.DataFrame,
    origin: pd.Timestamp,
    target_col: str,
    prefix: str = "origin",
) -> Dict[str, float]:
    s = hist.set_index("Date")[target_col].sort_index()
    feats: Dict[str, float] = {}

    # Lags relative to the forecast origin. origin_lag_1 is the last observed value.
    for lag in ORIGIN_LAGS:
        anchor = origin - pd.Timedelta(days=lag - 1)
        feats[f"{prefix}_{target_col}_lag_{lag}"] = value_at(s, anchor)

    for w in ORIGIN_ROLL_WINDOWS:
        window_start = origin - pd.Timedelta(days=w - 1)
        vals = s.loc[(s.index >= window_start) & (s.index <= origin)].values.astype(float)
        if len(vals) == w:
            feats[f"{prefix}_{target_col}_roll_mean_{w}"] = float(np.mean(vals))
            feats[f"{prefix}_{target_col}_roll_std_{w}"] = float(np.std(vals, ddof=1)) if w > 1 else np.nan
            feats[f"{prefix}_{target_col}_roll_min_{w}"] = float(np.min(vals))
            feats[f"{prefix}_{target_col}_roll_max_{w}"] = float(np.max(vals))
            feats[f"{prefix}_{target_col}_roll_median_{w}"] = float(np.median(vals))
        else:
            for stat in ["mean", "std", "min", "max", "median"]:
                feats[f"{prefix}_{target_col}_roll_{stat}_{w}"] = np.nan

    # Momentum/range summaries at the origin.
    m7 = feats.get(f"{prefix}_{target_col}_roll_mean_7", np.nan)
    m30 = feats.get(f"{prefix}_{target_col}_roll_mean_30", np.nan)
    m90 = feats.get(f"{prefix}_{target_col}_roll_mean_90", np.nan)
    std30 = feats.get(f"{prefix}_{target_col}_roll_std_30", np.nan)
    min30 = feats.get(f"{prefix}_{target_col}_roll_min_30", np.nan)
    max30 = feats.get(f"{prefix}_{target_col}_roll_max_30", np.nan)
    min90 = feats.get(f"{prefix}_{target_col}_roll_min_90", np.nan)
    max90 = feats.get(f"{prefix}_{target_col}_roll_max_90", np.nan)

    feats[f"{prefix}_{target_col}_m7_minus_m30"] = m7 - m30 if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    feats[f"{prefix}_{target_col}_m7_over_m30"] = m7 / (m30 + 1e-6) if not (np.isnan(m7) or np.isnan(m30)) else np.nan
    feats[f"{prefix}_{target_col}_m30_over_m90"] = m30 / (m90 + 1e-6) if not (np.isnan(m30) or np.isnan(m90)) else np.nan
    feats[f"{prefix}_{target_col}_vol_ratio_30"] = std30 / (m30 + 1e-6) if not (np.isnan(std30) or np.isnan(m30)) else np.nan
    feats[f"{prefix}_{target_col}_range_ratio_30"] = (max30 - min30) / (m30 + 1e-6) if not (np.isnan(max30) or np.isnan(min30) or np.isnan(m30)) else np.nan
    feats[f"{prefix}_{target_col}_range_ratio_90"] = (max90 - min90) / (m90 + 1e-6) if not (np.isnan(max90) or np.isnan(min90) or np.isnan(m90)) else np.nan

    # Recent level versus seasonal long lag.
    lag1 = feats.get(f"{prefix}_{target_col}_lag_1", np.nan)
    lag364 = feats.get(f"{prefix}_{target_col}_lag_364", np.nan)
    lag728 = feats.get(f"{prefix}_{target_col}_lag_728", np.nan)
    feats[f"{prefix}_{target_col}_lag1_over_lag364"] = lag1 / (lag364 + 1e-6) if not (np.isnan(lag1) or np.isnan(lag364)) else np.nan
    feats[f"{prefix}_{target_col}_lag364_over_lag728"] = lag364 / (lag728 + 1e-6) if not (np.isnan(lag364) or np.isnan(lag728)) else np.nan
    return feats


def seasonal_template_maps(
    hist: pd.DataFrame,
    origin: pd.Timestamp,
    target_col: str,
    trailing_days: int = TRAILING_TEMPLATE_DAYS,
) -> Dict[str, Dict]:
    start = origin - pd.Timedelta(days=trailing_days - 1)
    sub = hist[(hist["Date"] >= start) & (hist["Date"] <= origin)].copy()
    sub["month"] = sub["Date"].dt.month
    sub["dow"] = sub["Date"].dt.dayofweek
    sub["weekofyear"] = sub["Date"].dt.isocalendar().week.astype(int)
    sub["dayofyear"] = sub["Date"].dt.dayofyear

    maps: Dict[str, Dict] = {}
    maps["month_mean"] = sub.groupby("month")[target_col].mean().to_dict()
    maps["month_median"] = sub.groupby("month")[target_col].median().to_dict()
    maps["dow_mean"] = sub.groupby("dow")[target_col].mean().to_dict()
    maps["dow_median"] = sub.groupby("dow")[target_col].median().to_dict()
    maps["month_dow_mean"] = sub.groupby(["month", "dow"])[target_col].mean().to_dict()
    maps["month_dow_median"] = sub.groupby(["month", "dow"])[target_col].median().to_dict()
    maps["weekofyear_mean"] = sub.groupby("weekofyear")[target_col].mean().to_dict()
    maps["weekofyear_median"] = sub.groupby("weekofyear")[target_col].median().to_dict()

    # Store a dayofyear table for +/- window lookup.
    maps["doy_frame"] = sub[["dayofyear", target_col]].copy()
    return maps


def seasonal_template_features(
    target_date: pd.Timestamp,
    target_col: str,
    maps: Dict[str, Dict],
    prefix: str = "seasonal_template",
) -> Dict[str, float]:
    month = target_date.month
    dow = target_date.dayofweek
    week = int(target_date.isocalendar().week)
    doy = int(target_date.dayofyear)
    feats: Dict[str, float] = {
        f"{prefix}_{target_col}_month_mean": maps["month_mean"].get(month, np.nan),
        f"{prefix}_{target_col}_month_median": maps["month_median"].get(month, np.nan),
        f"{prefix}_{target_col}_dow_mean": maps["dow_mean"].get(dow, np.nan),
        f"{prefix}_{target_col}_dow_median": maps["dow_median"].get(dow, np.nan),
        f"{prefix}_{target_col}_month_dow_mean": maps["month_dow_mean"].get((month, dow), np.nan),
        f"{prefix}_{target_col}_month_dow_median": maps["month_dow_median"].get((month, dow), np.nan),
        f"{prefix}_{target_col}_weekofyear_mean": maps["weekofyear_mean"].get(week, np.nan),
        f"{prefix}_{target_col}_weekofyear_median": maps["weekofyear_median"].get(week, np.nan),
    }
    doy_frame = maps["doy_frame"]
    # circular distance on 365-day calendar; good enough for this dataset.
    dist = np.minimum(abs(doy_frame["dayofyear"].values - doy), 365 - abs(doy_frame["dayofyear"].values - doy))
    vals = doy_frame.loc[dist <= 7, target_col].values.astype(float)
    feats[f"{prefix}_{target_col}_doy_window_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    feats[f"{prefix}_{target_col}_doy_window_median"] = float(np.median(vals)) if len(vals) > 0 else np.nan
    return feats


def seasonal_anchor_features(
    series: pd.Series,
    origin: pd.Timestamp,
    target_date: pd.Timestamp,
    target_col: str,
    prefix: str = "seasonal_anchor",
) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    vals = {}
    for lag in SEASONAL_ANCHOR_LAGS:
        anchor = target_date - pd.Timedelta(days=lag)
        # Strict availability rule: anchor must be observed no later than origin.
        vals[lag] = value_at(series, anchor) if anchor <= origin else np.nan
        feats[f"{prefix}_{target_col}_lag_{lag}"] = vals[lag]

    # Small anchor interactions.
    v364, v365, v728, v729 = vals.get(364, np.nan), vals.get(365, np.nan), vals.get(728, np.nan), vals.get(729, np.nan)
    if not (np.isnan(v364) and np.isnan(v365)):
        feats[f"{prefix}_{target_col}_lag364_365_mean"] = np.nanmean([v364, v365])
    else:
        feats[f"{prefix}_{target_col}_lag364_365_mean"] = np.nan
    if not (np.isnan(v728) and np.isnan(v729)):
        feats[f"{prefix}_{target_col}_lag728_729_mean"] = np.nanmean([v728, v729])
    else:
        feats[f"{prefix}_{target_col}_lag728_729_mean"] = np.nan
    a1 = feats[f"{prefix}_{target_col}_lag364_365_mean"]
    a2 = feats[f"{prefix}_{target_col}_lag728_729_mean"]
    feats[f"{prefix}_{target_col}_anchor_1y_over_2y"] = a1 / (a2 + 1e-6) if not (np.isnan(a1) or np.isnan(a2)) else np.nan
    return feats


def build_rows_for_origin(
    hist: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    target_col: str,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
    require_target_available: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[pd.Timestamp]]:
    """Build direct rows for one origin over h=1..horizon.

    If require_target_available=True, only rows whose target_date exists in hist are kept
    and y is returned. If False, all h rows are returned and y is None.
    """
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist_idx = hist.set_index("Date")
    series = hist_idx[target_col]
    y_values: List[float] = []
    rows: List[Dict[str, float]] = []
    target_dates: List[pd.Timestamp] = []

    origin_hist = hist[hist["Date"] <= origin].copy()
    if len(origin_hist) == 0:
        return pd.DataFrame(), np.array([]) if require_target_available else None, []

    origin_feats = origin_state_features(origin_hist, origin, target_col)
    template_maps = seasonal_template_maps(origin_hist, origin, target_col)

    for h in range(1, horizon + 1):
        td = origin + pd.Timedelta(days=h)
        if require_target_available and td not in set(series.index):
            # For train folds, skip incomplete horizons if any.
            continue
        feat: Dict[str, float] = {}
        feat.update(horizon_features(h, horizon))
        feat.update(calendar_features(td, prefix="target"))
        feat.update(promo_features(td, promo_dict))
        feat.update(origin_feats)
        feat.update(seasonal_anchor_features(series, origin, td, target_col))
        feat.update(seasonal_template_features(td, target_col, template_maps))
        rows.append(feat)
        target_dates.append(td)
        if require_target_available:
            y_values.append(float(series.loc[td]))

    X = pd.DataFrame(rows)
    y = np.asarray(y_values, dtype=float) if require_target_available else None
    return X, y, target_dates


def build_training_table(
    sales: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    target_col: str,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
    origin_step_days: int = ORIGIN_STEP_DAYS,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    sales_cut = sales[sales["Date"] <= cutoff].copy()
    first = sales_cut["Date"].min()
    latest_origin = cutoff - pd.Timedelta(days=horizon)
    earliest_origin = first + pd.Timedelta(days=MIN_HISTORY_DAYS)

    if latest_origin < earliest_origin:
        raise ValueError(f"Not enough history to build direct table for cutoff={cutoff.date()}")

    origins = list(pd.date_range(earliest_origin, latest_origin, freq=f"{origin_step_days}D"))
    if origins[-1] != latest_origin:
        origins.append(latest_origin)

    frames: List[pd.DataFrame] = []
    targets: List[np.ndarray] = []
    print(f"  Direct train origins: {len(origins)} from {origins[0].date()} to {origins[-1].date()} ({target_col})")
    for i, origin in enumerate(origins):
        X_o, y_o, _ = build_rows_for_origin(
            sales_cut, origin, horizon, target_col, promo_dict, require_target_available=True
        )
        if len(X_o) > 0:
            frames.append(X_o)
            targets.append(y_o)
        if (i + 1) % 50 == 0:
            print(f"    built origin {i+1}/{len(origins)}")

    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets)
    feature_names = list(X.columns)
    print(f"  Direct training table: {X.shape[0]:,} rows x {X.shape[1]} features")
    return X, y, feature_names


def build_forecast_table(
    sales_hist: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    target_col: str,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    X, _, dates = build_rows_for_origin(
        sales_hist, origin, horizon, target_col, promo_dict, require_target_available=False
    )
    # Align columns exactly to training feature list.
    for col in feature_names:
        if col not in X.columns:
            X[col] = np.nan
    extra = [c for c in X.columns if c not in feature_names]
    if extra:
        X = X.drop(columns=extra)
    return X[list(feature_names)], dates


def train_lgbm(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMRegressor:
    ensure_lgbm()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model


def fit_direct_model(
    sales: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    target_col: str,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[lgb.LGBMRegressor, List[str]]:
    X, y, feats = build_training_table(sales, cutoff, horizon, target_col, promo_dict)
    model = train_lgbm(X, y)
    return model, feats


def predict_direct(
    model: lgb.LGBMRegressor,
    feature_names: Sequence[str],
    sales_hist: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    target_col: str,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    X, dates = build_forecast_table(sales_hist, origin, horizon, target_col, promo_dict, feature_names)
    pred = model.predict(X)
    return np.maximum(pred, 0), dates


def make_folds(sales: pd.DataFrame, horizon: int) -> List[FoldSpec]:
    max_date = sales["Date"].max()
    val_end_1 = max_date
    val_start_1 = val_end_1 - pd.Timedelta(days=horizon - 1)
    origin_1 = val_start_1 - pd.Timedelta(days=1)

    val_end_2 = origin_1
    val_start_2 = val_end_2 - pd.Timedelta(days=horizon - 1)
    origin_2 = val_start_2 - pd.Timedelta(days=1)

    return [
        FoldSpec("Fold1_Recent", origin_1, val_start_1, val_end_1),
        FoldSpec("Fold2_Past", origin_2, val_start_2, val_end_2),
    ]


def compute_scores(actual_rev: np.ndarray, pred_rev: np.ndarray, actual_cogs: np.ndarray, pred_cogs: np.ndarray) -> Dict[str, float]:
    rev_mae = mean_absolute_error(actual_rev, pred_rev)
    cogs_mae = mean_absolute_error(actual_cogs, pred_cogs)
    rev_rmse = mean_squared_error(actual_rev, pred_rev, squared=False)
    cogs_rmse = mean_squared_error(actual_cogs, pred_cogs, squared=False)
    return {
        "Revenue_MAE": rev_mae,
        "COGS_MAE": cogs_mae,
        "Composite_MAE": (rev_mae + cogs_mae) / 2,
        "Revenue_RMSE": rev_rmse,
        "COGS_RMSE": cogs_rmse,
        "Composite_RMSE": (rev_rmse + cogs_rmse) / 2,
    }


def scenario_direct_rev_cogs(
    sales: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    rev_model, rev_feats = fit_direct_model(sales, origin, horizon, "Revenue", promo_dict)
    cogs_model, cogs_feats = fit_direct_model(sales, origin, horizon, "COGS", promo_dict)

    pred_rev, dates = predict_direct(rev_model, rev_feats, sales[sales["Date"] <= origin], origin, horizon, "Revenue", promo_dict)
    pred_cogs, _ = predict_direct(cogs_model, cogs_feats, sales[sales["Date"] <= origin], origin, horizon, "COGS", promo_dict)
    return pred_rev, pred_cogs, dates


def ratio_clip_bounds(train_sales: pd.DataFrame) -> Tuple[float, float]:
    ratio = train_sales["COGS_REV_RATIO"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 100:
        return RATIO_CLIP_FALLBACK
    lo = float(ratio.quantile(0.005))
    hi = float(ratio.quantile(0.995))
    # Keep conservative sanity bounds.
    lo = max(0.10, min(lo, RATIO_CLIP_FALLBACK[0]))
    hi = min(2.00, max(hi, RATIO_CLIP_FALLBACK[1]))
    return lo, hi


def scenario_direct_rev_ratio(
    sales_with_ratio: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    promo_dict: Dict[pd.Timestamp, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], Tuple[float, float]]:
    # Revenue model uses normal sales frame.
    rev_model, rev_feats = fit_direct_model(sales_with_ratio, origin, horizon, "Revenue", promo_dict)
    ratio_model, ratio_feats = fit_direct_model(sales_with_ratio, origin, horizon, "COGS_REV_RATIO", promo_dict)

    hist = sales_with_ratio[sales_with_ratio["Date"] <= origin].copy()
    pred_rev, dates = predict_direct(rev_model, rev_feats, hist, origin, horizon, "Revenue", promo_dict)
    pred_ratio_raw, _ = predict_direct(ratio_model, ratio_feats, hist, origin, horizon, "COGS_REV_RATIO", promo_dict)
    bounds = ratio_clip_bounds(hist)
    pred_ratio = np.clip(pred_ratio_raw, bounds[0], bounds[1])
    pred_cogs = np.maximum(pred_rev * pred_ratio, 0)
    return pred_rev, pred_cogs, pred_ratio, dates, bounds


def evaluate_backtest(sales: pd.DataFrame, promo_dict: Dict[pd.Timestamp, Dict[str, float]], horizon: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    folds = make_folds(sales, horizon)
    sales_ratio = add_ratio_column(sales)

    print("\n" + "=" * 90)
    print(f"DIRECT MULTI-STEP BACKTEST — horizon={horizon}")
    print("=" * 90)

    for fold in folds:
        print(f"\n{fold.label}: origin={fold.origin.date()}, val={fold.val_start.date()} to {fold.val_end.date()}")
        val = sales[(sales["Date"] >= fold.val_start) & (sales["Date"] <= fold.val_end)].copy()
        if len(val) != horizon:
            raise ValueError(f"{fold.label} validation length {len(val)} != horizon {horizon}")

        # Scenario 1: direct Revenue + direct COGS.
        print("\n  Scenario DIRECT_REV_COGS")
        pred_rev, pred_cogs, dates = scenario_direct_rev_cogs(sales, fold.origin, horizon, promo_dict)
        scores = compute_scores(val["Revenue"].values, pred_rev, val["COGS"].values, pred_cogs)
        scores.update({"Scenario": "DIRECT_REV_COGS", "Fold": fold.label})
        rows.append(scores)
        print(f"    Rev MAE={scores['Revenue_MAE']:,.0f} | COGS MAE={scores['COGS_MAE']:,.0f} | Comp={scores['Composite_MAE']:,.0f}")

        # Scenario 2: direct Revenue + direct ratio.
        print("\n  Scenario DIRECT_REV_RATIO")
        pred_rev2, pred_cogs2, pred_ratio, dates2, bounds = scenario_direct_rev_ratio(sales_ratio, fold.origin, horizon, promo_dict)
        scores2 = compute_scores(val["Revenue"].values, pred_rev2, val["COGS"].values, pred_cogs2)
        scores2.update({
            "Scenario": "DIRECT_REV_RATIO",
            "Fold": fold.label,
            "ratio_clip_lo": bounds[0],
            "ratio_clip_hi": bounds[1],
            "pred_ratio_mean": float(np.mean(pred_ratio)),
        })
        rows.append(scores2)
        print(f"    Rev MAE={scores2['Revenue_MAE']:,.0f} | COGS MAE={scores2['COGS_MAE']:,.0f} | Comp={scores2['Composite_MAE']:,.0f}")
        print(f"    ratio bounds={bounds[0]:.4f}..{bounds[1]:.4f}, pred_ratio_mean={np.mean(pred_ratio):.4f}")

    results = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("BACKTEST SUMMARY")
    print("=" * 90)
    summary = results.groupby("Scenario")[["Revenue_MAE", "COGS_MAE", "Composite_MAE", "Revenue_RMSE", "COGS_RMSE", "Composite_RMSE"]].mean().sort_values("Composite_MAE")
    print(summary.to_string(float_format=lambda x: f"{x:,.0f}"))
    return results


def generate_final_submissions(sales: pd.DataFrame, test: pd.DataFrame, promo_dict: Dict[pd.Timestamp, Dict[str, float]], horizon: int) -> None:
    origin = sales["Date"].max()
    sales_ratio = add_ratio_column(sales)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"FINAL DIRECT FORECAST — origin={origin.date()}, horizon={horizon}")
    print("=" * 90)

    # Scenario 1
    print("\nTraining final DIRECT_REV_COGS...")
    pred_rev, pred_cogs, dates = scenario_direct_rev_cogs(sales, origin, horizon, promo_dict)
    save_submission(test, dates, pred_rev, pred_cogs, "submission_direct_rev_cogs.csv")

    # Scenario 2
    print("\nTraining final DIRECT_REV_RATIO...")
    pred_rev2, pred_cogs2, pred_ratio, dates2, bounds = scenario_direct_rev_ratio(sales_ratio, origin, horizon, promo_dict)
    save_submission(test, dates2, pred_rev2, pred_cogs2, "submission_direct_rev_ratio.csv")
    print(f"  Final ratio bounds={bounds[0]:.4f}..{bounds[1]:.4f}, pred_ratio_mean={np.mean(pred_ratio):.4f}")


def save_submission(test: pd.DataFrame, pred_dates: List[pd.Timestamp], pred_rev: np.ndarray, pred_cogs: np.ndarray, filename: str) -> str:
    pred_df = pd.DataFrame({"Date": pred_dates, "Revenue": pred_rev, "COGS": pred_cogs})
    # Preserve sample_submission order exactly.
    out = test[["Date", "__orig_order"]].merge(pred_df, on="Date", how="left")
    if out[["Revenue", "COGS"]].isnull().any().any():
        missing = out[out[["Revenue", "COGS"]].isnull().any(axis=1)]["Date"].tolist()
        raise ValueError(f"Missing predictions for some submission dates: {missing[:5]}")
    out = out.sort_values("__orig_order")[["Date", "Revenue", "COGS"]]
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    path = os.path.join(OUTPUT_DIR, filename)
    out.to_csv(path, index=False)
    print(f"  Saved {path}: {len(out)} rows")
    print(f"    Revenue mean={out['Revenue'].mean():,.0f}, COGS mean={out['COGS'].mean():,.0f}")
    return path


def main() -> None:
    t0 = datetime.now()
    ensure_lgbm()
    sales, test = read_data()
    horizon = len(test)

    # Use the date range required by all folds + final test.
    min_date = sales["Date"].min()
    max_date = max(sales["Date"].max(), test["Date"].max())
    promo_calendar = build_promo_calendar(min_date, max_date)
    promo_dict = make_promo_dict(promo_calendar)

    print("\n" + "=" * 90)
    print("DIRECT MULTI-STEP DUAL-TARGET EXPERIMENT")
    print("=" * 90)
    print(f"Sales: {sales['Date'].min().date()} -> {sales['Date'].max().date()} ({len(sales):,} days)")
    print(f"Test : {test['Date'].min().date()} -> {test['Date'].max().date()} ({horizon:,} days)")
    print(f"Origin step days: {ORIGIN_STEP_DAYS}; min history: {MIN_HISTORY_DAYS}; trailing templates: {TRAILING_TEMPLATE_DAYS}")
    print(f"Feature switches: USE_PROMO={USE_PROMO}, USE_TET={USE_TET}, USE_VN_HOLIDAY_BASIC={USE_VN_HOLIDAY_BASIC}")

    results = evaluate_backtest(sales, promo_dict, horizon)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "direct_multistep_backtest_results.csv")
    results.to_csv(summary_path, index=False)
    print(f"\nSaved backtest results: {summary_path}")

    generate_final_submissions(sales, test, promo_dict, horizon)
    print(f"\nDone in {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == "__main__":
    main()

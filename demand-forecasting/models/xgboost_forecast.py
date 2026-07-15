"""
ML time-series forecasting: XGBoost with engineered lag + rolling features.

Strategy:
- Daily level, all 5 categories simultaneously (category as feature)
- Features: lag 1/7/14/28/365, rolling 7d/28d mean+std, calendar features, holiday flags
- Walk-forward expanding-window validation: 6 quarterly windows (2022-Q3 to 2023-Q4)
- Evaluate MAPE, RMSE, MAE per quarter and overall
- Feature importance chart + forecast-vs-actual chart
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "demand_data.csv"
CHARTS = ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

LAGS = [1, 7, 14, 28, 364]
ROLLING_WINDOWS = [7, 28]
CAT_FOCUS = "Electronics"   # for the forecast-vs-actual chart

VALIDATION_QUARTERS = [
    ("2022-Q3", "2022-07-01", "2022-09-30"),
    ("2022-Q4", "2022-10-01", "2022-12-31"),
    ("2023-Q1", "2023-01-01", "2023-03-31"),
    ("2023-Q2", "2023-04-01", "2023-06-30"),
    ("2023-Q3", "2023-07-01", "2023-09-30"),
    ("2023-Q4", "2023-10-01", "2023-12-31"),
]


def load_daily(path):
    df = pd.read_csv(path, parse_dates=["date"])
    # Aggregate to daily per category (sum across SKUs)
    daily = df.groupby(["date", "category"]).agg(
        units_sold=("units_sold", "sum"),
        is_holiday=("is_holiday", "max"),
    ).reset_index()
    return daily.sort_values(["category", "date"]).reset_index(drop=True)


def engineer_features(daily):
    """Add lag, rolling, and calendar features per category."""
    frames = []
    for cat in daily["category"].unique():
        sub = daily[daily["category"] == cat].copy().sort_values("date")
        y = sub["units_sold"].values.astype(float)

        # Lag features
        for lag in LAGS:
            sub[f"lag_{lag}"] = pd.Series(y).shift(lag).values

        # Rolling statistics (on the lag-1 shifted series to avoid leakage)
        shifted = pd.Series(y).shift(1)
        for w in ROLLING_WINDOWS:
            sub[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().values
            sub[f"roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().values

        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)

    # Calendar features
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year
    out["dayofyear"] = out["date"].dt.day_of_year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    # Cyclical encodings
    out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 365)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 365)
    out["year_trend"] = (out["year"] - 2021) + (out["dayofyear"] / 365)

    # Category encoding
    le = LabelEncoder()
    out["cat_enc"] = le.fit_transform(out["category"])
    out["_le_classes"] = out["cat_enc"]   # keep for reference

    return out, le


def get_feature_cols(df):
    lag_cols = [f"lag_{l}" for l in LAGS]
    roll_cols = [f"roll_mean_{w}" for w in ROLLING_WINDOWS] + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    cal_cols = ["dayofweek", "month", "quarter", "year_trend", "is_weekend",
                "sin_dow", "cos_dow", "sin_doy", "cos_doy", "is_holiday", "cat_enc"]
    return lag_cols + roll_cols + cal_cols


def mape(actual, predicted):
    mask = actual > 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def walk_forward_xgb(featured):
    feature_cols = get_feature_cols(featured)
    # Drop rows with NaN (from lag creation)
    max_lag = max(LAGS)

    all_actuals = []
    all_preds = []
    quarter_results = []

    model = None
    for q_name, q_start, q_end in VALIDATION_QUARTERS:
        train = featured[featured["date"] < q_start].dropna(subset=feature_cols)
        test = featured[(featured["date"] >= q_start) & (featured["date"] <= q_end)].dropna(subset=feature_cols)

        if len(train) < 500 or len(test) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["units_sold"].values
        X_test = test[feature_cols].values
        y_test = test["units_sold"].values

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        preds = np.maximum(model.predict(X_test), 0)

        q_mape = mape(y_test, preds)
        q_rmse = np.sqrt(mean_squared_error(y_test, preds))
        q_mae = mean_absolute_error(y_test, preds)

        all_actuals.extend(y_test)
        all_preds.extend(preds)

        quarter_results.append({
            "quarter": q_name,
            "n_days_x_cats": len(test),
            "MAPE": round(q_mape, 2),
            "RMSE": round(q_rmse, 1),
            "MAE": round(q_mae, 1),
        })
        print(f"  {q_name}: MAPE={q_mape:.1f}%  RMSE={q_rmse:.0f}  MAE={q_mae:.0f}  "
              f"(n={len(test)} day-category rows)")

    return np.array(all_actuals), np.array(all_preds), quarter_results, model, feature_cols


def plot_feature_importance(model, feature_cols):
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top = fi.tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values, color="#1f77b4", alpha=0.85, edgecolor="none")
    ax.set_title("XGBoost Feature Importance (Top 15)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gain")
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    out = CHARTS / "07_xgb_feature_importance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_xgb_forecast(featured, model, feature_cols, le):
    """Plot forecast vs actual for the focus category in 2023."""
    feature_cols_used = get_feature_cols(featured)
    cat_enc = le.transform([CAT_FOCUS])[0]
    sub = featured[
        (featured["category"] == CAT_FOCUS) &
        (featured["date"] >= "2022-07-01")
    ].dropna(subset=feature_cols_used).copy()

    preds = np.maximum(model.predict(sub[feature_cols_used].values), 0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"XGBoost Forecast vs Actual — {CAT_FOCUS} (Daily, 2022-H2 onward)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(sub["date"], sub["units_sold"], color="#aec7e8", lw=0.8, alpha=0.5, label="Actual (raw)")
    roll_actual = pd.Series(sub["units_sold"].values).rolling(7).mean()
    roll_pred = pd.Series(preds).rolling(7).mean()
    ax.plot(sub["date"], roll_actual.values, color="#1f77b4", lw=1.8, label="Actual (7d avg)")
    ax.plot(sub["date"], roll_pred.values, color="#d62728", lw=1.8, ls="--", label="XGB Forecast (7d avg)")
    ax.axvline(pd.Timestamp("2022-07-01"), color="gray", ls=":", lw=1)
    ax.set_ylabel("Daily Units Sold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Residuals
    ax = axes[1]
    resid = sub["units_sold"].values - preds
    ax.plot(sub["date"], resid, color="#9467bd", lw=0.8, alpha=0.7)
    ax.axhline(0, color="black", lw=1.0)
    roll_resid = pd.Series(resid).rolling(14).mean()
    ax.plot(sub["date"], roll_resid.values, color="#d62728", lw=1.5, label="14d rolling residual mean")
    ax.set_ylabel("Residual (Actual - Forecast)")
    ax.set_xlabel("Date")
    ax.set_title("Forecast Residuals")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    out = CHARTS / "08_xgb_forecast.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def main():
    print("\n=== XGBoost Forecast (all categories, daily) ===")
    daily = load_daily(DATA)
    print(f"Daily series: {len(daily):,} rows ({daily['date'].min().date()} to {daily['date'].max().date()})")
    print(f"Categories: {sorted(daily['category'].unique())}")

    print("\nEngineering features...")
    featured, le = engineer_features(daily)
    feature_cols = get_feature_cols(featured)
    print(f"Feature set: {len(feature_cols)} features")

    print("\nWalk-forward validation (expanding window, quarterly):")
    all_actuals, all_preds, quarter_results, model, _ = walk_forward_xgb(featured)

    overall_mape = mape(all_actuals, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    overall_mae = mean_absolute_error(all_actuals, all_preds)

    print(f"\nXGBoost Overall (daily, all categories):")
    print(f"  MAPE: {overall_mape:.2f}%")
    print(f"  RMSE: {overall_rmse:.1f} units/day/category")
    print(f"  MAE:  {overall_mae:.1f} units/day/category")

    print("\nGenerating charts...")
    plot_feature_importance(model, feature_cols)
    plot_xgb_forecast(featured, model, feature_cols, le)

    metrics = {
        "model": "XGBoost+LagFeatures",
        "granularity": "daily (all categories)",
        "MAPE": round(overall_mape, 2),
        "RMSE": round(overall_rmse, 1),
        "MAE": round(overall_mae, 1),
        "n_features": len(feature_cols),
    }
    return metrics, quarter_results, model, featured, le, feature_cols


if __name__ == "__main__":
    metrics, quarterly, *_ = main()
    print("\nXGBoost done.")

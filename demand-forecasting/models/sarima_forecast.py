"""
Classical time-series forecasting: SARIMA with calendar regressors (SARIMAX).

Strategy:
- Aggregate daily data to weekly totals per category
- Fit SARIMAX(1,1,1) with exogenous features: month dummies + holiday flag
- Walk-forward (expanding-window) validation: quarterly windows
- Evaluate MAPE, RMSE, MAE on held-out quarters
- Save forecast-vs-actual chart
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "demand_data.csv"
CHARTS = ROOT / "outputs" / "charts"
RESULTS_DIR = ROOT / "outputs"
CHARTS.mkdir(parents=True, exist_ok=True)

FOCUS_CAT = "Electronics"   # demonstrate SARIMA on one representative category


def load_weekly(path, category):
    df = pd.read_csv(path, parse_dates=["date"])
    cat = df[df["category"] == category].copy()
    # Aggregate to weekly
    cat["week"] = cat["date"].dt.to_period("W").dt.start_time
    weekly = cat.groupby("week").agg(
        units_sold=("units_sold", "sum"),
        is_holiday=("is_holiday", "max"),   # 1 if any holiday in the week
    ).reset_index()
    weekly = weekly.rename(columns={"week": "date"})
    weekly = weekly.sort_values("date").reset_index(drop=True)

    # Calendar features as exogenous variables
    weekly["month"] = weekly["date"].dt.month
    for m in range(2, 13):
        weekly[f"m{m}"] = (weekly["month"] == m).astype(int)
    weekly["year_trend"] = (weekly["date"].dt.year - 2021) + (weekly["date"].dt.dayofyear / 365)
    return weekly


def build_exog(df):
    month_dummies = [f"m{m}" for m in range(2, 13)]
    cols = month_dummies + ["is_holiday", "year_trend"]
    return df[cols].values


def mape(actual, predicted):
    mask = actual > 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def walk_forward_sarima(weekly):
    """Expanding-window validation: train on growing history, predict next quarter."""
    # Training starts with 2021 (52 weeks), then we expand quarter by quarter
    # Validation quarters: Q1-2023, Q2-2023, Q3-2023, Q4-2023
    validation_quarters = [
        ("2022-Q3", "2022-07-01", "2022-09-30"),
        ("2022-Q4", "2022-10-01", "2022-12-31"),
        ("2023-Q1", "2023-01-01", "2023-03-31"),
        ("2023-Q2", "2023-04-01", "2023-06-30"),
        ("2023-Q3", "2023-07-01", "2023-09-30"),
        ("2023-Q4", "2023-10-01", "2023-12-31"),
    ]

    all_actuals = []
    all_preds = []
    quarter_results = []

    for q_name, q_start, q_end in validation_quarters:
        train = weekly[weekly["date"] < q_start].copy()
        test = weekly[(weekly["date"] >= q_start) & (weekly["date"] <= q_end)].copy()

        if len(train) < 20 or len(test) == 0:
            continue

        y_train = train["units_sold"].values.astype(float)
        exog_train = build_exog(train)
        exog_test = build_exog(test)

        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=200)
            preds = fit.forecast(steps=len(test), exog=exog_test)
            preds = np.maximum(preds, 0)

            actual = test["units_sold"].values
            q_mape = mape(actual, preds)
            q_rmse = np.sqrt(mean_squared_error(actual, preds))
            q_mae = mean_absolute_error(actual, preds)

            all_actuals.extend(actual)
            all_preds.extend(preds)
            quarter_results.append({
                "quarter": q_name,
                "n_weeks": len(test),
                "MAPE": round(q_mape, 2),
                "RMSE": round(q_rmse, 1),
                "MAE": round(q_mae, 1),
            })
            print(f"  {q_name}: MAPE={q_mape:.1f}%  RMSE={q_rmse:.0f}  MAE={q_mae:.0f}")
        except Exception as e:
            print(f"  {q_name}: SARIMA failed — {e}")

    return np.array(all_actuals), np.array(all_preds), quarter_results


def plot_forecast(weekly, all_actuals, all_preds, quarter_results):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"SARIMA(1,1,1) with Calendar Regressors — {FOCUS_CAT} (Weekly)",
                 fontsize=13, fontweight="bold")

    # Full time series + validation period forecast
    ax = axes[0]
    # Find where validation starts
    val_start_idx = weekly[weekly["date"] >= "2022-07-01"].index[0]
    val_dates = weekly.loc[val_start_idx:, "date"].values
    val_actual = weekly.loc[val_start_idx:, "units_sold"].values

    ax.plot(weekly["date"], weekly["units_sold"],
            color="#aec7e8", lw=1.0, alpha=0.6, label="Full series")
    ax.axvline(pd.Timestamp("2022-07-01"), color="gray", ls="--", lw=1.2, label="Validation start")

    # Plot quarterly predictions
    ptr = 0
    for r in quarter_results:
        qname = r["quarter"]
        n = r["n_weeks"]
        q_preds = all_preds[ptr:ptr+n]
        q_dates = weekly[(weekly["date"] >= qname.replace("-", " Q").split()[0]) |
                         (weekly["date"] >= "2022-07-01")]["date"].values
        # Simpler: use position
        wk_slice = weekly.iloc[val_start_idx + ptr: val_start_idx + ptr + n]
        ax.plot(wk_slice["date"], q_preds, color="#d62728", lw=1.5, alpha=0.8)
        ptr += n

    ax.plot([], [], color="#d62728", lw=1.5, label="SARIMA Forecast")
    ax.set_ylabel("Weekly Units Sold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Actual vs predicted scatter
    ax = axes[1]
    ax.scatter(all_actuals, all_preds, alpha=0.6, color="#1f77b4", s=30, edgecolors="none")
    min_v = min(all_actuals.min(), all_preds.min())
    max_v = max(all_actuals.max(), all_preds.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="Perfect forecast")
    overall_mape = mape(all_actuals, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    ax.set_xlabel("Actual Weekly Units")
    ax.set_ylabel("Predicted Weekly Units")
    ax.set_title(f"Actual vs Predicted — Overall MAPE={overall_mape:.1f}%, RMSE={overall_rmse:.0f}")
    ax.legend()

    plt.tight_layout()
    out = CHARTS / "06_sarima_forecast.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")
    return overall_mape, overall_rmse, mean_absolute_error(all_actuals, all_preds)


def main():
    print(f"\n=== SARIMA Forecast: {FOCUS_CAT} (weekly) ===")
    weekly = load_weekly(DATA, FOCUS_CAT)
    print(f"Weekly series: {len(weekly)} points ({weekly['date'].min().date()} to {weekly['date'].max().date()})")

    print("\nWalk-forward validation (expanding window, quarterly):")
    all_actuals, all_preds, quarter_results = walk_forward_sarima(weekly)

    overall_mape = mape(all_actuals, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    overall_mae = mean_absolute_error(all_actuals, all_preds)

    print(f"\nSARIMA Overall (weekly {FOCUS_CAT}):")
    print(f"  MAPE: {overall_mape:.2f}%")
    print(f"  RMSE: {overall_rmse:.1f} units")
    print(f"  MAE:  {overall_mae:.1f} units")

    plot_forecast(weekly, all_actuals, all_preds, quarter_results)

    # Save metrics
    metrics = {
        "model": "SARIMA(1,1,1)+CalendarRegressors",
        "category": FOCUS_CAT,
        "granularity": "weekly",
        "MAPE": round(overall_mape, 2),
        "RMSE": round(overall_rmse, 1),
        "MAE": round(overall_mae, 1),
        "n_quarters": len(quarter_results),
    }
    return metrics, quarter_results


if __name__ == "__main__":
    metrics, quarterly = main()
    print("\nSARIMA done.")

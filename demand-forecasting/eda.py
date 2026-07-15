"""
Exploratory Data Analysis for demand forecasting project.
Generates: decomposition plot, ACF/PACF, seasonal patterns, category comparison.
Saves all charts to outputs/charts/.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "demand_data.csv"
CHARTS = ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
})


def load_and_aggregate(path):
    df = pd.read_csv(path, parse_dates=["date"])
    # Daily aggregate per category
    daily_cat = (
        df.groupby(["date", "category"])["units_sold"].sum().reset_index()
    )
    # Daily aggregate total
    daily_total = df.groupby("date")["units_sold"].sum().reset_index()
    daily_total.columns = ["date", "total_units"]
    return df, daily_cat, daily_total


def plot_time_series_overview(daily_cat, daily_total):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Total demand
    ax = axes[0]
    # 7-day rolling mean for clarity
    total_roll = daily_total.set_index("date")["total_units"].rolling(7).mean()
    ax.plot(daily_total["date"], daily_total["total_units"], alpha=0.25, color="#4c9ed9", lw=0.8)
    ax.plot(total_roll.index, total_roll.values, color="#1f77b4", lw=2.0, label="7-day rolling mean")
    ax.set_title("Total Daily Demand (All Categories)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Per-category stacked view
    ax = axes[1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    cats = sorted(daily_cat["category"].unique())
    for i, cat in enumerate(cats):
        s = daily_cat[daily_cat["category"] == cat].set_index("date")["units_sold"].rolling(14).mean()
        ax.plot(s.index, s.values, label=cat, color=colors[i], lw=1.5)
    ax.set_title("14-Day Rolling Mean by Category", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.legend(ncol=3, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Year-over-year comparison
    ax = axes[2]
    for yr, col in [(2021, "#aec7e8"), (2022, "#1f77b4"), (2023, "#08306b")]:
        mask = daily_total["date"].dt.year == yr
        sub = daily_total[mask].copy()
        sub["doy"] = sub["date"].dt.day_of_year
        roll = sub.set_index("doy")["total_units"].rolling(7).mean()
        ax.plot(roll.index, roll.values, label=str(yr), color=col, lw=1.8)
    ax.set_title("Year-over-Year Comparison (7-day rolling)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Units Sold")
    ax.legend()

    plt.tight_layout()
    out = CHARTS / "01_time_series_overview.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_decomposition(daily_total):
    ts = daily_total.set_index("date")["total_units"]
    ts = ts.asfreq("D").ffill()

    result = seasonal_decompose(ts, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    fig.suptitle("Seasonal Decomposition of Total Daily Demand (period=7 days)",
                 fontsize=13, fontweight="bold", y=1.01)

    titles = ["Observed", "Trend", "Seasonal (Weekly)", "Residual"]
    series = [result.observed, result.trend, result.seasonal, result.resid]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for ax, s, title, col in zip(axes, series, titles, colors):
        ax.plot(s.index, s.values, color=col, lw=1.2, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        if title == "Seasonal (Weekly)":
            ax.axhline(0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    out = CHARTS / "02_decomposition.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_acf_pacf(daily_total):
    ts = daily_total.set_index("date")["total_units"].asfreq("D").ffill()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(ts, lags=60, ax=axes[0], color="#1f77b4", title="Autocorrelation (ACF) — Total Daily Demand")
    plot_pacf(ts, lags=40, ax=axes[1], color="#ff7f0e", title="Partial Autocorrelation (PACF)", method="ywm")
    plt.tight_layout()
    out = CHARTS / "03_acf_pacf.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_seasonal_patterns(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Seasonal Demand Patterns", fontsize=13, fontweight="bold")

    # By day of week
    ax = axes[0]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    dow_mean = df.groupby("dow")["units_sold"].mean()
    bars = ax.bar(dow_names, dow_mean.values, color=["#4c9ed9"]*5 + ["#ff7f0e"]*2, edgecolor="white")
    ax.set_title("Average Demand by Day of Week", fontsize=11)
    ax.set_ylabel("Mean Units Sold (per SKU)")
    for bar, val in zip(bars, dow_mean.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # By month
    ax = axes[1]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mon_mean = df.groupby("month")["units_sold"].mean()
    bars = ax.bar(month_names, mon_mean.values, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_title("Average Demand by Month", fontsize=11)
    ax.set_ylabel("Mean Units Sold (per SKU)")
    for bar, val in zip(bars, mon_mean.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = CHARTS / "04_seasonal_patterns.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_holiday_impact(df):
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])

    # Holiday vs non-holiday comparison
    hol = df2[df2["is_holiday"] == 1]
    non_hol = df2[df2["is_holiday"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Holiday Demand Impact", fontsize=13, fontweight="bold")

    # Overall holiday lift
    ax = axes[0]
    cats = sorted(df2["category"].unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(cats))
    width = 0.35
    hol_means = [hol[hol["category"]==c]["units_sold"].mean() for c in cats]
    non_hol_means = [non_hol[non_hol["category"]==c]["units_sold"].mean() for c in cats]
    ax.bar(x - width/2, non_hol_means, width, label="Regular Days", color="#aec7e8", edgecolor="white")
    ax.bar(x + width/2, hol_means, width, label="Holiday Days", color="#ff7f0e", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:4] for c in cats])
    ax.set_title("Holiday vs Regular Days by Category")
    ax.set_ylabel("Mean Units Sold")
    ax.legend()

    # Named holiday breakdown
    ax = axes[1]
    hol_named = df2[df2["holiday_name"] != ""].groupby("holiday_name")["units_sold"].mean().sort_values(ascending=True)
    ax.barh(hol_named.index, hol_named.values, color="#d62728", alpha=0.8, edgecolor="white")
    ax.set_title("Average Demand by Holiday")
    ax.set_xlabel("Mean Units Sold")
    ax.axvline(non_hol["units_sold"].mean(), color="navy", lw=1.5, ls="--", label="Regular day avg")
    ax.legend()

    plt.tight_layout()
    out = CHARTS / "05_holiday_impact.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def main():
    print("Loading data...")
    df, daily_cat, daily_total = load_and_aggregate(DATA)

    print(f"\nDataset: {len(df):,} rows, {df['date'].nunique()} days, "
          f"{df['sku_id'].nunique()} SKUs, {df['category'].nunique()} categories")
    print(f"Total units sold: {df['units_sold'].sum():,}")
    print(f"Holiday days: {df[df['is_holiday']==1]['date'].nunique()}")

    print("\nGenerating EDA charts...")
    plot_time_series_overview(daily_cat, daily_total)
    plot_decomposition(daily_total)
    plot_acf_pacf(daily_total)
    plot_seasonal_patterns(df)
    plot_holiday_impact(df)
    print("\nEDA complete. Charts saved to outputs/charts/")


if __name__ == "__main__":
    main()

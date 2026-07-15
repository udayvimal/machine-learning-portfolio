"""
Generate the head-to-head comparison chart between SARIMA and XGBoost.
Called after both models have run and saved their metrics.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CHARTS = Path(__file__).parent.parent / "outputs" / "charts"


def plot_model_comparison(sarima_metrics, xgb_metrics, sarima_quarterly, xgb_quarterly):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SARIMA vs XGBoost: Head-to-Head Comparison\n(Walk-Forward Validation, 2022-Q3 to 2023-Q4)",
                 fontsize=12, fontweight="bold")

    metrics = ["MAPE", "RMSE", "MAE"]
    sarima_vals = [sarima_metrics["MAPE"], sarima_metrics["RMSE"], sarima_metrics["MAE"]]
    xgb_vals = [xgb_metrics["MAPE"], xgb_metrics["RMSE"], xgb_metrics["MAE"]]
    colors = ["#1f77b4", "#ff7f0e"]

    for ax, metric, sv, xv in zip(axes, metrics, sarima_vals, xgb_vals):
        bars = ax.bar(
            ["SARIMA\n(weekly, 1 cat)", "XGBoost\n(daily, all cats)"],
            [sv, xv],
            color=colors,
            width=0.5,
            edgecolor="white",
        )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        unit = "%" if metric == "MAPE" else "units"
        ax.set_ylabel(unit)
        for bar, val in zip(bars, [sv, xv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sv*0.02,
                    f"{val:.1f}{'' if metric == 'MAPE' else ''}{'%' if metric == 'MAPE' else ''}",
                    ha="center", va="bottom", fontweight="bold", fontsize=11)
        # Highlight the better one
        better_idx = 0 if sv < xv else 1
        bars[better_idx].set_edgecolor("#2ca02c")
        bars[better_idx].set_linewidth(2.5)

    plt.tight_layout()
    out = CHARTS / "09_model_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_quarterly_comparison(sarima_quarterly, xgb_quarterly):
    """Plot MAPE by quarter for both models."""
    # XGBoost quarterly is computed on all categories jointly; filter to comparable quarters
    xgb_quarters = {r["quarter"]: r["MAPE"] for r in xgb_quarterly}
    sarima_quarters = {r["quarter"]: r["MAPE"] for r in sarima_quarterly}
    shared = sorted(set(xgb_quarters) & set(sarima_quarters))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(shared))
    w = 0.35
    ax.bar(x - w/2, [sarima_quarters[q] for q in shared], w, label="SARIMA", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, [xgb_quarters[q] for q in shared], w, label="XGBoost", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(shared)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Quarter: SARIMA vs XGBoost", fontsize=12, fontweight="bold")
    ax.legend()
    ax.axhline(np.mean([sarima_quarters[q] for q in shared]), color="#1f77b4", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(np.mean([xgb_quarters[q] for q in shared]), color="#ff7f0e", ls="--", lw=1.2, alpha=0.6)
    plt.tight_layout()
    out = CHARTS / "10_quarterly_mape_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")

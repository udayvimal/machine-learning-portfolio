"""
End-to-end pipeline: generate data -> EDA -> SARIMA -> XGBoost -> comparison.
Run from the demand-forecasting/ directory:
    python run_pipeline.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATA_CSV = ROOT / "data" / "demand_data.csv"


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def main():
    step("STEP 1: Generating synthetic demand data")
    from data.generate_data import main as gen_data
    gen_data()

    step("STEP 2: Exploratory Data Analysis")
    import eda
    eda.main()

    step("STEP 3: SARIMA Forecast (classical, weekly aggregated)")
    from models.sarima_forecast import main as sarima_main
    sarima_metrics, sarima_quarterly = sarima_main()

    step("STEP 4: XGBoost Forecast (ML, daily with lag features)")
    from models.xgboost_forecast import main as xgb_main
    xgb_metrics, xgb_quarterly, model, featured, le, feature_cols = xgb_main()

    step("STEP 5: Head-to-head comparison charts")
    from models.comparison_chart import plot_model_comparison, plot_quarterly_comparison
    plot_model_comparison(sarima_metrics, xgb_metrics, sarima_quarterly, xgb_quarterly)
    plot_quarterly_comparison(sarima_quarterly, xgb_quarterly)

    step("FINAL SUMMARY")
    print(f"\n{'Model':<35} {'MAPE':>8} {'RMSE':>8} {'MAE':>8}")
    print("-" * 62)
    print(f"{'SARIMA (weekly, Electronics only)':<35} "
          f"{sarima_metrics['MAPE']:>7.2f}% "
          f"{sarima_metrics['RMSE']:>8.1f} "
          f"{sarima_metrics['MAE']:>8.1f}")
    print(f"{'XGBoost (daily, all categories)':<35} "
          f"{xgb_metrics['MAPE']:>7.2f}% "
          f"{xgb_metrics['RMSE']:>8.1f} "
          f"{xgb_metrics['MAE']:>8.1f}")
    print()

    winner_mape = "SARIMA" if sarima_metrics["MAPE"] < xgb_metrics["MAPE"] else "XGBoost"
    print(f"Lower MAPE: {winner_mape}")
    print("\nAll outputs saved to outputs/charts/")

    # Save metrics to JSON for README generation
    summary = {
        "sarima": sarima_metrics,
        "xgboost": xgb_metrics,
        "sarima_quarterly": sarima_quarterly,
        "xgboost_quarterly": xgb_quarterly,
    }
    out_json = ROOT / "outputs" / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {out_json}")


if __name__ == "__main__":
    main()

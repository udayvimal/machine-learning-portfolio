"""
Generates the four evaluation datasets for the Autonomous ML Analyst Agent.

  dataset_01_classification.csv  — customer churn (clean, binary target)
  dataset_02_regression.csv      — housing prices (clean, continuous target)
  dataset_03_messy.csv           — employee attrition (15–25% missing, noisy categoricals)
  dataset_04_leakage.csv         — customer churn with a subtle data leakage column

Run:  python data/generate_datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
DATA_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────
# Dataset 1 — Customer Churn Classification (clean)
# ─────────────────────────────────────────────────────────────

def generate_classification(n: int = 500) -> pd.DataFrame:
    age = np.random.randint(18, 72, n)
    tenure = np.random.randint(1, 121, n)
    charges = np.round(np.random.uniform(19.5, 118.8, n), 2)
    num_products = np.random.randint(1, 6, n)
    support_tickets = np.random.poisson(1.4, n).clip(0, 8)
    has_premium = np.random.choice(["Yes", "No"], n, p=[0.38, 0.62])
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.22, 0.22, 0.22],
    )

    # Churn probability driven by features (realistic correlations)
    logit = (
        -3.0
        + 0.015 * charges
        - 0.025 * tenure
        + 0.10 * support_tickets
        + 0.40 * (contract_type == "Month-to-month").astype(float)
        - 0.30 * (has_premium == "Yes").astype(float)
        + np.random.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "num_products": num_products,
        "support_tickets": support_tickets,
        "has_premium": has_premium,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churned": churned,
    })


# ─────────────────────────────────────────────────────────────
# Dataset 2 — Housing Price Regression (clean)
# ─────────────────────────────────────────────────────────────

def generate_regression(n: int = 400) -> pd.DataFrame:
    sqft = np.random.randint(650, 4200, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n)
    year_built = np.random.randint(1960, 2024, n)
    distance_to_center = np.round(np.random.exponential(8, n).clip(0.5, 40), 1)
    lot_size = np.round(np.random.uniform(0.1, 1.2, n), 2)
    neighborhood = np.random.choice(
        ["Downtown", "Suburbs", "Rural", "Waterfront"], n, p=[0.20, 0.50, 0.20, 0.10]
    )

    neigh_bonus = {"Downtown": 30_000, "Suburbs": 0, "Rural": -25_000, "Waterfront": 80_000}
    base_price = (
        80 * sqft
        + 8_000 * bedrooms
        + 12_000 * bathrooms
        + 500 * (year_built - 1960)
        - 3_500 * distance_to_center
        + 20_000 * lot_size
        + np.array([neigh_bonus[n_] for n_ in neighborhood])
        + np.random.normal(0, 18_000, n)
    ).clip(80_000)

    return pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "distance_to_center_km": distance_to_center,
        "lot_size_acres": lot_size,
        "neighborhood_type": neighborhood,
        "sale_price": np.round(base_price, -2).astype(int),
    })


# ─────────────────────────────────────────────────────────────
# Dataset 3 — Employee Attrition (messy: missing values + noisy categoricals)
# ─────────────────────────────────────────────────────────────

def generate_messy(n: int = 600) -> pd.DataFrame:
    age = np.random.randint(22, 62, n).astype(float)
    years_at_company = np.random.randint(0, 20, n).astype(float)
    salary_band = np.random.choice(["Low", "Medium", "High", "Senior"], n, p=[0.30, 0.40, 0.20, 0.10])
    overtime = np.random.choice(["Yes", "No"], n, p=[0.28, 0.72])
    last_promo_years = np.random.randint(0, 10, n).astype(float)
    wfh_days = np.random.choice([0, 1, 2, 3, 4, 5], n).astype(float)
    perf_rating = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.12, 0.40, 0.30, 0.13])

    # Noisy department names
    dept_clean = np.random.choice(
        ["Engineering", "Sales", "Marketing", "HR", "Finance"], n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    noise_map = {
        "Engineering": ["Engineering", "Eng", "Eng.", "engineering", "ENGINEERING"],
        "Sales": ["Sales", "sales", "Sales Dept", "SALES"],
        "Marketing": ["Marketing", "Mktg", "marketing", "Mktg."],
        "HR": ["HR", "Human Resources", "H.R.", "hr"],
        "Finance": ["Finance", "Fin", "finance", "FINANCE"],
    }
    dept = [np.random.choice(noise_map[d]) for d in dept_clean]

    job_role = np.random.choice(
        ["Analyst", "Manager", "Director", "Associate", "Specialist", "Lead"], n,
        p=[0.28, 0.22, 0.10, 0.20, 0.12, 0.08],
    )

    logit = (
        -2.5
        - 0.04 * years_at_company
        + 0.8 * (overtime == "Yes").astype(float)
        - 0.5 * (salary_band == "High").astype(float)
        - 0.7 * (salary_band == "Senior").astype(float)
        + 0.3 * last_promo_years
        - 0.2 * perf_rating
        + np.random.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    left = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "department": dept,
        "job_role": job_role,
        "salary_band": salary_band,
        "years_at_company": years_at_company,
        "overtime": overtime,
        "last_promotion_years": last_promo_years,
        "wfh_days_per_week": wfh_days,
        "performance_rating": perf_rating.astype(float),
        "left_company": left,
    })

    # Inject realistic missing values
    miss_mask = {
        "age": 0.05,
        "last_promotion_years": 0.18,
        "wfh_days_per_week": 0.22,
        "performance_rating": 0.12,
        "salary_band": 0.08,
    }
    rng = np.random.default_rng(99)
    for col, rate in miss_mask.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────────────────────
# Dataset 4 — Customer Churn with Subtle Data Leakage
# ─────────────────────────────────────────────────────────────

def generate_leakage(n: int = 450) -> pd.DataFrame:
    """
    Same structure as Dataset 1, but with an added `risk_assessment_score`
    column that is a near-perfect function of `churned` (the target).

    The column name sounds like a legitimate pre-computed business score,
    making it a realistic data-leakage scenario. The agent's critique node
    should detect this via feature importance and/or correlation analysis.
    """
    df_base = generate_classification(n)
    # Derive leakage column from target (added AFTER the fact, as if a risk system computed it)
    noise = np.random.normal(0, 5, n)
    df_base["risk_assessment_score"] = (
        df_base["churned"] * 82 + noise
    ).clip(0, 100).round(1)
    return df_base


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

DATASETS = [
    ("dataset_01_classification", generate_classification),
    ("dataset_02_regression", generate_regression),
    ("dataset_03_messy", generate_messy),
    ("dataset_04_leakage", generate_leakage),
]


def generate_all():
    for name, fn in DATASETS:
        df = fn()
        out = DATA_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  Generated {out}  {df.shape}")


if __name__ == "__main__":
    print("Generating evaluation datasets...")
    generate_all()
    print("Done.")

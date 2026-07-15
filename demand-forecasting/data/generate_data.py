"""
Generate synthetic daily retail demand data.
Produces 21,900 rows across 5 product categories x 4 SKUs x 1095 days (3 years).
Incorporates: upward trend, weekly seasonality, annual seasonality,
holiday spikes (Black Friday, Christmas, July 4th, etc.), and realistic noise.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).parent.parent
OUT_CSV = ROOT / "data" / "demand_data.csv"

dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
N = len(dates)  # 1095 days

CATEGORIES = {
    "Electronics": {
        "base": 300, "trend": 0.08, "annual_peak_doy": 358,  # late Dec
        "annual_amp": 0.40, "weekly_amp": 0.20, "weekend_lift": 0.25,
        "noise": 0.12, "n_skus": 4,
    },
    "Clothing": {
        "base": 450, "trend": 0.05, "annual_peak_doy": 315,  # mid-Nov
        "annual_amp": 0.28, "weekly_amp": 0.18, "weekend_lift": 0.30,
        "noise": 0.15, "n_skus": 4,
    },
    "Food": {
        "base": 800, "trend": 0.03, "annual_peak_doy": 359,  # Christmas
        "annual_amp": 0.18, "weekly_amp": 0.10, "weekend_lift": 0.15,
        "noise": 0.08, "n_skus": 4,
    },
    "Sports": {
        "base": 220, "trend": 0.06, "annual_peak_doy": 185,  # early July
        "annual_amp": 0.45, "weekly_amp": 0.28, "weekend_lift": 0.40,
        "noise": 0.18, "n_skus": 4,
    },
    "Home": {
        "base": 280, "trend": 0.04, "annual_peak_doy": 100,  # spring
        "annual_amp": 0.22, "weekly_amp": 0.16, "weekend_lift": 0.22,
        "noise": 0.13, "n_skus": 4,
    },
}

# Holiday multipliers per category
HOLIDAY_EFFECTS = {
    "Christmas":       {"Electronics": 3.2, "Clothing": 2.2, "Food": 2.8, "Sports": 1.8, "Home": 2.0},
    "Black Friday":    {"Electronics": 3.8, "Clothing": 3.2, "Food": 1.3, "Sports": 2.2, "Home": 2.8},
    "Cyber Monday":    {"Electronics": 2.8, "Clothing": 2.0, "Food": 1.1, "Sports": 1.8, "Home": 2.0},
    "Pre-Christmas":   {"Electronics": 1.8, "Clothing": 1.6, "Food": 1.6, "Sports": 1.3, "Home": 1.5},
    "New Year":        {"Electronics": 0.7, "Clothing": 1.1, "Food": 1.8, "Sports": 1.4, "Home": 0.8},
    "Valentine":       {"Electronics": 1.1, "Clothing": 1.4, "Food": 1.5, "Sports": 1.0, "Home": 1.2},
    "July 4th":        {"Electronics": 0.9, "Clothing": 1.0, "Food": 2.2, "Sports": 2.4, "Home": 1.3},
    "Halloween":       {"Electronics": 1.0, "Clothing": 1.5, "Food": 1.8, "Sports": 1.0, "Home": 1.6},
    "Post-Christmas":  {"Electronics": 1.8, "Clothing": 1.6, "Food": 1.2, "Sports": 1.3, "Home": 1.4},
}


def _black_friday_dates():
    bfs = []
    for yr in [2021, 2022, 2023]:
        # 4th Thursday in November
        nov = pd.date_range(f"{yr}-11-01", f"{yr}-11-30", freq="D")
        thursdays = [d for d in nov if d.dayofweek == 3]
        thanksgiving = thursdays[3]
        bfs.append(thanksgiving + pd.Timedelta(days=1))   # Black Friday
        bfs.append(thanksgiving + pd.Timedelta(days=4))   # Cyber Monday
    return bfs


def _build_holiday_map():
    """Return {date: holiday_name} dict covering 2021-2023."""
    hmap = {}
    for yr in [2021, 2022, 2023]:
        hmap[pd.Timestamp(f"{yr}-01-01")] = "New Year"
        hmap[pd.Timestamp(f"{yr}-02-14")] = "Valentine"
        hmap[pd.Timestamp(f"{yr}-07-04")] = "July 4th"
        hmap[pd.Timestamp(f"{yr}-10-31")] = "Halloween"
        hmap[pd.Timestamp(f"{yr}-12-25")] = "Christmas"
        hmap[pd.Timestamp(f"{yr}-12-26")] = "Post-Christmas"
        for d in range(20, 25):
            hmap[pd.Timestamp(f"{yr}-12-{d}")] = "Pre-Christmas"
    bf_dates = _black_friday_dates()
    for i in range(0, len(bf_dates), 2):
        hmap[bf_dates[i]] = "Black Friday"
        hmap[bf_dates[i + 1]] = "Cyber Monday"
    return hmap


HOLIDAY_MAP = _build_holiday_map()


def generate_sku_demand(dates, cat_name, cfg, sku_idx):
    N = len(dates)
    sku_factor = 0.6 + sku_idx * 0.3   # SKU size multiplier: 0.6, 0.9, 1.2, 1.5
    rng = np.random.default_rng(42 + sku_idx + sum(ord(c) for c in cat_name))

    records = []
    for i, dt in enumerate(dates):
        day_num = i
        doy = dt.day_of_year

        # Trend component
        trend = 1.0 + cfg["trend"] * (day_num / 365)

        # Annual seasonality (sine curve, peak at annual_peak_doy)
        shift = 2 * np.pi * (doy - cfg["annual_peak_doy"]) / 365
        annual = 1.0 + cfg["annual_amp"] * np.cos(shift)

        # Weekly seasonality
        dow = dt.dayofweek  # 0=Mon, 6=Sun
        if dow >= 5:  # weekend
            weekly = 1.0 + cfg["weekend_lift"]
        else:
            weekly = 1.0 - cfg["weekly_amp"] * np.sin(2 * np.pi * dow / 7) * 0.3

        # Holiday effect
        holiday_mult = 1.0
        if dt in HOLIDAY_MAP:
            h_name = HOLIDAY_MAP[dt]
            holiday_mult = HOLIDAY_EFFECTS[h_name].get(cat_name, 1.0)

        # Base demand
        base = cfg["base"] * sku_factor

        # Final demand with multiplicative noise
        noise = rng.lognormal(0, cfg["noise"])
        demand = max(1, round(base * trend * annual * weekly * holiday_mult * noise))

        # Revenue (price varies by SKU and has slight trend)
        price = (8 + sku_idx * 4) * (1 + 0.02 * day_num / 365)
        revenue = round(demand * price, 2)

        is_hol = 1 if dt in HOLIDAY_MAP else 0

        records.append({
            "date": dt,
            "category": cat_name,
            "sku_id": f"{cat_name[:3].upper()}_{sku_idx+1:02d}",
            "units_sold": demand,
            "revenue": revenue,
            "is_holiday": is_hol,
            "holiday_name": HOLIDAY_MAP.get(dt, ""),
            "day_of_week": dow,
            "month": dt.month,
            "quarter": dt.quarter,
            "year": dt.year,
        })
    return records


def main():
    all_records = []
    for cat_name, cfg in CATEGORIES.items():
        for sku_idx in range(cfg["n_skus"]):
            records = generate_sku_demand(dates, cat_name, cfg, sku_idx)
            all_records.extend(records)
        print(f"  Generated {cat_name} ({cfg['n_skus']} SKUs)")

    df = pd.DataFrame(all_records)
    df = df.sort_values(["date", "category", "sku_id"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_CSV}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Total SKUs: {df['sku_id'].nunique()}")
    print(df.groupby("category")["units_sold"].agg(["sum", "mean"]).round(1))
    return df


if __name__ == "__main__":
    main()

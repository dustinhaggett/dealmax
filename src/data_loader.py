"""
Loads listing CSVs into a DataFrame and synthesizes realistic data
when no CSV is provided.

Schema (required columns):
    property_id, address, city, state, zip, latitude, longitude,
    list_price, sqft, beds, baths, year_built, property_condition,
    days_on_market, estimated_monthly_rent
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "property_id", "address", "city", "state", "zip",
    "latitude", "longitude", "list_price", "sqft", "beds", "baths",
    "year_built", "property_condition", "days_on_market",
    "estimated_monthly_rent",
]

CONDITION_LEVELS = ["excellent", "good", "fair", "poor", "distressed"]


def load_listings(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Load the listings CSV. If the file is missing, fall back to
    synthetic data so the MVP still runs end-to-end."""
    path = Path(csv_path)
    if not path.exists():
        print(f"[data_loader] {path} not found. Generating synthetic data.")
        df = generate_synthetic_listings(n=100)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def generate_synthetic_listings(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a believable distressed-property dataset.

    We pick a handful of metro markets with different price levels so
    the comp logic has something to work with (it requires multiple
    listings inside the same ZIP / nearby radius).
    """
    rng = np.random.default_rng(seed)

    # (city, state, base_zip, base_lat, base_lon, base_$/sqft, base_rent_psf)
    markets = [
        ("Phoenix",     "AZ", 85001, 33.4484, -112.0740, 235, 1.30),
        ("Atlanta",     "GA", 30301, 33.7490,  -84.3880, 210, 1.25),
        ("Cleveland",   "OH", 44101, 41.4993,  -81.6944, 130, 1.05),
        ("Tampa",       "FL", 33601, 27.9506,  -82.4572, 260, 1.45),
        ("Indianapolis","IN", 46201, 39.7684,  -86.1581, 145, 1.10),
    ]

    rows = []
    for i in range(n):
        city, state, base_zip, lat0, lon0, base_psf, rent_psf = markets[i % len(markets)]

        # Cluster lat/lon tightly so comps behave like a real neighborhood.
        lat = lat0 + rng.normal(0, 0.02)
        lon = lon0 + rng.normal(0, 0.02)
        zip_code = base_zip + rng.integers(0, 25)

        sqft = int(np.clip(rng.normal(1650, 450), 700, 3500))
        beds = int(np.clip(round(sqft / 600 + rng.normal(0, 0.5)), 1, 6))
        baths = float(np.clip(round((beds - 0.5) * 2) / 2, 1.0, 4.0))
        year_built = int(np.clip(rng.normal(1975, 25), 1900, 2024))

        condition = rng.choice(
            CONDITION_LEVELS, p=[0.10, 0.25, 0.30, 0.25, 0.10]
        )

        # Distressed pricing: discount $/sqft based on condition. The
        # "true" market $/sqft is base_psf with neighborhood noise; the
        # listing is sold at a discount because it needs work.
        market_psf = base_psf * (1 + rng.normal(0, 0.08))
        condition_discount = {
            "excellent": 1.00,
            "good":      0.95,
            "fair":      0.85,
            "poor":      0.72,
            "distressed":0.58,
        }[condition]

        list_price = int(market_psf * sqft * condition_discount
                         * (1 + rng.normal(0, 0.05)))
        list_price = max(list_price, 30_000)

        days_on_market = int(np.clip(rng.exponential(45), 1, 400))
        monthly_rent = int(rent_psf * sqft * (1 + rng.normal(0, 0.10)))

        rows.append({
            "property_id":  f"P{i+1:04d}",
            "address":      f"{rng.integers(100, 9999)} {rng.choice(['Oak','Maple','Pine','Elm','Cedar'])} St",
            "city":         city,
            "state":        state,
            "zip":          int(zip_code),
            "latitude":     round(float(lat), 6),
            "longitude":    round(float(lon), 6),
            "list_price":   list_price,
            "sqft":         sqft,
            "beds":         beds,
            "baths":        baths,
            "year_built":   year_built,
            "property_condition": condition,
            "days_on_market":     days_on_market,
            "estimated_monthly_rent": monthly_rent,
        })

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

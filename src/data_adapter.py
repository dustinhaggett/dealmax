"""
Adapter that conforms a Kaggle 'USA Real Estate Dataset'-style CSV
into the project's required schema.

Recommended source dataset
--------------------------
    Kaggle: ahmedshahriarsakib/usa-real-estate-dataset
    Columns: brokered_by, status, price, bed, bath, acre_lot, street,
             city, state, zip_code, house_size, prev_sold_date

The Kaggle source notably lacks several fields the project requires.
We fill them in as follows:

    latitude / longitude    geocoded from `zip_code` via pgeocode
    year_built              not in the source -> defaults to 1985
    property_condition      inferred from $/sqft deviation vs the
                            local (city) median; year_built buckets
                            adjust the baseline up or down
    days_on_market          derived from `prev_sold_date` if present,
                            else default 30
    estimated_monthly_rent  0.7% of list_price (national rule of thumb)

Cleaning:
    - Keep only rows where status == 'for_sale' (skip sold/pending).
    - Drop rows missing price, sqft, beds, or location.
    - Generate a stable property_id (`K{idx:06d}`).
    - Coerce ZIPs to 5-digit zero-padded strings then back to ints
      so geocoding works.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Required output schema (must match data_loader.REQUIRED_COLUMNS).
TARGET_COLUMNS = [
    "property_id", "address", "city", "state", "zip",
    "latitude", "longitude", "list_price", "sqft", "beds", "baths",
    "year_built", "property_condition", "days_on_market",
    "estimated_monthly_rent",
]

# Direct column renames Kaggle -> project schema.
KAGGLE_RENAME = {
    "price":       "list_price",
    "bed":         "beds",
    "bath":        "baths",
    "house_size":  "sqft",
    "city":        "city",
    "state":       "state",
    "zip_code":    "zip",
    "street":      "address",
}

# Approximate fix-and-flip rule of thumb: 0.7% of price per month rent.
RENT_TO_PRICE = 0.007

CONDITION_TIERS = ["distressed", "poor", "fair", "good", "excellent"]
DEFAULT_YEAR_BUILT = 1985


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _geocode_zips(zips: pd.Series) -> pd.DataFrame:
    """ZIP -> (latitude, longitude) via pgeocode (offline, US dataset).

    Returns a DataFrame with `latitude` and `longitude` aligned to `zips`.
    Falls back to NaN for unknown ZIPs; caller drops those rows.
    """
    try:
        import pgeocode  # lazy import: pgeocode downloads a US data
                          # file from a CDN on first use.
    except ImportError as exc:
        raise ImportError(
            "pgeocode is required for the Kaggle adapter. "
            "Install it with: pip install pgeocode>=0.4"
        ) from exc

    nomi = pgeocode.Nominatim("us")
    zips_str = zips.astype("Int64").astype(str).str.zfill(5)
    results = nomi.query_postal_code(zips_str.tolist())
    return results[["latitude", "longitude"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Condition inference
# ---------------------------------------------------------------------------

def _condition_from_year(year_built: float) -> int:
    """Map year_built to a baseline condition tier index (0..4)."""
    if pd.isna(year_built):
        return 3  # "good"
    y = int(year_built)
    if y < 1960:
        return 1  # poor
    if y < 1990:
        return 2  # fair
    if y < 2010:
        return 3  # good
    return 4      # excellent


def _infer_conditions(df: pd.DataFrame) -> pd.Series:
    """Infer property_condition from year_built bucket + $/sqft deviation
    versus the local (city) median.

    Steep negative deviations (priced well below the local market) imply
    a distressed or poor-condition property. A modest premium implies a
    well-renovated property. We adjust the year-based baseline by up to
    +/- 2 tiers and clip into [0, 4].
    """
    psf = df["list_price"] / df["sqft"]
    city_median_psf = psf.groupby(df["city"]).transform("median")
    deviation = (psf - city_median_psf) / city_median_psf

    base = df["year_built"].apply(_condition_from_year).astype(int)

    adjustment = np.zeros(len(df), dtype=int)
    adjustment[deviation < -0.30] = -2
    adjustment[(deviation >= -0.30) & (deviation < -0.15)] = -1
    adjustment[(deviation > 0.20)] = 1

    tier = (base + adjustment).clip(0, 4)
    return tier.map(lambda i: CONDITION_TIERS[int(i)])


# ---------------------------------------------------------------------------
# DOM derivation
# ---------------------------------------------------------------------------

def _days_on_market(prev_sold: pd.Series,
                    today: Optional[dt.date] = None) -> pd.Series:
    """Days since `prev_sold_date`, capped at 400; default 30 if missing.

    Note: Kaggle's `prev_sold_date` is the last *sold* date, not the
    listing date. Using it as DOM is an approximation — a long gap
    since the last sale is correlated with stale listings or
    long-time owners reluctant to negotiate. Good enough for an MVP.
    """
    if today is None:
        today = dt.date.today()
    parsed = pd.to_datetime(prev_sold, errors="coerce")
    days = (pd.Timestamp(today) - parsed).dt.days
    days = days.fillna(30).clip(lower=1, upper=400).astype(int)
    return days


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_kaggle_realtor(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Load a Kaggle 'USA Real Estate Dataset'-style CSV and conform it
    to the project's required schema.

    Parameters
    ----------
    csv_path:
        Path to the raw Kaggle CSV (or any CSV with the same columns).

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns exactly match `TARGET_COLUMNS`. Rows
        that cannot be cleaned (missing critical fields) are dropped.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {path}")

    raw = pd.read_csv(path)

    # 1) Active listings only (drop sold/pending if status column exists).
    if "status" in raw.columns:
        raw = raw[raw["status"].astype(str).str.lower() == "for_sale"].copy()

    # 2) Rename the columns we want to keep.
    keep = {k: v for k, v in KAGGLE_RENAME.items() if k in raw.columns}
    df = raw.rename(columns=keep)

    # 3) Carry through prev_sold_date for DOM derivation as a temp
    #    column so it survives the reset_index calls below.
    if "prev_sold_date" in raw.columns:
        df["_prev_sold"] = raw["prev_sold_date"]
    else:
        df["_prev_sold"] = pd.NaT

    # 4) Drop rows missing critical fields.
    must_have = ["list_price", "sqft", "beds", "city", "state", "zip"]
    df = df.dropna(subset=must_have).copy()

    # 5) Coerce types.
    df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce")
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["baths"] = pd.to_numeric(df.get("baths"), errors="coerce")
    df["zip"] = pd.to_numeric(df["zip"], errors="coerce")
    df = df.dropna(subset=["list_price", "sqft", "beds", "zip"]).copy()
    df["zip"] = df["zip"].astype(int)
    df["beds"] = df["beds"].astype(int)
    df["baths"] = df["baths"].fillna(df["beds"].clip(lower=1)).astype(float)

    # Filter obvious garbage / outliers.
    df = df[(df["list_price"] >= 20_000) & (df["list_price"] <= 5_000_000)]
    df = df[(df["sqft"] >= 400) & (df["sqft"] <= 15_000)]

    # 6) Geocode lat/lon.
    df = df.reset_index(drop=True)
    coords = _geocode_zips(df["zip"]).reset_index(drop=True)
    df["latitude"] = coords["latitude"]
    df["longitude"] = coords["longitude"]
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # 7) Synthesize the missing fields.
    df["property_id"] = [f"K{i:06d}" for i in range(len(df))]
    df["address"] = df.get("address", pd.Series([""] * len(df))).fillna("")
    df["year_built"] = DEFAULT_YEAR_BUILT  # not in Kaggle source.
    df["property_condition"] = _infer_conditions(df)
    df["days_on_market"] = _days_on_market(df["_prev_sold"])
    df["estimated_monthly_rent"] = (df["list_price"] * RENT_TO_PRICE).round(0)

    # 8) Final ordering.
    df = df[TARGET_COLUMNS].reset_index(drop=True)
    return df

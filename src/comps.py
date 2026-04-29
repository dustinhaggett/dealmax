"""
Comparable-sales (comps) engine.

ARV = median price_per_sqft of nearby/comparable properties * subject_sqft

Comps are pulled from the same dataset using simple filters that mirror
how an investor screens comps in the MLS:
    1. same ZIP (or within a small lat/lon radius as a fallback),
    2. sqft within +/- 25% of the subject,
    3. similar bed count (+/- 1).

We strip the subject row out, take the median $/sqft of what's left,
and multiply by the subject sqft. Median (not mean) keeps the ARV
robust to one weird flip or one trashed house in the bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approx miles per degree of latitude. Used for the fallback radius
# search when ZIP-level comps are too thin.
MILES_PER_DEG_LAT = 69.0
COMP_RADIUS_MILES = 1.5
MIN_COMPS = 3


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def _select_comps(subject: pd.Series, pool: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered DataFrame of comps for one subject property."""
    pool = pool[pool["property_id"] != subject["property_id"]]

    sqft_lo = subject["sqft"] * 0.75
    sqft_hi = subject["sqft"] * 1.25
    bed_lo = subject["beds"] - 1
    bed_hi = subject["beds"] + 1

    # 1) Try same ZIP first.
    same_zip = pool[
        (pool["zip"] == subject["zip"]) &
        (pool["sqft"].between(sqft_lo, sqft_hi)) &
        (pool["beds"].between(bed_lo, bed_hi))
    ]
    if len(same_zip) >= MIN_COMPS:
        return same_zip

    # 2) Fallback: radius search.
    dist = _haversine_miles(
        subject["latitude"], subject["longitude"],
        pool["latitude"].values, pool["longitude"].values,
    )
    nearby = pool.assign(_dist=dist)
    nearby = nearby[
        (nearby["_dist"] <= COMP_RADIUS_MILES) &
        (nearby["sqft"].between(sqft_lo, sqft_hi)) &
        (nearby["beds"].between(bed_lo, bed_hi))
    ]
    if len(nearby) >= MIN_COMPS:
        return nearby.drop(columns="_dist")

    # 3) Final fallback: just the closest 5 in the same metro (city).
    metro = pool[pool["city"] == subject["city"]].copy()
    metro["_dist"] = _haversine_miles(
        subject["latitude"], subject["longitude"],
        metro["latitude"].values, metro["longitude"].values,
    )
    return metro.nsmallest(5, "_dist").drop(columns="_dist")


def estimate_arv(df: pd.DataFrame) -> pd.DataFrame:
    """Add `comp_psf` (median $/sqft of comps) and `arv` columns.

    Important: we use comparable LIST prices as a proxy for sold prices
    here. In production you'd swap in MLS sold-comps. The structure of
    the calculation is identical.
    """
    df = df.copy()
    df["price_per_sqft"] = df["list_price"] / df["sqft"]

    comp_psf = np.zeros(len(df))
    n_comps = np.zeros(len(df), dtype=int)

    for i, subject in df.iterrows():
        comps = _select_comps(subject, df)
        # Adjust upward: comps are listed (often distressed too), so
        # we anchor on the upper half of the distribution to approximate
        # a renovated-condition ARV. Using the 60th percentile is a
        # simple, transparent way to do this.
        if len(comps) > 0:
            comp_psf[i] = float(np.percentile(comps["price_per_sqft"], 60))
            n_comps[i] = len(comps)
        else:
            comp_psf[i] = float(subject["price_per_sqft"])
            n_comps[i] = 0

    df["comp_psf"] = comp_psf
    df["n_comps"] = n_comps
    df["arv"] = (df["comp_psf"] * df["sqft"]).round(0)
    return df

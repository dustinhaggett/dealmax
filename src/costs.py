"""
Cost estimation for a fix-and-flip underwrite.

We model four cost buckets:

    rehab_cost     - condition-driven $/sqft estimate
    closing_costs  - % of purchase price (title, lender, inspections)
    holding_costs  - monthly carry * holding period
    selling_costs  - % of ARV (agent commission + seller closing)

These are intentionally simple, transparent rules of thumb. Real
underwriting would plug in lender-specific rates, market-specific
commission splits, and a contractor's line-item rehab scope. The
structure here is set up so each constant can be swapped for a
data-driven model later without touching the optimizer.
"""

from __future__ import annotations

import pandas as pd

# Rehab $/sqft by condition. These are conservative national averages
# for a light-to-medium cosmetic flip.
REHAB_PSF = {
    "excellent":   5.0,
    "good":       15.0,
    "fair":       30.0,
    "poor":       55.0,
    "distressed": 85.0,
}

# Percentages applied to purchase price / ARV.
CLOSING_PCT_OF_PURCHASE = 0.02   # title, lender fees, inspections
SELLING_PCT_OF_ARV      = 0.07   # 6% commission + ~1% seller closing

# Holding model: assume hard-money + utilities + taxes + insurance.
HOLDING_MONTHS_DEFAULT  = 6
HOLDING_RATE_PCT_PURCHASE = 0.012   # ~1.2% / month of purchase price
                                    # (12% APR hard money on ~100% LTC,
                                    # plus ~1% blended utilities/tax/ins)


def rehab_cost(row: pd.Series) -> float:
    psf = REHAB_PSF.get(str(row["property_condition"]).lower(), 30.0)
    return float(psf * row["sqft"])


def closing_costs(purchase_price: float) -> float:
    return CLOSING_PCT_OF_PURCHASE * purchase_price


def holding_costs(purchase_price: float,
                  months: int = HOLDING_MONTHS_DEFAULT) -> float:
    return HOLDING_RATE_PCT_PURCHASE * purchase_price * months


def selling_costs(arv: float) -> float:
    return SELLING_PCT_OF_ARV * arv


def total_costs(purchase_price: float,
                rehab: float,
                arv: float,
                holding_months: int = HOLDING_MONTHS_DEFAULT) -> float:
    """Sum of rehab + closing + holding + selling for a given P, ARV."""
    return (rehab
            + closing_costs(purchase_price)
            + holding_costs(purchase_price, holding_months)
            + selling_costs(arv))


def attach_rehab(df: pd.DataFrame) -> pd.DataFrame:
    """Add `rehab_cost` column derived from condition + sqft."""
    df = df.copy()
    df["rehab_cost"] = df.apply(rehab_cost, axis=1).round(0)
    return df

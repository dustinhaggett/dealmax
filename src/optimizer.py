"""
Bid optimizer.

For each property we want the LARGEST purchase price P that still
satisfies all of the investor's constraints simultaneously:

    (1) ROI    = profit / (P + rehab)         >= target_roi
    (2) profit = ARV - P - rehab - costs(P,A) >= minimum_profit
    (3) capital = P + rehab                   <= available_budget

Costs are not constant in P -- closing and holding are percentages of
purchase price. Letting:

    c_close   = CLOSING_PCT_OF_PURCHASE
    c_hold    = HOLDING_RATE_PCT * holding_months
    c_sell    = SELLING_PCT_OF_ARV

we have:
    profit = (1 - c_sell) * ARV - P*(1 + c_close + c_hold) - rehab

Each constraint reduces to a closed-form upper bound on P, and the
binding constraint is the minimum of those bounds. This is just a
1-D linear program with three half-plane constraints, so we solve it
analytically rather than calling out to a solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import (
    CLOSING_PCT_OF_PURCHASE,
    HOLDING_RATE_PCT_PURCHASE,
    HOLDING_MONTHS_DEFAULT,
    SELLING_PCT_OF_ARV,
    total_costs,
)


def _max_p_from_roi(arv: float, rehab: float,
                    target_roi: float, holding_months: int) -> float:
    """Largest P satisfying constraint (1).

    Derivation:
        profit >= target_roi * (P + rehab)
        (1 - c_sell)*ARV - P*(1+c_close+c_hold) - rehab
            >= target_roi*P + target_roi*rehab
        => P * (1 + c_close + c_hold + target_roi)
           <= (1 - c_sell)*ARV - rehab*(1 + target_roi)
    """
    c_close = CLOSING_PCT_OF_PURCHASE
    c_hold  = HOLDING_RATE_PCT_PURCHASE * holding_months
    c_sell  = SELLING_PCT_OF_ARV

    numerator   = (1 - c_sell) * arv - rehab * (1 + target_roi)
    denominator = 1 + c_close + c_hold + target_roi
    return numerator / denominator


def _max_p_from_profit(arv: float, rehab: float,
                       min_profit: float, holding_months: int) -> float:
    """Largest P satisfying constraint (2)."""
    c_close = CLOSING_PCT_OF_PURCHASE
    c_hold  = HOLDING_RATE_PCT_PURCHASE * holding_months
    c_sell  = SELLING_PCT_OF_ARV

    numerator   = (1 - c_sell) * arv - rehab - min_profit
    denominator = 1 + c_close + c_hold
    return numerator / denominator


def _max_p_from_budget(rehab: float, budget: float) -> float:
    """Largest P satisfying constraint (3): P + rehab <= budget."""
    return budget - rehab


def optimize_row(row: pd.Series,
                 target_roi: float,
                 min_profit: float,
                 budget: float,
                 holding_months: int = HOLDING_MONTHS_DEFAULT) -> pd.Series:
    arv   = float(row["arv"])
    rehab = float(row["rehab_cost"])
    list_price = float(row["list_price"])

    p_roi    = _max_p_from_roi(arv, rehab, target_roi, holding_months)
    p_profit = _max_p_from_profit(arv, rehab, min_profit, holding_months)
    p_budget = _max_p_from_budget(rehab, budget)

    # Max bid the investor SHOULD offer = tightest of the three.
    max_bid = min(p_roi, p_profit, p_budget)
    binding = ["roi", "profit", "budget"][int(np.argmin([p_roi, p_profit, p_budget]))]

    # Realistic offer = min(max_bid, list_price). You'll never need to
    # bid above list to win a distressed deal in this model.
    offer_price = max(0.0, min(max_bid, list_price))

    # Recompute the actual cost stack at the chosen offer price.
    costs_at_offer = total_costs(offer_price, rehab, arv, holding_months)
    profit_at_offer = arv - offer_price - costs_at_offer
    capital_at_offer = offer_price + rehab
    roi_at_offer = profit_at_offer / capital_at_offer if capital_at_offer > 0 else 0.0

    feasible = (
        max_bid > 0
        and offer_price > 0
        and roi_at_offer >= target_roi - 1e-9
        and profit_at_offer >= min_profit - 1e-6
        and capital_at_offer <= budget + 1e-6
    )

    return pd.Series({
        "max_bid":           round(max(0.0, max_bid), 0),
        "offer_price":       round(offer_price, 0),
        "binding_constraint": binding,
        "total_costs":       round(costs_at_offer, 0),
        "expected_profit":   round(profit_at_offer, 0),
        "expected_roi":      round(roi_at_offer, 4),
        "capital_required":  round(capital_at_offer, 0),
        "feasible":          bool(feasible),
    })


def optimize(df: pd.DataFrame,
             target_roi: float,
             min_profit: float,
             budget: float,
             holding_months: int = HOLDING_MONTHS_DEFAULT) -> pd.DataFrame:
    """Run the optimizer on every row of `df`.

    Expects df to already have `arv` and `rehab_cost` columns.
    """
    out = df.apply(
        lambda r: optimize_row(r, target_roi, min_profit, budget, holding_months),
        axis=1,
    )
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

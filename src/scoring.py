"""
Deal scoring + ranking.

The score is a weighted, normalized blend of four signals an investor
actually cares about:

    1. ROI                  - higher is better
    2. Absolute profit $    - bigger wins beat small wins
    3. List-price discount  - how far below ARV is the seller asking?
                              Proxy for motivated seller / equity room.
    4. Days on market       - older listings are more negotiable

Each signal is min-max scaled to [0, 1] across the dataset and then
combined with fixed weights. Min-max keeps the score readable
(0 = worst in this batch, 1 = best in this batch) and avoids one
outlier blowing up the ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "roi":        0.40,
    "profit":     0.30,
    "discount":   0.20,
    "dom":        0.10,
}


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def score_deals(df: pd.DataFrame) -> pd.DataFrame:
    """Add `deal_score` and rank-sort the DataFrame.

    Infeasible deals (those the optimizer flagged as not meeting the
    investor's constraints) are kept in the output but pushed to the
    bottom by zeroing their score.
    """
    df = df.copy()

    discount = (df["arv"] - df["list_price"]) / df["arv"]

    s_roi      = _minmax(df["expected_roi"].clip(lower=0))
    s_profit   = _minmax(df["expected_profit"].clip(lower=0))
    s_discount = _minmax(discount.clip(lower=0))
    s_dom      = _minmax(df["days_on_market"])

    score = (WEIGHTS["roi"]      * s_roi
             + WEIGHTS["profit"] * s_profit
             + WEIGHTS["discount"] * s_discount
             + WEIGHTS["dom"]    * s_dom)

    df["list_to_arv_discount"] = discount.round(4)
    df["deal_score"] = score.round(4)
    df.loc[~df["feasible"], "deal_score"] = 0.0

    df = df.sort_values(
        ["feasible", "deal_score"], ascending=[False, False]
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df

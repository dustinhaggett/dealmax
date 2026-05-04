"""Monte Carlo risk simulation for deal economics.

This module samples ARV, rehab costs, and holding duration around the
underwrite point estimates to produce profit/ROI distributions and a
feasibility probability for each deal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import total_costs


def simulate_deal(row: pd.Series,
                  n_sims: int = 1000,
                  target_roi: float = 0.18,
                  min_profit: float = 25_000,
                  holding_months: int = 6,
                  budget: float = 250_000,
                  seed: int = 42,
                  arv_sigma: float = 0.10,
                  rehab_sigma: float = 0.30,
                  holding_sigma: float = 1.5,
                  rng: np.random.Generator | None = None) -> dict[str, float]:
    """Monte Carlo simulate profit + ROI at the deal's deterministic offer."""
    if rng is None:
        rng = np.random.default_rng(seed)

    arv = float(row["arv"])
    rehab = float(row["rehab_cost"])
    offer_price = float(row["offer_price"])

    arvs = rng.normal(arv, arv_sigma * arv, size=n_sims)
    arvs = np.clip(arvs, 0.0, None)

    rehabs = rng.normal(rehab, rehab_sigma * rehab, size=n_sims)
    rehabs = np.clip(rehabs, 0.0, None)

    holding = rng.normal(holding_months, holding_sigma, size=n_sims)
    holding = np.clip(holding, 1.0, 18.0)

    costs = total_costs(offer_price, rehabs, arvs, holding)
    profit = arvs - offer_price - costs
    capital = offer_price + rehabs
    roi = np.divide(profit, capital, out=np.zeros_like(profit), where=capital > 0)

    feasible = (
        (profit >= min_profit)
        & (roi >= target_roi)
        & (capital <= budget)
    )

    profit_p5, profit_p50, profit_p95 = np.percentile(profit, [5, 50, 95])
    roi_p5, roi_p50, roi_p95 = np.percentile(roi, [5, 50, 95])

    return {
        "p_feasible": round(float(feasible.mean()), 4),
        "profit_p5": round(float(profit_p5), 0),
        "profit_p50": round(float(profit_p50), 0),
        "profit_p95": round(float(profit_p95), 0),
        "roi_p5": round(float(roi_p5), 4),
        "roi_p50": round(float(roi_p50), 4),
        "roi_p95": round(float(roi_p95), 4),
        "mean_profit": round(float(profit.mean()), 0),
        "mean_roi": round(float(roi.mean()), 4),
    }


def simulate_all(df: pd.DataFrame,
                 target_roi: float,
                 min_profit: float,
                 holding_months: int,
                 budget: float,
                 n_sims: int = 1000,
                 seed: int = 42) -> pd.DataFrame:
    """Run Monte Carlo simulation for every row in the DataFrame."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    risk_rows = []

    for _, row in df.iterrows():
        risk_rows.append(
            simulate_deal(
                row,
                n_sims=n_sims,
                target_roi=target_roi,
                min_profit=min_profit,
                holding_months=holding_months,
                budget=budget,
                rng=rng,
            )
        )

    risk_df = pd.DataFrame(risk_rows)
    return pd.concat([df.reset_index(drop=True), risk_df], axis=1)


def risk_adjusted_score(df: pd.DataFrame) -> pd.Series:
    """Risk-adjusted expected profit = expected profit * probability feasible."""
    return df["expected_profit"] * df["p_feasible"]

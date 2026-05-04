"""Tests for Monte Carlo risk simulation and deterministic equality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.costs import total_costs
from src.risk import simulate_deal


def test_simulate_deal_with_zero_sigma_returns_deterministic_profit():
    row = pd.Series({
        "arv": 320_000.0,
        "rehab_cost": 40_000.0,
        "offer_price": 150_000.0,
    })

    result = simulate_deal(
        row,
        n_sims=1000,
        target_roi=0.18,
        min_profit=25_000,
        holding_months=6,
        budget=250_000,
        seed=123,
        arv_sigma=0.0,
        rehab_sigma=0.0,
        holding_sigma=0.0,
    )

    deterministic_profit = (
        row["arv"]
        - row["offer_price"]
        - total_costs(row["offer_price"], row["rehab_cost"], row["arv"], 6)
    )

    assert result["profit_p50"] == pytest.approx(deterministic_profit, abs=1e-6)
    assert result["profit_p5"] == pytest.approx(deterministic_profit, abs=1e-6)
    assert result["profit_p95"] == pytest.approx(deterministic_profit, abs=1e-6)
    assert result["mean_profit"] == pytest.approx(deterministic_profit, abs=1e-6)

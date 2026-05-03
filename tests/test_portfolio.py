"""
Tests for src/portfolio.py.

Verifies that:
1. The MILP solution is at least as good as the greedy solution (optimality).
2. Both solvers respect the bankroll constraint.
3. Both handle edge cases: empty feasible set, bankroll = 0, single deal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import select_portfolio, select_portfolio_greedy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(profits, capitals, feasible=None):
    """Build a minimal DataFrame that portfolio functions expect."""
    n = len(profits)
    if feasible is None:
        feasible = [True] * n
    return pd.DataFrame({
        "property_id":    [f"P{i:04d}" for i in range(n)],
        "expected_profit": list(profits),
        "capital_required": list(capitals),
        "expected_roi":   [p / c if c > 0 else 0.0 for p, c in zip(profits, capitals)],
        "feasible":       list(feasible),
    })


def _portfolio_profit(df):
    return df.loc[df["selected"], "expected_profit"].sum()


def _portfolio_capital(df):
    return df.loc[df["selected"], "capital_required"].sum()


# ---------------------------------------------------------------------------
# Core correctness: MILP >= greedy
# ---------------------------------------------------------------------------

class TestMilpVsGreedy:
    """MILP must find at least as much profit as the greedy heuristic."""

    def test_milp_beats_or_ties_greedy_small(self):
        # Classic counter-example for greedy: item 0 has best ROI but is
        # too expensive; items 1+2 together fit and yield more profit.
        profits  = [100, 70, 60]
        capitals = [80,  50, 40]
        bankroll = 90
        df = _make_df(profits, capitals)

        milp_df   = select_portfolio(df, bankroll)
        greedy_df = select_portfolio_greedy(df, bankroll)

        assert _portfolio_profit(milp_df) >= _portfolio_profit(greedy_df), (
            "MILP must achieve at least as much profit as greedy"
        )

    def test_milp_beats_greedy_counter_example(self):
        # Greedy picks item 0 (ROI=1.0, profit=100, capital=100).
        # But items 1+2 together give profit=120 and fit in bankroll=110.
        profits  = [100, 70, 50]
        capitals = [100, 60, 50]
        bankroll = 110
        df = _make_df(profits, capitals)

        milp_df   = select_portfolio(df, bankroll)
        greedy_df = select_portfolio_greedy(df, bankroll)

        milp_profit   = _portfolio_profit(milp_df)
        greedy_profit = _portfolio_profit(greedy_df)

        assert milp_profit >= greedy_profit
        # Specifically, MILP should find 120 here.
        assert milp_profit == pytest.approx(120, abs=1e-3)

    def test_milp_greedy_agree_when_all_fit(self):
        # If everything fits in the bankroll, both should select everything.
        profits  = [50, 80, 30]
        capitals = [40, 60, 20]
        bankroll = 500
        df = _make_df(profits, capitals)

        milp_df   = select_portfolio(df, bankroll)
        greedy_df = select_portfolio_greedy(df, bankroll)

        assert milp_df["selected"].sum() == 3
        assert greedy_df["selected"].sum() == 3
        assert _portfolio_profit(milp_df) == pytest.approx(160, abs=1e-3)

    def test_milp_vs_greedy_random(self):
        """Random synthetic instances: MILP profit >= greedy profit."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 20)
            profits  = rng.uniform(10_000, 100_000, size=n).tolist()
            capitals = rng.uniform(20_000, 200_000, size=n).tolist()
            bankroll = float(rng.uniform(100_000, 500_000))

            df = _make_df(profits, capitals)
            milp_df   = select_portfolio(df, bankroll)
            greedy_df = select_portfolio_greedy(df, bankroll)

            assert _portfolio_profit(milp_df) >= _portfolio_profit(greedy_df) - 1e-3, (
                "MILP profit should be >= greedy profit on every random instance"
            )


# ---------------------------------------------------------------------------
# Constraint satisfaction
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_milp_respects_bankroll(self):
        profits  = [50, 80, 30, 60]
        capitals = [40, 60, 20, 55]
        bankroll = 100
        df = _make_df(profits, capitals)
        out = select_portfolio(df, bankroll)
        assert _portfolio_capital(out) <= bankroll + 1e-6

    def test_greedy_respects_bankroll(self):
        profits  = [50, 80, 30, 60]
        capitals = [40, 60, 20, 55]
        bankroll = 100
        df = _make_df(profits, capitals)
        out = select_portfolio_greedy(df, bankroll)
        assert _portfolio_capital(out) <= bankroll + 1e-6

    def test_portfolio_rank_is_contiguous(self):
        profits  = [50, 80, 30, 60]
        capitals = [40, 60, 20, 55]
        bankroll = 200
        df = _make_df(profits, capitals)
        out = select_portfolio(df, bankroll)
        ranks = sorted(out.loc[out["selected"], "portfolio_rank"].tolist())
        assert ranks == list(range(1, len(ranks) + 1))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_feasible_set(self):
        df = _make_df([100, 200], [50, 80], feasible=[False, False])
        out = select_portfolio(df, 300)
        assert out["selected"].sum() == 0

    def test_zero_bankroll(self):
        df = _make_df([100, 200], [50, 80])
        out = select_portfolio(df, 0)
        assert out["selected"].sum() == 0

    def test_single_deal_fits(self):
        df = _make_df([100], [50])
        out = select_portfolio(df, 100)
        assert out["selected"].iloc[0] is True or out["selected"].iloc[0] == True

    def test_single_deal_does_not_fit(self):
        df = _make_df([100], [200])
        out = select_portfolio(df, 100)
        assert out["selected"].sum() == 0

    def test_infeasible_rows_never_selected(self):
        df = _make_df([1_000_000], [1], feasible=[False])
        out = select_portfolio(df, 1_000_000)
        assert out["selected"].sum() == 0

    def test_greedy_empty_feasible_set(self):
        df = _make_df([100], [50], feasible=[False])
        out = select_portfolio_greedy(df, 500)
        assert out["selected"].sum() == 0

    def test_output_columns_present(self):
        df = _make_df([50, 80], [40, 60])
        out = select_portfolio(df, 200)
        assert "selected" in out.columns
        assert "portfolio_rank" in out.columns

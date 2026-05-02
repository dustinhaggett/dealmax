"""
Portfolio-level optimizer for fix-and-flip deal selection.

Problem formulation
-------------------
Given a set of n feasible deals (each with a capital requirement c_i and an
expected profit p_i), and a total bankroll B, select the subset that maximises
total expected profit without exceeding the bankroll:

    maximise   sum_i  p_i * x_i
    subject to sum_i  c_i * x_i  <= B
               x_i in {0, 1}   for all i

This is the classic 0/1 knapsack problem, a member of the NP-hard family of
integer programmes.

Why it is non-convex
--------------------
The feasible set {x in {0,1}^n} is a finite (discrete) set of 2^n vertices of
the unit hypercube [0,1]^n.  Discrete feasible sets are never convex (you
cannot form a continuous convex combination of two feasible points and stay
feasible), so the problem as a whole is non-convex, and standard gradient-based
or convex-programming algorithms cannot guarantee a global optimum.

The LP relaxation is convex
---------------------------
If we relax the integrality constraint from x_i in {0,1} to 0 <= x_i <= 1 we
obtain a linear programme (LP):

    maximise   p^T x
    subject to c^T x  <= B
               0 <= x <= 1

The feasible region of an LP is a convex polytope (intersection of half-spaces),
and the objective is linear (hence both convex and concave), so the LP
relaxation is a convex optimisation problem solvable in polynomial time.  The
LP optimum is an upper bound on the MILP optimum and can be used to bound the
branch-and-bound search tree.  PuLP uses CBC (or another MIP solver) which
exploits this LP/MILP relationship internally.

Implementation notes
--------------------
* ``select_portfolio`` – exact MILP via PuLP / CBC.
* ``select_portfolio_greedy`` – greedy ROI-sorted heuristic used as a baseline.
  The greedy solution is always feasible but may be suboptimal.
"""

from __future__ import annotations

import pandas as pd


def select_portfolio(df: pd.DataFrame, total_bankroll: float) -> pd.DataFrame:
    """Select the profit-maximising subset of feasible deals within *total_bankroll*.

    Solves the 0/1 knapsack MILP with PuLP (CBC solver).  Only rows where
    ``feasible == True`` are eligible; all other rows are passed through with
    ``selected = False`` and ``portfolio_rank = 0``.

    Parameters
    ----------
    df:
        Output of :func:`src.scoring.score_deals` (must contain ``feasible``,
        ``capital_required``, and ``expected_profit`` columns).
    total_bankroll:
        Maximum total capital that can be deployed across all selected deals.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with two new columns:
        * ``selected``       – bool, True iff the deal is in the optimal portfolio.
        * ``portfolio_rank`` – int 1..k for selected deals (ranked by expected
          profit descending); 0 for non-selected / infeasible rows.
    """
    try:
        import pulp
    except ImportError as exc:
        raise ImportError(
            "PuLP is required for portfolio optimisation. "
            "Install it with: pip install pulp>=2.7"
        ) from exc

    df = df.copy()
    df["selected"] = False
    df["portfolio_rank"] = 0

    feasible_mask = df["feasible"].astype(bool)
    candidates = df[feasible_mask].copy()

    if candidates.empty:
        return df

    n = len(candidates)
    idx = candidates.index.tolist()
    profits = candidates["expected_profit"].tolist()
    capitals = candidates["capital_required"].tolist()

    prob = pulp.LpProblem("portfolio_knapsack", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximise total expected profit.
    prob += pulp.lpSum(profits[i] * x[i] for i in range(n))

    # Constraint: total capital <= bankroll.
    prob += pulp.lpSum(capitals[i] * x[i] for i in range(n)) <= total_bankroll

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    selected_flags = [pulp.value(x[i]) for i in range(n)]

    for i, orig_idx in enumerate(idx):
        if selected_flags[i] is not None and selected_flags[i] > 0.5:
            df.at[orig_idx, "selected"] = True

    # Rank selected deals by expected profit (highest first).
    selected_rows = df[df["selected"]].sort_values("expected_profit", ascending=False)
    for rank, orig_idx in enumerate(selected_rows.index, start=1):
        df.at[orig_idx, "portfolio_rank"] = rank

    return df


def select_portfolio_greedy(df: pd.DataFrame, total_bankroll: float) -> pd.DataFrame:
    """Greedy ROI-sorted baseline for the portfolio knapsack.

    Sorts feasible deals by ``expected_roi`` descending and greedily adds each
    deal while the bankroll allows.  Runs in O(n log n) — useful as a fast
    baseline and sanity-check against the MILP solution.

    Parameters
    ----------
    df:
        Same schema requirements as :func:`select_portfolio`.
    total_bankroll:
        Maximum total capital to deploy.

    Returns
    -------
    pd.DataFrame
        Same augmented schema as :func:`select_portfolio`.
    """
    df = df.copy()
    df["selected"] = False
    df["portfolio_rank"] = 0

    feasible_mask = df["feasible"].astype(bool)
    candidates = df[feasible_mask].sort_values("expected_roi", ascending=False)

    remaining = total_bankroll
    rank = 1
    for orig_idx, row in candidates.iterrows():
        cap = float(row["capital_required"])
        if cap <= remaining:
            df.at[orig_idx, "selected"] = True
            df.at[orig_idx, "portfolio_rank"] = rank
            remaining -= cap
            rank += 1

    return df

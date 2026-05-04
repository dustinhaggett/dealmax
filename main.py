"""
DealMax -- distressed real estate deal finder & bid optimizer.

Pipeline:
    1. load listings (or synthesize them)
    2. estimate ARV from comps
    3. estimate rehab cost from condition
    4. optimize the max bid against ROI / profit / budget constraints
    5. score and rank
    6. write ranked CSV + charts

Usage:
    python main.py --input data/listings.csv \\
                   --budget 250000 \\
                   --target-roi 0.18 \\
                   --min-profit 25000 \\
                   --holding-months 6 \\
                   --output outputs/ranked_deals.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_listings
from src.data_adapter import load_kaggle_realtor
from src.comps import estimate_arv
from src.costs import attach_rehab, HOLDING_MONTHS_DEFAULT
from src.optimizer import optimize
from src.risk import risk_adjusted_score, simulate_all
from src.scoring import score_deals
from src.visualize import plot_all, plot_risk_fan
from src.portfolio import select_portfolio


OUTPUT_COLUMNS = [
    "rank", "property_id", "address", "city", "state", "zip",
    "list_price", "sqft", "beds", "baths", "property_condition",
    "days_on_market", "n_comps", "comp_psf", "arv",
    "rehab_cost", "total_costs", "max_bid", "offer_price",
    "expected_profit", "expected_roi", "capital_required",    "p_feasible", "profit_p5", "profit_p50", "profit_p95",
    "roi_p5", "roi_p50", "roi_p95",
    "mean_profit", "mean_roi", "risk_adjusted_score",    "list_to_arv_discount", "binding_constraint",
    "feasible", "deal_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/listings.csv",
                   help="Path to input listings CSV. If missing, synthetic data is generated.")
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "kaggle-realtor"],
                   help="Where the input CSV comes from. 'synthetic' (default) "
                        "expects the project's native schema; 'kaggle-realtor' "
                        "expects the Kaggle 'USA Real Estate Dataset' columns "
                        "and routes through src.data_adapter.")
    p.add_argument("--output", default="outputs/ranked_deals.csv",
                   help="Where to write the ranked deals CSV.")
    p.add_argument("--charts-dir", default="outputs",
                   help="Directory for chart PNGs.")
    p.add_argument("--budget", type=float, default=250_000,
                   help="Total capital available per deal (purchase + rehab).")
    p.add_argument("--target-roi", type=float, default=0.18,
                   help="Minimum acceptable ROI (e.g. 0.18 = 18%%).")
    p.add_argument("--min-profit", type=float, default=25_000,
                   help="Minimum acceptable expected profit in dollars.")
    p.add_argument("--holding-months", type=int, default=HOLDING_MONTHS_DEFAULT,
                   help="Months of holding cost to underwrite.")
    p.add_argument("--total-bankroll", type=float, default=None,
                   help="Total capital available across ALL deals. When set, "
                        "runs the portfolio MILP optimizer and writes "
                        "outputs/portfolio.csv.")
    p.add_argument("--simulate", action="store_true",
                   help="Run Monte Carlo risk simulation and add uncertainty metrics to the output.")
    p.add_argument("--n-sims", type=int, default=1000,
                   help="Number of Monte Carlo draws per deal when --simulate is enabled.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[main] Loading listings from {args.input} (source={args.source}) ...")
    if args.source == "kaggle-realtor":
        df = load_kaggle_realtor(args.input)
    else:
        df = load_listings(args.input)
    print(f"[main]   {len(df)} listings loaded.")

    print("[main] Estimating ARV from comps ...")
    df = estimate_arv(df)

    print("[main] Estimating rehab costs ...")
    df = attach_rehab(df)

    print(f"[main] Optimizing bids "
          f"(budget=${args.budget:,.0f}, target_roi={args.target_roi:.0%}, "
          f"min_profit=${args.min_profit:,.0f}, holding={args.holding_months}mo) ...")
    df = optimize(
        df,
        target_roi=args.target_roi,
        min_profit=args.min_profit,
        budget=args.budget,
        holding_months=args.holding_months,
    )

    if args.simulate:
        print(f"[main] Running Monte Carlo risk simulation ({args.n_sims} draws per deal) ...")
        df = simulate_all(
            df,
            target_roi=args.target_roi,
            min_profit=args.min_profit,
            holding_months=args.holding_months,
            budget=args.budget,
            n_sims=args.n_sims,
            seed=42,
        )
        df["risk_adjusted_score"] = risk_adjusted_score(df)

    print("[main] Scoring & ranking ...")
    df = score_deals(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df[cols].to_csv(out_path, index=False)
    print(f"[main] Wrote ranked deals -> {out_path}")

    print("[main] Generating charts ...")
    chart_paths = plot_all(df, args.charts_dir)
    for p in chart_paths:
        print(f"[main]   chart -> {p}")

    if args.simulate:
        risk_fan_path = plot_risk_fan(df, top_n=10, out_dir=args.charts_dir)
        chart_paths.append(risk_fan_path)
        print(f"[main]   chart -> {risk_fan_path}")

    feasible = df[df["feasible"]]
    print()
    print("=" * 60)
    print(f"Total properties analyzed:  {len(df)}")
    print(f"Feasible deals:             {len(feasible)}")
    if len(feasible) > 0:
        print(f"Median expected ROI:        {feasible['expected_roi'].median():.1%}")
        print(f"Median expected profit:     ${feasible['expected_profit'].median():,.0f}")
        top = feasible.iloc[0]
        print(f"Top deal:                   {top['property_id']} in {top['city']}, {top['state']}")
        print(f"  list / max_bid / ARV:     ${top['list_price']:,.0f} / ${top['max_bid']:,.0f} / ${top['arv']:,.0f}")
        print(f"  expected ROI / profit:    {top['expected_roi']:.1%} / ${top['expected_profit']:,.0f}")
    print("=" * 60)

    if args.total_bankroll is not None:
        print()
        print(f"[main] Running portfolio MILP optimizer (bankroll=${args.total_bankroll:,.0f}) ...")
        df = select_portfolio(df, args.total_bankroll)

        portfolio_path = Path(args.charts_dir) / "portfolio.csv"
        portfolio_cols = [c for c in OUTPUT_COLUMNS + ["selected", "portfolio_rank"]
                          if c in df.columns]
        portfolio_df = df[df["selected"]].sort_values("portfolio_rank")
        portfolio_df[portfolio_cols].to_csv(portfolio_path, index=False)
        print(f"[main] Wrote portfolio -> {portfolio_path}")

        selected = df[df["selected"]]
        capital_deployed = selected["capital_required"].sum()
        total_profit = selected["expected_profit"].sum()
        n_deals = len(selected)
        avg_roi = selected["expected_roi"].mean() if n_deals > 0 else 0.0

        print()
        print("=" * 60)
        print("Portfolio summary")
        print(f"  Deals selected:           {n_deals}")
        print(f"  Capital deployed:         ${capital_deployed:,.0f}  "
              f"({capital_deployed / args.total_bankroll:.1%} of bankroll)")
        print(f"  Total expected profit:    ${total_profit:,.0f}")
        print(f"  Average ROI (selected):   {avg_roi:.1%}")
        print("=" * 60)


if __name__ == "__main__":
    main()

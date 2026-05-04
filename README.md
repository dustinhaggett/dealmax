# DealMax — Distressed Real Estate Deal Finder & Bid Optimizer

DealMax ingests a CSV of U.S. property listings and answers the only
question a fix-and-flip investor actually cares about:

> **What is the most I can pay for this house and still make money?**

For every listing, it estimates after-repair value (ARV) from local
comps, estimates rehab cost from the property's condition, models the
full transaction cost stack, and then solves for the maximum purchase
price that satisfies the investor's ROI, profit, and capital
constraints. Deals are ranked, exported to CSV, and visualized.

---

## Quick start

```bash
pip install -r requirements.txt

# Will auto-generate 100 synthetic listings if the CSV doesn't exist.
python main.py --input data/listings.csv --budget 250000 --target-roi 0.18

# With portfolio selection across a $1 M bankroll:
python main.py --input data/listings.csv --budget 250000 --target-roi 0.18 \
               --total-bankroll 1000000
```

Outputs:

- `outputs/ranked_deals.csv` — every property scored and ranked
- `outputs/top_roi.png` — top 20 deals by ROI
- `outputs/bid_vs_list.png` — max bid vs list price scatter
- `outputs/profit_distribution.png` — histogram of expected profit

CLI flags:

| flag | default | meaning |
|---|---|---|
| `--input` | `data/listings.csv` | input listings CSV |
| `--source` | `synthetic` | `synthetic` (project schema) or `kaggle-realtor` (raw Kaggle USA Real Estate columns) |
| `--output` | `outputs/ranked_deals.csv` | ranked output CSV |
| `--budget` | `250000` | total capital per deal (purchase + rehab) |
| `--target-roi` | `0.18` | minimum acceptable ROI |
| `--min-profit` | `25000` | minimum acceptable expected profit ($) |
| `--holding-months` | `6` | months of carry to underwrite |
| `--total-bankroll` | *(skip)* | total capital across all deals; enables portfolio MILP optimizer |
| `--simulate` | `false` | run Monte Carlo risk simulation and emit uncertainty metrics |
| `--n-sims` | `1000` | number of Monte Carlo draws per deal when `--simulate` is enabled |

---

## Objective function

For each candidate property, find the maximum purchase price `P` that
maximizes ranked deal score subject to all of:

```
ROI    = profit / (P + rehab)         >= target_roi
profit = ARV - P - rehab - costs(P,A) >= min_profit
capital = P + rehab                   <= budget
```

`costs(P, ARV)` decomposes into four buckets:

```
rehab_cost     = REHAB_PSF[condition] * sqft
closing_costs  = 0.02 * P
holding_costs  = 0.012 * P * holding_months
selling_costs  = 0.07 * ARV
```

Because closing and holding are linear in `P` and selling is constant
in `ARV`, each constraint reduces to a closed-form upper bound on `P`.
The optimal max bid is the **minimum** of those three bounds — the
binding constraint is recorded for every deal so the investor can see
*why* a bid is capped (priced for ROI, priced for profit, or priced
out by budget). The math is implemented in `src/optimizer.py`; see the
docstring for the full derivation.

## Portfolio selection

When `--total-bankroll` is provided, DealMax runs a second optimisation pass
across all feasible deals to select the portfolio that **maximises total
expected profit** without exceeding the bankroll.

### MILP formulation

Let *n* be the number of feasible deals, *p_i* the expected profit and *c_i*
the capital required for deal *i*, and *B* the total bankroll:

```
maximise   Σ  p_i · x_i
subject to Σ  c_i · x_i  ≤  B
           x_i ∈ {0, 1}   ∀ i
```

This is the **0/1 knapsack** problem.  It is **NP-hard** because the feasible
set `{x ∈ {0,1}^n}` is discrete — you cannot form a continuous convex
combination of two integer-feasible points and stay feasible, so the problem is
non-convex and gradient methods cannot guarantee a global optimum.

The **LP relaxation** (replace `x_i ∈ {0,1}` with `0 ≤ x_i ≤ 1`) *is*
convex: the feasible region becomes a convex polytope and the objective is
linear.  The LP optimal value is an upper bound on the MILP and is used
internally by the branch-and-bound solver (PuLP / CBC).

Outputs:
- `outputs/portfolio.csv` — selected deals with `portfolio_rank` column
- Console summary: capital deployed, total expected profit, # deals, avg ROI.

A greedy ROI-sorted heuristic (`select_portfolio_greedy`) is included as a
baseline.  The MILP is always at least as good as the greedy solution.

## ARV estimation

```
ARV = (median price-per-sqft of comps) * subject_sqft
```

Comps are pulled from the same dataset using a tiered filter:

1. Same ZIP, sqft within ±25%, beds within ±1.
2. Within a 1.5-mile radius (haversine), same sqft/beds filter.
3. Closest 5 properties in the same metro as a final fallback.

We use the **60th percentile** of comp $/sqft rather than a flat
median, on the assumption that the comp pool itself contains
distressed inventory and a renovated subject would price into the
upper half of the local distribution. Median is robust to outliers in
a way that mean isn't — one bombed-out comp won't tank the ARV.

## Cost model

Rehab is condition-driven $/sqft:

| condition | $/sqft |
|---|---|
| excellent | 5 |
| good | 15 |
| fair | 30 |
| poor | 55 |
| distressed | 85 |

All other costs are simple percentages chosen to match common
fix-and-flip rules of thumb. Each constant lives at the top of
`src/costs.py` and can be swapped for a data-driven model
(per-market commissions, lender-specific rates, line-item rehab
scope) without touching the optimizer.

## Monte Carlo risk simulation

When `--simulate` is enabled, DealMax replaces point-estimate profit and
ROI with probabilistic distributions by sampling:

- ARV from `N(arv, 0.10 * arv)`
- rehab cost from `N(rehab_cost, 0.30 * rehab_cost)`
- holding months from `N(holding_months, 1.5)` clipped to `[1, 18]`

The simulation evaluates every draw at the deal's deterministic
`offer_price`, then reports:

- `p_feasible` — probability the sampled deal still meets ROI, profit,
  and budget criteria
- `profit_p5`, `profit_p50`, `profit_p95` — profit percentiles
- `roi_p5`, `roi_p50`, `roi_p95` — ROI percentiles
- `mean_profit`, `mean_roi`
- `risk_adjusted_score` = `expected_profit * p_feasible`

A new chart is emitted as `outputs/risk_fan.png`, showing the profit
uncertainty band for the top deals.

## Deal score

Once every deal has a max bid and projected economics, the score
blends four normalized signals:

| signal | weight | rationale |
|---|---|---|
| Expected ROI | 40% | core return metric |
| Expected profit ($) | 30% | absolute dollar size matters |
| List-to-ARV discount | 20% | proxy for motivated seller / equity |
| Days on market | 10% | older listings negotiate softer |

Each signal is min-max scaled across the dataset so the score reads
as 0 (worst in batch) to 1 (best in batch). Infeasible deals — those
that fail one of the constraints above — keep their row but score 0
and sort to the bottom.

---

## Dataset

Required columns:

```
property_id, address, city, state, zip, latitude, longitude,
list_price, sqft, beds, baths, year_built, property_condition,
days_on_market, estimated_monthly_rent
```

If the input CSV doesn't exist, `data_loader` synthesizes 100
listings across five metros (Phoenix, Atlanta, Cleveland, Tampa,
Indianapolis) with realistic $/sqft, condition mix, and DOM
distributions so the comp logic and optimizer have meaningful
clustering to work with.

`property_condition` is one of `excellent | good | fair | poor | distressed`.

### Real data (Kaggle adapter)

To run DealMax against real US listings, use the
[USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)
on Kaggle (~2.2M rows). Download the CSV, then:

```bash
python main.py --input path/to/realtor-data.csv \
               --source kaggle-realtor \
               --budget 250000 --target-roi 0.18
```

`src/data_adapter.py` conforms the raw Kaggle schema to the project's
required schema by:

- keeping only `status == 'for_sale'` rows
- dropping listings missing price / sqft / beds / location
- geocoding `zip_code` → `latitude / longitude` via `pgeocode`
- inferring `property_condition` from `$/sqft` deviation against the
  city median (price-per-sqft is a strong proxy for condition when
  controlling for location)
- deriving `days_on_market` from `prev_sold_date` (or defaulting to 30)
- estimating `estimated_monthly_rent` as `0.7%` of list price (a
  national fix-and-flip rule of thumb that lines up well with median
  US gross rent yields)
- defaulting `year_built` to 1985 (the Kaggle dataset doesn't include
  it; the condition inference still works because the price-deviation
  channel dominates)

A pre-cleaned 200-row sample lives at `data/listings_kaggle_sample.csv`
for offline demos, with the matching raw input at
`data/realtor_sample_raw.csv`.

## Project structure

```
.
├── main.py                    # CLI entry point
├── requirements.txt
├── data/                      # input CSVs (synthetic + Kaggle samples)
├── outputs/                   # ranked CSV + charts + portfolio.csv
├── tests/
│   ├── test_portfolio.py      # MILP + greedy portfolio tests
│   ├── test_risk.py           # Monte Carlo simulation tests
│   └── test_data_adapter.py   # Kaggle adapter schema/cleaning tests
└── src/
    ├── data_loader.py         # CSV load + synthetic generator
    ├── data_adapter.py        # Kaggle USA Real Estate -> project schema
    ├── comps.py               # ARV from comparable sales
    ├── costs.py               # rehab + transaction cost model
    ├── optimizer.py           # closed-form max-bid solver (per deal)
    ├── scoring.py             # weighted multi-signal ranking
    ├── portfolio.py           # 0/1 knapsack MILP + greedy baseline
    ├── risk.py                # Monte Carlo profit/ROI distributions
    └── visualize.py           # all charts (top-ROI, bid-vs-list, profit, fan)
```

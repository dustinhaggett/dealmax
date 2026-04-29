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
| `--output` | `outputs/ranked_deals.csv` | ranked output CSV |
| `--budget` | `250000` | total capital per deal (purchase + rehab) |
| `--target-roi` | `0.18` | minimum acceptable ROI |
| `--min-profit` | `25000` | minimum acceptable expected profit ($) |
| `--holding-months` | `6` | months of carry to underwrite |

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

## Project structure

```
.
├── main.py                    # CLI entry point
├── requirements.txt
├── data/                      # input CSVs
├── outputs/                   # ranked CSV + charts
└── src/
    ├── data_loader.py         # CSV load + synthetic generator
    ├── comps.py               # ARV from comparable sales
    ├── costs.py               # rehab + transaction cost model
    ├── optimizer.py           # closed-form max-bid solver
    ├── scoring.py             # weighted multi-signal ranking
    └── visualize.py           # the 3 required charts
```

---

## How this could become a business

The MVP is the underwriting brain. The business is the data + the
distribution channel wrapped around it.

**Where the moat lives.**

1. **Off-market lead supply.** The hard part of distressed real
   estate isn't underwriting — it's sourcing. Layer in pre-foreclosure
   filings, tax delinquency rolls, probate records, code violations,
   and absentee-owner skip-tracing data. Properties that show up in
   3+ of those lists are the ones that actually trade at a discount.
2. **A real comp engine.** Swap the same-CSV comps for live MLS sold
   data via an MLS aggregator (RESO Web API, Zillow's Bridge API).
   ARV accuracy is the single biggest lever on output quality.
3. **Rehab estimates from photos.** A vision model that scores
   listing photos for condition (roof, kitchen, bath, foundation
   cues) replaces the static $/sqft table with something defensible.
4. **Hyperlocal cost calibration.** Closing costs, taxes, insurance,
   commission splits, and contractor rates vary by market. Per-MSA
   constants derived from real closings beat national averages by
   100-300 bps of margin.

**Who pays.**

- **Solo flippers / small funds** ($99–$499/mo SaaS): a daily ranked
  list of feasible deals in their target ZIPs, with one-click offer
  generation.
- **iBuyers and institutional SFR buyers** (enterprise): bulk API
  access to feed their acquisition pipelines.
- **Hard money lenders** (rev share): same underwriting they're
  already doing internally, white-labeled — the lender gets better
  loan files, the borrower gets faster approvals.
- **Wholesalers** (per-deal): pay-per-feasible-lead in markets they
  cover.

**What the road to revenue looks like.**

1. Pick one metro. Source 5–10k listings/week. Run them through
   DealMax. Hand the top decile to 3–5 active flippers for free in
   exchange for outcome data (did they buy, what did rehab actually
   cost, what did it sell for).
2. Use that ground-truth data to recalibrate every constant in
   `costs.py` and the comp percentile in `comps.py`. Ship v2 with
   measurably better ARV and rehab MAE.
3. Charge. Expand to adjacent metros once the unit economics in the
   first one are clean.

The current MVP is the skeleton — the value compounds as the
constants in `comps.py` and `costs.py` get replaced by models trained
on real outcomes.

# DATATHON 2026 - ROUND 1 — Competition Overview

## 1) What this competition is about

This competition asks participants to **forecast future revenue** for a simulated **Vietnamese fashion e-commerce business**. The dataset covers more than a decade of operations and is intentionally organized as a **multi-table business database**, not a single flat training table.

The prediction target is **Revenue**, and the required forecasting granularity is **by product, date, and location**. In practice, this means the model must learn both temporal patterns and business structure: product catalog effects, customer/location effects, promotion effects, fulfillment effects, and operational constraints.

## 2) Official task framing

- **Problem type:** supervised regression / time-series forecasting with relational data
- **Target:** `Revenue`
- **Primary leaderboard metric:** **MAE (Mean Absolute Error)**
- **Forecast grain:** **product × date × location**
- **Training horizon (analytical table):** `sales.csv` covers **2012–2022**
- **Test horizon (analytical table):** `sample_submission.csv` covers **2023–2024**, with the target hidden

## 3) Dataset structure

The Kaggle description states that the competition data is split into **15 CSV files**, organized into **4 layers**, plus a submission template:

### Master data
Reference tables describing the business entities.

Expected files:
- `products.csv`
- `customers.csv`
- `promotions.csv`
- `locations.csv`

These tables supply relatively slow-changing business context: product catalog, customer acquisition/channel metadata, promotion windows, and geographic mapping.

### Transaction data
Event-level business activity.

Expected files:
- `orders.csv`
- `order_items.csv`
- `payments.csv`
- `shipments.csv`
- `returns.csv`
- `reviews.csv`

These tables let us reconstruct demand, realized sales, discounting, payment behavior, fulfillment, returns, and customer feedback.

### Analytical data
Pre-assembled forecasting tables.

Expected files:
- `sales.csv`
- `sample_submission.csv`

These are the most direct starting point for leaderboard modeling because they already represent the prediction grain.

### Operational data
Business operations and website activity.

Expected files:
- `inventory.csv`
- `web_traffic.csv`

These tables can provide lagged supply-side and demand-intent signals.

### Submission template
- `sample_submission.csv`

## 4) What makes the competition interesting

This is **not** just a classical univariate time series problem. The competition combines:

- **Temporal forecasting** across a long history
- **Relational modeling** across many connected tables
- **Business process structure** (orders → items → payments / shipments / returns / reviews)
- **Geographic variation** via zip/location mapping
- **Operational constraints** through inventory and stock indicators
- **Demand signals** from promotions and website traffic

A strong solution will typically need both:
1. a **good forecasting strategy**; and
2. a **clean feature engineering / join strategy**.

## 5) Recommended modeling philosophy

### Start from the analytical tables first
The safest baseline is to begin with `sales.csv` and `sample_submission.csv`, because they are already aligned with the competition target.

Recommended first baseline:
- Parse date columns
- Build calendar/time features
- Add product/location keys as categorical features
- Train a time-aware regression model
- Evaluate with a time-based validation split using MAE

Only after the baseline works should you enrich it using features from the transactional and operational tables.

### Then add relational features
The second stage is to aggregate raw tables into the same grain as the analytical tables:
- product × date × location
- or product × month × location if the analytical table is monthly

Useful aggregated feature families:
- lagged units sold / lagged revenue proxies
- promotion activity and discount intensity
- recent return rate
- recent average review score
- shipping delays or fulfillment rate
- lagged stock / stockout / sell-through indicators
- lagged site traffic / visitor intensity

## 6) Validation strategy

Because the public task is forecasting, **random train/validation splits are not acceptable as the main validation protocol**. Use **time-based validation**.

Recommended approach:
- train on an earlier period
- validate on a later contiguous block
- keep the validation grain identical to the competition target
- score with **MAE on Revenue**

Suggested ladder:
1. **Baseline split:** train through 2020, validate on 2021
2. **Second split:** train through 2021, validate on 2022
3. **Final training:** retrain on all available `sales.csv`

If the data is monthly, validate by month blocks. If daily, validate by contiguous date windows.

## 7) Leakage risks to watch for

This competition contains several tables with **post-order information**. Those are extremely valuable, but also dangerous.

### High-risk leakage sources

- **`shipments.csv`**
  - shipping and delivery events often happen *after* the order date
  - same-period shipment outcomes may leak downstream realization

- **`returns.csv`**
  - returns and refunds happen *after* a sale
  - same-period refund information can directly encode future revenue erosion

- **`reviews.csv`**
  - reviews occur only after delivery / usage
  - contemporaneous review signals can leak future behavior or post-sale outcomes

- **`inventory.csv`**
  - if the snapshot date is at end-of-month, using it to predict that same month can leak information from later in the period

- **website traffic tables**
  - using future-known traffic rather than lagged traffic would also leak demand information

### Safe rule
Only use features from another table if they are:
- available **strictly before** the forecast timestamp, or
- known **ex ante** (for example, product metadata, location mapping, promotion calendar if the promotion schedule is assumed known in advance)

## 8) Practical baseline roadmap

### Baseline A — analytical-only
Use only `sales.csv`:
- date features: year, month, week, day-of-week, quarter, holiday flags
- lag features on Revenue or proxy columns
- rolling means / rolling medians
- product/location categorical encoding

### Baseline B — analytical + master
Join:
- `products.csv`
- `locations.csv`
- optionally `customers.csv` aggregates if compatible with the target grain

### Baseline C — analytical + aggregated transactions
Aggregate:
- order counts
- item quantities
- discount amounts
- payment mix
- return rates
- review counts / mean score
- shipping speed summaries

### Baseline D — full system
Add lagged operational signals:
- stock on hand
- stockout indicators
- reorder flags
- sell-through rate
- web sessions / unique visitors / conversion proxies

## 9) Join map to keep in mind

Core business process:
- `orders` → `order_items`
- `orders` → `payments`
- `orders` → `shipments`
- `orders` → `returns`
- `orders` → `reviews`

Entity linkage:
- `order_items.product_id` → `products.product_id`
- `orders.customer_id` → `customers.customer_id`
- `orders.zip` or equivalent location key → `locations.zip`
- `order_items.promo_id` and/or order-level promotion IDs → `promotions`

Competition target linkage:
- analytical tables should be treated as the **source of truth for leaderboard rows**
- raw tables are primarily used to create **lagged or ex-ante features** aligned to the analytical grain

## 10) What the agents should assume

When Copilot, Antigravity, or any local coding agent works in this repo, it should assume:

1. The task is **MAE minimization on Revenue**.
2. The safe default unit of work is the **analytical table grain**.
3. Every feature from transactional/operational data must pass a **time-availability check**.
4. The first deliverable is always a **reproducible analytical-only baseline**.
5. Multi-table joins must be documented and reversible.
6. No schema should be guessed silently; the agent must inspect the actual downloaded CSV headers before coding irreversible assumptions.

## 11) Deliverables we want from the agents

Every serious modeling iteration should produce:
- a reproducible script or notebook
- a saved feature manifest
- a validation MAE
- a list of tables used
- a list of lag windows used
- a leakage checklist
- a submission file built from `sample_submission.csv`

## 12) Bottom line

This competition is best approached as a **time-aware relational forecasting problem**.

The winning pattern will likely be:
1. build a strong analytical baseline,
2. add carefully lagged business-process features,
3. validate with strict temporal splits,
4. avoid leakage from post-order tables,
5. iterate quickly and keep the feature pipeline auditable.

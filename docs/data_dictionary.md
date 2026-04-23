# DATATHON 2026 - ROUND 1 — Data Dictionary for Agents

## 0) Scope and trust level

This document is designed for **coding agents** and is based on the competition's public **Overview/Data descriptions** plus the repository's modeling needs.

Important:
- The file list below matches the competition description.
- Several columns are **explicitly confirmed** from the public Kaggle snippets.
- Some table details are **partially confirmed** and must be verified against the actual downloaded CSV headers before hard-coding assumptions.

Therefore this file should be treated as:
- a **project knowledge base**, and
- a **first-pass schema guide**, not a substitute for inspecting the real CSV files.

---

## 1) File inventory

### Master layer
1. `products.csv`
2. `customers.csv`
3. `promotions.csv`

### Transaction layer
4. `orders.csv`
5. `order_items.csv`
6. `payments.csv`
7. `shipments.csv`
8. `returns.csv`
9. `reviews.csv`

### Analytical layer
10. `sales.csv`
11. `sample_submission.csv`

### Operational layer
12. `inventory.csv`
13. `web_traffic.csv`

### Submission template
14. `sample_submission.csv`

---

## 2) Global business picture

The dataset simulates a **Vietnam fashion e-commerce business**. The analytical task is to forecast **Revenue** at a granular level aligned with **product, date, and location**.

This implies three schema principles:
1. `sales.csv` / `sample_submission.csv` define the **leaderboard row grain**.
2. Raw transactional tables describe the **events that generate sales**.
3. Operational tables provide **auxiliary demand/supply signals** that should usually be lagged.

---

## 3) Canonical join map

### Entity keys
- `product_id` → product entity
- `customer_id` → customer entity
- `zip` → location entity
- `order_id` → order entity
- `review_id` → review entity
- `promo_id` / `promo_id_1` / `promo_id_2` → promotion entity

### Main joins
- `orders.order_id` ↔ `order_items.order_id`
- `orders.order_id` ↔ `payments.order_id`
- `orders.order_id` ↔ `shipments.order_id`
- `orders.order_id` ↔ `returns.order_id`
- `orders.order_id` ↔ `reviews.order_id`
- `order_items.product_id` ↔ `products.product_id`
- `orders.customer_id` ↔ `customers.customer_id`
- `orders.zip` ↔ `locations.zip`
- `order_items.promo_id` or order-level promo columns ↔ `promotions`

### Relationship hints from the competition description
- `orders ↔ shipments`: **1 : 0 or 1**
- `orders ↔ returns`: **1 : 0 or many**
- `orders ↔ reviews`: **1 : 0 or many**

---

## 4) Table-by-table notes

## 4.1 `products.csv`

**Purpose**
Reference table for the product catalog.

**Likely grain**
One row per `product_id`.

**Confirmed columns**
- `product_id` — primary key
- `product_name`
- `category`
- `segment`
- `price`
- `cogs`

**Agent notes**
- `price` is retail price.
- `cogs` is cost of goods sold and is described as lower than `price`.
- Good source for static product features.

**Safe feature ideas**
- category encoding
- segment encoding
- price band
- gross-margin proxy = `price - cogs`
- ratio features such as `cogs / price`

**Leakage risk**
Low, unless the table changes over time and no effective dating exists.

---

## 4.2 `customers.csv`

**Purpose**
Customer reference table.

**Likely grain**
One row per `customer_id`.

**Confirmed columns**
- `customer_id` — primary key
- `zip`
- `city`
- `acquisition_channel` (nullable)

**Partially confirmed / verify locally**
- there may be additional demographic or location descriptors

**Agent notes**
- The overview says customer information includes demographics and acquisition channel.
- Use for customer-origin aggregates, but not as a direct row-level join to the analytical table unless the target grain supports it.

**Safe feature ideas**
- number of historical customers contributing to a product-location cell
- customer acquisition-channel mix aggregated by time and location
- new-vs-returning customer ratios if derivable from order history

**Leakage risk**
Low to medium; depends on whether aggregates are computed strictly from historical windows.

---

## 4.3 `promotions.csv`

**Purpose**
Promotion / discount calendar.

**Likely grain**
One row per promotion campaign.

**Confirmed columns**
- promotion identifier is implied (`promo_id` or equivalent)
- `start_date`
- `end_date`

**Partially confirmed / verify locally**
- discount amount or percentage
- promotion type / channel / scope
- product/category applicability

**Agent notes**
- Promotion features can be very powerful.
- This is generally safe if promotion schedules are known ex ante.

**Safe feature ideas**
- promotion active flag on forecast date
- number of overlapping promotions
- discount intensity
- days since promo start / until promo end

**Leakage risk**
Low if the table is a planned calendar; verify business meaning locally.

---

## 4.4 `orders.csv`

**Purpose**
Order-level transaction table.

**Likely grain**
One row per `order_id`.

**Confirmed columns**
- `order_id` — primary key
- `order_date`
- `customer_id` — FK to customers
- `zip` — shipping zip / location key
- `order_status`
- `payment_method`
- `device_type`
- `order_source`

**Competition snippet suggests possible additional columns**
- nullable `promo_id_1`
- nullable `promo_id_2`

**Agent notes**
- This is the central fact table at order level.
- `payment_method` here may represent the chosen method, while `payments.csv` stores normalized payment detail.
- `order_source` likely refers to marketing / acquisition source.

**Safe feature ideas**
- order counts by date / product / location after joining items
- device mix
- source-channel mix
- status ratios using only historical windows

**Leakage risk**
Medium.
- `order_status` can become dangerous if it reflects downstream fulfillment outcomes that were unknown at forecast time.

---

## 4.5 `order_items.csv`

**Purpose**
Line-item table linking orders to products and realized item-level pricing.

**Likely grain**
One row per `order_id × product_id` line item (or per unique line within an order).

**Confirmed columns**
- `order_id` — FK to orders
- `product_id` — FK to products
- `quantity`
- `unit_price` — described as post-promotion unit price
- `discount_amount` — total discount on the line
- `promo_id` — FK to promotions

**Agent notes**
- This table is critical for reconstructing realized demand and discount behavior.
- Strong source for product-time-location aggregates after joining order date and zip from `orders.csv`.

**Safe feature ideas**
- lagged quantity sold
- lagged revenue proxy = `quantity * unit_price`
- average realized selling price
- average discount per unit
- promotion penetration rate

**Leakage risk**
Medium if same-period aggregates are used to predict that same period; low if properly lagged.

---

## 4.6 `payments.csv`

**Purpose**
Payment detail table.

**Likely grain**
Competition description says **1:1 with orders**.

**Confirmed facts**
- `order_id` exists and links to `orders`
- the table is described as `payments.csv — Thanh toán (quan hệ 1:1 với orders)`
- an installment-count field exists

**Partially confirmed / verify locally**
- payment amount field name
- payment-type field name
- authorization / settlement timestamps if any

**Agent notes**
- Use mainly for payment-behavior aggregates, not as a primary predictor unless available before forecast time.

**Safe feature ideas**
- installment usage rate by location / period
- payment-method mix
- historical average installment count

**Leakage risk**
Medium.
- Payment completion behavior may lag order creation.

---

## 4.7 `shipments.csv`

**Purpose**
Fulfillment / shipment tracking.

**Likely grain**
At most one row per order according to the relationship summary.

**Confirmed columns**
- `order_id` — FK to orders
- `ship_date`

**Confirmed business rule**
- present only for relevant statuses such as shipped / delivered / returned

**Partially confirmed / verify locally**
- delivery date
- carrier / warehouse / shipping fee
- delay or SLA fields

**Agent notes**
- Good for lagged logistics-quality signals.
- Dangerous for same-period forecasting.

**Safe feature ideas**
- historical ship-rate
- historical average order-to-ship delay
- lagged delivered fraction

**Leakage risk**
High if not lagged.

---

## 4.8 `returns.csv`

**Purpose**
Returned-item / refund data.

**Likely grain**
Potentially multiple return rows per order.

**Confirmed columns**
- `return_quantity`
- `refund_amount`
- relationship to `orders` exists via `order_id`

**Likely additional keys to verify locally**
- `order_id`
- possibly `product_id`
- return reason / return date

**Agent notes**
- Valuable for modeling product quality and net-revenue erosion.
- Should almost always be used only in lagged form.

**Safe feature ideas**
- trailing return rate by product
- trailing refund rate by category and region
- lagged refunded-amount ratio

**Leakage risk**
Very high if same-period or future returns are used.

---

## 4.9 `reviews.csv`

**Purpose**
Customer feedback and product reviews.

**Likely grain**
One row per review.

**Confirmed columns**
- `review_id` — primary key
- `order_id` — FK to orders
- `product_id` — FK to products
- `customer_id` — FK to customers
- `review_date`

**Partially confirmed / verify locally**
- review score / rating
- review text or sentiment-like field

**Agent notes**
- Very useful for lagged quality / satisfaction indicators.
- Same-period reviews are a leakage hazard.

**Safe feature ideas**
- trailing review count
- trailing average rating
- review recency features

**Leakage risk**
High if contemporaneous.

---

## 4.10 `sales.csv`

**Purpose**
Primary modeling table for training.

**Official role**
Analytical revenue dataset covering **2012–2022**.

**Expected grain**
A row keyed by **product × date × location** or an equivalent analytical grain consistent with the competition target.

**Confirmed facts**
- contains the target `Revenue`
- `Revenue` is the leaderboard target
- public snippet also shows a `COGS` field in the analytical revenue description

**Columns to verify locally before hard-coding**
- exact date column name
- exact location key column name (`zip` or another identifier)
- any additional pre-aggregated measures such as quantity, orders, margin, etc.

**Agent notes**
- This should be the default table for all first baselines.
- If this table already contains sufficient laggable signals, do not overcomplicate the pipeline early.

**Safe feature ideas**
- lags of Revenue
- rolling means / medians / std
- seasonality features
- product and region encodings
- COGS-derived margin features, if available and allowed

**Leakage risk**
Low if using only past rows within the train horizon.

---

## 4.11 `sample_submission.csv`

**Purpose**
Primary scoring table for inference.

**Official role**
Analytical revenue dataset covering **2023–2024**, with the target hidden.

**Expected grain**
Same as `sales.csv`.

**Confirmed facts**
- same analytical layer as `sales.csv`
- target is hidden for leaderboard scoring

**Columns to verify locally**
- whether non-target measures such as `COGS` remain present
- exact row ID / key columns used in submission generation

**Agent notes**
- All feature engineering must reproduce exactly the same columns as in train, except for the hidden target.
- Never use any future-derived aggregate that mixes train and test periods incorrectly.

**Leakage risk**
Not applicable as a source table, but feature-building pipelines can easily contaminate it if fitted on full data incorrectly.

---

## 4.12 `inventory.csv`

**Purpose**
Monthly inventory snapshot table.

**Likely grain**
One row per `snapshot_date × product_id` and possibly location, depending on the actual CSV.

**Confirmed columns / signals**
- `snapshot_date`
- `product_id`
- stock / stockout / supply indicators are mentioned in the overview
- `reorder_flag`
- `sell_through_rate`

**Partially confirmed / verify locally**
- exact stock quantity column name
- whether location is included
- stockout count / days-in-stock / replenishment fields

**Agent notes**
- Because this is an end-of-period snapshot, same-period usage can leak information.

**Safe feature ideas**
- previous-month stock on hand
- lagged stockout flag
- lagged sell-through rate
- rolling stock pressure indicators

**Leakage risk**
High unless lagged.

---

## 4.13 `web_traffic.csv`

**Purpose**
Daily website activity.

**Likely grain**
One row per date, or per date × channel if the actual CSV contains breakdowns.

**Confirmed columns**
- `date`
- `sessions`
- `unique_visitors`

**Partially confirmed / verify locally**
- conversion-related fields
- device / channel breakdowns
- bounce or engagement fields

**Agent notes**
- This table can provide top-of-funnel demand proxies.
- If traffic is only global daily traffic, treat it as a calendar-like exogenous series.

**Safe feature ideas**
- lagged sessions
- lagged unique visitors
- rolling traffic trend
- traffic spikes around promotions or holidays

**Leakage risk**
Medium to high if future traffic is used.

---

## 4.14 `sample_submission.csv`

**Purpose**
Submission template.

**Agent notes**
- Use this file to guarantee correct row ordering and target column naming.
- Do not assume the submission schema from memory; inspect the file directly.

**Expected behavior**
- copy required identifier columns if present
- fill predicted `Revenue`
- preserve original row order unless explicitly instructed otherwise

---

## 5) Recommended local schema verification step

Before any serious coding, run a local schema inspection script to record:
- filename
- row count
- column names
- dtypes
- min/max date columns
- missingness
- unique counts for keys

Minimum rule for agents:
- **Never hard-code a column name that is not first checked against the actual CSV header.**

---

## 6) Grain alignment rules

When building modeling features, always map tables to the analytical grain.

### Preferred target grain
Use the grain represented in `sales.csv` / `sample_submission.csv`.

### Aggregation examples
- `orders` + `order_items` → aggregate to date × product × zip
- `returns` → trailing return rate by product × zip
- `reviews` → trailing mean score by product × zip
- `inventory` → lagged product stock features
- `web_traffic` → lagged global or channel traffic by date

If a raw table lacks location or product directly, join via order-level keys before aggregation.

---

## 7) Leakage checklist by table

### Low leakage tables
- `products.csv`
- `locations.csv`
- usually `promotions.csv` if it is a planned calendar

### Medium leakage tables
- `customers.csv`
- `orders.csv`
- `order_items.csv`
- `payments.csv`
- `web_traffic.csv`

### High leakage tables
- `shipments.csv`
- `returns.csv`
- `reviews.csv`
- `inventory.csv` if same-period end snapshots are used

Operational rule:
- if the event can occur **after the sale**, treat it as **lagged only**.

---

## 8) Priority order for modeling

1. `sales.csv`
2. `products.csv`
3. `locations.csv`
4. aggregated `orders.csv` + `order_items.csv`
5. `promotions.csv`
6. lagged `inventory.csv`
7. lagged `web_traffic.csv`
8. lagged `returns.csv` / `reviews.csv` / `shipments.csv`

This priority order is designed to maximize signal while minimizing accidental leakage.

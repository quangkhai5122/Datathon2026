# AGENTS.md — Working Rules for Copilot, Antigravity, and Other Local Coding Agents

## Mission

We are working on **Kaggle: DATATHON 2026 - ROUND 1**.

The task is to **forecast `Revenue`** for a simulated Vietnam fashion e-commerce business using a **multi-table historical dataset**. The primary metric is **MAE on the `Revenue` column**.

Your job as an agent is not just to write code. Your job is to help build a **reproducible, leakage-safe, time-aware forecasting pipeline**.

---

## Required reading order before coding

Before proposing code, always read these files in this exact order:

1. `competition_overview.md`
2. `data_dictionary.md`
3. the real CSV headers in `data/raw/`
4. any existing training / feature scripts in `src/`

Do **not** skip the real CSV inspection step.

---

## Non-negotiable rules

### Rule 1 — Do not guess schema silently
Never assume a column exists just because it is common in Kaggle or commerce datasets.

Before using any column, inspect the actual CSV header.

If the real file differs from the markdown docs, the **real CSV wins** and the docs should be updated.

### Rule 2 — The analytical tables are the default source of truth
Start from:
- `sales.csv`
- `sample_submission.csv`

These define the competition row grain. Do not start by flattening the entire warehouse unless explicitly asked.

### Rule 3 — Time-aware validation only
Do **not** use random train/test splits as the primary evaluation method.

Always use time-based validation.

Accepted patterns:
- train on early periods, validate on later periods
- rolling-origin validation
- blocked temporal folds

### Rule 4 — MAE is the main optimization target
Every modeling experiment must report:
- validation MAE
- split definition
- target transformation if any
- feature set used

Do not optimize mainly for RMSE or R² unless explicitly asked.

### Rule 5 — Leakage prevention is mandatory
Any feature from another table must answer this question:

> Would this information have been available **before** the forecast timestamp?

If the answer is uncertain, default to **lagging the feature** or excluding it.

---

## Competition facts to keep in memory

- Target: `Revenue`
- Metric: `MAE`
- Forecast grain: **product × date × location**
- `sales.csv`: historical analytical table
- `sasample_submission.csv`: future analytical table with hidden target
- Multi-table business schema includes products, customers, promotions, locations, orders, order_items, payments, shipments, returns, reviews, inventory, and web traffic

---

## Leakage policy by table

### Usually safe
- `products.csv`
- `locations.csv`
- `promotions.csv` if used as a known-in-advance calendar

### Use with caution
- `customers.csv`
- `orders.csv`
- `order_items.csv`
- `payments.csv`
- `web_traffic.csv`

### Lag only unless explicitly justified otherwise
- `shipments.csv`
- `returns.csv`
- `reviews.csv`
- `inventory.csv` if the snapshot is end-of-period

### Hard rule
Do not use same-period shipment / return / review outcomes to predict that same period's revenue.

---

## Expected modeling workflow

## Phase 1 — Baseline
Build a minimal baseline using only `sales.csv`.

Required steps:
1. parse date columns
2. identify product and location keys
3. create calendar features
4. build lag features from historical rows only
5. train a reproducible baseline regressor
6. evaluate with time-based MAE
7. generate a valid submission from `sample_submission.csv`

Good first model choices:
- LightGBM / XGBoost / CatBoost if available
- regularized linear model on lag features
- tree-based scikit-learn baseline if external libs are unavailable

## Phase 2 — Safe enrichment
Add only low-risk exogenous features first:
- product metadata
- location metadata
- promotion calendar
- lagged web traffic
- lagged inventory signals

## Phase 3 — Aggregated transactional features
Aggregate raw event tables to the analytical grain.

Examples:
- order count last 7 / 30 / 90 periods
- units sold last 7 / 30 / 90 periods
- discount intensity last 30 periods
- trailing return rate
- trailing mean review score
- trailing ship delay features

---

## Coding standards

### File organization
Prefer this project structure:

- `src/data/` for loading and schema inspection
- `src/features/` for feature builders
- `src/models/` for training / inference
- `src/eval/` for validation utilities
- `notebooks/` only for exploration, not final production logic

### Reproducibility
Every training script must:
- set a random seed
- print input row counts
- print train/validation date ranges
- print feature count
- print validation MAE
- save model artifacts if appropriate

### Explicitness
Prefer explicit code over hidden magic.

Good:
- named join keys
- named lag windows
- separate functions for train and inference

Bad:
- giant notebook cells with hidden state
- feature generation that behaves differently on train and test without documentation
- fitting encoders on combined train+test unless intentionally safe and documented

---

## Required outputs from any serious code proposal

Whenever you propose or modify training code, include:

1. **Assumed input files**
2. **Assumed key columns**
3. **Target column**
4. **Validation split rule**
5. **Feature families added**
6. **Leakage check summary**
7. **Expected output files**

If any of the above cannot be stated clearly, pause and inspect the data instead of guessing.

---

## Join discipline

When joining raw tables, follow this order unless there is a strong reason not to:

1. `orders`
2. `order_items`
3. add product metadata from `products`
4. add location metadata from `locations`
5. optionally add payment detail
6. optionally add shipment / return / review aggregates using lagged windows only

Never perform a many-to-many join without checking row explosion.

After every join:
- print row counts before and after
- verify key uniqueness
- verify no accidental duplication of the target grain

---

## Validation discipline

### Accepted validation patterns
- year holdout
- rolling monthly holdout
- rolling-origin backtesting

### Not acceptable as primary evidence
- random split across all dates
- k-fold that mixes future dates into training for past validation rows

### Minimum validation report
Every experiment must report:
- train period
- validation period
- MAE
- whether target is log-transformed
- whether predictions were clipped

---

## Feature engineering guardrails

### Allowed by default
- calendar features
- lag features
- rolling statistics
- product metadata
- location metadata
- promotion-active indicators

### Allowed only with documented lagging
- payment behavior
- return behavior
- review behavior
- shipment behavior
- inventory snapshots
- web traffic

### Not allowed unless explicitly justified
- any feature computed using future rows relative to the prediction timestamp
- any aggregation that touches the validation horizon while building training features
- using leaderboard or submission feedback as training signal

---

## Submission rules

When creating a submission:
- start from `sample_submission.csv`
- preserve required row order
- write the exact target column name expected by the competition
- save to `submissions/`
- include model name and validation score in filename when practical

Example filename:
- `submissions/lgbm_baseline_mae_12.345.csv`

---

## Preferred communication style from the agent

When answering in chat or comments:
- be concise but explicit
- list assumptions clearly
- separate confirmed facts from inferred guesses
- call out leakage concerns early
- recommend the smallest reproducible next step

---

## If the data disagrees with the docs

If the downloaded CSV files differ from `competition_overview.md` or `data_dictionary.md`:
1. trust the real CSVs,
2. update the docs,
3. explain what changed,
4. then continue coding.

---

## Definition of success

A successful agent contribution is one that:
- improves validation MAE,
- keeps the pipeline leakage-safe,
- stays reproducible,
- and makes the next iteration easier rather than more confusing.

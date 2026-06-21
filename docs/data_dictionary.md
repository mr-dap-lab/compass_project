# COMPASS — Engineered Feature Data Dictionary

One row per association (300 rows per table). Feature window = months 1–12; outcome window = months 13–18 (strict temporal split, no leakage).

## Model 1 — `model1_delinquency_features` (18 predictors + target)

| Feature | Meaning |
|---|---|
| association_type | HOA / CONDO / PUD / MIXED |
| state | US state code |
| total_units | Units in the association |
| coastal_flag | Coastal location (structural-risk proxy) |
| avg_delinq_rate_12mo | Mean monthly serious-delinquency rate, months 1–12 |
| max_delinq_rate_12mo | Peak monthly serious-delinquency rate |
| last_delinq_rate | Most recent month's rate |
| delinq_trend_3mo | Slope of last 3 months |
| pct_months_serious_12mo | Fraction of months with serious delinquency > 5% (chronic vs one-off) — strongest predictor |
| delinq_trend_12mo | Full-window slope |
| delinq_volatility_12mo | Std dev of monthly serious-delinquency rate |
| avg_late_stage_rate_12mo | Mean monthly 91+ day rate (severity depth) |
| collection_ratio_12mo | Payments ÷ charges over the window |
| lien_rate | Share of delinquency snapshots with a lien filed |
| attorney_referral_rate | Share referred to an attorney |
| rented_pct / vacant_pct | Occupancy exposure |
| **target_delinquency** | 1 if serious (61+ day) delinquency rate exceeds 10% anytime in months 13–18 |

## Model 2 — `model2_reserve_features` (12 predictors + target)

| Feature | Meaning |
|---|---|
| association_type, coastal_flag | Categorical / structural-risk context |
| building_age | Derived from year built |
| avg_funded_pct / last_funded_pct | Reserve funding level (feature window) — top drivers |
| funding_trend | Direction of funding over the window |
| contribution_ratio | Annual contribution ÷ fully-funded target |
| total_open_workorders | Open work-order count |
| critical_deferred_count | Deferred high-priority orders |
| avg_deferred_days | Average deferral duration |
| open_workorder_cost | Estimated cost of open work |
| **target_reserve_failure** | 1 if reserve falls below 25% funded anytime in the outcome window |

## Model 3 — `model3_anomaly_features` (8 predictors, unsupervised)

| Feature | Meaning |
|---|---|
| vendor_hhi | Herfindahl index of vendor-payment concentration |
| outflow_count / outflow_total | Volume and sum of outflows |
| occupied_pct | Occupancy level |
| collection_rate | Collected ÷ billed |
| collection_gap | Occupancy minus collection (phantom-unit signal) |
| reserve_volatility | Variability of reserve balance |
| reserve_max_drop | Largest single reserve drop (embezzlement signal) |

Evaluated against `data/full/_fraud_key.csv` (6 embedded fraud cases). No target column.

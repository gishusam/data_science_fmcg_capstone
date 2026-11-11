# FMCG Global Demand Planning & Forecasting | Data Science Capstone

A reproducible, course-aligned analysis of FMCG sales, promotional behavior, inventory risks, regional sales, and one-step-ahead SKU demand prediction. Built from the user-supplied CSV ZIP.

## Dataset reality check

| Attribute | Course guide claims | Actual input |
|---|---:|---:|
| Sales records | 1.1M+ | 1,100,000 |
| SKUs | 350+ | 102 |
| Countries | 15 | 7 |
| Cities | 60+ | 9 |
| Stores | not specified | 13 |
| Time period | 3 years | 2021-01-01 to 2023-12-31 |

Other files in the ZIP (`Train.csv`, `Test.csv`, `climate_features.csv`, `data_dictionary.csv`) are unrelated mortality/climate classification data, excluded from the capstone.

## Results from an executed run

- No null cells or duplicated `(store_id,sku_id,date)` records in the FMCG source.
- Sales transaction promotion share **8.02%**, stockout flag share **3.01%**.
- Observational promotion transactions averaged **104.27** units vs **55.26** units without promotion (Welch p-value < 0.001). **Not a causal lift estimate.**
- Approximate adjusted price-sales log association: **−0.790** (95% robust CI −0.901 to −0.679). Observational, not causal elasticity.
- On 10 representative SKUs, average one-step-ahead MAE: **85.75 units** (Histogram Gradient Boosting) vs **87.75 units** (Linear Regression). Test dates: 2023-07-01 through 2023-12-31.
- A model's actual future performance will differ if same-day price, weather and promotion features are not available, or if lagged sales must be recursively estimated.

## Setup and rerun

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python scripts/pipeline.py --zip /path/to/fmcg_raw_data.zip
python scripts/advanced_analysis.py --zip /path/to/fmcg_raw_data.zip
python scripts/coursework_extensions.py --zip /path/to/fmcg_raw_data.zip
jupyter notebook notebooks/FMCG_Capstone_Analysis.ipynb
```

The notebook is an inspectable report of the reproducible analyses produced by the scripts. The generated results in `reports/` are included for immediate review, and regenerate when the scripts are rerun.

## Repository

- `scripts/pipeline.py`: dataset audit, ABC/XYZ analysis, regional metrics, Welch test, 10-SKU forecasting and visuals.
- `scripts/advanced_analysis.py`: controlled log-price association, channel stockout equity checks, feature-importance diagnostic.
- `notebooks/FMCG_Capstone_Analysis.ipynb`: illustrated notebook with explained outputs and business interpretation.
- `sql/analysis_queries.sql`: retrieval, duplicate-grain validation, aggregation and trend SQL.
- `reports/`: executed CSV/JSON evidence and labeled figures.
- `data/README.md`: source-data handling; raw 1.1M rows are not bundled.

## Methodological caveats

- ABC thresholds allocate complete SKU classes using cumulative revenue before each SKU; boundary items can cause classes to exceed the nominal 80%/15%/5% shares.
- XYZ thresholds use weekly sold-unit coefficient of variation, X ≤ 0.5, Y ≤ 1.0, Z > 1.0; these are stated heuristic thresholds.
- Daily SKU aggregation pools multiple stores; it describes demand **recorded as sales**, which may be censored by stockouts.
- Two models use chronological holdout but **one-step-ahead features based on actual observed prior sales and same-day price/promo/weather**. This is not an autonomous 184-day forecast.
- MAPE excludes zero-demand dates; WAPE is also provided. Compare model quality per SKU before concluding a pooled model superiority.
- ADF is diagnostic. Tree boosting is trained on original levels with lagged histories, without blanket differencing.
- There is no explicitly labeled Traditional Trade category, no actual promotional calendar separate from flag, and insufficient geographic coverage to claim 15-country conclusions.
- No inference of causal effects, realized service-level improvement, stock savings, or revenue gains is justified from these experiments.

## Coursework extensions (executed)

Run the third script to generate a coordinate-based sales-density map, a relative logistics-hotspot table, and a stockout-classification backtest:

```bash
python scripts/coursework_extensions.py --zip /path/to/fmcg_raw_data.zip
```

- `reports/figures/geospatial_demand.png`: geolocated bubbles for all 13 stores. The hotspot flag uses >=75th percentile store-level units **and** lead-time SD, not a formal accessibility/desert claim.
- `reports/stockout_classification_metrics.csv`: all-negative baseline and balanced Logistic Regression (0.50 threshold), with precision, recall and F1 on the chronological 2023-07-01 through 2023-12-31 test split.
- `reports/figures/stockout_confusion_matrix.png`: class errors; `reports/stockout_channel_fairness.csv`: per-channel performance.
- `reports/preprocessing_and_stockout_notes.json`: scaler/encoder and experiment definitions.
- The stockout model excludes same-day `stock_on_hand`, `units_sold`, and `lead_time_days` from predictors; prior-record lags may span unequal calendar gaps. **Its low precision makes it unsuitable for live allocation decisions without further work.**
- For SKU forecasting, `Linear Regression` now uses training-only `StandardScaler`, both models include month-boundary flags, and the model-comparison CSV and notebook reflect the updated rerun. Categorical location/channel encoding is shown in the store-level stockout pipeline, not forced into the aggregated SKU forecast.

The coursework's independently observed holiday calendars, literal logistics-desert map, causally identified promotional lift, and realized inventory savings cannot be established from this dataset; these limitations are disclosed rather than invented.

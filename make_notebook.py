from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(n.v4.new_markdown_cell(s))
def code(s):cells.append(n.v4.new_code_cell(s))
md('''# FMCG Global Demand Planning and Forecasting — Capstone

## TL;DR
This notebook documents an end-to-end sales-demand analysis over 1,100,000 records (2021–2023), with SKU segmentation, promotional comparisons, demand modeling, and supply-chain recommendations. **The supplied file has 7 countries and 102 SKUs, not the 15 countries and 350+ SKUs described in the brief.** Models are evaluated using a chronological six-month holdout with **one-step-ahead** predictions and observed same-day covariates; this is **not** a six-month deployment forecast.

## Context & methods
- Input: `fmcg_raw_data.zip`, main file `data/fmcg_sales_3years_1M_rows.csv`.
- Other CSVs inside the ZIP describe mortality/climate records, not FMCG sales, and are excluded.
- Data grain: one record per `(store_id, sku_id, date)`.
- ABC: revenue-share bands 80% / next 15% / remaining 5%; XYZ: weekly sales coefficient of variation, X ≤ 0.5, Y ≤ 1.0, Z > 1.0 (heuristic thresholds).
- Test: **2023-07-01 to 2023-12-31**; training uses earlier records. The 30-day lag and rolling averages use strictly previous observations.
- Forecast features include *actual same-day promotions, price and weather*: assumed known for the backtest, but requiring forecasts or planning scenarios in real deployment.
- Outputs can be regenerated with `python scripts/pipeline.py --zip /path/to/fmcg_raw_data.zip` and `python scripts/advanced_analysis.py --zip ...`.
''')
code('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
REPORTS = ROOT / 'reports'
assert (REPORTS / 'data_audit.json').exists(), 'Run scripts/pipeline.py first'
def read_json(name):
    return json.loads((REPORTS / name).read_text())
def figure(name):
    display(Image(filename=str(REPORTS / 'figures' / name)))''')
md('## 1. Data preparation & SQL retrieval\nThe SQL queries in `sql/analysis_queries.sql` demonstrate filtering, grain validation, grouping and aggregation. A `store_id` key is needed to avoid false duplicates: multiple stores can share the same city, channel, SKU, and date.')
code('''audit = read_json('data_audit.json')
display(pd.Series(audit, name='value').to_frame())
assert audit['duplicate_store_sku_day'] == 0
assert audit['missing_cells'] == 0''')
md('## 2. Exploratory data analysis and market dynamics\nInspect the time series, geography, revenue concentration, weekly variability, and store-level logistics patterns. The dataset has only 13 stores, so geographical analysis is indicative rather than a high-resolution logistics-desert assessment.')
code("figure('daily_sales.png')")
code('''regions = pd.read_csv(REPORTS/'country_metrics.csv')
display(regions.round(3))
figure('country_revenue.png')''')
code('''segments = pd.read_csv(REPORTS/'sku_segmentation.csv')
display(segments[['sku_id','revenue','cv','ABC','XYZ','segment']].head(12).round(3))
figure('abc_xyz.png')''')
code('''logistics = pd.read_csv(REPORTS/'logistics_store_metrics.csv')
display(logistics.sort_values('stockout_rate',ascending=False).head(8).round(3))
figure('logistics_risk.png')''')
md('## 3. Hypothesis testing and price elasticity\n**Null hypothesis:** promotional transactions and non-promotional transactions have equal mean units. Welch’s two-sample test measures a difference in recorded mean sales, **not causal promotional lift**. Promotions may be allocated to high-volume stores or seasonal periods; transactions are clustered and not independently randomized. A separate price regression includes SKU, month and weekday fixed effects, but remains observational.')
code('''promo = read_json('promotion_test.json')
price = read_json('price_association.json')
display(pd.Series(promo, name='promotion analysis').to_frame())
display(pd.Series(price, name='price association').to_frame())''')
md('## 4. Feature engineering and time-aware preprocessing\nBy SKU, we create 1-, 7-, and 30-day lags and a shifted 7-day rolling mean. No scaler is needed for tree boosting; linear regression is not regularized and therefore does not require scaling. **Never shuffle time series.** The day-level SKU panel uses only observed sale-days; missing days would need explicit calendar completion before training in another dataset.')
code('''print('Features: lags 1/7/30, shifted rolling 7, weekday, month, weekend, holiday, promotion share, discounted price, rainfall and temperature.')
print('Train dates: through 2023-06-30; test dates: 2023-07-01 through 2023-12-31.')''')
md('## 5. Machine learning forecasting\nWe compare Linear Regression with Histogram Gradient Boosting for 10 revenue/volatility-diverse SKUs. ADF stationarity tests are diagnostics; differencing is not automatically applied to the tree model, because doing so can obscure demand levels. Model accuracy is measured in original demand units using MAE, RMSE, non-zero-denominator MAPE, and WAPE.')
code('''metrics = pd.read_csv(REPORTS/'model_metrics.csv')
summary = pd.read_csv(REPORTS/'model_comparison.csv')
display(metrics.round(2))
display(summary.round(2))
figure('model_comparison.png')''')
code('''stationarity = pd.read_csv(REPORTS/'adf_stationarity.csv')
display(stationarity.round(4))
figure('forecast_example.png')''')
md('## 6. Explainability, fairness and business recommendations\nPermutation importance illustrates what raises error when shuffled for **one representative SKU**, not a universal ranking. Review measured stockout rates by channel, but note that this dataset labels its channels Hypermarket, Supermarket, Convenience and E-commerce: a dedicated “Traditional Trade” category is missing.')
code('''importance = pd.read_csv(REPORTS/'feature_importance_example.csv')
equity = pd.read_csv(REPORTS/'channel_equity.csv')
display(importance.round(2))
figure('feature_importance.png')
display(equity.round(4))
figure('channel_stockout.png')''')
md('''## 7. Takeaways and limitations

1. Use the measured per-SKU backtest error to prioritize products for improved replenishment decisions; the results do not prove lower realized stockouts or inventory costs.
2. Apply ABC/XYZ-specific review cadences and safety-stock planning **only after validating lead-time demand and actual desired service levels**. The data contains lead times and stockout flags but does not establish a replenishment policy.
3. Promotions correlate with higher unit sales, but the observational Welch test does not establish incremental causal demand.
4. Check channel-level stockout disparity in deployment; don't systematically disadvantage small retailers. The supplied data cannot directly identify “Traditional Trade”.
5. **Deployment gap:** The current evaluation uses actual same-day prices, promotions and weather and actual lagged unit sales during the test period. For forecasts beyond one day, supply a promotion calendar, price plan, weather scenario and recursive/iterative predictions, then evaluate that separate realistic horizon.
6. **Data gaps:** only 7 countries, 9 cities, 13 stores and 102 SKUs versus the larger scope in the guide; stockouts may censor true demand. The source currency is not specified; revenue figures should not be assigned an assumed currency.

**Presentation storyline:** problem → data audit → ABC/XYZ / regional patterns → promotional test → 10-SKU forecasting → feature importance / channel equity → inventory-management next steps.
''')
nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
p=R/'notebooks'/'FMCG_Capstone_Analysis.ipynb';n.write(nb,p)
print(p)

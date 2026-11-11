-- Load CSV into a database table named fmcg_sales first; syntax is SQLite compatible.
-- 1. Validate store-SKU-day grain (expected no duplicate keys).
SELECT date,store_id,sku_id,COUNT(*) AS records
FROM fmcg_sales GROUP BY date,store_id,sku_id HAVING COUNT(*)>1;

-- 2. Country-level commercial and stockout summary.
SELECT country, SUM(units_sold) AS units_sold,
       SUM(net_sales) AS net_revenue,
       AVG(stock_out_flag * 1.0) AS stockout_rate
FROM fmcg_sales
WHERE date BETWEEN '2021-01-01' AND '2023-12-31'
GROUP BY country ORDER BY net_revenue DESC;

-- 3. Monthly sales by channel for temporal evaluation.
SELECT SUBSTR(date,1,7) AS month, channel,
       SUM(units_sold) AS units_sold, SUM(net_sales) AS net_sales
FROM fmcg_sales GROUP BY SUBSTR(date,1,7),channel ORDER BY month,channel;

-- 4. Aggregate BEFORE joining a dimension to avoid fan-out.
SELECT sku_id, SUM(net_sales) AS revenue, SUM(units_sold) AS units
FROM fmcg_sales GROUP BY sku_id ORDER BY revenue DESC;

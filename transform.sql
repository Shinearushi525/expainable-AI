-- ============================================================
-- transform.sql — staging + mart layers
-- Run after load_to_bigquery.py, in the BigQuery console.
-- ============================================================

-- ---------- STAGING ----------
CREATE SCHEMA IF NOT EXISTS `stg` OPTIONS (location = 'US');

CREATE OR REPLACE TABLE `stg.orders` AS
SELECT
  order_id,
  order_date,
  customer_age,
  TRIM(customer_type)               AS customer_type,
  TRIM(customer_city)               AS customer_city,
  TRIM(product_category)            AS product_category,
  TRIM(product_subcategory)         AS product_subcategory,
  ROUND(product_price, 2)           AS product_price,
  discount_percent,
  order_quantity,
  TRIM(payment_method)              AS payment_method,
  TRIM(shipping_type)               AS shipping_type,
  delivery_days,
  seller_rating,
  customer_rating,
  (warranty_available = 'Yes')        AS has_warranty,
  (product_size_available = 'Yes')    AS size_available,
  is_returned,
  TRIM(return_reason)               AS return_reason,
  (discount_percent >= 40)           AS high_discount_flag,
  (delivery_days > 5)                AS late_delivery_flag,
  (seller_rating < 3.5)              AS low_seller_rating_flag,
  (customer_rating <= 2)             AS low_customer_rating_flag,
  (payment_method = 'Cash on Delivery') AS is_cod,
  (product_price * (1 - discount_percent / 100.0)) AS post_discount_price,
  EXTRACT(MONTH FROM order_date) IN (10, 11) AS festival_season_flag
FROM `raw.orders`
WHERE order_id IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_date DESC) = 1;

-- ---------- MART ----------
CREATE SCHEMA IF NOT EXISTS `mart` OPTIONS (location = 'US');

CREATE OR REPLACE TABLE `mart.dim_product` AS
SELECT DISTINCT product_category, product_subcategory FROM `stg.orders`;

CREATE OR REPLACE TABLE `mart.dim_date` AS
SELECT
  d AS date,
  EXTRACT(YEAR FROM d) AS year,
  EXTRACT(MONTH FROM d) AS month,
  FORMAT_DATE('%B', d) AS month_name,
  EXTRACT(QUARTER FROM d) AS quarter,
  EXTRACT(DAYOFWEEK FROM d) AS day_of_week
FROM UNNEST(GENERATE_DATE_ARRAY('2021-01-01', '2026-12-31')) AS d;

CREATE OR REPLACE TABLE `mart.fact_orders` AS
SELECT
  order_id, order_date, customer_age, customer_type, customer_city,
  product_category, product_subcategory, product_price, post_discount_price,
  discount_percent, order_quantity, payment_method, shipping_type,
  delivery_days, seller_rating, customer_rating, has_warranty, size_available,
  is_returned, return_reason, high_discount_flag, late_delivery_flag,
  low_seller_rating_flag, low_customer_rating_flag, is_cod, festival_season_flag
FROM `stg.orders`;

-- Sanity check
SELECT COUNT(*) AS total_orders,
       SUM(is_returned) AS total_returns,
       ROUND(SAFE_DIVIDE(SUM(is_returned), COUNT(*)) * 100, 2) AS return_rate_pct
FROM `mart.fact_orders`;

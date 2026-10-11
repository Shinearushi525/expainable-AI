-- ============================================================
-- schema.sql — raw table schema
-- ============================================================

CREATE SCHEMA IF NOT EXISTS `raw` OPTIONS (location = 'US');

CREATE OR REPLACE TABLE `raw.orders` (
  order_id                STRING,
  order_date              DATE,
  customer_age            INT64,
  customer_type           STRING,
  customer_city           STRING,
  product_category        STRING,
  product_subcategory     STRING,
  product_price           FLOAT64,
  discount_percent        INT64,
  order_quantity          INT64,
  payment_method          STRING,
  shipping_type           STRING,
  delivery_days           INT64,
  seller_rating           FLOAT64,
  customer_rating         INT64,
  warranty_available       STRING,
  product_size_available   STRING,
  is_returned              INT64,
  return_reason            STRING
);

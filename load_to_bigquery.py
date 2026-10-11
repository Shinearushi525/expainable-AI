"""
load_to_bigquery.py

Loads ecommerce_product_returns.csv into BigQuery (raw.orders).

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
    export GCP_PROJECT_ID="your-gcp-project-id"
    python load_to_bigquery.py
"""

import os
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
DATASET = "raw"
TABLE = "orders"
CSV_PATH = "ecommerce_product_returns.csv"


def load_csv_to_bigquery():
    client = bigquery.Client(project=PROJECT_ID)

    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    dataset_ref.location = "US"
    client.create_dataset(dataset_ref, exists_ok=True)

    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("order_id", "STRING"),
            bigquery.SchemaField("order_date", "DATE"),
            bigquery.SchemaField("customer_age", "INT64"),
            bigquery.SchemaField("customer_type", "STRING"),
            bigquery.SchemaField("customer_city", "STRING"),
            bigquery.SchemaField("product_category", "STRING"),
            bigquery.SchemaField("product_subcategory", "STRING"),
            bigquery.SchemaField("product_price", "FLOAT64"),
            bigquery.SchemaField("discount_percent", "INT64"),
            bigquery.SchemaField("order_quantity", "INT64"),
            bigquery.SchemaField("payment_method", "STRING"),
            bigquery.SchemaField("shipping_type", "STRING"),
            bigquery.SchemaField("delivery_days", "INT64"),
            bigquery.SchemaField("seller_rating", "FLOAT64"),
            bigquery.SchemaField("customer_rating", "INT64"),
            bigquery.SchemaField("warranty_available", "STRING"),
            bigquery.SchemaField("product_size_available", "STRING"),
            bigquery.SchemaField("is_returned", "INT64"),
            bigquery.SchemaField("return_reason", "STRING"),
        ],
    )

    with open(CSV_PATH, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")


if __name__ == "__main__":
    load_csv_to_bigquery()

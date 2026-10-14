"""
bq_live.py

Live BigQuery access for the app's BigQuery tab. The heavy lifting (counts,
averages, joins) runs INSIDE BigQuery as SQL; only small result tables come
back to the app.
"""

import json
import os


def get_client(project, info=None):
    """Build a BigQuery client from a credentials dict, an env var, or the default."""
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if info:
        creds = service_account.Credentials.from_service_account_info(dict(info))
        return bigquery.Client(project=project, credentials=creds)
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if raw:
        creds = service_account.Credentials.from_service_account_info(json.loads(raw))
        return bigquery.Client(project=project, credentials=creds)
    return bigquery.Client(project=project)


def _run(client, sql):
    return client.query(sql).to_dataframe()


def fetch_all(client, project):
    """Returns (kpis, by_category, scored) DataFrames computed from the mart layer."""
    orders = f"`{project}.mart.fact_orders`"
    preds = f"`{project}.mart.fact_predictions`"

    kpis = _run(client, f"""
        SELECT COUNT(*) AS total_orders,
               ROUND(AVG(is_returned) * 100, 2) AS return_rate_pct
        FROM {orders}""")

    by_category = _run(client, f"""
        SELECT product_category,
               COUNT(*) AS orders,
               ROUND(AVG(is_returned) * 100, 1) AS return_rate_pct
        FROM {orders}
        GROUP BY product_category
        ORDER BY return_rate_pct DESC""")

    scored = _run(client, f"""
        SELECT p.order_id,
               p.predicted_return_probability,
               p.risk_tier,
               o.product_category,
               o.product_price,
               o.discount_percent,
               o.delivery_days
        FROM {preds} AS p
        JOIN {orders} AS o ON p.order_id = o.order_id""")

    return kpis, by_category, scored

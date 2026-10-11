"""
predict.py

Loads the model trained by train_model.py (best_model.pkl + encoders.pkl),
scores every order in BigQuery's mart.fact_orders, and writes predictions
back as mart.fact_predictions.

Usage:
    python train_model.py     # run once first, produces best_model.pkl
    python predict.py
"""

import os
import joblib
import pandas as pd
from google.cloud import bigquery

from train_model import engineer_features, encode_features, FEATURE_COLS

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"


def risk_tier(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def main():
    print("Loading trained model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    model_name = bundle["model_name"]
    encoders = joblib.load(ENCODERS_PATH)

    client = bigquery.Client(project=PROJECT_ID)

    print("Loading orders from BigQuery mart layer...")
    query = f"""
        SELECT order_id, order_date, customer_age, customer_type, customer_city,
               product_category, product_subcategory, product_price, discount_percent,
               order_quantity, payment_method, shipping_type, delivery_days,
               seller_rating, customer_rating,
               has_warranty, size_available, is_cod, high_discount_flag,
               late_delivery_flag, low_seller_rating_flag, low_customer_rating_flag,
               festival_season_flag, post_discount_price
        FROM `{PROJECT_ID}.mart.fact_orders`
    """
    df = client.query(query).to_dataframe()

    # Re-map booleans back to Yes/No so encode_features (which expects the
    # raw categorical text columns) works the same as during training.
    # Since mart.fact_orders already has engineered flags as booleans,
    # convert directly to int rather than re-running engineer_features().
    for col in ["has_warranty", "size_available", "is_cod", "high_discount_flag",
                "late_delivery_flag", "low_seller_rating_flag",
                "low_customer_rating_flag", "festival_season_flag"]:
        df[col] = df[col].astype(int)

    df_enc, _ = encode_features(df, encoders=encoders, fit=False)

    print(f"Scoring {len(df_enc)} orders with {model_name}...")
    df_enc["predicted_return_probability"] = model.predict_proba(df_enc[FEATURE_COLS])[:, 1]
    df_enc["predicted_label"] = model.predict(df_enc[FEATURE_COLS])
    df_enc["risk_tier"] = df_enc["predicted_return_probability"].apply(risk_tier)
    df_enc["model_version"] = model_name
    df_enc["prediction_date"] = pd.Timestamp.now(tz="UTC")

    result = df_enc[[
        "order_id", "predicted_return_probability", "predicted_label",
        "risk_tier", "model_version", "prediction_date",
    ]]

    table_id = f"{PROJECT_ID}.mart.fact_predictions"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(result, table_id, job_config=job_config)
    job.result()

    print(f"Wrote {len(result)} predictions to {table_id}")
    print(result["risk_tier"].value_counts())


if __name__ == "__main__":
    main()

"""
train_model.py

Trains and compares 5 classification models on the e-commerce return
dataset, auto-selects the best by ROC-AUC, and saves it (plus encoders
and feature list) for reuse by app.py and predict.py.

Usage:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "ecommerce_product_returns.csv"
MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"
LEADERBOARD_PATH = "model_leaderboard.csv"

TARGET = "is_returned"

CATEGORICAL_COLS = [
    "customer_type", "customer_city", "product_category", "product_subcategory",
    "payment_method", "shipping_type",
]
BOOLEAN_SOURCE_COLS = ["warranty_available", "product_size_available"]
NUMERIC_COLS = [
    "customer_age", "product_price", "discount_percent", "order_quantity",
    "delivery_days", "seller_rating", "customer_rating",
]


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    return df


def engineer_features(df):
    df = df.copy()
    df["has_warranty"] = (df["warranty_available"] == "Yes").astype(int)
    df["size_available"] = (df["product_size_available"] == "Yes").astype(int)
    df["high_discount_flag"] = (df["discount_percent"] >= 40).astype(int)
    df["late_delivery_flag"] = (df["delivery_days"] > 5).astype(int)
    df["low_seller_rating_flag"] = (df["seller_rating"] < 3.5).astype(int)
    df["low_customer_rating_flag"] = (df["customer_rating"] <= 2).astype(int)
    df["is_cod"] = (df["payment_method"] == "Cash on Delivery").astype(int)
    df["post_discount_price"] = df["product_price"] * (1 - df["discount_percent"] / 100.0)
    df["festival_season_flag"] = df["order_date"].dt.month.isin([10, 11]).astype(int)
    return df


def encode_features(df, encoders=None, fit=True):
    df = df.copy()
    encoders = encoders or {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    return df, encoders


FEATURE_COLS = (
    [c + "_enc" for c in CATEGORICAL_COLS]
    + ["has_warranty", "size_available", "high_discount_flag", "late_delivery_flag",
       "low_seller_rating_flag", "low_customer_rating_flag", "is_cod", "festival_season_flag"]
    + ["post_discount_price"] + NUMERIC_COLS
)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def main():
    print("Loading data...")
    df = load_data()
    df = engineer_features(df)
    df_enc, encoders = encode_features(df, fit=True)

    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    trained_models = {}
    print("Training and comparing models...\n")
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = model
        print(f"{name:22s} ROC-AUC={metrics['roc_auc']:.3f}  "
              f"Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}")

    leaderboard = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    print(f"\nLeaderboard saved to {LEADERBOARD_PATH}")

    best_name = leaderboard.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC={leaderboard.iloc[0]['roc_auc']:.3f})")

    joblib.dump({"model": best_model, "model_name": best_name, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved encoders -> {ENCODERS_PATH}")


if __name__ == "__main__":
    main()

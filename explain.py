"""
explain.py

Explains WHY the model flagged a specific order as risky, in plain
English — WITHOUT using the `shap` library.

Why not shap: shap depends on `numba`, which ships compiled DLLs that
some locked-down Windows machines block via Application Control /
Smart App Control policies (a security feature, not something pip can
fix). This module gets the same kind of answer using a simple,
dependency-free technique instead: PERTURBATION IMPORTANCE.

------------------------------------------------------------------------
HOW IT WORKS (in plain terms)
------------------------------------------------------------------------
For the order being explained, and for each feature one at a time:
    1. Temporarily replace that ONE feature's value with a "typical"
       baseline value (the median, computed from the training data)
    2. Ask the model for a new prediction with that swap
    3. The difference between the original prediction and this new
       one tells us how much that feature was pushing the risk score
       up or down

This is a simplified, honest, model-agnostic explanation method (in
the same spirit as SHAP/LIME) that only needs numpy/pandas/sklearn —
no compiled extensions, so it works everywhere including locked-down
Windows machines.
"""

import numpy as np
import pandas as pd

DATA_PATH = "ecommerce_product_returns.csv"

FEATURE_LABELS = {
    "customer_type_enc": "customer type",
    "customer_city_enc": "customer's city",
    "product_category_enc": "product category",
    "product_subcategory_enc": "product subcategory",
    "payment_method_enc": "payment method",
    "shipping_type_enc": "shipping type",
    "has_warranty": "warranty availability",
    "size_available": "size availability",
    "high_discount_flag": "high discount (≥40%)",
    "late_delivery_flag": "late delivery (>5 days)",
    "low_seller_rating_flag": "low seller rating (<3.5)",
    "low_customer_rating_flag": "low customer rating (≤2)",
    "is_cod": "Cash on Delivery payment",
    "festival_season_flag": "festival season timing",
    "post_discount_price": "final discounted price",
    "customer_age": "customer age",
    "product_price": "product price",
    "discount_percent": "discount percentage",
    "order_quantity": "order quantity",
    "delivery_days": "delivery days",
    "seller_rating": "seller rating",
    "customer_rating": "customer rating",
}


class PerturbationExplainer:
    """Holds the model + a cached baseline (median) feature vector."""

    def __init__(self, model, baseline: pd.Series):
        self.model = model
        self.baseline = baseline  # one row: median of each feature column


def get_explainer(model, background_df: pd.DataFrame = None) -> PerturbationExplainer:
    """
    Builds an explainer. If background_df (an already-encoded dataframe
    of FEATURE_COLS) isn't provided, it's built automatically from the
    training CSV so this still works with a single argument, matching
    the old shap-based call signature used elsewhere in the app.
    """
    if background_df is None:
        from train_model import engineer_features, encode_features, FEATURE_COLS
        raw = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
        feat = engineer_features(raw)
        enc, _ = encode_features(feat, fit=True)
        background_df = enc[FEATURE_COLS]

    baseline = background_df.median(numeric_only=True)
    return PerturbationExplainer(model, baseline)


def explain_order(explainer: PerturbationExplainer, X_row: pd.DataFrame, top_n: int = 3) -> dict:
    """
    Same signature/return shape as the old shap-based version, so
    nothing calling this needs to change.
    """
    model = explainer.model
    baseline = explainer.baseline
    original_prob = model.predict_proba(X_row)[0, 1]

    impacts = []
    for col in X_row.columns:
        if col not in baseline.index:
            continue
        perturbed = X_row.copy()
        perturbed[col] = baseline[col]
        perturbed_prob = model.predict_proba(perturbed)[0, 1]
        # Positive impact = this feature's actual value INCREASED risk
        # relative to a typical/baseline order
        impact = original_prob - perturbed_prob
        impacts.append((col, impact))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top = impacts[:top_n]

    parts = []
    for feat, val in top:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "increased" if val > 0 else "decreased"
        parts.append(f"{label} ({direction} risk by {abs(val):.3f})")

    sentence = "This order's risk was mainly driven by: " + "; ".join(parts) + "."

    return {
        "top_features": [{"feature": FEATURE_LABELS.get(f, f), "impact": float(v)} for f, v in top],
        "explanation": sentence,
    }


if __name__ == "__main__":
    import joblib
    from train_model import engineer_features, encode_features, FEATURE_COLS

    bundle = joblib.load("best_model.pkl")
    model = bundle["model"]
    encoders = joblib.load("encoders.pkl")

    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    df_feat = engineer_features(df)
    df_enc, _ = encode_features(df_feat, encoders=encoders, fit=False)

    explainer = get_explainer(model, df_enc[FEATURE_COLS])

    for i in range(3):
        row = df_enc[FEATURE_COLS].iloc[[i]]
        result = explain_order(explainer, row)
        print(f"\nOrder {df.iloc[i]['order_id']}:")
        print(result["explanation"])
